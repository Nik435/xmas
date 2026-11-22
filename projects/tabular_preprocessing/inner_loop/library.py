"""The fixed functions each pipeline stage can be filled with.

The slots follow the project concept, modelled on TabFM-Auto (arXiv:2609.37989):
``clean -> merge -> engineer -> sample -> predictor -> postprocess``. Every
function of a stage has the stage's signature, followed by keyword-only
parameters, which are what an agent chooses:

- ``clean(df, *, ...) -> df``: one database table, named by the call's
  ``table``. The table's keys and time column are hidden from the function, so
  cleaning can never break a join. Dropping rows is allowed.
- ``merge(ctx, seed, *, ...) -> new columns``: features for each label row
  from the database, one output row per seed row. Called once for train and
  once for test rows, so both are treated alike. Aggregates only see database
  rows strictly before the label row's own time, always over all of that history
  (no time windows: the rows to predict lie after the database's cutoff, where a
  short window would be empty). The seed rows carry no target; past labels come
  only from the fitting rows (``ctx.labels``), through ``target_history``.
- ``engineer(X_train, y_train, X_test, *, ...) -> (X_train, X_test)``: add
  columns to the merged frames. Label-free statistics may use train and test
  together (transductive).
- ``sample(X_train, y_train, X_test, max_rows, *, ...) -> list[row indices]``:
  the context views the predictor is fitted on, one fit per view.
- ``postprocess(pred, y_train, X_test, *, ...) -> pred``: adjust predictions;
  class probabilities ``(n, k)`` for classification, values ``(n,)`` for regression.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd

from tabular_preprocessing.inner_loop.data import Database, Table

STAGES: tuple[str, ...] = ("clean", "merge", "engineer", "sample", "postprocess")

# stage -> function name -> function
REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {stage: {} for stage in STAGES}


def register[F: Callable[..., Any]](stage: str) -> Callable[[F], F]:
    """Add a function to :data:`REGISTRY` under ``stage`` and its own name."""

    def decorate(fn: F) -> F:
        REGISTRY[stage][fn.__name__] = fn
        return fn

    return decorate


# --- clean -----------------------------------------------------------------------------


@register("clean")
def zero_as_missing(df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    """Treat 0 as a missing measurement in the given columns (e.g. a lap time of 0)."""
    df = df.copy()
    df[columns] = df[columns].mask(df[columns] == 0)
    return df


@register("clean")
def placeholder_as_missing(df: pd.DataFrame, *, values: list[Any], columns: list[str] | None = None) -> pd.DataFrame:
    r"""Treat placeholder values (e.g. -1, "\\N", "?", "unknown") as missing, in all or the given columns."""
    cols = columns if columns is not None else list(df.columns)
    df = df.copy()
    df[cols] = df[cols].mask(df[cols].isin(values))
    return df


@register("clean")
def to_numeric(df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    """Parse text columns that hold numbers; values that do not parse become missing."""
    df = df.copy()
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@register("clean")
def missing_indicators(df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    """Add a 0/1 ``<column>_missing`` column per given column, so missingness can be aggregated."""
    df = df.copy()
    for c in columns:
        df[f"{c}_missing"] = df[c].isna().astype(float)
    return df


@register("clean")
def drop_columns(df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    """Drop columns, e.g. free text or leaky ones."""
    return df.drop(columns=columns)


@register("clean")
def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that repeat an earlier row in everything but the primary key (foreign keys and time included).

    Two results of different drivers or races are never duplicates, even with equal values.
    """
    return df[~df.duplicated()]


# sees the foreign keys and the time column (read only; the harness checks they are unchanged)
drop_duplicate_rows.sees_keys = True  # type: ignore[attr-defined]


# --- merge -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeContext:
    """What the merge functions read: the cleaned database and the task's label columns.

    Attributes:
        db: The cleaned tables.
        con: DuckDB connection with one view per table.
        entity_table: The table the label rows refer to.
        entity_col: Label column holding the entity id.
        time_col: Label column holding the prediction time.
        labels: The fitting rows' labels (entity, time, target), the only labels any merge function
            may read; ``None`` where none are given.
        target_col: The target's column in ``labels``.
        horizon: How far ahead a label looks (see :attr:`RelationalTask.horizon`).
    """

    db: Database
    con: duckdb.DuckDBPyConnection
    entity_table: str
    entity_col: str
    time_col: str
    labels: pd.DataFrame | None = None
    target_col: str | None = None
    horizon: pd.Timedelta | None = None


AggOp = Literal[
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "nunique",
    "mode",
    "count_by",
    "last",
    "time_since_last",
    "time_since_first",
]
_SQL = {"sum": "sum", "mean": "avg", "min": "min", "max": "max", "std": "stddev", "nunique": "count(DISTINCT"}
COUNT_BY_TOP = 5  # values count_by makes a column for, by default: the most frequent ones
COUNT_BY_MAX = 20


def _entity_rows(ctx: MergeContext, seed: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The entity table's ``columns`` for each seed row (NaN where the entity is missing)."""
    table = ctx.db[ctx.entity_table]
    if table.pkey is None:
        raise ValueError(f"{ctx.entity_table!r} has no primary key to join on.")
    rows = table.df[[table.pkey, *columns]]
    if rows[table.pkey].duplicated().any():
        raise ValueError(f"{ctx.entity_table}.{table.pkey} is not unique; the join would multiply rows.")
    joined = seed[[ctx.entity_col]].merge(rows, how="left", left_on=ctx.entity_col, right_on=table.pkey)
    return joined[columns].set_axis(seed.index)


@register("merge")
def entity_columns(ctx: MergeContext, seed: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    """Columns of the entity's own row (e.g. a driver's nationality), as ``<entity_table>.<column>``."""
    columns = _unprefixed(columns, ctx.entity_table, ctx.db[ctx.entity_table].df.columns)
    return _entity_rows(ctx, seed, columns).add_prefix(f"{ctx.entity_table}.")


@register("merge")
def lookup(ctx: MergeContext, seed: pd.DataFrame, *, fk_column: str, columns: list[str]) -> pd.DataFrame:
    """Columns of the row a foreign key *of the entity table itself* points to (n:1), as ``<table>.<column>``.

    E.g. a driver's team through the driver's teamId. Only the entity table's own foreign keys work
    here; a table that points to the entity is aggregated (aggregate) or, with one row per entity,
    copied (child_columns).
    """
    fkeys = ctx.db[ctx.entity_table].fkeys or {}
    if fk_column not in fkeys:
        raise ValueError(
            f"{ctx.entity_table!r} has no foreign key {fk_column!r}; its foreign keys are {sorted(fkeys) or 'none'}. "
            "lookup only follows the entity table's own foreign keys; for a table that points to the entity "
            "use aggregate (several rows per entity) or child_columns (one row per entity)."
        )
    parent = ctx.db[fkeys[fk_column]]
    if parent.pkey is None:
        raise ValueError(f"{fkeys[fk_column]!r} has no primary key to join on.")
    columns = _unprefixed(columns, fkeys[fk_column], parent.df.columns)
    keys = _entity_rows(ctx, seed, [fk_column])
    rows = parent.df[[parent.pkey, *columns]].drop_duplicates(parent.pkey)
    joined = keys.merge(rows, how="left", left_on=fk_column, right_on=parent.pkey)
    return joined[columns].set_axis(seed.index).add_prefix(f"{fkeys[fk_column]}.")


@register("merge")
def aggregate(
    ctx: MergeContext,
    seed: pd.DataFrame,
    *,
    table: str,
    fk_column: str,
    aggregations: list[dict[str, Any]],
    where: dict[str, list[Any]] | None = None,
    via: str | None = None,
) -> pd.DataFrame:
    """Aggregate a table's rows that belong to the entity (1:n), per label row, over all history before its time.

    ``aggregations`` is a list like ``[{"op": "count"}, {"op": "mean", "column": "points"}]``; ops:
    count, sum, mean, min, max, std (numeric columns), nunique, mode (the most frequent value, also of
    text), count_by (one count per value of a categorical column: ``{"op": "count_by", "column":
    "status"}`` for its 5 most frequent values, or ``"values": [...]``), last (the column's value in the
    latest row), time_since_last (days since the latest row) and time_since_first (days since the
    first row: how long the entity has had rows, e.g. a product's age). ``where`` keeps only rows with one of
    the given values, e.g. ``{"status": ["yes"]}``.

    ``via`` reaches a table two steps away: ``fk_column`` is then the key of the ``via`` table to the
    entity, and the ``via`` table holds a foreign key to ``table``; e.g. ``table="events",
    via="event_attendees", fk_column="user_id"`` aggregates the events a user attended. Columns (in
    ``aggregations`` and ``where``) may then come from either table. Columns are named
    ``<table>.<op>(<column>)``, with ``<via>><table>`` for ``via`` and ``[<column>=<values>]`` for ``where``;
    a table with two keys to the entity (e.g. a user's friends, by ``user`` and by ``friend``) is named
    ``<table>[<fk_column>]``, so both directions can be aggregated.
    """
    child = ctx.db.get(table)
    if child is None:
        raise ValueError(f"No table {table!r}; tables are {sorted(ctx.db)}.")
    if via is None:
        if (child.fkeys or {}).get(fk_column) != ctx.entity_table:
            raise ValueError(f"{table}.{fk_column} is not a foreign key to {ctx.entity_table!r}.")
        sources = {"f": child}
        joins = f'JOIN "{table}" f ON f."{fk_column}" = s."{ctx.entity_col}"'
        label = _keyed(table, child, fk_column, ctx.entity_table)
    else:
        bridge = ctx.db.get(via)
        if bridge is None:
            raise ValueError(f"No table {via!r} for via; tables are {sorted(ctx.db)}.")
        if (bridge.fkeys or {}).get(fk_column) != ctx.entity_table:
            raise ValueError(f"{via}.{fk_column} is not a foreign key to {ctx.entity_table!r}.")
        links = [c for c, target in (bridge.fkeys or {}).items() if target == table]
        if len(links) != 1:
            raise ValueError(
                f"via needs exactly one foreign key from {via!r} to {table!r}; {via} has {bridge.fkeys or 'none'}."
            )
        if child.pkey is None:
            raise ValueError(f"{table!r} has no primary key to join on.")
        sources = {"f": child, "b": bridge}  # the target table first: its columns win a name clash
        label = table
        joins = (
            f'JOIN "{via}" b ON b."{fk_column}" = s."{ctx.entity_col}" '
            f'JOIN "{table}" f ON f."{child.pkey}" = b."{links[0]}"'
        )

    def resolve(column: Any) -> tuple[str, pd.Series]:
        for alias, source in sources.items():
            if column in source.df.columns:
                return alias, source.df[column]
        names = sorted({c for source in sources.values() for c in source.df.columns})
        raise ValueError(f"{' or '.join(_name(table, via))} has no column {column!r}; columns are {names}.")

    timed = next(((a, s) for a, s in sources.items() if s.time_col), None)
    exprs, names, kinds, params = [], [], [], []
    for agg in aggregations:
        op, column = agg.get("op"), agg.get("column")
        if op == "count":
            items = [("count(*)", f"{label}.count()", "count")]
        elif op in ("time_since_last", "time_since_first", "last"):
            if timed is None:
                raise ValueError(f"{table!r} has no time column for {op}.")
            ta, tcol = timed[0], timed[1].time_col
            if op in ("time_since_last", "time_since_first"):
                column, edge = tcol, "max" if op == "time_since_last" else "min"
                items = [(f'date_diff(\'day\', {edge}({ta}."{tcol}"), s."{ctx.time_col}")', None, "number")]
            else:
                alias, values = resolve(column)
                items = [(f'arg_max({_sql_value(alias, column, values)}, {ta}."{tcol}")', None, "mode")]
        elif op == "count_by":
            alias, values = resolve(column)
            chosen = agg.get("values")
            if isinstance(chosen, str):
                chosen = [chosen]
            if not chosen:
                top = min(int(agg.get("top", COUNT_BY_TOP)), COUNT_BY_MAX)
                chosen = values.value_counts().head(top).index.tolist()
            if not chosen:
                raise ValueError(f"count_by: {table}.{column} has no values.")
            items = []
            for value in chosen:
                items.append(
                    (
                        f'sum(CASE WHEN {alias}."{column}" = ? THEN 1 ELSE 0 END)',
                        f"{label}.count_by({column}={value})",
                        "count",
                    )
                )
                params.append(_param(value, values))
        elif op == "mode":
            alias, values = resolve(column)
            items = [(f'mode({alias}."{column}")', None, "mode")]
        elif op in _SQL:
            alias, values = resolve(column)
            if op != "nunique" and not pd.api.types.is_numeric_dtype(values):
                raise ValueError(
                    f"{op} needs a numeric column; {table}.{column} is {values.dtype} (use mode or count_by)."
                )
            value = f'{alias}."{column}"' if op == "nunique" else _sql_value(alias, column, values)
            items = [(f"{_SQL[op]}({value})" + (")" if op == "nunique" else ""), None, "number")]
        else:
            raise ValueError(f"Unknown aggregation op {op!r}; choose from {list(AggOp.__args__)}.")  # type: ignore[attr-defined]
        for expr, name, kind in items:
            exprs.append(expr)
            names.append(name or f"{label}.{op}({column or ''})")
            kinds.append(kind)
    if not exprs:
        raise ValueError("aggregations must not be empty.")
    prefix = f"{_keyed(via, ctx.db[via], fk_column, ctx.entity_table)}>" if via else ""
    suffix = "".join(f"[{c}={'|'.join(map(str, v))}]" for c, v in (where or {}).items())
    names = [prefix + n + suffix for n in names]
    if len(set(names)) < len(names):
        raise ValueError(f"aggregations repeat a column: {names}.")

    # strictly before the label row's time: a row at the prediction time is the future
    filters = [f' AND {a}."{s.time_col}" < s."{ctx.time_col}"' for a, s in sources.items() if s.time_col]
    for column, values in (where or {}).items():
        if not values:
            raise ValueError(f"where: no values given for {column!r}.")
        alias, series = resolve(column)
        filters.append(f' AND {alias}."{column}" IN ({", ".join("?" * len(values))})')
        params.extend(_param(v, series) for v in values)
    frame = seed[[ctx.entity_col, ctx.time_col]].reset_index(drop=True)
    frame["__rid__"] = np.arange(len(frame))
    ctx.con.register("__seed__", frame)
    try:
        select = ", ".join(f'{e} AS "{_quoted(n)}"' for e, n in zip(exprs, names, strict=True))
        sql = (
            f'SELECT s."__rid__", {select} FROM "__seed__" s {joins}{"".join(filters)} '
            f'GROUP BY s."__rid__", s."{ctx.time_col}"'
        )
        result = ctx.con.execute(sql, params).df()
    finally:
        ctx.con.unregister("__seed__")
    out = frame[["__rid__"]].merge(result, how="left", on="__rid__")[names]
    for name, kind in zip(names, kinds, strict=True):
        if kind == "count":  # no matching rows means a count of 0, not unknown
            out[name] = out[name].fillna(0).astype(float)
        elif kind == "number" or pd.api.types.is_numeric_dtype(out[name]):
            out[name] = out[name].astype(float)
    return out.set_axis(seed.index)


def _keyed(name: str, table: Table, fk_column: str, entity: str) -> str:
    """A table's name in feature names: with the key when it has two keys to the entity (``user_friends[friend]``)."""
    to_entity = [c for c, target in (table.fkeys or {}).items() if target == entity]
    return f"{name}[{fk_column}]" if len(to_entity) > 1 else name


def _sql_value(alias: str, column: str, values: pd.Series) -> str:
    """A column in SQL, booleans as numbers: DuckDB has no avg / sum / stddev of BOOLEAN."""
    if pd.api.types.is_bool_dtype(values):
        return f'CAST({alias}."{column}" AS DOUBLE)'
    return f'{alias}."{column}"'


def _unprefixed(columns: list[str], table: str, available: Any) -> list[str]:
    """``columns`` with a ``<table>.`` prefix removed where only the bare name exists (as a model often writes them)."""
    have = set(available)
    out = []
    for c in columns:
        bare = c[len(table) + 1 :] if c.startswith(f"{table}.") else c
        out.append(bare if c not in have and bare in have else c)
    return out


def _quoted(name: str) -> str:
    return name.replace('"', '""')


def _name(table: str, via: str | None) -> list[str]:
    return [table] if via is None else [table, via]


def _param(value: Any, column: pd.Series) -> Any:
    """A ``where`` / ``count_by`` value in the column's type: a model often writes numbers as text."""
    if pd.api.types.is_numeric_dtype(column) and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{value!r} is not a number, but the column {column.name!r} is numeric.") from None
    return value.item() if isinstance(value, np.generic) else value


@register("merge")
def child_columns(
    ctx: MergeContext, seed: pd.DataFrame, *, table: str, fk_column: str, columns: list[str]
) -> pd.DataFrame:
    """Columns of a table with at most one row per entity (1:1, e.g. a study's design), as ``<table>.<column>``.

    The table's foreign key ``fk_column`` points to the entity table. A row dated at or after the label
    row's time is not used (its values are left missing), as in aggregate.
    """
    child = ctx.db.get(table)
    if child is None:
        raise ValueError(f"No table {table!r}; tables are {sorted(ctx.db)}.")
    if (child.fkeys or {}).get(fk_column) != ctx.entity_table:
        raise ValueError(f"{table}.{fk_column} is not a foreign key to {ctx.entity_table!r}.")
    columns = _unprefixed(columns, table, child.df.columns)
    keys = [c for c in columns if c in child.keys]
    if keys:
        raise ValueError(f"child_columns copies data columns, not keys or the time column: {keys}.")
    if child.df[fk_column].dropna().duplicated().any():
        raise ValueError(
            f"{table} has several rows per entity; aggregate them instead (child_columns is for 1:1 tables)."
        )
    time = [child.time_col] if child.time_col else []
    rows = child.df[[fk_column, *time, *columns]].rename(columns={child.time_col: "__time__"} if time else {})
    joined = seed[[ctx.entity_col, ctx.time_col]].merge(rows, how="left", left_on=ctx.entity_col, right_on=fk_column)
    out = joined[columns].copy()
    if time:  # a row from the label time on is the future
        future = ~(pd.to_datetime(joined["__time__"]) < pd.to_datetime(joined[ctx.time_col]))
        out = out.mask(future)
    return out.set_axis(seed.index).add_prefix(f"{table}.")


HistoryOp = Literal["count", "mean", "sum", "min", "max", "std", "last", "time_since_last"]


@register("merge")
def target_history(ctx: MergeContext, seed: pd.DataFrame, *, ops: list[HistoryOp]) -> pd.DataFrame:
    """The entity's own past labels, aggregated per label row: e.g. how often it was positive before, its last label.

    Only labels of the fitting rows are used, and only once they are known: a label at time t describes
    the task's horizon after t, so it counts from t + horizon on. A row therefore never sees its own
    target. ops: count, mean, sum, min, max, std, last (the most recent known label), time_since_last
    (days since that label's time). Columns are named ``target_history.<op>(<target>)``.
    """
    if ctx.labels is None or ctx.target_col is None:
        raise ValueError("No past labels are available here.")
    if not ops:
        raise ValueError("ops must not be empty.")
    labels = ctx.labels
    numeric = pd.api.types.is_numeric_dtype(labels[ctx.target_col])
    exprs, names = [], []
    for op in ops:
        if op in ("mean", "sum", "min", "max", "std") and not numeric:
            raise ValueError(f"{op} needs a numeric target; {ctx.target_col} is {labels[ctx.target_col].dtype}.")
        expr = {
            "count": "count(*)",
            "last": 'arg_max(l."y", l."t")',
            "time_since_last": f'date_diff(\'day\', max(l."t"), s."{ctx.time_col}")',
        }.get(op) or f'{_SQL[op]}(l."y")'
        names.append(f"target_history.{op}({ctx.target_col})")
        exprs.append(f'{expr} AS "{_quoted(names[-1])}"')
    hist = pd.DataFrame(
        {
            "e": labels[ctx.entity_col].to_numpy(),
            "t": pd.to_datetime(labels[ctx.time_col]).to_numpy(),
            # booleans as numbers: DuckDB has no avg of BOOLEAN
            "y": labels[ctx.target_col].astype(float).to_numpy()
            if pd.api.types.is_bool_dtype(labels[ctx.target_col])
            else labels[ctx.target_col].to_numpy(),
        }
    )
    # known from the end of its period on; without a horizon, from just after its own time
    hist["known"] = hist["t"] + ctx.horizon if ctx.horizon is not None else hist["t"]
    known = 'l."known" <= s.' if ctx.horizon is not None else 'l."known" < s.'
    frame = seed[[ctx.entity_col, ctx.time_col]].reset_index(drop=True)
    frame["__rid__"] = np.arange(len(frame))
    ctx.con.register("__seed__", frame)
    ctx.con.register("__labels__", hist)
    try:
        sql = (
            f'SELECT s."__rid__", {", ".join(exprs)} FROM "__seed__" s JOIN "__labels__" l '
            f'ON l."e" = s."{ctx.entity_col}" AND {known}"{ctx.time_col}" '
            f'GROUP BY s."__rid__", s."{ctx.time_col}"'
        )
        result = ctx.con.sql(sql).df()
    finally:
        ctx.con.unregister("__seed__")
        ctx.con.unregister("__labels__")
    out = frame[["__rid__"]].merge(result, how="left", on="__rid__")[names]
    count = f"target_history.count({ctx.target_col})"
    if count in out:  # no known label means a count of 0, not unknown
        out[count] = out[count].fillna(0)
    for name in out.columns:  # last of a text target stays text
        if pd.api.types.is_numeric_dtype(out[name]):
            out[name] = out[name].astype(float)
    return out.set_axis(seed.index)


# --- engineer --------------------------------------------------------------------------


def _require(x: pd.DataFrame, *columns: str) -> None:
    """Fail with the closest feature names if a column is not a feature (merged names are easy to misspell)."""
    missing = [c for c in columns if c not in x.columns]
    if not missing:
        return
    hints = []
    for c in missing:
        close = difflib.get_close_matches(c, [str(f) for f in x.columns], n=3, cutoff=0.4)
        hints.append(f"{c!r} (closest: {', '.join(close) or 'none'})")
    raise ValueError(f"no feature column {'; '.join(hints)}. The features are: {', '.join(map(str, x.columns))}.")


def _both(x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    """Train and test stacked, for label-free statistics over all rows."""
    return pd.concat([x_train, x_test], axis=0, ignore_index=True)


def _split(both: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(x_train)
    return both.iloc[:n].set_axis(x_train.index), both.iloc[n:].set_axis(x_test.index)


@register("engineer")
def combine(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,
    *,
    a: str,
    b: str,
    op: Literal["add", "sub", "mul", "div"],
    name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A new column from two numeric columns: ``a + b``, ``a - b``, ``a * b`` or ``a / b`` (NaN where b is 0)."""
    _require(x_train, a, b)
    out = name or f"{a}_{op}_{b}"
    x_train, x_test = x_train.copy(), x_test.copy()
    for x in (x_train, x_test):
        left, right = x[a].astype(float), x[b].astype(float)
        x[out] = {
            "add": left + right,
            "sub": left - right,
            "mul": left * right,
            "div": left / right.where(right != 0),
        }[op]
    return x_train, x_test


@register("engineer")
def group_stat(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,
    *,
    group: str,
    value: str,
    stat: Literal["mean", "median", "std", "min", "max"] = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per row, a statistic of ``value`` within its ``group``, over train and test (label-free)."""
    _require(x_train, group, value)
    both = _both(x_train, x_test)
    both[f"{value}_{stat}_by_{group}"] = both.groupby(group, dropna=False)[value].transform(stat)
    return _split(both, x_train, x_test)


@register("engineer")
def count_encode(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,
    *,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``<column>_count``: how often the row's value occurs, over train and test."""
    _require(x_train, *columns)
    both = _both(x_train, x_test)
    for c in columns:
        both[f"{c}_count"] = both.groupby(c, dropna=False)[c].transform("size").astype(float)
    return _split(both, x_train, x_test)


@register("engineer")
def log_transform(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,
    *,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace numeric columns by their signed ``log1p``, for heavy-tailed values."""
    _require(x_train, *columns)
    x_train, x_test = x_train.copy(), x_test.copy()
    for x in (x_train, x_test):
        values = x[columns].astype(float)
        x[columns] = np.sign(values) * np.log1p(values.abs())
    return x_train, x_test


@register("engineer")
def datetime_parts(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,
    *,
    column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a date column's year, month, day of week and hour as numeric columns."""
    _require(x_train, column)
    x_train, x_test = x_train.copy(), x_test.copy()
    for x in (x_train, x_test):
        parsed = pd.to_datetime(x[column], errors="coerce")
        for part in ("year", "month", "day_of_week", "hour"):
            x[f"{column}_{part}"] = getattr(parsed.dt, part).astype(float)
    return x_train, x_test


@register("engineer")
def drop_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,
    *,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop feature columns, e.g. after a validator flags them as leaky or useless."""
    _require(x_train, *columns)
    return x_train.drop(columns=columns), x_test.drop(columns=columns)


# --- sample ----------------------------------------------------------------------------


@register("sample")
def full(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,  # noqa: ARG001
    max_rows: int,
    *,
    seed: int = 0,
) -> list[np.ndarray]:
    """One view: every training row, or a random ``max_rows`` of them if there are more."""
    n = len(x_train)
    if n <= max_rows:
        return [np.arange(n)]
    return [np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))]


@register("sample")
def random_views(
    x_train: pd.DataFrame,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,  # noqa: ARG001
    max_rows: int,
    *,
    n_views: int = 4,
    seed: int = 0,
) -> list[np.ndarray]:
    """``n_views`` random subsets of ``max_rows`` rows each; their predictions are averaged."""
    rng = np.random.default_rng(seed)
    n = len(x_train)
    size = min(n, max_rows)
    return [np.sort(rng.choice(n, size=size, replace=False)) for _ in range(n_views)]


@register("sample")
def recent(
    x_train: pd.DataFrame,  # noqa: ARG001
    y_train: pd.Series,
    x_test: pd.DataFrame,  # noqa: ARG001
    max_rows: int,
) -> list[np.ndarray]:
    """One view: the latest ``max_rows`` training rows (label rows are in time order), for drifting data."""
    n = len(y_train)
    return [np.arange(max(0, n - max_rows), n)]


@register("sample")
def stratified_views(
    x_train: pd.DataFrame,  # noqa: ARG001
    y_train: pd.Series,
    x_test: pd.DataFrame,  # noqa: ARG001
    max_rows: int,
    *,
    n_views: int = 4,
    seed: int = 0,
) -> list[np.ndarray]:
    """``n_views`` subsets that keep the class proportions, each up to ``max_rows`` rows (classification)."""
    rng = np.random.default_rng(seed)
    labels = y_train.to_numpy()
    share = min(1.0, max_rows / len(labels))
    views = []
    for _ in range(n_views):
        rows = [
            rng.choice(idx, size=max(1, round(share * len(idx))), replace=False)
            for idx in (np.flatnonzero(labels == c) for c in np.unique(labels))
        ]
        views.append(np.sort(np.concatenate(rows)))
    return views


@register("sample")
def oversample_minority(
    x_train: pd.DataFrame,  # noqa: ARG001
    y_train: pd.Series,
    x_test: pd.DataFrame,  # noqa: ARG001
    max_rows: int,
    *,
    n_views: int = 1,
    seed: int = 0,
) -> list[np.ndarray]:
    """Views with every class repeated up to the size of the largest, within ``max_rows`` (classification)."""
    rng = np.random.default_rng(seed)
    labels = y_train.to_numpy()
    classes = np.unique(labels)
    per_class = min(max(int((labels == c).sum()) for c in classes), max_rows // len(classes))
    views = []
    for _ in range(n_views):
        rows = [
            rng.choice(idx, size=per_class, replace=len(idx) < per_class)
            for idx in (np.flatnonzero(labels == c) for c in classes)
        ]
        views.append(np.sort(np.concatenate(rows)))
    return views


# --- postprocess -----------------------------------------------------------------------


def _classification(pred: np.ndarray, fn: str) -> None:
    if pred.ndim != 2:
        raise ValueError(f"{fn} needs class probabilities; this is a regression task.")


@register("postprocess")
def prior_shift(
    pred: np.ndarray,
    y_train: pd.Series,
    x_test: pd.DataFrame,  # noqa: ARG001
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Shift probabilities so their mean moves toward the training class priors (classification).

    ``strength`` 0 leaves them unchanged, 1 matches the priors in log-odds.
    """
    _classification(pred, "prior_shift")
    priors = y_train.value_counts(normalize=True).sort_index().to_numpy()
    current = pred.mean(axis=0).clip(1e-12)
    shifted = pred * (priors / current) ** strength
    out: np.ndarray = shifted / shifted.sum(axis=1, keepdims=True)
    return out


@register("postprocess")
def temperature(
    pred: np.ndarray,
    y_train: pd.Series,  # noqa: ARG001
    x_test: pd.DataFrame,  # noqa: ARG001
    *,
    t: float = 1.0,
) -> np.ndarray:
    """Temperature scaling of probabilities: ``t > 1`` softens them, ``t < 1`` sharpens them (classification)."""
    _classification(pred, "temperature")
    logits = np.log(pred.clip(1e-12)) / t
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    out: np.ndarray = exp / exp.sum(axis=1, keepdims=True)
    return out


@register("postprocess")
def clip_to_train_range(
    pred: np.ndarray,
    y_train: pd.Series,
    x_test: pd.DataFrame,  # noqa: ARG001
) -> np.ndarray:
    """Clip predictions to the range of the training target (regression)."""
    if pred.ndim != 1:
        raise ValueError("clip_to_train_range needs a regression task.")
    out: np.ndarray = pred.clip(y_train.min(), y_train.max())
    return out


# --- examples --------------------------------------------------------------------------

# One example call per function, shown to the agents at the "examples" level of the tool/slot
# description (a stage-2 factor). Table and column names are illustrations, not this task's.
EXAMPLES: dict[str, dict[str, Any]] = {
    "zero_as_missing": {"table": "measurements", "columns": ["blood_pressure"]},
    "placeholder_as_missing": {"table": "users", "values": [-1, "unknown"], "columns": ["age", "country"]},
    "to_numeric": {"table": "products", "columns": ["price"]},
    "missing_indicators": {"table": "users", "columns": ["income"]},
    "drop_columns": {"table": "posts", "columns": ["body_text"]},
    "drop_duplicate_rows": {"table": "events"},
    "entity_columns": {"columns": ["country", "signup_channel"]},
    "lookup": {"fk_column": "team_id", "columns": ["league"]},
    "aggregate": {
        "table": "orders",
        "fk_column": "user_id",
        "aggregations": [{"op": "count"}, {"op": "mean", "column": "amount"}, {"op": "count_by", "column": "channel"}],
        "where": {"status": ["paid"]},
    },
    "child_columns": {"table": "profiles", "fk_column": "user_id", "columns": ["plan", "region"]},
    "target_history": {"ops": ["count", "mean", "last"]},
    "combine": {"a": "orders.sum(amount)", "b": "orders.count()", "op": "div", "name": "avg_order"},
    "group_stat": {"group": "users.country", "value": "orders.count()", "stat": "mean"},
    "count_encode": {"columns": ["users.country"]},
    "log_transform": {"columns": ["orders.sum(amount)"]},
    "datetime_parts": {"column": "date"},
    "drop_features": {"columns": ["users.signup_channel"]},
    "full": {},
    "random_views": {"n_views": 4},
    "recent": {},
    "stratified_views": {"n_views": 4},
    "oversample_minority": {"n_views": 2},
    "prior_shift": {"strength": 0.5},
    "temperature": {"t": 1.5},
    "clip_to_train_range": {},
}
