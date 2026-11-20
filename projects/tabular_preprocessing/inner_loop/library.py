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
  rows strictly before the label row's own time.
- ``engineer(X_train, y_train, X_test, *, ...) -> (X_train, X_test)``: add
  columns to the merged frames. Label-free statistics may use train and test
  together (transductive).
- ``sample(X_train, y_train, X_test, max_rows, *, ...) -> list[row indices]``:
  the context views the predictor is fitted on, one fit per view.
- ``postprocess(pred, y_train, X_test, *, ...) -> pred``: adjust predictions;
  class probabilities ``(n, k)`` for classification, values ``(n,)`` for regression.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd

from tabular_preprocessing.inner_loop.data import Database

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
    """

    db: Database
    con: duckdb.DuckDBPyConnection
    entity_table: str
    entity_col: str
    time_col: str


AggOp = Literal["count", "sum", "mean", "min", "max", "std", "nunique", "time_since_last"]
_SQL = {"sum": "sum", "mean": "avg", "min": "min", "max": "max", "std": "stddev", "nunique": "count(DISTINCT"}


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
    return _entity_rows(ctx, seed, columns).add_prefix(f"{ctx.entity_table}.")


@register("merge")
def lookup(ctx: MergeContext, seed: pd.DataFrame, *, fk_column: str, columns: list[str]) -> pd.DataFrame:
    """Columns of the row a foreign key of the entity table points to (n:1), as ``<table>.<column>``."""
    fkeys = ctx.db[ctx.entity_table].fkeys or {}
    if fk_column not in fkeys:
        raise ValueError(f"{ctx.entity_table!r} has no foreign key {fk_column!r}; it has {sorted(fkeys)}.")
    parent = ctx.db[fkeys[fk_column]]
    if parent.pkey is None:
        raise ValueError(f"{fkeys[fk_column]!r} has no primary key to join on.")
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
    aggregations: list[dict[str, str]],
    window_days: int | None = None,
) -> pd.DataFrame:
    """Aggregate a table's rows that point to the entity (1:n), per label row and before its time.

    ``aggregations`` is a list like ``[{"op": "count"}, {"op": "mean", "column": "points"}]``;
    ops: count, sum, mean, min, max, std, nunique, time_since_last (days since the
    latest row, by the table's time column). With ``window_days``, only rows of
    the last that many days count. Columns are named ``<table>.<op>(<column>)``,
    plus ``@<window_days>d``.
    """
    child = ctx.db[table]
    if (child.fkeys or {}).get(fk_column) != ctx.entity_table:
        raise ValueError(f"{table}.{fk_column} is not a foreign key to {ctx.entity_table!r}.")
    if window_days is not None and (not isinstance(window_days, int) or window_days <= 0):
        raise ValueError("window_days must be a positive integer.")

    exprs, names, counts = [], [], []
    for agg in aggregations:
        op, column = agg.get("op"), agg.get("column")
        if op == "count":
            expr = "count(*)"
        elif op == "time_since_last":
            if child.time_col is None:
                raise ValueError(f"{table!r} has no time column for time_since_last.")
            column = child.time_col
            expr = f'date_diff(\'day\', max(f."{column}"), s."{ctx.time_col}")'
        elif op in _SQL:
            if column not in child.df.columns:
                raise ValueError(f"{table!r} has no column {column!r}.")
            if op != "nunique" and not pd.api.types.is_numeric_dtype(child.df[column]):
                raise ValueError(f"{op} needs a numeric column; {table}.{column} is {child.df[column].dtype}.")
            expr = f'{_SQL[op]}(f."{column}")' + (")" if op == "nunique" else "")
        else:
            raise ValueError(f"Unknown aggregation op {op!r}; choose from {list(AggOp.__args__)}.")  # type: ignore[attr-defined]
        name = f"{table}.{op}({column or ''})" + (f"@{window_days}d" if window_days else "")
        exprs.append(f'{expr} AS "{name}"')
        names.append(name)
        counts.append(op == "count")
    if not exprs:
        raise ValueError("aggregations must not be empty.")

    # strictly before the label row's time: a row at the prediction time is the future
    time_filter = ""
    if child.time_col is not None:
        time_filter = f' AND f."{child.time_col}" < s."{ctx.time_col}"'
        if window_days:
            time_filter += f' AND f."{child.time_col}" >= s."{ctx.time_col}" - INTERVAL \'{window_days} days\''
    frame = seed[[ctx.entity_col, ctx.time_col]].reset_index(drop=True)
    frame["__rid__"] = np.arange(len(frame))
    ctx.con.register("__seed__", frame)
    try:
        sql = (
            f'SELECT s."__rid__", {", ".join(exprs)} FROM "__seed__" s '
            f'JOIN "{table}" f ON f."{fk_column}" = s."{ctx.entity_col}"{time_filter} '
            f'GROUP BY s."__rid__", s."{ctx.time_col}"'
        )
        result = ctx.con.sql(sql).df()
    finally:
        ctx.con.unregister("__seed__")
    out = frame[["__rid__"]].merge(result, how="left", on="__rid__")[names]
    for name, is_count in zip(names, counts, strict=True):
        if is_count:  # no matching rows means a count of 0, not unknown
            out[name] = out[name].fillna(0)
    return out.astype(float).set_axis(seed.index)


# --- engineer --------------------------------------------------------------------------


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
        "aggregations": [{"op": "count"}, {"op": "mean", "column": "amount"}, {"op": "time_since_last"}],
        "window_days": 90,
    },
    "combine": {"a": "orders.sum(amount)@90d", "b": "orders.count()@90d", "op": "div", "name": "avg_order_90d"},
    "group_stat": {"group": "users.country", "value": "orders.count()@90d", "stat": "mean"},
    "count_encode": {"columns": ["users.country"]},
    "log_transform": {"columns": ["orders.sum(amount)@90d"]},
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
