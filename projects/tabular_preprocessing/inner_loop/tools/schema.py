"""Tool set: understand the schema (the Schema & Join Analyst; an executor without planner inherits it).

What the merge and feature plan needs: which tables the merge functions can reach
from the entity table (directly, as 1:1 tables, or two steps away through ``via``),
how many rows a 1:n relation has per entity, how much of it the rows to predict
still see, and how the tables' times relate to the label times. Everything is
measured on the cleaned database (after the draft's clean slot), because that is
what merging will read.
"""

# no `from __future__ import annotations`: xmas builds the tool schemas from the hints
import pandas as pd

from tabular_preprocessing.inner_loop.data import Database
from tabular_preprocessing.inner_loop.tools.text import bullets, fmt, pct
from tabular_preprocessing.inner_loop.tools.workspace import Workspace
from xmas.tools import Tool, tool


def schema_tools(workspace: Workspace, group: str) -> list[Tool]:
    """Build one group's schema tools.

    Args:
        workspace: The run's workspace.
        group: The tool-name prefix.

    Returns:
        ``<group>_list_relations``, ``<group>_inspect_relation`` and ``<group>_time_overview``.
    """
    task = workspace.tool_task  # the training rows, split by time: the validation rows stay unseen
    predict = workspace.task.val  # the rows the run is finally scored on: only their ids and times are read

    def db() -> Database:
        cleaned, _ = workspace.cleaned()
        return cleaned

    def list_relations() -> str:
        """What the merge functions can reach from the entity table: its own columns, n:1 lookups, 1:n aggregations."""
        tables = db()
        entity = tables[task.entity_table]
        own = [c for c in entity.df.columns if c not in entity.keys or c in (entity.fkeys or {})]
        out = [
            f"Entity table {task.entity_table!r} (label rows refer to it by {task.entity_col!r}).",
            "entity_columns can take: " + (", ".join(own) or "none"),
            "lookup (n:1, through a foreign key of the entity table):",
        ]
        lookups = [
            f"fk_column={fk} -> {parent}: columns "
            + ", ".join(c for c in tables[parent].df.columns if c not in tables[parent].keys)
            for fk, parent in (entity.fkeys or {}).items()
        ]
        out.append(bullets(lookups))
        out.append("aggregate (1:n, tables whose foreign key points to the entity table):")
        children, single, paths = [], [], []
        for name, t in tables.items():
            for fk, parent in (t.fkeys or {}).items():
                if parent != task.entity_table:
                    continue
                numeric = [c for c in t.df.columns if c not in t.keys and pd.api.types.is_numeric_dtype(t.df[c])]
                other = [c for c in t.df.columns if c not in t.keys and c not in numeric]
                per = t.df.groupby(fk).size()
                time = f"time column {t.time_col}" if t.time_col else "NO time column (rows are not time-filtered)"
                children.append(
                    f"table={name}, fk_column={fk}: {len(t.df)} rows, median {fmt(float(per.median()))} per entity, "
                    f"{time}; numeric columns: {', '.join(numeric) or 'none'}; categorical/text columns (mode, "
                    f"count_by, where): {', '.join(other) or 'none'}"
                )
                if per.max() <= 1:
                    single.append(f"table={name}, fk_column={fk}: columns {', '.join(numeric + other) or 'none'}")
                for link, target in (t.fkeys or {}).items():
                    if link != fk and target != task.entity_table and tables[target].pkey:
                        cols = [c for c in tables[target].df.columns if c not in tables[target].keys]
                        paths.append(f"table={target}, via={name}, fk_column={fk}: columns {', '.join(cols) or 'none'}")
        out.append(bullets(children))
        if single:
            out.append("child_columns (1:1, at most one row per entity: copy the columns):")
            out.append(bullets(single))
        if paths:
            out.append(
                "aggregate with via (two steps: the entity's rows in the via table, then the rows they point to):"
            )
            out.append(bullets(paths))
        out.append(
            f"target_history: the entity's own past labels ({task.target_col!r} of earlier label rows), "
            "e.g. how often it was positive before or its last label."
        )
        reached = {p.split(",", 1)[0].removeprefix("table=") for p in paths}
        unreachable = [
            n
            for n in tables
            if n != task.entity_table and not _touches(tables, n, task.entity_table) and n not in reached
        ]
        if unreachable:
            out.append(
                "Not reachable with the merge functions (no key to or from the entity table, nor through via): "
                + ", ".join(unreachable)
            )
        return "\n".join(out)

    def inspect_relation(table: str, fk_column: str) -> str:
        """How a 1:n table relates to the entity: rows per entity, coverage, and how much of it the rows to predict see.

        Compares the fitting rows with the rows the run is finally scored on: how many of the table's rows
        a label row sees before its time, and how old the latest one is. The rows to predict lie after the
        database's last rows, so they see older history; an aggregate is only as good as it is there.

        Args:
            table: The 1:n table.
            fk_column: Its foreign key to the entity table.
        """
        tables = db()
        if table not in tables:
            raise ValueError(f"No table {table!r}; tables are {sorted(tables)}.")
        t = tables[table]
        if (t.fkeys or {}).get(fk_column) != task.entity_table:
            raise ValueError(f"{table}.{fk_column} is not a foreign key to {task.entity_table!r}.")
        per = t.df.groupby(fk_column).size()
        entities = task.train[task.entity_col].nunique()
        covered = task.train[task.entity_col].isin(per.index).mean()
        out = [
            f"{table} via {fk_column}: {len(t.df)} rows over {per.size} entities; rows per entity: "
            f"median {fmt(float(per.median()))}, mean {fmt(float(per.mean()))}, max {int(per.max())}.",
            f"{pct(float(covered))} of the train label rows' entities ({entities} distinct) have at least one row.",
        ]
        if t.time_col is None:
            out.append("No time column: aggregate uses every row, also rows from after the label time (leakage risk).")
            return "\n".join(out)
        times = pd.to_datetime(t.df[t.time_col])
        out.append(f"Time column {t.time_col}: {times.min().date()} to {times.max().date()}.")
        rows = t.df[[fk_column, t.time_col]].rename(columns={fk_column: "_e", t.time_col: "_t"})
        for label, labels in (("fitting rows", task.train), ("rows to predict", predict)):
            out.append(f"{label}: " + _seen(labels, rows, task.entity_col, task.time_col))
        return "\n".join(out)

    def time_overview() -> str:
        """The label times (fitting and scoring rows) next to every table's time range, and tables without time."""
        tr, va = pd.to_datetime(task.train[task.time_col]), pd.to_datetime(task.val[task.time_col])
        final = pd.to_datetime(predict[task.time_col])
        out = [
            f"Label rows: fitting {tr.min().date()} to {tr.max().date()}, "
            f"scoring {va.min().date()} to {va.max().date()}.",
            f"Rows the run is finally scored on: {final.min().date()} to {final.max().date()}; they can only see "
            "the database's rows, which end where the tables' time ranges below end.",
            "Aggregates only see rows strictly before each label row's time.",
        ]
        lines = []
        for name, t in db().items():
            if t.time_col:
                times = pd.to_datetime(t.df[t.time_col])
                lines.append(f"{name}.{t.time_col}: {times.min().date()} to {times.max().date()}")
            else:
                lines.append(f"{name}: no time column (static, or changes are not dated)")
        return "\n".join(out) + "\n" + bullets(lines)

    fns = (list_relations, inspect_relation, time_overview)
    return [tool(name=f"{group}_{fn.__name__}")(fn) for fn in fns]


def _seen(labels: pd.DataFrame, rows: pd.DataFrame, entity_col: str, time_col: str) -> str:
    """What a sample of label rows sees of ``rows`` (``_e``, ``_t``): rows before its time, the latest one's age."""
    sample = labels.sample(n=min(300, len(labels)), random_state=0)[[entity_col, time_col]].reset_index()
    joined = sample.merge(rows, left_on=entity_col, right_on="_e", how="left")
    before = joined["_t"] < joined[time_col]
    counts = before.groupby(joined["index"]).sum()
    age = (joined[time_col] - joined["_t"]).dt.days.where(before).groupby(joined["index"]).min()
    latest = f"the latest is median {fmt(float(age.median()))} days old" if age.notna().any() else "none has any"
    return (
        f"a label row sees median {fmt(float(counts.median()))} rows, {pct(float((counts == 0).mean()))} see none; "
        f"{latest}"
    )


def _touches(tables: Database, name: str, entity: str) -> bool:
    t = tables[name]
    return entity in (t.fkeys or {}).values() or name in (tables[entity].fkeys or {}).values()
