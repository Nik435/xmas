"""Tool set: understand the schema (the Schema & Join Analyst; an executor without planner inherits it).

What the merge and feature plan needs: which tables the merge functions can reach
from the entity table, how many rows a 1:n relation has per entity and per time
window, and how the tables' times relate to the label times. Everything is
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
        children = []
        for name, t in tables.items():
            for fk, parent in (t.fkeys or {}).items():
                if parent != task.entity_table:
                    continue
                numeric = [c for c in t.df.columns if c not in t.keys and pd.api.types.is_numeric_dtype(t.df[c])]
                per = t.df.groupby(fk).size()
                time = f"time column {t.time_col}" if t.time_col else "NO time column (rows are not time-filtered)"
                children.append(
                    f"table={name}, fk_column={fk}: {len(t.df)} rows, median {fmt(float(per.median()))} per entity, "
                    f"{time}; numeric columns: {', '.join(numeric) or 'none'}"
                )
        out.append(bullets(children))
        unreachable = [n for n in tables if n != task.entity_table and not _touches(tables, n, task.entity_table)]
        if unreachable:
            out.append(
                "Not reachable with the merge functions (no direct key to or from the entity table): "
                + ", ".join(unreachable)
            )
        return "\n".join(out)

    def inspect_relation(table: str, fk_column: str) -> str:
        """How a 1:n table relates to the entity: rows per entity, coverage, time range, rows per time window.

        The windows (30, 90, 365 days, all history) show how many rows a label row sees before its
        time, which is what aggregate with window_days would aggregate.

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
        # what a label row typically sees: a sample of train label rows, rows strictly before its time
        sample = task.train.sample(n=min(300, len(task.train)), random_state=0)
        rows = t.df[[fk_column, t.time_col]].rename(columns={fk_column: "_e", t.time_col: "_t"})
        joined = (
            sample[[task.entity_col, task.time_col]]
            .reset_index()
            .merge(rows, left_on=task.entity_col, right_on="_e", how="left")
        )
        before = joined["_t"] < joined[task.time_col]
        age = (joined[task.time_col] - joined["_t"]).dt.days
        for window in (30, 90, 365, None):
            mask = before & (age <= window) if window else before
            counts = mask.groupby(joined["index"]).sum()
            label = f"last {window} days" if window else "all history"
            out.append(
                f"{label}: a label row sees median {fmt(float(counts.median()))} rows, "
                f"{pct(float((counts == 0).mean()))} see none"
            )
        return "\n".join(out)

    def time_overview() -> str:
        """The label times (fitting and scoring rows) next to every table's time range, and tables without time."""
        tr, va = pd.to_datetime(task.train[task.time_col]), pd.to_datetime(task.val[task.time_col])
        out = [
            f"Label rows: fitting {tr.min().date()} to {tr.max().date()}, "
            f"scoring {va.min().date()} to {va.max().date()}.",
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


def _touches(tables: Database, name: str, entity: str) -> bool:
    t = tables[name]
    return entity in (t.fkeys or {}).values() or name in (tables[entity].fkeys or {}).values()
