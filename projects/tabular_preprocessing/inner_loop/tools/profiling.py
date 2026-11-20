"""Tool set: profile the data (the Data Profiler; an executor without planner inherits it).

What the cleaning plan needs: types, missing values, placeholder values,
cardinality, duplicates, text columns that hold numbers, and the target. Every
tool can look at the raw tables or, with ``after_cleaning``, at the tables after
the draft's clean slot, so an executor can see what its steps changed.

Keys and time columns are listed but not profiled: cleaning can never change them.
"""

# no `from __future__ import annotations`: xmas builds the tool schemas from the hints
from typing import Any

import pandas as pd

from tabular_preprocessing.inner_loop.data import Database, Table
from tabular_preprocessing.inner_loop.tools.text import bullets, fmt, pct, table
from tabular_preprocessing.inner_loop.tools.workspace import Workspace
from xmas.tools import Tool, tool

# values that often stand for "missing" in real tables
PLACEHOLDERS: tuple[Any, ...] = (
    -1,
    -9,
    -99,
    -999,
    -9999,
    0,
    "",
    "?",
    "-",
    "\\N",
    "NA",
    "N/A",
    "null",
    "NULL",
    "None",
    "none",
    "unknown",
    "Unknown",
    "nan",
)


def profiling_tools(workspace: Workspace, group: str) -> list[Tool]:
    """Build one group's profiling tools.

    Args:
        workspace: The run's workspace.
        group: The tool-name prefix.

    Returns:
        ``<group>_list_tables``, ``<group>_profile_table``, ``<group>_profile_column``,
        ``<group>_find_placeholders``, ``<group>_target_distribution`` and ``<group>_sample_rows``.
    """
    task = workspace.tool_task  # the training rows, split by time: the validation rows stay unseen

    def tables(after_cleaning: bool) -> Database:
        if not after_cleaning:
            return task.db
        db, _ = workspace.cleaned()
        return db

    def get(name: str, after_cleaning: bool) -> Table:
        db = tables(after_cleaning)
        if name not in db:
            raise ValueError(f"No table {name!r}; tables are {sorted(db)}.")
        return db[name]

    def list_tables(after_cleaning: bool = False) -> str:
        """Every table: rows, columns, keys, time column, missing values and duplicated rows.

        Args:
            after_cleaning: Show the tables after the draft's clean steps instead of the raw ones.
        """
        lines = []
        for name, t in tables(after_cleaning).items():
            data = t.df.drop(columns=t.keys)
            missing = float(data.isna().mean().mean()) if data.size else 0.0
            dupes = _duplicates(t)
            keys = ", ".join(f"{k}" for k in t.keys) or "none"
            lines.append(
                f"{name}: {len(t.df)} rows, {data.shape[1]} data columns, missing {pct(missing)}, "
                f"{dupes} duplicated rows; keys/time: {keys}"
            )
        head = f"Entity table: {task.entity_table}. Tables" + (" after cleaning:" if after_cleaning else ":")
        return head + "\n" + bullets(lines)

    def profile_table(table_name: str, after_cleaning: bool = False) -> str:
        """Per data column of a table: type, missing share, distinct values, range or top values, and warnings.

        Warnings point at what cleaning may fix: placeholder values, numbers stored as text,
        constant columns, identifier-like text columns, heavy skew.

        Args:
            table_name: The table.
            after_cleaning: Profile the table after the draft's clean steps.
        """
        t = get(table_name, after_cleaning)
        data = t.df.drop(columns=t.keys)
        lines = [_column_line(data[c]) for c in data.columns]
        dupes = _duplicates(t)
        head = (
            f"{table_name}: {len(t.df)} rows; {dupes} rows repeat an earlier row in everything but the primary key. "
            f"Protected (keys/time): {', '.join(t.keys) or 'none'}."
        )
        return head + "\n" + bullets(lines)

    def profile_column(table_name: str, column: str, after_cleaning: bool = False) -> str:
        """One column in depth: the most frequent values with counts, quantiles, and placeholder candidates.

        Args:
            table_name: The table.
            column: The column.
            after_cleaning: Profile the column after the draft's clean steps.
        """
        t = get(table_name, after_cleaning)
        if column not in t.df.columns:
            raise ValueError(f"{table_name!r} has no column {column!r}; columns: {list(t.df.columns)}.")
        s = t.df[column]
        out = [_column_line(s)]
        counts = s.value_counts(dropna=False).head(15)
        out.append(
            "Most frequent values:\n" + bullets([f"{fmt(v)}: {n} ({pct(n / len(s))})" for v, n in counts.items()])
        )
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            q = s.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
            out.append("Quantiles: " + ", ".join(f"{int(k * 100)}%={fmt(float(v))}" for k, v in q.items()))
        found = _placeholders(s)
        out.append("Placeholder candidates: " + (", ".join(f"{fmt(v)} ({n} rows)" for v, n in found) or "none"))
        return "\n".join(out)

    def find_placeholders(after_cleaning: bool = False) -> str:
        """Every column of every table that holds values which often mean 'missing' (e.g. -1, 0, '?', 'unknown').

        A candidate is only a hint: 0 can be a real value. Check with profile_column.

        Args:
            after_cleaning: Search the tables after the draft's clean steps.
        """
        lines = []
        for name, t in tables(after_cleaning).items():
            for c in t.df.columns.drop(t.keys):
                found = _placeholders(t.df[c])
                if found:
                    lines.append(f"{name}.{c}: " + ", ".join(f"{fmt(v)} ({pct(n / len(t.df))})" for v, n in found))
        return "Placeholder candidates:\n" + bullets(lines)

    def target_distribution() -> str:
        """The target of the training rows: its distribution in your fitting and scoring rows, and over time."""
        y_train, y_val = task.train[task.target_col], task.val[task.target_col]
        out = [f"Target {task.target_col!r} ({task.task_type}); {len(y_train)} fitting and {len(y_val)} scoring rows."]
        for name, y in (("fitting rows", y_train), ("scoring rows (latest training rows)", y_val)):
            if task.task_type == "classification":
                shares = y.value_counts(normalize=True).sort_index()
                out.append(f"{name}: " + ", ".join(f"{k}: {pct(v)}" for k, v in shares.items()))
            else:
                out.append(
                    f"{name}: mean {fmt(float(y.mean()))}, median {fmt(float(y.median()))}, std {fmt(float(y.std()))}"
                )
        years = pd.to_datetime(task.train[task.time_col]).dt.year
        by_year = task.train.groupby(years)[task.target_col]
        stat = by_year.mean() if pd.api.types.is_numeric_dtype(y_train) else by_year.agg(lambda s: s.mode().iloc[0])
        out.append("Train target mean by year: " + ", ".join(f"{k}: {fmt(float(v))}" for k, v in stat.tail(12).items()))
        return "\n".join(out)

    def sample_rows(table_name: str, n: int = 5, after_cleaning: bool = False) -> str:
        """A few random rows of a table, every column.

        Args:
            table_name: The table.
            n: How many rows (at most 20).
            after_cleaning: Show the rows after the draft's clean steps.
        """
        t = get(table_name, after_cleaning)
        rows = t.df.sample(n=min(max(n, 1), 20, len(t.df)), random_state=0)
        return f"{table_name}, {len(rows)} random rows:\n{table(rows, rows=20)}"

    fns = (list_tables, profile_table, profile_column, find_placeholders, target_distribution, sample_rows)
    return [tool(name=f"{group}_{fn.__name__}")(fn) for fn in fns]


def _duplicates(t: Table) -> int:
    """Rows equal to an earlier row in everything but the primary key: what drop_duplicate_rows removes."""
    rest = t.df.drop(columns=[t.pkey]) if t.pkey else t.df
    return int(rest.duplicated().sum()) if rest.shape[1] else 0


def _placeholders(s: pd.Series) -> list[tuple[Any, int]]:
    found = []
    text = s.astype(str) if not pd.api.types.is_numeric_dtype(s) else None
    for value in PLACEHOLDERS:
        if text is not None and isinstance(value, str):
            n = int((text.str.strip() == value).sum())
        elif text is None and not isinstance(value, str):
            n = int((s == value).sum())
        else:
            continue
        # 0 is a placeholder candidate only where it is rare-ish and the column is otherwise positive
        if n and (value != 0 or (s[s != 0].min() > 0 and n / len(s) < 0.5)):
            found.append((value, n))
    return found


def _column_line(s: pd.Series) -> str:
    n = len(s)
    missing = float(s.isna().mean()) if n else 0.0
    unique = int(s.nunique())
    line = f"{s.name}: {s.dtype}, missing {pct(missing)}, {unique} distinct"
    warnings = []
    if unique <= 1:
        warnings.append("constant")
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        values = s.dropna().astype(float)
        if len(values):
            low, mid, high = (fmt(float(v)) for v in (values.min(), values.median(), values.max()))
            line += f", min {low}, median {mid}, max {high}"
            skew = float(values.skew()) if len(values) > 2 else 0.0
            if abs(skew) > 3:
                warnings.append(f"skewed ({fmt(skew)})")
    elif pd.api.types.is_datetime64_any_dtype(s):
        line += f", from {s.min()} to {s.max()}"
    else:
        top = s.value_counts().head(3)
        line += ", top: " + ", ".join(f"{fmt(v)} ({n})" for v, n in top.items())
        text = s.dropna().astype(str)
        numeric = float(pd.to_numeric(text, errors="coerce").notna().mean()) if len(text) else 0.0
        if numeric > 0.5:
            warnings.append(f"{pct(numeric)} parse as numbers (text column holding numbers?)")
        if n and unique / n > 0.9 and unique > 50:
            warnings.append("nearly unique text (identifier or free text?)")
    found = _placeholders(s)
    if found:
        warnings.append("placeholder candidates: " + ", ".join(fmt(v) for v, _ in found))
    if warnings:
        line += " | " + "; ".join(warnings)
    return line
