"""Relational tasks: a database, the entity rows to predict for, and their labels.

The loader goes through RelArena's own split code (as agentic-relational-ml
does), so temporal censoring matches RelArena's protocol: the database is
frozen at the validation cutoff, and the pipeline is fitted on the inner train
rows and scored on the inner validation rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class Table:
    """One database table.

    Attributes:
        df: The rows.
        pkey: Primary key column, if any.
        fkeys: Foreign key column -> the table it references.
        time_col: The column holding each row's timestamp, if the table has one.
    """

    df: pd.DataFrame
    pkey: str | None = None
    fkeys: dict[str, str] | None = None
    time_col: str | None = None

    @property
    def keys(self) -> list[str]:
        """The structural columns cleaning must leave alone: primary key, foreign keys, time."""
        cols = [self.pkey, *(self.fkeys or {}), self.time_col]
        return [c for c in cols if c is not None]


Database = dict[str, Table]


@dataclass(frozen=True)
class RelationalTask:
    """A prediction task over a database.

    Attributes:
        name: E.g. ``"rel-f1/driver-dnf"``.
        db: The tables, already censored to what the task may see.
        entity_table: The table whose rows are predicted for.
        entity_col: Column of the label rows holding the entity id.
        time_col: Column of the label rows holding each row's prediction time.
        target_col: Column of the label rows holding the target.
        task_type: ``"classification"`` or ``"regression"``.
        train: Label rows to fit on.
        val: Label rows to score on.
        horizon: How far ahead a label looks: the label at time ``t`` describes ``(t, t + horizon]``
            (RelBench's ``timedelta``). A past label is only known once its period is over, so
            ``target_history`` uses it from ``t + horizon`` on. ``None``: from just after ``t``.
    """

    name: str
    db: Database
    entity_table: str
    entity_col: str
    time_col: str
    target_col: str
    task_type: TaskType
    train: pd.DataFrame
    val: pd.DataFrame
    horizon: pd.Timedelta | None = None

    def tool_task(self, holdout: float = 0.2) -> RelationalTask:
        """The task the agents' tools score on: the training rows split by time, the validation rows unseen.

        The last ``holdout`` share of the training rows' time range becomes the tools' scoring rows,
        the rest their fitting rows. So agents can measure their drafts without ever touching the
        rows the run is finally scored on, and cannot tune to them. If the late rows hold only one
        class (classification), a seeded random split is used instead.

        Args:
            holdout: Share of the training rows to score on.

        Returns:
            The same task with ``train`` and ``val`` replaced by the two parts of ``train``.
        """
        times = pd.to_datetime(self.train[self.time_col])
        late = times >= times.quantile(1 - holdout)
        if (
            late.all()
            or not late.any()
            or (self.task_type == "classification" and self.train.loc[late, self.target_col].nunique() < 2)
        ):
            late = pd.Series(False, index=self.train.index)
            late[self.train.sample(frac=holdout, random_state=0).index] = True
        fit, score = self.train[~late], self.train[late]
        return replace(self, train=fit.reset_index(drop=True), val=score.reset_index(drop=True))

    def limit_train(self, max_rows: int | None) -> RelationalTask:
        """The task with at most ``max_rows`` training rows: the most recent ones (``None``: unchanged).

        For tasks with millions of label rows (rel-stack, rel-hm), whose features would otherwise be
        built for every row in every evaluation.
        """
        if max_rows is None or len(self.train) <= max_rows:
            return self
        recent = self.train.sort_values(self.time_col, kind="stable").tail(max_rows)
        return replace(self, train=recent.reset_index(drop=True))

    def describe(self) -> str:
        """The task and schema as text, without any rows: what agents read."""
        lines = [
            f"Task: {self.name} ({self.task_type})",
            f"Predict {self.target_col!r} for rows of {self.entity_table!r} (id column {self.entity_col!r}) "
            f"at time {self.time_col!r}. Training rows: {len(self.train)}.",
            "",
        ]
        for name, table in self.db.items():
            lines.append(f"## {name} ({len(table.df)} rows)")
            if table.pkey:
                lines.append(f"primary key: {table.pkey}")
            if table.time_col:
                lines.append(f"time column: {table.time_col}")
            for column, target in (table.fkeys or {}).items():
                lines.append(f"foreign key: {column} -> {target}")
            lines.append("columns: " + ", ".join(f"{c} ({t})" for c, t in table.df.dtypes.astype(str).items()))
            lines.append("")
        return "\n".join(lines)


def download_relbench_task(dataset: str, task: str) -> None:
    """Download a RelBench dataset and task into the relbench cache without loading them.

    Lets a cluster download once (e.g. on the login node) before many jobs read the cache.
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    get_dataset(dataset, download=True)
    get_task(dataset, task, download=True)


def load_relbench_task(dataset: str, task: str, split: str = "inner") -> RelationalTask:
    """Load a RelBench task through RelArena's own splits.

    Downloads the dataset on first use; ``relbench`` caches it afterwards. Only the requested
    split is built (each is a full, separately censored copy of the database).

    Args:
        dataset: E.g. ``"rel-f1"``.
        task: E.g. ``"driver-dnf"``.
        split: ``"inner"`` for the search: database frozen at the validation cutoff, fit on the
            train rows, score on the validation rows. ``"test"`` for the final evaluation: database
            frozen at the test cutoff, fit on train + validation rows, score on the test rows (their
            labels, hidden in RelArena's split, come from relbench).

    Returns:
        The task.

    Raises:
        ValueError: On an unknown split, or test rows whose labels cannot be matched.
    """
    # heavy imports (torch via relbench), only when a real task is loaded
    from relarena.dataset import RelBenchDatasetTask
    from relbench.base import TaskType as RelBenchTaskType

    source = RelBenchDatasetTask(dataset, task)
    rb = source.task
    if split == "inner":
        inner = source.inner_split()
        state, train, score = inner.db_state, inner.train_table.df, inner.eval_table.df
    elif split == "test":
        outer = source.outer_split()
        state = outer.db_state
        train = pd.concat([outer.train_table.df, outer.val_table.df], ignore_index=True)
        labels = rb.get_table("test", mask_input_cols=False).df
        keys = [rb.entity_col, rb.time_col]
        score = outer.eval_table.df[keys].merge(labels[[*keys, rb.target_col]], on=keys, how="left")
        if len(score) != len(outer.eval_table.df) or score[rb.target_col].isna().any():
            raise ValueError(f"{dataset}/{task}: the test labels do not match RelArena's test rows.")
    else:
        raise ValueError(f"Unknown split {split!r}; choose 'inner' or 'test'.")
    db = {
        name: Table(table.df, table.pkey_col, dict(table.fkey_col_to_pkey_table), table.time_col)
        for name, table in state.table_dict.items()
    }
    return RelationalTask(
        name=f"{dataset}/{task}",
        db=db,
        entity_table=rb.entity_table,
        entity_col=rb.entity_col,
        time_col=rb.time_col,
        target_col=rb.target_col,
        task_type="regression" if rb.task_type == RelBenchTaskType.REGRESSION else "classification",
        train=train.reset_index(drop=True),
        val=score.reset_index(drop=True),
        horizon=pd.Timedelta(rb.timedelta) if getattr(rb, "timedelta", None) is not None else None,
    )
