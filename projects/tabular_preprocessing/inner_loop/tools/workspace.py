"""The workspace: the state one system run works on.

Tools are plain functions closing over a workspace, so a run's tools all see the
same task and the same draft pipeline: group Clean's steps are there when group
Merge starts. xmas does not tell a tool which query it serves, so a workspace
belongs to exactly one run; build a fresh system with fresh tools per task.

The workspace also scores pipelines for the tools that need a score (validation,
orchestration): with the same predictor and limits as the inner loop, cached per
pipeline, and counted against an evaluation budget, so a system cannot buy its
result with unlimited trial and error. The tools score on :attr:`Workspace.tool_task`,
a time split of the training rows: the validation rows the run is finally scored on
are never seen by an agent, so no system can tune its pipeline to them.

One check does look at the rows the run is finally scored on, but never at their labels:
:meth:`Workspace.prediction_missing` builds the draft's features for a sample of them and
says how empty each feature is there. Those rows lie after the database's cutoff, the
tools' scoring rows inside it, so a feature can be full in every tool score and still be
empty where the run is scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tabular_preprocessing.inner_loop.data import Database, RelationalTask
from tabular_preprocessing.inner_loop.harness import (
    ModelFactory,
    StageRecord,
    build_tables,
    clean_database,
    default_model,
    evaluate,
)
from tabular_preprocessing.inner_loop.pipeline import Pipeline, PipelineError


@dataclass
class Edit:
    """One change a tool made to the draft pipeline.

    Attributes:
        tool: The tool that made it.
        stage: The slot it changed.
        action: ``"add"``, ``"replace"`` or ``"remove"``.
        step: The step added, the new step, or the step removed, as text.
    """

    tool: str
    stage: str
    action: str
    step: str


@dataclass(frozen=True)
class Scored:
    """The validation score of one pipeline, or why it has none.

    Attributes:
        score: Higher is better; ``None`` if the pipeline failed.
        metric: ``"roc_auc"`` or ``"neg_mae"``.
        features: The feature columns the pipeline built.
        error: Why the pipeline failed, else ``None``.
    """

    score: float | None
    metric: str | None
    features: tuple[str, ...] = ()
    error: str | None = None


DRY_RUN_ROWS = 200  # label rows a dry run builds features for
PREDICTION_ROWS = 300  # rows to predict that prediction_missing builds features for

# what group Merge may put into the sample slot when it fills it: selection and stratification, no oversampling
MERGE_SAMPLE_FUNCTIONS: tuple[str, ...] = ("full", "random_views", "recent", "stratified_views")


class BudgetExhaustedError(RuntimeError):
    """Every evaluation of the run's budget is used up."""


@dataclass
class Workspace:
    """The task, the draft pipeline and the evaluation budget of one run.

    Attributes:
        task: The task the system works on.
        pipeline: The draft; starts as the identity pipeline and is what the run submits.
        edits: Every change, in order, for the analysis of who did what.
        model_factory: The fixed predictor, as in the inner loop.
        max_rows: Most rows a context view may hold.
        max_columns: Most feature columns the predictor may receive.
        max_evaluations: How many distinct pipelines the tools may score in this run.
        evaluations: How many they have scored so far.
        submitted: Whether the orchestrator submitted the pipeline.
        summary: The orchestrator's summary when it submitted.
        tool_task: What the tools see and score on: the training rows split by time
            (see :meth:`RelationalTask.tool_task`), never the validation rows.
        tool_description: How much the function menu says: ``"brief"``, ``"detailed"`` or ``"examples"``.
        merge_fills_sample: Whether group Merge also fills the sample slot (selection and
            stratification only, see :data:`MERGE_SAMPLE_FUNCTIONS`).
    """

    task: RelationalTask
    pipeline: Pipeline = field(default_factory=Pipeline.identity)
    edits: list[Edit] = field(default_factory=list)
    model_factory: ModelFactory = default_model
    max_rows: int = 10_000
    max_columns: int = 500
    max_evaluations: int = 30
    evaluations: int = 0
    submitted: bool = False
    summary: str | None = None
    tool_description: str = "detailed"
    merge_fills_sample: bool = False
    _scores: dict[str, Scored] = field(default_factory=dict, repr=False)
    _cleaned: dict[str, tuple[Database, list[StageRecord]]] = field(default_factory=dict, repr=False)
    _features: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)
    _missing: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    tool_task: RelationalTask = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Split the training rows for the tools."""
        self.tool_task = self.task.tool_task()

    def evaluate(self, pipeline: Pipeline) -> Scored:
        """Score a pipeline on the tools' scoring rows (late training rows); a pipeline scored before is free.

        Raises:
            BudgetExhaustedError: If a new pipeline is asked for after the budget is used up.
        """
        key = json.dumps(pipeline.to_dict(), sort_keys=True, default=str)
        if key in self._scores:
            return self._scores[key]
        if self.evaluations >= self.max_evaluations:
            raise BudgetExhaustedError(
                f"The evaluation budget of {self.max_evaluations} is used up; pipelines scored before stay available."
            )
        self.evaluations += 1
        self._scores[key] = self._score(pipeline)
        return self._scores[key]

    def cleaned(self, pipeline: Pipeline | None = None) -> tuple[Database, list[StageRecord]]:
        """The task's database after the pipeline's clean slot (the draft's by default), cached by that slot.

        Cleaning a large table takes seconds and most edits do not touch the clean slot, so the
        dry run of every edit and the profiling and schema tools share this result.
        """
        pipeline = pipeline or self.pipeline
        key = json.dumps([c.to_dict() for c in pipeline.clean], sort_keys=True, default=str)
        if key not in self._cleaned:
            # the draft and the one before are enough; each state copies the cleaned tables, and
            # on large databases (rel-event, rel-hm) a few copies exceed the job's memory
            if len(self._cleaned) >= 2:
                self._cleaned.pop(next(iter(self._cleaned)))
            self._cleaned[key] = clean_database(pipeline, self.task)
        return self._cleaned[key]

    def features(self, pipeline: Pipeline | None = None) -> tuple[str, ...]:
        """The feature columns a pipeline (the draft's by default) builds, from a dry run; cached.

        The dry run builds the tables on a small sample of the tools' rows with the real harness, so a
        pipeline that would break on the data fails here. Uses no evaluation of the budget.

        Raises:
            PipelineError: If the pipeline does not run on the data.
        """
        pipeline = pipeline or self.pipeline
        key = _key(pipeline)
        if key not in self._features:
            task = self.tool_task
            train = task.train.sample(n=min(DRY_RUN_ROWS, len(task.train)), random_state=0)
            test = task.val.sample(n=min(DRY_RUN_ROWS // 2, len(task.val)), random_state=0)
            built = build_tables(
                pipeline, task, train, test, self.max_rows, self.max_columns, self.cleaned(pipeline), task.train
            )
            self._features[key] = tuple(str(c) for c in built.x_train.columns)
        return self._features[key]

    def prediction_missing(self, pipeline: Pipeline | None = None) -> dict[str, float]:
        """Per feature, its missing share on a sample of the rows the run is finally scored on; cached.

        Those rows' labels are never read (the merge stage does not see them); their past labels are
        the fitting rows' labels, as in the real run. Uses no evaluation of the budget.

        Raises:
            PipelineError: If the pipeline does not run on the data.
        """
        pipeline = pipeline or self.pipeline
        key = _key(pipeline)
        if key not in self._missing:
            task = self.task
            train = task.train.sample(n=min(DRY_RUN_ROWS, len(task.train)), random_state=0)
            test = task.val.sample(n=min(PREDICTION_ROWS, len(task.val)), random_state=0)
            built = build_tables(
                pipeline, task, train, test, self.max_rows, self.max_columns, self.cleaned(pipeline), task.train
            )
            self._missing[key] = {str(c): float(built.x_test[c].isna().mean()) for c in built.x_test.columns}
        return self._missing[key]

    def baseline(self) -> Scored:
        """The identity pipeline's score, the reference for every comparison; free of budget."""
        key = json.dumps(Pipeline.identity().to_dict(), sort_keys=True)
        if key not in self._scores:
            self._scores[key] = self._score(Pipeline.identity())
        return self._scores[key]

    def _score(self, pipeline: Pipeline) -> Scored:
        try:
            result = evaluate(
                pipeline, self.tool_task, self.model_factory, self.max_rows, self.max_columns, self.cleaned(pipeline)
            )
        except PipelineError as error:
            return Scored(None, None, error=str(error))
        return Scored(result.score, result.metric, tuple(result.run.features))


def _key(pipeline: Pipeline) -> str:
    return json.dumps(pipeline.to_dict(), sort_keys=True, default=str)
