"""The inner loop: one architecture builds one pipeline for one task, and the pipeline is scored.

    1. fill:     the architecture's system runs on the task and fills the pipeline
                 through its tools (a fresh workspace, so nothing leaks between runs)
    2. tables:   the filled pipeline builds the predictor's train and test tables
    3. predict:  the fixed predictor is fitted on the train table and predicts the test table
    4. score:    the predictions are scored, next to the identity pipeline's score

Train and test are the task's train and validation rows (RelArena's inner split);
the held-out test split stays untouched for the final evaluation. The agents' tools
never see the validation rows: they score drafts on a time split of the training
rows (see ``Workspace.tool_task``), so no system can tune its pipeline to the rows
it is judged on. A run that fails
is a result, not a crash: the reason is kept and the score is ``None``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tabular_preprocessing.inner_loop.architecture import Architecture
from tabular_preprocessing.inner_loop.data import RelationalTask
from tabular_preprocessing.inner_loop.harness import (
    ModelFactory,
    Tables,
    build_tables,
    default_model,
    evaluate,
    fit_predict,
    score,
)
from tabular_preprocessing.inner_loop.orchestrated import run_orchestrated
from tabular_preprocessing.inner_loop.pipeline import Pipeline, PipelineError
from tabular_preprocessing.inner_loop.tools import Edit, Workspace
from xmas import Trace


@dataclass
class InnerLoopResult:
    """Everything one inner-loop run produced.

    Attributes:
        task: The task's name.
        architecture: The architecture's code.
        pipeline: The pipeline the system submitted (the workspace's draft at the end of the run).
        trace: The system's run: every agent's prompt, answer and tool calls.
        edits: Every change to the pipeline, by tool, in order.
        score: The validation score of the pipeline, or ``None`` if the run failed.
        metric: ``"roc_auc"`` or ``"neg_mae"``.
        baseline: The identity pipeline's score, if computed.
        features: The feature columns of the tables the pipeline built.
        error: Why the run failed (the system or the pipeline), else ``None``.
        seconds: Wall time per phase: fill, tables, predict, baseline.
        tokens: Input and output tokens of all agents.
        evaluations: How many pipelines the agents' tools scored during the run.
        orchestration: For an orchestrated system: why it stopped, the agent turns used, and every decision.
    """

    task: str
    architecture: str
    pipeline: Pipeline
    trace: Trace
    edits: list[Edit]
    score: float | None = None
    metric: str | None = None
    baseline: float | None = None
    features: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    evaluations: int = 0
    orchestration: dict[str, Any] | None = None

    @property
    def improvement(self) -> float | None:
        """Score minus the identity pipeline's score, if both are known."""
        if self.score is None or self.baseline is None:
            return None
        return self.score - self.baseline


def run_inner_loop(
    architecture: Architecture,
    task: RelationalTask,
    llm: Any,
    llms: dict[str, Any] | None = None,
    models: dict[str, str] | None = None,
    model_factory: ModelFactory = default_model,
    max_rows: int = 10_000,
    max_columns: int = 500,
    max_loops: int = 3,
    max_tool_rounds: int = 20,
    max_evaluations: int = 30,
    autonomy: str = "bounded",
    tool_description: str = "detailed",
    merge_fills_sample: bool = False,
    max_steps: int = 12,
    routing: str = "ordered",
    baseline: bool = True,
) -> InnerLoopResult:
    """Run one architecture on one task: fill the pipeline, build the tables, predict, score.

    Args:
        architecture: The system's architecture.
        task: The task.
        llm: The default model of every agent.
        llms: Models by name, for agents bound by ``models``.
        models: Optional model name per agent name.
        model_factory: Builds the fixed predictor.
        max_rows: Most rows a context view may hold.
        max_columns: Most feature columns the predictor may receive.
        max_loops: How often one agent may run, which bounds every back-edge loop.
        max_tool_rounds: Tool-call turns per agent visit.
        max_evaluations: How many distinct pipelines the agents' tools may score in the run.
        autonomy: ``"fixed"``, ``"bounded"`` or ``"free"``, told to every agent (the limits are the
            ``max_*`` arguments).
        tool_description: How much the function menu says: ``"brief"``, ``"detailed"`` or ``"examples"``.
        merge_fills_sample: Whether group Merge also fills the sample slot.
        max_steps: For an orchestrated system: most agent turns the orchestrator may delegate.
        routing: For an orchestrated system: ``"ordered"`` or ``"free"``.
        baseline: Whether to score the identity pipeline too.

    Returns:
        The result; ``score`` is ``None`` and ``error`` says why if the run failed.
    """
    seconds: dict[str, float] = {}

    # 1. fill
    start = time.perf_counter()
    workspace = Workspace(
        task,
        model_factory=model_factory,
        max_rows=max_rows,
        max_columns=max_columns,
        max_evaluations=max_evaluations,
        tool_description=tool_description,
        merge_fills_sample=merge_fills_sample,
    )
    orchestration = None
    if architecture.orchestrator:
        run_ = run_orchestrated(
            architecture, workspace, llm, llms, models, max_steps, max_tool_rounds, autonomy, routing
        )
        trace = run_.trace
        orchestration = {"stopped": run_.stopped, "steps": run_.steps, "decisions": run_.decisions}
    else:
        mas = architecture.build(workspace, llm, llms, models, max_loops, max_tool_rounds, autonomy)
        trace = mas.run(task.describe())
    seconds["fill"] = time.perf_counter() - start
    result = InnerLoopResult(
        task=task.name,
        architecture=architecture.code,
        pipeline=workspace.pipeline,
        trace=trace,
        edits=list(workspace.edits),
        error=f"system: {trace.error}" if trace.error else None,
        seconds=seconds,
        tokens=_tokens(trace),
        evaluations=workspace.evaluations,
        orchestration=orchestration,
    )

    # 2. tables, 3. predict, 4. score; a failed system run may still have left a pipeline worth scoring
    try:
        start = time.perf_counter()
        tables: Tables = build_tables(workspace.pipeline, task, task.train, task.val, max_rows, max_columns)
        seconds["tables"] = time.perf_counter() - start
        start = time.perf_counter()
        run = fit_predict(tables, workspace.pipeline, task.task_type, model_factory)
        evaluation = score(run, task.val[task.target_col].to_numpy())
        seconds["predict"] = time.perf_counter() - start
        result.score, result.metric, result.features = evaluation.score, evaluation.metric, run.features
    except PipelineError as error:
        result.error = "; ".join(e for e in (result.error, f"pipeline: {error}") if e)

    if baseline:
        start = time.perf_counter()
        # on the validation rows, like the pipeline (the workspace's baseline is on the tools' rows)
        identity = evaluate(Pipeline.identity(), task, model_factory, max_rows, max_columns)
        seconds["baseline"] = time.perf_counter() - start
        result.baseline, result.metric = identity.score, identity.metric
    return result


def _tokens(trace: Trace) -> dict[str, int]:
    return {
        "input": sum(r.usage.input_tokens for r in trace.records),
        "output": sum(r.usage.output_tokens for r in trace.records),
    }
