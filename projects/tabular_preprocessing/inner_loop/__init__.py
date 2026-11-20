"""The inner loop: one architecture builds one pipeline for one task, and the pipeline is scored.

    fill (the architecture's agents, with their tools) -> build tables -> predictor -> score

| Module | What it holds |
|---|---|
| :mod:`.data` | the relational task: database, label rows, RelBench loader |
| :mod:`.library` | the fixed functions per pipeline slot |
| :mod:`.pipeline` | the pipeline as plain data (what the agents fill) |
| :mod:`.harness` | build_tables -> fit_predict -> score, around the fixed predictor |
| :mod:`.architecture` | the search space as xmas systems |
| :mod:`.orchestrated` | the orchestrator's loop (who works next, turn by turn) |
| :mod:`.agents` | the agent types |
| :mod:`.tools` | the agents' tool sets and the run's workspace |
| :mod:`.loop` | :func:`run_inner_loop`: all of it for one architecture and one task |
"""

from tabular_preprocessing.inner_loop.architecture import Architecture, GroupArchitecture
from tabular_preprocessing.inner_loop.data import Database, RelationalTask, Table, load_relbench_task
from tabular_preprocessing.inner_loop.harness import (
    Evaluation,
    RunResult,
    StageRecord,
    Tables,
    build_tables,
    default_model,
    evaluate,
    fit_predict,
    run,
    score,
)
from tabular_preprocessing.inner_loop.library import REGISTRY, STAGES
from tabular_preprocessing.inner_loop.loop import InnerLoopResult, run_inner_loop
from tabular_preprocessing.inner_loop.pipeline import Call, Pipeline, PipelineError

__all__ = [
    "REGISTRY",
    "STAGES",
    "Architecture",
    "Call",
    "Database",
    "Evaluation",
    "GroupArchitecture",
    "InnerLoopResult",
    "Pipeline",
    "PipelineError",
    "RelationalTask",
    "RunResult",
    "StageRecord",
    "Table",
    "Tables",
    "build_tables",
    "default_model",
    "evaluate",
    "fit_predict",
    "load_relbench_task",
    "run",
    "run_inner_loop",
    "score",
]
