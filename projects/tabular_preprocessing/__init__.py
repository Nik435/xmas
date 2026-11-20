"""MAS architecture search for a relational pipeline modelled on TabFM-Auto.

- :mod:`tabular_preprocessing.inner_loop`: one architecture builds one pipeline for one task,
  and the pipeline is scored (agents, tools, pipeline, harness, architectures).
- :mod:`tabular_preprocessing.outer_loop`: an optimizer (grid, random, ...) searches MAS
  architectures and hyperparameters; every candidate runs through the inner loop on every task.

The inner loop's public names are re-exported here.
"""

from tabular_preprocessing.inner_loop import (
    REGISTRY,
    STAGES,
    Architecture,
    Call,
    Database,
    Evaluation,
    GroupArchitecture,
    InnerLoopResult,
    Pipeline,
    PipelineError,
    RelationalTask,
    RunResult,
    StageRecord,
    Table,
    Tables,
    build_tables,
    default_model,
    evaluate,
    fit_predict,
    load_relbench_task,
    run,
    run_inner_loop,
    score,
)

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
