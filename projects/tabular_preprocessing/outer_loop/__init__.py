"""The outer loop: an optimizer searches MAS architectures and hyperparameters.

| Module | What it holds |
|---|---|
| :mod:`.config` | the study configuration: tasks, the search space, the optimizer, budgets |
| :mod:`.optimizers` | the swappable optimizers (``grid``, ``random``) behind one ask/tell interface |
| :mod:`.study` | the loop: ask a candidate, run the inner loop on every task, tell the objective |
| :mod:`.report` | the final test evaluation, hyperparameter effects, the export to stage 2, the plan |
| :mod:`.cli` | the command line |

Run it with ``python -m tabular_preprocessing.outer_loop <config.yaml>``.
"""

from tabular_preprocessing.outer_loop.config import (
    Candidate,
    ConfigError,
    ExperimentConfig,
    Factors,
    OptimizerSettings,
    RunSpec,
    SearchSpace,
)
from tabular_preprocessing.outer_loop.optimizers import OPTIMIZERS, GridSearch, Optimizer, RandomSearch, make_optimizer
from tabular_preprocessing.outer_loop.report import effects, export_stage2, final_evaluation, plan
from tabular_preprocessing.outer_loop.study import merge_shards, objective_stats, run_study, shard_of, summarize

__all__ = [
    "OPTIMIZERS",
    "Candidate",
    "ConfigError",
    "ExperimentConfig",
    "Factors",
    "GridSearch",
    "Optimizer",
    "OptimizerSettings",
    "RandomSearch",
    "RunSpec",
    "SearchSpace",
    "effects",
    "export_stage2",
    "final_evaluation",
    "make_optimizer",
    "merge_shards",
    "objective_stats",
    "plan",
    "run_study",
    "shard_of",
    "summarize",
]
