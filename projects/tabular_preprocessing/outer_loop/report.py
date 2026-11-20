"""After (or during) a study: the final test evaluation, the hyperparameter effects, the step to stage 2.

- :func:`final_evaluation`: the best candidates' pipelines, re-applied to RelArena's held-out test
  split (database frozen at the test cutoff, fit on train + validation rows, scored on the test rows).
  No agent runs again: a pipeline is plain data. Only these numbers are comparable to the RelArena
  leaderboard; the search's own values are measured on the validation rows it selected on.
- :func:`baselines`: references without any agent on every task, all with the study's predictor:
  identity (the task table only), entity table (plus the entity's own columns, RelBench's LightGBM
  recipe) and dfs (RDBLearn's Deep Feature Synthesis features).
- :func:`effects`: per hyperparameter and level, the mean value of the trials that had it (marginal
  means), e.g. whether a validator in group Merge helps on average.
- :func:`export_stage2`: a stage-2 configuration with the best stage-1 architectures, ready for
  varying the further factors.
- :func:`plan`: what a configuration will run, with a cost estimate from the runs so far.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tabular_preprocessing.inner_loop.data import RelationalTask, load_relbench_task
from tabular_preprocessing.inner_loop.harness import Evaluation, ModelFactory, Tables, evaluate, fit_predict
from tabular_preprocessing.inner_loop.harness import score as score_run
from tabular_preprocessing.inner_loop.library import REGISTRY
from tabular_preprocessing.inner_loop.pipeline import Pipeline, PipelineError
from tabular_preprocessing.outer_loop.config import PREDICTORS, ExperimentConfig
from tabular_preprocessing.outer_loop.study import _append, n_trials_of, objective_stats, read_jsonl

TestLoader = Callable[[str, str], RelationalTask]


def _test_split(dataset: str, task: str) -> RelationalTask:
    return load_relbench_task(dataset, task, split="test")


def entity_table_pipeline(task: RelationalTask) -> Pipeline:
    """The fixed pipeline that adds every column of the entity's own row and nothing else.

    "Prediction on just the target table": no other table, no cleaning, no aggregate. Keys and the
    entity table's time column are left out (the label table already has the time).
    """
    table = task.db[task.entity_table]
    columns = [c for c in table.df.columns if c not in table.keys]
    if not columns:
        return Pipeline.identity()
    return Pipeline.from_dict({"merge": [{"fn": "entity_columns", "columns": columns}]})


BASELINES: dict[str, Callable[[RelationalTask], Pipeline]] = {
    "identity": lambda _task: Pipeline.identity(),
    "entity_table": entity_table_pipeline,
}


DFS_DEPTH = 2  # RDBLearn's default depth


def dfs_evaluation(task: RelationalTask, split: str, model_factory: ModelFactory, max_rows: int) -> Evaluation:
    """RDBLearn's features for the task's rows, fitted and scored like any pipeline.

    The features come from RelArena's implementation of RDBLearn (Deep Feature Synthesis over the
    foreign keys at depth 2, past-label aggregates per entity, time-until-cutoff diffs, the entity
    key and calendar features of the cutoff); only RDBLearn's predictor (TabPFN) is replaced by the
    study's. Same training rows, same sampling, same metric as every run.

    Args:
        task: The task (its rows; the database is loaded again in RelBench's form).
        split: ``"val"`` (RelArena's inner split) or ``"test"`` (the outer split).
        model_factory: The study's predictor.
        max_rows: Most training rows per fit, as in the runs.
    """
    from relarena.dataset import RelBenchDatasetTask
    from relarena_core.featurization.dfs import build_dfs_features
    from relbench.base import Table as RelBenchTable

    dataset, name = task.name.split("/")
    source = RelBenchDatasetTask(dataset, name)
    rb = source.task
    db = (source.inner_split() if split == "val" else source.outer_split()).db_state

    def table(df: pd.DataFrame) -> RelBenchTable:
        return RelBenchTable(
            df=df.reset_index(drop=True),
            fkey_col_to_pkey_table={rb.entity_col: rb.entity_table},
            pkey_col=None,
            time_col=rb.time_col,
        )

    train = table(task.train)
    x_train, x_test = (
        build_dfs_features(
            rb, db, t, depth=DFS_DEPTH, max_depth=DFS_DEPTH, history_table=train, keep_anchor_columns=True
        )[0]
        for t in (train, table(task.val))
    )
    if len(x_train) != len(task.train) or len(x_test) != len(task.val):
        raise PipelineError("dfs: the features do not have one row per label row.")
    x_train, x_test = x_train.reset_index(drop=True), x_test.reindex(columns=x_train.columns).reset_index(drop=True)
    y_train = task.train[task.target_col].reset_index(drop=True)
    views = REGISTRY["sample"]["full"](x_train, y_train, x_test, max_rows)
    result = fit_predict(
        Tables(x_train, y_train, x_test, views, []), Pipeline.identity(), task.task_type, model_factory
    )
    return score_run(result, task.val[task.target_col].to_numpy())


def baselines(
    config: ExperimentConfig,
    task_loader: Callable[[str, str], RelationalTask] = load_relbench_task,
    test_loader: TestLoader | None = None,
    log: Callable[[str], None] = print,
    dfs: Callable[[RelationalTask, str, ModelFactory, int], Evaluation] | None = dfs_evaluation,
) -> pd.DataFrame:
    """Score the references on every task, on the validation and the test rows.

    The fixed pipelines of :data:`BASELINES`, plus ``dfs`` (:func:`dfs_evaluation`; ``None`` leaves
    it out). Same predictor and limits as the study for all of them, no agent and no LLM: ``val`` is
    the study's own split (comparable to the runs' scores), ``test`` RelArena's held-out split
    (comparable to ``--final``). Results go to ``<output_dir>/baselines.jsonl``; what is already
    there is not scored again, so a crash on one task (e.g. a big database) keeps the others.

    Returns:
        One row per task, split and baseline, with its relative improvement over the identity pipeline.
    """
    path = config.output_dir / "baselines.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(path)
    done = {(row["task"], row["split"]) for row in rows}
    factory = PREDICTORS[config.predictor]
    loaders = {"val": task_loader, "test": test_loader or _test_split}
    for name in config.tasks:
        for split, loader in loaders.items():  # one task and split in memory at a time
            if (name, split) in done:
                continue
            log(f"Baselines on {name} ({split}) ...")
            task = loader(*name.split("/")).limit_train(config.max_train_rows)
            scored = {}
            for baseline, make in BASELINES.items():
                try:
                    result = evaluate(make(task), task, factory, config.max_rows, config.max_columns)
                    scored[baseline] = (result.score, result.metric, len(result.run.features))
                except PipelineError as error:
                    log(f"  {baseline}: {error}")
                    scored[baseline] = (None, None, 0)
            rows_only = replace(task, db={})  # dfs loads the database itself: never two copies at once
            del task
            if dfs is not None:
                try:
                    result = dfs(rows_only, split, factory, config.max_rows)
                    scored["dfs"] = (result.score, result.metric, len(result.run.features))
                except Exception as error:  # an optional reference: report it, keep the others
                    log(f"  dfs: {type(error).__name__}: {error}")
                    scored["dfs"] = (None, None, 0)
            base = scored["identity"][0]
            for baseline, (score, metric, n_features) in scored.items():
                relative = None if score is None or not base else (score - base) / abs(base)
                row = {
                    "task": name,
                    "split": split,
                    "baseline": baseline,
                    "score": score,
                    "metric": metric,
                    "n_features": n_features,
                    "relative_improvement": relative,
                }
                _append(path, row)
                rows.append(row)
    return pd.DataFrame(rows)


def _study_rows(config: ExperimentConfig, name: str) -> list[dict[str, Any]]:
    # the merged file if there is one, else every shard's, each row once
    merged = config.output_dir / name
    paths = [merged] if merged.exists() else sorted(config.output_dir.glob(f"shards/*/{name}"))
    seen: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            seen[row.get("run_id") or row["candidate"]] = row
    return list(seen.values())


def compare(config: ExperimentConfig) -> pd.DataFrame:
    """Per task, the test scores of the systems next to every baseline, all with the same predictor.

    Columns: the references (``identity``, ``entity_table``, ``dfs``, from :func:`baselines`) and
    the multi-agent systems (``mas_mean``: every architecture and repeat; ``mas_selected``: the
    architecture with the best search value, so chosen without the test rows; ``solo``), from
    ``--final``. AUC is higher-is-better, MAE is shown positive and lower-is-better, as on the
    RelArena leaderboard.
    """
    table: dict[str, dict[str, Any]] = {name: {"task": name} for name in config.tasks}

    def put(task: str, column: str, score: float | None, metric: str | None = None) -> None:
        if task not in table or score is None or pd.isna(score):
            return
        if metric == "neg_mae":
            score, metric = -score, "mae"
        if metric:
            table[task].setdefault("metric", metric)
        table[task][column] = score

    for row in read_jsonl(config.output_dir / "baselines.jsonl"):
        if row["split"] == "test":
            put(row["task"], row["baseline"], row["score"], row["metric"])

    final = pd.DataFrame(_study_rows(config, "final.jsonl"))
    trials = [t for t in _study_rows(config, "trials.jsonl") if t["architecture"] != "solo"]
    if not final.empty:
        metrics = {r["task"]: r["metric"] for r in _study_rows(config, "results.jsonl")}
        final["solo"] = final["candidate"].str.startswith("solo|")
        selected = max(trials, key=lambda t: t["value"])["candidate"] if trials else None
        for task, rows in final.groupby("task"):
            put(task, "mas_mean", rows.loc[~rows["solo"], "score"].mean(), metrics.get(task))
            put(task, "mas_selected", rows.loc[rows["candidate"] == selected, "score"].mean(), metrics.get(task))
            put(task, "solo", rows.loc[rows["solo"], "score"].mean(), metrics.get(task))

    return pd.DataFrame(list(table.values()))


def final_evaluation(
    config: ExperimentConfig,
    top: int = 5,
    task_loader: TestLoader = _test_split,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Re-apply the best ``top`` candidates' pipelines to the test split and score them there.

    Every run of a candidate (each task and repeat) contributes its own pipeline. Results go to
    ``<output_dir>/final.jsonl`` (rewritten each time).

    Args:
        config: The study.
        top: How many of the best trials (by their search value) to evaluate.
        task_loader: Loads a task's test split from dataset and task name.
        log: Where progress lines go.

    Returns:
        One row per candidate: its search value and its value on the test split, best search value first.
    """
    trials = sorted(read_jsonl(config.output_dir / "trials.jsonl"), key=lambda t: t["value"], reverse=True)[:top]
    results = {r["run_id"]: r for r in read_jsonl(config.output_dir / "results.jsonl")}
    factory = PREDICTORS[config.predictor]
    out_path = config.output_dir / "final.jsonl"
    out_path.unlink(missing_ok=True)
    by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for trial in trials:
        for run_id in trial["runs"]:
            run = results[run_id]
            by_task.setdefault(run["task"], []).append((trial, run))

    rows: dict[str, dict[str, dict[str, Any]]] = {trial["candidate"]: {} for trial in trials}
    for name, pairs in by_task.items():  # one task in memory at a time
        log(f"Loading the test split of {name} ...")
        task = task_loader(*name.split("/")).limit_train(config.max_train_rows)
        base = evaluate(Pipeline.identity(), task, factory, config.max_rows).score
        for trial, run in pairs:
            try:
                score: float | None = evaluate(
                    Pipeline.from_dict(run["pipeline"]), task, factory, config.max_rows, config.max_columns
                ).score
            except PipelineError as error:
                log(f"  {run['run_id']}: the pipeline does not run on the test split: {error}")
                score = None
            row = {
                "trial": trial["trial"],
                "candidate": trial["candidate"],
                "run_id": run["run_id"],
                "task": name,
                "repeat": run["repeat"],
                "validation_score": run["score"],
                "score": score,
                "baseline": base,
                "improvement": None if score is None else score - base,
            }
            _append(out_path, row)
            rows[trial["candidate"]][run["run_id"]] = row
        del task

    summary = []
    for trial in trials:
        stats = objective_stats([rows[trial["candidate"]][run_id] for run_id in trial["runs"]], config.objective)
        summary.append(
            {
                "trial": trial["trial"],
                "candidate": trial["candidate"],
                "search_value": trial["value"],
                "test_value": stats["value"],
                "test_value_se": stats["value_se"],
                **{f"test:{task}": v for task, v in stats["per_task"].items()},
            }
        )
        log(f"{trial['candidate']}: search {trial['value']:+.4f} -> test {stats['value']:+.4f}")
    return pd.DataFrame(summary)


def effects(config: ExperimentConfig) -> pd.DataFrame:
    """Per hyperparameter and level: how many trials had it, and their mean value with its standard error.

    Marginal means over the trials so far; hyperparameters with a single level are left out, and
    inactive ones (e.g. ``role`` at size 1) appear as their own level ``-``. The single-agent
    baseline (``solo``) is left out: it has no groups, so it would count as a team of size 1.
    """
    trials = [t for t in read_jsonl(config.output_dir / "trials.jsonl") if t["architecture"] != "solo"]
    if not trials:
        return pd.DataFrame()
    frame = pd.DataFrame([{**t["config"], "value": t["value"]} for t in trials])
    frame = frame.drop(columns=["architecture"], errors="ignore")
    rows = []
    for column in frame.columns.drop("value"):
        levels = frame[column].map(lambda v: "-" if v is None else str(v))
        if levels.nunique() < 2:
            continue
        for level, values in frame["value"].groupby(levels):
            se = values.std() / math.sqrt(len(values)) if len(values) > 1 else None
            rows.append(
                {"hyperparameter": column, "level": level, "trials": len(values), "mean": values.mean(), "se": se}
            )
    return pd.DataFrame(rows)


def export_stage2(config: ExperimentConfig, top: int, path: str | Path) -> Path:
    """Write a stage-2 configuration: the ``top`` best architectures of this study, the same factors.

    Edit its ``factors`` (give the ones to test several levels) before running it.

    Returns:
        The path written.
    """
    trials = sorted(read_jsonl(config.output_dir / "trials.jsonl"), key=lambda t: t["value"], reverse=True)
    codes: list[str] = []
    for trial in trials:
        code = trial["architecture"].removesuffix("|orchestrator")
        if code not in codes:
            codes.append(code)
        if len(codes) == top:
            break
    data = dict(config.source)
    data["name"] = f"{config.name}_stage2"
    data["output_dir"] = f"{config.output_dir}_stage2"
    data["architecture"] = {"codes": codes}
    data["optimizer"] = {"name": "grid", "seed": config.optimizer.seed}
    header = (
        f"# Stage 2 from {config.name}: its {len(codes)} best architectures. Give the factors to test several\n"
        "# levels (e.g. orchestrator: [off, on], temperature: [0, 0.5, 1]), then run this configuration.\n\n"
    )
    target = Path(path)
    target.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def plan(config: ExperimentConfig) -> list[str]:
    """What the study will run, with a cost estimate from its runs so far (if any)."""
    n_trials = n_trials_of(config)
    size = config.space.size()
    per_trial = len(config.tasks) * config.repeats
    lines = [
        f"{config.name}: optimizer {config.optimizer.name} (seed {config.optimizer.seed}), {n_trials} trials x "
        f"{per_trial} runs ({len(config.tasks)} tasks x {config.repeats} repeats) = {n_trials * per_trial} runs; "
        f"space: {size if size is not None else 'infinite (ranges)'} candidates; objective: {config.objective}; "
        f"{config.parallel} run(s) at a time -> {config.output_dir}"
    ]
    results = read_jsonl(config.output_dir / "results.jsonl")
    done = len(read_jsonl(config.output_dir / "trials.jsonl"))
    if results:
        tokens = sum(r["tokens"] for r in results) / len(results)
        seconds = sum(r["seconds"] for r in results) / len(results)
        left = max(0, n_trials - done) * per_trial
        lines.append(
            f"Estimate from {len(results)} runs so far ({tokens:,.0f} tokens, {seconds:.0f} s each): "
            f"{left} runs left ~ {left * tokens / 1e6:.1f} M tokens, {left * seconds / 3600 / config.parallel:.1f} h."
        )
    else:
        lines.append("No runs yet to estimate the cost from; try --trials 1 first.")
    return lines
