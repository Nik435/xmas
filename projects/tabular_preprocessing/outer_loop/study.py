"""The outer loop: an optimizer searches the space of MAS architectures and hyperparameters.

    for each trial:
        candidate = optimizer.ask()                 # an architecture + factor combination
        for each task (x repeat):                   # the candidate as an xmas system,
            run_inner_loop(candidate, task)         #   filling, building and scoring a pipeline
        optimizer.tell(candidate, objective)        # mean (relative) improvement over the identity pipeline

Output in ``<output_dir>``:

- ``trials.jsonl``: one line per trial: the candidate, its objective value with standard error, its
  value per task, its runs;
- ``results.jsonl``: one line per run (candidate x task x repeat): scores, hyperparameters, cost,
  pipeline; ``runs/NNNNN.json``: each run's details (edits, agents' answers and tool calls);
- ``infra_errors.jsonl``: runs that failed for infrastructure reasons (API errors) after their retries.

A run that fails because of the API (timeouts, rate limits, server errors) is retried; if it still
fails, it is not recorded as a result and its trial is not finished, so the next invocation runs it
again. A failure of the system itself (an invalid pipeline, an agent error) is a result: improvement 0.
An authentication error (401/403) stops the study, as retrying cannot help.

An interrupted study continues where it stopped: finished trials are replayed to the optimizer, and
runs already in ``results.jsonl`` are not run again. See :mod:`.report` for the final evaluation on
the test split, the hyperparameter effects and the export to stage 2.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tabular_preprocessing.inner_loop.agents.base import ROLES
from tabular_preprocessing.inner_loop.architecture import Architecture
from tabular_preprocessing.inner_loop.data import RelationalTask, load_relbench_task
from tabular_preprocessing.inner_loop.loop import InnerLoopResult, run_inner_loop
from tabular_preprocessing.outer_loop.config import PREDICTORS, Candidate, ExperimentConfig, Factors, RunSpec
from tabular_preprocessing.outer_loop.optimizers import ListOptimizer, Optimizer, make_optimizer, planned_candidates

LLMFactory = Callable[[str, float | None], Any]
TaskLoader = Callable[[str, str], RelationalTask]

# errors of the API, not of the system: worth retrying, not a result
INFRA = re.compile(
    r"timed? ?out|timeout|rate.?limit|\b429\b|\b50[0234]\b|connection|APIConnection|APIStatus|InternalServer|"
    r"ServiceUnavailable|Bad Gateway|overloaded|LLM batch failed",
    re.IGNORECASE,
)
MAX_OUTAGES = 3  # trials in a row with API errors before the study stops
TOOL_RESULT_CHARS = 500  # of each tool reply kept in a run's detail: enough to see errors and warnings
AUTH = re.compile(r"\b40[13]\b|Authentication|PermissionDenied|invalid api key", re.IGNORECASE)


class StudyAbortedError(RuntimeError):
    """The study cannot go on (e.g. the API refuses the key); retrying would not help."""


class InfraError(RuntimeError):
    """A run failed for infrastructure reasons even after its retries."""


def make_llm(model: str, temperature: float | None) -> Any:
    """The default LLM factory: xmas's APILLM (endpoint and key from XMAS_BASE_URL / XMAS_API_KEY)."""
    from xmas import APILLM

    call_kwargs = {"temperature": temperature} if temperature is not None else None
    return APILLM(model=model, call_kwargs=call_kwargs)


def run_study(
    config: ExperimentConfig,
    llm_factory: LLMFactory = make_llm,
    task_loader: TaskLoader = load_relbench_task,
    max_new_trials: int | None = None,
    log: Callable[[str], None] = print,
    shard: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Run the outer loop until the trials are done, a budget is used up, or ``max_new_trials`` more.

    Args:
        config: The study.
        llm_factory: Builds an LLM from a model name and a temperature.
        task_loader: Loads a task from dataset and task name.
        max_new_trials: Stop after this many new trials (for a first try).
        log: Where progress lines go.
        shard: ``(i, n)``: run only shard ``i`` of ``n`` (1-based), for cluster job arrays. The study's
            candidates are planned deterministically (grid or random) and every ``n``-th one, starting at
            ``i``, is run into ``<output_dir>/shards/<i>-of-<n>/``. Combine the shards with
            :func:`merge_shards`.

    Returns:
        The trials made in this call, as written to ``trials.jsonl``.

    Raises:
        StudyAbortedError: If the API refuses access.
    """
    if shard is not None:
        config, optimizer = shard_of(config, *shard)
    else:
        optimizer = _optimizer(config)
    out = config.output_dir
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config.source, sort_keys=False), encoding="utf-8")
    trials_path, results_path = out / "trials.jsonl", out / "results.jsonl"
    results = {row["run_id"]: row for row in read_jsonl(results_path)}

    done = read_jsonl(trials_path)
    for trial in done:  # resume: the optimizer learns what was already evaluated
        optimizer.tell(candidate_from(trial), trial["value"])
    n_trials = len(optimizer.candidates) if isinstance(optimizer, ListOptimizer) else n_trials_of(config)
    log(
        f"{config.name}: optimizer {config.optimizer.name}, objective {config.objective}, {n_trials} trials x "
        f"{len(config.tasks) * config.repeats} runs each; {len(done)} trials done before."
    )

    tasks: dict[str, RelationalTask] = {}
    llms: dict[tuple[str, float | None], Any] = {}
    made: list[dict[str, Any]] = []
    outages = 0  # trials in a row that hit API errors
    while len(done) + len(made) < n_trials and (max_new_trials is None or len(made) < max_new_trials):
        exceeded = _budget_exceeded(config, results)
        if exceeded:
            log(f"Stopping: {exceeded}.")
            break
        candidate = optimizer.ask()
        if candidate is None:
            log("The optimizer has no candidate left: the space is exhausted.")
            break
        todo = [spec for spec in config.runs_of(candidate) if spec.run_id not in results]
        numbers = {spec.run_id: len(results) + i + 1 for i, spec in enumerate(todo)}
        by_task: dict[str, list[RunSpec]] = {}
        for spec in todo:
            by_task.setdefault(spec.task, []).append(spec)

        infra_failed = False
        for name, specs in by_task.items():
            # One task in memory at a time: big databases (rel-hm, rel-event) would not fit together.
            if name not in tasks:
                tasks.clear()
                tasks[name] = task_loader(*name.split("/")).limit_train(config.max_train_rows)
            task = tasks[name]

            def one(spec: RunSpec, task: RelationalTask = task, numbers: dict[str, int] = numbers) -> dict[str, Any]:
                return _run_with_retries(config, spec, task, llm_factory, llms, numbers[spec.run_id])

            with ThreadPoolExecutor(max_workers=config.parallel) as pool:
                futures = [(spec, pool.submit(one, spec)) for spec in specs]
                for spec, future in futures:
                    try:
                        row = future.result()
                    except InfraError as error:
                        infra_failed = True
                        _append(out / "infra_errors.jsonl", {"run_id": spec.run_id, "error": str(error)})
                        log(f"  {spec.run_id}: infrastructure error after {config.retries} retries: {error}")
                        continue
                    _append(results_path, row)
                    results[spec.run_id] = row
        if infra_failed:
            optimizer.skip(candidate)
            outages += 1
            log(f"[trial skipped] {candidate.key}: some runs hit API errors; they run again next time.")
            if outages >= MAX_OUTAGES:
                log(f"Stopping: {outages} trials in a row hit API errors. Start the study again later.")
                break
            continue
        outages = 0

        rows = [results[spec.run_id] for spec in config.runs_of(candidate)]
        stats = objective_stats(rows, config.objective)
        optimizer.tell(candidate, stats["value"])
        trial = {
            "trial": len(done) + len(made) + 1,
            "candidate": candidate.key,
            "architecture": candidate.architecture.code,
            "factors": candidate.factors.as_dict(),
            "config": candidate.as_dict(),
            "objective": config.objective,
            **stats,
            "failed_runs": sum(r["score"] is None for r in rows),
            "tokens": sum(r["tokens"] for r in rows),
            "runs": [r["run_id"] for r in rows],
        }
        _append(trials_path, trial)
        made.append(trial)
        best = optimizer.best
        se = f" ± {trial['value_se']:.4f}" if trial["value_se"] is not None else ""
        log(
            f"[trial {trial['trial']}/{n_trials}] {candidate.key}: {config.objective} {trial['value']:+.4f}{se}"
            + (f" | best so far {best[1]:+.4f} ({best[0].key})" if best else "")
        )
    return made


def shard_of(config: ExperimentConfig, i: int, n: int) -> tuple[ExperimentConfig, ListOptimizer]:
    """The configuration and the candidate list of shard ``i`` of ``n`` (1-based).

    Raises:
        ValueError: On an invalid shard, or an optimizer that cannot be split.
    """
    if not 1 <= i <= n:
        raise ValueError(f"Shard {i} of {n} does not exist; shards are numbered 1..{n}.")
    settings = config.optimizer
    planned = planned_candidates(settings.name, config.space, n_trials_of(config), settings.seed, **settings.params)
    sharded = replace(config, output_dir=config.output_dir / "shards" / f"{i}-of-{n}")
    return sharded, ListOptimizer(config.space, planned[i - 1 :: n])


def merge_shards(config: ExperimentConfig, log: Callable[[str], None] = print) -> int:
    """Combine every shard of the study into its output directory, so the reports see the whole study.

    Trials and runs are taken from ``<output_dir>/shards/*/`` (and any made unsharded in ``<output_dir>``
    itself), each once; trials are renumbered, and detail paths point into the shard folders. Running
    it again after more shards finished is safe.

    Returns:
        The number of trials in the merged study.
    """
    out = config.output_dir
    shards = sorted(p for p in (out / "shards").glob("*-of-*") if p.is_dir()) if (out / "shards").exists() else []
    results: dict[str, dict[str, Any]] = {r["run_id"]: r for r in read_jsonl(out / "results.jsonl")}
    trials: dict[str, dict[str, Any]] = {t["candidate"]: t for t in read_jsonl(out / "trials.jsonl")}
    infra: list[dict[str, Any]] = []
    for shard in shards:
        prefix = shard.relative_to(out)
        for row in read_jsonl(shard / "results.jsonl"):
            row["detail"] = (
                str(prefix / row["detail"]) if not str(row.get("detail", "")).startswith("shards") else row["detail"]
            )
            results.setdefault(row["run_id"], row)
        for trial in read_jsonl(shard / "trials.jsonl"):
            trials.setdefault(trial["candidate"], trial)
        infra += read_jsonl(shard / "infra_errors.jsonl")
    for number, trial in enumerate(trials.values(), start=1):
        trial["trial"] = number
    for name, rows in (("results.jsonl", results.values()), ("trials.jsonl", trials.values())):
        (out / name).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    if infra:
        (out / "infra_errors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in infra), encoding="utf-8")
    log(f"Merged {len(shards)} shards: {len(trials)} trials, {len(results)} runs -> {out}")
    return len(trials)


def objective_stats(rows: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    """A candidate's value from its runs: per task the mean over repeats, then the mean over tasks.

    A run whose pipeline failed counts as 0 (no improvement). With repeats, the standard error of
    the value combines the per-task standard errors; with one repeat per task it is ``None``.

    Args:
        rows: The candidate's runs (results rows).
        objective: ``"mean_relative_improvement"`` or ``"mean_improvement"``.

    Returns:
        ``value``, ``value_se``, ``per_task`` (task -> mean) and ``mean_score``.
    """
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(run_value(row, objective))
    means = {task: sum(v) / len(v) for task, v in by_task.items()}
    value = sum(means.values()) / len(means)
    if all(len(v) > 1 for v in by_task.values()):
        variance = sum(pd.Series(v).var() / len(v) for v in by_task.values()) / len(by_task) ** 2
        se: float | None = math.sqrt(variance)
    else:
        se = None
    scores = [r["score"] for r in rows if r["score"] is not None]
    return {
        "value": value,
        "value_se": se,
        "per_task": means,
        "mean_score": sum(scores) / len(scores) if scores else None,
    }


def run_value(row: dict[str, Any], objective: str) -> float:
    """One run's contribution to the objective; a failed pipeline counts as 0."""
    if row["improvement"] is None:
        return 0.0
    if objective == "mean_improvement":
        return float(row["improvement"])
    base = abs(row["baseline"]) if row["baseline"] else 0.0
    return float(row["improvement"]) / base if base else float(row["improvement"])


def candidate_from(trial: dict[str, Any]) -> Candidate:
    """The candidate a trial line describes."""
    factors = dict(trial["factors"])
    factors["models"] = tuple(sorted(factors["models"].items()))
    return Candidate(Architecture.from_code(trial["architecture"]), Factors(**factors))


def summarize(config: ExperimentConfig) -> pd.DataFrame:
    """The trials so far, best first: candidate, value with standard error, mean score, failures, cost."""
    trials = read_jsonl(config.output_dir / "trials.jsonl")
    if not trials:
        return pd.DataFrame()
    df = pd.DataFrame(trials)
    columns = ["trial", "candidate", "value", "value_se", "failed_runs", "tokens"]
    return df[[c for c in columns if c in df]].sort_values("value", ascending=False).reset_index(drop=True)


def n_trials_of(config: ExperimentConfig) -> int:
    """How many trials the study runs: ``optimizer.n_trials``, or the whole (finite) space."""
    if config.optimizer.n_trials is not None:
        return config.optimizer.n_trials
    size = config.space.size()
    assert size is not None
    return size


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The lines of a JSON-lines file (none if it does not exist)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _optimizer(config: ExperimentConfig) -> Optimizer:
    settings = config.optimizer
    return make_optimizer(settings.name, config.space, settings.seed, **settings.params)


def _budget_exceeded(config: ExperimentConfig, results: dict[str, dict[str, Any]]) -> str | None:
    tokens = sum(r["tokens"] for r in results.values())
    hours = sum(r["seconds"] for r in results.values()) / 3600
    if config.max_tokens is not None and tokens >= config.max_tokens:
        return f"the token budget is used up ({tokens} of {config.max_tokens})"
    if config.max_hours is not None and hours >= config.max_hours:
        return f"the time budget is used up ({hours:.2f} of {config.max_hours} hours)"
    return None


def _run_with_retries(
    config: ExperimentConfig,
    spec: RunSpec,
    task: RelationalTask,
    llm_factory: LLMFactory,
    cache: dict[tuple[str, float | None], Any],
    number: int,
) -> dict[str, Any]:
    """One run; retried while it fails for infrastructure reasons.

    Raises:
        StudyAbortedError: On an authentication error.
        InfraError: If it still fails for infrastructure reasons after ``config.retries`` retries.
    """
    error = ""
    for attempt in range(config.retries + 1):
        if attempt:
            time.sleep(min(60, 5 * 2**attempt))
        try:
            row = _run(config, spec, task, llm_factory, cache, number)
        except Exception as exc:  # e.g. the LLM cannot even be built
            row = {"error": f"system: {type(exc).__name__}: {exc}", "_raised": exc}
        error = str(row.get("error") or "")
        if AUTH.search(error):
            raise StudyAbortedError(f"The API refused access ({error}); check XMAS_API_KEY and the model name.")
        if not (error.startswith("system:") and INFRA.search(error)):
            if "_raised" in row:
                raise row["_raised"]
            return row
    raise InfraError(error)


def _run(
    config: ExperimentConfig,
    spec: RunSpec,
    task: RelationalTask,
    llm_factory: LLMFactory,
    cache: dict[tuple[str, float | None], Any],
    number: int,
) -> dict[str, Any]:
    """One inner-loop run of a candidate on a task: the result row, with its detail file written."""
    start = time.perf_counter()
    default, per_name, models = _llms(spec, llm_factory, cache)
    budget = config.budgets[spec.factors.autonomy]
    result = run_inner_loop(
        spec.architecture,
        task,
        default,
        llms=per_name,
        models=models,
        model_factory=PREDICTORS[config.predictor],
        max_rows=config.max_rows,
        max_columns=config.max_columns,
        max_loops=budget.max_loops,
        max_tool_rounds=budget.max_tool_rounds,
        max_evaluations=budget.max_evaluations,
        autonomy=spec.factors.autonomy,
        tool_description=spec.factors.tool_description,
        merge_fills_sample=spec.factors.merge_fills_sample,
        max_steps=budget.max_steps,
        routing=spec.factors.orchestrator_routing,
        baseline=config.baseline,
    )
    row = _row(spec, result, time.perf_counter() - start)
    detail = config.output_dir / "runs" / f"{number:05d}.json"
    row["detail"] = str(detail.relative_to(config.output_dir))
    detail.write_text(json.dumps(_detail(row, result), indent=1, default=str), encoding="utf-8")
    return row


def _llms(
    spec: RunSpec, factory: LLMFactory, cache: dict[tuple[str, float | None], Any]
) -> tuple[Any, dict[str, Any], dict[str, str]]:
    """The default LLM, the LLMs by model name, and the model name per agent (model types per role)."""
    temperature = spec.factors.temperature

    def get(model: str) -> Any:
        if not model:
            raise ValueError("factors.models.default names no model; set one in the configuration.")
        if (model, temperature) not in cache:
            cache[(model, temperature)] = factory(model, temperature)
        return cache[(model, temperature)]

    default = get(spec.factors.model("default"))
    models = {f"{g}_{ROLES[r]}": spec.factors.model(r) for g in ("clean", "merge") for r in ("P", "E", "V")}
    models["orchestrator"] = spec.factors.model("C")
    models["solo_executor"] = spec.factors.model("E")
    return default, {name: get(name) for name in set(models.values())}, models


def _row(spec: RunSpec, result: InnerLoopResult, seconds: float) -> dict[str, Any]:
    hp = {f"{g}.{k}": v for g in ("clean", "merge") for k, v in getattr(spec.architecture, g).hp.items()}
    relative = None
    if result.improvement is not None and result.baseline:
        relative = result.improvement / abs(result.baseline)
    return {
        "run_id": spec.run_id,
        "task": spec.task,
        "architecture": spec.architecture.code,
        **hp,
        "factors": spec.factors.key,
        **{f"factor.{k}": json.dumps(v) if isinstance(v, dict) else v for k, v in spec.factors.as_dict().items()},
        "repeat": spec.repeat,
        "score": result.score,
        "baseline": result.baseline,
        "improvement": result.improvement,
        "relative_improvement": relative,
        "metric": result.metric,
        "error": result.error,
        "n_features": len(result.features),
        "n_edits": len(result.edits),
        "agent_calls": len(result.trace.records),
        "evaluations": result.evaluations,
        "tokens": sum(result.tokens.values()),
        "seconds": round(seconds, 2),
        "stopped": (result.orchestration or {}).get("stopped"),
        "orchestrator_steps": (result.orchestration or {}).get("steps"),
        "pipeline": result.pipeline.to_dict(),
    }


def _detail(row: dict[str, Any], result: InnerLoopResult) -> dict[str, Any]:
    return {
        **row,
        "edits": [vars(e) for e in result.edits],
        "trace": [
            {
                "agent": r.agent,
                "output": next(iter(r.outputs.values()), ""),
                "tool_calls": [
                    {
                        "tool": c.tool,
                        "arguments": c.arguments,
                        "failed": c.failed,
                        "result": c.result[:TOOL_RESULT_CHARS],
                    }
                    for c in r.tool_calls
                ],
                "tokens": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
            }
            for r in result.trace.records
        ],
        "system_error": result.trace.error,
        "decisions": (result.orchestration or {}).get("decisions"),
    }
