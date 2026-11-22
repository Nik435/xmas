"""The outer loop: the configuration and search space, the optimizers, and the study loop."""

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from conftest import TeamLLM
from tabular_preprocessing import Architecture, Evaluation, Pipeline, RelationalTask, evaluate
from tabular_preprocessing.inner_loop.agents import AGENT_TYPES, CLEANING_CODER, Team
from tabular_preprocessing.inner_loop.tools import Toolbox, Workspace, filling_tools
from tabular_preprocessing.outer_loop import (
    ConfigError,
    ExperimentConfig,
    GridSearch,
    RandomSearch,
    effects,
    export_stage2,
    final_evaluation,
    make_optimizer,
    objective_stats,
    run_study,
    summarize,
)
from tabular_preprocessing.outer_loop.cli import main
from tabular_preprocessing.outer_loop.report import baselines, compare, entity_table_pipeline
from tabular_preprocessing.outer_loop.study import StudyAbortedError, merge_shards, shard_of
from xmas.errors import XmasError
from xmas.tools import LocalTools

CONFIGS = Path(__file__).parent.parent / "configs"
STEPS = {
    "merge": [
        {
            "stage": "merge",
            "step": {
                "fn": "aggregate",
                "table": "results",
                "fk_column": "driverId",
                "aggregations": [{"op": "mean", "column": "points"}],
            },
        }
    ]
}


def config(**overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "test",
        "tasks": ["synthetic/driver-good"],
        "architecture": {"codes": ["clean=E|merge=E", "clean=EV-VE|merge=PE"]},
        "factors": {"models": {"default": "m"}},
    }
    data.update(overrides)
    return ExperimentConfig.from_dict(data)


def fake(made: list[tuple[str, float | None]] | None = None):  # type: ignore[no-untyped-def]
    def factory(model: str, temperature: float | None) -> TeamLLM:
        if made is not None:
            made.append((model, temperature))
        return TeamLLM(STEPS)

    return factory


def study(cfg: ExperimentConfig, task: RelationalTask, **kwargs: Any) -> list[dict[str, Any]]:
    return run_study(
        cfg, llm_factory=kwargs.pop("factory", fake()), task_loader=lambda d, t: task, log=lambda _: None, **kwargs
    )


# ------------------------------------------------------------------------ the search space


def test_the_stage1_configuration_is_a_grid_of_the_81_systems() -> None:
    cfg = ExperimentConfig.from_yaml(CONFIGS / "stage1.yaml")
    assert cfg.optimizer.name == "grid"
    assert cfg.space.size() == 81
    assert len(cfg.space.factor_grid()) == 1
    factors = cfg.space.factor_grid()[0]
    assert (factors.orchestrator, factors.temperature, factors.tool_description) == (False, 0.25, "detailed")
    assert (factors.autonomy, factors.order, factors.merge_fills_sample) == ("bounded", "clean_first", False)
    assert len(cfg.runs()) == len(cfg.tasks) * 81


def test_the_random_configuration_samples_a_space_with_a_range() -> None:
    cfg = ExperimentConfig.from_yaml(CONFIGS / "random.yaml")
    assert (cfg.optimizer.name, cfg.optimizer.n_trials) == ("random", 30)
    assert cfg.space.size() is None
    temperatures = {cfg.space.sample(random.Random(i)).factors.temperature for i in range(20)}
    assert len(temperatures) > 10
    assert all(0 <= t <= 1 for t in temperatures)  # type: ignore[operator]


def test_a_list_gives_choices() -> None:
    cfg = config(factors={"models": {"default": "m"}, "temperature": [0, 0.5, 1], "autonomy": ["bounded", "free"]})
    assert len(cfg.space.factor_grid()) == 6
    assert cfg.space.size() == 2 * 6
    assert len({c.key for c in cfg.space.grid()}) == 12


def test_per_group_hyperparameters_collapse_inactive_values() -> None:
    cfg = config(architecture={"clean": {"size": 2, "role": ["planner", "validator"], "fb_E": [0, 1]}})
    assert [c.architecture.code for c in cfg.space.grid()] == [
        "clean=PE|merge=E",
        "clean=PE-EP|merge=E",
        "clean=EV|merge=E",  # fb_E is inactive without a planner, so it adds nothing here
    ]


def test_the_order_factor_reorders_the_groups() -> None:
    cfg = config(factors={"models": {"default": "m"}, "order": "merge_first"})
    assert cfg.space.grid()[0].architecture.code == "merge=E|clean=E"


def test_random_samples_respect_active_when() -> None:
    cfg = ExperimentConfig.from_yaml(CONFIGS / "random.yaml")
    rng = random.Random(1)
    for _ in range(200):
        candidate = cfg.space.sample(rng)
        hp = candidate.as_dict()
        for group in ("clean", "merge"):
            if hp[f"{group}.size"] != 2:
                assert hp[f"{group}.role"] is None
            if candidate.factors.orchestrator:
                assert "-" not in candidate.architecture.code  # no back edges with an orchestrator
        if not candidate.factors.orchestrator:
            assert candidate.factors.orchestrator_routing == "ordered"  # inactive: at its default


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"factors": {"models": {"default": "m"}, "orchestrator_routing": "chaotic"}}, "orchestrator_routing"),
        ({"factors": {"models": {"default": "m"}, "order": "pretest"}}, "pre-test"),
        ({"factors": {"models": {"default": "m"}, "tool_description": "verbose"}}, "tool_description"),
        ({"factors": {"models": {"P": "m"}}}, "default"),
        ({"factors": {"temprature": 0.5}}, "Unknown keys"),
        ({"factors": {"models": {"default": "m"}, "autonomy": {"low": 0, "high": 1}}}, "only temperature"),
        ({"factors": {"models": {"default": "m"}, "temperature": {"low": 0, "high": 1}}}, "ranges"),  # grid
        (
            {
                "factors": {"models": {"default": "m"}, "temperature": {"low": 0, "high": 1}},
                "optimizer": {"name": "random"},
            },
            "n_trials",
        ),
        ({"optimizer": {"name": "smac"}}, "optimizer 'smac' is unknown"),
        ({"run": {"baseline": False}}, "needs run.baseline"),
        ({"objective": "max_score"}, "objective"),
        ({"tasks": ["rel-f1"]}, "dataset"),
        ({"architecture": {"clean": {"size": 4}}}, "size must be"),
        ({"architecture": {"clean": {"size": 2}}}, "role"),
        ({"budgets": {"lavish": {"max_loops": 9}}}, "autonomy level"),
        ({"run": {"predictor": "tabfm"}}, "predictor"),
    ],
)
def test_a_configuration_that_cannot_run_says_why(overrides: dict[str, Any], reason: str) -> None:
    with pytest.raises(ConfigError, match=reason):
        config(**overrides)


# ------------------------------------------------------------------------- the optimizers


def test_grid_proposes_every_candidate_once_in_order() -> None:
    cfg = config()
    grid = make_optimizer("grid", cfg.space)
    assert isinstance(grid, GridSearch)
    first = grid.ask()
    assert first is not None
    grid.tell(first, 0.1)
    second = grid.ask()
    assert second is not None
    grid.tell(second, 0.3)
    assert [first.architecture.code, second.architecture.code] == ["clean=E|merge=E", "clean=EV-VE|merge=PE"]
    assert grid.ask() is None
    assert grid.best == (second, 0.3)


def test_random_never_proposes_a_candidate_twice_and_is_seeded() -> None:
    cfg = ExperimentConfig.from_yaml(CONFIGS / "stage1.yaml")
    a, b = RandomSearch(cfg.space, seed=7), RandomSearch(cfg.space, seed=7)
    keys = []
    for _ in range(81):
        candidate = a.ask()
        assert candidate is not None
        keys.append(candidate.key)
        assert candidate.key == b.ask().key  # type: ignore[union-attr]
    assert len(set(keys)) == 81  # the whole space, no repeats
    assert a.ask() is None  # exhausted


def row(task: str, improvement: float | None, baseline: float, score: float | None = 1.0) -> dict[str, Any]:
    return {"task": task, "improvement": improvement, "baseline": baseline, "score": score}


def test_the_relative_objective_weighs_tasks_alike_and_counts_failures_as_zero() -> None:
    rows = [row("auc", 0.1, 0.5), row("mae", 0.4, -4.0), row("mae", None, -4.0, None)]
    stats = objective_stats(rows, "mean_relative_improvement")
    # auc: 0.1 / 0.5 = 20%; mae: mean of 0.4 / 4 = 10% and a failure (0) = 5%; mean over tasks 12.5%
    assert stats["per_task"] == {"auc": pytest.approx(0.2), "mae": pytest.approx(0.05)}
    assert stats["value"] == pytest.approx(0.125)
    absolute = objective_stats(rows, "mean_improvement")
    assert absolute["value"] == pytest.approx((0.1 + 0.2) / 2)


def test_repeats_give_a_standard_error() -> None:
    assert objective_stats([row("a", 0.1, 0.5)], "mean_relative_improvement")["value_se"] is None
    stats = objective_stats([row("a", 0.1, 0.5), row("a", 0.2, 0.5)], "mean_relative_improvement")
    assert stats["value"] == pytest.approx(0.3)
    assert stats["value_se"] == pytest.approx(0.1)  # std of (0.2, 0.4) / sqrt(2)


# ----------------------------------------------------------------- factors in the system


def test_tool_description_levels_change_the_menu(task: RelationalTask) -> None:
    def menu(level: str) -> str:
        tools = LocalTools(filling_tools(Workspace(task, tool_description=level), "merge"))
        return tools.call("merge_list_functions", {"stage": "merge"})

    assert " -- " not in menu("brief")
    assert " -- " in menu("detailed")
    assert "example: {" not in menu("detailed")
    assert 'example: {"fn": "aggregate"' in menu("examples")


def test_merge_fills_sample_gives_merge_a_restricted_sample_slot(task: RelationalTask) -> None:
    workspace = Workspace(task, merge_fills_sample=True)
    tools = LocalTools(filling_tools(workspace, "merge"))
    assert "oversample_minority" not in tools.call("merge_list_functions", {"stage": "sample"})
    tools.call("merge_add_step", {"stage": "sample", "step": {"fn": "stratified_views", "n_views": 2}})
    assert str(workspace.pipeline.sample) == "stratified_views(n_views=2)"
    with pytest.raises(XmasError, match="no oversampling"):
        tools.call("merge_add_step", {"stage": "sample", "step": {"fn": "oversample_minority"}})
    without = LocalTools(filling_tools(Workspace(task), "merge"))
    with pytest.raises(XmasError):
        without.call("merge_add_step", {"stage": "sample", "step": {"fn": "full"}})


def test_autonomy_is_told_to_every_agent(task: RelationalTask) -> None:
    toolbox = Toolbox(Workspace(task))
    fixed = CLEANING_CODER.build(Team(autonomy="fixed"), toolbox, AGENT_TYPES["clean"])
    assert "Work strictly through the steps under 'How to work'" in fixed.prompt
    agents, _ = Architecture().agents(toolbox, autonomy="free")
    assert all("large budget" in a.prompt for a in agents)
    with pytest.raises(ValueError, match="autonomy"):
        Team(autonomy="anarchic")


# ------------------------------------------------------------------------------ the study


def test_a_grid_study_evaluates_every_candidate_and_resumes(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "study"))
    made: list[tuple[str, float | None]] = []
    trials = study(cfg, task, factory=fake(made))
    assert [t["architecture"] for t in trials] == ["clean=E|merge=E", "clean=EV-VE|merge=PE"]
    assert all(t["value"] > 0.2 and t["failed_runs"] == 0 for t in trials)
    assert trials[1]["config"]["clean.role"] == "validator"
    assert trials[1]["config"]["clean.fb_E"] is None  # inactive
    assert made == [("m", 0.25)]  # one model, one temperature: built once, reused

    out = tmp_path / "study"
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [r["run_id"] for r in rows] == [run for t in trials for run in t["runs"]]
    assert Pipeline.from_dict(rows[0]["pipeline"]).merge[0].fn == "aggregate"
    detail = json.loads((out / rows[1]["detail"]).read_text())
    assert [step["agent"] for step in detail["trace"]][:2] == ["clean_executor", "clean_validator"]
    calls = [c for step in detail["trace"] for c in step["tool_calls"]]
    assert calls and all(0 < len(c["result"]) <= 500 for c in calls)  # the tools' replies are kept, shortened
    assert rows[0]["dropped_features"] == []  # features left out for being empty on the rows to predict
    assert yaml.safe_load((out / "config.yaml").read_text())["name"] == "test"

    assert study(cfg, task) == []  # every trial done
    summary = summarize(cfg)
    assert list(summary["value"]) == sorted(summary["value"], reverse=True)


def test_runs_are_reused_when_trials_are_lost(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "s"))
    study(cfg, task)
    (tmp_path / "s" / "trials.jsonl").unlink()
    made: list[tuple[str, float | None]] = []
    trials = study(cfg, task, factory=fake(made))
    assert len(trials) == 2
    assert made == []  # every run came from results.jsonl; no LLM was built
    assert len((tmp_path / "s" / "results.jsonl").read_text().splitlines()) == 2


def test_a_random_study_samples_new_candidates(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(
        output_dir=str(tmp_path / "r"),
        factors={"models": {"default": "m"}, "temperature": {"low": 0.0, "high": 1.0}},
        optimizer={"name": "random", "n_trials": 3, "seed": 1},
    )
    made: list[tuple[str, float | None]] = []
    trials = study(cfg, task, factory=fake(made))
    assert len(trials) == 3
    assert len({t["candidate"] for t in trials}) == 3
    assert len({temperature for _, temperature in made}) == 3  # each trial its own temperature


def test_max_new_trials_stops_early_and_a_rerun_continues(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "t"))
    assert len(study(cfg, task, max_new_trials=1)) == 1
    rest = study(cfg, task)
    assert [t["trial"] for t in rest] == [2]


def test_dry_run_shows_the_plan(capsys: pytest.CaptureFixture[str]) -> None:
    main([str(CONFIGS / "stage1.yaml"), "--dry-run"])
    out = capsys.readouterr().out
    assert out.startswith("stage1: optimizer grid (seed 0), 81 trials x 3 runs (3 tasks x 1 repeats) = 243 runs")
    assert "objective: mean_relative_improvement" in out
    assert "clean=E|merge=E|" in out


# ------------------------------------------------------------- robustness, budgets, reports


class FlakyLLM(TeamLLM):
    """Fails like an overloaded API for the first ``failures`` tool turns."""

    def __init__(self, failures: int, message: str = "LLM batch failed: 503 Service Unavailable") -> None:
        """Script the failures."""
        super().__init__(STEPS)
        self.failures, self.message = failures, message

    def get_tool_response(self, messages: list[dict[str, Any]], tools: Any) -> Any:
        """Fail while failures are left, then answer as the scripted team."""
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError(self.message)
        return super().get_tool_response(messages, tools)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tabular_preprocessing.outer_loop.study.time.sleep", lambda _: None)


def test_an_api_error_is_retried_not_scored(task: RelationalTask, tmp_path: Path, no_sleep: None) -> None:
    cfg = config(output_dir=str(tmp_path / "f"), architecture={"codes": ["clean=E|merge=E"]})
    flaky = FlakyLLM(failures=1)
    trials = study(cfg, task, factory=lambda m, t: flaky)
    assert len(trials) == 1
    assert trials[0]["failed_runs"] == 0  # the retry succeeded
    assert trials[0]["value"] > 0


def test_a_run_that_keeps_failing_on_the_api_leaves_its_trial_open(
    task: RelationalTask, tmp_path: Path, no_sleep: None
) -> None:
    cfg = config(output_dir=str(tmp_path / "g"), run={"retries": 1})
    logged: list[str] = []
    down = FlakyLLM(failures=10_000)
    trials = run_study(cfg, llm_factory=lambda m, t: down, task_loader=lambda d, t: task, log=logged.append)
    assert trials == []
    assert not (tmp_path / "g" / "results.jsonl").exists()  # not a result: no zero for the candidates
    assert len((tmp_path / "g" / "infra_errors.jsonl").read_text().splitlines()) == 2  # each candidate once
    assert sum("[trial skipped]" in line for line in logged) == 2  # a skipped candidate is not asked again
    assert len(study(cfg, task)) == 2  # the next invocation runs them again


def test_a_lasting_outage_stops_the_study(task: RelationalTask, tmp_path: Path, no_sleep: None) -> None:
    codes = ["clean=E|merge=E", "clean=EV|merge=E", "clean=PE|merge=E", "clean=E|merge=PE"]
    cfg = config(output_dir=str(tmp_path / "o"), architecture={"codes": codes}, run={"retries": 0})
    logged: list[str] = []
    down = FlakyLLM(failures=10_000)
    run_study(cfg, llm_factory=lambda m, t: down, task_loader=lambda d, t: task, log=logged.append)
    assert sum("[trial skipped]" in line for line in logged) == 3
    assert "Stopping: 3 trials in a row hit API errors" in logged[-1]


def test_an_authentication_error_stops_the_study(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "a"))
    with pytest.raises(StudyAbortedError, match="refused access"):
        study(cfg, task, factory=lambda m, t: FlakyLLM(failures=99, message="Error code: 401 - invalid api key"))


def test_a_token_budget_stops_new_trials(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "b"), run={"max_tokens": 1})
    assert len(study(cfg, task)) == 1  # the first trial uses the budget up; the second is not started


def test_parallel_runs_and_train_row_limit(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(
        output_dir=str(tmp_path / "p"),
        architecture={"codes": ["clean=E|merge=E"]},
        repeats=2,
        run={"parallel": 2, "max_train_rows": 300},
    )
    trials = study(cfg, task)
    assert len(trials) == 1
    assert len(trials[0]["runs"]) == 2
    assert trials[0]["value_se"] is not None  # two repeats: a standard error
    rows = [json.loads(line) for line in (tmp_path / "p" / "results.jsonl").read_text().splitlines()]
    assert {r["repeat"] for r in rows} == {1, 2}


def test_final_evaluation_reapplies_the_best_pipelines_to_the_test_split(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "e"))
    study(cfg, task)
    # a stand-in test split: the synthetic task's validation rows play the held-out test rows
    final = final_evaluation(cfg, top=1, task_loader=lambda d, t: task, log=lambda _: None)
    assert len(final) == 1
    assert final["test_value"].iloc[0] > 0
    assert "test:synthetic/driver-good" in final.columns
    lines = (tmp_path / "e" / "final.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["score"] is not None


def fake_dfs(task: RelationalTask, split: str, factory: Any, max_rows: int) -> Evaluation:
    # a stand-in for RDBLearn's features (they need RelBench): the rows only, as dfs gets them
    assert task.db == {}
    return evaluate(Pipeline.identity(), task, factory, max_rows)


def test_baselines_score_fixed_pipelines_once_per_task(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "b"))
    loads: list[str] = []

    def loader(d: str, t: str) -> RelationalTask:
        loads.append(t)
        return task

    table = baselines(cfg, task_loader=loader, test_loader=loader, log=lambda _: None, dfs=fake_dfs)
    assert len(table) == 6  # val and test x identity, entity_table and dfs
    val = table[table["split"] == "val"].set_index("baseline")
    assert val.loc["entity_table", "n_features"] == 2  # drivers.nationality and the time column; keys stay hidden
    assert val.loc["identity", "relative_improvement"] == 0
    assert val.loc["dfs", "score"] == val.loc["identity", "score"]
    assert entity_table_pipeline(task).merge[0].params["columns"] == ["nationality"]
    baselines(cfg, task_loader=loader, test_loader=loader, log=lambda _: None, dfs=fake_dfs)  # nothing again
    assert loads == ["driver-good", "driver-good"]


def test_a_failing_dfs_keeps_the_other_baselines(task: RelationalTask, tmp_path: Path) -> None:
    def broken(*args: Any) -> Evaluation:
        raise RuntimeError("no relbench here")

    cfg = config(output_dir=str(tmp_path / "f"))
    table = baselines(cfg, task_loader=lambda d, t: task, test_loader=lambda d, t: task, log=lambda _: None, dfs=broken)
    scores = table[table["split"] == "val"].set_index("baseline")["score"]
    assert pd.isna(scores["dfs"]) and not pd.isna(scores["identity"])


def test_compare_puts_systems_and_every_baseline_side_by_side(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "c"), architecture={"codes": ["clean=E|merge=E", "solo"]})
    study(cfg, task)
    baselines(cfg, task_loader=lambda d, t: task, test_loader=lambda d, t: task, log=lambda _: None, dfs=fake_dfs)
    final_evaluation(cfg, top=2, task_loader=lambda d, t: task, log=lambda _: None)
    row = compare(cfg).iloc[0]
    assert row["metric"] == "roc_auc"
    assert {"identity", "entity_table", "dfs", "mas_mean", "mas_selected", "solo"} <= set(row.index)
    assert row["mas_mean"] == row["mas_selected"]  # one architecture besides solo


def test_effects_leave_out_the_single_agent_baseline(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(
        output_dir=str(tmp_path / "solo"), architecture={"codes": ["clean=E|merge=E", "clean=PEV|merge=EV-VE", "solo"]}
    )
    study(cfg, task)
    assert len(summarize(cfg)) == 3
    table = effects(cfg)
    assert table.set_index(["hyperparameter", "level"])["trials"].groupby(level=0).sum().eq(2).all()


def test_effects_and_stage2_export(task: RelationalTask, tmp_path: Path) -> None:
    cfg = config(output_dir=str(tmp_path / "x"))
    study(cfg, task)
    table = effects(cfg)
    assert set(table["hyperparameter"]) >= {"clean.size", "merge.size"}
    assert table.loc[table["hyperparameter"] == "clean.size", "trials"].sum() == 2
    path = export_stage2(cfg, 1, tmp_path / "stage2.yaml")
    stage2 = ExperimentConfig.from_yaml(path)
    assert len(stage2.space.architectures()) == 1
    assert stage2.output_dir == Path(f"{cfg.output_dir}_stage2")


# ------------------------------------------------------------------------------- sharding


def test_shards_split_the_study_and_merge_back(task: RelationalTask, tmp_path: Path) -> None:
    codes = ["clean=E|merge=E", "clean=EV|merge=E", "clean=PE|merge=E"]
    cfg = config(output_dir=str(tmp_path / "s"), architecture={"codes": codes})
    first = run_study(cfg, fake(), lambda d, t: task, log=lambda _: None, shard=(1, 2))
    second = run_study(cfg, fake(), lambda d, t: task, log=lambda _: None, shard=(2, 2))
    assert [t["architecture"] for t in first] == ["clean=E|merge=E", "clean=PE|merge=E"]  # every 2nd from 1
    assert [t["architecture"] for t in second] == ["clean=EV|merge=E"]
    assert (tmp_path / "s" / "shards" / "1-of-2" / "trials.jsonl").exists()
    assert not (tmp_path / "s" / "trials.jsonl").exists()  # the shards never write to the study folder

    assert merge_shards(cfg, log=lambda _: None) == 3
    assert merge_shards(cfg, log=lambda _: None) == 3  # merging again changes nothing
    assert len(summarize(cfg)) == 3
    rows = [json.loads(line) for line in (tmp_path / "s" / "results.jsonl").read_text().splitlines()]
    assert all((tmp_path / "s" / r["detail"]).exists() for r in rows)  # detail paths point into the shards
    final = final_evaluation(cfg, top=1, task_loader=lambda d, t: task, log=lambda _: None)
    assert len(final) == 1


def test_a_random_study_shards_the_same_candidates_in_every_job() -> None:
    cfg = ExperimentConfig.from_yaml(CONFIGS / "random.yaml")
    jobs = [shard_of(cfg, i, 3)[1].candidates for i in (1, 2, 3)]
    keys = [c.key for job in jobs for c in job]
    assert len(keys) == cfg.optimizer.n_trials == len(set(keys))  # all trials, each once
    assert [c.key for c in shard_of(cfg, 2, 3)[1].candidates] == [c.key for c in jobs[1]]  # deterministic
    with pytest.raises(ValueError, match=r"numbered 1\.\.3"):
        shard_of(cfg, 4, 3)


def test_tabpfn_v2_is_a_predictor_choice_and_weights_can_be_fetched() -> None:
    from tabular_preprocessing.outer_loop.config import PREDICTORS, fetch_predictor

    assert config(run={"predictor": "tabpfn_v2"}).predictor == "tabpfn_v2"
    with pytest.raises(ConfigError, match="unknown"):
        config(run={"predictor": "tabpfn_v9"})
    fetch_predictor(PREDICTORS["hist_gradient_boosting"])  # the same call fits TabPFN once on the login node
