"""The search space as xmas systems, and the inner loop: fill, build tables, predict, score."""

import pytest

from conftest import TeamLLM
from tabular_preprocessing import Architecture, GroupArchitecture, RelationalTask, run_inner_loop
from tabular_preprocessing.inner_loop.tools import Toolbox, Workspace
from xmas import END, MAS, START
from xmas.graph import back_edges

CLEAN_STEP = {"stage": "clean", "step": {"fn": "placeholder_as_missing", "table": "results", "values": [-1.0]}}
MERGE_STEPS = [
    {"stage": "merge", "step": {"fn": "entity_columns", "columns": ["nationality"]}},
    {
        "stage": "merge",
        "step": {
            "fn": "aggregate",
            "table": "results",
            "fk_column": "driverId",
            "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
        },
    },
]
STEPS = {"clean": [CLEAN_STEP], "merge": MERGE_STEPS}


def test_nine_variants_per_group_and_81_architectures() -> None:
    variants = GroupArchitecture.variants()
    assert [v.code for v in variants] == ["E", "PE", "PE-EP", "EV", "EV-VE", "PEV", "PEV-VE", "PEV-EP", "PEV-EP-VE"]
    codes = [a.code for a in Architecture.stage1()]
    assert len(set(codes)) == 81
    assert all(Architecture.from_code(code).code == code for code in codes)


@pytest.mark.parametrize("code", ["X", "PE-VE", "E-EP", "PEV-XX", ""])
def test_invalid_group_codes_are_rejected(code: str) -> None:
    with pytest.raises(ValueError):
        GroupArchitecture.from_code(code)


def test_wiring_of_a_full_and_a_two_agent_group(task: RelationalTask) -> None:
    architecture = Architecture.from_code("clean=PEV-EP-VE|merge=PE-EP")
    toolbox = Toolbox(Workspace(task))
    agents, edges = architecture.agents(toolbox)
    assert [a.name for a in agents] == [
        "clean_planner",
        "clean_executor",
        "clean_validator",
        "merge_planner",
        "merge_executor",
    ]
    assert set(edges) == {
        (START, "clean_planner"),
        ("clean_planner", "clean_executor"),
        ("clean_executor", "clean_validator"),
        ("clean_executor", "clean_planner"),
        ("clean_validator", "clean_executor"),
        ("clean_validator", "merge_planner"),
        ("merge_planner", "merge_executor"),
        ("merge_executor", "merge_planner"),
        ("merge_executor", END),
    }
    mas = MAS(agents, edges=edges, llm=TeamLLM(), tools=toolbox.tools())
    # xmas reads the back edges the same way: these are the loops taken only on <continue>
    assert back_edges(mas.graph) == {
        ("clean_executor", "clean_planner"),
        ("clean_validator", "clean_executor"),
        ("merge_executor", "merge_planner"),
    }
    # the merge planner reads Clean's last word, and its executor's objection (E->P)
    assert agents[3].inputs == ["query", "clean_review", "merge_work"]


def test_every_stage1_architecture_builds(task: RelationalTask) -> None:
    for architecture in Architecture.stage1():
        mas = architecture.build(Workspace(task), TeamLLM())
        assert len(mas.agents) == len(architecture.clean.roles) + len(architecture.merge.roles)


def test_merge_first_reverses_who_works_first(task: RelationalTask) -> None:
    architecture = Architecture(order=("merge", "clean"))
    agents, edges = architecture.agents(Toolbox(Workspace(task)))
    assert edges == [(START, "merge_executor"), ("merge_executor", "clean_executor"), ("clean_executor", END)]
    assert agents[1].inputs == ["query", "merge_work"]


def test_an_orchestrated_system_is_not_a_fixed_graph(task: RelationalTask) -> None:
    with pytest.raises(ValueError, match="run_orchestrated"):
        Architecture(orchestrator=True).build(Workspace(task), TeamLLM())
    with pytest.raises(ValueError, match="no back edges"):
        Architecture.from_code("clean=EV-VE|merge=E|orchestrator")


def test_inner_loop_fills_builds_predicts_and_scores(task: RelationalTask) -> None:
    result = run_inner_loop(Architecture(), task, TeamLLM(STEPS))
    assert result.error is None
    assert result.architecture == "clean=E|merge=E"
    assert [r.agent for r in result.trace.records] == ["clean_executor", "merge_executor"]
    assert [e.tool for e in result.edits] == ["clean_add_step", "merge_add_step", "merge_add_step"]
    assert result.features == ["date", "drivers.nationality", "results.count()", "results.mean(points)"]
    assert result.metric == "roc_auc"
    assert result.score is not None
    assert result.baseline is not None
    assert result.improvement is not None
    assert result.improvement > 0.2
    assert set(result.seconds) == {"fill", "tables", "predict", "baseline"}
    assert result.tokens["input"] > 0


def test_the_run_is_scored_without_features_empty_on_the_rows_to_predict(task: RelationalTask) -> None:
    # the validation rows are new drivers: their features from the database are empty
    fresh = RelationalTask(**{**task.__dict__, "val": task.val.assign(driverId=task.val.driverId + 1000)})
    result = run_inner_loop(Architecture(), fresh, TeamLLM(STEPS), baseline=False)
    assert result.error is None
    assert result.dropped == ["drivers.nationality", "results.mean(points)"]
    assert result.features == ["date", "results.count()"]


def test_a_rejection_sends_the_work_back_once(task: RelationalTask) -> None:
    llm = TeamLLM(STEPS, verdicts={"clean": ["<continue> drop the placeholder step"]})
    result = run_inner_loop(Architecture.from_code("clean=EV-VE|merge=E"), task, llm, baseline=False)
    assert result.error is None
    assert [r.agent for r in result.trace.records] == [
        "clean_executor",
        "clean_validator",
        "clean_executor",
        "clean_validator",
        "merge_executor",
    ]
    # the second executor visit sees the review that sent it back
    assert "<continue> drop the placeholder step" in result.trace.records[2].rendered_prompt
    assert result.baseline is None


def test_a_step_that_does_not_fit_the_data_is_refused_and_the_run_still_scores(task: RelationalTask) -> None:
    bad = {"stage": "merge", "step": {"fn": "entity_columns", "columns": ["no_such_column"]}}
    result = run_inner_loop(Architecture(), task, TeamLLM({"merge": [bad]}))
    calls = [c for r in result.trace.records for c in r.tool_calls]
    assert [(c.tool, c.failed) for c in calls] == [("merge_add_step", True)]
    assert "Refused: with this change the pipeline would not run" in calls[0].result
    assert result.pipeline.merge == []  # the draft never held the broken step
    assert result.error is None
    assert result.score is not None


def test_the_single_agent_baseline_fills_every_slot_alone(task: RelationalTask) -> None:
    solo = Architecture.from_code("solo")
    assert solo.code == "solo"
    toolbox = Toolbox(Workspace(task))
    agents, edges = solo.agents(toolbox)
    assert [a.name for a in agents] == ["solo_executor"]
    assert edges == [(START, "solo_executor"), ("solo_executor", END)]
    tools = agents[0].tools
    assert len(tools) == 6 + 6 + 3 + 4  # filling (with list_features), profiling, schema, validation
    assert all(t.startswith("solo_") for t in tools)
    steps = {"solo": [{"stage": s["stage"], "step": s["step"]} for s in [CLEAN_STEP, *MERGE_STEPS]]}
    result = run_inner_loop(solo, task, TeamLLM(steps))
    assert result.error is None
    assert [r.agent for r in result.trace.records] == ["solo_executor"]
    assert result.pipeline.clean and result.pipeline.merge
    assert result.improvement is not None
    assert result.improvement > 0.2
    with pytest.raises(ValueError, match="single-agent"):
        Architecture(solo=True, orchestrator=True)
