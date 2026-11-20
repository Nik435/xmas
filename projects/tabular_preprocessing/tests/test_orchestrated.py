"""Orchestrated systems: the orchestrator decides after every turn who works next."""

import json
from pathlib import Path

import pytest
import yaml

from conftest import TeamLLM
from tabular_preprocessing import Architecture, RelationalTask, run_inner_loop
from tabular_preprocessing.inner_loop.orchestrated import DecisionError, parse_decision, run_orchestrated
from tabular_preprocessing.inner_loop.tools import Workspace
from tabular_preprocessing.outer_loop.config import ExperimentConfig

STEPS = {
    "clean": [{"stage": "clean", "step": {"fn": "placeholder_as_missing", "table": "results", "values": [-1.0]}}],
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
    ],
}
LONE = Architecture(orchestrator=True)


def to(agent: str, instruction: str = "do your part") -> str:
    return f"I will delegate.\n{json.dumps({'next': agent, 'instruction': instruction})}"


def run(task: RelationalTask, decisions: list[str], architecture: Architecture = LONE, **kwargs: object):  # type: ignore[no-untyped-def]
    workspace = Workspace(task)
    result = run_orchestrated(architecture, workspace, TeamLLM(STEPS, decisions=decisions), **kwargs)  # type: ignore[arg-type]
    return workspace, result


def test_parse_decision_takes_the_last_json_object_with_next() -> None:
    text = 'Thinking about {"a": 1}...\n{"next": "clean_executor", "instruction": "use {braces}"}'
    decision = parse_decision(text)
    assert (decision.next, decision.instruction) == ("clean_executor", "use {braces}")
    assert parse_decision('{"next": "submit", "summary": "good"}').summary == "good"
    with pytest.raises(DecisionError, match="No decision"):
        parse_decision("I think we are done.")


def test_the_orchestrator_delegates_turn_by_turn_and_submits(task: RelationalTask) -> None:
    workspace, result = run(task, [to("clean_executor", "replace -1 in results"), to("merge_executor")])
    assert result.stopped == "submitted"
    assert result.steps == 2
    assert [r.agent for r in result.trace.records] == [
        "orchestrator",
        "clean_executor",
        "orchestrator",
        "merge_executor",
        "orchestrator",
    ]
    assert workspace.submitted
    assert workspace.summary == "done"
    assert [c.fn for c in workspace.pipeline.clean] == ["placeholder_as_missing"]
    assert [c.fn for c in workspace.pipeline.merge] == ["aggregate"]
    records = result.trace.records
    # the agent gets the instruction; the orchestrator then sees the agent's full answer and the decisions so far
    assert "## The orchestrator's instruction for this turn\nreplace -1 in results" in records[1].rendered_prompt
    assert "clean visit 1: done." in records[2].rendered_prompt
    assert "1. clean_executor: replace -1 in results" in records[2].rendered_prompt
    assert "Turns used: 1 of 12." in records[2].rendered_prompt


def test_ordered_routing_refuses_the_second_group_first(task: RelationalTask) -> None:
    _, result = run(task, [to("merge_executor"), to("clean_executor"), to("merge_executor")])
    assert result.decisions[0]["error"].startswith("merge_executor belongs to group merge")
    assert "invalid decision: merge_executor belongs to group merge" in result.trace.records[1].rendered_prompt
    assert [r.agent for r in result.trace.records if r.agent != "orchestrator"] == ["clean_executor", "merge_executor"]
    assert result.stopped == "submitted"


def test_free_routing_lets_the_second_group_start(task: RelationalTask) -> None:
    _, result = run(task, [to("merge_executor"), to("clean_executor")], routing="free")
    assert [r.agent for r in result.trace.records if r.agent != "orchestrator"] == ["merge_executor", "clean_executor"]
    assert all(d["error"] is None for d in result.decisions)


def test_the_step_budget_ends_the_run(task: RelationalTask) -> None:
    workspace, result = run(task, [to("clean_executor"), to("merge_executor")], max_steps=1)
    assert result.stopped == "budget"
    assert result.steps == 1
    assert not workspace.submitted
    assert workspace.pipeline.clean  # the draft counts as it is


def test_repeated_invalid_decisions_end_the_run(task: RelationalTask) -> None:
    _, result = run(task, ["no idea", to("teleporter"), "{}"])
    assert result.stopped == "invalid"
    assert len(result.decisions) == 3
    assert all(d["error"] for d in result.decisions)


def test_a_full_team_sees_the_latest_answers_without_back_edges(task: RelationalTask) -> None:
    architecture = Architecture.from_code("clean=PEV|merge=E|orchestrator")
    decisions = [to("clean_planner"), to("clean_executor"), to("clean_validator"), to("clean_executor", "fix it")]
    _, result = run(task, decisions, architecture=architecture)
    executor_turns = [r for r in result.trace.records if r.agent == "clean_executor"]
    assert len(executor_turns) == 2
    assert "## The validator's review, latest version (empty if it has not run yet)\nAPPROVED" in (
        executor_turns[1].rendered_prompt
    )
    assert "<continue>" not in executor_turns[0].rendered_prompt
    assert "Plan for clean" in executor_turns[0].rendered_prompt


def test_inner_loop_runs_an_orchestrated_architecture(task: RelationalTask) -> None:
    llm = TeamLLM(STEPS, decisions=[to("clean_executor"), to("merge_executor")])
    result = run_inner_loop(LONE, task, llm, max_steps=5)
    assert result.error is None
    assert result.architecture == "clean=E|merge=E|orchestrator"
    assert result.orchestration is not None
    assert result.orchestration["stopped"] == "submitted"
    assert result.orchestration["steps"] == 2
    assert result.improvement is not None
    assert result.improvement > 0.2


def test_with_an_orchestrator_back_edges_collapse_to_16_architectures() -> None:
    data = yaml.safe_load((Path(__file__).parent.parent / "configs" / "stage1.yaml").read_text())
    data["factors"]["orchestrator"] = True
    codes = [c.architecture.code for c in ExperimentConfig.from_dict(data).space.grid()]
    assert len(codes) == 16
    assert all(code.endswith("|orchestrator") and "-" not in code for code in codes)


def test_routing_without_an_orchestrator_adds_no_candidates() -> None:
    data = {
        "tasks": ["rel-f1/driver-dnf"],
        "architecture": {"codes": ["clean=E|merge=E"]},
        "factors": {
            "models": {"default": "m"},
            "orchestrator": [False, True],
            "orchestrator_routing": ["ordered", "free"],
        },
    }
    assert len(ExperimentConfig.from_dict(data).space.grid()) == 3  # off, on+ordered, on+free
