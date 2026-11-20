"""Agent types: the prompts, inputs and tools each role gets in a team, and a scripted run through xmas."""

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

import pytest
from promptolution.llms.base_llm import BaseLLM

from tabular_preprocessing import RelationalTask
from tabular_preprocessing.inner_loop.agents import (
    AGENT_TYPES,
    CLEANING_CODER,
    CLEANING_VALIDATOR,
    DATA_PROFILER,
    FEATURE_ENGINEER,
    ORCHESTRATOR,
    SCHEMA_JOIN_ANALYST,
    MissingToolSetWarning,
    Team,
)
from tabular_preprocessing.inner_loop.tools import Toolbox, Workspace
from xmas import MAS
from xmas.tools import Tool, ToolCallRequest

FULL = Team(frozenset({"P", "E", "V"}), fb_e=True, fb_v=True)


@pytest.fixture
def toolbox(task: RelationalTask) -> Toolbox:
    return Toolbox(Workspace(task))


def test_the_seven_agent_types_of_the_concept() -> None:
    titles = {(g, r): t.title for g, roles in AGENT_TYPES.items() for r, t in roles.items() if g != "solo"}
    assert titles == {
        ("clean", "P"): "Data Profiler",
        ("clean", "E"): "Cleaning Coder",
        ("clean", "V"): "Cleaning Validator",
        ("merge", "P"): "Schema & Join Analyst",
        ("merge", "E"): "Feature Engineer",
        ("merge", "V"): "Leakage & Feature Validator",
        ("orchestrator", "C"): "Orchestrator",
    }
    assert AGENT_TYPES["solo"]["E"].title == "Generalist"  # the single-agent baseline, outside the concept
    assert [t.name for t in AGENT_TYPES["clean"].values()] == ["clean_planner", "clean_executor", "clean_validator"]
    assert [t.output for t in AGENT_TYPES["merge"].values()] == ["merge_plan", "merge_work", "merge_review"]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"roles": frozenset({"P", "V"})}, "executor"),
        ({"roles": frozenset({"E", "X"})}, "Unknown roles"),
        ({"roles": frozenset({"E"}), "fb_e": True}, "planner"),
        ({"roles": frozenset({"P", "E"}), "fb_v": True}, "validator"),
    ],
)
def test_teams_follow_the_search_space_rules(kwargs: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        Team(**kwargs)


def test_a_lone_executor_takes_over_the_missing_roles(toolbox: Toolbox) -> None:
    team = Team(frozenset({"E"}))
    assert CLEANING_CODER.tool_sets_for(team, AGENT_TYPES["clean"]) == ("filling", "profiling", "validation")
    agent = CLEANING_CODER.build(team, toolbox, AGENT_TYPES["clean"])
    assert agent.inputs == ["query"]
    assert agent.tools == [
        # filling: its own
        "clean_list_functions",
        "clean_show_pipeline",
        "clean_add_step",
        "clean_replace_step",
        "clean_remove_step",
        # profiling: the missing planner's
        "clean_list_tables",
        "clean_profile_table",
        "clean_profile_column",
        "clean_find_placeholders",
        "clean_target_distribution",
        "clean_sample_rows",
        # validation: the missing validator's
        "clean_check_pipeline",
        "clean_score_pipeline",
        "clean_score_steps",
        "clean_feature_report",
    ]
    assert "## Also yours, as there is no planner" in agent.prompt
    assert "## Also yours, as there is no validator" in agent.prompt
    assert "<continue>" not in agent.prompt


def test_a_full_group_with_both_back_edges(toolbox: Toolbox) -> None:
    types = AGENT_TYPES["clean"]
    planner, executor, validator = (t.build(FULL, toolbox, types) for t in types.values())
    assert planner.inputs == ["query", "clean_work"]
    assert executor.inputs == ["query", "clean_plan", "clean_review"]
    assert validator.inputs == ["query", "clean_plan", "clean_work"]
    assert "objection to your plan" in planner.prompt
    assert "<continue> followed by what must change" in executor.prompt
    assert "<continue> followed by concrete fixes" in validator.prompt
    assert "Also yours" not in executor.prompt
    assert executor.tools == [t for t in executor.tools if t.startswith("clean_")]
    assert len(executor.tools) == 5  # only filling: planner and validator bring their own sets
    assert planner.tools[0] == "clean_list_tables"
    assert validator.tools == [
        "clean_check_pipeline",
        "clean_score_pipeline",
        "clean_score_steps",
        "clean_feature_report",
    ]


def test_a_validator_without_back_edge_only_gives_a_verdict() -> None:
    prompt = CLEANING_VALIDATOR.prompt(Team(frozenset({"E", "V"})))
    assert "Answer APPROVED or REJECTED" in prompt
    assert "nobody revises the work after you" in prompt
    orchestrated = CLEANING_VALIDATOR.prompt(Team(frozenset({"E", "V"}), orchestrated=True))
    assert "the orchestrator decides what happens next" in orchestrated


def test_the_second_group_reads_the_first_groups_result() -> None:
    team = Team(frozenset({"P", "E"}), upstream="clean_review")
    assert SCHEMA_JOIN_ANALYST.inputs(team) == ["query", "clean_review"]
    assert FEATURE_ENGINEER.inputs(team) == ["query", "clean_review", "merge_plan"]
    assert "## Result of the previous group\n{clean_review}" in FEATURE_ENGINEER.prompt(team)


def test_every_input_has_a_placeholder_in_the_prompt() -> None:
    for types in AGENT_TYPES.values():
        for agent_type in types.values():
            if agent_type.role == "C":
                continue
            team = Team(frozenset({"P", "E", "V"}), fb_e=True, fb_v=True, upstream="other_review")
            prompt = agent_type.prompt(team)
            assert all(f"{{{var}}}" in prompt for var in agent_type.inputs(team)), agent_type.key


def test_the_orchestrator_reads_every_member(toolbox: Toolbox) -> None:
    team = Team(members=(("Cleaning Coder", "clean_work"), ("Feature Engineer", "merge_work")))
    agent = ORCHESTRATOR.build(team, toolbox, AGENT_TYPES["orchestrator"])
    assert agent.name == "orchestrator"
    assert agent.inputs == ["query", "clean_work", "merge_work"]
    assert '{"next": "submit"' in agent.prompt


def test_profiler_planner_does_not_get_filling_tools(toolbox: Toolbox) -> None:
    agent = DATA_PROFILER.build(FULL, toolbox, AGENT_TYPES["clean"])
    assert "clean_add_step" not in agent.tools
    assert len(agent.tools) == 6


def test_an_unknown_tool_set_warns_and_is_skipped(toolbox: Toolbox) -> None:
    custom = replace(DATA_PROFILER, tool_sets=("profiling", "telepathy"))
    with pytest.warns(MissingToolSetWarning, match="'telepathy'"):
        agent = custom.build(FULL, toolbox, AGENT_TYPES["clean"])
    assert len(agent.tools) == 6


class ScriptedLLM(BaseLLM):
    """Each executor adds one step through its own tools, then reports; every other call just answers."""

    STEPS: ClassVar[dict[str, dict[str, Any]]] = {
        "clean_add_step": {
            "stage": "clean",
            "step": {"fn": "placeholder_as_missing", "table": "results", "values": [-1.0]},
        },
        "merge_add_step": {
            "stage": "merge",
            "step": {"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]},
        },
    }

    def __init__(self) -> None:
        """Start with no prompts seen."""
        super().__init__()
        self.prompts: list[str] = []

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        """Answer plain calls (agents without tools)."""
        return ["ok" for _ in prompts]

    def get_tool_response(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest]]:
        """First turn: call the group's add_step; then report."""
        if len(messages) <= 2:  # first turn: the rendered prompt
            self.prompts.append(messages[-1]["content"])
            name = next(t.name for t in tools if t.name.endswith("_add_step"))
            return "", [ToolCallRequest("call-1", name, json.dumps(self.STEPS[name]))]
        return "Added one step.", []


def test_two_lone_executors_fill_one_pipeline_through_xmas(task: RelationalTask) -> None:
    workspace = Workspace(task)
    toolbox = Toolbox(workspace)
    clean = CLEANING_CODER.build(Team(frozenset({"E"})), toolbox, AGENT_TYPES["clean"])
    merge = FEATURE_ENGINEER.build(Team(frozenset({"E"}), upstream="clean_work"), toolbox, AGENT_TYPES["merge"])
    llm = ScriptedLLM()
    mas = MAS([clean, merge], edges=[("clean_executor", "merge_executor")], llm=llm, tools=toolbox.tools())

    trace = mas.run(task.describe())

    assert trace.error is None, trace.error
    assert [r.agent for r in trace.records] == ["clean_executor", "merge_executor"]
    assert [(c.tool, c.failed) for r in trace.records for c in r.tool_calls] == [
        ("clean_add_step", False),
        ("merge_add_step", False),
    ]
    assert workspace.pipeline.describe().splitlines()[:2] == [
        "clean        placeholder_as_missing[results](values=[-1.0])",
        "merge        aggregate(table='results', fk_column='driverId', aggregations=[{'op': 'count'}])",
    ]
    assert "## Result of the previous group\nAdded one step." in llm.prompts[1]
