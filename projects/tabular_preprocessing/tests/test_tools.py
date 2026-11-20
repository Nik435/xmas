"""The filling tool set, called the way a MAS calls it: through a tool source, with JSON-like arguments."""

import pytest

from tabular_preprocessing import RelationalTask, evaluate
from tabular_preprocessing.inner_loop.tools import Workspace, filling_tools
from xmas.errors import XmasError
from xmas.tools import LocalTools, index_tools

AGGREGATE = {
    "fn": "aggregate",
    "table": "results",
    "fk_column": "driverId",
    "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
}


@pytest.fixture
def workspace(task: RelationalTask) -> Workspace:
    return Workspace(task)


@pytest.fixture
def clean(workspace: Workspace) -> LocalTools:
    return LocalTools(filling_tools(workspace, "clean"))


@pytest.fixture
def merge(workspace: Workspace) -> LocalTools:
    return LocalTools(filling_tools(workspace, "merge"))


def test_tool_names_carry_the_group_and_both_groups_fit_in_one_system(workspace: Workspace) -> None:
    clean, merge = filling_tools(workspace, "clean"), filling_tools(workspace, "merge")
    assert [t.name for t in clean] == [
        "clean_list_functions",
        "clean_show_pipeline",
        "clean_add_step",
        "clean_replace_step",
        "clean_remove_step",
    ]
    assert len(index_tools([*clean, *merge])) == 10


def test_the_schema_offers_only_the_groups_own_slots(clean: LocalTools, merge: LocalTools) -> None:
    def stage_enum(source: LocalTools, name: str) -> list[str]:
        spec = next(t for t in source.tools() if t.name == name)
        stage = spec.parameters["properties"]["stage"]
        return stage["enum"] if "enum" in stage else [stage["const"]]  # one allowed value is written as const

    assert stage_enum(clean, "clean_add_step") == ["clean"]
    assert stage_enum(merge, "merge_add_step") == ["merge", "engineer"]
    with pytest.raises(XmasError):
        merge.call("merge_add_step", {"stage": "clean", "step": {"fn": "drop_duplicate_rows", "table": "results"}})


def test_list_functions_is_the_menu_of_one_slot(merge: LocalTools) -> None:
    menu = merge.call("merge_list_functions", {"stage": "merge"})
    assert "- aggregate(table: str, fk_column: str" in menu
    assert "zero_as_missing" not in menu


def test_groups_fill_one_shared_draft(workspace: Workspace, clean: LocalTools, merge: LocalTools) -> None:
    clean.call(
        "clean_add_step",
        {"stage": "clean", "step": {"fn": "placeholder_as_missing", "table": "results", "values": [-1.0]}},
    )
    merge.call("merge_add_step", {"stage": "merge", "step": {"fn": "entity_columns", "columns": ["nationality"]}})
    out = merge.call("merge_add_step", {"stage": "merge", "step": AGGREGATE, "position": 0})
    assert out.startswith("Added to merge: aggregate(")
    assert "[0] aggregate(" in out
    assert "[1] entity_columns(" in out

    shown = merge.call("merge_show_pipeline", {})
    assert "clean (read-only):\n  [0] placeholder_as_missing[results](values=[-1.0])" in shown
    assert "merge (yours):" in shown
    assert [(e.tool, e.action) for e in workspace.edits] == [
        ("clean_add_step", "add"),
        ("merge_add_step", "add"),
        ("merge_add_step", "add"),
    ]
    # the draft is a real pipeline: it runs and beats the identity baseline
    assert evaluate(workspace.pipeline, workspace.task).score > 0.7


def test_replace_and_remove_by_number(workspace: Workspace, merge: LocalTools) -> None:
    merge.call("merge_add_step", {"stage": "merge", "step": AGGREGATE})
    merge.call("merge_add_step", {"stage": "engineer", "step": {"fn": "log_transform", "columns": ["results.count()"]}})
    out = merge.call(
        "merge_replace_step",
        {"stage": "engineer", "index": 0, "step": {"fn": "drop_features", "columns": ["results.count()"]}},
    )
    assert "Replaced engineer[0] with drop_features(" in out
    merge.call("merge_remove_step", {"stage": "engineer", "index": 0})
    assert workspace.pipeline.engineer == []
    assert [e.action for e in workspace.edits] == ["add", "add", "replace", "remove"]


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"stage": "merge", "step": {"fn": "teleport"}}, "merge has no function 'teleport'"),
        ({"stage": "merge", "step": {"fn": "lookup", "fk_column": "teamId"}}, "needs parameters"),
        ({"stage": "merge", "step": {"columns": ["x"]}}, "'fn' key"),
        ({"stage": "merge", "step": AGGREGATE, "position": 5}, "position must be between 0 and 0"),
    ],
)
def test_a_wrong_step_comes_back_as_an_error_and_changes_nothing(
    workspace: Workspace, merge: LocalTools, arguments: dict[str, object], reason: str
) -> None:
    with pytest.raises(XmasError, match=reason):
        merge.call("merge_add_step", arguments)
    assert workspace.pipeline.to_dict() == workspace.pipeline.identity().to_dict()
    assert workspace.edits == []


def test_a_wrong_index_is_an_error(merge: LocalTools) -> None:
    with pytest.raises(XmasError, match="merge has no step 0"):
        merge.call("merge_remove_step", {"stage": "merge", "index": 0})


def test_clean_steps_need_a_table(clean: LocalTools) -> None:
    with pytest.raises(XmasError, match="needs a 'table'"):
        clean.call("clean_add_step", {"stage": "clean", "step": {"fn": "drop_duplicate_rows"}})


def test_sample_holds_one_step(workspace: Workspace) -> None:
    tools = LocalTools(filling_tools(workspace, "merge", slots=["merge", "engineer", "sample"]))
    tools.call("merge_add_step", {"stage": "sample", "step": {"fn": "random_views", "n_views": 2}})
    tools.call("merge_add_step", {"stage": "sample", "step": {"fn": "recent"}})
    assert str(workspace.pipeline.sample) == "recent()"
    tools.call("merge_remove_step", {"stage": "sample", "index": 0})
    assert str(workspace.pipeline.sample) == "full()"


def test_unknown_groups_and_slots_are_rejected(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="No slots"):
        filling_tools(workspace, "orchestrator")
    with pytest.raises(ValueError, match="Unknown slots"):
        filling_tools(workspace, "clean", slots=["preprocess"])


# the three ways pipelines broke in the first real test study, now refused when the step is added


def test_an_unknown_aggregation_is_refused(workspace: Workspace, merge: LocalTools) -> None:
    last = {**AGGREGATE, "aggregations": [{"op": "last", "column": "points"}]}
    with pytest.raises(XmasError, match=r"Refused.*Unknown aggregation op 'last'"):
        merge.call("merge_add_step", {"stage": "merge", "step": last})
    assert workspace.pipeline.merge == []
    assert workspace.edits == []


def test_a_misspelled_merged_column_is_refused(workspace: Workspace, merge: LocalTools) -> None:
    merge.call("merge_add_step", {"stage": "merge", "step": AGGREGATE})
    typo = {"fn": "combine", "a": "results.count", "b": "results.mean(points)", "op": "div"}  # no "()"
    with pytest.raises(XmasError, match="Refused"):
        merge.call("merge_add_step", {"stage": "engineer", "step": typo})
    assert workspace.pipeline.engineer == []


def test_a_column_removed_by_clean_cannot_be_merged_and_vice_versa(
    workspace: Workspace, clean: LocalTools, merge: LocalTools
) -> None:
    drop = {"fn": "drop_columns", "table": "drivers", "columns": ["nationality"]}
    clean.call("clean_add_step", {"stage": "clean", "step": drop})
    with pytest.raises(XmasError, match=r"Refused.*nationality"):
        merge.call("merge_add_step", {"stage": "merge", "step": {"fn": "entity_columns", "columns": ["nationality"]}})
    # the other way round: Clean may not remove a column a merge step already uses
    clean.call("clean_remove_step", {"stage": "clean", "index": 0})
    merge.call("merge_add_step", {"stage": "merge", "step": {"fn": "entity_columns", "columns": ["nationality"]}})
    with pytest.raises(XmasError, match="Refused"):
        clean.call("clean_add_step", {"stage": "clean", "step": drop})


def test_removing_a_step_a_later_step_needs_is_refused(workspace: Workspace, merge: LocalTools) -> None:
    merge.call("merge_add_step", {"stage": "merge", "step": AGGREGATE})
    merge.call("merge_add_step", {"stage": "engineer", "step": {"fn": "log_transform", "columns": ["results.count()"]}})
    with pytest.raises(XmasError, match="Refused"):
        merge.call("merge_remove_step", {"stage": "merge", "index": 0})
    assert len(workspace.pipeline.merge) == 1
