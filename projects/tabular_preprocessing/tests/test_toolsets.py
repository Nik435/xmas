"""The profiling, schema, validation and orchestration tool sets, called the way a MAS calls them."""

import pandas as pd
import pytest

from tabular_preprocessing import Architecture, Pipeline, RelationalTask
from tabular_preprocessing.inner_loop.tools import TOOL_SETS, Toolbox, Workspace
from tabular_preprocessing.inner_loop.tools.validation import _signal
from xmas.errors import XmasError
from xmas.tools import LocalTools, index_tools

AGGREGATE = {
    "fn": "aggregate",
    "table": "results",
    "fk_column": "driverId",
    "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
}
PLACEHOLDER = {"fn": "placeholder_as_missing", "table": "results", "values": [-1.0]}


@pytest.fixture
def workspace(task: RelationalTask) -> Workspace:
    return Workspace(task)


def tools(workspace: Workspace, group: str, name: str) -> LocalTools:
    return LocalTools(TOOL_SETS[name](workspace, group))


# ------------------------------------------------------------------------- profiling


def test_profiling_finds_the_placeholder_and_sees_it_gone_after_cleaning(workspace: Workspace) -> None:
    p = tools(workspace, "clean", "profiling")
    assert "results.points: -1" in p.call("clean_find_placeholders", {})
    workspace.pipeline = Pipeline.from_dict({"clean": [PLACEHOLDER]})
    assert "results.points" not in p.call("clean_find_placeholders", {"after_cleaning": True})


def test_profiling_tables_columns_target_and_rows(workspace: Workspace) -> None:
    p = tools(workspace, "clean", "profiling")
    overview = p.call("clean_list_tables", {})
    assert overview.startswith("Entity table: drivers.")
    assert "- results: 1800 rows" in overview

    profile = p.call("clean_profile_table", {"table_name": "results"})
    assert "Protected (keys/time): resultId, driverId, raceId, date." in profile
    assert "- points: float64" in profile
    assert "placeholder candidates: -1" in profile

    column = p.call("clean_profile_column", {"table_name": "results", "column": "points"})
    assert "Most frequent values:\n- -1: 164" in column
    assert "Quantiles: 0%=" in column

    target = p.call("clean_target_distribution", {})
    assert "Target 'good' (classification)" in target
    assert "fitting rows: 0:" in target
    assert "scoring rows (latest training rows): 0:" in target

    assert "teams, 3 random rows:" in p.call("clean_sample_rows", {"table_name": "teams", "n": 3})


def test_profiling_flags_numbers_stored_as_text(task: RelationalTask) -> None:
    drivers = task.db["drivers"]
    as_text = drivers.df.assign(age=[str(20 + i) for i in range(len(drivers.df))])
    db = {**task.db, "drivers": type(drivers)(as_text, drivers.pkey, drivers.fkeys, drivers.time_col)}
    p = tools(Workspace(RelationalTask(**{**task.__dict__, "db": db})), "clean", "profiling")
    assert "parse as numbers" in p.call("clean_profile_table", {"table_name": "drivers"})


def test_profiling_names_an_unknown_table(workspace: Workspace) -> None:
    with pytest.raises(XmasError, match="No table 'nope'"):
        tools(workspace, "clean", "profiling").call("clean_profile_table", {"table_name": "nope"})


# ---------------------------------------------------------------------------- schema


def test_schema_lists_what_the_merge_functions_can_reach(workspace: Workspace) -> None:
    out = tools(workspace, "merge", "schema").call("merge_list_relations", {})
    assert "entity_columns can take: teamId, nationality" in out
    assert "- fk_column=teamId -> teams: columns country" in out
    assert "- table=results, fk_column=driverId: 1800 rows, median 60 per entity, time column date" in out
    # races are two steps away: results point to them
    assert "- table=races, via=results, fk_column=driverId: columns" in out
    assert "target_history: the entity's own past labels ('good' of earlier label rows)" in out
    assert "Not reachable" not in out


def test_schema_inspects_a_relation_for_fitting_rows_and_rows_to_predict(workspace: Workspace) -> None:
    out = tools(workspace, "merge", "schema").call(
        "merge_inspect_relation", {"table": "results", "fk_column": "driverId"}
    )
    assert "results via driverId: 1800 rows over 30 entities" in out
    assert "window" not in out and "last 30 days" not in out
    # races are weekly, so the latest result is a week old for both
    assert "fitting rows: a label row sees median" in out
    assert "rows to predict: a label row sees median" in out
    assert "the latest is median 7 days old" in out


def test_schema_rejects_a_key_that_does_not_point_to_the_entity(workspace: Workspace) -> None:
    with pytest.raises(XmasError, match="not a foreign key to 'drivers'"):
        tools(workspace, "merge", "schema").call("merge_inspect_relation", {"table": "results", "fk_column": "raceId"})


def test_schema_time_overview(workspace: Workspace) -> None:
    out = tools(workspace, "merge", "schema").call("merge_time_overview", {})
    assert out.startswith("Label rows: fitting 2020-03-11")
    assert "- teams: no time column" in out
    assert "- results.date: 2020-01-01 to" in out


# ------------------------------------------------------------------------ validation


def test_check_pipeline_reports_contracts_and_failures(workspace: Workspace) -> None:
    v = tools(workspace, "merge", "validation")
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE]})
    ok = v.call("merge_check_pipeline", {})
    assert ok.startswith("Runs:")
    assert "Features: 3 of the budget of 500." in ok
    workspace.pipeline = Pipeline.from_dict({"merge": [{"fn": "entity_columns", "columns": ["nope"]}]})
    assert v.call("merge_check_pipeline", {}).startswith("FAILS: merge/entity_columns")
    assert workspace.evaluations == 0  # checking is free


def test_score_pipeline_compares_with_identity_and_without_the_groups_slots(workspace: Workspace) -> None:
    v = tools(workspace, "merge", "validation")
    workspace.pipeline = Pipeline.from_dict({"clean": [PLACEHOLDER], "merge": [AGGREGATE]})
    out = v.call("merge_score_pipeline", {})
    assert "draft: roc_auc 0." in out
    assert "draft without your slots (merge, engineer): roc_auc" in out
    assert "Evaluations used: 2 of 30." in out  # the draft, and the draft without merge; identity is free
    v.call("merge_score_pipeline", {})
    assert workspace.evaluations == 2  # the same pipelines again come from the cache


def test_score_steps_measures_each_step(workspace: Workspace) -> None:
    v = tools(workspace, "merge", "validation")
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE]})
    assert "merge[0] aggregate(" in v.call("merge_score_steps", {})
    assert "(helps)" in v.call("merge_score_steps", {})

    # a log of a count changes nothing for trees; without the aggregate, the log step has no column
    log_count = {"fn": "log_transform", "columns": ["results.count()"]}
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE], "engineer": [log_count]})
    out = v.call("merge_score_steps", {})
    assert "merge[0] aggregate(" in out
    assert "needed by later steps" in out
    assert "engineer[0] log_transform(columns=['results.count()']): adds +0.0000 (no effect)" in out


def test_the_budget_stops_new_evaluations_but_not_cached_ones(task: RelationalTask) -> None:
    workspace = Workspace(task, max_evaluations=1)
    v = tools(workspace, "merge", "validation")
    workspace.pipeline = Pipeline.from_dict({"clean": [PLACEHOLDER], "merge": [AGGREGATE]})
    with pytest.raises(XmasError, match="budget of 1 is used up"):
        v.call("merge_score_pipeline", {})  # the draft fits the budget; the draft without merge does not
    assert workspace.evaluate(workspace.pipeline).score is not None  # scored before: still free
    assert workspace.baseline().score is not None  # the identity reference is always free
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE]})
    with pytest.raises(XmasError, match="used up"):
        v.call("merge_score_steps", {})  # a new draft is refused


def test_features_empty_on_the_rows_to_predict_are_errors(task: RelationalTask) -> None:
    # the rows to predict are new drivers: the database has nothing about them, though every tool score
    # (on the training rows) sees full features
    fresh = task.val.assign(driverId=task.val.driverId + 1000)
    workspace = Workspace(RelationalTask(**{**task.__dict__, "val": fresh}))
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE]})
    v = tools(workspace, "merge", "validation")
    out = v.call("merge_check_pipeline", {})
    assert out.startswith("ERRORS, fix before you finish")
    assert "results.mean(points): missing for 100.0% of the rows to predict" in out
    assert "results.count()" not in out.split("\n\n")[0].split("Runs:")[0]  # a count of 0 is a value
    report = v.call("merge_feature_report", {})
    assert "results.mean(points): missing 0.0% (rows to predict: 100.0%)" in report
    assert "EMPTY on the rows to predict" in report
    assert workspace.evaluations == 0  # free of budget


def test_no_errors_when_the_rows_to_predict_have_features(workspace: Workspace) -> None:
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE]})
    assert tools(workspace, "merge", "validation").call("merge_check_pipeline", {}).startswith("Runs:")


def test_feature_report_flags_static_columns(workspace: Workspace) -> None:
    workspace.pipeline = Pipeline.from_dict(
        {"merge": [{"fn": "entity_columns", "columns": ["nationality"]}, AGGREGATE]}
    )
    out = tools(workspace, "merge", "validation").call("merge_feature_report", {})
    assert "drivers.nationality: missing 0.0% (rows to predict: 0.0%), signal" in out
    assert "static column" in out
    assert "results.mean(points): missing" in out


def test_a_near_perfect_feature_counts_as_suspicious() -> None:
    y = pd.Series([0, 1] * 50)
    assert _signal(y.astype(float), y, "classification") == 1.0
    assert _signal(pd.Series(["a", "b"] * 50), y, "classification") == 1.0
    assert _signal(pd.Series([1.0] * 100), y, "classification") is None


# --------------------------------------------------------------------- orchestration


def test_orchestrator_status_and_submit(workspace: Workspace) -> None:
    o = tools(workspace, "orchestrator", "orchestration")
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE]})
    status = o.call("orchestrator_status", {})
    assert "Draft pipeline:\nclean        -" in status
    assert "Identity pipeline score:" in status
    assert "Not submitted yet." in status
    out = o.call("orchestrator_submit", {"summary": "History of results."})
    assert out.startswith("Submitted:")
    assert workspace.submitted
    assert workspace.summary == "History of results."


def test_orchestrator_validation_looks_at_every_slot(workspace: Workspace) -> None:
    workspace.pipeline = Pipeline.from_dict({"clean": [PLACEHOLDER], "merge": [AGGREGATE]})
    out = tools(workspace, "orchestrator", "validation").call("orchestrator_score_pipeline", {})
    assert "draft without your slots (clean, merge, engineer, postprocess)" in out


# ---------------------------------------------------------------------------- toolbox


def test_a_full_system_offers_every_tool_once(task: RelationalTask) -> None:
    toolbox = Toolbox(Workspace(task))
    Architecture.from_code("clean=PEV-EP-VE|merge=PEV-EP-VE").agents(toolbox)
    names = [t.name for t in toolbox.tools()]
    # filling (merge also has list_features), validation, profiling, schema
    assert len(index_tools(toolbox.tools())) == len(names) == 2 * (5 + 4) + 1 + 6 + 3
    assert {n.split("_", 1)[0] for n in names} == {"clean", "merge"}


def test_the_tools_never_see_the_validation_rows(workspace: Workspace) -> None:
    task, seen = workspace.task, workspace.tool_task
    assert len(seen.train) + len(seen.val) == len(task.train)  # only the training rows, split
    assert seen.train["date"].max() < seen.val["date"].min()  # split by time: fit on early, score on late
    assert seen.val["date"].max() < task.val["date"].min()  # every scoring row lies before the validation rows
    workspace.pipeline = Pipeline.from_dict({"merge": [AGGREGATE]})
    tools(workspace, "merge", "validation").call("merge_score_pipeline", {})
    assert workspace.baseline().score is not None


def test_a_task_can_keep_only_its_most_recent_training_rows(task: RelationalTask) -> None:
    limited = task.limit_train(100)
    assert len(limited.train) == 100
    assert limited.train["date"].min() >= task.train["date"].sort_values().iloc[-100]
    assert task.limit_train(None) is task
