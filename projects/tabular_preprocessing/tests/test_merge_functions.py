"""The merge functions added for the toolset improvements, and the rule that keeps past labels honest.

target_history, aggregate's where / count_by / mode / via, child_columns; and the harness rule that no
merge function sees a label of the rows it builds features for.
"""

import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import DAY, T0, with_licences
from tabular_preprocessing import REGISTRY, Pipeline, PipelineError, RelationalTask, Table, run
from tabular_preprocessing.inner_loop.harness import build_tables

WEEK = 7 * DAY


def features(pipeline: Pipeline, task: RelationalTask, seed: pd.DataFrame, train: pd.DataFrame | None = None):
    """The features built for ``seed`` (as rows to predict), fitted on ``train`` (default: the task's)."""
    return build_tables(pipeline, task, task.train if train is None else train, seed).x_test


def merge(*steps: dict) -> Pipeline:
    return Pipeline.from_dict({"merge": list(steps)})


# ---------------------------------------------------------------------------- target_history

HISTORY = {"fn": "target_history", "ops": ["count", "mean", "last", "time_since_last"]}


def test_target_history_aggregates_the_entitys_earlier_labels(task: RelationalTask) -> None:
    at = T0 + 20 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [0]})
    out = features(merge(HISTORY), task, seed)
    past = task.train[(task.train.driverId == 0) & (task.train.date < at)].sort_values("date")
    assert out["target_history.count(good)"].tolist() == [len(past)]  # races 10..19
    assert out["target_history.mean(good)"].tolist() == pytest.approx([past.good.mean()])
    assert out["target_history.last(good)"].tolist() == [past.good.iloc[-1]]
    assert out["target_history.time_since_last(good)"].tolist() == [7.0]


def test_a_label_counts_only_once_its_period_is_over(task: RelationalTask) -> None:
    # with a two-week horizon, the label of race 19 (one week before) is not known yet at race 20
    at = T0 + 20 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [0]})
    out = features(merge(HISTORY), replace(task, horizon=2 * WEEK), seed)
    known = task.train[(task.train.driverId == 0) & (task.train.date + 2 * WEEK <= at)]
    assert out["target_history.count(good)"].tolist() == [len(known)] == [9]  # races 10..18
    assert out["target_history.time_since_last(good)"].tolist() == [14.0]


def test_a_fitting_row_never_sees_its_own_label(task: RelationalTask) -> None:
    built = build_tables(merge(HISTORY), replace(task, horizon=WEEK), task.train, task.val)
    first = task.train.date == task.train.date.min()
    assert (built.x_train.loc[first, "target_history.count(good)"] == 0).all()  # nothing before the first race
    # every fitting row sees exactly its driver's labels whose week is over, never its own
    row = task.train.index[(task.train.driverId == 3) & (task.train.date == T0 + 15 * WEEK)][0]
    assert built.x_train.loc[row, "target_history.count(good)"] == 5  # races 10..14


def test_the_labels_of_the_rows_to_predict_are_never_used(task: RelationalTask) -> None:
    # the rows to predict carry their labels (scoring needs them); scrambling them changes no feature
    pipeline = merge(
        HISTORY,
        {"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]},
    )
    honest = build_tables(pipeline, task, task.train, task.val).x_test
    scrambled = task.val.assign(good=1 - task.val.good)
    assert build_tables(pipeline, task, task.train, scrambled).x_test.equals(honest)


def test_no_merge_function_sees_a_target_column(task: RelationalTask, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def peek(ctx, seed):  # a merge function that records what it is given
        seen.append(list(seed.columns))
        return pd.DataFrame(index=seed.index)

    monkeypatch.setitem(REGISTRY["merge"], "peek", peek)
    build_tables(merge({"fn": "peek"}), task, task.train, task.val)
    assert seen and all("good" not in columns for columns in seen)


def test_target_history_without_labels_or_ops_is_refused(task: RelationalTask) -> None:
    with pytest.raises(PipelineError, match="ops must not be empty"):
        features(merge({"fn": "target_history", "ops": []}), task, task.val.head(3))


# ------------------------------------------------------------- aggregate: where, count_by, mode


def results_before(task: RelationalTask, driver: int, at: pd.Timestamp) -> pd.DataFrame:
    r = task.db["results"].df
    return r[(r.driverId == driver) & (r.date < at)]


def test_where_keeps_only_the_given_values_and_converts_numbers_given_as_text(task: RelationalTask) -> None:
    at = T0 + 30 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [2]})
    step = {
        "fn": "aggregate",
        "table": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
        "where": {"position": ["1", "2", "3"]},
    }
    out = features(merge(step), task, seed)
    rows = results_before(task, 2, at).query("position <= 3")
    assert out["results.count()[position=1|2|3]"].tolist() == [len(rows)]
    assert out["results.mean(points)[position=1|2|3]"].tolist() == pytest.approx([rows.points.mean()])


def test_count_by_and_mode_summarise_a_categorical_column(task: RelationalTask) -> None:
    results = task.db["results"]
    status = np.where(results.df.position > 5, "dnf", "finished")  # about 3 of 4 are dnf: no tie
    db = {**task.db, "results": replace(results, df=results.df.assign(status=status))}
    on = replace(task, db=db)
    at = T0 + 30 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [4]})
    step = {
        "fn": "aggregate",
        "table": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "count_by", "column": "status"}, {"op": "mode", "column": "status"}],
    }
    out = features(merge(step), on, seed)
    counts = results_before(on, 4, at).status.value_counts()
    assert out["results.count_by(status=finished)"].tolist() == [counts.get("finished", 0)]
    assert out["results.count_by(status=dnf)"].tolist() == [counts.get("dnf", 0)]
    assert counts.iloc[0] != counts.iloc[-1]  # no tie, so the mode is well defined
    assert out["results.mode(status)"].tolist() == [counts.idxmax()]


def test_an_op_without_its_column_is_refused(task: RelationalTask) -> None:
    step = {"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "mean"}]}
    with pytest.raises(PipelineError, match="has no column None"):
        features(merge(step), task, task.val.head(2))


# ------------------------------------------------------------------------ aggregate: via


def test_via_reaches_a_table_two_steps_away(task: RelationalTask) -> None:
    # drivers -> results -> races: the races a driver took part in, before the label time
    at = T0 + 3 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [0]})
    step = {
        "fn": "aggregate",
        "table": "races",
        "via": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "count"}, {"op": "time_since_last"}, {"op": "max", "column": "points"}],
    }
    out = features(merge(step), task, seed)
    assert out["results>races.count()"].tolist() == [3]  # races 0, 1, 2; race 3 is at the label time
    assert out["results>races.time_since_last(date)"].tolist() == [7.0]
    # a column of the via table (points) can be aggregated too
    assert out["results>races.max(points)"].tolist() == pytest.approx([results_before(task, 0, at).points.max()])


def test_via_needs_a_key_from_the_via_table_to_the_target(task: RelationalTask) -> None:
    step = {
        "fn": "aggregate",
        "table": "teams",
        "via": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "count"}],
    }
    with pytest.raises(PipelineError, match="exactly one foreign key from 'results' to 'teams'"):
        features(merge(step), task, task.val.head(2))


# ---------------------------------------------------------------------------- child_columns


def test_child_columns_copy_a_one_to_one_table_before_the_label_time(task: RelationalTask) -> None:
    on = with_licences(task)
    seed = pd.DataFrame({"date": [T0 + 3 * WEEK, T0 + 3 * WEEK], "driverId": [1, 5]})
    step = {"fn": "child_columns", "table": "licences", "fk_column": "driverId", "columns": ["grade"]}
    out = features(merge(step), on, seed)
    # driver 1's licence was issued in week 1, driver 5's only in week 5: after the label time
    assert out["licences.grade"].tolist()[0] == 1
    assert np.isnan(out["licences.grade"].tolist()[1])


def test_child_columns_refuses_a_table_with_several_rows_per_entity(task: RelationalTask) -> None:
    step = {"fn": "child_columns", "table": "results", "fk_column": "driverId", "columns": ["points"]}
    with pytest.raises(PipelineError, match="several rows per entity"):
        features(merge(step), task, task.val.head(2))


# --------------------------------------------------------------------- steps and messages


@pytest.mark.parametrize(
    "step",
    [
        {"function": "entity_columns", "columns": ["nationality"]},
        {"entity_columns": {"columns": ["nationality"]}},
    ],
)
def test_two_common_wrong_step_forms_are_read_as_meant(step: dict) -> None:
    assert Pipeline.from_dict({"merge": [step]}).merge[0].fn == "entity_columns"


def test_a_step_without_fn_gets_a_correct_example() -> None:
    with pytest.raises(PipelineError, match=re.escape('e.g. {"fn": "entity_columns"')):
        Pipeline.from_dict({"merge": [{"colums": ["x"]}]})  # fits no function


def test_a_misspelled_feature_name_gets_the_closest_names(task: RelationalTask) -> None:
    pipeline = Pipeline.from_dict(
        {
            "merge": [
                {"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]}
            ],
            "engineer": [{"fn": "log_transform", "columns": ["results.count"]}],
        }
    )
    with pytest.raises(PipelineError, match=r"closest: results\.count\(\)"):
        run(pipeline, task, task.train, task.val)


# ------------------------------------------------------------ fixes after grid81_v2


@pytest.mark.parametrize(
    ("stage", "step", "fn"),
    [
        # no fn at all: only aggregate takes table, fk_column and aggregations
        ("merge", {"table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]}, "aggregate"),
        ("merge", {"ops": ["count"]}, "target_history"),
        ("clean", {"table": "results", "values": [-1.0]}, "placeholder_as_missing"),
        # the function name as a key set to true
        ("merge", {"entity_columns": "true", "columns": ["nationality"]}, "entity_columns"),
        ("engineer", {"count_encode": True, "columns": ["drivers.nationality"]}, "count_encode"),
    ],
)
def test_a_step_without_fn_is_read_when_only_one_function_fits(stage: str, step: dict, fn: str) -> None:
    assert getattr(Pipeline.from_dict({stage: [step]}), stage)[0].fn == fn


def test_a_step_that_fits_several_functions_is_still_refused() -> None:
    # log_transform, count_encode, missing_indicators ... all take just columns
    with pytest.raises(PipelineError, match="'fn' names the function"):
        Pipeline.from_dict({"engineer": [{"columns": ["results.count()"]}]})


def test_last_is_the_value_of_the_latest_row(task: RelationalTask) -> None:
    at = T0 + 12 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [6]})
    step = {
        "fn": "aggregate",
        "table": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "last", "column": "points"}],
    }
    out = features(merge(step), task, seed)
    latest = results_before(task, 6, at).sort_values("date").points.iloc[-1]  # race 11: before the label time
    assert out["results.last(points)"].tolist() == pytest.approx([latest])


def test_boolean_columns_and_targets_are_aggregated_as_numbers(task: RelationalTask) -> None:
    results = task.db["results"]
    db = {**task.db, "results": replace(results, df=results.df.assign(podium=results.df.position <= 3))}
    on = replace(task, db=db, train=task.train.assign(good=task.train.good.astype(bool)))
    at = T0 + 30 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [1]})
    pipeline = merge(
        {
            "fn": "aggregate",
            "table": "results",
            "fk_column": "driverId",
            "aggregations": [{"op": "mean", "column": "podium"}, {"op": "sum", "column": "podium"}],
        },
        {"fn": "target_history", "ops": ["mean", "last"]},
    )
    out = features(pipeline, on, seed)
    rows = results_before(on, 1, at)
    assert out["results.mean(podium)"].tolist() == pytest.approx([rows.podium.mean()])
    assert out["results.sum(podium)"].tolist() == [rows.podium.sum()]
    past = on.train[(on.train.driverId == 1) & (on.train.date < at)]
    assert out["target_history.mean(good)"].tolist() == pytest.approx([past.good.mean()])


def test_a_table_with_two_keys_to_the_entity_is_named_by_the_key(task: RelationalTask) -> None:
    duels = pd.DataFrame({"duelId": range(4), "winner": [0, 0, 1, 2], "loser": [1, 2, 0, 0], "date": [T0] * 4})
    db = {**task.db, "duels": Table(duels, "duelId", {"winner": "drivers", "loser": "drivers"}, "date")}
    on = replace(task, db=db)
    seed = pd.DataFrame({"date": [T0 + WEEK], "driverId": [0]})
    pipeline = merge(
        {"fn": "aggregate", "table": "duels", "fk_column": "winner", "aggregations": [{"op": "count"}]},
        {"fn": "aggregate", "table": "duels", "fk_column": "loser", "aggregations": [{"op": "count"}]},
    )
    out = features(pipeline, on, seed)  # before, both were duels.count(): refused as a repeated column
    assert out["duels[winner].count()"].tolist() == [2]
    assert out["duels[loser].count()"].tolist() == [2]
    # a table with one key to the entity keeps its plain name
    plain = features(
        merge({"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]}),
        task,
        seed,
    )
    assert "results.count()" in plain


@pytest.mark.parametrize(
    "step",
    [
        {"fn": "entity_columns", "columns": ["drivers.nationality"]},
        {"fn": "lookup", "fk_column": "teamId", "columns": ["teams.country"]},
        {"fn": "child_columns", "table": "licences", "fk_column": "driverId", "columns": ["licences.grade"]},
    ],
)
def test_a_table_prefix_on_a_column_is_removed(task: RelationalTask, step: dict) -> None:
    out = features(merge(step), with_licences(task), pd.DataFrame({"date": [T0 + 9 * WEEK], "driverId": [1]}))
    prefixed = step["columns"][0]
    assert list(out.columns)[-1] == prefixed  # the feature keeps its usual name, from the bare column


def test_a_column_built_again_with_the_same_values_is_skipped(task: RelationalTask) -> None:
    count = {"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]}
    both = {**count, "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}]}
    built = build_tables(merge(count, both), task, task.train, task.val)
    assert list(built.x_train.columns) == ["date", "results.count()", "results.mean(points)"]
    assert built.skipped == {str(Pipeline.from_dict({"merge": [both]}).merge[0]): ["results.count()"]}


def test_a_column_built_again_with_other_values_is_refused(
    task: RelationalTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    def impostor(ctx, seed):  # same name as the real count, other values
        return pd.DataFrame({"results.count()": 99.0}, index=seed.index)

    monkeypatch.setitem(REGISTRY["merge"], "impostor", impostor)
    count = {"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]}
    with pytest.raises(PipelineError, match="already built with other values"):
        build_tables(merge(count, {"fn": "impostor"}), task, task.train, task.val)


# --------------------------------------------------------------- round 3: after grid81_v2


def test_time_since_first_is_the_age_of_the_first_row(task: RelationalTask) -> None:
    at = T0 + 12 * WEEK
    seed = pd.DataFrame({"date": [at], "driverId": [3]})
    step = {
        "fn": "aggregate",
        "table": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "time_since_first"}, {"op": "time_since_last"}],
    }
    out = features(merge(step), task, seed)
    assert out["results.time_since_first(date)"].tolist() == [84.0]  # first race in week 0
    assert out["results.time_since_last(date)"].tolist() == [7.0]


def test_features_empty_on_the_rows_to_predict_are_dropped_when_asked(task: RelationalTask) -> None:
    fresh = task.val.assign(driverId=task.val.driverId + 1000)  # new drivers: the database knows nothing of them
    pipeline = merge(
        {"fn": "entity_columns", "columns": ["nationality"]},
        {
            "fn": "aggregate",
            "table": "results",
            "fk_column": "driverId",
            "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
        },
    )
    kept = build_tables(pipeline, task, task.train, fresh)
    assert "results.mean(points)" in kept.x_train and kept.dropped == []  # off by default (the baselines)
    built = build_tables(pipeline, task, task.train, fresh, drop_empty=0.95)
    # a count of 0 is a value; the time column always stays
    assert built.dropped == ["drivers.nationality", "results.mean(points)"]
    assert list(built.x_train.columns) == list(built.x_test.columns) == ["date", "results.count()"]
    # on rows the database knows, nothing is dropped
    assert build_tables(pipeline, task, task.train, task.val, drop_empty=0.95).dropped == []
