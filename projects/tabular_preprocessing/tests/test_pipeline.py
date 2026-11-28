import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import DAY, T0, with_licences
from tabular_preprocessing import (
    REGISTRY,
    STAGES,
    Pipeline,
    PipelineError,
    RelationalTask,
    evaluate,
    run,
)
from tabular_preprocessing.inner_loop.harness import clean_database

BASE_MERGE = [
    {"fn": "entity_columns", "columns": ["nationality"]},
    {
        "fn": "aggregate",
        "table": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
    },
]

# one valid call per library function, on top of BASE_MERGE
EXAMPLES: dict[str, dict[str, object]] = {
    "zero_as_missing": {"table": "results", "columns": ["points"]},
    "placeholder_as_missing": {"table": "results", "values": [-1.0]},
    "to_numeric": {"table": "results", "columns": ["position"]},
    "missing_indicators": {"table": "results", "columns": ["points"]},
    "drop_columns": {"table": "results", "columns": ["position"]},
    "drop_duplicate_rows": {"table": "results"},
    "entity_columns": {"columns": ["teamId"]},
    "lookup": {"fk_column": "teamId", "columns": ["country"]},
    "aggregate": {
        "table": "results",
        "fk_column": "driverId",
        "aggregations": [{"op": "max", "column": "points"}],
    },
    "child_columns": {"table": "licences", "fk_column": "driverId", "columns": ["grade"]},
    "target_history": {"ops": ["count", "mean", "last"]},
    "combine": {"a": "results.mean(points)", "b": "results.count()", "op": "div"},
    "group_stat": {"group": "drivers.nationality", "value": "results.mean(points)"},
    "count_encode": {"columns": ["drivers.nationality"]},
    "log_transform": {"columns": ["results.count()"]},
    "datetime_parts": {"column": "date"},
    "drop_features": {"columns": ["drivers.nationality"]},
    "full": {},
    "random_views": {"n_views": 2},
    "recent": {},
    "stratified_views": {"n_views": 2},
    "oversample_minority": {},
    "prior_shift": {"strength": 0.5},
    "temperature": {"t": 2.0},
    "clip_to_train_range": {},
}


def pipeline_with(stage: str, fn: str) -> Pipeline:
    call = {"fn": fn, **EXAMPLES[fn]}
    data: dict[str, object] = {"merge": list(BASE_MERGE)}
    if stage == "sample":
        data["sample"] = call
    else:
        data[stage] = [*data.get(stage, []), call]  # type: ignore[misc]
    return Pipeline.from_dict(data)  # type: ignore[arg-type]


def test_every_function_has_an_example() -> None:
    assert set(EXAMPLES) == {fn for fns in REGISTRY.values() for fn in fns}


@pytest.mark.parametrize("stage", STAGES)
def test_every_function_runs_inside_the_harness(task: RelationalTask, stage: str) -> None:
    regression = replace(task, task_type="regression")
    for fn in REGISTRY[stage]:
        on = with_licences(regression if fn == "clip_to_train_range" else task)
        result = evaluate(pipeline_with(stage, fn), on)
        assert np.isfinite(result.score), fn


def test_identity_uses_only_the_prediction_time(task: RelationalTask) -> None:
    result = evaluate(Pipeline.identity(), task)
    assert result.metric == "roc_auc"
    assert result.run.features == ["date"]
    assert [r.stage for r in result.run.log] == ["sample", "predictor"]


def test_merged_history_beats_the_identity_baseline(task: RelationalTask) -> None:
    merged = evaluate(Pipeline.from_dict({"merge": BASE_MERGE}), task)
    assert merged.score > evaluate(Pipeline.identity(), task).score + 0.2


def test_aggregate_sees_only_rows_strictly_before_the_label_time(task: RelationalTask) -> None:
    seed = pd.DataFrame({"date": [T0 + 7 * 3 * DAY], "driverId": [0], "good": [1]})
    pipeline = Pipeline.from_dict(
        {
            "merge": [
                {
                    "fn": "aggregate",
                    "table": "results",
                    "fk_column": "driverId",
                    "aggregations": [{"op": "count"}, {"op": "time_since_last"}],
                },
            ]
        }
    )
    out = _features(pipeline, task, seed)
    assert list(out.columns) == ["date", "results.count()", "results.time_since_last(date)"]
    # races in weeks 0, 1, 2 are before; the race in week 3 is at the label time, so excluded
    assert out["results.count()"].tolist() == [3.0]
    assert out["results.time_since_last(date)"].tolist() == [7.0]


def test_aggregate_has_no_time_windows() -> None:
    step = {"fn": "aggregate", "table": "results", "fk_column": "driverId", "aggregations": [{"op": "count"}]}
    with pytest.raises(PipelineError, match="takes no parameters \\['window_days'\\]"):
        Pipeline.from_dict({"merge": [{**step, "window_days": 30}]})


def test_lookup_and_entity_columns_follow_the_keys(task: RelationalTask) -> None:
    seed = pd.DataFrame({"date": [T0, T0], "driverId": [3, 999], "good": [1, 0]})
    pipeline = Pipeline.from_dict(
        {
            "merge": [
                {"fn": "entity_columns", "columns": ["teamId"]},
                {"fn": "lookup", "fk_column": "teamId", "columns": ["country"]},
            ]
        }
    )
    out = _features(pipeline, task, seed)
    team = task.db["drivers"].df.set_index("driverId").teamId[3]
    assert out["drivers.teamId"].iloc[0] == team
    assert out["teams.country"].iloc[0] == "ABCDE"[team]
    assert out[["drivers.teamId", "teams.country"]].iloc[1].isna().all()  # unknown driver: no row, not an error


def test_cleaning_runs_before_merging(task: RelationalTask) -> None:
    seed = task.train.head(50)
    raw = _features(Pipeline.from_dict({"merge": BASE_MERGE[1:]}), task, seed)
    cleaned = _features(
        Pipeline.from_dict(
            {"clean": [{"fn": "placeholder_as_missing", "table": "results", "values": [-1.0]}], "merge": BASE_MERGE[1:]}
        ),
        task,
        seed,
    )
    assert (cleaned["results.mean(points)"] != raw["results.mean(points)"]).any()


def test_cleaning_cannot_touch_keys(task: RelationalTask) -> None:
    pipeline = Pipeline.from_dict({"clean": [{"fn": "drop_columns", "table": "results", "columns": ["driverId"]}]})
    with pytest.raises(PipelineError, match="clean/drop_columns"):
        evaluate(pipeline, task)


@pytest.mark.parametrize(
    "call",
    [
        {
            "fn": "aggregate",
            "table": "results",
            "fk_column": "raceId",
            "aggregations": [{"op": "count"}],
        },  # not to drivers
        {
            "fn": "aggregate",
            "table": "results",
            "fk_column": "driverId",
            "aggregations": [{"op": "mean", "column": "nope"}],
        },
        {
            "fn": "aggregate",
            "table": "results",
            "fk_column": "driverId",
            "aggregations": [{"op": "median", "column": "points"}],
        },
        {"fn": "lookup", "fk_column": "nationality", "columns": ["country"]},  # not a foreign key
    ],
)
def test_invalid_merges_fail_with_their_reason(task: RelationalTask, call: dict[str, object]) -> None:
    with pytest.raises(PipelineError, match=f"merge/{call['fn']}"):
        evaluate(Pipeline.from_dict({"merge": [call]}), task)


def test_a_duplicated_entity_key_is_a_join_explosion(task: RelationalTask) -> None:
    drivers = task.db["drivers"]
    db = {**task.db, "drivers": replace(drivers, df=pd.concat([drivers.df, drivers.df.head(1)]))}
    with pytest.raises(PipelineError, match="not unique"):
        evaluate(Pipeline.from_dict({"merge": BASE_MERGE[:1]}), replace(task, db=db))


def test_the_column_budget_is_enforced(task: RelationalTask) -> None:
    with pytest.raises(PipelineError, match="budget"):
        evaluate(Pipeline.from_dict({"merge": BASE_MERGE}), task, max_columns=2)


def test_pipeline_round_trips_through_json() -> None:
    data = {
        "clean": [{"fn": "placeholder_as_missing", "table": "results", "values": [-1]}],
        "merge": BASE_MERGE,
        "engineer": [{"fn": "log_transform", "columns": ["results.count()"]}],
        "sample": {"fn": "random_views", "n_views": 2},
        "postprocess": [{"fn": "temperature", "t": 2.0}],
        "model_kwargs": {"max_iter": 10},
    }
    pipeline = Pipeline.from_dict(data)
    assert Pipeline.from_dict(json.loads(json.dumps(pipeline.to_dict()))).to_dict() == data
    assert pipeline.describe().splitlines()[0] == "clean        placeholder_as_missing[results](values=[-1])"


@pytest.mark.parametrize(
    "data",
    [
        {"clean": [{"fn": "teleport", "table": "results"}]},
        {"clean": [{"fn": "drop_duplicate_rows"}]},  # no table
        {"clean": [{"fn": "aggregate", "table": "results"}]},  # wrong stage
        {"merge": [{"fn": "lookup", "fk_column": "teamId", "columns": ["country"], "colour": "red"}]},
        {"merge": [{"fn": "lookup", "fk_column": "teamId"}]},  # missing parameter
        {"preprocess": []},  # old slot name
        {"merge": {"fn": "lookup"}},  # not a list
    ],
)
def test_invalid_pipelines_are_rejected(data: dict[str, object]) -> None:
    with pytest.raises(PipelineError):
        Pipeline.from_dict(data)


def test_options_describe_every_slot() -> None:
    options = Pipeline.options()
    assert list(options) == list(STAGES)
    assert options["clean"]["zero_as_missing"].startswith("(table: str, columns: list[str])")
    assert "where: dict[str, list[typing.Any]] | None = None" in options["merge"]["aggregate"]
    assert "via: str | None = None" in options["merge"]["aggregate"]
    assert "window_days" not in options["merge"]["aggregate"]


def test_describe_renders_the_schema(task: RelationalTask) -> None:
    text = task.describe()
    assert "foreign key: driverId -> drivers" in text
    assert "time column: date" in text


def _features(pipeline: Pipeline, task: RelationalTask, seed: pd.DataFrame) -> pd.DataFrame:
    """The merged features for ``seed``, read back through a predictor that records its input."""
    seen: list[pd.DataFrame] = []

    class Recorder:
        classes_ = np.array([0, 1])

        def fit(self, x: pd.DataFrame, y: pd.Series) -> "Recorder":
            return self

        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            seen.append(x)
            return np.full((len(x), 2), 0.5)

    run(pipeline, task, seed, seed, model_factory=lambda *a, **k: Recorder())
    out = seen[0].copy()
    for c in out.columns:  # undo the predictor's categorical encoding for readable assertions
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object)
    return out


def test_parameters_are_converted_to_their_declared_types() -> None:
    # a model wrote a number as text in the first real run; accepted steps must also run
    pipeline = Pipeline.from_dict({"sample": {"fn": "random_views", "n_views": "4"}})
    assert pipeline.sample.params["n_views"] == 4
    with pytest.raises(PipelineError, match="parameter n_views='four'"):
        Pipeline.from_dict({"sample": {"fn": "random_views", "n_views": "four"}})
    with pytest.raises(PipelineError, match="parameter op='pow'"):
        Pipeline.from_dict({"engineer": [{"fn": "combine", "a": "x", "b": "y", "op": "pow"}]})


def test_duplicates_are_equal_in_everything_but_the_primary_key(task: RelationalTask) -> None:
    # results of different drivers or races with equal values are not duplicates
    results = task.db["results"].df
    copy = results.head(5).assign(resultId=range(10_000, 10_005))  # a true duplicate: only the id differs
    db = {**task.db, "results": replace(task.db["results"], df=pd.concat([results, copy], ignore_index=True))}
    pipeline = Pipeline.from_dict({"clean": [{"fn": "drop_duplicate_rows", "table": "results"}]})
    cleaned, _ = clean_database(pipeline, replace(task, db=db))
    assert len(cleaned["results"].df) == len(results)
    assert list(cleaned["results"].df.columns) == list(results.columns)


def test_a_table_with_only_constant_features_gets_a_constant_prediction(task: RelationalTask) -> None:
    # all label rows at one cutoff: the identity pipeline's only feature is constant; TabPFN refuses that
    def refuses_constant(task_type: str, **kwargs: object) -> object:
        raise AssertionError("the predictor must not be fitted on constant features")

    one_cutoff = replace(task, train=task.train.assign(**{task.time_col: task.train[task.time_col].max()}))
    result = evaluate(Pipeline.identity(), one_cutoff, refuses_constant)
    assert result.score == 0.5  # the prior for every row
