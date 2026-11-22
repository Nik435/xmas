"""The fixed harness: runs a pipeline around a fixed predictor, checks its contracts, scores it.

As in TabFM-Auto, the pipeline never calls the predictor; the harness does::

    clean (per table) -> merge (train and test label rows alike) -> engineer
    -> sample (views) -> predictor per view, predictions averaged -> postprocess

The harness enforces the contracts a validator checks, so a broken pipeline
fails loudly instead of scoring quietly:

- cleaning never sees a table's keys or time column, so it cannot break a join;
- merging returns exactly one row per label row (no lost rows, no join explosion),
  and the same calls run on train and test rows; a column an earlier step already built
  with the same values is skipped, one with the same name but other values is refused;
- the feature count stays within ``max_columns``;
- with ``drop_empty`` (on wherever an agent's pipeline is scored), a feature that is empty for
  nearly all rows to predict is left out: it cannot help there, and agents tend to keep it.

The two halves are separate functions: :func:`build_tables` turns a pipeline into
the predictor's train and test tables, :func:`fit_predict` fits the predictor on
them and predicts. :func:`run` is both; :func:`evaluate` runs on the task's train
and validation rows (RelArena's inner split) and :func:`score` scores the result, so
pipelines can be compared with :meth:`Pipeline.identity`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

from tabular_preprocessing.inner_loop.data import Database, RelationalTask, Table, TaskType
from tabular_preprocessing.inner_loop.library import REGISTRY, MergeContext
from tabular_preprocessing.inner_loop.pipeline import Call, Pipeline, PipelineError

ModelFactory = Callable[..., Any]
EMPTY_FEATURE = 0.95  # drop_empty: a feature at least this empty on the rows to predict is left out


def default_model(task_type: TaskType, **kwargs: Any) -> Any:
    """The fixed predictor: sklearn's histogram gradient boosting.

    A stand-in until the predictor is decided (TabFM, TabPFN, LightGBM): like a
    tabular foundation model, it takes raw numeric and categorical columns with
    missing values. Any factory ``(task_type, **model_kwargs) -> estimator`` with
    ``fit`` and ``predict`` / ``predict_proba`` can replace it.
    """
    if task_type == "classification":
        return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0, **kwargs)
    return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0, **kwargs)


TABPFN_PREDICT_ROWS = 8192  # rows per prediction batch: TabPFN's default (32,768) can exceed GPU memory


def tabpfn_v2_model(task_type: TaskType, **kwargs: Any) -> Any:
    """TabPFN v2, the tabular foundation model RDBLearn uses by default; on a GPU if there is one.

    Fit on at most ``max_rows`` rows (10,000, TabPFN v2's pretraining limit, as in RDBLearn). It
    reads the same frames as the gradient boosting: numbers, pandas categoricals, missing values.
    The weights download on first use (no login for v2). Prior Labs License: see the tabpfn package.
    """
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings

    settings.tabpfn.max_batched_test_rows = TABPFN_PREDICT_ROWS
    estimator = TabPFNClassifier if task_type == "classification" else TabPFNRegressor
    return estimator.create_default_for_version(
        ModelVersion.V2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=0,
        ignore_pretraining_limits=True,
        **kwargs,
    )


@dataclass(frozen=True)
class StageRecord:
    """What one call did.

    Attributes:
        stage: The stage.
        call: The call, as text.
        shape: For ``clean``, the table's shape after the call; for ``sample``, (views,
            rows in the largest); otherwise the training features' shape.
        seconds: Wall time.
    """

    stage: str
    call: str
    shape: tuple[int, int]
    seconds: float


@dataclass
class RunResult:
    """Predictions of one run, plus its per-call log.

    Attributes:
        pred: Class probabilities ``(n, k)`` (columns in ``classes`` order) or values ``(n,)``.
        classes: The class labels, for classification.
        features: The training features the predictor saw.
        log: One record per call and one for the predictor.
    """

    pred: np.ndarray
    classes: np.ndarray | None
    features: list[str]
    log: list[StageRecord] = field(default_factory=list)


@dataclass
class Tables:
    """The predictor's input, built by a pipeline: one row per label row, one column per feature.

    Attributes:
        x_train: Training features.
        y_train: Training target.
        x_test: Test features, same columns as ``x_train``.
        views: Row indices of ``x_train`` per context view; the predictor is fitted once per view.
        log: One record per pipeline call that built them.
        skipped: Per merge call (as text), the columns it built again with the values an earlier call
            had already given them; they were left out.
        dropped: The features left out because they were empty for nearly all rows to predict.
    """

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    views: list[np.ndarray]
    log: list[StageRecord] = field(default_factory=list)
    skipped: dict[str, list[str]] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)


def clean_database(pipeline: Pipeline, task: RelationalTask) -> tuple[Database, list[StageRecord]]:
    """Apply the pipeline's clean slot to the task's database.

    Args:
        pipeline: The pipeline; only its ``clean`` calls are used.
        task: The task whose database is cleaned (it is not changed; a new one is returned).

    Returns:
        The cleaned database and one record per clean call.

    Raises:
        PipelineError: If a call names an unknown table, fails or adds rows.
    """
    log: list[StageRecord] = []
    db = dict(task.db)
    for call in pipeline.clean:
        start = time.perf_counter()
        if call.table not in db:
            raise PipelineError(f"clean/{call.fn}: no table {call.table!r}; tables are {sorted(db)}.")
        db[call.table] = _clean(call, db[call.table])
        log.append(_record("clean", call, db[call.table].df.shape, start))
    return db, log


def build_tables(
    pipeline: Pipeline,
    task: RelationalTask,
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_rows: int = 10_000,
    max_columns: int = 500,
    cleaned: tuple[Database, list[StageRecord]] | None = None,
    history: pd.DataFrame | None = None,
    drop_empty: float | None = None,
) -> Tables:
    """Build the predictor's train and test tables from a pipeline: clean, merge, engineer, sample.

    Args:
        pipeline: The pipeline.
        task: The task: database and label columns.
        train: Label rows to fit on (entity id, time, target).
        test: Label rows to predict (entity id, time; the target is ignored).
        max_rows: Most rows a context view may hold.
        max_columns: Most feature columns the predictor may receive.
        cleaned: The result of :func:`clean_database` for this pipeline, if already known (a cache: the
            clean slot of a large database can take seconds, and many drafts share it).
        history: The label rows whose labels merge functions may use as past labels (``target_history``);
            ``train`` by default. A check on a sample of ``train`` passes all fitting rows here, so the
            sampled rows see the past labels a real run would.
        drop_empty: Leave out every feature (except the prediction time) that is missing for at least
            this share of the ``test`` rows; ``None`` keeps them all. Label-free, like the engineer stage.

    Returns:
        The tables, the context views and the log.

    Raises:
        PipelineError: If a call fails or breaks a contract.
    """
    db, log = cleaned if cleaned is not None else clean_database(pipeline, task)
    log = list(log)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    x_train = train[[task.time_col]].copy()
    x_test = test[[task.time_col]].copy()
    y_train = train[task.target_col]
    # the merge functions see the label rows without their target: the rows to predict may carry
    # theirs (scoring needs it), and no merge function may read it. Past labels reach them only as
    # ctx.labels, the fitting rows' labels.
    seeds = [rows.drop(columns=[task.target_col], errors="ignore") for rows in (train, test)]
    labels = (train if history is None else history)[[task.entity_col, task.time_col, task.target_col]]

    skipped: dict[str, list[str]] = {}
    # as many threads as this process may use (a cluster job's cores, not the whole node)
    con = duckdb.connect(":memory:", config={"threads": len(os.sched_getaffinity(0))})
    try:
        for name, table in db.items():
            con.register(name, table.df)
        ctx = MergeContext(
            db, con, task.entity_table, task.entity_col, task.time_col, labels, task.target_col, task.horizon
        )
        for call in pipeline.merge:
            start = time.perf_counter()
            new_train = _invoke("merge", call, ctx, seeds[0])
            new_test = _invoke("merge", call, ctx, seeds[1])
            for new, seed in ((new_train, train), (new_test, test)):
                if len(new) != len(seed) or not new.index.equals(seed.index):
                    raise PipelineError(f"merge/{call} must return exactly one row per label row.")
            clash = [c for c in new_train.columns if c in x_train.columns]
            same = [c for c in clash if new_train[c].equals(x_train[c]) and new_test[c].equals(x_test[c])]
            if len(same) < len(clash):
                other = sorted(set(clash) - set(same))
                raise PipelineError(
                    f"merge/{call} builds columns {other} that an earlier step already built with other values "
                    "(e.g. a lookup of the same table through another key); rename or remove one of the steps."
                )
            if same:  # the same name is the same computation: keep the first copy
                skipped[str(call)] = same
                new_train, new_test = new_train.drop(columns=same), new_test.drop(columns=same)
            x_train = pd.concat([x_train, new_train], axis=1)
            x_test = pd.concat([x_test, new_test], axis=1)
            log.append(_record("merge", call, x_train.shape, start))
    finally:
        con.close()

    for call in pipeline.engineer:
        start = time.perf_counter()
        x_train, x_test = _invoke("engineer", call, x_train, y_train, x_test)
        if len(x_train) != len(train) or len(x_test) != len(test):
            raise PipelineError(f"engineer/{call} must keep every row.")
        if list(x_train.columns) != list(x_test.columns):
            raise PipelineError(f"engineer/{call} must give train and test the same columns.")
        log.append(_record("engineer", call, x_train.shape, start))

    dropped: list[str] = []
    if drop_empty is not None and len(x_test):
        dropped = [c for c in x_test.columns if c != task.time_col and x_test[c].isna().mean() >= drop_empty]
        x_train, x_test = x_train.drop(columns=dropped), x_test.drop(columns=dropped)

    if x_train.shape[1] > max_columns:
        raise PipelineError(f"{x_train.shape[1]} feature columns exceed the budget of {max_columns}.")

    start = time.perf_counter()
    views: list[np.ndarray] = _invoke("sample", pipeline.sample, x_train, y_train, x_test, max_rows)
    if not views or any(len(v) == 0 for v in views):
        raise PipelineError(f"sample/{pipeline.sample} returned an empty view.")
    log.append(_record("sample", pipeline.sample, (len(views), max(len(v) for v in views)), start))
    return Tables(x_train, y_train, x_test, views, log, skipped, dropped)


def fit_predict(
    tables: Tables,
    pipeline: Pipeline,
    task_type: TaskType,
    model_factory: ModelFactory = default_model,
) -> RunResult:
    """Fit the predictor on each view of the train table, predict the test table, postprocess.

    Args:
        tables: The tables a pipeline built.
        pipeline: The same pipeline, for its ``model_kwargs`` and ``postprocess`` calls.
        task_type: ``"classification"`` or ``"regression"``.
        model_factory: Builds the fixed predictor from ``task_type`` and ``pipeline.model_kwargs``.

    Returns:
        Predictions for the test table and the log (the table-building records first).

    Raises:
        PipelineError: If a postprocess call fails.
    """
    log = list(tables.log)
    start = time.perf_counter()
    features = list(tables.x_train.columns)
    x_train, x_test = _model_frames(tables.x_train, tables.x_test)
    y_train = tables.y_train
    classes = np.unique(y_train) if task_type == "classification" else None
    preds = []
    for view in tables.views:
        model = model_factory(task_type, **pipeline.model_kwargs)
        model.fit(x_train.iloc[view], y_train.iloc[view])
        preds.append(_predict(model, x_test, classes))
    pred = np.mean(preds, axis=0)
    fit = Call(f"fit x{len(tables.views)}", dict(pipeline.model_kwargs))
    log.append(_record("predictor", fit, x_train.shape, start))

    for call in pipeline.postprocess:
        start = time.perf_counter()
        pred = _invoke("postprocess", call, pred, y_train, x_test)
        log.append(_record("postprocess", call, x_train.shape, start))

    return RunResult(pred, classes, features, log)


def run(
    pipeline: Pipeline,
    task: RelationalTask,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_factory: ModelFactory = default_model,
    max_rows: int = 10_000,
    max_columns: int = 500,
    cleaned: tuple[Database, list[StageRecord]] | None = None,
    drop_empty: float | None = None,
) -> RunResult:
    """Run a pipeline once: :func:`build_tables`, then :func:`fit_predict`.

    Args:
        pipeline: The pipeline.
        task: The task: database and label columns.
        train: Label rows to fit on (entity id, time, target).
        test: Label rows to predict (entity id, time; the target is ignored).
        model_factory: Builds the fixed predictor from ``task_type`` and ``pipeline.model_kwargs``.
        max_rows: Most rows a context view may hold.
        max_columns: Most feature columns the predictor may receive.
        cleaned: The result of :func:`clean_database` for this pipeline, if already known (a cache).
        drop_empty: See :func:`build_tables`.

    Returns:
        Predictions for ``test`` and the log.

    Raises:
        PipelineError: If a call fails or breaks a contract.
    """
    tables = build_tables(pipeline, task, train, test, max_rows, max_columns, cleaned, drop_empty=drop_empty)
    return fit_predict(tables, pipeline, task.task_type, model_factory)


@dataclass(frozen=True)
class Evaluation:
    """The validation score of one pipeline (higher is better), plus the run behind it.

    Attributes:
        score: The metric on the validation rows.
        metric: ``"roc_auc"`` for classification, ``"neg_mae"`` for regression (RelBench's metrics).
        run: The run, with its log and features.
    """

    score: float
    metric: str
    run: RunResult


def evaluate(
    pipeline: Pipeline,
    task: RelationalTask,
    model_factory: ModelFactory = default_model,
    max_rows: int = 10_000,
    max_columns: int = 500,
    cleaned: tuple[Database, list[StageRecord]] | None = None,
    drop_empty: float | None = None,
) -> Evaluation:
    """Fit on the task's train rows and score its validation rows.

    Args:
        pipeline: The pipeline.
        task: The task.
        model_factory: Builds the fixed predictor.
        max_rows: Most rows a context view may hold.
        max_columns: Most feature columns the predictor may receive.
        cleaned: The result of :func:`clean_database` for this pipeline, if already known (a cache).
        drop_empty: See :func:`build_tables`.

    Returns:
        The score and the run.
    """
    result = run(pipeline, task, task.train, task.val, model_factory, max_rows, max_columns, cleaned, drop_empty)
    return score(result, task.val[task.target_col].to_numpy())


def score(result: RunResult, y_true: np.ndarray) -> Evaluation:
    """Score a run's predictions: ROC AUC (one-vs-rest beyond two classes) or negative MAE, higher is better.

    Args:
        result: The run.
        y_true: The true targets of the predicted rows.

    Returns:
        The score and the run.
    """
    if result.classes is None:
        return Evaluation(-float(mean_absolute_error(y_true, result.pred)), "neg_mae", result)
    if len(result.classes) == 2:
        value = roc_auc_score(y_true == result.classes[1], result.pred[:, 1])
    else:
        value = roc_auc_score(y_true, result.pred, multi_class="ovr", labels=result.classes)
    return Evaluation(float(value), "roc_auc", result)


def _clean(call: Call, table: Table) -> Table:
    # The function does not see the keys and time column; they are put back on the rows it keeps.
    # A function marked `sees_keys` (duplicate detection) also sees the foreign keys and time
    # column, read only, so that rows of different entities or times are never "equal".
    sees_keys = getattr(REGISTRY["clean"][call.fn], "sees_keys", False)
    hidden = [table.pkey] if sees_keys and table.pkey else ([] if sees_keys else table.keys)
    shown = [k for k in table.keys if k not in hidden]
    out = _invoke("clean", call, table.df.drop(columns=hidden))
    if not out.index.isin(table.df.index).all():
        raise PipelineError(f"clean/{call} may drop rows but not add them.")
    if shown and not out[shown].equals(table.df.loc[out.index, shown]):
        raise PipelineError(f"clean/{call} changed key or time columns {shown}.")
    df = pd.concat([table.df.loc[out.index, hidden], out], axis=1)
    # the table's own column order, minus dropped columns, plus added ones (e.g. missing indicators)
    order = [c for c in table.df.columns if c in df.columns] + [c for c in df.columns if c not in table.df.columns]
    return Table(df[order], table.pkey, table.fkeys, table.time_col)


def _invoke(stage: str, call: Call, *args: Any) -> Any:
    try:
        return REGISTRY[stage][call.fn](*args, **call.params)
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(f"{stage}/{call} failed: {type(error).__name__}: {error}") from error


def _record(stage: str, call: Call, shape: tuple[int, ...], start: float) -> StageRecord:
    return StageRecord(stage, str(call), (int(shape[0]), int(shape[1])), time.perf_counter() - start)


def _model_frames(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fixed part of the predictor, not of the pipeline: dates become days since the
    # epoch, other non-numeric columns pandas categoricals (shared categories over
    # train and test) the model reads natively; beyond 255 levels, the gradient
    # boosting limit, they become codes.
    x_train, x_test = x_train.copy(), x_test.copy()
    for c in x_train.columns:
        if pd.api.types.is_datetime64_any_dtype(x_train[c]):
            for x in (x_train, x_test):
                x[c] = (x[c] - pd.Timestamp(0)).dt.total_seconds() / 86_400
            continue
        if pd.api.types.is_numeric_dtype(x_train[c]) and not pd.api.types.is_bool_dtype(x_train[c]):
            continue
        levels = pd.concat([x_train[c], x_test[c]]).dropna().astype(str).unique()
        dtype = pd.CategoricalDtype(sorted(levels))
        for x in (x_train, x_test):
            x[c] = x[c].astype(str).where(x[c].notna()).astype(dtype)
            if len(levels) > 255:
                x[c] = x[c].cat.codes.replace(-1, np.nan).astype(float)
    return x_train, x_test


def _predict(model: Any, x_test: pd.DataFrame, classes: np.ndarray | None) -> np.ndarray:
    if classes is None:
        return np.asarray(model.predict(x_test), dtype=float)
    # a view may miss a class; place each model column under its global class
    proba = np.zeros((len(x_test), len(classes)))
    proba[:, np.searchsorted(classes, model.classes_)] = model.predict_proba(x_test)
    return proba
