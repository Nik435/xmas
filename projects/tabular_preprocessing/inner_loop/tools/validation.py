"""Tool set: validate the pipeline (both validators; an executor without validator inherits it).

- ``check_pipeline``: the contracts, without fitting the predictor: does the pipeline
  run, how many rows does cleaning remove, are train and validation treated alike,
  is the feature count within budget, which features are empty or constant, and, as
  errors, which features are empty for nearly all rows the run is finally scored on.
- ``score_pipeline``: the validation score of the draft, next to the identity
  pipeline and to the draft without this group's slots (what the group added).
- ``score_steps``: leave-one-out per step of the group's slots (what each step adds).
- ``feature_report``: per feature its missing share and its signal on the train
  rows, with flags for leakage risks (suspiciously strong, static columns, tables
  without time column) and for drift between train and validation.

Scores come from the workspace: cached per pipeline and counted against its budget.
"""

# no `from __future__ import annotations`: xmas builds the tool schemas from the hints
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tabular_preprocessing.inner_loop.data import RelationalTask
from tabular_preprocessing.inner_loop.harness import Tables, build_tables
from tabular_preprocessing.inner_loop.library import STAGES
from tabular_preprocessing.inner_loop.pipeline import Pipeline, PipelineError
from tabular_preprocessing.inner_loop.tools.filling import GROUP_SLOTS
from tabular_preprocessing.inner_loop.tools.text import bullets, fmt, pct
from tabular_preprocessing.inner_loop.tools.workspace import Scored, Workspace
from xmas.tools import Tool, tool

_LISTS = ("clean", "merge", "engineer", "postprocess")
PREDICTION_EMPTY = 0.95  # a feature at least this empty on the rows to predict cannot help there


def validation_tools(workspace: Workspace, group: str) -> list[Tool]:
    """Build one group's validation tools.

    Args:
        workspace: The run's workspace.
        group: The tool-name prefix; its slots (:data:`GROUP_SLOTS`) are what ``score_steps``
            and the "without your slots" comparison look at. A group without slots (the
            orchestrator) looks at every slot.

    Returns:
        ``<group>_check_pipeline``, ``<group>_score_pipeline``, ``<group>_score_steps`` and
        ``<group>_feature_report``.
    """
    task = workspace.tool_task  # the training rows, split by time: the validation rows stay unseen
    owned = [s for s in GROUP_SLOTS.get(group, STAGES) if s in _LISTS]

    def tables() -> Tables:
        return build_tables(
            workspace.pipeline,
            task,
            task.train,
            task.val,
            workspace.max_rows,
            workspace.max_columns,
            workspace.cleaned(),
        )

    def check_pipeline() -> str:
        """Check the draft's contracts without fitting the predictor (uses no evaluation of the budget).

        Does it run, how many rows cleaning removes, train and validation treated alike, the feature count
        within budget, and features that are empty, constant or much more missing in validation.
        """
        try:
            built = tables()
        except PipelineError as error:
            return f"FAILS: {error}\nFix or remove that step; nothing after it can run."
        out = ["Runs: every step applies to the data; one row per label row in train and validation."]
        sizes = {name: len(t.df) for name, t in task.db.items()}
        for record in built.log:
            if record.stage == "clean":
                table = record.call.split("[", 1)[1].split("]", 1)[0]
                removed = 1 - record.shape[0] / sizes[table] if sizes[table] else 0.0
                note = f" ({pct(removed)} of its rows removed so far)" if removed else ""
                out.append(f"clean {record.call}: {table} has {record.shape[0]} rows{note}")
        n = built.x_train.shape[1]
        out.append(f"Features: {n} of the budget of {workspace.max_columns}.")
        problems = []
        for c in built.x_train.columns:
            tr, va = built.x_train[c], built.x_test[c]
            if tr.isna().all():
                problems.append(f"{c}: empty in train")
            elif tr.nunique() <= 1:
                problems.append(f"{c}: constant in train")
            elif va.isna().mean() - tr.isna().mean() > 0.3:
                problems.append(
                    f"{c}: missing in {pct(va.isna().mean())} of validation vs {pct(tr.isna().mean())} of train rows"
                )
        out.append("Feature problems:\n" + bullets(problems))
        errors = _empty_where_predicted(workspace)
        if errors:
            out.insert(
                0,
                "ERRORS, fix before you finish: these features are empty for nearly all rows the run is "
                "finally scored on (those rows lie after the database's last rows), so they cannot help "
                "there; the tool scores do not show this. Remove or replace their steps:\n" + bullets(errors),
            )
        return "\n".join(out)

    def score_pipeline() -> str:
        """The draft's validation score, next to the identity pipeline and to the draft without your slots.

        Each pipeline not scored before uses one evaluation of the run's budget.
        """
        draft = workspace.evaluate(workspace.pipeline)
        base = workspace.baseline()
        out = [_line("draft", draft), _line("identity (no preprocessing)", base)]
        if draft.score is not None and base.score is not None:
            out.append(f"draft vs identity: {draft.score - base.score:+.4f}")
        if any(getattr(workspace.pipeline, s) for s in owned):
            data = workspace.pipeline.to_dict()
            for s in owned:
                data[s] = []
            without = _evaluate_or_fail(workspace, data)
            out.append(_line(f"draft without your slots ({', '.join(owned)})", without))
            if draft.score is not None and without.score is not None:
                out.append(f"your slots add: {draft.score - without.score:+.4f}")
        out.append(_budget(workspace))
        return "\n".join(out)

    def score_steps() -> str:
        """Leave-one-out for every step of your slots: the score without that step, and what the step adds.

        A step whose removal breaks a later step (e.g. an engineer step that uses its column) is reported
        as needed. Each removal uses one evaluation of the budget; stops early when the budget runs out.
        """
        draft = workspace.evaluate(workspace.pipeline)
        if draft.score is None:
            return f"The draft fails, so its steps cannot be compared: {draft.error}"
        out = [_line("draft", draft)]
        lines = []
        for stage in owned:
            for i, call in enumerate(getattr(workspace.pipeline, stage)):
                data = workspace.pipeline.to_dict()
                data[stage] = [c for j, c in enumerate(data[stage]) if j != i]
                if workspace.evaluations >= workspace.max_evaluations:
                    lines.append(f"{stage}[{i}] {call}: not scored, budget used up")
                    continue
                without = _evaluate_or_fail(workspace, data)
                if without.score is None:
                    lines.append(f"{stage}[{i}] {call}: needed by later steps ({without.error})")
                else:
                    delta = draft.score - without.score
                    verdict = "helps" if delta > 0.001 else "hurts" if delta < -0.001 else "no effect"
                    lines.append(f"{stage}[{i}] {call}: adds {delta:+.4f} ({verdict})")
        out.append("Per step (positive = the step helps):\n" + bullets(lines))
        out.append(_budget(workspace))
        return "\n".join(out)

    def feature_report() -> str:
        """Per feature of the draft: missing share, signal on the train rows, and flags for leakage and drift.

        Signal is the ROC AUC of the feature alone (binary target; 0.5 = none, far from 0.5 = strong) or the
        absolute Spearman correlation (regression). A feature with near-perfect signal is suspicious.
        """
        try:
            built = tables()
        except PipelineError as error:
            return f"The draft fails: {error}"
        static = _static_prefixes(workspace.pipeline, task)
        untimed = {name for name, t in task.db.items() if t.time_col is None}
        try:
            predicted = workspace.prediction_missing()
        except PipelineError:
            predicted = {}
        lines = []
        for c in built.x_train.columns:
            tr, va = built.x_train[c], built.x_test[c]
            signal = _signal(tr, built.y_train, task.task_type)
            line = (
                f"{c}: missing {pct(float(tr.isna().mean()))} (rows to predict: {pct(predicted.get(str(c), 0.0))}), "
                f"signal {fmt(signal) if signal is not None else 'n/a'}"
            )
            flags = []
            if predicted.get(str(c), 0.0) >= PREDICTION_EMPTY:
                flags.append("EMPTY on the rows to predict: cannot help there, remove it")
            if signal is not None and (signal > 0.95 or (task.task_type == "classification" and signal < 0.05)):
                flags.append("near-perfect signal: check for target leakage")
            if c.split(".", 1)[0] in static and "(" not in c:
                flags.append("static column: is its value known at prediction time?")
            if "(" in c and c.split(".", 1)[0] in untimed:
                flags.append("aggregate over a table without time column: includes rows after the label time")
            if va.isna().mean() - tr.isna().mean() > 0.3:
                flags.append("much more missing in validation (drift)")
            lines.append(line + (" | " + "; ".join(flags) if flags else ""))
        head = f"{built.x_train.shape[1]} features, {len(built.x_train)} train rows."
        return head + "\n" + bullets(lines, limit=80)

    fns = (check_pipeline, score_pipeline, score_steps, feature_report)
    return [tool(name=f"{group}_{fn.__name__}")(fn) for fn in fns]


def _evaluate_or_fail(workspace: Workspace, data: dict[str, object]) -> Scored:
    try:
        pipeline = Pipeline.from_dict(data)  # type: ignore[arg-type]
    except PipelineError as error:
        return Scored(None, None, error=str(error))
    return workspace.evaluate(pipeline)


def _line(label: str, scored: Scored) -> str:
    if scored.score is None:
        return f"{label}: fails ({scored.error})"
    return f"{label}: {scored.metric} {scored.score:.4f} ({len(scored.features)} features)"


def _budget(workspace: Workspace) -> str:
    return f"Evaluations used: {workspace.evaluations} of {workspace.max_evaluations}."


def _static_prefixes(pipeline: Pipeline, task: RelationalTask) -> set[str]:
    """Table prefixes of columns copied as they are (entity_columns, lookup), not aggregated over time."""
    prefixes = set()
    fkeys = task.db[task.entity_table].fkeys or {}
    for call in pipeline.merge:
        if call.fn == "entity_columns":
            prefixes.add(task.entity_table)
        elif call.fn == "lookup":
            prefixes.add(fkeys.get(call.params.get("fk_column", ""), ""))
    return prefixes


def _signal(x: pd.Series, y: pd.Series, task_type: str) -> float | None:
    values = x
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        if pd.api.types.is_datetime64_any_dtype(values):
            values = (values - pd.Timestamp(0)).dt.days
        else:
            # categories: rank them by their target mean on the train rows (an upper bound on their signal)
            if task_type != "classification" or y.nunique() != 2:
                return None
            positive = (y == np.sort(y.unique())[1]).astype(float)
            values = values.astype(str).map(positive.groupby(values.astype(str)).mean())
    values = values.astype(float)
    filled = values.fillna(values.median() if values.notna().any() else 0.0)
    if filled.nunique() <= 1:
        return None
    if task_type == "classification":
        if y.nunique() != 2:
            return None
        return float(roc_auc_score(y == np.sort(y.unique())[1], filled))
    return float(abs(filled.corr(y.astype(float), method="spearman")))


def _empty_where_predicted(workspace: Workspace) -> list[str]:
    """The draft's features that are at least :data:`PREDICTION_EMPTY` empty on the rows to predict."""
    try:
        missing = workspace.prediction_missing()
    except PipelineError as error:
        return [f"the pipeline does not run on the rows to predict: {error}"]
    return [
        f"{c}: missing for {pct(share)} of the rows to predict"
        for c, share in missing.items()
        if share >= PREDICTION_EMPTY
    ]
