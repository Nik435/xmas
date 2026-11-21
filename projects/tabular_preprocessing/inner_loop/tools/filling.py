"""Tool set: fill the pipeline.

The executor's tools: read the function menu, then add, replace and remove
steps in the slots its group owns. Every edit is validated against the library
at once (function exists in that slot, parameters known and complete), so a
wrong step comes back to the model as an error instead of reaching the harness.
Whether the pipeline *helps* (scores, leakage) is the validation tool set's job.

Every change is also tried at once: the draft with the change is built on a small sample of
label rows (a dry run with the real harness). A change that would break the pipeline (an unknown
aggregation, a column another step removed, a misspelled merged column, removing a step a later
step needs) is refused with the reason, and the draft stays as it was. A change that runs is
answered with the feature columns it added or removed, by their exact names, so engineer steps
can refer to them; ``list_features`` lists all of the draft's columns.

One call to :func:`filling_tools` gives one group its tools, names prefixed with
the group (``clean_add_step``, ``merge_add_step``, ...), so both groups' tools
can sit in one system without clashing.
"""

# no `from __future__ import annotations`: the tools' type hints are set at build
# time (the stage parameter only lists the group's own slots) and must stay objects
from collections.abc import Sequence
from typing import Any, Literal

from tabular_preprocessing.inner_loop.library import STAGES
from tabular_preprocessing.inner_loop.pipeline import Pipeline, PipelineError
from tabular_preprocessing.inner_loop.tools.workspace import MERGE_SAMPLE_FUNCTIONS, Edit, Workspace
from xmas.tools import Tool, tool

# which slots each group fills, as in the project concept; "sample" joins merge
# when the "Merge fills sample()" factor is on
GROUP_SLOTS: dict[str, tuple[str, ...]] = {
    "clean": ("clean",),
    "merge": ("merge", "engineer"),
    "solo": ("clean", "merge", "engineer"),  # the single-agent baseline fills everything
}


def filling_tools(workspace: Workspace, group: str, slots: Sequence[str] | None = None) -> list[Tool]:
    """Build one group's tools for filling the pipeline.

    Args:
        workspace: The run's workspace; the tools edit its draft pipeline.
        group: The group, used as the tool-name prefix (``clean``, ``merge``, ...).
        slots: The slots this group may edit; defaults to :data:`GROUP_SLOTS`.

    Returns:
        ``<group>_list_functions``, ``<group>_show_pipeline``, ``<group>_list_features`` (groups that
        fill merge or engineer), ``<group>_add_step``, ``<group>_replace_step`` and ``<group>_remove_step``.

    Raises:
        ValueError: If the group has no default slots and none are given, or a slot is unknown.
    """
    owned = tuple(slots) if slots is not None else GROUP_SLOTS.get(group)
    if slots is None and group == "merge" and workspace.merge_fills_sample:
        owned = (*GROUP_SLOTS["merge"], "sample")
    if not owned:
        raise ValueError(f"No slots for group {group!r}; pass slots=, choosing from {list(STAGES)}.")
    unknown = [s for s in owned if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown slots {unknown}; the slots are {list(STAGES)}.")
    stage_type: Any = Literal[owned]  # the schema the model sees lists only these
    slot_list = ", ".join(owned)

    def list_functions(stage: str) -> str:
        """The functions a slot can hold, with their parameters and what they do.

        Args:
            stage: The slot.
        """
        menu = Pipeline.options(workspace.tool_description)[stage]
        if stage == "sample" and workspace.merge_fills_sample:
            menu = {name: doc for name, doc in menu.items() if name in MERGE_SAMPLE_FUNCTIONS}
        head = f"Functions for {stage}" + (" (every step also needs a 'table'):" if stage == "clean" else ":")
        return "\n".join([head, *(f"- {name}{doc}" for name, doc in menu.items())])

    def show_pipeline() -> str:
        """The whole draft pipeline, every slot with its numbered steps."""
        return _render(workspace.pipeline, owned)

    def list_features() -> str:
        """The feature columns the draft builds now, by the exact names engineer steps refer to."""
        columns = _features(workspace, workspace.pipeline)
        return f"{len(columns)} feature columns:\n" + "\n".join(f"- {c}" for c in columns)

    def add_step(stage: str, step: dict[str, Any], position: int | None = None) -> str:
        """Add a step to a slot. For sample, which holds one step, this replaces it.

        Args:
            stage: The slot.
            step: The step as an object: {"fn": <function>, ...its parameters}; clean steps also need "table".
            position: Where to insert it (0 = first); at the end if omitted.
        """
        steps = _steps(workspace.pipeline, stage)
        at = 0 if stage == "sample" else len(steps) if position is None else position
        if not 0 <= at <= len(steps):
            raise PipelineError(f"position must be between 0 and {len(steps)}.")
        if stage == "sample":
            _check_sample(workspace, step)
            steps = [step]
        else:
            steps.insert(at, step)
        before = _features(workspace, workspace.pipeline)
        added = _commit(workspace, stage, steps, at)
        workspace.edits.append(Edit(f"{group}_add_step", stage, "add", added))
        change = _change(before, _features(workspace, workspace.pipeline))
        return f"Added to {stage}: {added}\n{change}{_render_slot(workspace.pipeline, stage)}"

    def replace_step(stage: str, index: int, step: dict[str, Any]) -> str:
        """Replace one step of a slot by a new one.

        Args:
            stage: The slot.
            index: The step's number, as show_pipeline lists it.
            step: The new step as an object: {"fn": <function>, ...its parameters}; clean steps also need "table".
        """
        steps = _steps(workspace.pipeline, stage)
        _check_index(stage, index, steps)
        if stage == "sample":
            _check_sample(workspace, step)
        steps[index] = step
        before = _features(workspace, workspace.pipeline)
        new = _commit(workspace, stage, steps, index)
        workspace.edits.append(Edit(f"{group}_replace_step", stage, "replace", new))
        change = _change(before, _features(workspace, workspace.pipeline))
        return f"Replaced {stage}[{index}] with {new}\n{change}{_render_slot(workspace.pipeline, stage)}"

    def remove_step(stage: str, index: int) -> str:
        """Remove one step of a slot. Removing the sample step restores the default (full).

        Args:
            stage: The slot.
            index: The step's number, as show_pipeline lists it.
        """
        steps = _steps(workspace.pipeline, stage)
        _check_index(stage, index, steps)
        removed = str(_calls(workspace.pipeline, stage)[index])
        if stage == "sample":
            steps = [{"fn": "full"}]
        else:
            del steps[index]
        before = _features(workspace, workspace.pipeline)
        _commit(workspace, stage, steps, None)
        workspace.edits.append(Edit(f"{group}_remove_step", stage, "remove", removed))
        change = _change(before, _features(workspace, workspace.pipeline))
        return f"Removed {stage}[{index}]: {removed}\n{change}{_render_slot(workspace.pipeline, stage)}"

    fns = (list_functions, show_pipeline, add_step, replace_step, remove_step)
    if {"merge", "engineer"} & set(owned):  # merged column names matter to whoever writes engineer steps
        fns = (list_functions, show_pipeline, list_features, add_step, replace_step, remove_step)
    tools = []
    for fn in fns:
        if "stage" in fn.__annotations__:
            fn.__annotations__["stage"] = stage_type
            fn.__doc__ = (fn.__doc__ or "").replace("stage: The slot.", f"stage: The slot ({slot_list}).")
        tools.append(tool(name=f"{group}_{fn.__name__}")(fn))
    return tools


def _check_sample(workspace: Workspace, step: dict[str, Any]) -> None:
    if workspace.merge_fills_sample and step.get("fn") not in MERGE_SAMPLE_FUNCTIONS:
        raise PipelineError(
            f"sample may only hold {list(MERGE_SAMPLE_FUNCTIONS)} (selection and stratification, no oversampling)."
        )


def _steps(pipeline: Pipeline, stage: str) -> list[dict[str, Any]]:
    data = pipeline.to_dict()[stage]
    return [data] if stage == "sample" else list(data)


def _calls(pipeline: Pipeline, stage: str) -> list[Any]:
    return [pipeline.sample] if stage == "sample" else list(getattr(pipeline, stage))


def _check_index(stage: str, index: int, steps: list[dict[str, Any]]) -> None:
    if not 0 <= index < len(steps):
        raise PipelineError(f"{stage} has no step {index}; it has {len(steps)} step(s), numbered from 0.")


def _commit(workspace: Workspace, stage: str, steps: list[dict[str, Any]], at: int | None) -> str:
    """Rebuild the pipeline with ``steps`` in ``stage``, check that it runs, and return step ``at`` as text.

    The new draft is built on a small sample of label rows with the real harness (a dry run). If that
    fails, the change is refused with the reason and the draft stays as it was: a draft only ever holds
    steps that run, so a run never ends with a pipeline that breaks on the data.

    Raises:
        PipelineError: If the step is invalid, or the draft with it would not run on the data.
    """
    data = workspace.pipeline.to_dict()
    data[stage] = steps[0] if stage == "sample" else steps
    draft = Pipeline.from_dict(data)
    _dry_run(workspace, draft)
    workspace.pipeline = draft
    return "" if at is None else str(_calls(workspace.pipeline, stage)[at])


def _dry_run(workspace: Workspace, draft: Pipeline) -> None:
    """Build the draft's tables on a small sample of the tools' rows; raise if any step breaks."""
    _features(workspace, draft, refused=True)


def _features(workspace: Workspace, pipeline: Pipeline, refused: bool = False) -> tuple[str, ...]:
    """The pipeline's feature columns (a cached dry run); a failure is worded as a refused change."""
    try:
        return workspace.features(pipeline)
    except PipelineError as error:
        if not refused:
            raise
        raise PipelineError(
            f"Refused: with this change the pipeline would not run: {error} The draft is unchanged."
        ) from None


def _change(before: tuple[str, ...], after: tuple[str, ...]) -> str:
    """The feature columns a change added and removed, by name; empty if it changed none."""
    added = [c for c in after if c not in before]
    removed = [c for c in before if c not in after]
    lines = []
    if added:
        lines.append("New feature columns: " + ", ".join(added))
    if removed:
        lines.append("Removed feature columns: " + ", ".join(removed))
    return "".join(f"{line}\n" for line in lines)


def _render_slot(pipeline: Pipeline, stage: str) -> str:
    calls = _calls(pipeline, stage)
    lines = [f"{stage}:"] + [f"  [{i}] {c}" for i, c in enumerate(calls)]
    return "\n".join(lines if calls else [f"{stage}: (empty)"])


def _render(pipeline: Pipeline, owned: Sequence[str]) -> str:
    parts = []
    for stage in STAGES:
        mark = " (yours)" if stage in owned else " (read-only)"
        parts.append(_render_slot(pipeline, stage).replace(f"{stage}:", f"{stage}{mark}:", 1))
    parts.append(f"model_kwargs: {pipeline.model_kwargs or '-'}")
    return "\n".join(parts)
