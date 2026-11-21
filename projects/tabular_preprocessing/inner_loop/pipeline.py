r"""The pipeline: what the agents fill in, as plain data.

The slots of the project concept (modelled on TabFM-Auto), each given as calls
to fixed functions from the library::

    {
        "clean": [{"fn": "placeholder_as_missing", "table": "results", "values": ["\\N"]}],
        "merge": [
            {"fn": "entity_columns", "columns": ["nationality"]},
            {
                "fn": "aggregate",
                "table": "results",
                "fk_column": "driverId",
                "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
            },
        ],
        "engineer": [{"fn": "log_transform", "columns": ["results.count()"]}],
        "sample": {"fn": "full"},
        "postprocess": [],
        "model_kwargs": {},
    }

``clean``, ``merge``, ``engineer`` and ``postprocess`` are lists, applied in
order; ``sample`` is one call. A ``clean`` call names the table it cleans.
Group Clean fills ``clean``; group Merge fills ``merge`` and ``engineer`` (and
``sample``, if that factor is on). :meth:`Pipeline.identity` is the baseline.
"""

from __future__ import annotations

import functools
import inspect
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError

from tabular_preprocessing.inner_loop.library import EXAMPLES, REGISTRY, STAGES

_LISTS = ("clean", "merge", "engineer", "postprocess")
DESCRIPTION_LEVELS = ("brief", "detailed", "examples")


class PipelineError(ValueError):
    """An invalid pipeline, or a stage function that failed while running."""


@dataclass(frozen=True)
class Call:
    """One library function and the parameters chosen for it.

    Attributes:
        fn: Function name in its stage's library.
        params: Keyword-only parameters.
        table: The table a ``clean`` call works on; ``None`` for every other stage.
    """

    fn: str
    params: dict[str, Any] = field(default_factory=dict)
    table: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The call as ``{"fn": ..., ["table": ...,] **params}``."""
        head: dict[str, Any] = {"fn": self.fn} if self.table is None else {"fn": self.fn, "table": self.table}
        return {**head, **self.params}

    @classmethod
    def from_dict(cls, data: dict[str, Any], stage: str) -> Call:
        """Read a call from ``{"fn": ..., **params}``; for ``clean``, ``table`` is the call's own field.

        Two forms models often write instead are read as meant: ``{"function": ..., **params}`` and the
        function name as the only key, ``{"aggregate": {...params}}``.
        """
        if isinstance(data, dict) and "fn" not in data:
            if "function" in data:
                data = {"fn": data["function"], **{k: v for k, v in data.items() if k != "function"}}
            elif (
                len(data) == 1
                and next(iter(data)) in REGISTRY.get(stage, {})
                and isinstance(next(iter(data.values())), dict)
            ):
                ((name, params_),) = data.items()
                data = {"fn": name, **params_}
        if not isinstance(data, dict) or "fn" not in data:
            raise PipelineError(
                f"A {stage} step must be one object: 'fn' names the function and its parameters sit next to it, "
                f"e.g. {json.dumps(_example(stage))}. Got {data!r}."
            )
        params = {k: v for k, v in data.items() if k != "fn"}
        table = params.pop("table", None) if stage == "clean" else None
        return cls(data["fn"], params, table)

    def __str__(self) -> str:
        """The call as ``fn[table](param=value, ...)``."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.fn}{f'[{self.table}]' if self.table else ''}({args})"


@dataclass(frozen=True)
class Pipeline:
    """The slots and the predictor settings; validated on construction.

    Attributes:
        clean: Per-table cleaning calls, in order.
        merge: Calls that bring database features into the label rows, in order.
        engineer: Feature calls on the merged frames, in order.
        sample: The one call that picks the context views.
        postprocess: Prediction calls, in order.
        model_kwargs: Settings for the fixed predictor.
    """

    clean: list[Call] = field(default_factory=list)
    merge: list[Call] = field(default_factory=list)
    engineer: list[Call] = field(default_factory=list)
    sample: Call = field(default_factory=lambda: Call("full"))
    postprocess: list[Call] = field(default_factory=list)
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check every call against its stage's library, and convert its parameters to their declared types.

        A model often writes a number as text (``"365"``); it is converted here, so a step that is
        accepted also runs. A value that cannot be converted is refused with the reason.
        """
        for stage in _LISTS:
            calls = getattr(self, stage)
            calls[:] = [_check(stage, call) for call in calls]
        object.__setattr__(self, "sample", _check("sample", self.sample))

    @classmethod
    def identity(cls) -> Pipeline:
        """The pipeline that adds nothing: the baseline every other pipeline is compared with."""
        return cls()

    @staticmethod
    def options(detail: str = "detailed") -> dict[str, dict[str, str]]:
        """Per stage, each function's parameters and description: the menu an agent reads.

        Args:
            detail: How much the menu says, the "tool/slot description" factor: ``"brief"`` (parameters
                only), ``"detailed"`` (parameters and description) or ``"examples"`` (and an example call).

        Raises:
            ValueError: On an unknown level.
        """
        if detail not in DESCRIPTION_LEVELS:
            raise ValueError(f"Unknown description level {detail!r}; choose from {list(DESCRIPTION_LEVELS)}.")
        menu: dict[str, dict[str, str]] = {}
        for stage, fns in REGISTRY.items():
            menu[stage] = {}
            for name, fn in fns.items():
                text = _params_doc(stage, fn)
                if detail != "brief":
                    text += f" -- {inspect.getdoc(fn) or ''}"
                if detail == "examples":
                    text += "\n  example: " + json.dumps({"fn": name, **EXAMPLES[name]})
                menu[stage][name] = text
        return menu

    def calls(self) -> list[tuple[str, Call]]:
        """Every call with its stage, in execution order."""
        return [
            *(("clean", c) for c in self.clean),
            *(("merge", c) for c in self.merge),
            *(("engineer", c) for c in self.engineer),
            ("sample", self.sample),
            *(("postprocess", c) for c in self.postprocess),
        ]

    def to_dict(self) -> dict[str, Any]:
        """The pipeline as plain data (JSON-ready if the parameters are)."""
        data: dict[str, Any] = {stage: [c.to_dict() for c in getattr(self, stage)] for stage in _LISTS}
        return {**data, "sample": self.sample.to_dict(), "model_kwargs": dict(self.model_kwargs)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """Read a pipeline from plain data; missing slots are left empty.

        Raises:
            PipelineError: On unknown keys, functions or parameters.
        """
        unknown = set(data) - {*STAGES, "model_kwargs"}
        if unknown:
            raise PipelineError(f"Unknown pipeline keys {sorted(unknown)}; the slots are {[*STAGES, 'model_kwargs']}.")
        for stage in _LISTS:
            if not isinstance(data.get(stage, []), list):
                raise PipelineError(f"{stage!r} must be a list of calls.")
        lists = {stage: [Call.from_dict(c, stage) for c in data.get(stage, [])] for stage in _LISTS}
        return cls(
            **lists,
            sample=Call.from_dict(data.get("sample", {"fn": "full"}), "sample"),
            model_kwargs=dict(data.get("model_kwargs", {})),
        )

    def describe(self) -> str:
        """The pipeline as text, one line per slot."""
        lines = [f"{stage:12s} {' -> '.join(str(c) for c in getattr(self, stage)) or '-'}" for stage in _LISTS]
        lines.insert(3, f"{'sample':12s} {self.sample}")
        lines.append(f"{'model_kwargs':12s} {self.model_kwargs or '-'}")
        return "\n".join(lines)


def _example(stage: str) -> dict[str, Any]:
    """A correct step for a slot, for error messages: its first function with an example call."""
    for name in REGISTRY.get(stage, {}):
        if name in EXAMPLES:
            return {"fn": name, **EXAMPLES[name]}
    return {"fn": "..."}


def _check(stage: str, call: Call) -> Call:
    """The call with its parameters converted to their declared types.

    Raises:
        PipelineError: On an unknown function, missing or unknown parameters, or a value of the wrong type.
    """
    fns = REGISTRY[stage]
    if call.fn not in fns:
        raise PipelineError(f"{stage} has no function {call.fn!r}; choose from {list(fns)}.")
    if stage == "clean" and not call.table:
        raise PipelineError(f"clean/{call.fn} needs a 'table'.")
    keyword = {name: p for name, p in inspect.signature(fns[call.fn]).parameters.items() if p.kind is p.KEYWORD_ONLY}
    unknown = set(call.params) - set(keyword)
    if unknown:
        raise PipelineError(f"{stage}/{call.fn} takes no parameters {sorted(unknown)}; it takes {list(keyword)}.")
    missing = [name for name, p in keyword.items() if p.default is p.empty and name not in call.params]
    if missing:
        raise PipelineError(f"{stage}/{call.fn} needs parameters {missing}.")
    params = {}
    for name, value in call.params.items():
        try:
            params[name] = _adapter(fns[call.fn], name).validate_python(value)
        except ValidationError as error:
            problem = "; ".join(e["msg"] for e in error.errors())
            raise PipelineError(f"{stage}/{call.fn}: parameter {name}={value!r}: {problem}.") from None
    return Call(call.fn, params, call.table)


@functools.cache
def _adapter(fn: Any, name: str) -> TypeAdapter[Any]:
    """A validator for one parameter's declared type (lax: "365" becomes 365 for an int)."""
    return TypeAdapter(inspect.signature(fn, eval_str=True).parameters[name].annotation)


def _params_doc(stage: str, fn: Any) -> str:
    # eval_str: the library uses postponed annotations, which would print as quoted strings
    params = [str(p) for p in inspect.signature(fn, eval_str=True).parameters.values() if p.kind is p.KEYWORD_ONLY]
    if stage == "clean":
        params.insert(0, "table: str")
    return "(" + ", ".join(params) + ")"
