"""Study configuration: the search space of the outer loop and how to search it, read from YAML.

A configuration names

- the tasks every candidate is evaluated on;
- the **search space**: the architecture hyperparameters (section 1 of the concept: per
  group ``size``, ``role``, ``fb_E``, ``fb_V``, or system ``codes``) and the further factors
  (section 2: orchestrator and its routing, temperature, model types, tool/slot description,
  autonomy, order, Merge fills ``sample()``). A single value is fixed; a list gives the
  choices; ``{low: .., high: ..}`` gives a range (temperature only, for samplers);
- the **optimizer** that searches the space (``grid``, ``random``; see
  :mod:`tabular_preprocessing.outer_loop.optimizers`) and its number of trials;
- the budgets per autonomy level and the harness settings.

A *candidate* is one point of the space: one architecture (an xmas system, built by the
inner loop's :class:`Architecture`) with one factor combination. Inactive hyperparameters
follow the concept's "active when": ``role`` only at size 2, ``fb_E`` only with a planner,
``fb_V`` only with a validator, no back edges and ``orchestrator_routing`` only with the
orchestrator on. See ``configs/stage1.yaml`` and ``configs/random.yaml``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from tabular_preprocessing.inner_loop.architecture import Architecture, GroupArchitecture
from tabular_preprocessing.inner_loop.harness import ModelFactory, default_model
from tabular_preprocessing.inner_loop.orchestrated import ROUTING
from tabular_preprocessing.inner_loop.pipeline import DESCRIPTION_LEVELS

ORDERS: dict[str, tuple[str, str]] = {"clean_first": ("clean", "merge"), "merge_first": ("merge", "clean")}
AUTONOMY_LEVELS: tuple[str, ...] = ("fixed", "bounded", "free")
MODEL_ROLES: tuple[str, ...] = ("default", "P", "E", "V", "C")
# mean_relative_improvement: per task (score - identity) / |identity|, so ROC AUC and MAE tasks weigh alike;
# mean_improvement: per task score - identity, in the metric's own unit
OBJECTIVES: tuple[str, ...] = ("mean_relative_improvement", "mean_improvement")
FACTOR_NAMES: tuple[str, ...] = (
    "orchestrator",
    "orchestrator_routing",
    "temperature",
    "models",
    "tool_description",
    "autonomy",
    "order",
    "merge_fills_sample",
)

# the fixed predictor, chosen by name; add TabPFN / LightGBM / TabFM here once decided
PREDICTORS: dict[str, ModelFactory] = {"hist_gradient_boosting": default_model}


class ConfigError(ValueError):
    """A configuration that cannot be run, with the reason."""


@dataclass(frozen=True)
class Budget:
    """The limits one autonomy level grants.

    Attributes:
        max_tool_rounds: Tool-call turns per agent visit.
        max_loops: How often one agent may run, which bounds every back-edge loop.
        max_evaluations: How many distinct pipelines the agents' tools may score per run.
        max_steps: With an orchestrator: most agent turns it may delegate per run.
    """

    max_tool_rounds: int
    max_loops: int
    max_evaluations: int
    max_steps: int


DEFAULT_BUDGETS: dict[str, Budget] = {
    "fixed": Budget(max_tool_rounds=10, max_loops=1, max_evaluations=10, max_steps=8),
    "bounded": Budget(max_tool_rounds=20, max_loops=3, max_evaluations=30, max_steps=12),
    "free": Budget(max_tool_rounds=50, max_loops=6, max_evaluations=100, max_steps=24),
}


@dataclass(frozen=True)
class Factors:
    """One combination of the further hyperparameters (section 2 of the concept).

    Attributes:
        orchestrator: Whether an orchestrator leads the system.
        orchestrator_routing: With an orchestrator: ``"ordered"`` (the second group starts only after the
            first group's executor has worked) or ``"free"``.
        temperature: Sampling temperature of every agent; ``None`` sends none (e.g. reasoning models).
        models: Model per role: ``default`` and optionally ``P``, ``E``, ``V``, ``C``.
        tool_description: ``"brief"``, ``"detailed"`` or ``"examples"``.
        autonomy: ``"fixed"``, ``"bounded"`` or ``"free"``.
        order: ``"clean_first"`` or ``"merge_first"``.
        merge_fills_sample: Whether group Merge also fills the sample slot.
    """

    orchestrator: bool = False
    orchestrator_routing: str = "ordered"
    temperature: float | None = 0.25
    models: tuple[tuple[str, str], ...] = (("default", ""),)
    tool_description: str = "detailed"
    autonomy: str = "bounded"
    order: str = "clean_first"
    merge_fills_sample: bool = False

    def model(self, role: str) -> str:
        """The model of a role (``P``, ``E``, ``V``, ``C``), falling back to the default."""
        models = dict(self.models)
        return models.get(role) or models["default"]

    def as_dict(self) -> dict[str, Any]:
        """Plain data, with ``models`` as a mapping."""
        return {**asdict(self), "models": dict(self.models)}

    @property
    def key(self) -> str:
        """A short, stable id of this combination."""
        return hashlib.sha1(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:8]


@dataclass(frozen=True)
class Candidate:
    """One point of the search space: an architecture with a factor combination.

    The architecture already carries the factors that shape it (group order, orchestrator).
    """

    architecture: Architecture
    factors: Factors

    @property
    def key(self) -> str:
        """A stable id, e.g. ``clean=PE|merge=E|6bdc5fe3``."""
        return f"{self.architecture.code}|{self.factors.key}"

    def as_dict(self) -> dict[str, Any]:
        """Plain data: the architecture code, the per-group hyperparameters, the factors."""
        hp = {f"{g}.{k}": v for g in ("clean", "merge") for k, v in getattr(self.architecture, g).hp.items()}
        return {"architecture": self.architecture.code, **hp, **self.factors.as_dict()}


@dataclass(frozen=True)
class RunSpec:
    """One run: a candidate on a task, one repeat."""

    task: str
    architecture: Architecture
    factors: Factors
    repeat: int

    @property
    def run_id(self) -> str:
        """A stable id: the same run in a later invocation has the same id, so it is not run again."""
        return f"{self.task}|{self.architecture.code}|{self.factors.key}|r{self.repeat}"


class SearchSpace:
    """The hyperparameters and their choices; can list every candidate or draw one at random."""

    def __init__(self, architecture: dict[str, Any], factors: dict[str, Any]) -> None:
        """Read and check the ``architecture`` and ``factors`` sections.

        Raises:
            ConfigError: On an unknown key or a value that cannot be run.
        """
        _known(architecture, {"clean", "merge", "codes"}, "architecture.")
        _known(factors, set(FACTOR_NAMES), "factors.")
        self.codes: list[Architecture] | None = None
        self.groups: dict[str, dict[str, list[Any]]] = {}
        if "codes" in architecture:
            if "clean" in architecture or "merge" in architecture:
                raise ConfigError("architecture: give either codes or per-group hyperparameters, not both.")
            try:
                self.codes = [Architecture.from_code(code) for code in _list(architecture["codes"])]
            except ValueError as error:
                raise ConfigError(f"architecture.codes: {error}") from error
        else:
            for group in ("clean", "merge"):
                hp = architecture.get(group) or {"size": 1}
                _known(hp, {"size", "role", "fb_E", "fb_V"}, f"architecture.{group}.")
                self.groups[group] = {
                    "size": [int(s) for s in _list(hp.get("size", 1))],
                    "role": [str(r) for r in _list(hp.get("role", []))],
                    "fb_E": [bool(v) for v in _list(hp.get("fb_E", 0))],
                    "fb_V": [bool(v) for v in _list(hp.get("fb_V", 0))],
                }
                try:
                    GroupArchitecture.grid(*self.groups[group].values())
                except ValueError as error:
                    raise ConfigError(f"architecture.{group}: {error}") from error

        defaults = Factors()
        self.levels: dict[str, list[Any]] = {}
        self.ranges: dict[str, tuple[float, float]] = {}
        for key in FACTOR_NAMES:
            raw = factors.get(key, getattr(defaults, key) if key != "models" else {"default": ""})
            if isinstance(raw, dict) and key != "models":
                self.ranges[key] = _range(key, raw)
                self.levels[key] = [self.ranges[key][0]]  # a stand-in for the checks below
            else:
                self.levels[key] = raw if isinstance(raw, list) else [raw]
        for key, levels in self.levels.items():  # every level on its own must be runnable
            for level in levels:
                _check_factors({**{k: v[0] for k, v in self.levels.items()}, key: level})

    @property
    def is_finite(self) -> bool:
        """Whether every candidate can be listed (no ranges)."""
        return not self.ranges

    def architectures(self) -> list[Architecture]:
        """Every architecture of the space, before the factors shape it."""
        if self.codes is not None:
            return list(self.codes)
        clean, merge = (GroupArchitecture.grid(*self.groups[g].values()) for g in ("clean", "merge"))
        return [Architecture(c, m) for c, m in itertools.product(clean, merge)]

    def factor_grid(self) -> list[Factors]:
        """Every distinct factor combination (inactive ones collapse).

        Raises:
            ConfigError: If a factor is a range, which has no finite grid.
        """
        if self.ranges:
            raise ConfigError(f"factors {sorted(self.ranges)} are ranges: use a sampling optimizer, not grid.")
        out: list[Factors] = []
        for combo in itertools.product(*self.levels.values()):
            factors = _collapse(_check_factors(dict(zip(self.levels, combo, strict=True))))
            if factors not in out:
                out.append(factors)
        return out

    def grid(self) -> list[Candidate]:
        """Every distinct candidate, in a stable order."""
        out: list[Candidate] = []
        for factors in self.factor_grid():
            for arch in self.architectures():
                candidate = Candidate(_shape(arch, factors), factors)
                if candidate not in out:
                    out.append(candidate)
        return out

    def sample(self, rng: random.Random) -> Candidate:
        """One candidate drawn at random: each active hyperparameter uniformly from its choices or range."""
        if self.codes is not None:
            arch = rng.choice(self.codes)
        else:
            groups = {}
            for group, hp in self.groups.items():
                size = rng.choice(hp["size"])
                role = rng.choice(hp["role"]) if size == 2 else None
                variant = GroupArchitecture.from_hp(size, role)
                fb_e = rng.choice(hp["fb_E"]) if "P" in variant.roles else False
                fb_v = rng.choice(hp["fb_V"]) if "V" in variant.roles else False
                groups[group] = GroupArchitecture.from_hp(size, role, fb_e, fb_v)
            arch = Architecture(groups["clean"], groups["merge"])
        values = {}
        for key, levels in self.levels.items():
            if key in self.ranges:
                low, high = self.ranges[key]
                values[key] = round(rng.uniform(low, high), 3)
            else:
                values[key] = rng.choice(levels)
        factors = _collapse(_check_factors(values))
        return Candidate(_shape(arch, factors), factors)

    def size(self) -> int | None:
        """How many distinct candidates the space holds, or ``None`` if it has ranges."""
        return len(self.grid()) if self.is_finite else None


@dataclass(frozen=True)
class OptimizerSettings:
    """How the outer loop searches the space.

    Attributes:
        name: The optimizer, a key of :data:`tabular_preprocessing.outer_loop.optimizers.OPTIMIZERS`.
        n_trials: How many candidates to evaluate; ``None`` for all (grid only).
        seed: Seed of the optimizer's randomness.
        params: Further settings for the optimizer.
    """

    name: str = "grid"
    n_trials: int | None = None
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """A parsed and checked study configuration.

    Attributes:
        name: The study's name.
        output_dir: Where results go (``trials.jsonl``, ``results.jsonl`` and one detail file per run).
        tasks: RelBench tasks as ``"<dataset>/<task>"``; every candidate runs on all of them.
        repeats: How often each candidate runs on each task.
        space: The search space.
        optimizer: How to search it.
        objective: What a trial's value is: ``"mean_relative_improvement"`` (default) or ``"mean_improvement"``,
            over the identity pipeline, averaged over repeats per task and then over tasks.
        budgets: The limits per autonomy level.
        predictor: The fixed predictor's name in :data:`PREDICTORS`.
        max_rows: Most rows a context view may hold.
        max_columns: Most feature columns the predictor may receive.
        baseline: Whether to score the identity pipeline in every run.
        max_train_rows: Keep only the most recent training label rows (for tasks with millions of them).
        retries: How often a run that failed for infrastructure reasons (API errors) is retried.
        parallel: How many runs of a trial run at the same time.
        max_tokens: Stop starting new trials once the study's runs used this many tokens.
        max_hours: Stop starting new trials once the study's runs took this many hours.
        source: The configuration as read, for the record.
    """

    name: str
    output_dir: Path
    tasks: list[str]
    repeats: int
    space: SearchSpace
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    objective: str = "mean_relative_improvement"
    budgets: dict[str, Budget] = field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    predictor: str = "hist_gradient_boosting"
    max_rows: int = 10_000
    max_columns: int = 500
    baseline: bool = True
    max_train_rows: int | None = None
    retries: int = 2
    parallel: int = 1
    max_tokens: int | None = None
    max_hours: float | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Read and check a configuration file.

        Raises:
            ConfigError: If the file cannot be run, with the reason.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Check a configuration given as plain data.

        Raises:
            ConfigError: If it cannot be run, with the reason.
        """
        allowed = {"name", "output_dir", "tasks", "repeats", "architecture", "factors", "optimizer", "objective"}
        _known(data, allowed | {"budgets", "run"}, "")
        name = str(data.get("name") or "study")
        tasks = _list(data.get("tasks"))
        if not tasks or not all(isinstance(t, str) and t.count("/") == 1 for t in tasks):
            raise ConfigError("tasks must list RelBench tasks as '<dataset>/<task>', e.g. 'rel-f1/driver-dnf'.")
        repeats = int(data.get("repeats", 1))
        if repeats < 1:
            raise ConfigError("repeats must be at least 1.")

        budgets = dict(DEFAULT_BUDGETS)
        for level, values in (data.get("budgets") or {}).items():
            if level not in AUTONOMY_LEVELS:
                raise ConfigError(f"budgets.{level}: unknown autonomy level; choose from {list(AUTONOMY_LEVELS)}.")
            _known(values, {"max_tool_rounds", "max_loops", "max_evaluations", "max_steps"}, f"budgets.{level}.")
            budgets[level] = replace(budgets[level], **{k: int(v) for k, v in values.items()})

        run = data.get("run") or {}
        _known(
            run,
            {"predictor", "max_rows", "max_columns", "baseline", "max_train_rows", "retries", "parallel"}
            | {"max_tokens", "max_hours"},
            "run.",
        )
        predictor = str(run.get("predictor", "hist_gradient_boosting"))
        if predictor not in PREDICTORS:
            raise ConfigError(f"run.predictor {predictor!r} is unknown; available: {sorted(PREDICTORS)}.")

        objective = str(data.get("objective", "mean_relative_improvement"))
        if objective not in OBJECTIVES:
            raise ConfigError(f"objective {objective!r} is unknown; available: {list(OBJECTIVES)}.")
        baseline = bool(run.get("baseline", True))
        if not baseline:
            raise ConfigError(f"objective {objective} needs run.baseline: true (the identity score).")
        parallel, retries = int(run.get("parallel", 1)), int(run.get("retries", 2))
        if parallel < 1 or retries < 0:
            raise ConfigError("run.parallel must be at least 1 and run.retries at least 0.")

        raw = data.get("optimizer") or {}
        if isinstance(raw, str):
            raw = {"name": raw}
        _known(raw, {"name", "n_trials", "seed", "params"}, "optimizer.")
        optimizer = OptimizerSettings(
            name=str(raw.get("name", "grid")),
            n_trials=None if raw.get("n_trials") is None else int(raw["n_trials"]),
            seed=int(raw.get("seed", 0)),
            params=dict(raw.get("params") or {}),
        )
        space = SearchSpace(data.get("architecture") or {}, data.get("factors") or {})
        _check_optimizer(optimizer, space)

        return cls(
            name=name,
            output_dir=Path(data.get("output_dir") or f"runs/{name}"),
            tasks=tasks,
            repeats=repeats,
            space=space,
            optimizer=optimizer,
            objective=objective,
            budgets=budgets,
            predictor=predictor,
            max_rows=int(run.get("max_rows", 10_000)),
            max_columns=int(run.get("max_columns", 500)),
            baseline=baseline,
            max_train_rows=_optional(run.get("max_train_rows"), int),
            retries=retries,
            parallel=parallel,
            max_tokens=_optional(run.get("max_tokens"), int),
            max_hours=_optional(run.get("max_hours"), float),
            source=data,
        )

    def runs_of(self, candidate: Candidate) -> list[RunSpec]:
        """The runs that evaluate one candidate: every task, every repeat."""
        return [
            RunSpec(task, candidate.architecture, candidate.factors, repeat)
            for task, repeat in itertools.product(self.tasks, range(1, self.repeats + 1))
        ]

    def runs(self) -> list[RunSpec]:
        """Every run of the whole grid (a finite space only): candidate x task x repeat."""
        return [spec for candidate in self.space.grid() for spec in self.runs_of(candidate)]


def _check_optimizer(settings: OptimizerSettings, space: SearchSpace) -> None:
    # imported here: optimizers imports this module for the space and candidate types
    from tabular_preprocessing.outer_loop.optimizers import OPTIMIZERS

    if settings.name not in OPTIMIZERS:
        raise ConfigError(f"optimizer {settings.name!r} is unknown; available: {sorted(OPTIMIZERS)}.")
    if settings.n_trials is None and not space.is_finite:
        raise ConfigError("optimizer.n_trials is needed when the space has ranges.")
    if settings.n_trials is not None and settings.n_trials < 1:
        raise ConfigError("optimizer.n_trials must be at least 1.")
    if settings.name == "grid" and not space.is_finite:
        raise ConfigError(f"factors {sorted(space.ranges)} are ranges: grid cannot list them; use random.")


def _shape(arch: Architecture, factors: Factors) -> Architecture:
    """The architecture as the factors shape it: group order, and with an orchestrator no back edges."""
    if arch.solo:  # one agent: no groups to order, no orchestrator to add
        return arch
    arch = replace(arch, order=ORDERS[factors.order])
    if factors.orchestrator:
        arch = replace(
            arch,
            clean=GroupArchitecture(arch.clean.roles),
            merge=GroupArchitecture(arch.merge.roles),
            orchestrator=True,
        )
    return arch


def _collapse(factors: Factors) -> Factors:
    """Inactive factors to their default, so equal systems have equal keys."""
    if not factors.orchestrator:
        factors = replace(factors, orchestrator_routing=Factors.orchestrator_routing)
    return factors


def _check_factors(values: dict[str, Any]) -> Factors:
    if values["orchestrator"] not in (True, False):
        raise ConfigError("factors.orchestrator must be on or off.")
    if values["orchestrator_routing"] not in ROUTING:
        raise ConfigError(f"factors.orchestrator_routing must be one of {list(ROUTING)}.")
    temperature = values["temperature"]
    if temperature is not None and not (isinstance(temperature, int | float) and 0 <= temperature <= 2):
        raise ConfigError(f"factors.temperature must be between 0 and 2, or null; got {temperature!r}.")
    models = values["models"]
    if isinstance(models, str):
        models = {"default": models}
    if not isinstance(models, dict) or "default" not in models or set(models) - set(MODEL_ROLES):
        raise ConfigError(f"factors.models must map 'default' (and optionally P, E, V, C) to a model; got {models!r}.")
    if values["tool_description"] not in DESCRIPTION_LEVELS:
        raise ConfigError(f"factors.tool_description must be one of {list(DESCRIPTION_LEVELS)}.")
    if values["autonomy"] not in AUTONOMY_LEVELS:
        raise ConfigError(f"factors.autonomy must be one of {list(AUTONOMY_LEVELS)}.")
    if values["order"] == "pretest":
        raise ConfigError("factors.order: run the pre-test first, then set clean_first or merge_first.")
    if values["order"] not in ORDERS:
        raise ConfigError(f"factors.order must be one of {list(ORDERS)}.")
    if values["merge_fills_sample"] not in (True, False):
        raise ConfigError("factors.merge_fills_sample must be yes or no.")
    return Factors(
        orchestrator=values["orchestrator"],
        orchestrator_routing=values["orchestrator_routing"],
        temperature=None if temperature is None else float(temperature),
        models=tuple(sorted((k, str(v or "")) for k, v in models.items())),
        tool_description=values["tool_description"],
        autonomy=values["autonomy"],
        order=values["order"],
        merge_fills_sample=values["merge_fills_sample"],
    )


def _range(key: str, raw: dict[str, Any]) -> tuple[float, float]:
    if key != "temperature":
        raise ConfigError(f"factors.{key}: only temperature can be a range; give a list of choices.")
    _known(raw, {"low", "high"}, f"factors.{key}.")
    low, high = float(raw["low"]), float(raw["high"])
    if not 0 <= low <= high <= 2:
        raise ConfigError(f"factors.{key}: need 0 <= low <= high <= 2, got {low}..{high}.")
    return low, high


def _optional(value: Any, kind: Any) -> Any:
    return None if value is None else kind(value)


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def _known(section: Any, allowed: set[str], where: str) -> None:
    if not isinstance(section, dict):
        raise ConfigError(f"{where or 'the configuration'} must be a mapping.")
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"Unknown keys {sorted(f'{where}{k}' for k in unknown)}; allowed: {sorted(allowed)}.")
