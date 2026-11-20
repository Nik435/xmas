"""The outer loop's optimizers: swappable, one interface.

An optimizer proposes candidates (:meth:`Optimizer.ask`) and learns their objective
values (:meth:`Optimizer.tell`). The study loop does the rest: it builds each candidate
as an xmas system, runs the inner loop on every task and tells the optimizer the mean
improvement. Adding a method means one class with ``ask`` and ``tell``, registered in
:data:`OPTIMIZERS` under the name a configuration uses (``optimizer: {name: ...}``).

- ``grid``: every candidate of a finite space, in order.
- ``random``: candidates drawn uniformly from the space (each active hyperparameter from
  its choices or range), never the same candidate twice.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from tabular_preprocessing.outer_loop.config import Candidate, SearchSpace


class Optimizer(ABC):
    """Proposes candidates and learns their values; higher values are better."""

    def __init__(self, space: SearchSpace, seed: int = 0, **params: Any) -> None:
        """Initialize the optimizer.

        Args:
            space: The search space.
            seed: Seed of the optimizer's randomness.
            **params: Method-specific settings (``optimizer.params`` in the configuration).
        """
        self.space = space
        self.seed = seed
        self.params = params
        self.history: list[tuple[Candidate, float]] = []
        self.seen: set[str] = set()

    @abstractmethod
    def ask(self) -> Candidate | None:
        """The next candidate to evaluate, or ``None`` when the method has nothing left to propose."""

    def tell(self, candidate: Candidate, value: float) -> None:
        """Learn a candidate's objective value (also used to replay earlier trials when a study resumes)."""
        self.history.append((candidate, value))
        self.seen.add(candidate.key)

    def skip(self, candidate: Candidate) -> None:
        """Set a candidate aside for this session without a value (its runs hit API errors).

        It is not proposed again until the study is started anew, which then evaluates it.
        """
        self.seen.add(candidate.key)

    @property
    def best(self) -> tuple[Candidate, float] | None:
        """The best candidate so far with its value."""
        return max(self.history, key=lambda item: item[1]) if self.history else None


class GridSearch(Optimizer):
    """Every candidate of a finite space, in a stable order; ones already told are skipped."""

    def __init__(self, space: SearchSpace, seed: int = 0, **params: Any) -> None:
        """Initialize the grid.

        Raises:
            ValueError: If the space has ranges.
        """
        super().__init__(space, seed, **params)
        self._grid = space.grid()

    def ask(self) -> Candidate | None:
        """The next candidate of the grid not evaluated yet."""
        return next((c for c in self._grid if c.key not in self.seen), None)


class RandomSearch(Optimizer):
    """Candidates drawn uniformly from the space; a candidate is never proposed twice.

    Params:
        max_draws: Draws per ``ask`` before the space counts as exhausted (default 1000).
    """

    def __init__(self, space: SearchSpace, seed: int = 0, **params: Any) -> None:
        """Initialize the sampler with its own random generator."""
        super().__init__(space, seed, **params)
        self.rng = random.Random(seed)
        self.max_draws = int(params.get("max_draws", 1000))
        self._asked: set[str] = set()

    def ask(self) -> Candidate | None:
        """A random candidate not proposed or evaluated before, or ``None`` if none is found."""
        for _ in range(self.max_draws):
            candidate = self.space.sample(self.rng)
            if candidate.key not in self.seen and candidate.key not in self._asked:
                self._asked.add(candidate.key)
                return candidate
        return None


class ListOptimizer(Optimizer):
    """A fixed list of candidates, in order: one shard of a study split across cluster jobs."""

    def __init__(self, space: SearchSpace, candidates: list[Candidate]) -> None:
        """Initialize with the candidates this shard evaluates."""
        super().__init__(space)
        self.candidates = candidates

    def ask(self) -> Candidate | None:
        """The next candidate of the list not evaluated yet."""
        return next((c for c in self.candidates if c.key not in self.seen), None)


# optimizers whose candidates do not depend on earlier results: a study with them can be split into shards
SHARDABLE: tuple[str, ...] = ("grid", "random")


def planned_candidates(name: str, space: SearchSpace, n: int, seed: int = 0, **params: Any) -> list[Candidate]:
    """The first ``n`` candidates the optimizer would propose without any feedback.

    For ``grid`` and ``random`` that is exactly what they propose in a study, so the list can be split
    across jobs; it is the same in every job, as it only depends on the space and the seed.

    Raises:
        ValueError: For an optimizer that learns from results (its candidates are not known in advance).
    """
    if name not in SHARDABLE:
        raise ValueError(f"Optimizer {name!r} learns from results and cannot be split into shards; use {SHARDABLE}.")
    optimizer = make_optimizer(name, space, seed, **params)
    out = []
    while len(out) < n and (candidate := optimizer.ask()) is not None:
        optimizer.skip(candidate)
        out.append(candidate)
    return out


# name in the configuration -> optimizer class
OPTIMIZERS: dict[str, type[Optimizer]] = {"grid": GridSearch, "random": RandomSearch}


def make_optimizer(name: str, space: SearchSpace, seed: int = 0, **params: Any) -> Optimizer:
    """Build the optimizer a configuration names.

    Raises:
        KeyError: If the name is unknown.
    """
    if name not in OPTIMIZERS:
        raise KeyError(f"Unknown optimizer {name!r}; available: {sorted(OPTIMIZERS)}.")
    return OPTIMIZERS[name](space, seed, **params)
