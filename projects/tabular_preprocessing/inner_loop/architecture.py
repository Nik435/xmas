"""Architectures: the stage-1 search space, built as xmas systems.

Each group (Clean, Merge) uses the role chain Planner -> Executor -> Validator. The
executor is always present and the flow forward is fixed; only the back edges to the
direct predecessor are free (E->P, V->E, each on or off, taken on rejection and
bounded by ``max_loops``). That gives 9 variants per group::

    E          PE   PE-EP          EV   EV-VE          PEV   PEV-EP   PEV-VE   PEV-EP-VE

and 9 x 9 = 81 architectures. The groups are connected by one edge, the last agent of
the first group to the first agent of the second.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from tabular_preprocessing.inner_loop.agents import AGENT_TYPES, Team
from tabular_preprocessing.inner_loop.tools import Toolbox, Workspace
from xmas import END, MAS, START, Agent

GROUPS: tuple[str, ...] = ("clean", "merge")
_ROLE_ORDER = ("P", "E", "V")


@dataclass(frozen=True)
class GroupArchitecture:
    """One group's variant: which roles are present and which back edges are on.

    Attributes:
        roles: A subset of ``{"P", "E", "V"}`` that contains ``"E"``.
        fb_e: Back edge executor -> planner.
        fb_v: Back edge validator -> executor.
    """

    roles: frozenset[str] = frozenset({"E"})
    fb_e: bool = False
    fb_v: bool = False

    def __post_init__(self) -> None:
        """Check the variant against the search space's rules (the same as :class:`Team`'s)."""
        Team(self.roles, self.fb_e, self.fb_v)

    @property
    def chain(self) -> list[str]:
        """The roles in flow order, e.g. ``["P", "E", "V"]``."""
        return [r for r in _ROLE_ORDER if r in self.roles]

    @property
    def code(self) -> str:
        """Short code, e.g. ``"PEV-EP-VE"``; parsed back by :meth:`from_code`."""
        return "-".join(["".join(self.chain), *(["EP"] if self.fb_e else []), *(["VE"] if self.fb_v else [])])

    @classmethod
    def from_code(cls, code: str) -> GroupArchitecture:
        """Read a variant from its code, e.g. ``"PE-EP"``.

        Raises:
            ValueError: On an unknown code.
        """
        chain, *edges = code.split("-")
        if not chain or set(chain) - set(_ROLE_ORDER) or set(edges) - {"EP", "VE"}:
            raise ValueError(f"Unknown group code {code!r}; examples: {[v.code for v in cls.variants()]}.")
        return cls(frozenset(chain), "EP" in edges, "VE" in edges)

    @classmethod
    def from_hp(cls, size: int, role: str | None = None, fb_e: bool = False, fb_v: bool = False) -> GroupArchitecture:
        """Build a variant from the concept's hyperparameters; inactive ones are ignored.

        Args:
            size: Agents in the group: 1 (E), 2 (E plus ``role``) or 3 (P, E, V).
            role: ``"planner"`` or ``"validator"``; active only when ``size`` is 2.
            fb_e: Back edge E->P; active only when a planner is present.
            fb_v: Back edge V->E; active only when a validator is present.

        Raises:
            ValueError: On an unknown size or role.
        """
        if size == 1:
            roles = {"E"}
        elif size == 2:
            if role not in ("planner", "validator"):
                raise ValueError(f"size 2 needs role 'planner' or 'validator', got {role!r}.")
            roles = {"P", "E"} if role == "planner" else {"E", "V"}
        elif size == 3:
            roles = {"P", "E", "V"}
        else:
            raise ValueError(f"size must be 1, 2 or 3, got {size!r}.")
        return cls(frozenset(roles), bool(fb_e) and "P" in roles, bool(fb_v) and "V" in roles)

    @property
    def hp(self) -> dict[str, object]:
        """The concept's hyperparameters of this variant; inactive ones are ``None``."""
        size = len(self.roles)
        role = None if size != 2 else ("planner" if "P" in self.roles else "validator")
        return {
            "size": size,
            "role": role,
            "fb_E": int(self.fb_e) if "P" in self.roles else None,
            "fb_V": int(self.fb_v) if "V" in self.roles else None,
        }

    @classmethod
    def grid(
        cls,
        size: list[int],
        role: list[str],
        fb_e: list[bool],
        fb_v: list[bool],
    ) -> list[GroupArchitecture]:
        """Every distinct variant from a grid of hyperparameter values; inactive values collapse.

        ``grid([1, 2, 3], ["planner", "validator"], [0, 1], [0, 1])`` gives the 9 variants.
        """
        out: list[GroupArchitecture] = []
        for s, r, e, v in itertools.product(size, role or [None], fb_e or [False], fb_v or [False]):
            variant = cls.from_hp(s, r if s == 2 else None, e, v)
            if variant not in out:
                out.append(variant)
        return out

    @classmethod
    def variants(cls) -> list[GroupArchitecture]:
        """All 9 variants: E; P->E and E->V with their back edge off and on; P->E->V with both off and on."""
        out = []
        for roles in (["E"], ["P", "E"], ["E", "V"], ["P", "E", "V"]):
            for fb_e, fb_v in itertools.product([False, True], repeat=2):
                if (fb_e and "P" not in roles) or (fb_v and "V" not in roles):
                    continue
                out.append(cls(frozenset(roles), fb_e, fb_v))
        return out


@dataclass(frozen=True)
class Architecture:
    """One system of the search space.

    Attributes:
        clean: Group Clean's variant.
        merge: Group Merge's variant.
        order: Which group works first (a stage-2 factor; stage 1: clean first).
        orchestrator: Whether an orchestrator leads the system (a stage-2 factor). It then routes every turn,
            so back edges are off; such a system is run by :func:`.orchestrated.run_orchestrated`.
        solo: The single-agent baseline (code ``"solo"``): one Generalist fills every slot with every tool;
            the groups do not apply. Outside the concept's search space, for comparison.
    """

    clean: GroupArchitecture = GroupArchitecture()
    merge: GroupArchitecture = GroupArchitecture()
    order: tuple[str, str] = GROUPS
    orchestrator: bool = False
    solo: bool = False

    def __post_init__(self) -> None:
        """Check the group order.

        Raises:
            ValueError: If ``order`` is not the two groups.
        """
        if sorted(self.order) != sorted(GROUPS):
            raise ValueError(f"order must name both groups once, got {self.order}.")
        if self.solo and (self.orchestrator or self.clean != GroupArchitecture() or self.merge != GroupArchitecture()):
            raise ValueError("The single-agent baseline has no groups and no orchestrator.")
        if self.orchestrator and any(v.fb_e or v.fb_v for v in (self.clean, self.merge)):
            raise ValueError("With an orchestrator there are no back edges: it decides every next turn.")

    @property
    def code(self) -> str:
        """Short code, e.g. ``"clean=PE-EP|merge=PEV-VE"``, groups in working order; ``"solo"`` for the baseline."""
        if self.solo:
            return "solo"
        code = "|".join(f"{g}={getattr(self, g).code}" for g in self.order)
        return code + ("|orchestrator" if self.orchestrator else "")

    @classmethod
    def from_code(cls, code: str) -> Architecture:
        """Read an architecture from its code (see :attr:`code`).

        Raises:
            ValueError: On an unknown code.
        """
        if code == "solo":
            return cls(solo=True)
        parts = code.split("|")
        orchestrator = parts[-1] == "orchestrator"
        groups = dict(part.split("=", 1) for part in (parts[:-1] if orchestrator else parts))
        if sorted(groups) != sorted(GROUPS):
            raise ValueError(f"An architecture code names both groups, e.g. 'clean=E|merge=PEV-VE'; got {code!r}.")
        return cls(
            clean=GroupArchitecture.from_code(groups["clean"]),
            merge=GroupArchitecture.from_code(groups["merge"]),
            order=tuple(groups),  # type: ignore[arg-type]
            orchestrator=orchestrator,
        )

    @classmethod
    def stage1(cls) -> list[Architecture]:
        """The 81 architectures of stage 1: every Clean variant with every Merge variant, Clean first."""
        variants = GroupArchitecture.variants()
        return [cls(clean, merge) for clean, merge in itertools.product(variants, variants)]

    def agents(
        self, toolbox: Toolbox, models: dict[str, str] | None = None, autonomy: str = "bounded"
    ) -> tuple[list[Agent], list[tuple[str, str]]]:
        """The system's agents and edges, with tools from ``toolbox``.

        Args:
            toolbox: The run's toolbox.
            models: Optional model name per agent name (e.g. ``{"clean_executor": "big"}``);
                agents not listed use the system's default model.
            autonomy: ``"fixed"``, ``"bounded"`` or ``"free"``, told to every agent.

        Returns:
            The agents and the edges, including the explicit ``START`` and ``END`` edges.

        Raises:
            ValueError: For an architecture with orchestrator (see :mod:`.orchestrated`).
        """
        if self.solo:
            agent = AGENT_TYPES["solo"]["E"].build(
                Team(autonomy=autonomy), toolbox, AGENT_TYPES["solo"], model=(models or {}).get("solo_executor")
            )
            return [agent], [(START, agent.name), (agent.name, END)]
        if self.orchestrator:
            raise ValueError(
                "An orchestrated system is not a fixed graph; run it with orchestrated.run_orchestrated "
                "(run_inner_loop does so)."
            )
        models = models or {}
        agents: list[Agent] = []
        edges: list[tuple[str, str]] = []
        upstream: str | None = None
        previous: str | None = None
        for group in self.order:
            variant: GroupArchitecture = getattr(self, group)
            types = AGENT_TYPES[group]
            team = Team(variant.roles, variant.fb_e, variant.fb_v, upstream=upstream, autonomy=autonomy)
            names = {role: types[role].name for role in variant.chain}
            for role in variant.chain:
                agent_type = types[role]
                agents.append(agent_type.build(team, toolbox, types, model=models.get(agent_type.name)))
            chain = [names[r] for r in variant.chain]
            edges += list(itertools.pairwise(chain))
            if variant.fb_e:
                edges.append((names["E"], names["P"]))
            if variant.fb_v:
                edges.append((names["V"], names["E"]))
            # explicit, because a back edge gives the first agent a predecessor and the
            # last a successor, so xmas would not wire them to START and END itself
            edges.append((previous, chain[0]) if previous else (START, chain[0]))
            previous = chain[-1]
            upstream = types[variant.chain[-1]].output
        assert previous is not None
        edges.append((previous, END))
        return agents, edges

    def build(
        self,
        workspace: Workspace,
        llm: Any,
        llms: dict[str, Any] | None = None,
        models: dict[str, str] | None = None,
        max_loops: int = 3,
        max_tool_rounds: int = 20,
        autonomy: str = "bounded",
    ) -> MAS:
        """The xmas system for one run on ``workspace``.

        Args:
            workspace: The run's workspace; its tools edit the workspace's draft pipeline.
            llm: The default model of every agent.
            llms: Models by name, for agents bound by ``models``.
            models: Optional model name per agent name.
            max_loops: How often one agent may run in one query, which bounds every back-edge loop.
            max_tool_rounds: Tool-call turns per agent visit (an executor adds one step per call).
            autonomy: ``"fixed"``, ``"bounded"`` or ``"free"``, told to every agent.

        Returns:
            The system; run it with ``mas.run(task.describe())``.
        """
        toolbox = Toolbox(workspace)
        agents, edges = self.agents(toolbox, models, autonomy)
        return MAS(
            agents,
            edges=edges,
            llm=llm,
            llms=llms,
            tools=toolbox.tools(),
            max_visits=max_loops,
            max_tool_rounds=max_tool_rounds,
        )
