"""Agent types: the roles of the project concept, turned into xmas agents.

An :class:`AgentType` is one role of one group: what it is for, which tool sets it
uses, and the parts of its prompt. It becomes an :class:`xmas.Agent` only for a
concrete :class:`Team`, because what an agent reads and must do depends on who
else is there:

- an executor without a planner gets the planner's duties and tool sets, and one
  without a validator gets the validator's (only the division of labour varies,
  not the capabilities);
- a back edge adds the input it carries (the executor's objection for E->P, the
  validator's review for V->E) and the instruction to take it with ``<continue>``;
- the first agent of the second group reads the first group's result.

State variables are named ``<group>_<plan|work|review>``; the task itself (its
description and schema) is the system's ``query``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from tabular_preprocessing.inner_loop.tools.toolbox import TOOL_SETS, Toolbox
from xmas import Agent
from xmas.runtime import CONTINUE_MARKER

ROLES: dict[str, str] = {"P": "planner", "E": "executor", "V": "validator", "C": "orchestrator"}
OUTPUTS: dict[str, str] = {"P": "plan", "E": "work", "V": "review", "C": "decision"}

INTRO = (
    "You are the {title}, the {role} of group {group} in a team of agents that builds a preprocessing "
    "pipeline for a relational prediction task. The pipeline is fixed: "
    "clean (per table) -> merge (database features for each label row) -> engineer -> sample -> "
    "predictor (fixed) -> postprocess. Each slot is filled with steps from a fixed library of functions. "
    "Group clean fills clean; group merge fills merge and engineer."
)


# the prompt line per autonomy level; the run sets the matching limits (tool rounds, loops, evaluations)
AUTONOMY: dict[str, str] = {
    "fixed": (
        "Work strictly through the steps under 'How to work', in that order, using only the tools named there. "
        "Your tool calls are limited."
    ),
    "bounded": "Choose your tools and steps freely, but your tool calls and evaluations are limited.",
    "free": "Choose your own steps and tools; you have a large budget of tool calls and evaluations.",
}


class MissingToolSetWarning(UserWarning):
    """An agent type names a tool set that does not exist; the agent runs without it."""


@dataclass(frozen=True)
class Team:
    """Who else is in the system, from one agent's point of view.

    Attributes:
        roles: The roles present in the agent's group, a subset of ``{"P", "E", "V"}``.
        fb_e: Back edge executor -> planner (on rejection of the plan).
        fb_v: Back edge validator -> executor (on rejection of the work).
        upstream: State variable holding the previous group's result, if this group comes second.
        orchestrated: Whether an orchestrator delegates to the agent and reads its answer.
        members: For the orchestrator: ``(title, state variable)`` of every other agent.
        autonomy: How freely the agent works (a stage-2 factor): ``"fixed"`` (the steps under "How to
            work", in order), ``"bounded"`` (free tool choice, limited steps) or ``"free"`` (own steps,
            larger budget). The limits themselves are set by the run, not the prompt.
    """

    roles: frozenset[str] = frozenset({"E"})
    fb_e: bool = False
    fb_v: bool = False
    upstream: str | None = None
    orchestrated: bool = False
    members: tuple[tuple[str, str], ...] = ()
    autonomy: str = "bounded"

    def __post_init__(self) -> None:
        """Check the team against the search space's rules.

        Raises:
            ValueError: On an unknown role, a group without executor, or a back edge without its role.
        """
        unknown = set(self.roles) - {"P", "E", "V"}
        if unknown:
            raise ValueError(f"Unknown roles {sorted(unknown)}; a group has P, E and V.")
        if "E" not in self.roles:
            raise ValueError("Every group has an executor.")
        if self.fb_e and "P" not in self.roles:
            raise ValueError("The back edge E->P needs a planner.")
        if self.fb_v and "V" not in self.roles:
            raise ValueError("The back edge V->E needs a validator.")
        if self.autonomy not in AUTONOMY:
            raise ValueError(f"Unknown autonomy {self.autonomy!r}; choose from {list(AUTONOMY)}.")


@dataclass(frozen=True)
class AgentType:
    """One role of one group.

    Attributes:
        key: Identifier, e.g. ``"data_profiler"``.
        title: Name in the concept, e.g. ``"Data Profiler"``.
        group: ``"clean"``, ``"merge"`` or ``"orchestrator"``.
        role: ``"P"``, ``"E"``, ``"V"`` or ``"C"``.
        task: What the role is for, one line (the concept's task column).
        tool_sets: The tool sets the role itself uses.
        instructions: How to do the job, written for the agent.
        answer: What the answer must contain, written for the agent.
        absent_duties: Extra instructions for an executor, per role missing from its group.
    """

    key: str
    title: str
    group: str
    role: str
    task: str
    tool_sets: tuple[str, ...]
    instructions: str
    answer: str
    absent_duties: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The agent's name in a system, e.g. ``clean_planner``."""
        return "orchestrator" if self.role == "C" else f"{self.group}_{ROLES[self.role]}"

    @property
    def output(self) -> str:
        """The state variable the agent writes, e.g. ``clean_plan``."""
        return f"{self.group}_{OUTPUTS[self.role]}"

    def tool_sets_for(self, team: Team, types: dict[str, AgentType]) -> tuple[str, ...]:
        """The tool sets the agent gets in ``team``: its own, plus those of missing roles for an executor.

        Args:
            team: The team.
            types: The group's agent types by role, to look up the missing roles' sets.
        """
        sets = list(self.tool_sets)
        if self.role == "E":
            for role in ("P", "V"):
                if role not in team.roles and role in types:
                    sets += [s for s in types[role].tool_sets if s not in sets]
        return tuple(sets)

    def inputs(self, team: Team) -> list[str]:
        """The state variables the agent reads in ``team``."""
        if self.role == "C":
            return ["query", *(var for _, var in team.members)]
        g = self.group
        inputs = ["query"]
        if team.orchestrated:
            inputs.append("instruction")
        first = "P" if "P" in team.roles else "E"
        if team.upstream and self.role in (first, "E"):
            inputs.append(team.upstream)
        # with an orchestrator any agent may run again, so it sees the answer it would react to,
        # as a back edge would carry it
        if self.role == "P" and (team.fb_e or team.orchestrated):
            inputs.append(f"{g}_work")
        if self.role == "E":
            if "P" in team.roles:
                inputs.append(f"{g}_plan")
            if team.fb_v or (team.orchestrated and "V" in team.roles):
                inputs.append(f"{g}_review")
        if self.role == "V":
            if "P" in team.roles:
                inputs.append(f"{g}_plan")
            inputs.append(f"{g}_work")
        return inputs

    def prompt(self, team: Team) -> str:
        """The prompt template for ``team``; ``{placeholders}`` are the inputs."""
        parts = [INTRO.format(title=self.title, role=ROLES[self.role], group=self.group), f"Your job: {self.task}."]
        if team.orchestrated:
            parts.append("An orchestrator delegates the work and reads your answer.")
        parts.append(AUTONOMY[team.autonomy])
        parts.append("## Task and database\n{query}")
        for var in self.inputs(team)[1:]:
            parts.append(f"## {_heading(var, self.group, team)}\n{{{var}}}")
        parts.append(f"## How to work\n{self.instructions}")
        if self.role == "E":
            for role in ("P", "V"):
                if role not in team.roles and role in self.absent_duties:
                    parts.append(f"## Also yours, as there is no {ROLES[role]}\n{self.absent_duties[role]}")
        parts.append(f"## Your answer\n{self.answer}")
        loop = _loop_rule(self.role, team)
        if loop:
            parts.append(loop)
        return "\n\n".join(parts)

    def build(self, team: Team, toolbox: Toolbox, types: dict[str, AgentType], model: str | None = None) -> Agent:
        """The xmas agent for ``team``, with its tools taken from ``toolbox``.

        Tool sets that do not exist are skipped with a :class:`MissingToolSetWarning`.

        Args:
            team: The team.
            toolbox: The run's toolbox.
            types: The group's agent types by role.
            model: Model name for this agent, or ``None`` for the system's default.
        """
        if self.role != "C" and self.role not in team.roles:
            raise ValueError(f"{self.title} ({self.role}) is not part of a team with roles {sorted(team.roles)}.")
        tools: list[str] = []
        for name in self.tool_sets_for(team, types):
            if name not in TOOL_SETS:
                warnings.warn(
                    f"Tool set {name!r} of the {self.title} does not exist; the agent runs without it.",
                    MissingToolSetWarning,
                    stacklevel=2,
                )
                continue
            tools += [t.name for t in toolbox.get(self.group, name)]
        return Agent(
            self.name, self.prompt(team), output=self.output, inputs=self.inputs(team), tools=tools, model=model
        )


def _heading(var: str, group: str, team: Team) -> str:
    if var == "instruction":
        return "The orchestrator's instruction for this turn"
    if var == team.upstream:
        if team.orchestrated:
            return "Result of the other group (empty if it has not worked yet)"
        return "Result of the previous group"
    if team.orchestrated:
        kind = var.removeprefix(f"{group}_")
        latest = {"plan": "planner's plan", "work": "executor's report", "review": "validator's review"}
        if kind in latest:
            return f"The {latest[kind]}, latest version (empty if it has not run yet)"
    kind = var.removeprefix(f"{group}_")
    titles = {
        "plan": "The planner's plan",
        "work": "The executor's report",
        "review": "The validator's review of your last attempt (empty on the first attempt)",
    }
    if kind == "work" and team.fb_e:
        return "The executor's report or objection to your plan (empty on the first attempt)"
    return titles.get(kind, var)


def _loop_rule(role: str, team: Team) -> str | None:
    if team.orchestrated:
        if role == "V":
            return "Answer APPROVED or REJECTED, with your reasons; the orchestrator decides what happens next."
        return None
    if role == "E" and team.fb_e:
        return (
            f"If the plan cannot be carried out with the available functions or data, answer with "
            f"{CONTINUE_MARKER} followed by what must change; the planner then revises the plan."
        )
    if role == "V" and team.fb_v:
        return (
            f"If the work must be fixed, answer with {CONTINUE_MARKER} followed by concrete fixes; the "
            "executor then revises it. Otherwise answer APPROVED with a short summary."
        )
    if role == "V":
        after = (
            "the orchestrator decides what happens next" if team.orchestrated else "nobody revises the work after you"
        )
        return f"Answer APPROVED or REJECTED, with your reasons; {after}."
    return None
