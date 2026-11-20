"""Orchestrated systems: the orchestrator decides after every agent's turn what happens next.

    orchestrator -> decides {"next": agent, "instruction": ...} -> that agent works -> orchestrator -> ...
                 -> {"next": "submit"} ends the run

xmas runs fixed graphs (forward edges, and back edges taken on ``<continue>``); it has no
edge whose target an agent chooses at run time. So this module drives the hub-and-spoke
loop itself, with xmas's pieces: the agents, tools and model bindings of an xmas ``MAS``
(used as a container, so xmas checks them as usual), xmas's tool loop for each turn, and an
xmas ``Trace`` with one record per turn as the result.

Routing (a configuration factor):

- ``ordered`` (default): no agent of the second group works before the first group's
  executor has worked once; going back to the first group later is allowed.
- ``free``: any agent at any time.

The orchestrator sees every agent's full latest answer and the list of decisions so far.
A decision it cannot make (unknown agent, against the order, no JSON) is fed back to it as
an error; after ``max_invalid`` such answers in a row the run stops. The run also stops when
``max_steps`` agent turns are used up; the draft as it is then counts as submitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from promptolution.utils.token_counter import get_token_counter

from tabular_preprocessing.inner_loop.agents import AGENT_TYPES, ORCHESTRATOR, Team
from tabular_preprocessing.inner_loop.architecture import Architecture
from tabular_preprocessing.inner_loop.tools import Toolbox, Workspace
from xmas import MAS, Agent, AgentRecord, TokenUsage, Trace
from xmas.prompts import safe_format
from xmas.tools import run_tool_loop

ROUTING: tuple[str, ...] = ("ordered", "free")
SUBMIT = "submit"


class DecisionError(ValueError):
    """An orchestrator answer that is not a decision it may make."""


@dataclass(frozen=True)
class Decision:
    """What the orchestrator decided.

    Attributes:
        next: The agent's name, or ``"submit"``.
        instruction: What the agent should do now.
        summary: Why this pipeline, when submitting.
    """

    next: str
    instruction: str = ""
    summary: str = ""


@dataclass
class OrchestratedRun:
    """The result of an orchestrated run.

    Attributes:
        trace: One record per turn (orchestrator and agents), in order.
        decisions: Every decision, valid or not: ``{"turn", "next", "instruction", "error"}``.
        steps: Agent turns used.
        stopped: ``"submitted"``, ``"budget"`` (``max_steps`` used up), ``"invalid"`` (too many
            invalid decisions in a row) or ``"error"`` (a turn failed).
    """

    trace: Trace
    decisions: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    stopped: str = ""


def parse_decision(text: str) -> Decision:
    """The last JSON object with a ``"next"`` key in an orchestrator's answer.

    Raises:
        DecisionError: If there is none.
    """
    decoder = json.JSONDecoder()
    for start in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "next" in obj:
            return Decision(str(obj["next"]), str(obj.get("instruction", "")), str(obj.get("summary", "")))
    raise DecisionError('No decision found: end the answer with {"next": ..., "instruction": ...}.')


def run_orchestrated(
    architecture: Architecture,
    workspace: Workspace,
    llm: Any,
    llms: dict[str, Any] | None = None,
    models: dict[str, str] | None = None,
    max_steps: int = 12,
    max_tool_rounds: int = 20,
    autonomy: str = "bounded",
    routing: str = "ordered",
    max_invalid: int = 3,
) -> OrchestratedRun:
    """Run an orchestrated system on ``workspace`` until it submits or its budget is used up.

    Args:
        architecture: Which roles each group has (back edges must be off: the orchestrator routes).
        workspace: The run's workspace; the agents' tools edit its draft pipeline.
        llm: The default model of every agent.
        llms: Models by name, for agents bound by ``models``.
        models: Optional model name per agent name (``"orchestrator"`` for the orchestrator).
        max_steps: Most agent turns the orchestrator may delegate.
        max_tool_rounds: Tool-call turns per agent turn.
        autonomy: ``"fixed"``, ``"bounded"`` or ``"free"``, told to every agent.
        routing: ``"ordered"`` or ``"free"``.
        max_invalid: Invalid decisions in a row before the run stops.

    Returns:
        The trace, the decisions and why the run stopped.

    Raises:
        ValueError: On an unknown routing, or an architecture with back edges.
    """
    if routing not in ROUTING:
        raise ValueError(f"Unknown routing {routing!r}; choose from {list(ROUTING)}.")
    variants = [getattr(architecture, g) for g in architecture.order]
    if any(v.fb_e or v.fb_v for v in variants):
        raise ValueError("An orchestrated system has no back edges: the orchestrator routes every turn.")
    models = models or {}
    toolbox = Toolbox(workspace)

    # the agents, each seeing the other group's latest result
    first, second = architecture.order
    last_output = {g: AGENT_TYPES[g][getattr(architecture, g).chain[-1]].output for g in architecture.order}
    agents: list[Agent] = []
    group_of: dict[str, str] = {}
    titles: dict[str, str] = {}
    for group in architecture.order:
        variant = getattr(architecture, group)
        other = second if group == first else first
        upstream = last_output[other] if (group == second or routing == "free") else None
        team = Team(variant.roles, upstream=upstream, orchestrated=True, autonomy=autonomy)
        for role in variant.chain:
            agent_type = AGENT_TYPES[group][role]
            agent = agent_type.build(team, toolbox, AGENT_TYPES[group], model=models.get(agent_type.name))
            agents.append(agent)
            group_of[agent.name] = group
            titles[agent.name] = agent_type.title

    # the orchestrator, reading every agent's latest answer and the decisions so far
    team = Team(members=tuple((titles[a.name], a.output) for a in agents), autonomy=autonomy)
    orchestrator = ORCHESTRATOR.build(team, toolbox, AGENT_TYPES["orchestrator"], model=models.get("orchestrator"))
    orchestrator.prompt += "\n\n" + _rules(agents, titles, group_of, architecture.order, routing, max_steps)
    orchestrator.inputs.append("history")

    # a container, so xmas checks tools and model bindings; its graph is never run (the loop below routes)
    edges = [(a.name, orchestrator.name) for a in agents]
    mas = MAS([orchestrator, *agents], edges=edges, llm=llm, llms=llms, tools=toolbox.tools())

    by_name = {a.name: a for a in agents}
    executor = AGENT_TYPES[first]["E"].name
    state: dict[str, str] = {"query": workspace.task.describe()}
    records: list[AgentRecord] = []
    run = OrchestratedRun(Trace(query=state["query"], output=""))
    history: list[str] = []
    worked: set[str] = set()
    invalid = 0
    answer = ""
    while True:
        state["history"] = _history(history, run.steps, max_steps)
        try:
            answer = _turn(mas, orchestrator, state, records, max_tool_rounds)
        except Exception as error:  # a failed turn ends the run; the draft still counts
            return _finish(run, records, answer, "error", f"{type(error).__name__}: {error}")
        if workspace.submitted:  # the orchestrator called its submit tool
            return _finish(run, records, answer, "submitted")
        try:
            decision = parse_decision(answer)
            _allowed(decision, by_name, group_of, executor, worked, second, routing)
        except DecisionError as error:
            invalid += 1
            run.decisions.append({"turn": len(records) - 1, "next": None, "instruction": "", "error": str(error)})
            history.append(f"invalid decision: {error}")
            if invalid >= max_invalid:
                return _finish(run, records, answer, "invalid")
            continue
        invalid = 0
        run.decisions.append(
            {"turn": len(records) - 1, "next": decision.next, "instruction": decision.instruction, "error": None}
        )
        if decision.next == SUBMIT:
            workspace.submitted, workspace.summary = True, decision.summary
            return _finish(run, records, answer, "submitted")
        if run.steps >= max_steps:
            return _finish(run, records, answer, "budget")
        run.steps += 1
        agent = by_name[decision.next]
        state["instruction"] = decision.instruction
        try:
            state[agent.output] = _turn(mas, agent, state, records, max_tool_rounds)
        except Exception as error:
            return _finish(run, records, answer, "error", f"{type(error).__name__}: {error}")
        worked.add(agent.name)
        history.append(f"{agent.name}: {decision.instruction}")


def _turn(mas: MAS, agent: Agent, state: dict[str, str], records: list[AgentRecord], max_tool_rounds: int) -> str:
    """One agent's turn, recorded as xmas records it."""
    llm = mas.resolve_llm(agent)
    count = get_token_counter(llm)
    inputs = {name: state.get(name, "") for name in agent.inputs}
    rendered = safe_format(agent.prompt, inputs)
    calls: tuple[Any, ...] = ()
    if agent.tools:
        response, calls, usage = run_tool_loop(
            llm, rendered, mas.resolve_tools(agent), max_tool_rounds, DEFAULT_SYS_PROMPT, count
        )
    else:
        response = llm.get_response([rendered])[0]
        usage = TokenUsage(input_tokens=int(count(rendered)), output_tokens=int(count(response)))
    records.append(
        AgentRecord(
            agent=agent.name,
            step=len(records),
            prompt=agent.prompt,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: response},
            tool_calls=calls,
            usage=usage,
            model=agent.model,
            raw_response=response,
        )
    )
    return str(response)


def _allowed(
    decision: Decision,
    agents: dict[str, Agent],
    group_of: dict[str, str],
    executor: str,
    worked: set[str],
    second: str,
    routing: str,
) -> None:
    if decision.next == SUBMIT:
        return
    if decision.next not in agents:
        raise DecisionError(f"Unknown agent {decision.next!r}; choose from {[*agents, SUBMIT]}.")
    if routing == "ordered" and group_of[decision.next] == second and executor not in worked:
        raise DecisionError(
            f"{decision.next} belongs to group {second}, which may only start after {executor} has worked once."
        )


def _rules(
    agents: list[Agent],
    titles: dict[str, str],
    group_of: dict[str, str],
    order: tuple[str, str],
    routing: str,
    max_steps: int,
) -> str:
    lines = ["## The agents you can delegate to"]
    lines += [f"- {a.name}: the {titles[a.name]} of group {group_of[a.name]}" for a in agents]
    if routing == "ordered":
        rule = (
            f"Group {order[0]} works first: no agent of group {order[1]} may work before "
            f"{order[0]}_executor has worked once. You may go back to group {order[0]} later."
        )
    else:
        rule = "You may delegate to any agent at any time, in any order."
    lines += ["", "## Rules", rule, f"You may delegate at most {max_steps} turns in total; then the draft counts."]
    lines += ["", "## Decisions so far", "{history}"]
    return "\n".join(lines)


def _history(history: list[str], steps: int, max_steps: int) -> str:
    done = "\n".join(f"{i}. {line}" for i, line in enumerate(history, start=1)) or "(none yet)"
    return f"{done}\nTurns used: {steps} of {max_steps}."


def _finish(
    run: OrchestratedRun, records: list[AgentRecord], output: str, stopped: str, error: str | None = None
) -> OrchestratedRun:
    run.trace = Trace(query=run.trace.query, output=output, records=tuple(records), error=error)
    run.stopped = stopped
    return run
