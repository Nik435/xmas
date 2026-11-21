"""Shared fixtures: a small synthetic relational task (no download needed) and a scripted model."""

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from promptolution.llms.base_llm import BaseLLM

from tabular_preprocessing import RelationalTask, Table
from xmas.tools import Tool, ToolCallRequest

DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2020-01-01")


@pytest.fixture
def task() -> RelationalTask:
    """Drivers with a hidden skill; results reveal it; the label is a good next result."""
    rng = np.random.default_rng(0)
    n_drivers, n_races = 30, 60
    skill = rng.normal(0, 1, n_drivers)
    teams = pd.DataFrame({"teamId": range(5), "country": list("ABCDE")})
    drivers = pd.DataFrame(
        {
            "driverId": range(n_drivers),
            "teamId": rng.integers(0, 5, n_drivers),
            "nationality": rng.choice(["x", "y", "z"], n_drivers),
        }
    )
    races = pd.DataFrame({"raceId": range(n_races), "date": [T0 + 7 * i * DAY for i in range(n_races)]})
    rows = [
        (r * n_drivers + d, d, r, races.date[r], skill[d] + rng.normal(0, 0.5), rng.integers(1, 20))
        for r in range(n_races)
        for d in range(n_drivers)
    ]
    results = pd.DataFrame(rows, columns=["resultId", "driverId", "raceId", "date", "points", "position"])
    results.loc[results.index % 11 == 0, "points"] = -1.0  # a placeholder for "not recorded"
    db = {
        "teams": Table(teams, "teamId"),
        "drivers": Table(drivers, "driverId", {"teamId": "teams"}),
        "races": Table(races, "raceId", time_col="date"),
        "results": Table(results, "resultId", {"driverId": "drivers", "raceId": "races"}, "date"),
    }
    labels = pd.DataFrame(
        [
            (races.date[r], d, int(skill[d] + rng.normal(0, 0.3) > 0))
            for r in range(10, n_races)
            for d in range(n_drivers)
        ],
        columns=["date", "driverId", "good"],
    )
    cut = T0 + 7 * 45 * DAY
    return RelationalTask(
        "synthetic/driver-good",
        db,
        "drivers",
        "driverId",
        "date",
        "good",
        "classification",
        train=labels[labels.date < cut].reset_index(drop=True),
        val=labels[labels.date >= cut].reset_index(drop=True),
    )


def with_licences(task: RelationalTask) -> RelationalTask:
    """The task plus a 1:1 table: one licence per driver, issued in week ``driverId``, grade ``driverId % 3``."""
    n = len(task.db["drivers"].df)
    licences = pd.DataFrame(
        {
            "licenceId": range(n),
            "driverId": range(n),
            "issued": [T0 + 7 * d * DAY for d in range(n)],
            "grade": [d % 3 for d in range(n)],
        }
    )
    db = {**task.db, "licences": Table(licences, "licenceId", {"driverId": "drivers"}, "issued")}
    return RelationalTask(**{**task.__dict__, "db": db})


class TeamLLM(BaseLLM):
    """A scripted model for whole systems: answers each agent by its role, from the prompt it gets.

    - Executors (they have tools): on their first visit, call ``<group>_add_step`` once per
      step scripted for their group; on later visits only look at the pipeline. Then report.
    - Validators: give the next scripted verdict of their group, then ``APPROVED`` (no tool calls).
    - Planners: a fixed plan (no tool calls).
    - The orchestrator: the next scripted decision, then submit (no tool calls).
    """

    def __init__(
        self,
        steps: dict[str, list[dict[str, Any]]] | None = None,
        verdicts: dict[str, list[str]] | None = None,
        decisions: list[str] | None = None,
    ) -> None:
        """Script the run.

        Args:
            steps: Per group, the steps its executor adds (``{"stage": ..., "step": ...}`` each).
            verdicts: Per group, the validator's answers in order; ``APPROVED`` after they run out.
            decisions: The orchestrator's answers in order; a submit decision after they run out.
        """
        super().__init__()
        self.decisions = list(decisions or [])
        self.steps = steps or {}
        self.verdicts = {group: list(answers) for group, answers in (verdicts or {}).items()}
        self.visits: dict[str, int] = {}

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        """Planners and validators, which have no tools yet."""
        return [self._answer(prompt) for prompt in prompts]

    def _answer(self, prompt: str) -> str:
        if ", the orchestrator of group" in prompt:
            return self.decisions.pop(0) if self.decisions else '{"next": "submit", "summary": "done"}'
        group = "clean" if "of group clean" in prompt else "merge"
        if ", the validator of" in prompt:
            answers = self.verdicts.get(group, [])
            return answers.pop(0) if answers else "APPROVED"
        return f"Plan for {group}: use the obvious steps."

    def get_tool_response(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest]]:
        """Executors add the scripted steps on their first visit, then report; planners and validators answer."""
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        if ", the executor of" not in prompt:
            return self._answer(prompt), []
        group = tools[0].name.split("_", 1)[0]
        # visits are counted per run: every run builds its own tools, so the tool object tells runs apart
        # (one model object serves many runs in a study, as a real, stateless LLM does)
        key = f"{group}:{id(tools[0])}"
        if len(messages) <= 2:  # a new visit: the rendered prompt is the only user message
            self.visits[key] = self.visits.get(key, 0) + 1
            calls = (
                [(f"{group}_add_step", args) for args in self.steps.get(group, [])]
                if self.visits[key] == 1
                else [(f"{group}_show_pipeline", {})]
            )
            if calls:
                return "", [ToolCallRequest(f"c{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)]
        return f"{group} visit {self.visits[key]}: done.", []
