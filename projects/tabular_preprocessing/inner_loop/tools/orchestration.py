"""Tool set: lead the run (the Orchestrator).

The orchestrator delegates, manages the budget, decides when to stop and submits
the pipeline. Its tools give it the overview (the draft, who changed what, the
evaluation budget and the scores known so far) and the act of submitting. It also
gets the validation tool set, to measure the draft itself.

How the orchestrator delegates to other agents (its routing) is not decided yet;
these tools do not depend on that decision.
"""

# no `from __future__ import annotations`: xmas builds the tool schemas from the hints
from collections import Counter

from tabular_preprocessing.inner_loop.tools.text import bullets
from tabular_preprocessing.inner_loop.tools.workspace import Workspace
from xmas.tools import Tool, tool


def orchestration_tools(workspace: Workspace, group: str) -> list[Tool]:
    """Build the orchestration tools.

    Args:
        workspace: The run's workspace.
        group: The tool-name prefix (``orchestrator``).

    Returns:
        ``<group>_status`` and ``<group>_submit``.
    """

    def status() -> str:
        """The overview of the run: draft pipeline, changes per group, budget, identity score, submitted or not.

        Uses no evaluation of the budget.
        """
        by_group = Counter(edit.tool.split("_", 1)[0] for edit in workspace.edits)
        base = workspace.baseline()
        out = [
            "Draft pipeline:",
            workspace.pipeline.describe(),
            "Changes by group: " + (", ".join(f"{g}: {n}" for g, n in by_group.items()) or "none yet"),
            "Last changes:\n" + bullets([f"{e.tool} {e.action} {e.stage}: {e.step}" for e in workspace.edits[-8:]]),
            f"Identity pipeline score: {base.score:.4f} ({base.metric})" if base.score is not None else "",
            f"Evaluations used: {workspace.evaluations} of {workspace.max_evaluations}.",
            "Submitted." if workspace.submitted else "Not submitted yet.",
        ]
        return "\n".join(line for line in out if line)

    def submit(summary: str) -> str:
        """Submit the draft pipeline as the run's result; the run should end after this.

        Args:
            summary: Why this pipeline: what it does and how it scored, in a few sentences.
        """
        workspace.submitted = True
        workspace.summary = summary
        return "Submitted:\n" + workspace.pipeline.describe()

    fns = (status, submit)
    return [tool(name=f"{group}_{fn.__name__}")(fn) for fn in fns]
