"""The toolbox: every tool set of one run, built once per group.

Agent types name the tool sets they need; the toolbox builds each (group, set)
pair once for the run's workspace, so two agents of one group share the same
tool objects and the system offers every tool name exactly once.
"""

from __future__ import annotations

from collections.abc import Callable

from tabular_preprocessing.inner_loop.tools.filling import filling_tools
from tabular_preprocessing.inner_loop.tools.orchestration import orchestration_tools
from tabular_preprocessing.inner_loop.tools.profiling import profiling_tools
from tabular_preprocessing.inner_loop.tools.schema import schema_tools
from tabular_preprocessing.inner_loop.tools.validation import validation_tools
from tabular_preprocessing.inner_loop.tools.workspace import Workspace
from xmas.tools import Tool

# tool set name -> builder (workspace, group) -> tools
TOOL_SETS: dict[str, Callable[[Workspace, str], list[Tool]]] = {
    "filling": filling_tools,
    "profiling": profiling_tools,
    "schema": schema_tools,
    "validation": validation_tools,
    "orchestration": orchestration_tools,
}


class Toolbox:
    """The tool sets of one run, built on demand."""

    def __init__(self, workspace: Workspace) -> None:
        """Initialize an empty toolbox.

        Args:
            workspace: The run's workspace every tool set is built for.
        """
        self.workspace = workspace
        self._built: dict[tuple[str, str], list[Tool]] = {}

    def get(self, group: str, name: str) -> list[Tool]:
        """One group's tools of one set, built the first time they are asked for.

        Raises:
            KeyError: If the tool set does not exist (yet).
        """
        if name not in TOOL_SETS:
            raise KeyError(f"Tool set {name!r} does not exist; available: {sorted(TOOL_SETS)}.")
        key = (group, name)
        if key not in self._built:
            self._built[key] = TOOL_SETS[name](self.workspace, group)
        return self._built[key]

    def tools(self) -> list[Tool]:
        """Every tool built so far, for ``MAS(tools=...)``."""
        return [t for tools in self._built.values() for t in tools]
