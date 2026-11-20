"""Agent types, one module per group, as in the project concept.

| Group | Role | Agent type | Module |
|---|---|---|---|
| clean | P | Data Profiler | :mod:`.clean` |
| clean | E | Cleaning Coder | :mod:`.clean` |
| clean | V | Cleaning Validator | :mod:`.clean` |
| merge | P | Schema & Join Analyst | :mod:`.merge` |
| merge | E | Feature Engineer | :mod:`.merge` |
| merge | V | Leakage & Feature Validator | :mod:`.merge` |
| C | - | Orchestrator | :mod:`.orchestrator` |

A type becomes an :class:`xmas.Agent` with :meth:`AgentType.build` for a concrete
:class:`Team`; wiring agents into the 81 architectures is the next step.
"""

from tabular_preprocessing.inner_loop.agents.base import AgentType, MissingToolSetWarning, Team
from tabular_preprocessing.inner_loop.agents.clean import CLEAN_TYPES, CLEANING_CODER, CLEANING_VALIDATOR, DATA_PROFILER
from tabular_preprocessing.inner_loop.agents.merge import (
    FEATURE_ENGINEER,
    LEAKAGE_FEATURE_VALIDATOR,
    MERGE_TYPES,
    SCHEMA_JOIN_ANALYST,
)
from tabular_preprocessing.inner_loop.agents.orchestrator import ORCHESTRATOR
from tabular_preprocessing.inner_loop.agents.solo import GENERALIST, SOLO_TYPES

# group -> role -> agent type
AGENT_TYPES: dict[str, dict[str, AgentType]] = {
    "clean": CLEAN_TYPES,
    "merge": MERGE_TYPES,
    "orchestrator": {"C": ORCHESTRATOR},
    "solo": SOLO_TYPES,  # the single-agent baseline, outside the concept's search space
}

__all__ = [
    "AGENT_TYPES",
    "CLEANING_CODER",
    "CLEANING_VALIDATOR",
    "DATA_PROFILER",
    "FEATURE_ENGINEER",
    "GENERALIST",
    "LEAKAGE_FEATURE_VALIDATOR",
    "ORCHESTRATOR",
    "SCHEMA_JOIN_ANALYST",
    "AgentType",
    "MissingToolSetWarning",
    "Team",
]
