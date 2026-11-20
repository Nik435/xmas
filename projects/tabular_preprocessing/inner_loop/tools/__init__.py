"""Tool sets for the agents, one module per set.

Each set is built for one run's :class:`Workspace` and returns plain
:class:`xmas.tools.Tool` objects, ready for ``MAS(tools=...)`` and ``Agent(tools=[names])``.
Tool names carry the group as prefix (``clean_add_step``, ``merge_score_pipeline``), so
several groups' sets fit in one system. The :class:`Toolbox` builds each set once per
group, as the agent types ask for them.

| Set | Module | For | Tools |
|---|---|---|---|
| filling | :mod:`.filling` | executors | list_functions, show_pipeline, add_step, replace_step, remove_step |
| profiling | :mod:`.profiling` | Data Profiler | list_tables, profile_table, profile_column, find_placeholders, |
|  |  |  | target_distribution, sample_rows |
| schema | :mod:`.schema` | Schema & Join Analyst | list_relations, inspect_relation, time_overview |
| validation | :mod:`.validation` | validators, orchestrator | check_pipeline, score_pipeline, score_steps, |
|  |  |  | feature_report |
| orchestration | :mod:`.orchestration` | Orchestrator | status, submit |

An executor without planner or validator gets their sets too (only the division of
labour varies, not the capabilities).
"""

from tabular_preprocessing.inner_loop.tools.filling import GROUP_SLOTS, filling_tools
from tabular_preprocessing.inner_loop.tools.orchestration import orchestration_tools
from tabular_preprocessing.inner_loop.tools.profiling import profiling_tools
from tabular_preprocessing.inner_loop.tools.schema import schema_tools
from tabular_preprocessing.inner_loop.tools.toolbox import TOOL_SETS, Toolbox
from tabular_preprocessing.inner_loop.tools.validation import validation_tools
from tabular_preprocessing.inner_loop.tools.workspace import BudgetExhaustedError, Edit, Scored, Workspace

__all__ = [
    "GROUP_SLOTS",
    "TOOL_SETS",
    "BudgetExhaustedError",
    "Edit",
    "Scored",
    "Toolbox",
    "Workspace",
    "filling_tools",
    "orchestration_tools",
    "profiling_tools",
    "schema_tools",
    "validation_tools",
]
