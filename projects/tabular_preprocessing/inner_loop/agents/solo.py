"""The single-agent baseline: one Generalist fills every slot with every tool.

Not part of the concept's search space (which drops a separate generalist: a group with only
an executor is that group's generalist). It answers a different question: does dividing the
work among agents help at all, compared with one agent that does everything?
"""

from tabular_preprocessing.inner_loop.agents.base import AgentType

GENERALIST = AgentType(
    key="generalist",
    title="Generalist",
    group="solo",
    role="E",
    task=(
        "build the whole preprocessing pipeline alone: clean the tables, merge database features into the "
        "label rows and engineer derived features, using only the library's functions"
    ),
    tool_sets=("filling", "profiling", "schema", "validation"),
    instructions=(
        "You fill the clean, merge and engineer slots yourself; your tools are prefixed solo_.\n"
        "1. Clean: solo_list_tables and solo_find_placeholders for the overview, solo_profile_table and "
        "solo_profile_column for details. Fix real problems only (placeholder values, numbers stored as text, "
        "noisy text columns) with clean steps; every clean step names its table. Keys and time columns are "
        "protected.\n"
        "2. Merge: solo_list_relations shows what the merge functions can reach from the entity table; "
        "solo_inspect_relation shows how many rows a label row sees per time window. Add entity_columns, "
        "lookups and aggregates (with time windows) that should predict the target. Merged columns are named "
        "<table>.<column> or <table>.<op>(<column>)[@<days>d].\n"
        "3. Engineer: derived features on the merged columns (ratios, logs of skewed counts, ...), by those "
        "names.\n"
        "Use solo_list_functions for each slot's functions, solo_add_step / solo_replace_step / "
        "solo_remove_step to edit, solo_show_pipeline to check.\n"
        "4. Check your work: solo_check_pipeline (contracts, feature budget), solo_feature_report (signal per "
        "feature, leakage flags), solo_score_pipeline and solo_score_steps (what your slots and each step add). "
        "Remove steps that break the pipeline, leak or do not help."
    ),
    answer="A short report: the steps you added per slot and why, and the score of your pipeline.",
)

SOLO_TYPES: dict[str, AgentType] = {"E": GENERALIST}
