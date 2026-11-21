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
        "2. Merge: solo_list_relations shows what the merge functions can reach from the entity table (also "
        "1:1 tables, tables two steps away through via, and the entity's past labels); solo_inspect_relation "
        "shows how many rows a label row sees, also on the rows the run is finally scored on. Add "
        "entity_columns, lookups, child_columns, aggregates (over all history; where, mode and count_by for "
        "categories) and target_history where they should predict the target.\n"
        "3. Engineer: derived features on the merged columns (ratios, logs of skewed counts, ...). Every change "
        "answers with the columns it added by their exact names; solo_list_features lists all of them.\n"
        "Use solo_list_functions for each slot's functions, solo_add_step / solo_replace_step / "
        "solo_remove_step to edit, solo_show_pipeline to check.\n"
        "4. Check your work: solo_check_pipeline (contracts, feature budget; its ERRORS are features empty on "
        "the rows to predict, remove them), solo_feature_report (signal per "
        "feature, leakage flags), solo_score_pipeline and solo_score_steps (what your slots and each step add). "
        "Remove steps that break the pipeline, leak or do not help."
    ),
    answer="A short report: the steps you added per slot and why, and the score of your pipeline.",
)

SOLO_TYPES: dict[str, AgentType] = {"E": GENERALIST}
