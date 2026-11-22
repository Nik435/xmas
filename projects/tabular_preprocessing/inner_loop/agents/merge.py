"""Group Merge: Schema & Join Analyst (P) -> Feature Engineer (E) -> Leakage & Feature Validator (V).

Fills the merge and engineer slots (and sample, if the "Merge fills sample()" factor is on).
"""

from tabular_preprocessing.inner_loop.agents.base import AgentType

SCHEMA_JOIN_ANALYST = AgentType(
    key="schema_join_analyst",
    title="Schema & Join Analyst",
    group="merge",
    role="P",
    task="work out the keys, join paths, 1:n relations and timestamps, and turn them into a merge and feature plan",
    tool_sets=("schema",),
    instructions=(
        "Start with merge_list_relations: the entity table's own columns, the n:1 lookups, the 1:n tables, "
        "the 1:1 tables (child_columns), the tables two steps away (aggregate with via) and the entity's own "
        "past labels (target_history). For each promising 1:n table, merge_inspect_relation shows how many "
        "rows a label row sees, for the fitting rows and for the rows the run is finally scored on, which lie "
        "after the database's last rows; merge_time_overview relates the tables' times to the label times. "
        "Decide which columns to aggregate, with which operations (count, sum, mean, min, max, std, nunique, "
        "time_since_last, time_since_first for numbers and times; last, mode and count_by also for "
        "categories) and which rows to keep with where (e.g. only attended events), given what the target "
        "is about. Every aggregate uses all history before the label row's time. If the target is about "
        "the entity repeating its behaviour, plan target_history. "
        "Then name derived features worth building (ratios, differences, logs of skewed counts, date parts)."
    ),
    answer=(
        "A merge and feature plan, most promising first: per step the function, table, key and columns, and "
        "one line on why it should predict the target. Keep it within a few dozen feature columns."
    ),
)

FEATURE_ENGINEER = AgentType(
    key="feature_engineer",
    title="Feature Engineer",
    group="merge",
    role="E",
    task=(
        "write the merge and engineer slots: joins, aggregations, past labels and domain features, "
        "using only the library's functions"
    ),
    tool_sets=("filling",),
    instructions=(
        "Call merge_list_functions for the merge and the engineer slot. Add merge steps first "
        "(entity_columns, lookup, child_columns, aggregate, target_history), then engineer steps that build on "
        "their columns. Every change answers with the feature columns it added, by their exact names; "
        "merge_list_features lists all of them; engineer steps must use those names. Use merge_show_pipeline "
        "to check the result and merge_replace_step or merge_remove_step to correct it. If you are revising "
        "earlier work (a review or an objection is above), start with merge_show_pipeline: your earlier steps "
        "are still in the draft; change them with merge_replace_step or merge_remove_step instead of adding "
        "them again. Prefer a few strong features over many weak ones."
    ),
    answer=(
        "A short report: the steps you added (function, table, columns, why) and anything from the plan you "
        "did not do, with the reason."
    ),
    absent_duties={
        "P": (
            "Work out the plan yourself first: merge_list_relations for what is reachable from the entity "
            "table (also 1:1 tables, two steps through via, and past labels), merge_inspect_relation for how "
            "many rows a label row sees, also on the rows to predict, merge_time_overview for the times. Then "
            "fill the slots."
        ),
        "V": (
            "Check your own work before you answer: merge_check_pipeline (contracts, feature budget, empty "
            "features; its ERRORS are features empty on the rows to predict, remove them), merge_feature_report "
            "(signal per feature, leakage flags), merge_score_pipeline and merge_score_steps (what your slots "
            "and each step add). Remove steps that leak or do not help."
        ),
    },
)

LEAKAGE_FEATURE_VALIDATOR = AgentType(
    key="leakage_feature_validator",
    title="Leakage & Feature Validator",
    group="merge",
    role="V",
    task=(
        "check for leakage (time, target), join explosion and the column budget, measure the improvement on "
        "the validation score, and select features"
    ),
    tool_sets=("validation",),
    instructions=(
        "Run merge_check_pipeline first (joins keep one row per label row, feature count within budget, empty "
        "or constant features; its ERRORS list features that are empty on the rows the run is finally scored "
        "on: they must go, whatever the tool scores say). Then merge_feature_report: a feature with "
        "near-perfect signal may carry the target; a static column (copied from the entity table or a "
        "lookup) may hold a value that is only known after the label time; an aggregate over a table "
        "without time column includes future rows. "
        "Then merge_score_pipeline for what the merge and engineer slots add, and merge_score_steps to find "
        "steps that add nothing or hurt. Each score uses the run's evaluation budget."
    ),
    answer=(
        "Your verdict with evidence (the numbers from the tools): leakage found or not, contract problems, the "
        "score change, and the concrete fix for each problem (which step to remove or change, and how)."
    ),
)

MERGE_TYPES: dict[str, AgentType] = {
    t.role: t for t in (SCHEMA_JOIN_ANALYST, FEATURE_ENGINEER, LEAKAGE_FEATURE_VALIDATOR)
}
