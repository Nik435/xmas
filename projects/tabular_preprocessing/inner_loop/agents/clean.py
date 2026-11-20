"""Group Clean: Data Profiler (P) -> Cleaning Coder (E) -> Cleaning Validator (V). Fills the clean slot."""

from tabular_preprocessing.inner_loop.agents.base import AgentType

DATA_PROFILER = AgentType(
    key="data_profiler",
    title="Data Profiler",
    group="clean",
    role="P",
    task=(
        "profile every table (types, missing values, placeholders, cardinality, duplicates, string columns, "
        "target distribution) and turn the findings into a cleaning plan"
    ),
    tool_sets=("profiling",),
    instructions=(
        "Start with clean_list_tables and clean_find_placeholders for the overview, then clean_profile_table "
        "for the tables that matter for the task, and clean_profile_column where a column needs a closer look "
        "(is 0 or -1 a real value or a placeholder?). clean_target_distribution shows the target. Look for: "
        "numbers stored as text, placeholder values that mean 'missing', columns with many missing values, "
        "duplicated rows, identifier-like or free-text columns that only add noise, constant columns.\n"
        "Keys and time columns are protected: cleaning never changes them, so leave them out of the plan."
    ),
    answer=(
        "A cleaning plan: per table, the problems you found with evidence (column, value, share of rows) and "
        "the fix for each, in the order to apply them. Name the library function for each fix where you can. "
        "Say explicitly which tables need no cleaning."
    ),
)

CLEANING_CODER = AgentType(
    key="cleaning_coder",
    title="Cleaning Coder",
    group="clean",
    role="E",
    task=(
        "write the clean slot: placeholders to missing values, types, corrupted rows, string columns, "
        "using only the library's clean functions"
    ),
    tool_sets=("filling",),
    instructions=(
        "Call clean_list_functions to see the clean functions and their parameters. Add one step per fix with "
        "clean_add_step; every clean step names its table. Steps of one table run in the order you add them, "
        "so convert types before steps that compare numbers. Use clean_show_pipeline to check the result and "
        "clean_replace_step or clean_remove_step to correct it.\n"
        "Keys and time columns are hidden from the clean functions and cannot be changed. Do not clean what is "
        "fine: every step must fix a real problem."
    ),
    answer=(
        "A short report: the steps you added (table, function, why) and anything from the plan you did not do, "
        "with the reason."
    ),
    absent_duties={
        "P": (
            "Work out the cleaning plan yourself first: clean_list_tables and clean_find_placeholders for the "
            "overview, clean_profile_table and clean_profile_column for the details (wrong types, placeholder "
            "values, missing values, duplicates, noisy text columns). Then fill the slot. With "
            "after_cleaning=true, the profiling tools show what your steps changed."
        ),
        "V": (
            "Check your own work before you answer: clean_check_pipeline (does every step apply, how many rows "
            "it removes), clean_score_pipeline (the score with and without your clean steps) and "
            "clean_score_steps (what each step adds). Remove steps that break a contract or do not help."
        ),
    },
)

CLEANING_VALIDATOR = AgentType(
    key="cleaning_validator",
    title="Cleaning Validator",
    group="clean",
    role="V",
    task=(
        "check the clean slot against its contracts (schema intact, row loss, train and test treated alike) "
        "and its effect on the validation score"
    ),
    tool_sets=("validation",),
    instructions=(
        "Run clean_check_pipeline first: every step must apply to an existing table and column, and no step may "
        "remove a large share of a table's rows (train and validation rows are cleaned by the same steps; the "
        "harness guarantees it). Then clean_score_pipeline: does the clean slot improve the validation score "
        "over leaving the data as it is? With clean_score_steps, find steps that do not help. Each score uses "
        "the run's evaluation budget, so do not repeat a measurement."
    ),
    answer=(
        "Your verdict with evidence (the numbers from the tools): which steps hold, which break a contract or "
        "cost score, and the concrete fix for each (remove, replace with what)."
    ),
)

CLEAN_TYPES: dict[str, AgentType] = {t.role: t for t in (DATA_PROFILER, CLEANING_CODER, CLEANING_VALIDATOR)}
