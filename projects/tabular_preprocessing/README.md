# MAS architecture search for a TabFM-Auto-style pipeline

A multi-agent system (built with xmas) fills a fixed relational pipeline, modelled on
[TabFM-Auto](https://arxiv.org/abs/2609.37989), with **fixed functions** from a library.
Groups Clean and Merge (Planner → Executor → Validator) are the architecture under search.
This folder currently holds the pipeline, the function library, the harness, the agent types,
all five tool sets, the 81 architectures, the inner loop, and a configuration-driven study runner.

## Layout

This folder is the package itself (`import tabular_preprocessing`), with no `src/` folder.
`inner_loop/` holds everything one run needs. `outer_loop/` holds the search over architectures
and hyperparameters.

```
tabular_preprocessing/
├── inner_loop/          # one architecture, one task: fill -> tables -> predictor -> score
│   ├── loop.py          # run_inner_loop: all of it for one run
│   ├── data.py          # RelationalTask, Table, load_relbench_task
│   ├── library.py       # the fixed functions per slot
│   ├── pipeline.py      # Pipeline and Call: the pipeline as plain data
│   ├── harness.py       # build_tables -> fit_predict -> score, around the fixed predictor
│   ├── architecture.py  # the search space as xmas systems
│   ├── orchestrated.py  # the orchestrator's loop: it decides after every turn who works next
│   ├── agents/          # agent types, one module per group
│   │   ├── base.py         # AgentType, Team
│   │   ├── clean.py        # Data Profiler, Cleaning Coder, Cleaning Validator
│   │   ├── merge.py        # Schema & Join Analyst, Feature Engineer, Leakage & Feature Validator
│   │   └── orchestrator.py
│   └── tools/           # tool sets for the agents, one module per set
│       ├── workspace.py      # Workspace: task, draft, edit log, evaluation budget
│       ├── toolbox.py        # builds each set once per group
│       ├── filling.py        # executors
│       ├── profiling.py      # Data Profiler
│       ├── schema.py         # Schema & Join Analyst
│       ├── validation.py     # validators, orchestrator
│       └── orchestration.py  # Orchestrator
├── outer_loop/          # an optimizer searches architectures + hyperparameters
│   ├── config.py        # the study: tasks, search space, optimizer, budgets (from YAML)
│   ├── optimizers.py    # swappable optimizers behind ask/tell: grid, random
│   ├── study.py         # the loop: ask -> run the inner loop on every task -> tell
│   ├── report.py        # final test evaluation, hyperparameter effects, export to stage 2, plan
│   └── cli.py           # the command line
├── configs/
│   ├── stage1.yaml      # stage 1: grid over the 81 systems, every further factor at its default
│   ├── random.yaml      # random search over architectures and the stage-2 factors
│   ├── smoke.yaml       # the cheapest real run: one architecture, one task
│   ├── smoke_multi.yaml # one architecture on the three rel-f1 tasks
│   ├── optimization_test.yaml # complete test: random, 20 trials x 20 RelArena tasks
│   ├── grid81.yaml      # all 81 architectures + solo, 3 repeats, 10 tasks
│   └── smoke_tabpfn.yaml # the smoke test with TabPFN v2: checks the GPU setup
├── cluster/             # SLURM: test_job.sh, optimization_job.sh, grid_job.sh + start_grid.sh (the grid81 study)
└── tests/
```

The inner loop's public names are re-exported at the top level, for example
`from tabular_preprocessing import Architecture, Pipeline, run_inner_loop`. Submodules are
under `tabular_preprocessing.inner_loop`, for example `tabular_preprocessing.inner_loop.tools`.

## The pipeline

```
clean (per table) -> merge -> engineer -> sample -> predictor (fixed) -> postprocess
```

| Slot | Filled by | Signature | Functions |
|---|---|---|---|
| `clean` (list) | Clean | `(table df) -> df`, call names its `table` | `zero_as_missing`, `placeholder_as_missing`, `to_numeric`, `missing_indicators`, `drop_columns`, `drop_duplicate_rows` |
| `merge` (list) | Merge | `(ctx, label rows) -> new columns` | `entity_columns`, `lookup` (n:1), `child_columns` (1:1), `aggregate` (1:n over all history; `where`, `count_by`, `mode`; two steps with `via`), `target_history` (the entity's past labels) |
| `engineer` (list) | Merge | `(X_train, y_train, X_test) -> (X_train, X_test)` | `combine`, `group_stat`, `count_encode`, `log_transform`, `datetime_parts`, `drop_features` |
| `sample` (one) | default (Merge, if that factor is on) | `(X_train, y_train, X_test, max_rows) -> [row indices]` | `full`, `random_views`, `recent`, `stratified_views`, `oversample_minority` |
| `postprocess` (list) | to be decided | `(pred, y_train, X_test) -> pred` | `prior_shift`, `temperature`, `clip_to_train_range` |
| `model_kwargs` | | dict | settings for the fixed predictor |

A pipeline is plain data, which is what the executors write:

```python
from tabular_preprocessing import Pipeline, evaluate, load_relbench_task

task = load_relbench_task("rel-f1", "driver-dnf")  # RelArena's inner split
print(task.describe())  # schema, keys, time columns: what the agents read
pipeline = Pipeline.from_dict(
    {
        "merge": [
            {"fn": "entity_columns", "columns": ["nationality"]},
            {
                "fn": "aggregate",
                "table": "results",
                "fk_column": "driverId",
                "aggregations": [{"op": "count"}, {"op": "mean", "column": "points"}],
            },
        ],
    }
)
baseline = evaluate(Pipeline.identity(), task)  # fit on train rows, score validation rows
result = evaluate(pipeline, task)
print(result.score, result.run.features, result.run.log)
```

`Pipeline.options()` lists every function with its parameters and description (the
menu for the agents).

## The outer loop: searching architectures and hyperparameters

```
for each trial:
    candidate = optimizer.ask()            # an architecture + factor combination
    for each task (x repeat):
        run_inner_loop(candidate, task)    # the candidate as an xmas system fills and scores a pipeline
    optimizer.tell(candidate, mean improvement over the identity pipeline)
```

```bash
uv run python -m tabular_preprocessing.outer_loop CONFIG --dry-run          # the plan and a cost estimate
uv run python -m tabular_preprocessing.outer_loop CONFIG --trials 1         # a first try
uv run python -m tabular_preprocessing.outer_loop CONFIG                    # the study (continues where it stopped)
uv run python -m tabular_preprocessing.outer_loop CONFIG --summary          # the trials, best first
uv run python -m tabular_preprocessing.outer_loop CONFIG --effects          # mean value per hyperparameter level
uv run python -m tabular_preprocessing.outer_loop CONFIG --final 5          # the best 5 on the held-out test split
uv run python -m tabular_preprocessing.outer_loop CONFIG --export-stage2 5 configs/stage2.yaml
```

### What makes the results valid

- **The agents never see the rows they are judged on.** Their tools (`score_pipeline`,
  `score_steps`, `feature_report`, ...) fit on the early training rows and score on the latest
  ones, a time split of the training rows. The run's score uses the validation rows, which no
  agent sees.
- **Comparable across tasks.** The default objective `mean_relative_improvement` takes
  (score − identity) / |identity| per task, then the mean over tasks, so ROC AUC tasks and MAE
  tasks weigh alike. `mean_improvement` (absolute) can still be selected.
- **Repeats give a standard error.** With `repeats: 2` or more, each trial reports
  `value_se`: per task the mean over repeats, the standard errors combined.
- **API errors are not results.** A run that fails on the API (timeouts, rate limits, 5xx) is
  retried (`run.retries`). If it still fails, it is logged in `infra_errors.jsonl` and its trial
  stays open for the next invocation. It is not scored as a failure of the candidate. After 3 such
  trials in a row the study stops; on 401/403 it stops at once.
- **Every pipeline edit is tried at once.** The draft with the change is built on a small sample
  (a dry run with the real harness). A change that would break the pipeline is refused with the
  reason: an unknown aggregation, a column Clean removed, a misspelled merged column, or removing
  a step a later step needs. So a run never ends with a pipeline that fails on the data.
- **Final numbers on the test split.** `--final K` re-applies the pipelines of the best K
  candidates to RelArena's outer split: the database frozen at the test cutoff, fit on train +
  validation rows, scored on the test rows, with labels from relbench. No agent runs again,
  since a pipeline is plain data. Only these numbers are comparable to the RelArena leaderboard.

### Run settings (`run:`)

| Key | Default | What it does |
|---|---|---|
| `max_train_rows` | none | keep only the most recent training label rows; needed for rel-stack (1.4M) and rel-hm (3.8M) |
| `retries` | 2 | retries of a run that failed on the API |
| `parallel` | 1 | runs of a trial at the same time (check your endpoint's rate limit) |
| `max_tokens`, `max_hours` | none | stop starting new trials once the study's runs used this much |

**The configuration is the search space.** A single value is fixed; a list gives the
choices; `{low, high}` gives a range (temperature only). Inactive hyperparameters follow the
concept's "active when": `role` only at size 2, `fb_E` only with a planner, `fb_V` only with a
validator, and with the orchestrator on no back edges and a `routing`.

**The optimizer can be swapped** with one line:

```yaml
optimizer:
  name: random      # grid | random
  n_trials: 30      # grid: every candidate if omitted
  seed: 0
objective: mean_relative_improvement   # per task (score - identity) / |identity|, then the mean; a failed run counts as 0
```

| Optimizer | What it does |
|---|---|
| `grid` | every candidate of a finite space, in order ([configs/stage1.yaml](configs/stage1.yaml)) |
| `random` | candidates drawn uniformly, never the same twice, seeded ([configs/random.yaml](configs/random.yaml)) |

Adding a method means one class in `outer_loop/optimizers.py` with `ask()` (the next candidate)
and `tell(candidate, value)`, registered in `OPTIMIZERS`. The study loop, resuming and the
records stay the same.

| Section | Key | Values (stage-1 default) | Wired to |
|---|---|---|---|
| `architecture.<group>` | `size`, `role`, `fb_E`, `fb_V` | grid; inactive values collapse, so the full grid gives 9 variants per group | `GroupArchitecture.grid` |
| `architecture` | `codes` | instead of the above: systems by code, e.g. `clean=PE-EP\|merge=PEV` | `Architecture.from_code` |
| `factors` | `orchestrator` | on / **off** | on: the orchestrator routes every turn (see below) |
| | `orchestrator_routing` | **ordered** / free | only active when on |
| | `temperature` | e.g. 0 / 0.5 / 1 (**0.25**); `null` sends none | the LLM's `temperature` |
| | `models` | `default` (**one model**), optional `P`, `E`, `V`, `C` | one LLM per model; each agent bound by role |
| | `tool_description` | brief / **detailed** / examples | the function menu (`list_functions`) |
| | `autonomy` | fixed / **bounded** / free | a prompt line for every agent, plus that level's `budgets` |
| | `order` | **clean_first** / merge_first | which group works first (decided by the pre-test) |
| | `merge_fills_sample` | yes / **no** | Merge also gets the sample slot (no oversampling) |
| `budgets.<level>` | `max_tool_rounds`, `max_loops`, `max_evaluations`, `max_steps` | per autonomy level | the inner loop's limits; `max_steps` = turns the orchestrator may delegate |
| `run` | `predictor`, `max_rows`, `max_columns`, `baseline` | | the fixed predictor and harness limits |

- **Results:**
  - `<output_dir>/trials.jsonl` has one line per trial: the candidate, its value, its runs.
  - `<output_dir>/results.jsonl` has one line per run (candidate × task × repeat): scores,
    hyperparameters, cost, pipeline.
  - `<output_dir>/runs/NNNNN.json` holds each run's details: edits, every agent's answer and
    tool calls.
- **Resuming:** an interrupted study continues where it stopped. Finished trials are replayed
  to the optimizer, and runs already in `results.jsonl` are not run again.
- **LLMs:** they come from `XMAS_BASE_URL` / `XMAS_API_KEY`. One LLM object is built per model
  and temperature and reused across runs.

## Running on a SLURM cluster

The study splits into **shards**: `--shard i/n` runs every n-th of the study's candidates, starting
at the i-th, into `<output_dir>/shards/i-of-n/`. Grid and random search plan their candidates
deterministically from the seed, so every job knows its share and no two jobs write to the same
file. `--merge` combines the shards into the study folder; `--summary`, `--effects` and `--final`
then cover the whole study.

```bash
# once, on the login node (from the project folder)
uv sync
export RELBENCH_CACHE_DIR="$(ws_find tabulardata-mas)/relbench"
.venv/bin/python -m tabular_preprocessing.outer_loop configs/optimization_test.yaml --download
mkdir -p log

sbatch cluster/test_job.sh                                          # smoke test, one run
sbatch cluster/optimization_job.sh                                  # array 1-20: one trial each
sbatch -a 0 --dependency=afterany:JOBID cluster/optimization_job.sh # merge, summary, effects, final

# all 81 architectures + solo + the baselines (configs/grid81.yaml), in one go: 82 architecture
# jobs (at most 20 at once), the baselines job alongside, the report job when all have ended
bash cluster/start_grid.sh

# the 10 best of grid81_v3 + solo on all 20 tasks with TabPFN v2, the real comparison with RDBLearn;
# start_grid.sh reads the number of architectures and gives TabPFN 16 CPU cores per job
CONFIG=configs/top10_tabpfn.yaml bash cluster/start_grid.sh
```

**Predictor.** `run.predictor` is the same for every system and every baseline:
`hist_gradient_boosting` (fast, CPU) or `tabpfn_v2` (TabPFN v2, RDBLearn's default, 10,000 rows
per fit; about 1 min per fit on a CPU, so it needs a GPU). `--download`
also fetches TabPFN's weights, for compute nodes without internet. TabPFN is under the Prior Labs
License.

```bash
sbatch -p ml_gpu-rtx2080 --gres=gpu:1 cluster/test_job.sh configs/smoke_tabpfn.yaml   # check the GPU setup
```

`--baselines` scores references without any agent on every task, on the validation and the test
split, **all with the study's predictor, training rows and sampling**, so only the features differ:

| baseline | features |
|---|---|
| `identity` | the task table only (the timestamp; keys stay hidden) |
| `entity_table` | plus every column of the entity's own row (RelBench's LightGBM recipe) |
| `dfs` | RDBLearn's features from RelArena: Deep Feature Synthesis at depth 2, past-label aggregates, time-until-cutoff diffs, entity key and cutoff calendar |

The single-agent `solo` runs as a normal candidate; `--effects` leaves it out. `--compare` shows per
task, on the test split, every baseline next to `mas_mean` (every architecture and repeat),
`mas_selected` (the architecture with the best search value) and `solo`. AUC is higher-is-better;
MAE is shown positive, lower-is-better, as on the RelArena leaderboard.

- **Continuing:** resubmitting the same configuration continues it. Every shard skips what it
  already finished, so time limits, preemption (`--requeue`) and API outages lose at most the runs
  in progress.
- **The LLM endpoint** must be reachable from the compute nodes. All jobs share its rate limit, so
  start with few jobs; the run lines in `log/` and `infra_errors.jsonl` show whether the endpoint
  keeps up.
- **Pinned versions:** xmas's `.gitignore` excludes `uv.lock`. To install the same versions on the
  cluster, commit this project's lock file with `git add -f uv.lock`.
- **Memory per job:** about 1 GB for rel-f1, 5 GB for rel-trial and rel-avito, 12 GB for rel-event.
  rel-stack and rel-hm need `run.max_train_rows`.

## The inner loop

One run of one architecture on one task:

```
1. fill     the architecture's system runs on the task and fills the pipeline through its tools
2. tables   the filled pipeline builds the predictor's train and test tables  (build_tables)
3. predict  the fixed predictor is fitted on the train table, predicts the test table  (fit_predict)
4. score    the predictions are scored, next to the identity pipeline  (score)
```

```python
from tabular_preprocessing import Architecture, run_inner_loop, load_relbench_task
from xmas import APILLM

task = load_relbench_task("rel-f1", "driver-dnf")
llm = APILLM(model="...")  # endpoint and key from XMAS_BASE_URL / XMAS_API_KEY
result = run_inner_loop(Architecture.from_code("clean=PE|merge=PEV-VE"), task, llm)
print(result.score, result.baseline, result.improvement, result.error)
print(result.pipeline.describe())  # what the system built
print([r.agent for r in result.trace.records], result.edits, result.tokens)
```

- **Train and test** are the task's train and validation rows (RelArena's inner split).
  The held-out test split is left for the final evaluation.
- **Each run gets a fresh workspace and system,** so nothing leaks between runs.
- **A failed run is a result, not a crash.** If the system errors or the pipeline does not
  fit the data, `error` says why and `score` is `None`. The baseline is still computed.
- **`max_loops`** (xmas's `max_visits`) bounds every back-edge loop, and
  **`max_tool_rounds`** bounds the tool calls per agent visit.

## Architectures

`Architecture.stage1()` gives the 81 systems. Each group is one of 9 variants, written as
codes: `E`, `PE`, `PE-EP`, `EV`, `EV-VE`, `PEV`, `PEV-EP`, `PEV-VE`, `PEV-EP-VE` (`EP` and `VE` are
the back edges). A system code reads `clean=PE-EP|merge=PEV-VE`, groups in working order.
The groups are joined by one edge: the last agent of the first group feeds the first agent
of the second. `order=("merge", "clean")` is the stage-2 order factor.

## The orchestrator

With `orchestrator: on`, the orchestrator is the hub. After every agent's turn it reads every
agent's full latest answer and the decisions so far, and answers with
`{"next": "<agent>", "instruction": "..."}` or `{"next": "submit", "summary": "..."}`. The chosen
agent works with its usual tools, plus the instruction as an extra prompt section, and its
answer goes back to the orchestrator.

- **Routing:** `ordered` (default) means no agent of the second group works before the first
  group's executor has worked once; going back later is allowed. `free` allows any agent at any time.
- **Budget:** at most `max_steps` agent turns. When they are used up, the draft counts as it
  is (`stopped: budget`).
- **Invalid decisions** (unknown agent, against the order, no JSON) go back to the orchestrator
  as an error. Three in a row end the run (`stopped: invalid`).
- **Back edges are off.** The orchestrator decides every next turn, and each agent sees the
  latest answers it would react to (the planner sees the executor's report, the executor the
  validator's review). So `fb_E`/`fb_V` collapse: there are 4 variants per group (E, PE, EV, PEV)
  and 16 orchestrated systems.
- **How it runs:** xmas has no edge whose target an agent chooses at run time, so
  `orchestrated.py` drives the loop itself. It uses the agents, tools and model bindings of an
  xmas `MAS` and xmas's tool loop, and returns an xmas `Trace` with one record per turn.
  `InnerLoopResult.orchestration` holds why the run stopped, the turns used and every decision.

## Agent types

`agents/` holds the seven agent types of the concept, one module per group:

| Group | Role | Agent type | Tool sets |
|---|---|---|---|
| clean | P | Data Profiler | profiling |
| clean | E | Cleaning Coder | filling |
| clean | V | Cleaning Validator | validation |
| merge | P | Schema & Join Analyst | schema |
| merge | E | Feature Engineer | filling |
| merge | V | Leakage & Feature Validator | validation |
| C | – | Orchestrator | orchestration, validation |

An agent type becomes an `xmas.Agent` for a concrete `Team`, which says who else is there:

```python
from tabular_preprocessing.inner_loop.agents import AGENT_TYPES, FEATURE_ENGINEER, Team
from tabular_preprocessing.inner_loop.tools import Toolbox, Workspace

toolbox = Toolbox(Workspace(task))
team = Team(frozenset({"P", "E", "V"}), fb_v=True, upstream="clean_review")
agent = FEATURE_ENGINEER.build(team, toolbox, AGENT_TYPES["merge"])
# then: MAS([...agents], edges=[...], llm=llm, tools=toolbox.tools())
```

- **Inputs follow the team.** The executor reads the plan only when there is a planner and
  the review only with a V→E back edge. The planner reads the executor's objection only with
  E→P. The second group reads the first group's result.
- **An executor takes over missing roles,** both their duties (an extra prompt section) and
  their tool sets. Only the division of labour varies, not the capabilities.
- **Back edges are taken by answering with `<continue>`,** which is xmas's marker; the prompt
  says so only where such an edge exists.
- **Names:** agents are `<group>_<planner|executor|validator>`, and they write
  `<group>_<plan|work|review>`.

## Tool sets for the agents

`tools/` holds one module per tool set. Every set is built for one run's `Workspace`: the
task, the draft pipeline all agents of the run share, the edit log and the evaluation
budget. Tool names carry the group as a prefix (`clean_add_step`, `merge_score_pipeline`), so
both groups' sets fit in one system. An executor without a planner or validator also gets
their sets.

| Set | For | Tools | What they give |
|---|---|---|---|
| `filling` | executors | `list_functions`, `show_pipeline`, `list_features` (merge/engineer groups), `add_step`, `replace_step`, `remove_step` | edit the group's own slots; every edit is checked against the library at once and answers with the feature columns it added or removed |
| `profiling` | Data Profiler | `list_tables`, `profile_table`, `profile_column`, `find_placeholders`, `target_distribution`, `sample_rows` | types, missing values, placeholder candidates, numbers stored as text, duplicates, the target; with `after_cleaning=true`, as the draft's clean steps leave them |
| `schema` | Schema & Join Analyst | `list_relations`, `inspect_relation`, `time_overview` | what entity_columns / lookup / child_columns / aggregate (also via) / target_history can reach, rows per entity, what the fitting rows and the rows to predict see before their time, time ranges |
| `validation` | validators, orchestrator | `check_pipeline`, `score_pipeline`, `score_steps`, `feature_report` | contracts without fitting (free); the score against identity and without the group's slots; leave-one-out per step; per-feature signal with leakage and drift flags |
| `orchestration` | Orchestrator | `status`, `submit` | the overview of the run, and submitting the pipeline |

**Evaluation budget.** Every score uses the same predictor and limits as the inner loop.
Scores are cached per pipeline, and each new pipeline counts against `max_evaluations`
(default 30, set through `run_inner_loop(..., max_evaluations=...)`). When the budget is used up,
new pipelines are refused and the agent is told so. Cached pipelines and the identity
baseline stay free. `InnerLoopResult.evaluations` records how many a run used.

## Contracts the harness enforces

Each one is a `PipelineError` naming the failing call, which a validator can hand back
to its executor:

- `clean` never sees a table's keys or time column, so it cannot break a join.
- `merge` returns exactly one row per label row: no lost rows, no join explosion. The
  same calls run on the train and the validation rows.
- Aggregates only use database rows **strictly before** each label row's own time.
- The feature count stays within `max_columns` (default 500).

The fixed predictor (`default_model`) is sklearn's `HistGradientBoosting*`, a stand-in until
the predictor is decided; swap it with `model_factory=...`.

## Development

```bash
uv sync
uv run pytest      # synthetic relational data, no download needed
uv run ruff check . && uv run ruff format --check .
```
