#!/bin/bash
# Start the whole grid81 study at once, from the login node in the project folder:
#   bash cluster/start_grid.sh
# Another grid study (any number of architectures; TabPFN gets 16 cores instead of 4):
#   CONFIG=configs/grid81_v2.yaml bash cluster/start_grid.sh
#   CONFIG=configs/top10_tabpfn.yaml bash cluster/start_grid.sh
# On a GPU instead of the CPU partition (faster, but only 8-11 GB of GPU memory):
#   CONFIG=configs/top10_tabpfn.yaml GPU_PARTITION=ml_gpu-rtx2080 bash cluster/start_grid.sh
# Submits one job per architecture (at most 20 at a time), the baselines job alongside
# them, and the report job, which starts when all of them have ended. Running it again continues a
# stopped study: every job skips what is already done.
set -e
cd "$(dirname "$0")/.."
mkdir -p log
export CONFIG="${CONFIG:-configs/grid81.yaml}"   # passed on to every job (sbatch exports the environment)
NAME="$(basename "$CONFIG" .yaml)"
export RELBENCH_CACHE_DIR="$(ws_find tabulardata-mas)/relbench"
set -a; source ../../.env; set +a   # TABPFN_TOKEN, for downloading gated TabPFN weights here (v2 needs none)
.venv/bin/python -m tabular_preprocessing.outer_loop "$CONFIG" --download   # quick when cached
# the number of architectures and the predictor, from the config
read -r SHARDS PREDICTOR < <(.venv/bin/python -c "
from tabular_preprocessing.outer_loop.config import ExperimentConfig
from tabular_preprocessing.outer_loop.study import n_trials_of
c = ExperimentConfig.from_yaml('$CONFIG'); print(n_trials_of(c), c.predictor)")
export SHARDS
RES=()                                                          # gradient boosting: the script's 4 cores
if [[ -n "$GPU_PARTITION" ]]; then
    RES=(-p "$GPU_PARTITION" --gres=gpu:1)
elif [[ "$PREDICTOR" == tabpfn* ]]; then                       # TabPFN on the CPU: it scales with cores
    RES=(-c 16)
fi
ARCHS=$(sbatch --parsable -a "1-$SHARDS%20" "${RES[@]}" -J "mas-$NAME" cluster/grid_job.sh)
BASE=$(sbatch --parsable -a 0 "${RES[@]}" -J "mas-$NAME-baselines" cluster/grid_job.sh baselines)
REPORT=$(sbatch --parsable -a 0 -J "mas-$NAME-report" --dependency=afterany:$ARCHS:$BASE cluster/grid_job.sh report)
echo "$CONFIG ($PREDICTOR ${RES[*]}): architectures $ARCHS ($SHARDS jobs)   baselines: $BASE   report: $REPORT (after both)"
echo "watch:   squeue -u $USER"
echo "report:  log/mas-$NAME-report.$REPORT.0.out"
