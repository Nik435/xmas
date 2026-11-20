#!/bin/bash
# Start the whole grid81 study at once, from the login node in the project folder:
#   bash cluster/start_grid.sh
# Submits the 82 architecture jobs (81 + solo, at most 20 at a time), the baselines job alongside
# them, and the report job, which starts when all of them have ended. Running it again continues a
# stopped study: every job skips what is already done.
set -e
cd "$(dirname "$0")/.."
mkdir -p log
export RELBENCH_CACHE_DIR="$(ws_find tabulardata-mas)/relbench"
.venv/bin/python -m tabular_preprocessing.outer_loop configs/grid81.yaml --download   # quick when cached
ARCHS=$(sbatch --parsable cluster/grid_job.sh)
BASE=$(sbatch --parsable -a 0 -J mas-grid81-baselines cluster/grid_job.sh baselines)
REPORT=$(sbatch --parsable -a 0 -J mas-grid81-report --dependency=afterany:$ARCHS:$BASE cluster/grid_job.sh report)
echo "architectures: $ARCHS (82 jobs)   baselines: $BASE   report: $REPORT (after both)"
echo "watch:   squeue -u $USER"
echo "report:  log/mas-grid81-report.$REPORT.0.out"
