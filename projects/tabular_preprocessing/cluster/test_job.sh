#!/bin/bash
#SBATCH -p gki_cpu-cascadelake          # partition (queue)
#SBATCH --mem 8000                     # memory (8 GB)
#SBATCH -t 0-01:00                     # time (D-HH:MM)
#SBATCH -c 4                           # number of cores
#SBATCH -o log/%x.%N.%j.out            # STDOUT (the folder log must exist)
#SBATCH -e log/%x.%N.%j.err            # STDERR
#SBATCH -J mas-test-job                # job name

# Test job: one real run of the smoke test (configs/smoke.yaml).
# Submit from the project folder:   mkdir -p log && sbatch cluster/test_job.sh

echo "Started at $(date) on $(hostname), job $SLURM_JOB_ID"

export RELBENCH_CACHE_DIR="$(ws_find tabulardata-mas)/relbench"   # data in the workspace jehled-tabulardata-mas
set -a; source ../../.env; set +a                                  # XMAS_BASE_URL, XMAS_API_KEY (xmas/.env)
export PYTHONUNBUFFERED=1

.venv/bin/python -m tabular_preprocessing.outer_loop configs/smoke.yaml
.venv/bin/python -m tabular_preprocessing.outer_loop configs/smoke.yaml --summary

echo "Finished at $(date)"
