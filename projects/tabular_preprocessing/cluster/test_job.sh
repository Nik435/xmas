#!/bin/bash
#SBATCH -p gki_cpu-cascadelake          # partition (queue)
#SBATCH --mem 8000                     # memory (8 GB)
#SBATCH -t 0-01:00                     # time (D-HH:MM)
#SBATCH -c 4                           # number of cores
#SBATCH -o log/%x.%N.%j.out            # STDOUT (the folder log must exist)
#SBATCH -e log/%x.%N.%j.err            # STDERR
#SBATCH -J mas-test-job                # job name

# Test job: one real run of the smoke test (configs/smoke.yaml, or the config given).
# Submit from the project folder:   mkdir -p log && sbatch cluster/test_job.sh
# With TabPFN on a GPU:   sbatch -p ml_gpu-rtx2080 --gres=gpu:1 cluster/test_job.sh configs/smoke_tabpfn.yaml

echo "Started at $(date) on $(hostname), job $SLURM_JOB_ID"

export RELBENCH_CACHE_DIR="$(ws_find tabulardata-mas)/relbench"   # data in the workspace jehled-tabulardata-mas
set -a; source ../../.env; set +a                                  # XMAS_BASE_URL, XMAS_API_KEY, TABPFN_TOKEN (xmas/.env)
export TABPFN_NO_BROWSER=1                                       # a job cannot log in through a browser: TABPFN_TOKEN instead
export PYTHONUNBUFFERED=1

CONFIG=${1:-configs/smoke.yaml}
nvidia-smi -L 2>/dev/null                                          # the GPU, if the job has one
.venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG
.venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG --summary

echo "Finished at $(date)"
