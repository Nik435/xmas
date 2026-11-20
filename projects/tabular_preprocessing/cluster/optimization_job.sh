#!/bin/bash
#SBATCH -p gki_cpu-cascadelake          # partition (queue)
#SBATCH --mem 64000                    # memory (64 GB: rel-event and rel-hm are big)
#SBATCH -t 1-00:00                     # time (D-HH:MM)
#SBATCH -c 4                           # number of cores
#SBATCH -a 1-20                        # array: one trial per job (20 trials)
#SBATCH -o log/%x.%A.%a.out            # STDOUT (the folder log must exist)
#SBATCH -e log/%x.%A.%a.err            # STDERR
#SBATCH -J mas-optimization            # job name
# Complete optimization test (configs/optimization_test.yaml): 20 trials x 20 tasks.
# Array jobs 1-20 each run one trial; job 0 combines the results afterwards.
# Submit from the project folder:
#   mkdir -p log && sbatch cluster/optimization_job.sh                             # prints the JOBID
#   sbatch -a 0 --dependency=afterany:JOBID cluster/optimization_job.sh            # report when all are done
echo "Started at $(date) on $(hostname), job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
export RELBENCH_CACHE_DIR="$(ws_find tabulardata-mas)/relbench"   # data in the workspace jehled-tabulardata-mas
set -a; source ../../.env; set +a                                  # XMAS_BASE_URL, XMAS_API_KEY (xmas/.env)
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1                                        # on a crash (segfault), print where it happened
CONFIG=configs/optimization_test.yaml
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    .venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG --merge
    .venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG --summary
    .venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG --effects
    .venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG --final 3
else
    .venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG --shard $SLURM_ARRAY_TASK_ID/20
fi
status=$?
echo "Finished at $(date) with exit code $status"
exit $status                                                       # so sacct shows FAILED when Python crashed
