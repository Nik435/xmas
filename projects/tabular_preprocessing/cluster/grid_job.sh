#!/bin/bash
#SBATCH -p gki_cpu-cascadelake          # partition (queue); for tabpfn_v2 a GPU: -p ml_gpu-rtx2080 --gres=gpu:1
#SBATCH --mem 64000                    # memory (64 GB: rel-event is big)
#SBATCH -t 1-00:00                     # time (D-HH:MM)
#SBATCH -c 4                           # number of cores
#SBATCH -a 1-82%20                     # array: one architecture per job (start_grid.sh sets it per study)
#SBATCH --requeue                      # restart a job whose node fails; it continues where it stopped
#SBATCH -o log/%x.%A.%a.out            # STDOUT (the folder log must exist)
#SBATCH -e log/%x.%A.%a.err            # STDERR
#SBATCH -J mas-grid81                  # job name
# One part of a grid study ($CONFIG, configs/grid81.yaml by default). Start everything with cluster/start_grid.sh;
# it submits this script three ways:
#   (no argument)  array jobs 1-N: one architecture each, then its test scores (no LLM)
#   baselines      identity, entity_table and dfs on every task (no LLM), alongside the architectures
#   report         after all of them: merge, summary, effects, comparison
echo "Started at $(date) on $(hostname), job $SLURM_JOB_ID ($SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID) ${1:-architecture}"
nvidia-smi -L 2>/dev/null                                          # the GPU, if the job has one
export RELBENCH_CACHE_DIR="$(ws_find tabulardata-mas)/relbench"   # data in the workspace jehled-tabulardata-mas
set -a; source ../../.env; set +a                                  # XMAS_BASE_URL, XMAS_API_KEY, TABPFN_TOKEN (xmas/.env)
export TABPFN_NO_BROWSER=1                                       # a job cannot log in through a browser: TABPFN_TOKEN instead
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1                                        # on a crash (segfault), print where it happened
CONFIG="${CONFIG:-configs/grid81.yaml}"                           # set by cluster/start_grid.sh
SHARDS="${SHARDS:-82}"                                            # the study's architectures, likewise
RUN=".venv/bin/python -m tabular_preprocessing.outer_loop $CONFIG"
case "$1" in
    baselines)
        $RUN --baselines
        status=$?
        ;;
    report)
        $RUN --merge
        $RUN --summary
        $RUN --effects
        $RUN --compare
        status=$?
        ;;
    *)
        $RUN --shard $SLURM_ARRAY_TASK_ID/$SHARDS
        status=$?
        if [ $status -eq 0 ]; then                                 # the trial is complete: its test scores
            $RUN --shard $SLURM_ARRAY_TASK_ID/$SHARDS --final 1
            status=$?
        fi
        ;;
esac
echo "Finished at $(date) with exit code $status"
exit $status                                                       # so sacct shows FAILED when Python crashed
