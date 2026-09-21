#!/bin/bash
# Submit from the experiment directory after running `mkdir -p logs`.
# Uses the Juno H200 checkout and avcil-h200 environment.
# Each array element runs up to 4 experiments concurrently on one GPU.
# Submit: sbatch --array=1-21 run_log1p_multi.slurm
# Adjust RUNS_PER_GPU to available memory; array size = ceil(81 / RUNS_PER_GPU).
#SBATCH -J vgg_log1p_h200_multi
#SBATCH -p h200
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --array=1-21
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

set -euo pipefail

# Changing this also requires updating the array range at submission.
RUNS_PER_GPU=4

: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch as an array job}"

if [[ ! "${RUNS_PER_GPU}" =~ ^[1-9][0-9]*$ || ! "${SLURM_ARRAY_TASK_ID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: RUNS_PER_GPU and SLURM_ARRAY_TASK_ID must be positive integers." >&2
  exit 1
fi
: "${SLURM_CPUS_PER_TASK:?Submit with --cpus-per-task set}"
CPUS_PER_RUN=$(( SLURM_CPUS_PER_TASK / RUNS_PER_GPU ))
if (( CPUS_PER_RUN < 1 )); then
  echo "Error: request at least RUNS_PER_GPU CPU cores." >&2
  exit 1
fi
export OMP_NUM_THREADS=${CPUS_PER_RUN}
export MKL_NUM_THREADS=${CPUS_PER_RUN}
export OPENBLAS_NUM_THREADS=${CPUS_PER_RUN}
export NUMEXPR_NUM_THREADS=${CPUS_PER_RUN}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"

source /groups/ytian/lujing/miniconda3/etc/profile.d/conda.sh
conda activate avcil-h200

cd /groups/ytian/lujing/lujing/Projects/AV-CIL_ICCV2023_share_h200/experiments_phase_8_rdcrosssdc_modular_gridsearch

mkdir -p logs logs_log1p

nvidia-smi

CMD_FILE="grid_commands/commands_log1p.txt"

# One array element owns one GPU; its child processes inherit the same
# CUDA_VISIBLE_DEVICES from Slurm. Do not assign a GPU to each child process.
TOTAL_COMMANDS=$(awk 'END { print NR + 0 }' "${CMD_FILE}")
TOTAL_GROUPS=$(( (TOTAL_COMMANDS + RUNS_PER_GPU - 1) / RUNS_PER_GPU ))
if (( SLURM_ARRAY_TASK_ID > TOTAL_GROUPS )); then
  echo "Error: array task ${SLURM_ARRAY_TASK_ID} exceeds ${TOTAL_GROUPS} command groups." >&2
  exit 1
fi

FIRST_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * RUNS_PER_GPU + 1 ))
LAST_LINE=$(( FIRST_LINE + RUNS_PER_GPU - 1 ))
if (( LAST_LINE > TOTAL_COMMANDS )); then
  LAST_LINE=${TOTAL_COMMANDS}
fi

# Validate the complete group before starting any training processes.
COMMANDS=()
for (( LINE=FIRST_LINE; LINE<=LAST_LINE; LINE++ )); do
  CMD=$(sed -n "${LINE}p" "${CMD_FILE}")
  if [[ "${CMD}" =~ ^[[:space:]]*$ || "${CMD}" =~ ^[[:space:]]*# ]]; then
    echo "Error: line ${LINE} in ${CMD_FILE} is empty or a comment." >&2
    exit 1
  fi
  COMMANDS+=("${CMD}")
done

echo "Running lines ${FIRST_LINE}-${LAST_LINE} on one GPU; ${CPUS_PER_RUN} CPU threads per process."
PIDS=()
for (( INDEX=0; INDEX<${#COMMANDS[@]}; INDEX++ )); do
  echo "Starting command line $(( FIRST_LINE + INDEX )): ${COMMANDS[INDEX]}"
  # Each command already redirects output to its own experiment log.
  bash -c "${COMMANDS[INDEX]}" &
  PIDS+=("$!")
done

# Keep the allocation until every child finishes; propagate any failure to Slurm.
STATUS=0
for (( INDEX=0; INDEX<${#PIDS[@]}; INDEX++ )); do
  if wait "${PIDS[INDEX]}"; then
    echo "Command line $(( FIRST_LINE + INDEX )) completed."
  else
    CHILD_STATUS=$?
    echo "Command line $(( FIRST_LINE + INDEX )) failed with exit code ${CHILD_STATUS}." >&2
    STATUS=1
  fi
done

echo "End time: $(date)"
exit "${STATUS}"
