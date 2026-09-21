#!/bin/bash
# Submit from the experiment directory after running `mkdir -p logs`.
# Uses the Juno H200 checkout and avcil-h200 environment.
# Each array element runs one experiment on one GPU.
# Submit: sbatch --array=1-81 run_exp.slurm
#SBATCH -J vgg_exp_h200
#SBATCH -p h200
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --array=1-81
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch as an array job}"

if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID must be a positive integer." >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"

source /groups/ytian/lujing/miniconda3/etc/profile.d/conda.sh
conda activate avcil-h200

cd /groups/ytian/lujing/lujing/Projects/AV-CIL_ICCV2023_share_h200/experiments_phase_8_rdcrosssdc_modular_gridsearch

mkdir -p logs logs_exp

nvidia-smi

CMD_FILE="grid_commands/commands_exp.txt"

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CMD_FILE}")

if [[ "${CMD}" =~ ^[[:space:]]*$ || "${CMD}" =~ ^[[:space:]]*# ]]; then
  echo "Error: no command found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  echo "CMD_FILE=${CMD_FILE}"
  exit 1
fi

echo "Running command:"
echo "${CMD}"

eval "${CMD}"

echo "End time: $(date)"
