#!/bin/bash
#SBATCH -J vgg_wo_lowerbound_balance
#SBATCH -p gpu-preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1

#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"

source /groups/ytian/lujing/miniconda3/etc/profile.d/conda.sh
conda activate avcil

cd /groups/ytian/lujing/lujing/Projects/AV-CIL_ICCV2023/experiments_phase_8_rdcrosssdc_modular_gridsearch

mkdir -p logs logs_hinge grid_commands

nvidia-smi

CMD_FILE="grid_commands/commands_hinge.txt"

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CMD_FILE}")

if [ -z "${CMD}" ]; then
  echo "Error: no command found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  echo "CMD_FILE=${CMD_FILE}"
  exit 1
fi

echo "Running command:"
echo "${CMD}"

eval "${CMD}"

echo "End time: $(date)"