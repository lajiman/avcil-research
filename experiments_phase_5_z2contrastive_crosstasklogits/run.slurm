#!/bin/bash
#SBATCH -J vgg_z2_contrastive_alpha1p0_balance
#SBATCH -p gpu-preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=g-01-05,g-06-01,g-06-02,g-06-03,g-07-02,g-07-03,g-07-04,g-07-05,g-07-06,g-08-01,g-08-02,g-08-03
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

source /scratch/ganymede2/dal400642/miniconda3/etc/profile.d/conda.sh
conda activate avcil

cd /scratch/ganymede2/dal400642/Projects/AV-CIL_ICCV2023/experiments_phase_5_z2contrastive_crosstasklogits

mkdir -p logs grid_commands

nvidia-smi

CMD_FILE="grid_commands/z2_contrastive_balance_commands.txt"

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