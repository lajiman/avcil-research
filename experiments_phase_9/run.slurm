#!/bin/bash
#SBATCH -J avcil_phase9
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail
# [P9 新增] 每次 sbatch 申请 1 GPU、12 CPU、300G 主机内存，三个 seed 共享。
# 一个 Slurm step 内启动三个独立训练进程，继承同一个 CUDA_VISIBLE_DEVICES。
# Slurm 的 --mem 不是 GPU 显存。先激活 Python 环境，并在提交前创建 logs。
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from experiments_phase_9}"
mkdir -p logs
command_file="${1:?Pass grid_commands/commands_periodic_fixed.txt or commands_periodic_sample_aware.txt}"
# 不再启动三个独占 GPU 的 srun；单个 step 继承作业的 GPU 和 300G 总内存。
# CPU 绑定由 Slurm 保持在本作业分配范围内，三个子进程均可使用该 step 的 12 CPU。
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
    bash run_shared_gpu.sh "$command_file"
