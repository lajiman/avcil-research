#!/bin/bash
#SBATCH -J avcil_phase9
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=h200
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail
# [P9 新增] 每次 sbatch 申请 1 GPU、12 CPU、180G 主机内存，三个 seed 共享。
# 两个作业合计 360G，可同时放入 Juno 的 375G 双 H200 节点。
# 一个 Slurm step 内启动三个独立训练进程，继承同一个 CUDA_VISIBLE_DEVICES。
# Slurm 在脚本运行前打开输出文件，因此调度日志写入已存在的提交目录。
# 训练日志则在下方自动创建的 experiments_phase_9/logs 中。
# sbatch 会复制脚本到 spool，不能用 BASH_SOURCE 定位原仓库。
project_dir="${AVCIL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f "$project_dir/experiments_phase_9/run_shared_gpu.sh" ]]; then
    phase_dir="$project_dir/experiments_phase_9"
elif [[ -f "$project_dir/run_shared_gpu.sh" ]]; then
    phase_dir="$project_dir"
else
    echo "Submit from the repository root or experiments_phase_9, or set AVCIL_PROJECT_ROOT." >&2
    exit 1
fi
cd "$phase_dir"
mkdir -p logs
command_file="${1:?Pass a three-seed command file from grid_commands (periodic or uniform_cl_history)}"

# 独立初始化指定的 Conda，不依赖提交终端当前激活的环境或 .bashrc。
conda_root="${AVCIL_CONDA_ROOT:-/groups/ytian/lujing/miniconda3}"
conda_env="${AVCIL_CONDA_ENV:-$conda_root/envs/avcil-h200}"
if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $conda_root/etc/profile.d/conda.sh" >&2
    exit 1
fi
# Conda/环境激活钩子可能读取未定义变量；仅在激活期间关闭 nounset。
set +u
source "$conda_root/etc/profile.d/conda.sh"
conda activate "$conda_env"
set -u
printf 'Working directory: %s\nConda environment: %s\nPython: %s\n' \
    "$PWD" "$CONDA_PREFIX" "$(command -v python)"

# 不再启动三个独占 GPU 的 srun；单个 step 继承作业的 GPU 和 180G 总内存。
# CPU 绑定由 Slurm 保持在本作业分配范围内，三个子进程均可使用该 step 的 12 CPU。
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
    bash run_shared_gpu.sh "$command_file"
