#!/bin/bash
#SBATCH --job-name=avcil_phase9_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h200
#SBATCH --reservation=xie
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail
# [P9 分析] 一个作业占用预留节点的两张 GPU，每卡一个长期推理进程。
# 训练输出始终只读；CPU 汇总、checkpoint 推理、最终报告按顺序自动完成。
project_dir="${AVCIL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f "$project_dir/experiments_phase_9/run_analysis.py" ]]; then
    phase_dir="$project_dir/experiments_phase_9"
elif [[ -f "$project_dir/run_analysis.py" ]]; then
    phase_dir="$project_dir"
else
    echo "Submit from the repository root or experiments_phase_9, or set AVCIL_PROJECT_ROOT." >&2
    exit 1
fi
cd "$phase_dir"

conda_root="${AVCIL_CONDA_ROOT:-/groups/ytian/lujing/miniconda3}"
conda_env="${AVCIL_CONDA_ENV:-$conda_root/envs/avcil-h200}"
if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $conda_root/etc/profile.d/conda.sh" >&2
    exit 1
fi
set +u
source "$conda_root/etc/profile.d/conda.sh"
conda activate "$conda_env"
set -u

threads=$(( ${SLURM_CPUS_PER_TASK:-12} / 2 ))
if (( threads < 1 )); then threads=1; fi
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
export PYTHONUNBUFFERED=1
extra=()
if [[ -n "${AVCIL_FEATURE_ROOT:-}" ]]; then extra+=(--feature-root "$AVCIL_FEATURE_ROOT"); fi
if [[ -n "${AVCIL_META_ROOT:-}" ]]; then extra+=(--meta-root "$AVCIL_META_ROOT"); fi
results_dir="$phase_dir/results/analysis_${SLURM_JOB_ID:-manual}"
printf 'Phase directory: %s\nPython: %s\nDefault results: %s\n' "$phase_dir" "$(command -v python)" "$results_dir"

# 一个 Slurm step 继承两张可见 GPU；Python 内部使用 cuda:0 和 cuda:1。
# 不写物理 GPU 编号，不开启 DataParallel，不在脚本中重新训练。
# "$@" 允许覆盖默认结果目录，以便中断后复用已验证的推理缓存。
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
    python -u run_analysis.py --mode all --device cuda --gpus 2 \
    --cpu-threads "$threads" --results-dir "$results_dir" "${extra[@]}" "$@"
