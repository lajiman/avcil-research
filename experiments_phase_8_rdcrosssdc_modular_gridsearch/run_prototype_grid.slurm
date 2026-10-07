#!/bin/bash
# Adapted from run_exp_multi.slurm and the existing Juno shared-GPU protocol.
# One submission owns one GPU; B and C are submitted as two independent jobs.
#SBATCH -J avcil_proto_BC
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=h200
#SBATCH --reservation=xie
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
set -euo pipefail

project_dir="${AVCIL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
phase_name="experiments_phase_8_rdcrosssdc_modular_gridsearch"
if [[ -f "$project_dir/$phase_name/run_prototype_grid.py" ]]; then
    phase_dir="$project_dir/$phase_name"
elif [[ -f "$project_dir/run_prototype_grid.py" ]]; then
    phase_dir="$project_dir"
else
    echo "Submit from the repository root or $phase_name, or set AVCIL_PROJECT_ROOT." >&2
    exit 1
fi
cd "$phase_dir"
command_file="${1:?Pass grid_commands/commands_prototype_B_s3_s4.txt or commands_prototype_C_s3_s4.txt}"
shift

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

total_cpus="${SLURM_CPUS_PER_TASK:-12}"
if [[ ! "$total_cpus" =~ ^[0-9]+$ ]] || (( total_cpus < 3 )); then
    echo "At least 3 allocated CPUs are required for three concurrent runs." >&2
    exit 1
fi
threads_per_run=$(( total_cpus / 3 ))
extra=()
# Optional array: each element owns one setting/tolerance's three seeds.
# --array=1-6%1 keeps B (or C) at one GPU while giving each group its own time limit.
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && extra+=(--group "$SLURM_ARRAY_TASK_ID")
[[ -n "${AVCIL_FEATURE_ROOT:-}" ]] && extra+=(--feature-root "$AVCIL_FEATURE_ROOT")
[[ -n "${AVCIL_META_ROOT:-}" ]] && extra+=(--meta-root "$AVCIL_META_ROOT")
printf 'Working directory: %s\nConda environment: %s\nPython: %s\n' \
    "$PWD" "$CONDA_PREFIX" "$(command -v python)"
printf 'One allocated GPU, 180G host memory by default; CPU threads per run: %s\n' "$threads_per_run"

# A single step owns the allocated GPU. All three Python children inherit its
# exact CUDA_VISIBLE_DEVICES. No three exclusive GPU srun steps or manual IDs.
srun --nodes=1 --ntasks=1 --cpus-per-task="$total_cpus" \
    python -u run_prototype_grid.py --commands "$command_file" \
    --threads-per-run "$threads_per_run" "${extra[@]}" "$@"
