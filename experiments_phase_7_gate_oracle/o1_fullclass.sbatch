#!/bin/bash
#SBATCH -J o1_gate_fullclass
#SBATCH -p gpu-preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=g-01-05,g-06-01,g-06-02,g-06-03,g-07-02,g-07-03,g-07-04,g-07-05,g-07-06,g-08-01,g-08-02,g-08-03
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"

BOOTSTRAP_DIR="${O1_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$BOOTSTRAP_DIR/o1_job_env.sh" ]]; then
  echo "Error: missing $BOOTSTRAP_DIR/o1_job_env.sh" >&2
  exit 1
fi
source "$BOOTSTRAP_DIR/o1_job_env.sh"

nvidia-smi

CMD_FILE="$O1_PROJECT_DIR/grid_commands/o1_fullclass_command.txt"
if [[ ! -f "$CMD_FILE" ]]; then
  echo "Error: command file not found: $CMD_FILE" >&2
  exit 1
fi

CMD="$(sed -n '1p' "$CMD_FILE")"
if [[ -z "$CMD" ]]; then
  echo "Error: no full-class command found in $CMD_FILE" >&2
  exit 1
fi

echo "Running full-class gate discovery command:"
echo "$CMD"
eval "$CMD"

if [[ ! -s "$O1_GATE_TABLE" ]]; then
  echo "Error: full-class job finished but gate table was not created: $O1_GATE_TABLE" >&2
  exit 1
fi

echo "Verified gate table: $O1_GATE_TABLE"
echo "End time: $(date)"