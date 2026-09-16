#!/bin/bash
#SBATCH -J o1_gate_cil
#SBATCH -p gpu-preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
#SBATCH --exclude=g-01-05,g-06-01,g-06-02,g-06-03,g-07-02,g-07-03,g-07-04,g-07-05,g-07-06,g-08-01,g-08-02,g-08-03
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
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

if [[ ! -s "$O1_GATE_TABLE" ]]; then
  echo "Error: required gate table is missing or empty: $O1_GATE_TABLE" >&2
  exit 1
fi

CMD_FILE="$O1_PROJECT_DIR/grid_commands/o1_cil_commands.txt"
if [[ ! -f "$CMD_FILE" ]]; then
  echo "Error: command file not found: $CMD_FILE" >&2
  exit 1
fi

CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMD_FILE")"
if [[ -z "$CMD" ]]; then
  echo "Error: no command found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  echo "CMD_FILE=$CMD_FILE" >&2
  exit 1
fi

echo "Running CIL oracle command:"
echo "$CMD"
eval "$CMD"

echo "End time: $(date)"