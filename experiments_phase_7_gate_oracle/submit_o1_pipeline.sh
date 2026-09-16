#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
cd "$SCRIPT_DIR"
mkdir -p logs grid_commands

# Optional local overrides.  This file is deliberately not required and may be
# kept outside version control after a machine/code move.
if [[ -f "$SCRIPT_DIR/o1_local_paths.env" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/o1_local_paths.env"
fi

export O1_PROJECT_DIR="${O1_PROJECT_DIR:-$SCRIPT_DIR}"
export O1_CONDA_ENV="${O1_CONDA_ENV:-avcil}"

# Export the current Conda installation when discoverable, so compute nodes do
# not depend on the former hard-coded Miniconda path.
if [[ -z "${O1_CONDA_SH:-}" && -n "${CONDA_EXE:-}" ]]; then
  _base="$(cd "$(dirname "$CONDA_EXE")/.." >/dev/null 2>&1 && pwd -P || true)"
  if [[ -f "$_base/etc/profile.d/conda.sh" ]]; then
    export O1_CONDA_SH="$_base/etc/profile.d/conda.sh"
  fi
fi
if [[ -z "${O1_CONDA_SH:-}" ]] && command -v conda >/dev/null 2>&1; then
  _base="$(conda info --base 2>/dev/null || true)"
  if [[ -f "$_base/etc/profile.d/conda.sh" ]]; then
    export O1_CONDA_SH="$_base/etc/profile.d/conda.sh"
  fi
fi

for required in \
  o1_job_env.sh \
  o1_fullclass.sbatch \
  o1_cil_array.sbatch \
  grid_commands/o1_fullclass_command.txt \
  grid_commands/o1_cil_commands.txt; do
  if [[ ! -f "$SCRIPT_DIR/$required" ]]; then
    echo "Missing $SCRIPT_DIR/$required" >&2
    exit 1
  fi
done

if [[ "${1:-}" == "--cil-only" ]]; then
  # Useful after moving an already-completed full-class gate table.
  source "$SCRIPT_DIR/o1_job_env.sh"
  if [[ ! -s "$O1_GATE_TABLE" ]]; then
    echo "Cannot submit --cil-only: missing $O1_GATE_TABLE" >&2
    exit 1
  fi
  CIL_ARRAY_JOB_ID="$(sbatch --parsable --export=ALL o1_cil_array.sbatch)"
  echo "Submitted CIL-only array job: $CIL_ARRAY_JOB_ID"
  echo "Monitor with: squeue -j $CIL_ARRAY_JOB_ID"
  exit 0
fi

FULLCLASS_JOB_ID="$(sbatch --parsable --export=ALL o1_fullclass.sbatch)"
echo "Submitted full-class gate discovery job: $FULLCLASS_JOB_ID"

CIL_ARRAY_JOB_ID="$(sbatch --parsable --export=ALL \
  --dependency=afterok:${FULLCLASS_JOB_ID} \
  o1_cil_array.sbatch)"
echo "Submitted dependent CIL array job: $CIL_ARRAY_JOB_ID"
echo "The three CIL tasks become eligible only after job $FULLCLASS_JOB_ID succeeds."
echo
echo "Monitor with:"
echo "  squeue -j ${FULLCLASS_JOB_ID},${CIL_ARRAY_JOB_ID}"