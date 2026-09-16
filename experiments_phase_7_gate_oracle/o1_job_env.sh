#!/bin/bash
# Shared runtime setup for both O1 Slurm jobs.
# This file is sourced inside the allocated job.

set -euo pipefail

_o1_abs_dir() {
  local p="$1"
  (cd "$p" >/dev/null 2>&1 && pwd -P)
}

_o1_first_existing_dir() {
  local p
  for p in "$@"; do
    [[ -n "$p" ]] || continue
    if [[ -d "$p" ]]; then
      _o1_abs_dir "$p"
      return 0
    fi
  done
  return 1
}

# SLURM_SUBMIT_DIR is reliable even though Slurm executes a copied spool script.
O1_PROJECT_DIR="${O1_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "$O1_PROJECT_DIR" ]]; then
  echo "Error: O1_PROJECT_DIR does not exist: $O1_PROJECT_DIR" >&2
  exit 1
fi
O1_PROJECT_DIR="$(_o1_abs_dir "$O1_PROJECT_DIR")"
export O1_PROJECT_DIR

# ---------------------------------------------------------------------------
# Conda activation.  A moved installation is accepted through O1_CONDA_SH.
# The launcher exports the currently discoverable Conda base automatically.
# ---------------------------------------------------------------------------
O1_CONDA_ENV="${O1_CONDA_ENV:-avcil}"
O1_CONDA_SH="${O1_CONDA_SH:-}"

if [[ -z "$O1_CONDA_SH" && -n "${CONDA_EXE:-}" ]]; then
  _conda_base="$(cd "$(dirname "$CONDA_EXE")/.." >/dev/null 2>&1 && pwd -P || true)"
  if [[ -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
    O1_CONDA_SH="${_conda_base}/etc/profile.d/conda.sh"
  fi
fi

if [[ -z "$O1_CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
    O1_CONDA_SH="${_conda_base}/etc/profile.d/conda.sh"
  fi
fi

if [[ -z "$O1_CONDA_SH" ]]; then
  for _candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/scratch/ganymede2/${USER}/miniconda3/etc/profile.d/conda.sh" \
    "/scratch/ganymede2/${USER}/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$_candidate" ]]; then
      O1_CONDA_SH="$_candidate"
      break
    fi
  done
fi

if [[ -z "$O1_CONDA_SH" || ! -f "$O1_CONDA_SH" ]]; then
  cat >&2 <<EOF
Error: unable to locate conda.sh.
Set it before submission, for example:
  export O1_CONDA_SH=/new/miniconda3/etc/profile.d/conda.sh
  export O1_CONDA_ENV=avcil
EOF
  exit 1
fi

source "$O1_CONDA_SH"
conda activate "$O1_CONDA_ENV"
export O1_CONDA_SH O1_CONDA_ENV

# ---------------------------------------------------------------------------
# Dataset paths.  Explicit O1_* variables always win.  Otherwise locate the
# common AV-CIL layouts relative to the moved experiment folder.
# ---------------------------------------------------------------------------
if [[ -z "${O1_FEATURE_ROOT:-}" ]]; then
  O1_FEATURE_ROOT="$(_o1_first_existing_dir \
    "$O1_PROJECT_DIR/../../../datasets/VGGSound" \
    "$O1_PROJECT_DIR/../../datasets/VGGSound" \
    "$O1_PROJECT_DIR/../datasets/VGGSound" \
    "/scratch/ganymede2/${USER}/datasets/VGGSound" \
    "/mnt/data2/wpian/dataset/VGGSound" \
    || true)"
fi

if [[ -z "${O1_META_EASY2HARD:-}" ]]; then
  O1_META_EASY2HARD="$(_o1_first_existing_dir \
    "$O1_PROJECT_DIR/../data_easy2hard/balance" \
    "$O1_PROJECT_DIR/data_easy2hard/balance" \
    "$O1_PROJECT_DIR/../../data_easy2hard/balance" \
    || true)"
fi

if [[ -z "${O1_META_HARD2EASY:-}" ]]; then
  O1_META_HARD2EASY="$(_o1_first_existing_dir \
    "$O1_PROJECT_DIR/../data_hard2easy/balance" \
    "$O1_PROJECT_DIR/data_hard2easy/balance" \
    "$O1_PROJECT_DIR/../../data_hard2easy/balance" \
    || true)"
fi

if [[ -z "${O1_META_DATA2:-}" ]]; then
  O1_META_DATA2="$(_o1_first_existing_dir \
    "$O1_PROJECT_DIR/../data2/balance" \
    "$O1_PROJECT_DIR/data2/balance" \
    "$O1_PROJECT_DIR/../../data2/balance" \
    || true)"
fi

for _var in O1_FEATURE_ROOT O1_META_EASY2HARD O1_META_HARD2EASY O1_META_DATA2; do
  _value="${!_var:-}"
  if [[ -z "$_value" || ! -d "$_value" ]]; then
    echo "Error: $_var is not set to an existing directory: ${_value:-<empty>}" >&2
    echo "Export $_var before running submit_o1_pipeline.sh." >&2
    exit 1
  fi
done

O1_GATE_TABLE="${O1_GATE_TABLE:-$O1_PROJECT_DIR/save/VGGSound_fullclass_trainaware_class_oracle_seed42/oracle_gate_table.json}"
export O1_FEATURE_ROOT O1_META_EASY2HARD O1_META_HARD2EASY O1_META_DATA2 O1_GATE_TABLE

cd "$O1_PROJECT_DIR"
mkdir -p logs grid_commands save

# Fail early if the moved folder is incomplete.  The original project has
# appeared in both self-contained and parent-package layouts, so accept either
# layout but require the base and oracle model to live in the same model package.
for _required in train_fullclass_oracle_gate.py train_incremental_oracle_class_gate.py; do
  if [[ ! -f "$O1_PROJECT_DIR/$_required" ]]; then
    echo "Error: required file missing after code move: $O1_PROJECT_DIR/$_required" >&2
    exit 1
  fi
done

_O1_PARENT_DIR="$(dirname "$O1_PROJECT_DIR")"
if [[ -f "$O1_PROJECT_DIR/model/audio_visual_model_incremental.py"    && -f "$O1_PROJECT_DIR/model/audio_visual_model_incremental_oracle_gate.py" ]]; then
  O1_PYTHON_ROOT="$O1_PROJECT_DIR"
elif [[ -f "$_O1_PARENT_DIR/model/audio_visual_model_incremental.py"      && -f "$_O1_PARENT_DIR/model/audio_visual_model_incremental_oracle_gate.py" ]]; then
  O1_PYTHON_ROOT="$_O1_PARENT_DIR"
else
  cat >&2 <<EOF
Error: the following two files must be placed in the same model/ directory:
  audio_visual_model_incremental.py
  audio_visual_model_incremental_oracle_gate.py
Checked:
  $O1_PROJECT_DIR/model
  $_O1_PARENT_DIR/model
EOF
  exit 1
fi

if [[ ! -f "$O1_PROJECT_DIR/dataloader_ours.py"    && ! -f "$_O1_PARENT_DIR/dataloader_ours.py" ]]; then
  echo "Error: dataloader_ours.py was not found in the experiment folder or its parent." >&2
  exit 1
fi

export O1_PYTHON_ROOT
export PYTHONPATH="$O1_PROJECT_DIR:$_O1_PARENT_DIR:$O1_PYTHON_ROOT:${PYTHONPATH:-}"

echo "O1_PROJECT_DIR=$O1_PROJECT_DIR"
echo "O1_CONDA_SH=$O1_CONDA_SH"
echo "O1_CONDA_ENV=$O1_CONDA_ENV"
echo "Python=$(command -v python)"
echo "O1_FEATURE_ROOT=$O1_FEATURE_ROOT"
echo "O1_META_EASY2HARD=$O1_META_EASY2HARD"
echo "O1_META_HARD2EASY=$O1_META_HARD2EASY"
echo "O1_META_DATA2=$O1_META_DATA2"
echo "O1_GATE_TABLE=$O1_GATE_TABLE"