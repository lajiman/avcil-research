#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE="${SCRIPT_DIR}/model_patch/audio_visual_model_incremental.py"
TARGET="${PROJECT_ROOT}/model/audio_visual_model_incremental.py"
BACKUP="${TARGET}.before_phase5_modality_analysis"

if [[ ! -f "${SOURCE}" ]]; then
  echo "Patch source not found: ${SOURCE}" >&2
  exit 1
fi
if [[ ! -f "${TARGET}" ]]; then
  echo "Target model file not found: ${TARGET}" >&2
  exit 1
fi
if [[ ! -f "${BACKUP}" ]]; then
  cp "${TARGET}" "${BACKUP}"
  echo "Backup created: ${BACKUP}"
else
  echo "Backup already exists: ${BACKUP}"
fi
cp "${SOURCE}" "${TARGET}"
echo "Patched model file: ${TARGET}"
echo "The patch is backward-compatible with existing tuple/tensor outputs."
