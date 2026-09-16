#!/usr/bin/env bash
set -euo pipefail

# Expected location:
#   Projects/AV-CIL_ICCV2023/experiments_phase_5_modality_analysis/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_ROOT="${FEATURE_ROOT:-../../../datasets/VGGSound}"
OUT_BASE="${OUT_BASE:-${SCRIPT_DIR}/results}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PERMUTATION_REPEATS="${PERMUTATION_REPEATS:-5}"
STEPS="${STEPS:-}"
COMPUTE_INPUT_ZERO="${COMPUTE_INPUT_ZERO:-0}"
CACHE_FEATURES="${CACHE_FEATURES:-0}"
SAVE_SAMPLE_COUNTERFACTUAL="${SAVE_SAMPLE_COUNTERFACTUAL:-0}"

COMMON_ARGS=(
  --feature_root "${FEATURE_ROOT}"
  --num_classes 100
  --class_num_per_step 10
  --infer_batch_size "${INFER_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --seed 42
  --topk_centroid 5
  --knn_k 10
  --knn_chunk_size 512
  --z1_interventions zero,mean,perm
  --permutation_repeats "${PERMUTATION_REPEATS}"
)

if [[ -n "${STEPS}" ]]; then
  COMMON_ARGS+=(--steps "${STEPS}")
fi
if [[ "${COMPUTE_INPUT_ZERO}" == "1" ]]; then
  COMMON_ARGS+=(--compute_input_zero)
fi
if [[ "${CACHE_FEATURES}" == "1" ]]; then
  COMMON_ARGS+=(--cache_features)
fi
if [[ "${SAVE_SAMPLE_COUNTERFACTUAL}" == "1" ]]; then
  COMMON_ARGS+=(--save_sample_counterfactual)
fi

run_order() {
  local order_name="$1"
  local dataset="$2"
  local meta_root="$3"
  local full_dir="$4"
  local no_z1_dir="$5"

  echo "============================================================"
  echo "Running modality diagnosis: ${order_name}"
  echo "============================================================"

  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_modality_difficulty.py" \
    --order_name "${order_name}" \
    --dataset "${dataset}" \
    --meta_root "${meta_root}" \
    --full_ckpt_dir "${full_dir}" \
    --no_z1_ckpt_dir "${no_z1_dir}" \
    --out_root "${OUT_BASE}/${order_name}" \
    "${COMMON_ARGS[@]}"
}

run_order \
  random \
  VGGSound_data2_balance_modality_analysis_seed42 \
  "${PROJECT_ROOT}/data2/balance" \
  "${PROJECT_ROOT}/experiments_phase_3_wo_lowerbound/save/VGGSound_data2_balance_lowerbound_c_wo_seed42" \
  "${PROJECT_ROOT}/experiments_phase_3_wo_lowerbound/save/VGGSound_data2_balance_attn_only_seed42"

run_order \
  easy2hard \
  VGGSound_easy2hard_balance_modality_analysis_seed42 \
  "${PROJECT_ROOT}/data_easy2hard/balance" \
  "${PROJECT_ROOT}/experiments_phase_3_wo_lowerbound/save/VGGSound_easy2hard_balance_lowerbound_c_wo_seed42" \
  "${PROJECT_ROOT}/experiments_phase_3_wo_lowerbound/save/VGGSound_easy2hard_balance_attn_only_seed42"

run_order \
  hard2easy \
  VGGSound_hard2easy_balance_modality_analysis_seed42 \
  "${PROJECT_ROOT}/data_hard2easy/balance" \
  "${PROJECT_ROOT}/experiments_phase_3_wo_lowerbound/save/VGGSound_hard2easy_balance_lowerbound_c_wo_seed42" \
  "${PROJECT_ROOT}/experiments_phase_3_wo_lowerbound/save/VGGSound_hard2easy_balance_attn_only_seed42"

echo "All analyses completed. Results: ${OUT_BASE}"