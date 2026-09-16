# Phase 5: Modality-Difficulty Analysis

This folder implements the matched **Full AVCIL vs No-z1 AVCIL** diagnosis discussed in the meeting notes.

## Files

- `analyze_modality_difficulty.py` — main analysis program.
- `run_all_modality_analysis.sh` — runs random, easy-to-hard, and hard-to-easy using the supplied checkpoint directories.
- `plot_modality_analysis.py` — creates four compact diagnostic figures from the CSV outputs.
- `model_patch/audio_visual_model_incremental.py` — backward-compatible model implementation that exposes raw and normalized z0/z1/z2 tensors through the existing dictionary output.
- `apply_model_patch.sh` — backs up and replaces `model/audio_visual_model_incremental.py`.

The analysis program contains a fallback implementation that reconstructs the same raw features from the model modules. Therefore, the model patch is **recommended but not mandatory** for running this analysis. Applying it makes future analysis code simpler and gives one canonical feature interface.

## What the code diagnoses

### 1. z0 intrinsic modality difficulty

The script uses:

- audio z0: the fixed audio input embedding;
- visual z0: a uniform average over temporal and spatial visual tokens.

The visual z0 reference is deliberately independent of audio-guided attention. The final-step table contains all 100 classes.

### 2. z1 modality mismatch in Full AVCIL

For audio and visual z1, it computes:

- intra-class dispersion;
- nearest-centroid and top-k centroid distance;
- normalized margin;
- kNN purity;
- continuous difficulty percentile;
- Easy/Hard and Easy/Medium/Hard states;
- paired and centroid cross-modal cosine similarity;
- audio/visual/z2 class-neighborhood overlap.

### 3. continual-learning dynamics

For every checkpoint and class, it records:

- centroid drift from the previous checkpoint and first-seen checkpoint;
- audio and visual difficulty changes;
- difficulty-gap changes;
- joint recall/F1 and their forgetting values.

For valid temporal interpretation, analyze all checkpoints. When `--steps` skips checkpoints, “previous” means the previous **analyzed** checkpoint.

### 4. counterfactual modality contribution

No temporary classifier is trained. The original joint classifier is retained.

The primary interventions are applied to the **raw z1 branches**:

- `z1_zero`: replace one direct branch with zero;
- `z1_mean`: replace one direct branch with the test-set mean feature;
- `z1_permutation`: replace one direct branch with features sampled from other classes while preserving the feature distribution.

The script measures:

- true-class logit drop;
- true-class probability drop;
- classification-margin drop;
- prediction-change rate;
- necessity rate: joint correct, intervention wrong;
- interference rate: joint wrong, intervention correct;
- synergy and redundancy rates.

`z1_permutation` is the preferred primary result because the replacement remains on the empirical feature distribution.

Optional `--compute_input_zero` performs end-to-end z0 zero-input interventions. This is a sensitivity test, not audio-only or visual-only accuracy.

### 5. effect of symmetric z1 contrastive learning

Full and No-z1 runs are compared while both retain attention distillation. Results are grouped using the **No-z1 Easy/Hard state**, avoiding grouping solely by a representation already modified by the treatment.

A matched No-attention run is still required for a causal claim about attention distillation itself.

## Model output patch

From this folder, run:

```bash
./apply_model_patch.sh
```

It first creates:

```text
model/audio_visual_model_incremental.py.before_phase5_modality_analysis
```

The new dictionary keys are:

```text
z0_audio_raw
z0_visual_uniform_raw
z0_audio_norm
z0_visual_uniform_norm
attn_visual_pooled_raw
attn_visual_pooled_norm
z1_audio_raw
z1_visual_raw
z1_audio_norm
z1_visual_norm
z2_fusion_raw
z2_fusion_norm
```

Existing tuple/tensor outputs and existing dictionary keys are unchanged.

## Run all three orders

Enter the analysis folder and activate the same environment used to train AVCIL:

```bash
cd Projects/AV-CIL_ICCV2023/experiments_phase_5_modality_analysis
conda activate avcil
./run_all_modality_analysis.sh
```

The default feature path is the same relative path used by the existing experiment folders:

```text
../../../datasets/VGGSound
```

Override it when necessary:

```bash
FEATURE_ROOT=/absolute/path/to/VGGSound ./run_all_modality_analysis.sh
```

Useful environment switches:

```bash
# Fast initial check: final checkpoint only
STEPS=9 ./run_all_modality_analysis.sh

# All checkpoints plus end-to-end input-zero intervention
COMPUTE_INPUT_ZERO=1 ./run_all_modality_analysis.sh

# Save raw feature arrays for later t-SNE or custom analysis
CACHE_FEATURES=1 ./run_all_modality_analysis.sh

# Reduce GPU memory usage
INFER_BATCH_SIZE=8 ./run_all_modality_analysis.sh
```

For the final dynamic diagnosis, do **not** set `STEPS`; all steps 0–9 should be analyzed.

## Run one order manually

Example for random/data2:

```bash
python -u analyze_modality_difficulty.py \
  --dataset VGGSound_data2_balance_modality_analysis_seed42 \
  --order_name random \
  --num_classes 100 \
  --class_num_per_step 10 \
  --feature_root ../../../datasets/VGGSound \
  --meta_root ../data2/balance \
  --full_ckpt_dir ../experiments_phase_3_wo_lowerbound/save/VGGSound_data2_balance_lowerbound_c_wo_seed42 \
  --no_z1_ckpt_dir ../experiments_phase_3_wo_lowerbound/save/VGGSound_data2_balance_attn_only_seed42 \
  --out_root results/random \
  --z1_interventions zero,mean,perm \
  --permutation_repeats 5 \
  --infer_batch_size 32 \
  --num_workers 0
```

## Plot results

```bash
python plot_modality_analysis.py --result_root results/random
python plot_modality_analysis.py --result_root results/easy2hard
python plot_modality_analysis.py --result_root results/hard2easy
```

## Main output files

Inside each order’s result directory:

### `core/01_z0_intrinsic_difficulty_final_step.csv`

Static input-space modality difficulty for all 100 classes.

### `core/02_full_z1_modality_mismatch_by_step.csv`

The main Full-AVCIL table. Use it to show that modality mismatch persists and changes across CL steps.

### `core/03_counterfactual_modality_contribution_by_class.csv`

Class-level modality contribution and interference for Full and No-z1.

### `core/04_difficulty_contribution_correlations.csv`

Tests whether:

- signed audio-minus-visual difficulty predicts signed contribution asymmetry;
- absolute difficulty mismatch predicts intervention-based interference.

### `core/05_z1_contrastive_effect_by_status.csv`

Full minus No-z1 effects grouped by the No-z1 modality state.

### `detail/modality_difficulty_by_class_step.csv`

Complete class-step geometry, alignment, neighborhood overlap, dynamics, and joint performance.

### `detail/z1_contrastive_effect_by_class.csv`

Class-level matched Full versus No-z1 differences.

### `detail/checkpoint_consistency.csv`

Includes `max_abs_classifier_reconstruction_error`. This should be near zero. A large value means the extracted raw z2 does not match the actual classifier input and the analysis stops rather than silently reporting invalid interventions.

## Recommended execution sequence

1. Run `STEPS=9` for all three orders as a setup and memory check.
2. Inspect `detail/checkpoint_consistency.csv` and confirm reconstruction errors are near zero.
3. Run all steps without `STEPS` for the dynamic diagnosis.
4. Treat `z1_permutation` as the primary contribution analysis; use zero and mean as robustness checks.
5. Only enable `COMPUTE_INPUT_ZERO=1` after the main z1 analysis succeeds, because it requires two additional full forward passes per batch.
