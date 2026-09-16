#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-level modality-difficulty diagnosis for AV-CIL.

This script analyzes a matched pair of runs:

    Full AVCIL:
        classification + logits distillation + attention distillation
        + z1 instance/class contrastive losses

    No-z1 AVCIL:
        classification + logits distillation + attention distillation
        (z1 instance/class contrastive losses removed)

The analysis is deliberately based on the original joint classifier. It does
NOT add audio-only or visual-only classifiers. Modality contribution is instead
measured with counterfactual interventions on the original model.

Five diagnostic layers
----------------------
1. z0 intrinsic modality difficulty
   - fixed input audio feature
   - audio-independent uniform-pooled visual input feature

2. z1 modality difficulty mismatch in Full AVCIL
   - class geometry of audio and visual z1 representations
   - continuous difficulty gap and E/H or E/M/H states
   - cross-modal alignment and class-neighborhood overlap

3. continual-learning dynamics
   - class-centroid drift
   - change in modality difficulty and mismatch from first seen / previous step
   - joint per-class recall/F1 forgetting

4. counterfactual modality contribution
   - no temporary classifier is trained
   - z1 direct-branch intervention: zero, mean, cross-class permutation
   - optional z0 end-to-end zero intervention
   - true-class logit/probability/margin drop, necessity, interference, synergy

5. effect of the original symmetric z1 contrastive losses
   - Full AVCIL versus matched No-z1 AVCIL
   - geometry, mismatch, alignment, joint performance, and intervention metrics

Important interpretation
------------------------
* z1 geometry uses cosine distance after internal L2 normalization.
* counterfactual z1 intervention uses RAW z1 vectors because those are the
  vectors actually added and passed into the classifier.
* z1 audio removal only removes the direct audio branch. The visual z1 branch
  remains audio-conditioned by the original attention module. The output is
  therefore named a direct-branch intervention, not audio-only inference.
* z0 zero intervention is an end-to-end sensitivity test, not a calibrated
  single-modality accuracy measurement.

Expected checkpoint layout
--------------------------
    <checkpoint_dir>/step_0_best_model.pkl
    ...
    <checkpoint_dir>/step_9_best_model.pkl

The script should live at:
    Projects/AV-CIL_ICCV2023/experiments_phase_5_modality_analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader_ours import IcaAVELoader  # noqa: E402
from model.audio_visual_model_incremental import IncreAudioVisualNet  # noqa: F401,E402


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_RE = re.compile(r"^step[_-]?(\d+)_best_model(?:\.pkl)?$", re.IGNORECASE)
EPS = 1e-12


# =============================================================================
# Generic utilities
# =============================================================================
def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not value.strip():
        return None
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def l2_normalize_np(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_distance_vec(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(1.0 - np.dot(a, b) / denom)


def average_rank_percentile(values: Sequence[float]) -> np.ndarray:
    """Average-tie percentile ranks in [0, 1], low value -> low percentile."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and np.isclose(sorted_x[end], sorted_x[start], rtol=0.0, atol=1e-12):
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks / float(n - 1)


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.isclose(x_arr.std(), 0.0) or np.isclose(y_arr.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3:
        return float("nan")
    return pearson_corr(average_rank_percentile(x_arr), average_rank_percentile(y_arr))


def dataset_type(value: str) -> str:
    if value in {"AVE", "ksounds"} or "VGGSound" in value:
        return value
    raise argparse.ArgumentTypeError("dataset must be AVE, ksounds, or contain VGGSound")


# =============================================================================
# Checkpoints and model compatibility
# =============================================================================
def discover_checkpoints(checkpoint_dir: str | Path) -> Dict[int, str]:
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_dir}")
    result: Dict[int, str] = {}
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        match = CHECKPOINT_RE.match(path.name)
        if match:
            step = int(match.group(1))
            if step in result:
                raise ValueError(f"Duplicate checkpoint for step {step}: {path}")
            result[step] = str(path)
    if not result:
        raise FileNotFoundError(f"No step_<k>_best_model.pkl in {checkpoint_dir}")
    return dict(sorted(result.items()))


def load_model(checkpoint_path: str) -> torch.nn.Module:
    try:
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def linear_classifier_parameters(model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    base = unwrap_model(model)
    classifier = base.classifier
    if not isinstance(classifier, torch.nn.Linear):
        raise TypeError(
            "Counterfactual analysis currently requires nn.Linear classifier; "
            f"got {type(classifier).__name__}."
        )
    weight = classifier.weight.detach().float().cpu().numpy()
    if classifier.bias is None:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
    else:
        bias = classifier.bias.detach().float().cpu().numpy()
    return weight, bias


def classify_np(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float32) @ weight.T + bias[None, :]


def _fallback_analysis_forward(
    model: torch.nn.Module,
    visual: torch.Tensor,
    audio: torch.Tensor,
    out_logits: bool = True,
) -> Dict[str, torch.Tensor]:
    """Reconstruct analysis tensors using the existing model modules.

    This fallback makes the analysis runnable even before the optional model patch
    is copied into model/audio_visual_model_incremental.py.
    """
    base = unwrap_model(model)
    visual_4d = visual.view(visual.shape[0], 8, -1, 768)
    visual_z0_uniform = visual_4d.mean(dim=(1, 2))
    spatial_attn, temporal_attn = base.audio_visual_attention(audio, visual_4d)
    visual_pooled = torch.sum(spatial_attn * visual_4d, dim=2)
    visual_pooled = torch.sum(temporal_attn * visual_pooled, dim=1)
    z1_audio = F.relu(base.audio_proj(audio))
    z1_visual = F.relu(base.visual_proj(visual_pooled))
    z2 = z1_audio + z1_visual
    outputs: Dict[str, torch.Tensor] = {
        "z0_audio_raw": audio,
        "z0_visual_uniform_raw": visual_z0_uniform,
        "attn_visual_pooled_raw": visual_pooled,
        "z1_audio_raw": z1_audio,
        "z1_visual_raw": z1_visual,
        "z2_fusion_raw": z2,
        "z0_audio_norm": F.normalize(audio, dim=1),
        "z0_visual_uniform_norm": F.normalize(visual_z0_uniform, dim=1),
        "attn_visual_pooled_norm": F.normalize(visual_pooled, dim=1),
        "z1_audio_norm": F.normalize(z1_audio, dim=1),
        "z1_visual_norm": F.normalize(z1_visual, dim=1),
        "z2_fusion_norm": F.normalize(z2, dim=1),
    }
    if out_logits:
        outputs["logits"] = base.classifier(z2)
    return outputs


def analysis_forward(
    model: torch.nn.Module,
    visual: torch.Tensor,
    audio: torch.Tensor,
    out_logits: bool = True,
) -> Dict[str, torch.Tensor]:
    """Use the patched dict output when available, otherwise reconstruct it."""
    try:
        outputs = model(
            visual=visual,
            audio=audio,
            out_logits=out_logits,
            return_dict=True,
            out_analysis_features=True,
        )
        required = {
            "z0_audio_raw",
            "z0_visual_uniform_raw",
            "attn_visual_pooled_raw",
            "z1_audio_raw",
            "z1_visual_raw",
            "z2_fusion_raw",
        }
        if not isinstance(outputs, dict) or not required.issubset(outputs):
            return _fallback_analysis_forward(model, visual, audio, out_logits=out_logits)
        return outputs
    except TypeError:
        return _fallback_analysis_forward(model, visual, audio, out_logits=out_logits)


# =============================================================================
# Dataset and inference
# =============================================================================
def set_dataset_to_seen_classes(dataset: Any, num_seen_classes: int) -> None:
    """Match the previous analysis code: evaluate all classes seen so far."""
    dataset.current_step_class = np.arange(num_seen_classes)
    dataset.all_current_data_vids = []
    for class_idx in dataset.current_step_class:
        dataset.all_current_data_vids += dataset.all_classId_vid_dict[str(int(class_idx))]


@torch.no_grad()
def run_model_inference(
    model: torch.nn.Module,
    dataset: Any,
    batch_size: int,
    num_workers: int,
    description: str,
    compute_input_zero: bool,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    buffers: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
    raw_keys = [
        "z0_audio_raw",
        "z0_visual_uniform_raw",
        "attn_visual_pooled_raw",
        "z1_audio_raw",
        "z1_visual_raw",
        "z2_fusion_raw",
    ]

    for data, labels in tqdm(loader, desc=description, ncols=110):
        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)
        labels = labels.long()

        outputs = analysis_forward(model, visual, audio, out_logits=True)
        logits = outputs["logits"]

        buffers["labels"].append(labels.cpu())
        buffers["logits"].append(logits.detach().float().cpu())
        for key in raw_keys:
            buffers[key].append(outputs[key].detach().float().cpu())

        if compute_input_zero:
            logits_audio_zero = analysis_forward(
                model,
                visual=visual,
                audio=torch.zeros_like(audio),
                out_logits=True,
            )["logits"]
            logits_visual_zero = analysis_forward(
                model,
                visual=torch.zeros_like(visual),
                audio=audio,
                out_logits=True,
            )["logits"]
            buffers["input_zero_audio_logits"].append(
                logits_audio_zero.detach().float().cpu()
            )
            buffers["input_zero_visual_logits"].append(
                logits_visual_zero.detach().float().cpu()
            )

    output = {key: torch.cat(value, dim=0).numpy() for key, value in buffers.items()}
    output["preds"] = output["logits"].argmax(axis=1).astype(np.int64)
    return output


# =============================================================================
# Classification metrics and forgetting
# =============================================================================
def per_class_classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    labels = labels.astype(np.int64)
    preds = logits.argmax(axis=1).astype(np.int64)
    result: Dict[int, Dict[str, float]] = {}
    for class_id in range(num_classes):
        true_c = labels == class_id
        pred_c = preds == class_id
        tp = int(np.sum(true_c & pred_c))
        fp = int(np.sum((~true_c) & pred_c))
        fn = int(np.sum(true_c & (~pred_c)))
        support = int(np.sum(true_c))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        result[class_id] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


# =============================================================================
# Geometry and difficulty
# =============================================================================
def compute_knn_purity(
    x_norm: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    k: int,
    chunk_size: int,
    description: str,
) -> np.ndarray:
    n = len(labels)
    if n <= 1:
        return np.zeros(num_classes, dtype=np.float32)
    k_eff = min(k, n - 1)
    sample_purity = np.zeros(n, dtype=np.float32)
    for start in tqdm(range(0, n, chunk_size), desc=description, ncols=110, leave=False):
        end = min(start + chunk_size, n)
        sim = x_norm[start:end] @ x_norm.T
        local_rows = np.arange(end - start)
        global_cols = np.arange(start, end)
        sim[local_rows, global_cols] = -np.inf
        neighbors = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
        sample_purity[start:end] = (
            labels[neighbors] == labels[start:end, None]
        ).mean(axis=1)
    class_purity = np.zeros(num_classes, dtype=np.float32)
    for class_id in range(num_classes):
        idx = np.where(labels == class_id)[0]
        class_purity[class_id] = float(sample_purity[idx].mean()) if len(idx) else 0.0
    return class_purity


def compute_geometry(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    topk_centroid: int,
    knn_k: int,
    knn_chunk_size: int,
    description: str,
    difficulty_definition: str,
    easy_quantile: float,
    hard_quantile: float,
) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray]:
    num_classes = len(class_names)
    x_norm = l2_normalize_np(features)
    dim = x_norm.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)
    intra = np.zeros(num_classes, dtype=np.float32)

    for class_id in range(num_classes):
        idx = np.where(labels == class_id)[0]
        counts[class_id] = len(idx)
        if len(idx):
            centroid = x_norm[idx].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), EPS)
            centroids[class_id] = centroid
            intra[class_id] = float((1.0 - x_norm[idx] @ centroid).mean())

    centroid_distance = 1.0 - centroids @ centroids.T
    np.fill_diagonal(centroid_distance, np.inf)
    purity = compute_knn_purity(
        x_norm,
        labels.astype(np.int64),
        num_classes,
        knn_k,
        knn_chunk_size,
        description=f"{description}/kNN",
    )

    summary: Dict[int, Dict[str, Any]] = {}
    for class_id in range(num_classes):
        order = np.argsort(centroid_distance[class_id])
        order = [
            int(neighbor)
            for neighbor in order
            if np.isfinite(centroid_distance[class_id, neighbor])
        ][: min(topk_centroid, max(0, num_classes - 1))]
        distances = [float(centroid_distance[class_id, neighbor]) for neighbor in order]
        nearest_distance = distances[0] if distances else 0.0
        topk_mean_distance = float(np.mean(distances)) if distances else 0.0
        normalized_margin = nearest_distance / (float(intra[class_id]) + EPS)
        summary[class_id] = {
            "count": int(counts[class_id]),
            "intra_dispersion": float(intra[class_id]),
            "nearest_distance": nearest_distance,
            "topk_mean_distance": topk_mean_distance,
            "normalized_margin": float(normalized_margin),
            "knn_purity": float(purity[class_id]),
            "nearest_class_id": order[0] if order else "",
            "nearest_class_name": class_names[order[0]] if order else "",
            "neighbor_ids": order,
            "neighbor_names": [class_names[item] for item in order],
            "neighbor_distances": distances,
        }

    add_difficulty_scores(
        summary,
        difficulty_definition=difficulty_definition,
        easy_quantile=easy_quantile,
        hard_quantile=hard_quantile,
    )
    return summary, centroids


def add_difficulty_scores(
    summary: Dict[int, Dict[str, Any]],
    difficulty_definition: str,
    easy_quantile: float,
    hard_quantile: float,
) -> None:
    """Attach continuous and categorical class-difficulty fields.

    Main paper setting:
      - normalized_margin: difficulty is the reverse percentile of the
        normalized class margin (nearest-centroid distance / intra dispersion).
      - margin_purity: equal average of reverse-margin percentile and reverse
        kNN-purity percentile, followed by a percentile rank.

    The legacy four-component definition is retained only for sensitivity
    analysis and backward compatibility.

    Ternary states follow the requested 30/40/30 narrative by default:
      easy   : percentile < 0.30
      medium : 0.30 <= percentile < 0.70
      hard   : percentile >= 0.70
    """
    class_ids = sorted(summary)
    intra = np.asarray([summary[c]["intra_dispersion"] for c in class_ids])
    nearest = np.asarray([summary[c]["nearest_distance"] for c in class_ids])
    margin = np.asarray([summary[c]["normalized_margin"] for c in class_ids])
    purity = np.asarray([summary[c]["knn_purity"] for c in class_ids])

    intra_difficulty = average_rank_percentile(intra)
    inter_difficulty = 1.0 - average_rank_percentile(nearest)
    margin_difficulty = 1.0 - average_rank_percentile(margin)
    purity_difficulty = 1.0 - average_rank_percentile(purity)

    difficulty_components = np.stack(
        [
            intra_difficulty,
            inter_difficulty,
            margin_difficulty,
            purity_difficulty,
        ],
        axis=1,
    )

    if difficulty_definition == "normalized_margin":
        score = margin_difficulty
    elif difficulty_definition == "margin_purity":
        score = 0.5 * (margin_difficulty + purity_difficulty)
    elif difficulty_definition == "legacy_four_component":
        score = difficulty_components.mean(axis=1)
    else:
        raise ValueError(
            "difficulty_definition must be one of: normalized_margin, "
            "margin_purity, legacy_four_component"
        )

    score_pct = average_rank_percentile(score)

    for index, class_id in enumerate(class_ids):
        pct = float(score_pct[index])
        # Binary state is retained for backward compatibility only. The main
        # narrative and all new grouped outputs use the ternary state.
        binary = "hard" if pct >= 0.5 else "easy"
        if pct < easy_quantile:
            ternary = "easy"
        elif pct < hard_quantile:
            ternary = "medium"
        else:
            ternary = "hard"
        summary[class_id].update(
            {
                "difficulty_definition": difficulty_definition,
                "difficulty_easy_quantile": float(easy_quantile),
                "difficulty_hard_quantile": float(hard_quantile),
                "difficulty_score": float(score[index]),
                "difficulty_percentile": pct,
                "difficulty_binary": binary,
                "difficulty_ternary": ternary,
                "difficulty_intra_component": float(difficulty_components[index, 0]),
                "difficulty_inter_component": float(difficulty_components[index, 1]),
                "difficulty_margin_component": float(difficulty_components[index, 2]),
                "difficulty_purity_component": float(difficulty_components[index, 3]),
            }
        )


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    return safe_div(len(set_a & set_b), len(union)) if union else 0.0


def compute_crossmodal_alignment(
    audio_features: np.ndarray,
    visual_features: np.ndarray,
    labels: np.ndarray,
    audio_centroids: np.ndarray,
    visual_centroids: np.ndarray,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    audio_norm = l2_normalize_np(audio_features)
    visual_norm = l2_normalize_np(visual_features)
    paired = np.sum(audio_norm * visual_norm, axis=1)
    result: Dict[int, Dict[str, float]] = {}
    for class_id in range(num_classes):
        idx = np.where(labels == class_id)[0]
        result[class_id] = {
            "paired_av_cosine": float(paired[idx].mean()) if len(idx) else 0.0,
            "centroid_av_cosine": float(
                np.dot(audio_centroids[class_id], visual_centroids[class_id])
            ),
        }
    return result


# =============================================================================
# Counterfactual interventions
# =============================================================================
def true_class_values(logits: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    labels = labels.astype(np.int64)
    row = np.arange(len(labels))
    true_logits = logits[row, labels]

    masked = logits.copy()
    masked[row, labels] = -np.inf
    max_other = masked.max(axis=1)
    margin = true_logits - max_other

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)
    true_prob = probs[row, labels]
    pred = logits.argmax(axis=1).astype(np.int64)
    return {
        "true_logit": true_logits,
        "true_prob": true_prob,
        "margin": margin,
        "pred": pred,
        "correct": pred == labels,
    }


def cross_class_permutation(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = labels.astype(np.int64)
    classes = np.unique(labels)
    by_class = {class_id: np.where(labels == class_id)[0] for class_id in classes}
    result = np.empty(len(labels), dtype=np.int64)
    for index, class_id in enumerate(labels):
        choices = classes[classes != class_id]
        target_class = int(rng.choice(choices))
        result[index] = int(rng.choice(by_class[target_class]))
    return result


def build_counterfactual_logits(
    model_output: Dict[str, np.ndarray],
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    interventions: Sequence[str],
    permutation_repeats: int,
    seed: int,
    include_input_zero: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    audio = model_output["z1_audio_raw"]
    visual = model_output["z1_visual_raw"]
    labels = model_output["labels"].astype(np.int64)
    result: Dict[str, Dict[str, np.ndarray]] = {}

    if "zero" in interventions:
        result["z1_zero"] = {
            "remove_audio": classify_np(visual, classifier_weight, classifier_bias),
            "remove_visual": classify_np(audio, classifier_weight, classifier_bias),
        }

    if "mean" in interventions:
        mean_audio = audio.mean(axis=0, keepdims=True)
        mean_visual = visual.mean(axis=0, keepdims=True)
        result["z1_mean"] = {
            "remove_audio": classify_np(
                visual + mean_audio, classifier_weight, classifier_bias
            ),
            "remove_visual": classify_np(
                audio + mean_visual, classifier_weight, classifier_bias
            ),
        }

    if "perm" in interventions:
        rng = np.random.default_rng(seed)
        audio_logits_sum = np.zeros_like(model_output["logits"], dtype=np.float64)
        visual_logits_sum = np.zeros_like(model_output["logits"], dtype=np.float64)
        for _ in range(permutation_repeats):
            permutation = cross_class_permutation(labels, rng)
            audio_logits_sum += classify_np(
                audio[permutation] + visual,
                classifier_weight,
                classifier_bias,
            )
            permutation = cross_class_permutation(labels, rng)
            visual_logits_sum += classify_np(
                audio + visual[permutation],
                classifier_weight,
                classifier_bias,
            )
        result["z1_permutation"] = {
            "remove_audio": (audio_logits_sum / permutation_repeats).astype(np.float32),
            "remove_visual": (visual_logits_sum / permutation_repeats).astype(np.float32),
        }

    if include_input_zero:
        result["z0_input_zero"] = {
            "remove_audio": model_output["input_zero_audio_logits"],
            "remove_visual": model_output["input_zero_visual_logits"],
        }

    return result


def aggregate_counterfactual_by_class(
    normal_logits: np.ndarray,
    counterfactuals: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    step: int,
    class_num_per_step: int,
    save_sample_rows: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normal = true_class_values(normal_logits, labels)
    class_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for intervention, modal_logits in counterfactuals.items():
        removed_audio = true_class_values(modal_logits["remove_audio"], labels)
        removed_visual = true_class_values(modal_logits["remove_visual"], labels)

        per_sample: Dict[str, np.ndarray] = {}
        for modality, cf in [
            ("audio", removed_audio),
            ("visual", removed_visual),
        ]:
            per_sample[f"{modality}_true_logit_drop"] = (
                normal["true_logit"] - cf["true_logit"]
            )
            per_sample[f"{modality}_true_prob_drop"] = (
                normal["true_prob"] - cf["true_prob"]
            )
            per_sample[f"{modality}_margin_drop"] = normal["margin"] - cf["margin"]
            per_sample[f"{modality}_prediction_changed"] = normal["pred"] != cf["pred"]
            per_sample[f"{modality}_necessity"] = normal["correct"] & (~cf["correct"])
            per_sample[f"{modality}_interference"] = (~normal["correct"]) & cf["correct"]
            per_sample[f"{modality}_cf_correct"] = cf["correct"]

        synergy = (
            normal["correct"]
            & (~removed_audio["correct"])
            & (~removed_visual["correct"])
        )
        redundancy = (
            normal["correct"]
            & removed_audio["correct"]
            & removed_visual["correct"]
        )

        for class_id, class_name in enumerate(class_names):
            idx = np.where(labels == class_id)[0]
            if len(idx) == 0:
                continue
            row: Dict[str, Any] = {
                "model": model_name,
                "step": step,
                "class_id": class_id,
                "class_name": class_name,
                "first_seen_step": class_id // class_num_per_step,
                "task_age": step - class_id // class_num_per_step,
                "intervention": intervention,
                "support": len(idx),
                "joint_accuracy": float(normal["correct"][idx].mean()),
                "synergy_rate": float(synergy[idx].mean()),
                "redundancy_rate": float(redundancy[idx].mean()),
            }
            for modality in ["audio", "visual"]:
                for metric in [
                    "true_logit_drop",
                    "true_prob_drop",
                    "margin_drop",
                    "prediction_changed",
                    "necessity",
                    "interference",
                    "cf_correct",
                ]:
                    values = per_sample[f"{modality}_{metric}"][idx]
                    row[f"{modality}_{metric}"] = float(values.mean())
            row["audio_minus_visual_logit_drop"] = (
                row["audio_true_logit_drop"] - row["visual_true_logit_drop"]
            )
            row["audio_minus_visual_margin_drop"] = (
                row["audio_margin_drop"] - row["visual_margin_drop"]
            )
            row["audio_minus_visual_interference"] = (
                row["audio_interference"] - row["visual_interference"]
            )
            class_rows.append(row)

        if save_sample_rows:
            for index, label in enumerate(labels.astype(np.int64)):
                row = {
                    "model": model_name,
                    "step": step,
                    "sample_index": index,
                    "class_id": int(label),
                    "class_name": class_names[int(label)],
                    "intervention": intervention,
                    "joint_correct": int(normal["correct"][index]),
                    "synergy": int(synergy[index]),
                    "redundancy": int(redundancy[index]),
                }
                for modality in ["audio", "visual"]:
                    for metric in [
                        "true_logit_drop",
                        "true_prob_drop",
                        "margin_drop",
                        "prediction_changed",
                        "necessity",
                        "interference",
                        "cf_correct",
                    ]:
                        value = per_sample[f"{modality}_{metric}"][index]
                        row[f"{modality}_{metric}"] = (
                            int(value) if np.issubdtype(np.asarray(value).dtype, np.bool_) else float(value)
                        )
                sample_rows.append(row)

    return class_rows, sample_rows


# =============================================================================
# Row construction and cross-model comparison
# =============================================================================
def geometry_to_columns(prefix: str, geometry: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "count",
        "intra_dispersion",
        "nearest_distance",
        "topk_mean_distance",
        "normalized_margin",
        "knn_purity",
        "nearest_class_id",
        "nearest_class_name",
        "difficulty_definition",
        "difficulty_easy_quantile",
        "difficulty_hard_quantile",
        "difficulty_score",
        "difficulty_percentile",
        "difficulty_binary",
        "difficulty_ternary",
        "difficulty_intra_component",
        "difficulty_inter_component",
        "difficulty_margin_component",
        "difficulty_purity_component",
    ]
    return {f"{prefix}_{key}": geometry.get(key, "") for key in keys}


def build_modality_rows_for_model(
    model_name: str,
    step: int,
    class_names: Sequence[str],
    performance: Dict[int, Dict[str, float]],
    geometry: Dict[str, Dict[int, Dict[str, Any]]],
    centroids: Dict[str, np.ndarray],
    alignment: Dict[int, Dict[str, float]],
    class_num_per_step: int,
    state: Dict[str, Dict[int, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    num_classes = len(class_names)

    for class_id in range(num_classes):
        audio_g = geometry["z1_audio"][class_id]
        visual_g = geometry["z1_visual"][class_id]
        z2_g = geometry["z2_fusion"][class_id]
        attn_g = geometry["attn_visual_pooled"][class_id]

        gap_signed = (
            audio_g["difficulty_percentile"] - visual_g["difficulty_percentile"]
        )
        gap_abs = abs(gap_signed)
        binary_state = f"A_{audio_g['difficulty_binary']}_V_{visual_g['difficulty_binary']}"
        ternary_state = f"A_{audio_g['difficulty_ternary']}_V_{visual_g['difficulty_ternary']}"
        z2_ternary = z2_g["difficulty_ternary"]
        z1_ternary_mismatch = audio_g["difficulty_ternary"] != visual_g["difficulty_ternary"]
        z1_extreme_mismatch = {
            audio_g["difficulty_ternary"], visual_g["difficulty_ternary"]
        } == {"easy", "hard"}

        row: Dict[str, Any] = {
            "model": model_name,
            "step": step,
            "class_id": class_id,
            "class_name": class_names[class_id],
            "first_seen_step": class_id // class_num_per_step,
            "task_age": step - class_id // class_num_per_step,
            **performance[class_id],
            "z1_difficulty_gap_signed_audio_minus_visual": gap_signed,
            "z1_difficulty_gap_abs": gap_abs,
            "z1_binary_state": binary_state,
            "z1_ternary_state": ternary_state,
            "z1_ternary_mismatch": int(z1_ternary_mismatch),
            "z1_extreme_easy_hard_mismatch": int(z1_extreme_mismatch),
            "z2_difficulty_binary": z2_g["difficulty_binary"],
            "z2_difficulty_ternary": z2_ternary,
            "z1_z2_ternary_state": f"{ternary_state}_J_{z2_ternary}",
            "paired_av_cosine": alignment[class_id]["paired_av_cosine"],
            "centroid_av_cosine": alignment[class_id]["centroid_av_cosine"],
            "audio_visual_neighbor_jaccard": jaccard(
                audio_g["neighbor_ids"], visual_g["neighbor_ids"]
            ),
            "audio_z2_neighbor_jaccard": jaccard(
                audio_g["neighbor_ids"], z2_g["neighbor_ids"]
            ),
            "visual_z2_neighbor_jaccard": jaccard(
                visual_g["neighbor_ids"], z2_g["neighbor_ids"]
            ),
        }
        for layer in ["z1_audio", "z1_visual", "z2_fusion", "attn_visual_pooled"]:
            row.update(geometry_to_columns(layer, geometry[layer][class_id]))

        current_audio_centroid = centroids["z1_audio"][class_id]
        current_visual_centroid = centroids["z1_visual"][class_id]

        if class_id not in state["first_audio_centroid"]:
            state["first_audio_centroid"][class_id] = current_audio_centroid.copy()
            state["first_visual_centroid"][class_id] = current_visual_centroid.copy()
            state["first_audio_difficulty"][class_id] = audio_g["difficulty_percentile"]
            state["first_visual_difficulty"][class_id] = visual_g["difficulty_percentile"]
            state["first_gap"][class_id] = gap_signed
            drift_audio_first = 0.0
            drift_visual_first = 0.0
            difficulty_audio_from_first = 0.0
            difficulty_visual_from_first = 0.0
            gap_from_first = 0.0
        else:
            drift_audio_first = cosine_distance_vec(
                current_audio_centroid, state["first_audio_centroid"][class_id]
            )
            drift_visual_first = cosine_distance_vec(
                current_visual_centroid, state["first_visual_centroid"][class_id]
            )
            difficulty_audio_from_first = (
                audio_g["difficulty_percentile"]
                - state["first_audio_difficulty"][class_id]
            )
            difficulty_visual_from_first = (
                visual_g["difficulty_percentile"]
                - state["first_visual_difficulty"][class_id]
            )
            gap_from_first = gap_signed - state["first_gap"][class_id]

        if class_id not in state["previous_audio_centroid"]:
            drift_audio_previous: Any = ""
            drift_visual_previous: Any = ""
            difficulty_audio_from_previous: Any = ""
            difficulty_visual_from_previous: Any = ""
            gap_from_previous: Any = ""
        else:
            drift_audio_previous = cosine_distance_vec(
                current_audio_centroid, state["previous_audio_centroid"][class_id]
            )
            drift_visual_previous = cosine_distance_vec(
                current_visual_centroid, state["previous_visual_centroid"][class_id]
            )
            difficulty_audio_from_previous = (
                audio_g["difficulty_percentile"]
                - state["previous_audio_difficulty"][class_id]
            )
            difficulty_visual_from_previous = (
                visual_g["difficulty_percentile"]
                - state["previous_visual_difficulty"][class_id]
            )
            gap_from_previous = gap_signed - state["previous_gap"][class_id]

        row.update(
            {
                "z1_audio_centroid_drift_from_first": drift_audio_first,
                "z1_visual_centroid_drift_from_first": drift_visual_first,
                "z1_audio_centroid_drift_from_previous": drift_audio_previous,
                "z1_visual_centroid_drift_from_previous": drift_visual_previous,
                "z1_audio_difficulty_change_from_first": difficulty_audio_from_first,
                "z1_visual_difficulty_change_from_first": difficulty_visual_from_first,
                "z1_gap_change_from_first": gap_from_first,
                "z1_audio_difficulty_change_from_previous": difficulty_audio_from_previous,
                "z1_visual_difficulty_change_from_previous": difficulty_visual_from_previous,
                "z1_gap_change_from_previous": gap_from_previous,
            }
        )

        state["previous_audio_centroid"][class_id] = current_audio_centroid.copy()
        state["previous_visual_centroid"][class_id] = current_visual_centroid.copy()
        state["previous_audio_difficulty"][class_id] = audio_g["difficulty_percentile"]
        state["previous_visual_difficulty"][class_id] = visual_g["difficulty_percentile"]
        state["previous_gap"][class_id] = gap_signed
        rows.append(row)

    return rows


def initialize_dynamic_state() -> Dict[str, Dict[int, Any]]:
    return {
        "first_audio_centroid": {},
        "first_visual_centroid": {},
        "previous_audio_centroid": {},
        "previous_visual_centroid": {},
        "first_audio_difficulty": {},
        "first_visual_difficulty": {},
        "previous_audio_difficulty": {},
        "previous_visual_difficulty": {},
        "first_gap": {},
        "previous_gap": {},
    }


def build_z0_rows(
    step: int,
    class_names: Sequence[str],
    geometry: Dict[str, Dict[int, Dict[str, Any]]],
    class_num_per_step: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        audio_g = geometry["z0_audio"][class_id]
        visual_g = geometry["z0_visual"][class_id]
        signed = audio_g["difficulty_percentile"] - visual_g["difficulty_percentile"]
        row: Dict[str, Any] = {
            "step": step,
            "class_id": class_id,
            "class_name": class_name,
            "first_seen_step": class_id // class_num_per_step,
            "task_age": step - class_id // class_num_per_step,
            "z0_difficulty_gap_signed_audio_minus_visual": signed,
            "z0_difficulty_gap_abs": abs(signed),
            "z0_binary_state": f"A_{audio_g['difficulty_binary']}_V_{visual_g['difficulty_binary']}",
            "z0_ternary_state": f"A_{audio_g['difficulty_ternary']}_V_{visual_g['difficulty_ternary']}",
            "z0_ternary_mismatch": int(
                audio_g["difficulty_ternary"] != visual_g["difficulty_ternary"]
            ),
            "z0_extreme_easy_hard_mismatch": int(
                {audio_g["difficulty_ternary"], visual_g["difficulty_ternary"]}
                == {"easy", "hard"}
            ),
            "z0_audio_visual_neighbor_jaccard": jaccard(
                audio_g["neighbor_ids"], visual_g["neighbor_ids"]
            ),
        }
        row.update(geometry_to_columns("z0_audio", audio_g))
        row.update(geometry_to_columns("z0_visual", visual_g))
        rows.append(row)
    return rows


def build_contrastive_effect_rows(
    full_rows: Sequence[Dict[str, Any]],
    no_z1_rows: Sequence[Dict[str, Any]],
    full_counterfactual: Sequence[Dict[str, Any]],
    no_z1_counterfactual: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    full_idx = {(r["step"], r["class_id"]): r for r in full_rows}
    no_idx = {(r["step"], r["class_id"]): r for r in no_z1_rows}
    cf_full_idx = {
        (r["step"], r["class_id"], r["intervention"]): r
        for r in full_counterfactual
    }
    cf_no_idx = {
        (r["step"], r["class_id"], r["intervention"]): r
        for r in no_z1_counterfactual
    }

    detail_rows: List[Dict[str, Any]] = []
    geometry_metrics = [
        "recall",
        "f1",
        "z1_difficulty_gap_abs",
        "z1_difficulty_gap_signed_audio_minus_visual",
        "paired_av_cosine",
        "centroid_av_cosine",
        "audio_visual_neighbor_jaccard",
        "z1_audio_intra_dispersion",
        "z1_visual_intra_dispersion",
        "z1_audio_nearest_distance",
        "z1_visual_nearest_distance",
        "z1_audio_normalized_margin",
        "z1_visual_normalized_margin",
        "z1_audio_knn_purity",
        "z1_visual_knn_purity",
        "z1_audio_difficulty_percentile",
        "z1_visual_difficulty_percentile",
    ]

    for key in sorted(set(full_idx) & set(no_idx)):
        full = full_idx[key]
        no = no_idx[key]
        base = {
            "step": key[0],
            "class_id": key[1],
            "class_name": full["class_name"],
            "first_seen_step": full["first_seen_step"],
            "task_age": full["task_age"],
            "no_z1_binary_state": no["z1_binary_state"],
            "no_z1_ternary_state": no["z1_ternary_state"],
            "full_binary_state": full["z1_binary_state"],
            "full_ternary_state": full["z1_ternary_state"],
        }
        for metric in geometry_metrics:
            base[f"no_z1_{metric}"] = no.get(metric, "")
            base[f"full_{metric}"] = full.get(metric, "")
            if no.get(metric, "") != "" and full.get(metric, "") != "":
                base[f"delta_full_minus_no_z1_{metric}"] = float(
                    full[metric] - no[metric]
                )
            else:
                base[f"delta_full_minus_no_z1_{metric}"] = ""
        detail_rows.append(base)

        interventions = sorted(
            {
                intervention
                for step, class_id, intervention in cf_full_idx
                if (step, class_id) == key
            }
            & {
                intervention
                for step, class_id, intervention in cf_no_idx
                if (step, class_id) == key
            }
        )
        for intervention in interventions:
            full_cf = cf_full_idx[(key[0], key[1], intervention)]
            no_cf = cf_no_idx[(key[0], key[1], intervention)]
            cf_row = dict(base)
            cf_row["intervention"] = intervention
            for metric in [
                "audio_true_logit_drop",
                "visual_true_logit_drop",
                "audio_margin_drop",
                "visual_margin_drop",
                "audio_necessity",
                "visual_necessity",
                "audio_interference",
                "visual_interference",
                "synergy_rate",
                "redundancy_rate",
                "audio_minus_visual_logit_drop",
                "audio_minus_visual_margin_drop",
            ]:
                cf_row[f"no_z1_{metric}"] = no_cf[metric]
                cf_row[f"full_{metric}"] = full_cf[metric]
                cf_row[f"delta_full_minus_no_z1_{metric}"] = float(
                    full_cf[metric] - no_cf[metric]
                )
            detail_rows.append(cf_row)

    # Aggregate only rows with an intervention for counterfactual metrics, and
    # one geometry-only row per class for geometry metrics.
    grouped: DefaultDict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        intervention = row.get("intervention", "geometry")
        grouped[(int(row["step"]), str(row["no_z1_ternary_state"]), intervention)].append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for (step, status, intervention), rows in sorted(grouped.items()):
        aggregate: Dict[str, Any] = {
            "step": step,
            "no_z1_ternary_state": status,
            "intervention": intervention,
            "num_classes": len(rows),
        }
        delta_keys = sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("delta_full_minus_no_z1_")
                and row.get(key, "") != ""
            }
        )
        for key in delta_keys:
            values = [float(row[key]) for row in rows if row.get(key, "") != ""]
            aggregate[f"mean_{key}"] = float(np.mean(values)) if values else ""
        aggregate_rows.append(aggregate)

    return detail_rows, aggregate_rows


def build_z1_z2_ternary_composition(
    modality_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize the 3 x 3 modality states against z2 joint E/M/H.

    Each row corresponds to one Full-AVCIL step and one audio/visual ternary
    state. It reports how often z2 is easy, medium, or hard, making cases such
    as A_easy_V_hard_J_medium explicit.
    """
    grouped: DefaultDict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in modality_rows:
        if row.get("model") != "full":
            continue
        grouped[(int(row["step"]), str(row["z1_ternary_state"]))].append(row)

    output: List[Dict[str, Any]] = []
    for (step, av_state), rows in sorted(grouped.items()):
        total = len(rows)
        joint_counts = {
            state: sum(r["z2_difficulty_ternary"] == state for r in rows)
            for state in ["easy", "medium", "hard"]
        }
        output.append(
            {
                "step": step,
                "z1_ternary_state": av_state,
                "num_classes": total,
                "z2_easy_count": joint_counts["easy"],
                "z2_medium_count": joint_counts["medium"],
                "z2_hard_count": joint_counts["hard"],
                "z2_easy_ratio": safe_div(joint_counts["easy"], total),
                "z2_medium_ratio": safe_div(joint_counts["medium"], total),
                "z2_hard_ratio": safe_div(joint_counts["hard"], total),
                "mean_recall": float(np.mean([float(r["recall"]) for r in rows])),
                "mean_f1": float(np.mean([float(r["f1"]) for r in rows])),
                "mean_z1_difficulty_gap_abs": float(
                    np.mean([float(r["z1_difficulty_gap_abs"]) for r in rows])
                ),
                "mean_z2_difficulty_percentile": float(
                    np.mean([float(r["z2_fusion_difficulty_percentile"]) for r in rows])
                ),
            }
        )
    return output


def build_difficulty_contribution_correlations(
    modality_rows: Sequence[Dict[str, Any]],
    counterfactual_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    modality_idx = {
        (r["model"], r["step"], r["class_id"]): r for r in modality_rows
    }
    grouped: DefaultDict[Tuple[str, int, str], List[Dict[str, float]]] = defaultdict(list)
    for cf in counterfactual_rows:
        key = (cf["model"], cf["step"], cf["class_id"])
        if key not in modality_idx:
            continue
        difficulty = modality_idx[key]
        grouped[(cf["model"], int(cf["step"]), cf["intervention"])].append(
            {
                "signed_gap": float(
                    difficulty["z1_difficulty_gap_signed_audio_minus_visual"]
                ),
                "abs_gap": float(difficulty["z1_difficulty_gap_abs"]),
                "contribution_asymmetry": float(
                    cf["audio_minus_visual_logit_drop"]
                ),
                "margin_asymmetry": float(cf["audio_minus_visual_margin_drop"]),
                "mean_interference": 0.5
                * float(cf["audio_interference"] + cf["visual_interference"]),
                "max_interference": max(
                    float(cf["audio_interference"]),
                    float(cf["visual_interference"]),
                ),
            }
        )

    output: List[Dict[str, Any]] = []
    for (model_name, step, intervention), rows in sorted(grouped.items()):
        signed_gap = [r["signed_gap"] for r in rows]
        abs_gap = [r["abs_gap"] for r in rows]
        contribution = [r["contribution_asymmetry"] for r in rows]
        margin = [r["margin_asymmetry"] for r in rows]
        mean_interference = [r["mean_interference"] for r in rows]
        max_interference = [r["max_interference"] for r in rows]
        output.append(
            {
                "model": model_name,
                "step": step,
                "intervention": intervention,
                "num_classes": len(rows),
                "pearson_signed_gap_vs_audio_minus_visual_logit_drop": pearson_corr(
                    signed_gap, contribution
                ),
                "spearman_signed_gap_vs_audio_minus_visual_logit_drop": spearman_corr(
                    signed_gap, contribution
                ),
                "pearson_signed_gap_vs_audio_minus_visual_margin_drop": pearson_corr(
                    signed_gap, margin
                ),
                "spearman_signed_gap_vs_audio_minus_visual_margin_drop": spearman_corr(
                    signed_gap, margin
                ),
                "pearson_abs_gap_vs_mean_interference": pearson_corr(
                    abs_gap, mean_interference
                ),
                "spearman_abs_gap_vs_mean_interference": spearman_corr(
                    abs_gap, mean_interference
                ),
                "pearson_abs_gap_vs_max_interference": pearson_corr(
                    abs_gap, max_interference
                ),
                "spearman_abs_gap_vs_max_interference": spearman_corr(
                    abs_gap, max_interference
                ),
            }
        )
    return output


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=dataset_type, required=True)
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--meta_root", type=str, required=True)
    parser.add_argument("--full_ckpt_dir", type=str, required=True)
    parser.add_argument("--no_z1_ckpt_dir", type=str, required=True)
    parser.add_argument("--order_name", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)

    parser.add_argument("--modality", type=str, default="audio-visual")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--class_num_per_step", type=int, default=10)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--topk_centroid", type=int, default=5)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument(
        "--difficulty_definition",
        type=str,
        default="normalized_margin",
        choices=["normalized_margin", "margin_purity", "legacy_four_component"],
        help=(
            "Main difficulty statistic. normalized_margin is recommended for the "
            "paper; margin_purity is the equal average of normalized-margin and "
            "kNN-purity difficulty percentiles."
        ),
    )
    parser.add_argument(
        "--easy_quantile",
        type=float,
        default=0.30,
        help="Upper percentile boundary for Easy (default gives Easy 30%%).",
    )
    parser.add_argument(
        "--hard_quantile",
        type=float,
        default=0.70,
        help="Lower percentile boundary for Hard (default gives Hard 30%%).",
    )

    parser.add_argument(
        "--z1_interventions",
        type=str,
        default="zero,mean,perm",
        help="Comma-separated subset of zero,mean,perm",
    )
    parser.add_argument("--permutation_repeats", type=int, default=5)
    parser.add_argument(
        "--compute_input_zero",
        action="store_true",
        help="Also run end-to-end z0 audio/visual zero interventions.",
    )
    parser.add_argument(
        "--save_sample_counterfactual",
        action="store_true",
        help="Save large sample-level counterfactual CSV files.",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save labels/logits/raw z0-z2 arrays for each step and model.",
    )

    args = parser.parse_args()
    if not (0.0 < args.easy_quantile < args.hard_quantile < 1.0):
        raise ValueError(
            "Require 0 < --easy_quantile < --hard_quantile < 1."
        )
    setup_seed(args.seed)

    out_root = Path(args.out_root)
    core_dir = out_root / "core"
    detail_dir = out_root / "detail"
    cache_dir = out_root / "feature_cache"
    ensure_dir(core_dir)
    ensure_dir(detail_dir)
    if args.cache_features:
        ensure_dir(cache_dir)

    full_checkpoints = discover_checkpoints(args.full_ckpt_dir)
    no_z1_checkpoints = discover_checkpoints(args.no_z1_ckpt_dir)
    common_steps = sorted(set(full_checkpoints) & set(no_z1_checkpoints))
    requested_steps = parse_int_list(args.steps)
    analysis_steps = common_steps if requested_steps is None else requested_steps
    missing = [step for step in analysis_steps if step not in common_steps]
    if missing:
        raise ValueError(f"Steps absent from one checkpoint directory: {missing}")
    if not analysis_steps:
        raise ValueError("No common checkpoints to analyze")

    interventions = parse_str_list(args.z1_interventions)
    invalid_interventions = set(interventions) - {"zero", "mean", "perm"}
    if invalid_interventions:
        raise ValueError(f"Unknown z1 interventions: {sorted(invalid_interventions)}")
    if "perm" in interventions and args.permutation_repeats <= 0:
        raise ValueError("--permutation_repeats must be positive")

    print(f"[INFO] device: {device}")
    print(f"[INFO] order: {args.order_name}")
    print(f"[INFO] steps: {analysis_steps}")
    print(f"[INFO] full checkpoints: {args.full_ckpt_dir}")
    print(f"[INFO] no-z1 checkpoints: {args.no_z1_ckpt_dir}")

    test_set = IcaAVELoader(args=args, mode="test", modality=args.modality)
    id_to_category = {
        int(value): key for key, value in test_set.category_encode_dict.items()
    }

    z0_rows_all: List[Dict[str, Any]] = []
    modality_rows_all: List[Dict[str, Any]] = []
    counterfactual_rows_all: List[Dict[str, Any]] = []
    sample_counterfactual_rows_all: List[Dict[str, Any]] = []
    consistency_rows: List[Dict[str, Any]] = []

    dynamic_states = {
        "full": initialize_dynamic_state(),
        "no_z1": initialize_dynamic_state(),
    }
    best_f1: Dict[str, Dict[int, float]] = {"full": {}, "no_z1": {}}
    best_recall: Dict[str, Dict[int, float]] = {"full": {}, "no_z1": {}}

    for step in analysis_steps:
        num_seen_classes = min(
            args.num_classes, (step + 1) * args.class_num_per_step
        )
        set_dataset_to_seen_classes(test_set, num_seen_classes)
        class_names = [
            id_to_category.get(class_id, f"class_{class_id}")
            for class_id in range(num_seen_classes)
        ]
        print(
            f"\n[STEP {step}] seen_classes={num_seen_classes}, "
            f"test_samples={len(test_set)}"
        )

        step_outputs: Dict[str, Dict[str, np.ndarray]] = {}
        step_geometry_by_model: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
        step_centroids_by_model: Dict[str, Dict[str, np.ndarray]] = {}
        step_performance_by_model: Dict[str, Dict[int, Dict[str, float]]] = {}

        for model_name, checkpoint in [
            ("full", full_checkpoints[step]),
            ("no_z1", no_z1_checkpoints[step]),
        ]:
            model = load_model(checkpoint)
            output = run_model_inference(
                model,
                test_set,
                batch_size=args.infer_batch_size,
                num_workers=args.num_workers,
                description=f"[{model_name} step {step}]",
                compute_input_zero=args.compute_input_zero,
            )
            weight, bias = linear_classifier_parameters(model)

            performance = per_class_classification_metrics(
                output["labels"], output["logits"], num_seen_classes
            )
            for class_id in range(num_seen_classes):
                current_f1 = performance[class_id]["f1"]
                current_recall = performance[class_id]["recall"]
                previous_best_f1 = best_f1[model_name].get(class_id, current_f1)
                previous_best_recall = best_recall[model_name].get(
                    class_id, current_recall
                )
                performance[class_id]["best_f1"] = max(previous_best_f1, current_f1)
                performance[class_id]["forget_f1"] = max(
                    0.0, previous_best_f1 - current_f1
                )
                performance[class_id]["best_recall"] = max(
                    previous_best_recall, current_recall
                )
                performance[class_id]["forget_recall"] = max(
                    0.0, previous_best_recall - current_recall
                )
                best_f1[model_name][class_id] = performance[class_id]["best_f1"]
                best_recall[model_name][class_id] = performance[class_id][
                    "best_recall"
                ]

            geometry: Dict[str, Dict[int, Dict[str, Any]]] = {}
            centroids: Dict[str, np.ndarray] = {}
            layer_arrays = {
                "z1_audio": output["z1_audio_raw"],
                "z1_visual": output["z1_visual_raw"],
                "z2_fusion": output["z2_fusion_raw"],
                "attn_visual_pooled": output["attn_visual_pooled_raw"],
            }
            for layer, features in layer_arrays.items():
                print(f"[STEP {step}] geometry {model_name}/{layer}")
                geometry[layer], centroids[layer] = compute_geometry(
                    features,
                    output["labels"],
                    class_names,
                    topk_centroid=args.topk_centroid,
                    knn_k=args.knn_k,
                    knn_chunk_size=args.knn_chunk_size,
                    description=f"{model_name}/{layer}",
                    difficulty_definition=args.difficulty_definition,
                    easy_quantile=args.easy_quantile,
                    hard_quantile=args.hard_quantile,
                )

            alignment = compute_crossmodal_alignment(
                output["z1_audio_raw"],
                output["z1_visual_raw"],
                output["labels"],
                centroids["z1_audio"],
                centroids["z1_visual"],
                num_seen_classes,
            )
            modality_rows = build_modality_rows_for_model(
                model_name=model_name,
                step=step,
                class_names=class_names,
                performance=performance,
                geometry=geometry,
                centroids=centroids,
                alignment=alignment,
                class_num_per_step=args.class_num_per_step,
                state=dynamic_states[model_name],
            )
            modality_rows_all.extend(modality_rows)

            counterfactual_logits = build_counterfactual_logits(
                output,
                classifier_weight=weight,
                classifier_bias=bias,
                interventions=interventions,
                permutation_repeats=args.permutation_repeats,
                seed=args.seed + 1000 * step + (0 if model_name == "full" else 1),
                include_input_zero=args.compute_input_zero,
            )
            cf_class_rows, cf_sample_rows = aggregate_counterfactual_by_class(
                normal_logits=output["logits"],
                counterfactuals=counterfactual_logits,
                labels=output["labels"],
                class_names=class_names,
                model_name=model_name,
                step=step,
                class_num_per_step=args.class_num_per_step,
                save_sample_rows=args.save_sample_counterfactual,
            )
            counterfactual_rows_all.extend(cf_class_rows)
            sample_counterfactual_rows_all.extend(cf_sample_rows)

            overall_acc = float(
                (output["preds"] == output["labels"].astype(np.int64)).mean()
            )
            reconstructed_logits = classify_np(
                output["z2_fusion_raw"], weight, bias
            )
            max_logit_error = float(
                np.max(np.abs(reconstructed_logits - output["logits"]))
            )
            consistency_rows.append(
                {
                    "model": model_name,
                    "step": step,
                    "overall_accuracy": overall_acc,
                    "num_samples": len(output["labels"]),
                    "max_abs_classifier_reconstruction_error": max_logit_error,
                }
            )
            if max_logit_error > 1e-4:
                raise RuntimeError(
                    f"Classifier reconstruction mismatch at {model_name}/step {step}: "
                    f"{max_logit_error}. Raw z1/z2 extraction is inconsistent."
                )

            if args.cache_features:
                model_cache_dir = cache_dir / f"step_{step}" / model_name
                ensure_dir(model_cache_dir)
                for key in [
                    "labels",
                    "logits",
                    "z0_audio_raw",
                    "z0_visual_uniform_raw",
                    "attn_visual_pooled_raw",
                    "z1_audio_raw",
                    "z1_visual_raw",
                    "z2_fusion_raw",
                ]:
                    np.save(model_cache_dir / f"{key}.npy", output[key])

            step_outputs[model_name] = output
            step_geometry_by_model[model_name] = geometry
            step_centroids_by_model[model_name] = centroids
            step_performance_by_model[model_name] = performance

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not np.array_equal(
            step_outputs["full"]["labels"], step_outputs["no_z1"]["labels"]
        ):
            raise RuntimeError(f"Label ordering mismatch at step {step}")

        # z0 is model-independent. Use the Full run's input arrays once.
        z0_geometry: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for layer, features in [
            ("z0_audio", step_outputs["full"]["z0_audio_raw"]),
            ("z0_visual", step_outputs["full"]["z0_visual_uniform_raw"]),
        ]:
            print(f"[STEP {step}] geometry input/{layer}")
            z0_geometry[layer], _ = compute_geometry(
                features,
                step_outputs["full"]["labels"],
                class_names,
                topk_centroid=args.topk_centroid,
                knn_k=args.knn_k,
                knn_chunk_size=args.knn_chunk_size,
                description=f"input/{layer}",
                difficulty_definition=args.difficulty_definition,
                easy_quantile=args.easy_quantile,
                hard_quantile=args.hard_quantile,
            )
        z0_rows_all.extend(
            build_z0_rows(
                step,
                class_names,
                z0_geometry,
                class_num_per_step=args.class_num_per_step,
            )
        )

        # Persist after every checkpoint so a later failure does not erase work.
        write_csv(detail_dir / "z0_difficulty_by_step.csv", z0_rows_all)
        write_csv(
            detail_dir / "modality_difficulty_by_class_step.csv",
            modality_rows_all,
        )
        write_csv(
            detail_dir / "counterfactual_contribution_by_class.csv",
            counterfactual_rows_all,
        )
        write_csv(detail_dir / "checkpoint_consistency.csv", consistency_rows)
        if args.save_sample_counterfactual:
            write_csv(
                detail_dir / "counterfactual_contribution_by_sample.csv",
                sample_counterfactual_rows_all,
            )

        del step_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Core outputs
    # -------------------------------------------------------------------------
    final_step = max(analysis_steps)
    z0_final = [row for row in z0_rows_all if int(row["step"]) == final_step]
    full_mismatch = [
        row for row in modality_rows_all if row["model"] == "full"
    ]

    contrastive_detail, contrastive_aggregate = build_contrastive_effect_rows(
        [row for row in modality_rows_all if row["model"] == "full"],
        [row for row in modality_rows_all if row["model"] == "no_z1"],
        [row for row in counterfactual_rows_all if row["model"] == "full"],
        [row for row in counterfactual_rows_all if row["model"] == "no_z1"],
    )
    correlations = build_difficulty_contribution_correlations(
        modality_rows_all, counterfactual_rows_all
    )
    z1_z2_ternary_composition = build_z1_z2_ternary_composition(
        modality_rows_all
    )

    write_csv(core_dir / "01_z0_intrinsic_difficulty_final_step.csv", z0_final)
    write_csv(core_dir / "02_full_z1_modality_mismatch_by_step.csv", full_mismatch)
    write_csv(
        core_dir / "03_counterfactual_modality_contribution_by_class.csv",
        counterfactual_rows_all,
    )
    write_csv(
        core_dir / "04_difficulty_contribution_correlations.csv",
        correlations,
    )
    write_csv(
        core_dir / "05_z1_contrastive_effect_by_status.csv",
        contrastive_aggregate,
    )
    write_csv(
        core_dir / "06_z1_z2_ternary_composition.csv",
        z1_z2_ternary_composition,
    )
    write_csv(
        detail_dir / "z1_contrastive_effect_by_class.csv",
        contrastive_detail,
    )

    summary = {
        "order_name": args.order_name,
        "dataset": args.dataset,
        "feature_root": args.feature_root,
        "meta_root": args.meta_root,
        "full_ckpt_dir": args.full_ckpt_dir,
        "no_z1_ckpt_dir": args.no_z1_ckpt_dir,
        "available_common_steps": common_steps,
        "analyzed_steps": analysis_steps,
        "final_step_used_for_z0_intrinsic_table": final_step,
        "device": str(device),
        "z1_interventions": interventions,
        "permutation_repeats": args.permutation_repeats,
        "compute_input_zero": args.compute_input_zero,
        "difficulty_definition": args.difficulty_definition,
        "easy_quantile": args.easy_quantile,
        "hard_quantile": args.hard_quantile,
        "interpretation_notes": {
            "z0_visual": (
                "Uniform mean over temporal and spatial visual input tokens; "
                "independent of audio-guided attention."
            ),
            "z1_direct_branch_intervention": (
                "Uses raw z1 and the original classifier. Removing audio does not "
                "remove audio information already embedded in the attention-conditioned "
                "visual branch."
            ),
            "z0_input_zero": (
                "End-to-end zero-input sensitivity, not audio-only/visual-only accuracy."
            ),
            "difficulty": (
                "The selected within-step percentile definition is recorded in "
                "difficulty_definition. The default is reverse normalized-margin "
                "percentile. Ternary states use Easy/Medium/Hard = 30/40/30."
            ),
            "causal_limit": (
                "Full versus No-z1 isolates z1 contrastive because both runs retain "
                "attention distillation. Causal analysis of attention distillation "
                "requires a matched No-attention run."
            ),
        },
        "core_outputs": {
            "01_z0_intrinsic_difficulty_final_step.csv": (
                "Input-space audio/visual difficulty over all classes available at the final step."
            ),
            "02_full_z1_modality_mismatch_by_step.csv": (
                "Main evidence that mismatch remains in Full AVCIL and changes over CL steps."
            ),
            "03_counterfactual_modality_contribution_by_class.csv": (
                "Original-classifier branch/input interventions for Full and No-z1."
            ),
            "04_difficulty_contribution_correlations.csv": (
                "Class-level association between difficulty mismatch and contribution/interference."
            ),
            "05_z1_contrastive_effect_by_status.csv": (
                "Full minus No-z1 effects grouped by the No-z1 E/M/H state."
            ),
            "06_z1_z2_ternary_composition.csv": (
                "Counts and ratios of z2 joint Easy/Medium/Hard within every "
                "audio/visual z1 Easy/Medium/Hard state."
            ),
        },
    }
    save_json(summary, out_root / "summary.json")

    if args.dataset != "AVE" and hasattr(test_set, "close_visual_features_h5"):
        test_set.close_visual_features_h5()

    print("\n[DONE]")
    print(f"[INFO] Results written to: {out_root}")
    print(f"[INFO] Start with: {core_dir}")


if __name__ == "__main__":
    main()
