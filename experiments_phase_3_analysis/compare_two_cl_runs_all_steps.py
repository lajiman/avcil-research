#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two complete class-incremental runs across every available checkpoint.

The script is organized around five analysis questions:

1. Why does overall accuracy form a U shape?
   -> core/01_u_shape_diagnosis.csv

2. Which arrival tasks/classes create the gain of Model B over Model A?
   -> core/02_taskwise_accuracy_delta.csv
   -> detail/per_class_performance.csv

3. Does representation geometry improve over incremental steps?
   -> core/03_geometry_trajectory_by_group.csv
   -> detail/class_geometry_by_step.csv
   -> detail/centroid_neighbors_by_step.csv

4. Are old/hard classes confused with later/easier classes?
   -> core/04_confusion_direction_by_group.csv

5. Are per-class performance gains associated with geometry gains?
   -> core/05_geometry_performance_correlation.csv

Expected checkpoint layout:
    xxx/save/<experiment_name>/step_0_best_model.pkl
    ...
    xxx/save/<experiment_name>/step_9_best_model.pkl

Expected metrics CSV layout:
    xxx/save/metrics/<experiment_name>/per_class_metrics.csv

The CSV path is derived automatically from the checkpoint directory, but can
also be supplied explicitly.

Model-output assumption:
    model(..., out_features=True, out_feature_before_fusion=True) returns
        logits, z2_fusion, z1_audio, z1_visual
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from dataloader_ours import IcaAVELoader
from model.audio_visual_model_incremental import IncreAudioVisualNet  # noqa: F401


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Generic utilities
# ============================================================================
def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_csv(
    path: str,
    rows: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def dataset_type(value: str) -> str:
    if value in {"AVE", "ksounds"} or "VGGSound" in value:
        return value
    raise argparse.ArgumentTypeError(
        "dataset must be 'AVE', 'ksounds', or contain 'VGGSound'"
    )


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    return result


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_distance_vec(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(1.0 - np.dot(a, b) / denom)


def arrival_group_for_task(task_step: int, total_steps: int) -> str:
    """For ten tasks: early=0..2, middle=3..6, late=7..9."""
    if total_steps <= 1:
        return "all"
    early_cut = max(1, int(np.floor(total_steps * 0.3)))
    late_cut = max(early_cut + 1, int(np.floor(total_steps * 0.7)))
    if task_step < early_cut:
        return "early"
    if task_step < late_cut:
        return "middle"
    return "late"


def preferred_direction(metric: str) -> str:
    if metric in {
        "intra_dispersion",
        "centroid_drift_from_previous_analyzed_step",
        "centroid_drift_from_first_reference_step",
    }:
        return "lower"
    if metric in {
        "nearest_distance",
        "normalized_margin",
        "knn_purity",
    }:
        return "higher"
    return "descriptive"


def oriented_improvement(raw_delta_b_minus_a: float, metric: str) -> float:
    direction = preferred_direction(metric)
    if direction == "lower":
        return -raw_delta_b_minus_a
    return raw_delta_b_minus_a


# ============================================================================
# Checkpoint and metrics-path discovery
# ============================================================================
_CHECKPOINT_RE = re.compile(
    r"^step[_-]?(\d+)_best_model(?:\.pkl)?$",
    flags=re.IGNORECASE,
)


def discover_checkpoints(checkpoint_dir: str) -> Dict[int, str]:
    directory = Path(checkpoint_dir).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {directory}")

    result: Dict[int, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = _CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step in result:
            raise ValueError(
                f"Duplicate checkpoint files for step {step} in {directory}: "
                f"{result[step]} and {path}"
            )
        result[step] = str(path)

    if not result:
        raise FileNotFoundError(
            f"No files matching step_<k>_best_model.pkl were found in {directory}"
        )
    return result


def derive_metrics_csv_from_checkpoint_dir(checkpoint_dir: str) -> str:
    """
    Convert:
        xxx/save/<experiment_name>
    to:
        xxx/save/metrics/<experiment_name>/per_class_metrics.csv
    """
    experiment_dir = Path(checkpoint_dir).expanduser().resolve()
    save_dir = experiment_dir.parent

    return str(
        save_dir
        / "metrics"
        / experiment_dir.name
        / "per_class_metrics.csv"
    )


def resolve_metrics_csv(
    explicit_path: Optional[str],
    checkpoint_dir: str,
) -> str:
    path = explicit_path or derive_metrics_csv_from_checkpoint_dir(checkpoint_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"per_class_metrics.csv not found: {path}")
    return path


# ============================================================================
# per_class_metrics.csv
# ============================================================================
def read_per_class_metrics(path: str) -> List[Dict[str, Any]]:
    required = {
        "step",
        "class_id",
        "category_name",
        "first_seen_step",
        "support",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "best_f1",
        "forget_f1",
        "forgetting",
        "overall_acc",
    }
    int_fields = {
        "step",
        "class_id",
        "first_seen_step",
        "support",
        "tp",
        "fp",
        "fn",
    }
    float_fields = {
        "precision",
        "recall",
        "f1",
        "best_f1",
        "forget_f1",
        "forgetting",
        "overall_acc",
    }

    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for raw in reader:
            row: Dict[str, Any] = dict(raw)
            for key in int_fields:
                row[key] = int(raw[key])
            for key in float_fields:
                row[key] = float(raw[key])
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {path}")
    return rows


def index_metric_rows(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in rows:
        key = (int(row["step"]), int(row["class_id"]))
        if key in result:
            raise ValueError(f"Duplicate metrics row: step/class={key}")
        result[key] = row
    return result


def validate_metrics_pair(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
) -> None:
    idx_a = index_metric_rows(rows_a)
    idx_b = index_metric_rows(rows_b)
    if set(idx_a) != set(idx_b):
        raise ValueError(
            "The two metrics CSV files do not contain identical (step, class_id) keys."
        )
    for key, row_a in idx_a.items():
        row_b = idx_b[key]
        for field in ["first_seen_step", "support", "category_name"]:
            if row_a[field] != row_b[field]:
                raise ValueError(f"Metrics mismatch at {key}, field={field}")


def rows_at_step(
    rows: Sequence[Dict[str, Any]],
    step: int,
) -> List[Dict[str, Any]]:
    return [row for row in rows if int(row["step"]) == step]


def metric_map_at_step(
    rows: Sequence[Dict[str, Any]],
    step: int,
) -> Dict[int, Dict[str, Any]]:
    return {
        int(row["class_id"]): row
        for row in rows
        if int(row["step"]) == step
    }


def aggregate_metric_subset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "num_classes": 0,
            "support": 0,
            "tp": 0,
            "micro_acc": None,
            "macro_f1": None,
            "mean_forgetting": None,
        }
    support = int(sum(int(row["support"]) for row in rows))
    tp = int(sum(int(row["tp"]) for row in rows))
    return {
        "num_classes": len(rows),
        "support": support,
        "tp": tp,
        "micro_acc": safe_div(tp, support),
        "macro_f1": float(np.mean([float(row["f1"]) for row in rows])),
        "mean_forgetting": float(
            np.mean([float(row["forgetting"]) for row in rows])
        ),
    }


def unique_overall_acc(step_rows: Sequence[Dict[str, Any]], path: str, step: int) -> float:
    values = np.asarray([float(row["overall_acc"]) for row in step_rows])
    if len(values) == 0:
        raise ValueError(f"No rows for step {step}: {path}")
    if float(values.max() - values.min()) > 1e-7:
        raise ValueError(f"overall_acc is not constant at step {step}: {path}")
    return float(values[0])


def build_decomposition_single(
    rows: Sequence[Dict[str, Any]],
    csv_path: str,
) -> Dict[int, Dict[str, Any]]:
    steps = sorted({int(row["step"]) for row in rows})
    output: Dict[int, Dict[str, Any]] = {}
    previous_overall: Optional[float] = None

    for step in steps:
        current = rows_at_step(rows, step)
        seen = aggregate_metric_subset(current)
        old = aggregate_metric_subset(
            [row for row in current if int(row["first_seen_step"]) < step]
        )
        new = aggregate_metric_subset(
            [row for row in current if int(row["first_seen_step"]) == step]
        )
        overall = unique_overall_acc(current, csv_path, step)
        old_weight = safe_div(old["support"], seen["support"])
        new_weight = safe_div(new["support"], seen["support"])

        if previous_overall is None:
            overall_change = None
            old_effect = None
            new_effect = None
            residual = None
        else:
            overall_change = overall - previous_overall
            old_effect = (
                old_weight * (float(old["micro_acc"]) - previous_overall)
                if old["micro_acc"] is not None
                else None
            )
            new_effect = (
                new_weight * (float(new["micro_acc"]) - previous_overall)
                if new["micro_acc"] is not None
                else None
            )
            reconstructed = float(old_effect or 0.0) + float(new_effect or 0.0)
            residual = overall_change - reconstructed

        output[step] = {
            "overall_acc": overall,
            "overall_acc_recomputed": seen["micro_acc"],
            "old_acc": old["micro_acc"],
            "new_task_acc": new["micro_acc"],
            "old_macro_f1": old["macro_f1"],
            "new_task_macro_f1": new["macro_f1"],
            "old_mean_forgetting": old["mean_forgetting"],
            "overall_change": overall_change,
            "old_class_change_effect": old_effect,
            "new_task_composition_effect": new_effect,
            "decomposition_residual": residual,
            "num_seen_classes": seen["num_classes"],
        }
        previous_overall = overall
    return output


def dominant_driver(old_effect: Any, new_effect: Any) -> str:
    if old_effect is None or new_effect is None:
        return ""
    old_abs = abs(float(old_effect))
    new_abs = abs(float(new_effect))
    if np.isclose(old_abs, new_abs, rtol=1e-5, atol=1e-12):
        return "balanced"
    return "old_class_change" if old_abs > new_abs else "new_task_composition"


def build_u_shape_rows(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    csv_a: str,
    csv_b: str,
) -> List[Dict[str, Any]]:
    a = build_decomposition_single(rows_a, csv_a)
    b = build_decomposition_single(rows_b, csv_b)
    if set(a) != set(b):
        raise ValueError("Metrics CSV files have different steps.")

    output: List[Dict[str, Any]] = []
    for step in sorted(a):
        row: Dict[str, Any] = {
            "step": step,
            "num_seen_classes": a[step]["num_seen_classes"],
        }
        for field in [
            "overall_acc",
            "overall_acc_recomputed",
            "old_acc",
            "new_task_acc",
            "old_macro_f1",
            "new_task_macro_f1",
            "old_mean_forgetting",
            "overall_change",
            "old_class_change_effect",
            "new_task_composition_effect",
            "decomposition_residual",
        ]:
            value_a = a[step][field]
            value_b = b[step][field]
            row[f"model_a_{field}"] = "" if value_a is None else value_a
            row[f"model_b_{field}"] = "" if value_b is None else value_b
            row[f"delta_b_minus_a_{field}"] = (
                ""
                if value_a is None or value_b is None
                else float(value_b - value_a)
            )
        row["model_a_dominant_driver"] = dominant_driver(
            a[step]["old_class_change_effect"],
            a[step]["new_task_composition_effect"],
        )
        row["model_b_dominant_driver"] = dominant_driver(
            b[step]["old_class_change_effect"],
            b[step]["new_task_composition_effect"],
        )
        output.append(row)
    return output


def build_taskwise_rows(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    idx_a = index_metric_rows(rows_a)
    idx_b = index_metric_rows(rows_b)
    output: List[Dict[str, Any]] = []

    for eval_step in sorted({step for step, _ in idx_a}):
        current_a = [row for (step, _), row in idx_a.items() if step == eval_step]
        current_b = [row for (step, _), row in idx_b.items() if step == eval_step]
        tasks = sorted({int(row["first_seen_step"]) for row in current_a})
        for task in tasks:
            agg_a = aggregate_metric_subset(
                [row for row in current_a if int(row["first_seen_step"]) == task]
            )
            agg_b = aggregate_metric_subset(
                [row for row in current_b if int(row["first_seen_step"]) == task]
            )
            output.append(
                {
                    "eval_step": eval_step,
                    "task_first_seen_step": task,
                    "task_age": eval_step - task,
                    "num_classes": agg_a["num_classes"],
                    "model_a_task_accuracy": agg_a["micro_acc"],
                    "model_b_task_accuracy": agg_b["micro_acc"],
                    "delta_b_minus_a_task_accuracy": float(
                        agg_b["micro_acc"] - agg_a["micro_acc"]
                    ),
                    "model_a_task_macro_f1": agg_a["macro_f1"],
                    "model_b_task_macro_f1": agg_b["macro_f1"],
                    "delta_b_minus_a_task_macro_f1": float(
                        agg_b["macro_f1"] - agg_a["macro_f1"]
                    ),
                }
            )
    return output


def build_taskwise_matrix(
    taskwise_rows: Sequence[Dict[str, Any]],
    field: str,
) -> List[Dict[str, Any]]:
    eval_steps = sorted({int(row["eval_step"]) for row in taskwise_rows})
    task_steps = sorted({int(row["task_first_seen_step"]) for row in taskwise_rows})
    lookup = {
        (int(row["eval_step"]), int(row["task_first_seen_step"])): row
        for row in taskwise_rows
    }
    output: List[Dict[str, Any]] = []
    for eval_step in eval_steps:
        row: Dict[str, Any] = {"eval_step": eval_step}
        for task_step in task_steps:
            item = lookup.get((eval_step, task_step))
            row[f"task_{task_step}"] = "" if item is None else item[field]
        output.append(row)
    return output


def build_per_class_performance_rows(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    total_steps: int,
) -> List[Dict[str, Any]]:
    idx_a = index_metric_rows(rows_a)
    idx_b = index_metric_rows(rows_b)
    output: List[Dict[str, Any]] = []
    for key in sorted(idx_a):
        a = idx_a[key]
        b = idx_b[key]
        task = int(a["first_seen_step"])
        row: Dict[str, Any] = {
            "step": int(a["step"]),
            "class_id": int(a["class_id"]),
            "class_name": str(a["category_name"]),
            "first_seen_step": task,
            "arrival_group": arrival_group_for_task(task, total_steps),
            "task_age": int(a["step"]) - task,
            "support": int(a["support"]),
        }
        for metric in [
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "best_f1",
            "forget_f1",
            "forgetting",
        ]:
            value_a = a[metric]
            value_b = b[metric]
            row[f"model_a_{metric}"] = value_a
            row[f"model_b_{metric}"] = value_b
            row[f"delta_b_minus_a_{metric}"] = value_b - value_a
        output.append(row)
    return output


# ============================================================================
# Dataset, model inference, and prediction metrics
# ============================================================================
def set_dataset_to_seen_classes(dataset: Any, num_seen_classes: int) -> None:
    dataset.current_step_class = np.arange(num_seen_classes)
    dataset.all_current_data_vids = []
    for class_idx in dataset.current_step_class:
        dataset.all_current_data_vids += dataset.all_classId_vid_dict[str(int(class_idx))]


def load_model(checkpoint_path: str) -> Any:
    try:
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference_extract_reps(
    model: Any,
    dataset: Any,
    batch_size: int,
    num_workers: int,
    description: str,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    labels_all: List[torch.Tensor] = []
    preds_all: List[torch.Tensor] = []
    logits_all: List[torch.Tensor] = []
    z1_audio_all: List[torch.Tensor] = []
    z1_visual_all: List[torch.Tensor] = []
    z2_fusion_all: List[torch.Tensor] = []

    for data, labels in tqdm(loader, desc=description, ncols=100):
        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)
        labels = labels.long()

        out = model(
            visual=visual,
            audio=audio,
            out_logits=True,
            out_features=True,
            out_features_norm=False,
            out_feature_before_fusion=True,
        )
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise RuntimeError(
                "Expected at least (logits, fusion_feature, audio_feature, visual_feature)."
            )

        logits = out[0]
        z2_fusion = out[1]
        z1_audio = out[2]
        z1_visual = out[3]

        labels_all.append(labels.cpu())
        logits_all.append(logits.detach().float().cpu())
        preds_all.append(logits.argmax(dim=1).detach().long().cpu())
        z2_fusion_all.append(z2_fusion.detach().float().cpu())
        z1_audio_all.append(z1_audio.detach().float().cpu())
        z1_visual_all.append(z1_visual.detach().float().cpu())

    return {
        "labels": torch.cat(labels_all).numpy(),
        "preds": torch.cat(preds_all).numpy(),
        "logits": torch.cat(logits_all).numpy(),
        "z1_audio": torch.cat(z1_audio_all).numpy(),
        "z1_visual": torch.cat(z1_visual_all).numpy(),
        "z2_fusion": torch.cat(z2_fusion_all).numpy(),
    }


def compute_confusion(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels.astype(np.int64), preds.astype(np.int64)):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[true, pred] += 1
    return cm


# ============================================================================
# Geometry
# ============================================================================
def compute_centroids(
    x_norm: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = x_norm.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)
    for class_id in range(num_classes):
        indices = np.where(labels == class_id)[0]
        counts[class_id] = len(indices)
        if len(indices) > 0:
            centroids[class_id] = x_norm[indices].mean(axis=0)
    centroids = l2_normalize_np(centroids, eps=eps)
    return centroids, counts


def compute_knn_purity(
    x_norm: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    n = len(labels)
    if n <= 1:
        return np.zeros(num_classes, dtype=np.float32)
    k_eff = min(k, n - 1)
    sample_purity = np.zeros(n, dtype=np.float32)

    for start in tqdm(
        range(0, n, chunk_size),
        desc=f"[kNN purity k={k_eff}]",
        ncols=100,
    ):
        end = min(start + chunk_size, n)
        similarities = x_norm[start:end] @ x_norm.T
        local_rows = np.arange(end - start)
        global_cols = np.arange(start, end)
        similarities[local_rows, global_cols] = -np.inf
        neighbor_idx = np.argpartition(
            -similarities,
            kth=k_eff - 1,
            axis=1,
        )[:, :k_eff]
        sample_purity[start:end] = (
            labels[neighbor_idx] == labels[start:end, None]
        ).mean(axis=1)

    class_purity = np.zeros(num_classes, dtype=np.float32)
    for class_id in range(num_classes):
        indices = np.where(labels == class_id)[0]
        class_purity[class_id] = (
            float(sample_purity[indices].mean()) if len(indices) > 0 else 0.0
        )
    return class_purity


def compute_geometry(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    topk_centroid: int,
    knn_k: int,
    knn_chunk_size: int,
    eps: float,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
    num_classes = len(class_names)
    x_norm = l2_normalize_np(features, eps=eps)
    centroids, counts = compute_centroids(x_norm, labels, num_classes, eps)

    intra = np.zeros(num_classes, dtype=np.float32)
    for class_id in range(num_classes):
        indices = np.where(labels == class_id)[0]
        if len(indices) > 0:
            intra[class_id] = float(
                (1.0 - x_norm[indices] @ centroids[class_id]).mean()
            )

    centroid_distance = 1.0 - centroids @ centroids.T
    np.fill_diagonal(centroid_distance, np.inf)
    purity = compute_knn_purity(
        x_norm=x_norm,
        labels=labels.astype(np.int64),
        num_classes=num_classes,
        k=knn_k,
        chunk_size=knn_chunk_size,
    )

    summary: Dict[int, Dict[str, Any]] = {}
    neighbor_rows: List[Dict[str, Any]] = []
    for class_id in range(num_classes):
        if counts[class_id] == 0:
            order: List[int] = []
        else:
            order = np.argsort(centroid_distance[class_id])[:topk_centroid].tolist()
            order = [
                neighbor
                for neighbor in order
                if np.isfinite(centroid_distance[class_id, neighbor])
            ]

        nearest_id = int(order[0]) if order else -1
        nearest_distance = (
            float(centroid_distance[class_id, nearest_id]) if order else 0.0
        )
        normalized_margin = (
            nearest_distance / (float(intra[class_id]) + eps) if order else 0.0
        )
        summary[class_id] = {
            "intra_dispersion": float(intra[class_id]),
            "nearest_distance": nearest_distance,
            "nearest_class_id": nearest_id if nearest_id >= 0 else "",
            "nearest_class_name": (
                class_names[nearest_id] if nearest_id >= 0 else ""
            ),
            "normalized_margin": float(normalized_margin),
            "knn_purity": float(purity[class_id]),
        }
        for rank, neighbor_id in enumerate(order, start=1):
            neighbor_rows.append(
                {
                    "source_class_id": class_id,
                    "source_class_name": class_names[class_id],
                    "neighbor_rank": rank,
                    "neighbor_class_id": int(neighbor_id),
                    "neighbor_class_name": class_names[neighbor_id],
                    "centroid_distance": float(
                        centroid_distance[class_id, neighbor_id]
                    ),
                }
            )
    return summary, neighbor_rows, centroids


def neighbor_ratios_by_class(
    neighbor_rows: Sequence[Dict[str, Any]],
    class_to_task: Dict[int, int],
) -> Dict[int, Dict[str, float]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in neighbor_rows:
        grouped.setdefault(int(row["source_class_id"]), []).append(row)

    output: Dict[int, Dict[str, float]] = {}
    for class_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["neighbor_rank"]))
        source_task = class_to_task[class_id]
        neighbor_tasks = [
            class_to_task[int(row["neighbor_class_id"])] for row in rows
        ]
        denom = max(len(neighbor_tasks), 1)
        output[class_id] = {
            "later_neighbor_ratio_topk": safe_div(
                sum(task > source_task for task in neighbor_tasks), denom
            ),
            "cross_task_neighbor_ratio_topk": safe_div(
                sum(task != source_task for task in neighbor_tasks), denom
            ),
        }
    return output


# ============================================================================
# Multi-step geometry, drift, confusion, and correlations
# ============================================================================
def aggregate_class_geometry_rows(
    class_rows: Sequence[Dict[str, Any]],
    total_steps: int,
) -> List[Dict[str, Any]]:
    """Long-format trajectory table by task, arrival group, and overall."""
    output: List[Dict[str, Any]] = []
    steps = sorted({int(row["step"]) for row in class_rows})
    layers = ["z1_audio", "z1_visual", "z2_fusion"]
    metrics = [
        "intra_dispersion",
        "nearest_distance",
        "normalized_margin",
        "knn_purity",
        "later_neighbor_ratio_topk",
        "cross_task_neighbor_ratio_topk",
        "centroid_drift_from_previous_analyzed_step",
        "centroid_drift_from_first_reference_step",
    ]

    for step in steps:
        step_rows = [row for row in class_rows if int(row["step"]) == step]
        groups: List[Tuple[str, str, List[Dict[str, Any]]]] = []
        for task in sorted({int(row["first_seen_step"]) for row in step_rows}):
            groups.append(
                (
                    "task",
                    str(task),
                    [row for row in step_rows if int(row["first_seen_step"]) == task],
                )
            )
        for group in ["early", "middle", "late"]:
            selected = [row for row in step_rows if row["arrival_group"] == group]
            if selected:
                groups.append(("arrival_group", group, selected))
        groups.append(("overall", "all", step_rows))

        for level, group_id, selected in groups:
            mean_recall_gain = float(
                np.mean([float(row["delta_b_minus_a_recall"]) for row in selected])
            )
            mean_f1_gain = float(
                np.mean([float(row["delta_b_minus_a_f1"]) for row in selected])
            )
            for layer in layers:
                for metric in metrics:
                    key_a = f"model_a_{layer}_{metric}"
                    key_b = f"model_b_{layer}_{metric}"
                    values_a = np.asarray(
                        [
                            float(row[key_a])
                            for row in selected
                            if row.get(key_a, "") != ""
                        ],
                        dtype=np.float64,
                    )
                    values_b = np.asarray(
                        [
                            float(row[key_b])
                            for row in selected
                            if row.get(key_b, "") != ""
                        ],
                        dtype=np.float64,
                    )
                    if len(values_a) == 0 or len(values_b) == 0:
                        continue
                    mean_a = float(values_a.mean())
                    mean_b = float(values_b.mean())
                    delta = mean_b - mean_a
                    output.append(
                        {
                            "step": step,
                            "aggregation_level": level,
                            "group_id": group_id,
                            "num_classes": len(selected),
                            "layer": layer,
                            "geometry_metric": metric,
                            "preferred_direction": preferred_direction(metric),
                            "model_a_mean": mean_a,
                            "model_b_mean": mean_b,
                            "delta_b_minus_a_raw": delta,
                            "oriented_improvement_b_over_a": oriented_improvement(
                                delta, metric
                            ),
                            "mean_delta_b_minus_a_recall": mean_recall_gain,
                            "mean_delta_b_minus_a_f1": mean_f1_gain,
                        }
                    )
    return output


def confusion_direction_for_classes(
    cm: np.ndarray,
    source_classes: Sequence[int],
    class_to_task: Dict[int, int],
) -> Dict[str, float]:
    """Separate correct predictions from same-task errors."""
    support = int(cm[list(source_classes), :].sum())
    counts = {
        "correct": 0,
        "error_to_earlier_task": 0,
        "error_to_same_task_other_class": 0,
        "error_to_later_task": 0,
    }
    for source_class in source_classes:
        source_task = class_to_task[source_class]
        for predicted_class in range(cm.shape[1]):
            count = int(cm[source_class, predicted_class])
            if predicted_class == source_class:
                counts["correct"] += count
                continue
            predicted_task = class_to_task[predicted_class]
            if predicted_task < source_task:
                counts["error_to_earlier_task"] += count
            elif predicted_task > source_task:
                counts["error_to_later_task"] += count
            else:
                counts["error_to_same_task_other_class"] += count

    output: Dict[str, float] = {"support": support}
    for key, count in counts.items():
        output[f"{key}_count"] = count
        output[f"{key}_rate"] = safe_div(count, support)
    return output


def build_confusion_direction_rows(
    step: int,
    cm_a: np.ndarray,
    cm_b: np.ndarray,
    class_to_task: Dict[int, int],
    total_steps: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    groups: List[Tuple[str, str, List[int]]] = []

    for task in sorted(set(class_to_task.values())):
        groups.append(
            (
                "task",
                str(task),
                [class_id for class_id, t in class_to_task.items() if t == task],
            )
        )
    for group in ["early", "middle", "late"]:
        classes = [
            class_id
            for class_id, task in class_to_task.items()
            if arrival_group_for_task(task, total_steps) == group
        ]
        if classes:
            groups.append(("arrival_group", group, classes))
    groups.append(("overall", "all", sorted(class_to_task)))

    for level, group_id, classes in groups:
        a = confusion_direction_for_classes(cm_a, classes, class_to_task)
        b = confusion_direction_for_classes(cm_b, classes, class_to_task)
        row: Dict[str, Any] = {
            "step": step,
            "aggregation_level": level,
            "group_id": group_id,
            "num_source_classes": len(classes),
            "support": a["support"],
        }
        for metric in [
            "correct_rate",
            "error_to_earlier_task_rate",
            "error_to_same_task_other_class_rate",
            "error_to_later_task_rate",
        ]:
            row[f"model_a_{metric}"] = a[metric]
            row[f"model_b_{metric}"] = b[metric]
            row[f"delta_b_minus_a_{metric}"] = b[metric] - a[metric]
        output.append(row)
    return output


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def build_geometry_correlation_rows(
    class_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    layers = ["z1_audio", "z1_visual", "z2_fusion"]
    metrics = [
        "intra_dispersion",
        "nearest_distance",
        "normalized_margin",
        "knn_purity",
        "later_neighbor_ratio_topk",
        "cross_task_neighbor_ratio_topk",
        "centroid_drift_from_previous_analyzed_step",
        "centroid_drift_from_first_reference_step",
    ]

    for step in sorted({int(row["step"]) for row in class_rows}):
        step_rows = [row for row in class_rows if int(row["step"]) == step]
        subsets: List[Tuple[str, List[Dict[str, Any]]]] = [("overall", step_rows)]
        for group in ["early", "middle", "late"]:
            selected = [row for row in step_rows if row["arrival_group"] == group]
            if selected:
                subsets.append((group, selected))

        for subset_name, selected in subsets:
            delta_recall = np.asarray(
                [float(row["delta_b_minus_a_recall"]) for row in selected],
                dtype=np.float64,
            )
            delta_f1 = np.asarray(
                [float(row["delta_b_minus_a_f1"]) for row in selected],
                dtype=np.float64,
            )
            for layer in layers:
                for metric in metrics:
                    raw_values: List[float] = []
                    recall_values: List[float] = []
                    f1_values: List[float] = []
                    key = f"delta_b_minus_a_{layer}_{metric}"
                    for index, row in enumerate(selected):
                        value = row.get(key, "")
                        if value == "":
                            continue
                        raw_values.append(float(value))
                        recall_values.append(float(delta_recall[index]))
                        f1_values.append(float(delta_f1[index]))
                    if len(raw_values) < 3:
                        continue
                    raw = np.asarray(raw_values, dtype=np.float64)
                    oriented = np.asarray(
                        [oriented_improvement(value, metric) for value in raw],
                        dtype=np.float64,
                    )
                    recall = np.asarray(recall_values, dtype=np.float64)
                    f1 = np.asarray(f1_values, dtype=np.float64)
                    output.append(
                        {
                            "step": step,
                            "class_subset": subset_name,
                            "num_classes": len(raw),
                            "layer": layer,
                            "geometry_metric": metric,
                            "preferred_direction": preferred_direction(metric),
                            "pearson_with_delta_recall": pearson_corr(
                                oriented, recall
                            ),
                            "spearman_with_delta_recall": spearman_corr(
                                oriented, recall
                            ),
                            "pearson_with_delta_f1": pearson_corr(oriented, f1),
                            "spearman_with_delta_f1": spearman_corr(oriented, f1),
                        }
                    )
    return output


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=dataset_type, required=True)
    parser.add_argument(
        "--modality",
        type=str,
        default="audio-visual",
        choices=["audio-visual"],
    )
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--meta_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--class_num_per_step", type=int, default=10)

    parser.add_argument("--model_a_ckpt_dir", type=str, required=True)
    parser.add_argument("--model_b_ckpt_dir", type=str, required=True)
    parser.add_argument("--model_a_metrics_csv", type=str, default=None)
    parser.add_argument("--model_b_metrics_csv", type=str, default=None)
    parser.add_argument("--model_a_name", type=str, default="model_a")
    parser.add_argument("--model_b_name", type=str, default="model_b")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps. Default: every common step.",
    )
    parser.add_argument(
        "--cache_feature_steps",
        type=str,
        default=None,
        help=(
            "Optional comma-separated steps whose labels/z1/z2 arrays should be "
            "saved for later t-SNE analysis, e.g. 2,5,9."
        ),
    )
    parser.add_argument("--out_root", type=str, required=True)

    parser.add_argument("--infer_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk_centroid", type=int, default=5)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--save_confusion_matrices", action="store_true")

    args = parser.parse_args()
    setup_seed(args.seed)

    core_dir = os.path.join(args.out_root, "core")
    detail_dir = os.path.join(args.out_root, "detail")
    ensure_dir(core_dir)
    ensure_dir(detail_dir)

    checkpoints_a = discover_checkpoints(args.model_a_ckpt_dir)
    checkpoints_b = discover_checkpoints(args.model_b_ckpt_dir)
    common_steps = sorted(set(checkpoints_a).intersection(checkpoints_b))
    requested_steps = parse_int_list(args.steps)
    analysis_steps = common_steps if requested_steps is None else requested_steps
    missing = [step for step in analysis_steps if step not in common_steps]
    if missing:
        raise ValueError(
            f"Requested steps are not available in both checkpoint directories: {missing}"
        )

    cache_steps = set(parse_int_list(args.cache_feature_steps) or [])
    metrics_csv_a = resolve_metrics_csv(
        args.model_a_metrics_csv, args.model_a_ckpt_dir
    )
    metrics_csv_b = resolve_metrics_csv(
        args.model_b_metrics_csv, args.model_b_ckpt_dir
    )
    metrics_rows_a = read_per_class_metrics(metrics_csv_a)
    metrics_rows_b = read_per_class_metrics(metrics_csv_b)
    validate_metrics_pair(metrics_rows_a, metrics_rows_b)

    csv_steps = sorted({int(row["step"]) for row in metrics_rows_a})
    unavailable_csv_steps = [step for step in analysis_steps if step not in csv_steps]
    if unavailable_csv_steps:
        raise ValueError(
            f"Steps missing from per_class_metrics.csv: {unavailable_csv_steps}"
        )

    total_steps = int(np.ceil(args.num_classes / args.class_num_per_step))

    # Performance outputs use the complete CSV trajectories, even if --steps selects
    # a subset of checkpoints for expensive geometry extraction.
    u_shape_rows = build_u_shape_rows(
        metrics_rows_a, metrics_rows_b, metrics_csv_a, metrics_csv_b
    )
    taskwise_rows = build_taskwise_rows(metrics_rows_a, metrics_rows_b)
    taskwise_delta_matrix = build_taskwise_matrix(
        taskwise_rows, "delta_b_minus_a_task_accuracy"
    )
    taskwise_a_matrix = build_taskwise_matrix(
        taskwise_rows, "model_a_task_accuracy"
    )
    taskwise_b_matrix = build_taskwise_matrix(
        taskwise_rows, "model_b_task_accuracy"
    )
    per_class_performance = build_per_class_performance_rows(
        metrics_rows_a, metrics_rows_b, total_steps
    )

    write_csv(
        os.path.join(core_dir, "01_u_shape_diagnosis.csv"),
        u_shape_rows,
    )
    write_csv(
        os.path.join(core_dir, "02_taskwise_accuracy_delta.csv"),
        taskwise_delta_matrix,
    )
    write_csv(
        os.path.join(detail_dir, "taskwise_accuracy_model_a.csv"),
        taskwise_a_matrix,
    )
    write_csv(
        os.path.join(detail_dir, "taskwise_accuracy_model_b.csv"),
        taskwise_b_matrix,
    )
    write_csv(
        os.path.join(detail_dir, "taskwise_accuracy_long.csv"),
        taskwise_rows,
    )
    write_csv(
        os.path.join(detail_dir, "per_class_performance.csv"),
        per_class_performance,
    )

    print(f"[INFO] device = {device}")
    print(f"[INFO] model A = {args.model_a_name}")
    print(f"[INFO] model B = {args.model_b_name}")
    print(f"[INFO] analyzed checkpoint steps = {analysis_steps}")
    print(f"[INFO] model A metrics = {metrics_csv_a}")
    print(f"[INFO] model B metrics = {metrics_csv_b}")

    test_set = IcaAVELoader(args=args, mode="test", modality=args.modality)
    id_to_category = {
        int(value): key for key, value in test_set.category_encode_dict.items()
    }

    class_geometry_rows: List[Dict[str, Any]] = []
    centroid_neighbor_rows: List[Dict[str, Any]] = []
    confusion_direction_rows: List[Dict[str, Any]] = []
    checkpoint_consistency_rows: List[Dict[str, Any]] = []

    first_centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        "model_a": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
        "model_b": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
    }
    first_centroid_reference_step: Dict[str, Dict[str, Dict[int, int]]] = {
        "model_a": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
        "model_b": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
    }
    previous_centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        "model_a": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
        "model_b": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
    }
    previous_step_for_class: Dict[str, Dict[str, Dict[int, int]]] = {
        "model_a": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
        "model_b": {layer: {} for layer in ["z1_audio", "z1_visual", "z2_fusion"]},
    }

    for step in analysis_steps:
        num_seen_classes = min(
            args.num_classes,
            (step + 1) * args.class_num_per_step,
        )
        set_dataset_to_seen_classes(test_set, num_seen_classes)
        class_names = [
            id_to_category.get(class_id, f"class_{class_id}")
            for class_id in range(num_seen_classes)
        ]
        sample_ids = list(test_set.all_current_data_vids)
        csv_map_a = metric_map_at_step(metrics_rows_a, step)
        csv_map_b = metric_map_at_step(metrics_rows_b, step)
        expected_ids = set(range(num_seen_classes))
        if set(csv_map_a) != expected_ids or set(csv_map_b) != expected_ids:
            raise ValueError(
                f"At step {step}, CSV class IDs are not exactly 0..{num_seen_classes - 1}."
            )
        class_to_task = {
            class_id: int(csv_map_a[class_id]["first_seen_step"])
            for class_id in range(num_seen_classes)
        }

        print(
            f"\n[STEP {step}] seen classes={num_seen_classes}, "
            f"test samples={len(test_set)}"
        )

        model_a = load_model(checkpoints_a[step])
        output_a = run_inference_extract_reps(
            model_a,
            test_set,
            args.infer_batch_size,
            args.num_workers,
            description=f"[A step {step}]",
        )
        del model_a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_b = load_model(checkpoints_b[step])
        output_b = run_inference_extract_reps(
            model_b,
            test_set,
            args.infer_batch_size,
            args.num_workers,
            description=f"[B step {step}]",
        )
        del model_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        labels = output_a["labels"]
        if not np.array_equal(labels, output_b["labels"]):
            raise RuntimeError(f"Inference-label ordering mismatch at step {step}.")

        cm_a = compute_confusion(labels, output_a["preds"], num_seen_classes)
        cm_b = compute_confusion(labels, output_b["preds"], num_seen_classes)
        inference_acc_a = float((labels == output_a["preds"]).mean())
        inference_acc_b = float((labels == output_b["preds"]).mean())
        csv_acc_a = float(next(iter(csv_map_a.values()))["overall_acc"])
        csv_acc_b = float(next(iter(csv_map_b.values()))["overall_acc"])
        checkpoint_consistency_rows.append(
            {
                "step": step,
                "model_a_inference_acc": inference_acc_a,
                "model_a_csv_acc": csv_acc_a,
                "model_a_inference_minus_csv": inference_acc_a - csv_acc_a,
                "model_b_inference_acc": inference_acc_b,
                "model_b_csv_acc": csv_acc_b,
                "model_b_inference_minus_csv": inference_acc_b - csv_acc_b,
            }
        )

        confusion_direction_rows.extend(
            build_confusion_direction_rows(
                step,
                cm_a,
                cm_b,
                class_to_task,
                total_steps,
            )
        )

        if args.save_confusion_matrices:
            matrix_dir = os.path.join(args.out_root, "confusion_matrices")
            ensure_dir(matrix_dir)
            np.save(os.path.join(matrix_dir, f"step_{step}_model_a.npy"), cm_a)
            np.save(os.path.join(matrix_dir, f"step_{step}_model_b.npy"), cm_b)
            np.save(
                os.path.join(matrix_dir, f"step_{step}_delta_b_minus_a.npy"),
                cm_b.astype(np.int64) - cm_a.astype(np.int64),
            )

        step_geometry: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {
            "model_a": {},
            "model_b": {},
        }
        step_centroids: Dict[str, Dict[str, np.ndarray]] = {
            "model_a": {},
            "model_b": {},
        }
        step_neighbor_ratios: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {
            "model_a": {},
            "model_b": {},
        }

        for model_key, model_output in [
            ("model_a", output_a),
            ("model_b", output_b),
        ]:
            for layer in ["z1_audio", "z1_visual", "z2_fusion"]:
                print(f"[STEP {step}] geometry: {model_key}/{layer}")
                summary, neighbors, centroids = compute_geometry(
                    features=model_output[layer],
                    labels=labels,
                    class_names=class_names,
                    topk_centroid=args.topk_centroid,
                    knn_k=args.knn_k,
                    knn_chunk_size=args.knn_chunk_size,
                    eps=args.eps,
                )
                step_geometry[model_key][layer] = summary
                step_centroids[model_key][layer] = centroids
                ratios = neighbor_ratios_by_class(neighbors, class_to_task)
                step_neighbor_ratios[model_key][layer] = ratios

                for neighbor in neighbors:
                    source_id = int(neighbor["source_class_id"])
                    neighbor_id = int(neighbor["neighbor_class_id"])
                    source_task = class_to_task[source_id]
                    neighbor_task = class_to_task[neighbor_id]
                    centroid_neighbor_rows.append(
                        {
                            "step": step,
                            "model": model_key,
                            "layer": layer,
                            **neighbor,
                            "source_first_seen_step": source_task,
                            "neighbor_first_seen_step": neighbor_task,
                            "source_arrival_group": arrival_group_for_task(
                                source_task, total_steps
                            ),
                            "neighbor_arrival_group": arrival_group_for_task(
                                neighbor_task, total_steps
                            ),
                            "neighbor_task_relation": (
                                "earlier"
                                if neighbor_task < source_task
                                else "later"
                                if neighbor_task > source_task
                                else "same"
                            ),
                        }
                    )

        for class_id in range(num_seen_classes):
            task = class_to_task[class_id]
            row: Dict[str, Any] = {
                "step": step,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "first_seen_step": task,
                "arrival_group": arrival_group_for_task(task, total_steps),
                "task_age": step - task,
                "model_a_recall": float(csv_map_a[class_id]["recall"]),
                "model_b_recall": float(csv_map_b[class_id]["recall"]),
                "delta_b_minus_a_recall": float(
                    csv_map_b[class_id]["recall"] - csv_map_a[class_id]["recall"]
                ),
                "model_a_f1": float(csv_map_a[class_id]["f1"]),
                "model_b_f1": float(csv_map_b[class_id]["f1"]),
                "delta_b_minus_a_f1": float(
                    csv_map_b[class_id]["f1"] - csv_map_a[class_id]["f1"]
                ),
                "model_a_forgetting": float(csv_map_a[class_id]["forgetting"]),
                "model_b_forgetting": float(csv_map_b[class_id]["forgetting"]),
                "delta_b_minus_a_forgetting": float(
                    csv_map_b[class_id]["forgetting"]
                    - csv_map_a[class_id]["forgetting"]
                ),
            }

            for model_key in ["model_a", "model_b"]:
                for layer in ["z1_audio", "z1_visual", "z2_fusion"]:
                    geometry = step_geometry[model_key][layer][class_id]
                    ratios = step_neighbor_ratios[model_key][layer][class_id]
                    for metric in [
                        "intra_dispersion",
                        "nearest_distance",
                        "nearest_class_id",
                        "nearest_class_name",
                        "normalized_margin",
                        "knn_purity",
                    ]:
                        row[f"{model_key}_{layer}_{metric}"] = geometry[metric]
                    for metric, value in ratios.items():
                        row[f"{model_key}_{layer}_{metric}"] = value

                    current_centroid = step_centroids[model_key][layer][class_id]
                    if class_id not in first_centroids[model_key][layer]:
                        first_centroids[model_key][layer][class_id] = current_centroid.copy()
                        first_centroid_reference_step[model_key][layer][class_id] = step
                        drift_first = 0.0
                    else:
                        drift_first = cosine_distance_vec(
                            current_centroid,
                            first_centroids[model_key][layer][class_id],
                            args.eps,
                        )

                    if class_id not in previous_centroids[model_key][layer]:
                        drift_previous: Any = ""
                        previous_analyzed_step: Any = ""
                    else:
                        drift_previous = cosine_distance_vec(
                            current_centroid,
                            previous_centroids[model_key][layer][class_id],
                            args.eps,
                        )
                        previous_analyzed_step = previous_step_for_class[model_key][layer][
                            class_id
                        ]

                    row[
                        f"{model_key}_{layer}_centroid_drift_from_previous_analyzed_step"
                    ] = drift_previous
                    row[
                        f"{model_key}_{layer}_previous_analyzed_step"
                    ] = previous_analyzed_step
                    row[
                        f"{model_key}_{layer}_centroid_drift_from_first_reference_step"
                    ] = drift_first
                    row[
                        f"{model_key}_{layer}_first_centroid_reference_step"
                    ] = first_centroid_reference_step[model_key][layer][class_id]

                    previous_centroids[model_key][layer][class_id] = current_centroid.copy()
                    previous_step_for_class[model_key][layer][class_id] = step

            for layer in ["z1_audio", "z1_visual", "z2_fusion"]:
                for metric in [
                    "intra_dispersion",
                    "nearest_distance",
                    "normalized_margin",
                    "knn_purity",
                    "later_neighbor_ratio_topk",
                    "cross_task_neighbor_ratio_topk",
                    "centroid_drift_from_previous_analyzed_step",
                    "centroid_drift_from_first_reference_step",
                ]:
                    value_a = row.get(f"model_a_{layer}_{metric}", "")
                    value_b = row.get(f"model_b_{layer}_{metric}", "")
                    row[f"delta_b_minus_a_{layer}_{metric}"] = (
                        ""
                        if value_a == "" or value_b == ""
                        else float(value_b - value_a)
                    )
            class_geometry_rows.append(row)

        if step in cache_steps:
            cache_dir = os.path.join(args.out_root, "feature_cache", f"step_{step}")
            ensure_dir(cache_dir)
            np.save(os.path.join(cache_dir, "labels.npy"), labels)
            for model_key, output in [
                ("model_a", output_a),
                ("model_b", output_b),
            ]:
                np.save(os.path.join(cache_dir, f"{model_key}_preds.npy"), output["preds"])
                for layer in ["z1_audio", "z1_visual", "z2_fusion"]:
                    np.save(
                        os.path.join(cache_dir, f"{model_key}_{layer}.npy"),
                        output[layer],
                    )
            with open(
                os.path.join(cache_dir, "sample_ids.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                for sample_id in sample_ids:
                    f.write(str(sample_id) + "\n")

        del output_a, output_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    geometry_trajectory_rows = aggregate_class_geometry_rows(
        class_geometry_rows, total_steps
    )
    geometry_correlation_rows = build_geometry_correlation_rows(
        class_geometry_rows
    )

    write_csv(
        os.path.join(core_dir, "03_geometry_trajectory_by_group.csv"),
        geometry_trajectory_rows,
    )
    write_csv(
        os.path.join(core_dir, "04_confusion_direction_by_group.csv"),
        confusion_direction_rows,
    )
    write_csv(
        os.path.join(core_dir, "05_geometry_performance_correlation.csv"),
        geometry_correlation_rows,
    )
    write_csv(
        os.path.join(detail_dir, "class_geometry_by_step.csv"),
        class_geometry_rows,
    )
    write_csv(
        os.path.join(detail_dir, "centroid_neighbors_by_step.csv"),
        centroid_neighbor_rows,
    )
    write_csv(
        os.path.join(detail_dir, "checkpoint_csv_consistency.csv"),
        checkpoint_consistency_rows,
    )

    summary = {
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "model_a_ckpt_dir": args.model_a_ckpt_dir,
        "model_b_ckpt_dir": args.model_b_ckpt_dir,
        "model_a_metrics_csv": metrics_csv_a,
        "model_b_metrics_csv": metrics_csv_b,
        "available_common_steps": common_steps,
        "analyzed_geometry_steps": analysis_steps,
        "cached_feature_steps": sorted(cache_steps),
        "core_outputs": {
            "01_u_shape_diagnosis.csv": (
                "overall/old/new accuracy and exact decomposition of each step change"
            ),
            "02_taskwise_accuracy_delta.csv": (
                "Model B minus Model A task-wise accuracy matrix"
            ),
            "03_geometry_trajectory_by_group.csv": (
                "z1/z2 geometry, neighbor structure, and centroid drift across steps"
            ),
            "04_confusion_direction_by_group.csv": (
                "correct and error flow to earlier/same/later arrival tasks"
            ),
            "05_geometry_performance_correlation.csv": (
                "per-step association between class accuracy gains and geometry gains"
            ),
        },
        "important_note": (
            "Drift references depend on --steps. With the default all-step run, "
            "previous_analyzed_step is the true previous checkpoint and "
            "first_reference_step equals each class's first_seen_step."
        ),
    }
    save_json(summary, os.path.join(args.out_root, "summary.json"))

    if args.dataset != "AVE":
        test_set.close_visual_features_h5()

    print("\n[DONE]")
    print(f"[INFO] Outputs saved to: {args.out_root}")
    print(f"[INFO] Start with: {core_dir}")


if __name__ == "__main__":
    main()