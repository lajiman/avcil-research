#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build embedding-space diagnostic tables for VGGSound / AV-CIL baselines.

Outputs:
  1) class_summary.csv
  2) centroid_neighbors.csv
  3) confusion_neighbors.csv
  4) neighbor_confusion_alignment.csv

Representation names:
  z0_audio
  z0_visual
  full_z1_audio
  full_z1_visual
  full_z2_fusion
  cl_z1_audio
  cl_z1_visual
  cl_z2_fusion

Assumption for your current IncreAudioVisualNet:
  model(..., out_features=True, out_feature_before_fusion=True) returns:
      logits, audio_visual_features, norm_audio_feature, norm_visual_feature
  where audio_visual_features = visual_feature + audio_feature, i.e. z2_fusion.
"""

import os
import sys
import csv
import json
import argparse
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make this script runnable from experiments/ or a subfolder.
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from dataloader_ours import IcaAVELoader
# Keep this import so torch.load can resolve whole-model checkpoints saved with this class.
from model.audio_visual_model_incremental import IncreAudioVisualNet  # noqa: F401


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================
# Basic utilities
# =========================================================
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None):
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def dataset_type(s: str):
    if s in ["AVE", "ksounds"]:
        return s
    if "VGGSound" in s:
        return s
    raise argparse.ArgumentTypeError("dataset must be 'AVE', 'ksounds', or contain 'VGGSound'")


def set_dataset_to_all_classes(dataset, num_classes: int):
    """Force test dataset to include all classes instead of only current incremental step."""
    dataset.current_step_class = np.arange(num_classes)
    dataset.all_current_data_vids = []
    for class_idx in dataset.current_step_class:
        dataset.all_current_data_vids += dataset.all_classId_vid_dict[str(int(class_idx))]


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


# =========================================================
# Feature flattening for z0
# =========================================================
def flatten_visual(raw_visual: torch.Tensor) -> torch.Tensor:
    """
    Convert raw visual pretrained feature to [B, D].
    Your VGGSound visual feature is usually [B, T, S, 768].
    """
    raw_visual = raw_visual.detach().float().cpu()
    if raw_visual.ndim == 4:
        return raw_visual.mean(dim=(1, 2))
    if raw_visual.ndim == 3:
        return raw_visual.mean(dim=1)
    if raw_visual.ndim == 2:
        return raw_visual
    return raw_visual.view(raw_visual.shape[0], -1)


def flatten_audio(raw_audio: torch.Tensor) -> torch.Tensor:
    """Convert raw audio pretrained feature to [B, D]."""
    raw_audio = raw_audio.detach().float().cpu()
    if raw_audio.ndim > 2:
        reduce_dims = tuple(range(1, raw_audio.ndim - 1))
        return raw_audio.mean(dim=reduce_dims) if len(reduce_dims) > 0 else raw_audio
    return raw_audio


# =========================================================
# Model / inference
# =========================================================
def load_model(ckpt_path: str):
    """Load a whole-model checkpoint saved by torch.save(model, path)."""
    try:
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(ckpt_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference_extract_reps(
    model,
    dataset,
    batch_size: int,
    num_workers: int,
    prefix: str,
    extract_z0: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Returns labels, preds, logits, and representation arrays.

    For prefix='full':
      full_z1_audio, full_z1_visual, full_z2_fusion
    For prefix='cl':
      cl_z1_audio, cl_z1_visual, cl_z2_fusion

    If extract_z0=True, also returns z0_audio and z0_visual.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    labels_all, preds_all, logits_all = [], [], []
    z1_audio_all, z1_visual_all, z2_fusion_all = [], [], []
    z0_audio_all, z0_visual_all = [], []

    model.eval()
    for data, labels in tqdm(loader, desc=f"[Inference:{prefix}]", ncols=100):
        labels = labels.long()
        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)

        if extract_z0:
            z0_visual_all.append(flatten_visual(data[0]))
            z0_audio_all.append(flatten_audio(data[1]))

        # Your model's return order for audio-visual mode:
        # logits, audio_visual_features, F.normalize(audio_feature), F.normalize(visual_feature)
        out = model(
            visual=visual,
            audio=audio,
            out_logits=True,
            out_features=True,
            out_features_norm=False,
            out_feature_before_fusion=True,
        )
        if not (isinstance(out, (tuple, list)) and len(out) >= 4):
            raise RuntimeError(
                "Expected model(..., out_features=True, out_feature_before_fusion=True) "
                "to return at least (logits, fusion_feature, audio_feature, visual_feature)."
            )

        logits = out[0]
        fusion_feature = out[1]
        audio_feature = out[2]
        visual_feature = out[3]

        pred = logits.argmax(dim=1).detach().cpu().long()

        labels_all.append(labels.cpu())
        preds_all.append(pred)
        logits_all.append(logits.detach().float().cpu())
        z2_fusion_all.append(fusion_feature.detach().float().cpu())
        z1_audio_all.append(audio_feature.detach().float().cpu())
        z1_visual_all.append(visual_feature.detach().float().cpu())

    result = {
        "labels": torch.cat(labels_all, dim=0).numpy(),
        "preds": torch.cat(preds_all, dim=0).numpy(),
        "logits": torch.cat(logits_all, dim=0).numpy(),
        f"{prefix}_z1_audio": torch.cat(z1_audio_all, dim=0).numpy(),
        f"{prefix}_z1_visual": torch.cat(z1_visual_all, dim=0).numpy(),
        f"{prefix}_z2_fusion": torch.cat(z2_fusion_all, dim=0).numpy(),
    }

    if extract_z0:
        result["z0_audio"] = torch.cat(z0_audio_all, dim=0).numpy()
        result["z0_visual"] = torch.cat(z0_visual_all, dim=0).numpy()

    return result


# =========================================================
# Performance / confusion
# =========================================================
def compute_per_class_prf_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, np.ndarray]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        tp[c] = np.logical_and(true_c, pred_c).sum()
        fp[c] = np.logical_and(~true_c, pred_c).sum()
        fn[c] = np.logical_and(true_c, ~pred_c).sum()

    support = tp + fn
    precision = np.zeros(num_classes, dtype=np.float32)
    recall = np.zeros(num_classes, dtype=np.float32)
    f1 = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        p = safe_div(tp[c], tp[c] + fp[c])
        r = safe_div(tp[c], tp[c] + fn[c])
        precision[c] = p
        recall[c] = r
        f1[c] = safe_div(2 * p * r, p + r)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_confusion_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def build_confusion_neighbors(
    cm: np.ndarray,
    class_names: List[str],
    setting: str,
    topk: int,
) -> List[Dict[str, Any]]:
    rows = []
    num_classes = cm.shape[0]

    for i in range(num_classes):
        row = cm[i].astype(np.float64).copy()
        support = float(row.sum())
        row[i] = 0.0
        order = np.argsort(row)[::-1]
        order = [j for j in order if row[j] > 0][:topk]

        # Keep at least one empty row when there is no confusion, useful for complete joins.
        if len(order) == 0:
            rows.append({
                "source_class_id": i,
                "source_class_name": class_names[i],
                "evaluation_setting": setting,
                "confusion_rank": "",
                "predicted_class_id": "",
                "predicted_class_name": "",
                "confusion_count": 0,
                "confusion_rate": 0.0,
            })
            continue

        for rank, j in enumerate(order, start=1):
            rows.append({
                "source_class_id": i,
                "source_class_name": class_names[i],
                "evaluation_setting": setting,
                "confusion_rank": rank,
                "predicted_class_id": int(j),
                "predicted_class_name": class_names[j],
                "confusion_count": int(row[j]),
                "confusion_rate": float(row[j] / max(support, 1.0)),
            })

    return rows


# =========================================================
# Geometry metrics
# =========================================================
def compute_centroids(
    X_norm: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    D = X_norm.shape[1]
    centroids = np.zeros((num_classes, D), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        counts[c] = len(idx)
        if len(idx) > 0:
            centroids[c] = X_norm[idx].mean(axis=0)

    centroids = l2_normalize_np(centroids, eps=eps)
    return centroids, counts


def compute_knn_purity(
    X_norm: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    k: int,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Chunked cosine kNN purity. X_norm must be L2-normalized.
    Excludes the sample itself.
    """
    N = X_norm.shape[0]
    if N <= 1:
        return np.zeros(num_classes, dtype=np.float32)

    k_eff = min(k, N - 1)
    labels = labels.astype(np.int64)
    sample_purity = np.zeros(N, dtype=np.float32)

    for start in tqdm(range(0, N, chunk_size), desc=f"[kNN purity k={k_eff}]", ncols=100):
        end = min(start + chunk_size, N)
        sims = X_norm[start:end] @ X_norm.T  # [B, N]
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        sims[rows, cols] = -np.inf

        # Unsorted top-k indices, then purity does not need sorting.
        nn_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        same = labels[nn_idx] == labels[start:end, None]
        sample_purity[start:end] = same.mean(axis=1)

    class_purity = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        class_purity[c] = float(sample_purity[idx].mean()) if len(idx) > 0 else 0.0
    return class_purity


def compute_geometry_for_representation(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    representation: str,
    topk_centroid: int,
    knn_k: int,
    eps: float,
    knn_chunk_size: int,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      summary_by_class[c] = class-level geometry metrics for this representation
      centroid_neighbor_rows = long-format top-k nearest centroid table rows
    """
    num_classes = len(class_names)
    X_norm = l2_normalize_np(features, eps=eps)
    labels = labels.astype(np.int64)

    centroids, counts = compute_centroids(X_norm, labels, num_classes, eps=eps)

    # Intra-class cosine dispersion: mean 1 - cos(x_i, centroid_c)
    intra = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            intra[c] = 0.0
        else:
            sims = X_norm[idx] @ centroids[c]
            intra[c] = float((1.0 - sims).mean())

    # Centroid cosine distance matrix.
    centroid_sims = centroids @ centroids.T
    centroid_dist = 1.0 - centroid_sims
    np.fill_diagonal(centroid_dist, np.inf)

    # kNN local purity.
    knn_purity = compute_knn_purity(
        X_norm=X_norm,
        labels=labels,
        num_classes=num_classes,
        k=knn_k,
        chunk_size=knn_chunk_size,
    )

    summary_by_class: Dict[int, Dict[str, Any]] = {}
    neighbor_rows: List[Dict[str, Any]] = []

    for c in range(num_classes):
        if counts[c] == 0:
            order = []
        else:
            order = np.argsort(centroid_dist[c])[:topk_centroid].tolist()
            order = [j for j in order if np.isfinite(centroid_dist[c, j])]

        nearest_id = int(order[0]) if order else -1
        nearest_dist = float(centroid_dist[c, nearest_id]) if order else 0.0
        margin = float(nearest_dist / (float(intra[c]) + eps)) if order else 0.0

        summary_by_class[c] = {
            f"{representation}_intra_dispersion": float(intra[c]),
            f"{representation}_nearest_distance": nearest_dist,
            f"{representation}_nearest_class_id": nearest_id if nearest_id >= 0 else "",
            f"{representation}_nearest_class_name": class_names[nearest_id] if nearest_id >= 0 else "",
            f"{representation}_normalized_margin": margin,
            f"{representation}_knn_purity_k{knn_k}": float(knn_purity[c]),
        }

        for rank, j in enumerate(order, start=1):
            neighbor_rows.append({
                "source_class_id": c,
                "source_class_name": class_names[c],
                "representation": representation,
                "neighbor_rank": rank,
                "neighbor_class_id": int(j),
                "neighbor_class_name": class_names[j],
                "centroid_distance": float(centroid_dist[c, j]),
            })

    return summary_by_class, neighbor_rows


# =========================================================
# Alignment table
# =========================================================
def build_neighbor_maps(centroid_rows: List[Dict[str, Any]], topk: int) -> Dict[Tuple[str, int], List[int]]:
    d: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    for r in centroid_rows:
        rep = str(r["representation"])
        c = int(r["source_class_id"])
        rank = int(r["neighbor_rank"])
        nid = int(r["neighbor_class_id"])
        d.setdefault((rep, c), []).append((rank, nid))
    return {k: [nid for rank, nid in sorted(v)[:topk]] for k, v in d.items()}


def build_confusion_maps(confusion_rows: List[Dict[str, Any]], topk: int) -> Dict[Tuple[str, int], List[int]]:
    d: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    for r in confusion_rows:
        if r.get("confusion_rank", "") == "" or r.get("predicted_class_id", "") == "":
            continue
        setting = str(r["evaluation_setting"])
        c = int(r["source_class_id"])
        rank = int(r["confusion_rank"])
        pid = int(r["predicted_class_id"])
        d.setdefault((setting, c), []).append((rank, pid))
    return {k: [pid for rank, pid in sorted(v)[:topk]] for k, v in d.items()}


def build_alignment_rows(
    centroid_rows: List[Dict[str, Any]],
    confusion_rows: List[Dict[str, Any]],
    cms: Dict[str, np.ndarray],
    class_names: List[str],
    representations: List[str],
    settings: List[str],
    topk: int,
) -> List[Dict[str, Any]]:
    num_classes = len(class_names)
    neighbor_map = build_neighbor_maps(centroid_rows, topk=topk)
    confusion_map = build_confusion_maps(confusion_rows, topk=topk)

    rows = []
    for rep in representations:
        for setting in settings:
            cm = cms[setting]
            support = cm.sum(axis=1)
            for c in range(num_classes):
                nearest = neighbor_map.get((rep, c), [])
                confused = confusion_map.get((setting, c), [])

                def overlap_at(k: int) -> float:
                    if k <= 0:
                        return 0.0
                    return float(len(set(nearest[:k]).intersection(set(confused[:k]))) / k)

                nearest1 = nearest[0] if len(nearest) > 0 else -1
                confusion1 = confused[0] if len(confused) > 0 else -1
                nearest1_conf_rate = 0.0
                if nearest1 >= 0:
                    nearest1_conf_rate = float(cm[c, nearest1] / max(int(support[c]), 1))

                rows.append({
                    "class_id": c,
                    "class_name": class_names[c],
                    "representation": rep,
                    "evaluation_setting": setting,
                    "overlap_at1": overlap_at(1),
                    "overlap_at3": overlap_at(min(3, topk)) if topk >= 3 else overlap_at(topk),
                    "overlap_at5": overlap_at(min(5, topk)) if topk >= 5 else overlap_at(topk),
                    "nearest1_class_id": nearest1 if nearest1 >= 0 else "",
                    "nearest1_class_name": class_names[nearest1] if nearest1 >= 0 else "",
                    "confusion1_class_id": confusion1 if confusion1 >= 0 else "",
                    "confusion1_class_name": class_names[confusion1] if confusion1 >= 0 else "",
                    "nearest1_is_confusion1": int(nearest1 >= 0 and confusion1 >= 0 and nearest1 == confusion1),
                    "nearest1_confusion_rate": nearest1_conf_rate,
                })
    return rows


# =========================================================
# Main table assembly
# =========================================================
def build_class_summary(
    class_names: List[str],
    full_stats: Dict[str, np.ndarray],
    cl_stats: Dict[str, np.ndarray],
    geometry_summary: Dict[str, Dict[int, Dict[str, Any]]],
    knn_k: int,
) -> List[Dict[str, Any]]:
    num_classes = len(class_names)
    rows: List[Dict[str, Any]] = []

    for c in range(num_classes):
        row: Dict[str, Any] = {
            "class_id": c,
            "class_name": class_names[c],
            "support": int(full_stats["support"][c]),
            "full_precision": float(full_stats["precision"][c]),
            "full_recall": float(full_stats["recall"][c]),
            "full_f1": float(full_stats["f1"][c]),
            "full_tp": int(full_stats["tp"][c]),
            "full_fp": int(full_stats["fp"][c]),
            "full_fn": int(full_stats["fn"][c]),
            "cl_final_precision": float(cl_stats["precision"][c]),
            "cl_final_recall": float(cl_stats["recall"][c]),
            "cl_final_f1": float(cl_stats["f1"][c]),
            "cl_final_tp": int(cl_stats["tp"][c]),
            "cl_final_fp": int(cl_stats["fp"][c]),
            "cl_final_fn": int(cl_stats["fn"][c]),
        }
        row["precision_gap"] = row["full_precision"] - row["cl_final_precision"]
        row["recall_gap"] = row["full_recall"] - row["cl_final_recall"]
        row["f1_gap"] = row["full_f1"] - row["cl_final_f1"]

        for rep, summary in geometry_summary.items():
            row.update(summary[c])

        # Most important delta: CL fusion geometry vs full-class fusion geometry.
        prefix_a = "cl_z2_fusion"
        prefix_b = "full_z2_fusion"
        if (
            f"{prefix_a}_normalized_margin" in row
            and f"{prefix_b}_normalized_margin" in row
        ):
            row["delta_cl_z2_vs_full_z2_intra_dispersion"] = (
                row[f"{prefix_a}_intra_dispersion"] - row[f"{prefix_b}_intra_dispersion"]
            )
            row["delta_cl_z2_vs_full_z2_nearest_distance"] = (
                row[f"{prefix_a}_nearest_distance"] - row[f"{prefix_b}_nearest_distance"]
            )
            row["delta_cl_z2_vs_full_z2_normalized_margin"] = (
                row[f"{prefix_a}_normalized_margin"] - row[f"{prefix_b}_normalized_margin"]
            )
            row[f"delta_cl_z2_vs_full_z2_knn_purity_k{knn_k}"] = (
                row[f"{prefix_a}_knn_purity_k{knn_k}"] - row[f"{prefix_b}_knn_purity_k{knn_k}"]
            )

        # Optional rough deltas: full fusion vs initial single-modality geometry.
        for z0_rep in ["z0_audio", "z0_visual"]:
            if (
                f"full_z2_fusion_normalized_margin" in row
                and f"{z0_rep}_normalized_margin" in row
            ):
                row[f"delta_full_z2_vs_{z0_rep}_normalized_margin"] = (
                    row["full_z2_fusion_normalized_margin"] - row[f"{z0_rep}_normalized_margin"]
                )
                row[f"delta_full_z2_vs_{z0_rep}_knn_purity_k{knn_k}"] = (
                    row[f"full_z2_fusion_knn_purity_k{knn_k}"] - row[f"{z0_rep}_knn_purity_k{knn_k}"]
                )

        rows.append(row)

    return rows


def save_feature_cache(out_root: str, arrays: Dict[str, np.ndarray], sample_ids: List[str]):
    feature_dir = os.path.join(out_root, "features")
    array_dir = os.path.join(out_root, "arrays")
    ensure_dir(feature_dir)
    ensure_dir(array_dir)

    feature_keys = [
        "z0_audio", "z0_visual",
        "full_z1_audio", "full_z1_visual", "full_z2_fusion",
        "cl_z1_audio", "cl_z1_visual", "cl_z2_fusion",
    ]
    array_keys = ["labels", "full_preds", "cl_preds", "full_logits", "cl_logits"]

    for k in feature_keys:
        if k in arrays:
            np.save(os.path.join(feature_dir, f"{k}.npy"), arrays[k])
    for k in array_keys:
        if k in arrays:
            np.save(os.path.join(array_dir, f"{k}.npy"), arrays[k])

    with open(os.path.join(array_dir, "sample_ids.txt"), "w", encoding="utf-8") as f:
        for sid in sample_ids:
            f.write(str(sid) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=dataset_type, required=True)
    parser.add_argument("--modality", type=str, default="audio-visual", choices=["audio-visual"])
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--meta_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--class_num_per_step", type=int, default=10)

    parser.add_argument("--full_ckpt_path", type=str, required=True)
    parser.add_argument("--cl_ckpt_path", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)

    parser.add_argument("--infer_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--topk_centroid", type=int, default=5)
    parser.add_argument("--topk_confusion", type=int, default=5)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--no_cache_features", action="store_true")

    args = parser.parse_args()
    setup_seed(args.seed)
    ensure_dir(args.out_root)

    print(f"[INFO] device = {device}")
    print("[INFO] Building test dataset...")
    test_set = IcaAVELoader(args=args, mode="test", modality=args.modality)
    set_dataset_to_all_classes(test_set, args.num_classes)
    print(f"[INFO] test samples = {len(test_set)} | num_classes = {args.num_classes}")

    category_encode_dict = test_set.category_encode_dict
    id_to_category = {int(v): k for k, v in category_encode_dict.items()}
    class_names = [id_to_category.get(i, f"class_{i}") for i in range(args.num_classes)]
    sample_ids = list(test_set.all_current_data_vids)

    print(f"[INFO] Loading full-class checkpoint: {args.full_ckpt_path}")
    full_model = load_model(args.full_ckpt_path)
    full_out = run_inference_extract_reps(
        model=full_model,
        dataset=test_set,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        prefix="full",
        extract_z0=True,
    )

    print(f"[INFO] Loading CL-final checkpoint: {args.cl_ckpt_path}")
    cl_model = load_model(args.cl_ckpt_path)
    cl_out = run_inference_extract_reps(
        model=cl_model,
        dataset=test_set,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        prefix="cl",
        extract_z0=False,
    )

    labels = full_out["labels"]
    if not np.array_equal(labels, cl_out["labels"]):
        raise RuntimeError("Full and CL inference labels are not identical. Check test set ordering.")

    full_preds = full_out["preds"]
    cl_preds = cl_out["preds"]

    full_stats = compute_per_class_prf_np(labels, full_preds, args.num_classes)
    cl_stats = compute_per_class_prf_np(labels, cl_preds, args.num_classes)

    full_cm = compute_confusion_np(labels, full_preds, args.num_classes)
    cl_cm = compute_confusion_np(labels, cl_preds, args.num_classes)

    # Merge all representation arrays.
    rep_features: Dict[str, np.ndarray] = {
        "z0_audio": full_out["z0_audio"],
        "z0_visual": full_out["z0_visual"],
        "full_z1_audio": full_out["full_z1_audio"],
        "full_z1_visual": full_out["full_z1_visual"],
        "full_z2_fusion": full_out["full_z2_fusion"],
        "cl_z1_audio": cl_out["cl_z1_audio"],
        "cl_z1_visual": cl_out["cl_z1_visual"],
        "cl_z2_fusion": cl_out["cl_z2_fusion"],
    }
    representations = list(rep_features.keys())

    print("[INFO] Computing geometry metrics...")
    geometry_summary: Dict[str, Dict[int, Dict[str, Any]]] = {}
    centroid_rows: List[Dict[str, Any]] = []
    for rep in representations:
        print(f"[INFO] Representation: {rep} | shape={rep_features[rep].shape}")
        summary, rows = compute_geometry_for_representation(
            features=rep_features[rep],
            labels=labels,
            class_names=class_names,
            representation=rep,
            topk_centroid=args.topk_centroid,
            knn_k=args.knn_k,
            eps=args.eps,
            knn_chunk_size=args.knn_chunk_size,
        )
        geometry_summary[rep] = summary
        centroid_rows.extend(rows)

    print("[INFO] Building summary / neighbor / alignment tables...")
    class_summary_rows = build_class_summary(
        class_names=class_names,
        full_stats=full_stats,
        cl_stats=cl_stats,
        geometry_summary=geometry_summary,
        knn_k=args.knn_k,
    )

    confusion_rows = []
    confusion_rows.extend(build_confusion_neighbors(full_cm, class_names, "full", args.topk_confusion))
    confusion_rows.extend(build_confusion_neighbors(cl_cm, class_names, "cl_final", args.topk_confusion))

    alignment_rows = build_alignment_rows(
        centroid_rows=centroid_rows,
        confusion_rows=confusion_rows,
        cms={"full": full_cm, "cl_final": cl_cm},
        class_names=class_names,
        representations=representations,
        settings=["full", "cl_final"],
        topk=min(args.topk_centroid, args.topk_confusion),
    )

    # Save CSVs.
    write_csv(os.path.join(args.out_root, "class_summary.csv"), class_summary_rows)
    write_csv(os.path.join(args.out_root, "centroid_neighbors.csv"), centroid_rows)
    write_csv(os.path.join(args.out_root, "confusion_neighbors.csv"), confusion_rows)
    write_csv(os.path.join(args.out_root, "neighbor_confusion_alignment.csv"), alignment_rows)

    # Save arrays and metadata.
    arrays = {
        **rep_features,
        "labels": labels,
        "full_preds": full_preds,
        "cl_preds": cl_preds,
        "full_logits": full_out["logits"],
        "cl_logits": cl_out["logits"],
    }
    np.save(os.path.join(args.out_root, "full_confusion_matrix.npy"), full_cm)
    np.save(os.path.join(args.out_root, "cl_confusion_matrix.npy"), cl_cm)

    if not args.no_cache_features:
        save_feature_cache(args.out_root, arrays, sample_ids)

    summary = {
        "num_samples": int(len(labels)),
        "num_classes": int(args.num_classes),
        "representations": representations,
        "topk_centroid": int(args.topk_centroid),
        "topk_confusion": int(args.topk_confusion),
        "knn_k": int(args.knn_k),
        "full_overall_acc": float((labels == full_preds).mean()),
        "cl_final_overall_acc": float((labels == cl_preds).mean()),
        "outputs": [
            "class_summary.csv",
            "centroid_neighbors.csv",
            "confusion_neighbors.csv",
            "neighbor_confusion_alignment.csv",
        ],
    }
    save_json(summary, os.path.join(args.out_root, "summary.json"))

    if args.dataset != "AVE":
        test_set.close_visual_features_h5()

    print("[DONE]")
    print(f"[INFO] Outputs saved to: {args.out_root}")
    print(f"[INFO] full_overall_acc={summary['full_overall_acc']:.6f}")
    print(f"[INFO] cl_final_overall_acc={summary['cl_final_overall_acc']:.6f}")


if __name__ == "__main__":
    main()