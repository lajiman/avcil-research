#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small z0 experiment: identify class-conditional modality noise in VGGSound.

Claim supported by this script
------------------------------
This script does NOT prove that the raw waveform or raw video itself is corrupted.
It tests a narrower, model-independent claim:

    Some classes have a locally label-mixed / weakly structured z0 representation
    in one modality, while the other modality is substantially more structured.

z0 definitions
--------------
audio:
    the fixed input audio feature returned by IcaAVELoader.

visual:
    the fixed, audio-independent uniform average of the visual input tokens:
        visual.view(B, 8, -1, 768).mean(dim=(1, 2))

No checkpoint and no learned projection are used.

Main diagnostics
----------------
1. kNN purity:
       P_{m,c} = average same-class fraction among each sample's k nearest
                 neighbours in modality m.

2. Chance-corrected kNN purity:
       P*_{m,c} = (P_{m,c} - pi_c) / (1 - pi_c),
       pi_c = (N_c - 1) / (N - 1).

3. Normalized neighbour-label entropy:
       H_{m,c} in [0, 1].
   High H means neighbours are spread across many labels rather than being
   concentrated in one coherent competing class.

4. Normalized margin:
       M_{m,c} = nearest-centroid distance / intra-class dispersion.

Interpretation
--------------
- low margin + high purity: difficult but locally structured;
- low margin + low purity / high entropy: noisy or locally label-mixed;
- scatter-plot color directly encodes normalized neighbour-label entropy.
- a paired bootstrap CI for visual-minus-audio purity (and the reverse entropy
  gap) identifies statistically stable modality asymmetry within each class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent

# Allow the script to live either in the project root or in an experiment
# subdirectory one/two levels below it.
PROJECT_ROOT = None
for candidate in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (candidate / "dataloader_ours.py").is_file():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataloader_ours.py. Place this script in the "
        "AV-CIL project root or one of its experiment subdirectories."
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader_ours import IcaAVELoader  # noqa: E402


EPS = 1e-12


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def dataset_type(value: str) -> str:
    if value in {"AVE", "ksounds"} or "VGGSound" in value:
        return value
    raise argparse.ArgumentTypeError(
        "dataset must be AVE, ksounds, or contain VGGSound"
    )


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def average_rank_percentile(values: Sequence[float]) -> np.ndarray:
    """Average-tie percentile ranks in [0, 1]."""
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
        while (
            end < n
            and np.isclose(
                sorted_x[end], sorted_x[start], rtol=0.0, atol=1e-12
            )
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / float(n - 1)


def set_dataset_to_all_classes(dataset: Any, num_classes: int) -> None:
    """
    IcaAVELoader normally exposes one incremental block at a time.
    This diagnostic must use all 100 classes at once.
    """
    dataset.current_step_class = np.arange(num_classes)
    dataset.all_current_data_vids = []
    for class_id in dataset.current_step_class:
        key = str(int(class_id))
        if key not in dataset.all_classId_vid_dict:
            raise KeyError(f"Missing class ID {key} in all_classId_vid_dict.")
        dataset.all_current_data_vids += dataset.all_classId_vid_dict[key]


@torch.no_grad()
def extract_z0(
    dataset: Any,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    audio_all: List[torch.Tensor] = []
    visual_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    for data, labels in tqdm(loader, desc="Extract fixed z0", ncols=100):
        visual = data[0].float()
        audio = data[1].float()

        if audio.ndim != 2:
            raise ValueError(
                f"Expected audio shape (B, D), got {tuple(audio.shape)}"
            )
        if visual.shape[-1] != 768:
            raise ValueError(
                "Expected the final visual dimension to be 768, "
                f"got {tuple(visual.shape)}"
            )

        # Exact audio-independent z0 visual definition used by the model.
        visual_4d = visual.reshape(visual.shape[0], 8, -1, 768)
        visual_uniform = visual_4d.mean(dim=(1, 2))

        audio_all.append(audio.cpu())
        visual_all.append(visual_uniform.cpu())
        labels_all.append(labels.long().cpu())

    audio_np = torch.cat(audio_all, dim=0).numpy().astype(np.float32)
    visual_np = torch.cat(visual_all, dim=0).numpy().astype(np.float32)
    labels_np = torch.cat(labels_all, dim=0).numpy().astype(np.int64)

    if not (len(audio_np) == len(visual_np) == len(labels_np)):
        raise RuntimeError("z0 extraction produced inconsistent sample counts.")
    if not np.isfinite(audio_np).all() or not np.isfinite(visual_np).all():
        raise RuntimeError("Non-finite z0 features detected.")

    return audio_np, visual_np, labels_np


def optional_class_subsample(
    audio: np.ndarray,
    visual: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    max_samples_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples_per_class <= 0:
        return audio, visual, labels

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for class_id in range(num_classes):
        idx = np.where(labels == class_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Class {class_id} has no samples.")
        if len(idx) > max_samples_per_class:
            idx = rng.choice(
                idx, size=max_samples_per_class, replace=False
            )
        selected.extend(int(i) for i in idx)
    selected_np = np.asarray(sorted(selected), dtype=np.int64)
    return audio[selected_np], visual[selected_np], labels[selected_np]


def compute_sample_knn_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    k: int,
    chunk_size: int,
    description: str,
) -> Dict[str, np.ndarray]:
    x = l2_normalize(features)
    n = len(labels)
    if n <= 1:
        raise ValueError("At least two samples are required.")
    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError("Effective k must be at least 1.")

    sample_purity = np.zeros(n, dtype=np.float32)
    sample_entropy = np.zeros(n, dtype=np.float32)
    neighbor_labels_all = np.empty((n, k_eff), dtype=np.int64)

    entropy_denom = math.log(min(k_eff, num_classes))
    for start in tqdm(
        range(0, n, chunk_size),
        desc=description,
        ncols=100,
        leave=False,
    ):
        end = min(start + chunk_size, n)
        similarity = x[start:end] @ x.T

        local_rows = np.arange(end - start)
        global_cols = np.arange(start, end)
        similarity[local_rows, global_cols] = -np.inf

        neighbors = np.argpartition(
            -similarity, kth=k_eff - 1, axis=1
        )[:, :k_eff]
        neighbor_labels = labels[neighbors]
        neighbor_labels_all[start:end] = neighbor_labels

        target = labels[start:end, None]
        sample_purity[start:end] = (
            neighbor_labels == target
        ).mean(axis=1)

        if entropy_denom <= 0:
            sample_entropy[start:end] = 0.0
        else:
            for local_idx, row_labels in enumerate(neighbor_labels):
                counts = np.bincount(
                    row_labels, minlength=num_classes
                ).astype(np.float64)
                probs = counts[counts > 0] / float(k_eff)
                entropy = -float(np.sum(probs * np.log(probs)))
                sample_entropy[start + local_idx] = (
                    entropy / entropy_denom
                )

    return {
        "x_norm": x,
        "sample_purity": sample_purity,
        "sample_entropy": sample_entropy,
        "neighbor_labels": neighbor_labels_all,
        "k_eff": np.asarray([k_eff], dtype=np.int64),
    }


def compute_class_geometry(
    x_norm: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, np.ndarray]:
    dim = x_norm.shape[1]
    centroids = np.zeros((num_classes, dim), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)
    intra = np.zeros(num_classes, dtype=np.float64)

    for class_id in range(num_classes):
        idx = np.where(labels == class_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Class {class_id} has no samples.")
        counts[class_id] = len(idx)
        centroid = x_norm[idx].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), EPS)
        centroids[class_id] = centroid
        intra[class_id] = float(
            (1.0 - x_norm[idx] @ centroid).mean()
        )

    centroid_distance = 1.0 - centroids @ centroids.T
    np.fill_diagonal(centroid_distance, np.inf)
    nearest_class = np.argmin(centroid_distance, axis=1).astype(np.int64)
    nearest_distance = centroid_distance[
        np.arange(num_classes), nearest_class
    ].astype(np.float64)
    normalized_margin = nearest_distance / np.maximum(intra, EPS)

    return {
        "counts": counts,
        "centroids": centroids,
        "intra_dispersion": intra,
        "nearest_class": nearest_class,
        "nearest_distance": nearest_distance,
        "normalized_margin": normalized_margin,
    }


def chance_corrected_sample_purity(
    sample_purity: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
) -> np.ndarray:
    n = len(labels)
    corrected = np.zeros(n, dtype=np.float64)
    for class_id, count in enumerate(class_counts):
        idx = np.where(labels == class_id)[0]
        chance = (float(count) - 1.0) / max(float(n) - 1.0, 1.0)
        corrected[idx] = (
            sample_purity[idx].astype(np.float64) - chance
        ) / max(1.0 - chance, EPS)
    return corrected


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    ci: float,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(values.mean())
    if repeats <= 0 or n == 1:
        return estimate, estimate, estimate
    draw = rng.integers(0, n, size=(repeats, n))
    means = values[draw].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def top_confusion(
    neighbor_labels: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    num_classes: int,
) -> Tuple[int, float]:
    idx = np.where(labels == class_id)[0]
    flattened = neighbor_labels[idx].reshape(-1)
    counts = np.bincount(flattened, minlength=num_classes).astype(np.int64)
    counts[class_id] = 0
    top_id = int(np.argmax(counts))
    total = max(int(len(flattened)), 1)
    return top_id, float(counts[top_id] / total)


def geometry_type(
    margin_percentile: float,
    purity_percentile: float,
) -> str:
    if margin_percentile <= 1.0 / 3.0 and purity_percentile <= 1.0 / 3.0:
        return "noisy_hard"
    if margin_percentile <= 1.0 / 3.0 and purity_percentile >= 2.0 / 3.0:
        return "structured_hard"
    if margin_percentile >= 2.0 / 3.0 and purity_percentile >= 2.0 / 3.0:
        return "easy_structured"
    return "mixed"


def plot_purity_asymmetry(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    annotate_top: int,
) -> None:
    x = np.asarray(
        [float(r["audio_chance_corrected_purity"]) for r in rows]
    )
    y = np.asarray(
        [float(r["visual_chance_corrected_purity"]) for r in rows]
    )
    purity_gap = y - x
    entropy_gap = np.asarray(
        [float(r["audio_minus_visual_entropy"]) for r in rows]
    )

    # A diverging entropy-gap color scale makes the direction explicit:
    # positive -> audio neighbours are more diffusely mixed;
    # negative -> visual neighbours are more diffusely mixed.
    max_abs_entropy_gap = max(float(np.max(np.abs(entropy_gap))), 1e-6)

    fig, ax = plt.subplots(figsize=(8, 8))
    points = ax.scatter(
        x,
        y,
        c=entropy_gap,
        cmap="coolwarm",
        vmin=-max_abs_entropy_gap,
        vmax=max_abs_entropy_gap,
        s=55,
        edgecolors="black",
        linewidths=0.35,
    )
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Audio z0 chance-corrected kNN purity")
    ax.set_ylabel("Visual z0 chance-corrected kNN purity")
    ax.set_title("z0 modality-local structure by class")
    ax.grid(True, alpha=0.3)

    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label(
        "Audio minus visual neighbour-label entropy\n"
        "(positive: audio confusion is more diffuse)"
    )

    top = np.argsort(-np.abs(purity_gap))[: min(annotate_top, len(rows))]
    for idx in top:
        ax.annotate(
            str(rows[int(idx)]["class_name"]),
            (x[idx], y[idx]),
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_margin_purity(
    rows: Sequence[Mapping[str, Any]],
    modality: str,
    out_path: Path,
    annotate_top: int,
) -> None:
    margin = np.asarray(
        [float(r[f"{modality}_margin_percentile"]) for r in rows]
    )
    purity = np.asarray(
        [float(r[f"{modality}_purity_percentile"]) for r in rows]
    )
    # Use the actual normalized entropy in [0, 1] as the color value rather
    # than a point-size proxy. This keeps the third variable directly readable.
    entropy = np.asarray(
        [float(r[f"{modality}_neighbor_entropy"]) for r in rows]
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    points = ax.scatter(
        margin,
        purity,
        c=entropy,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=60,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.axvline(1.0 / 3.0, linestyle="--")
    ax.axhline(1.0 / 3.0, linestyle="--")
    ax.set_xlabel("Normalized-margin percentile (higher = easier)")
    ax.set_ylabel("kNN-purity percentile (higher = more structured)")
    ax.set_title(f"{modality.capitalize()} z0: structured-hard vs noisy-hard")
    ax.grid(True, alpha=0.3)

    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label(
        "Normalized neighbour-label entropy\n"
        "(higher: confusion is spread across more classes)"
    )

    # Prioritize annotations for points with both low purity and high entropy.
    noise_order = np.argsort(purity - entropy)
    for idx in noise_order[: min(annotate_top, len(rows))]:
        ax.annotate(
            str(rows[int(idx)]["class_name"]),
            (margin[idx], purity[idx]),
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_asymmetry_rank(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    top_n: int,
) -> None:
    ordered = sorted(
        rows,
        key=lambda r: float(r["visual_minus_audio_corrected_purity"]),
    )
    chosen = ordered[:top_n] + ordered[-top_n:]
    labels = [str(r["class_name"]) for r in chosen]
    values = [
        float(r["visual_minus_audio_corrected_purity"]) for r in chosen
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(chosen))))
    positions = np.arange(len(chosen))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, linestyle="--")
    ax.set_xlabel(
        "Visual minus audio chance-corrected kNN purity\n"
        "(positive: audio is more locally mixed)"
    )
    ax.set_title("Largest z0 modality asymmetries")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_noise_score_rank(
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    top_n: int,
) -> None:
    ordered = sorted(
        rows,
        key=lambda r: float(r["audio_minus_visual_noise_score"]),
    )
    chosen = ordered[:top_n] + ordered[-top_n:]
    labels = [str(r["class_name"]) for r in chosen]
    values = [
        float(r["audio_minus_visual_noise_score"]) for r in chosen
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(chosen))))
    positions = np.arange(len(chosen))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, linestyle="--")
    ax.set_xlabel(
        "Audio minus visual z0 noise score\n"
        "(positive: audio is noisier; negative: visual is noisier)"
    )
    ax.set_title("Largest class-conditional z0 noise asymmetries")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to diagnose. Train is recommended for the main result.",
    )
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--class_num_per_step", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--bootstrap_ci", type=float, default=0.95)
    parser.add_argument(
        "--noise_quantile",
        type=float,
        default=1.0 / 3.0,
        help=(
            "A strong asymmetric-noise flag requires the noisy modality to "
            "be in the bottom q purity and top q entropy."
        ),
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="0 uses all train samples; positive values cap each class.",
    )
    parser.add_argument("--annotate_top", type=int, default=8)
    parser.add_argument("--rank_top_n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        type=str,
        default="./save/z0_modality_noise",
    )
    args = parser.parse_args()

    if args.knn_k < 1:
        raise ValueError("--knn_k must be >= 1.")
    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap_repeats must be >= 0.")
    if not 0.0 < args.bootstrap_ci < 1.0:
        raise ValueError("--bootstrap_ci must be in (0, 1).")
    if not 0.0 < args.noise_quantile < 0.5:
        raise ValueError("--noise_quantile must be in (0, 0.5).")

    setup_seed(args.seed)
    out_dir = Path(args.output_root) / args.dataset / args.split
    ensure_dir(out_dir)

    train_set = IcaAVELoader(
        args=args, mode=args.split, modality=args.modality
    )
    set_dataset_to_all_classes(train_set, args.num_classes)

    id_to_category = {
        int(class_id): str(name)
        for name, class_id in train_set.category_encode_dict.items()
    }
    class_names = [
        id_to_category.get(class_id, f"class_{class_id}")
        for class_id in range(args.num_classes)
    ]

    audio, visual, labels = extract_z0(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    audio, visual, labels = optional_class_subsample(
        audio,
        visual,
        labels,
        num_classes=args.num_classes,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )

    expected = set(range(args.num_classes))
    observed = set(int(x) for x in np.unique(labels))
    if observed != expected:
        raise ValueError(
            f"Observed class IDs do not match 0..{args.num_classes - 1}: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )

    audio_knn = compute_sample_knn_metrics(
        audio,
        labels,
        args.num_classes,
        args.knn_k,
        args.knn_chunk_size,
        "Audio z0 kNN",
    )
    visual_knn = compute_sample_knn_metrics(
        visual,
        labels,
        args.num_classes,
        args.knn_k,
        args.knn_chunk_size,
        "Visual z0 kNN",
    )

    audio_geo = compute_class_geometry(
        audio_knn["x_norm"], labels, args.num_classes
    )
    visual_geo = compute_class_geometry(
        visual_knn["x_norm"], labels, args.num_classes
    )

    audio_corrected = chance_corrected_sample_purity(
        audio_knn["sample_purity"],
        labels,
        audio_geo["counts"],
    )
    visual_corrected = chance_corrected_sample_purity(
        visual_knn["sample_purity"],
        labels,
        visual_geo["counts"],
    )

    # Class-level arrays used to build transparent percentile states.
    class_audio_purity = np.zeros(args.num_classes, dtype=np.float64)
    class_visual_purity = np.zeros(args.num_classes, dtype=np.float64)
    class_audio_entropy = np.zeros(args.num_classes, dtype=np.float64)
    class_visual_entropy = np.zeros(args.num_classes, dtype=np.float64)
    for class_id in range(args.num_classes):
        idx = np.where(labels == class_id)[0]
        class_audio_purity[class_id] = float(audio_corrected[idx].mean())
        class_visual_purity[class_id] = float(visual_corrected[idx].mean())
        class_audio_entropy[class_id] = float(
            audio_knn["sample_entropy"][idx].mean()
        )
        class_visual_entropy[class_id] = float(
            visual_knn["sample_entropy"][idx].mean()
        )

    audio_purity_pct = average_rank_percentile(class_audio_purity)
    visual_purity_pct = average_rank_percentile(class_visual_purity)
    audio_entropy_pct = average_rank_percentile(class_audio_entropy)
    visual_entropy_pct = average_rank_percentile(class_visual_entropy)
    audio_margin_pct = average_rank_percentile(
        audio_geo["normalized_margin"]
    )
    visual_margin_pct = average_rank_percentile(
        visual_geo["normalized_margin"]
    )

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    q = args.noise_quantile

    for class_id in range(args.num_classes):
        idx = np.where(labels == class_id)[0]
        n_c = len(idx)
        chance = (n_c - 1.0) / max(len(labels) - 1.0, 1.0)

        a_p, a_p_lo, a_p_hi = bootstrap_mean_ci(
            audio_corrected[idx],
            args.bootstrap_repeats,
            rng,
            args.bootstrap_ci,
        )
        v_p, v_p_lo, v_p_hi = bootstrap_mean_ci(
            visual_corrected[idx],
            args.bootstrap_repeats,
            rng,
            args.bootstrap_ci,
        )
        a_h, a_h_lo, a_h_hi = bootstrap_mean_ci(
            audio_knn["sample_entropy"][idx],
            args.bootstrap_repeats,
            rng,
            args.bootstrap_ci,
        )
        v_h, v_h_lo, v_h_hi = bootstrap_mean_ci(
            visual_knn["sample_entropy"][idx],
            args.bootstrap_repeats,
            rng,
            args.bootstrap_ci,
        )

        # Paired bootstrap: the same video supplies audio and visual z0.
        purity_gap_values = visual_corrected[idx] - audio_corrected[idx]
        entropy_gap_values = (
            audio_knn["sample_entropy"][idx]
            - visual_knn["sample_entropy"][idx]
        )
        p_gap, p_gap_lo, p_gap_hi = bootstrap_mean_ci(
            purity_gap_values,
            args.bootstrap_repeats,
            rng,
            args.bootstrap_ci,
        )
        h_gap, h_gap_lo, h_gap_hi = bootstrap_mean_ci(
            entropy_gap_values,
            args.bootstrap_repeats,
            rng,
            args.bootstrap_ci,
        )

        audio_conf_id, audio_conf_fraction = top_confusion(
            audio_knn["neighbor_labels"],
            labels,
            class_id,
            args.num_classes,
        )
        visual_conf_id, visual_conf_fraction = top_confusion(
            visual_knn["neighbor_labels"],
            labels,
            class_id,
            args.num_classes,
        )

        # Transparent rank-based noise score. High values require all three:
        # low local purity, diffuse neighbour labels, and weak class margin.
        audio_noise_score = float(
            (1.0 - audio_purity_pct[class_id])
            * audio_entropy_pct[class_id]
            * (1.0 - audio_margin_pct[class_id])
        )
        visual_noise_score = float(
            (1.0 - visual_purity_pct[class_id])
            * visual_entropy_pct[class_id]
            * (1.0 - visual_margin_pct[class_id])
        )

        # A strong flag is deliberately conservative: the candidate noisy
        # modality must be locally mixed, diffusely confused, globally weak,
        # and significantly worse than the paired modality.
        audio_asymmetric_noise = bool(
            audio_purity_pct[class_id] <= q
            and audio_entropy_pct[class_id] >= 1.0 - q
            and audio_margin_pct[class_id] <= q
            and p_gap_lo > 0.0
            and h_gap_lo > 0.0
        )
        visual_asymmetric_noise = bool(
            visual_purity_pct[class_id] <= q
            and visual_entropy_pct[class_id] >= 1.0 - q
            and visual_margin_pct[class_id] <= q
            and p_gap_hi < 0.0
            and h_gap_hi < 0.0
        )

        if audio_asymmetric_noise:
            asymmetry_label = "audio_noisy_relative_to_visual"
        elif visual_asymmetric_noise:
            asymmetry_label = "visual_noisy_relative_to_audio"
        else:
            asymmetry_label = "no_strong_asymmetric_noise"

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "support": n_c,
                "knn_k": int(audio_knn["k_eff"][0]),
                "same_class_chance": chance,
                "audio_raw_knn_purity": float(
                    audio_knn["sample_purity"][idx].mean()
                ),
                "visual_raw_knn_purity": float(
                    visual_knn["sample_purity"][idx].mean()
                ),
                "audio_chance_corrected_purity": a_p,
                "audio_corrected_purity_ci_low": a_p_lo,
                "audio_corrected_purity_ci_high": a_p_hi,
                "visual_chance_corrected_purity": v_p,
                "visual_corrected_purity_ci_low": v_p_lo,
                "visual_corrected_purity_ci_high": v_p_hi,
                "visual_minus_audio_corrected_purity": p_gap,
                "purity_gap_ci_low": p_gap_lo,
                "purity_gap_ci_high": p_gap_hi,
                "audio_neighbor_entropy": a_h,
                "audio_entropy_ci_low": a_h_lo,
                "audio_entropy_ci_high": a_h_hi,
                "visual_neighbor_entropy": v_h,
                "visual_entropy_ci_low": v_h_lo,
                "visual_entropy_ci_high": v_h_hi,
                "audio_minus_visual_entropy": h_gap,
                "entropy_gap_ci_low": h_gap_lo,
                "entropy_gap_ci_high": h_gap_hi,
                "audio_intra_dispersion": float(
                    audio_geo["intra_dispersion"][class_id]
                ),
                "audio_nearest_centroid_distance": float(
                    audio_geo["nearest_distance"][class_id]
                ),
                "audio_normalized_margin": float(
                    audio_geo["normalized_margin"][class_id]
                ),
                "visual_intra_dispersion": float(
                    visual_geo["intra_dispersion"][class_id]
                ),
                "visual_nearest_centroid_distance": float(
                    visual_geo["nearest_distance"][class_id]
                ),
                "visual_normalized_margin": float(
                    visual_geo["normalized_margin"][class_id]
                ),
                "audio_purity_percentile": float(
                    audio_purity_pct[class_id]
                ),
                "visual_purity_percentile": float(
                    visual_purity_pct[class_id]
                ),
                "audio_entropy_percentile": float(
                    audio_entropy_pct[class_id]
                ),
                "visual_entropy_percentile": float(
                    visual_entropy_pct[class_id]
                ),
                "audio_margin_percentile": float(
                    audio_margin_pct[class_id]
                ),
                "visual_margin_percentile": float(
                    visual_margin_pct[class_id]
                ),
                "audio_noise_score": audio_noise_score,
                "visual_noise_score": visual_noise_score,
                "audio_minus_visual_noise_score": float(
                    audio_noise_score - visual_noise_score
                ),
                "audio_geometry_type": geometry_type(
                    float(audio_margin_pct[class_id]),
                    float(audio_purity_pct[class_id]),
                ),
                "visual_geometry_type": geometry_type(
                    float(visual_margin_pct[class_id]),
                    float(visual_purity_pct[class_id]),
                ),
                "audio_top_confusion_class_id": audio_conf_id,
                "audio_top_confusion_class_name": class_names[audio_conf_id],
                "audio_top_confusion_fraction_all_neighbors": (
                    audio_conf_fraction
                ),
                "visual_top_confusion_class_id": visual_conf_id,
                "visual_top_confusion_class_name": class_names[visual_conf_id],
                "visual_top_confusion_fraction_all_neighbors": (
                    visual_conf_fraction
                ),
                "audio_asymmetric_noise_flag": int(
                    audio_asymmetric_noise
                ),
                "visual_asymmetric_noise_flag": int(
                    visual_asymmetric_noise
                ),
                "asymmetric_noise_label": asymmetry_label,
            }
        )

    rows_by_abs_gap = sorted(
        rows,
        key=lambda r: abs(
            float(r["visual_minus_audio_corrected_purity"])
        ),
        reverse=True,
    )
    audio_flags = [
        r for r in rows if int(r["audio_asymmetric_noise_flag"]) == 1
    ]
    visual_flags = [
        r for r in rows if int(r["visual_asymmetric_noise_flag"]) == 1
    ]

    write_csv(out_dir / "z0_modality_noise_by_class.csv", rows)
    np.savez_compressed(
        out_dir / "z0_sample_neighbor_metrics.npz",
        labels=labels,
        audio_sample_purity=audio_knn["sample_purity"],
        visual_sample_purity=visual_knn["sample_purity"],
        audio_chance_corrected_purity=audio_corrected,
        visual_chance_corrected_purity=visual_corrected,
        audio_sample_entropy=audio_knn["sample_entropy"],
        visual_sample_entropy=visual_knn["sample_entropy"],
    )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": int(len(labels)),
        "num_classes": int(args.num_classes),
        "knn_k": int(audio_knn["k_eff"][0]),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "bootstrap_ci": float(args.bootstrap_ci),
        "noise_quantile": float(args.noise_quantile),
        "claim_scope": (
            "representation-level local label mixing in fixed z0; "
            "not direct raw-signal SNR or corruption"
        ),
        "mean_audio_chance_corrected_purity": float(
            class_audio_purity.mean()
        ),
        "mean_visual_chance_corrected_purity": float(
            class_visual_purity.mean()
        ),
        "num_audio_asymmetric_noise_classes": len(audio_flags),
        "num_visual_asymmetric_noise_classes": len(visual_flags),
        "audio_asymmetric_noise_classes": [
            {
                "class_id": int(r["class_id"]),
                "class_name": str(r["class_name"]),
                "audio_corrected_purity": float(
                    r["audio_chance_corrected_purity"]
                ),
                "visual_corrected_purity": float(
                    r["visual_chance_corrected_purity"]
                ),
                "purity_gap_ci": [
                    float(r["purity_gap_ci_low"]),
                    float(r["purity_gap_ci_high"]),
                ],
                "entropy_gap_ci": [
                    float(r["entropy_gap_ci_low"]),
                    float(r["entropy_gap_ci_high"]),
                ],
            }
            for r in audio_flags
        ],
        "visual_asymmetric_noise_classes": [
            {
                "class_id": int(r["class_id"]),
                "class_name": str(r["class_name"]),
                "audio_corrected_purity": float(
                    r["audio_chance_corrected_purity"]
                ),
                "visual_corrected_purity": float(
                    r["visual_chance_corrected_purity"]
                ),
                "purity_gap_ci": [
                    float(r["purity_gap_ci_low"]),
                    float(r["purity_gap_ci_high"]),
                ],
                "entropy_gap_ci": [
                    float(r["entropy_gap_ci_low"]),
                    float(r["entropy_gap_ci_high"]),
                ],
            }
            for r in visual_flags
        ],
        "largest_absolute_modality_gaps": [
            {
                "class_id": int(r["class_id"]),
                "class_name": str(r["class_name"]),
                "visual_minus_audio_corrected_purity": float(
                    r["visual_minus_audio_corrected_purity"]
                ),
                "asymmetric_noise_label": str(
                    r["asymmetric_noise_label"]
                ),
            }
            for r in rows_by_abs_gap[:10]
        ],
    }
    save_json(summary, out_dir / "z0_modality_noise_summary.json")

    plot_purity_asymmetry(
        rows,
        out_dir / "z0_audio_vs_visual_corrected_purity.png",
        args.annotate_top,
    )
    plot_margin_purity(
        rows,
        "audio",
        out_dir / "z0_audio_margin_vs_purity.png",
        args.annotate_top,
    )
    plot_margin_purity(
        rows,
        "visual",
        out_dir / "z0_visual_margin_vs_purity.png",
        args.annotate_top,
    )
    plot_asymmetry_rank(
        rows,
        out_dir / "z0_modality_asymmetry_rank.png",
        args.rank_top_n,
    )
    plot_noise_score_rank(
        rows,
        out_dir / "z0_noise_score_asymmetry_rank.png",
        args.rank_top_n,
    )

    print("\n=== z0 modality-noise summary ===")
    print(f"samples: {len(labels)}")
    print(f"classes: {args.num_classes}")
    print(f"audio asymmetric-noise classes: {len(audio_flags)}")
    print(f"visual asymmetric-noise classes: {len(visual_flags)}")
    print(f"outputs: {out_dir}")

    if args.dataset != "AVE":
        close_fn = getattr(train_set, "close_visual_features_h5", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()