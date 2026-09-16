#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-class diagnostic for audio-guided visual attention.

Goal
----
Measure the decision-level contribution of A->V attention guidance directly,
then test whether single-modality geometry (audio hard / visual easy, etc.)
actually predicts that contribution.

The trained model is NOT modified. For each sample we run two counterfactual
branches with the SAME audio feature, visual projection, fusion rule and
classifier:

  Guided:
      z_G = classifier(z1_audio + z1_visual_guided)

  Uniform / unguided:
      z_U = classifier(z1_audio + z1_visual_uniform)

where z1_visual_guided uses the model's original audio-guided spatial+temporal
attention and z1_visual_uniform uses uniform pooling over the same visual tokens.

For true class y, define target-vs-rest margin

  B(z, y) = z_y - logsumexp_{k != y} z_k

and A->V guidance contribution

  C^{A->V} = B(z_G, y) - B(z_U, y).

Positive C: guidance improves the true-vs-rest decision margin.
Negative C: guidance harms the true-vs-rest decision margin.

Recommended usage
-----------------
Train ONE ordinary full-class model first:
  --num_classes 100 --class_num_per_step 100

Then run this script on step_0_best_model.pkl. Geometry is estimated on the
training split by default; guidance contribution is measured on the test split
by default, so the geometry-vs-utility correlation is not a same-sample artifact.
"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: str, map_location="cpu"):
    """Compatible with both old PyTorch and PyTorch >=2.6 whole-model loading."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def ensure_audio_visual_model(model) -> None:
    required = [
        "audio_proj", "visual_proj", "audio_visual_attention", "classifier"
    ]
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(
            "Checkpoint is not the expected audio-visual model. Missing: {}".format(missing)
        )


def reshape_visual(visual: torch.Tensor) -> torch.Tensor:
    """Match the original model: visual -> (B, 8, spatial_tokens, 768)."""
    if visual.ndim == 4:
        if visual.shape[1] != 8 or visual.shape[-1] != 768:
            raise ValueError("Unexpected 4-D visual shape: {}".format(tuple(visual.shape)))
        return visual
    if visual.ndim == 3:
        if visual.shape[-1] != 768:
            raise ValueError("Expected last visual dim 768, got {}".format(visual.shape[-1]))
        if visual.shape[1] % 8 != 0:
            raise ValueError(
                "Visual token count {} is not divisible by 8 frames".format(visual.shape[1])
            )
        return visual.reshape(visual.shape[0], 8, -1, 768)
    raise ValueError("Unexpected visual tensor shape: {}".format(tuple(visual.shape)))


@torch.no_grad()
def compute_branches(model, visual: torch.Tensor, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute exact baseline guided branch and the controlled uniform-attention branch.

    IMPORTANT: the only intervention between logits_guided and logits_uniform is
    how visual tokens are pooled. Audio z1, visual_proj and classifier are shared.
    """
    visual_4d = reshape_visual(visual)

    # Original audio-guided visual attention.
    spatial_attn, temporal_attn = model.audio_visual_attention(audio, visual_4d)
    guided_pool = torch.sum(spatial_attn * visual_4d, dim=2)
    guided_pool = torch.sum(temporal_attn * guided_pool, dim=1)

    # Exact unguided reference: uniform average over all frames/spatial tokens.
    uniform_pool = visual_4d.mean(dim=(1, 2))

    # z0 representations (before model projections).
    z0_audio = audio
    z0_visual_uniform = uniform_pool

    # z1 representations used by the classifier.
    z1_audio = F.relu(model.audio_proj(audio))
    z1_visual_guided = F.relu(model.visual_proj(guided_pool))
    z1_visual_uniform = F.relu(model.visual_proj(uniform_pool))

    logits_guided = model.classifier(z1_audio + z1_visual_guided)
    logits_uniform = model.classifier(z1_audio + z1_visual_uniform)

    return {
        "z0_audio": z0_audio,
        "z0_visual_uniform": z0_visual_uniform,
        "z1_audio": z1_audio,
        "z1_visual_guided": z1_visual_guided,
        "z1_visual_uniform": z1_visual_uniform,
        "logits_guided": logits_guided,
        "logits_uniform": logits_uniform,
        "spatial_attn": spatial_attn,
        "temporal_attn": temporal_attn,
    }


@torch.no_grad()
def verify_guided_forward(model, dataset, device: torch.device, batch_size: int, num_workers: int, tol: float = 1e-6) -> float:
    """Verify our reconstructed guided branch exactly matches the native model forward."""
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    data, _ = next(iter(loader))
    visual = data[0].to(device, non_blocking=True)
    audio = data[1].to(device, non_blocking=True)
    native = model(visual=visual, audio=audio)
    rebuilt = compute_branches(model, visual, audio)["logits_guided"]
    max_err = float((native - rebuilt).abs().max().item())
    if max_err > tol:
        raise RuntimeError(
            "Reconstructed guided forward does not match native model: "
            f"max_abs_error={max_err:.3e} > tol={tol:.3e}"
        )
    return max_err


def target_vs_rest_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """B(z,y)=z_y-logsumexp_{k!=y} z_k."""
    if logits.ndim != 2:
        raise ValueError("logits must be 2-D")
    if logits.shape[1] < 2:
        raise ValueError("target-vs-rest margin requires at least two classes")

    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    rest_lse = torch.logsumexp(masked, dim=1)
    return true_logits - rest_lse


def class_cap_indices(labels: torch.Tensor, counts: np.ndarray, cap: int) -> List[int]:
    """Choose batch indices without exceeding cap samples per class. cap<=0 means all."""
    if cap <= 0:
        return list(range(labels.numel()))
    chosen = []
    for i, c in enumerate(labels.detach().cpu().tolist()):
        if 0 <= c < len(counts) and counts[c] < cap:
            counts[c] += 1
            chosen.append(i)
    return chosen


@torch.no_grad()
def extract_geometry_bank(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    max_samples_per_class: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract train-split geometry banks for z0 and unguided z1."""
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    counts = np.zeros(num_classes, dtype=np.int64)
    labels_all: List[torch.Tensor] = []
    banks = defaultdict(list)

    model.eval()
    for data, labels in loader:
        labels_cpu = labels.long().cpu()
        idx = class_cap_indices(labels_cpu, counts, max_samples_per_class)
        if not idx:
            if max_samples_per_class > 0 and np.all(counts >= max_samples_per_class):
                break
            continue

        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)
        out = compute_branches(model, visual, audio)
        take = torch.as_tensor(idx, dtype=torch.long, device=device)

        labels_all.append(labels_cpu[idx])
        for key in ["z0_audio", "z0_visual_uniform", "z1_audio", "z1_visual_uniform"]:
            banks[key].append(out[key].index_select(0, take).detach().cpu())

        if max_samples_per_class > 0 and np.all(counts >= max_samples_per_class):
            break

    if not labels_all:
        raise RuntimeError("No samples extracted for geometry bank")

    labels_np = torch.cat(labels_all, dim=0).numpy().astype(np.int64)
    bank_np = {k: torch.cat(v, dim=0).numpy().astype(np.float32) for k, v in banks.items()}
    return labels_np, bank_np


def chunked_knn_sample_purity(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """Global cosine-kNN purity for every sample, computed in chunks."""
    n = int(features.shape[0])
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    k_eff = min(int(k), n - 1)
    if k_eff <= 0:
        return np.ones(n, dtype=np.float32)

    x = torch.from_numpy(features).float()
    x = F.normalize(x, dim=1)
    # Keep the bank on the requested device if practical; 6400x768 is modest.
    x_dev = x.to(device)
    labels_dev = torch.from_numpy(labels).long().to(device)
    result = torch.empty(n, dtype=torch.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q = x_dev[start:end]
        sim = q @ x_dev.t()

        local_rows = torch.arange(end - start, device=device)
        global_rows = torch.arange(start, end, device=device)
        sim[local_rows, global_rows] = float("-inf")

        nn_idx = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False).indices
        nn_labels = labels_dev[nn_idx]
        q_labels = labels_dev[start:end, None]
        purity = (nn_labels == q_labels).float().mean(dim=1)
        result[start:end] = purity.cpu()

    return result.numpy()


def geometry_by_class(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    knn_k: int,
    knn_device: torch.device,
    knn_chunk_size: int,
    eps: float,
) -> Dict[str, np.ndarray]:
    """
    Geometry definition used for diagnosis:
      S_c = mean cosine distance to normalized class centroid
      B_c = nearest other-class centroid cosine distance
      M_c = B_c / (S_c + eps)
      D_c = S_c / (B_c + eps)   (higher = harder)
      P_c = mean global cosine-kNN label purity
    """
    x = torch.from_numpy(features).float()
    x = F.normalize(x, dim=1)
    y = torch.from_numpy(labels).long()

    centroids = torch.full((num_classes, x.shape[1]), float("nan"), dtype=torch.float32)
    support = np.zeros(num_classes, dtype=np.int64)
    intra = np.full(num_classes, np.nan, dtype=np.float64)

    for c in range(num_classes):
        mask = (y == c)
        nc = int(mask.sum().item())
        support[c] = nc
        if nc == 0:
            continue
        xc = x[mask]
        centroid = F.normalize(xc.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        centroids[c] = centroid
        intra[c] = float((1.0 - xc @ centroid).mean().item())

    nearest = np.full(num_classes, np.nan, dtype=np.float64)
    valid = np.where(support > 0)[0]
    if len(valid) >= 2:
        valid_centroids = centroids[valid]
        sim = valid_centroids @ valid_centroids.t()
        dist = 1.0 - sim
        dist.fill_diagonal_(float("inf"))
        mins = dist.min(dim=1).values.cpu().numpy()
        nearest[valid] = mins

    margin = nearest / (intra + eps)
    difficulty = intra / (nearest + eps)

    sample_purity = chunked_knn_sample_purity(
        features=features,
        labels=labels,
        k=knn_k,
        device=knn_device,
        chunk_size=knn_chunk_size,
    )
    purity = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if np.any(mask):
            purity[c] = float(np.mean(sample_purity[mask]))

    return {
        "support": support,
        "intra_dispersion": intra,
        "nearest_centroid_distance": nearest,
        "normalized_margin": margin,
        "difficulty": difficulty,
        "knn_purity": purity,
    }


@torch.no_grad()
def extract_contribution(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    max_samples_per_class: int,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    counts = np.zeros(num_classes, dtype=np.int64)
    acc = defaultdict(list)

    model.eval()
    for data, labels in loader:
        labels_cpu = labels.long().cpu()
        idx = class_cap_indices(labels_cpu, counts, max_samples_per_class)
        if not idx:
            if max_samples_per_class > 0 and np.all(counts >= max_samples_per_class):
                break
            continue

        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)
        labels_dev = labels.long().to(device, non_blocking=True)
        out = compute_branches(model, visual, audio)

        logits_g = out["logits_guided"]
        logits_u = out["logits_uniform"]
        if logits_g.shape[1] != num_classes:
            raise RuntimeError(
                "Checkpoint has {} classifier outputs but --num_classes={}".format(
                    logits_g.shape[1], num_classes
                )
            )

        margin_g = target_vs_rest_margin(logits_g, labels_dev)
        margin_u = target_vs_rest_margin(logits_u, labels_dev)
        contrib = margin_g - margin_u

        true_g = logits_g.gather(1, labels_dev[:, None]).squeeze(1)
        true_u = logits_u.gather(1, labels_dev[:, None]).squeeze(1)
        true_delta = true_g - true_u
        # Since C = delta(true logit) - delta(rest logsumexp), this is useful diagnostically.
        rest_delta = true_delta - contrib

        pred_g = logits_g.argmax(dim=1)
        pred_u = logits_u.argmax(dim=1)
        correct_g = pred_g.eq(labels_dev)
        correct_u = pred_u.eq(labels_dev)
        rescue = correct_g & (~correct_u)
        suppression = (~correct_g) & correct_u

        take = torch.as_tensor(idx, dtype=torch.long, device=device)
        tensors = {
            "label": labels_dev,
            "margin_guided": margin_g,
            "margin_uniform": margin_u,
            "guidance_contribution": contrib,
            "true_logit_delta": true_delta,
            "rest_lse_delta": rest_delta,
            "pred_guided": pred_g,
            "pred_uniform": pred_u,
            "correct_guided": correct_g.long(),
            "correct_uniform": correct_u.long(),
            "rescue": rescue.long(),
            "suppression": suppression.long(),
        }
        for key, tensor in tensors.items():
            acc[key].append(tensor.index_select(0, take).detach().cpu())

        if max_samples_per_class > 0 and np.all(counts >= max_samples_per_class):
            break

    if not acc:
        raise RuntimeError("No samples extracted for contribution diagnostic")
    return {k: torch.cat(v, dim=0).numpy() for k, v in acc.items()}


def mean_ci95(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else np.nan
    half = 1.96 * se
    return mean, std, mean - half, mean + half


def assign_exact_30_40_30_groups(values: np.ndarray) -> List[str]:
    """Exact rank split among finite classes: easiest 30%, middle 40%, hardest rest."""
    values = np.asarray(values, dtype=np.float64)
    groups = ["missing"] * len(values)
    valid = np.where(np.isfinite(values))[0]
    if len(valid) == 0:
        return groups
    order = valid[np.argsort(values[valid], kind="stable")]
    n = len(order)
    n_easy = int(round(0.30 * n))
    n_medium = int(round(0.40 * n))
    # Keep all valid classes assigned even under rounding.
    n_easy = min(max(n_easy, 0), n)
    n_medium = min(max(n_medium, 0), n - n_easy)
    for idx in order[:n_easy]:
        groups[int(idx)] = "easy"
    for idx in order[n_easy:n_easy + n_medium]:
        groups[int(idx)] = "medium"
    for idx in order[n_easy + n_medium:]:
        groups[int(idx)] = "hard"
    return groups


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x, y = x[mask], y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def simple_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with average tie handling, implemented without scipy."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return pearson(simple_ranks(x[mask]), simple_ranks(y[mask]))


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def aggregate_contribution_by_class(
    sample: Dict[str, np.ndarray],
    num_classes: int,
    id_to_category: Dict[int, str],
) -> List[Dict]:
    rows = []
    labels = sample["label"].astype(np.int64)
    for c in range(num_classes):
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "class_id": c,
                "category_name": id_to_category.get(c, str(c)),
                "support_test": 0,
            })
            continue

        contrib = sample["guidance_contribution"][mask]
        mean_c, std_c, ci_lo, ci_hi = mean_ci95(contrib)
        sign = "uncertain"
        if np.isfinite(ci_hi) and ci_hi < 0:
            sign = "negative"
        elif np.isfinite(ci_lo) and ci_lo > 0:
            sign = "positive"

        row = {
            "class_id": c,
            "category_name": id_to_category.get(c, str(c)),
            "support_test": n,
            "guided_margin_mean": float(np.mean(sample["margin_guided"][mask])),
            "uniform_margin_mean": float(np.mean(sample["margin_uniform"][mask])),
            "guidance_contribution_mean": mean_c,
            "guidance_contribution_std": std_c,
            "guidance_contribution_median": float(np.median(contrib)),
            "guidance_contribution_ci95_low": ci_lo,
            "guidance_contribution_ci95_high": ci_hi,
            "guidance_contribution_sign": sign,
            "negative_sample_fraction": float(np.mean(contrib < 0)),
            "true_logit_delta_mean": float(np.mean(sample["true_logit_delta"][mask])),
            "rest_lse_delta_mean": float(np.mean(sample["rest_lse_delta"][mask])),
            "guided_acc": float(np.mean(sample["correct_guided"][mask])),
            "uniform_acc": float(np.mean(sample["correct_uniform"][mask])),
            "delta_acc_guided_minus_uniform": float(
                np.mean(sample["correct_guided"][mask]) - np.mean(sample["correct_uniform"][mask])
            ),
            "rescue_rate": float(np.mean(sample["rescue"][mask])),
            "suppression_rate": float(np.mean(sample["suppression"][mask])),
            "rescue_minus_suppression_rate": float(
                np.mean(sample["rescue"][mask]) - np.mean(sample["suppression"][mask])
            ),
        }
        rows.append(row)
    return rows


def attach_geometry(
    rows: List[Dict],
    prefix: str,
    geom: Dict[str, np.ndarray],
) -> None:
    for row in rows:
        c = int(row["class_id"])
        row[prefix + "_support_geometry"] = int(geom["support"][c])
        for key in [
            "intra_dispersion", "nearest_centroid_distance", "normalized_margin",
            "difficulty", "knn_purity"
        ]:
            row[prefix + "_" + key] = float(geom[key][c]) if np.isfinite(geom[key][c]) else np.nan


def add_groups_and_gaps(rows: List[Dict], space: str) -> None:
    a = np.array([r.get(space + "_audio_difficulty", np.nan) for r in rows], dtype=np.float64)
    v = np.array([r.get(space + "_visual_difficulty", np.nan) for r in rows], dtype=np.float64)
    a_group = assign_exact_30_40_30_groups(a)
    v_group = assign_exact_30_40_30_groups(v)
    for i, row in enumerate(rows):
        row[space + "_audio_group"] = a_group[i]
        row[space + "_visual_group"] = v_group[i]
        row[space + "_difficulty_gap_audio_minus_visual"] = (
            float(a[i] - v[i]) if np.isfinite(a[i]) and np.isfinite(v[i]) else np.nan
        )
        row[space + "_naive_audio_hard_visual_easy"] = int(
            a_group[i] == "hard" and v_group[i] == "easy"
        )


def class_metric_array(rows: List[Dict], key: str) -> np.ndarray:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, np.nan)))
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals, dtype=np.float64)


def cross_table(rows: List[Dict], space: str) -> List[Dict]:
    result = []
    order = ["easy", "medium", "hard"]
    for ag in order:
        for vg in order:
            subset = [
                r for r in rows
                if r.get(space + "_audio_group") == ag and r.get(space + "_visual_group") == vg
                and int(r.get("support_test", 0)) > 0
            ]
            cvals = np.array([r.get("guidance_contribution_mean", np.nan) for r in subset], dtype=float)
            dacc = np.array([r.get("delta_acc_guided_minus_uniform", np.nan) for r in subset], dtype=float)
            neg = np.array([float(v < 0) for v in cvals if np.isfinite(v)], dtype=float)
            confneg = np.array([
                float(r.get("guidance_contribution_sign") == "negative") for r in subset
            ], dtype=float)
            result.append({
                "space": space,
                "audio_group": ag,
                "visual_group": vg,
                "num_classes": len(subset),
                "mean_guidance_contribution": float(np.nanmean(cvals)) if len(cvals) else np.nan,
                "negative_class_fraction": float(np.mean(neg)) if len(neg) else np.nan,
                "confident_negative_class_fraction": float(np.mean(confneg)) if len(confneg) else np.nan,
                "mean_delta_acc_guided_minus_uniform": float(np.nanmean(dacc)) if len(dacc) else np.nan,
            })
    return result


def summarize_space(rows: List[Dict], space: str) -> Dict:
    gap = class_metric_array(rows, space + "_difficulty_gap_audio_minus_visual")
    contrib = class_metric_array(rows, "guidance_contribution_mean")
    dacc = class_metric_array(rows, "delta_acc_guided_minus_uniform")
    naive = np.array([bool(r.get(space + "_naive_audio_hard_visual_easy", 0)) for r in rows])
    valid = np.isfinite(contrib)
    negative = valid & (contrib < 0)
    confident_negative = np.array([
        r.get("guidance_contribution_sign") == "negative" for r in rows
    ]) & valid

    non_extreme = valid & (~naive)
    summary = {
        "pearson_difficulty_gap_vs_guidance_contribution": pearson(gap, contrib),
        "spearman_difficulty_gap_vs_guidance_contribution": spearman(gap, contrib),
        "pearson_difficulty_gap_vs_delta_acc": pearson(gap, dacc),
        "spearman_difficulty_gap_vs_delta_acc": spearman(gap, dacc),
        "pearson_gap_vs_contribution_excluding_naive_group": pearson(gap[non_extreme], contrib[non_extreme]),
        "spearman_gap_vs_contribution_excluding_naive_group": spearman(gap[non_extreme], contrib[non_extreme]),
        "num_naive_audio_hard_visual_easy_classes": int(naive.sum()),
        "mean_contribution_naive_group": float(np.nanmean(contrib[naive])) if np.any(naive) else np.nan,
        "mean_contribution_rest": float(np.nanmean(contrib[~naive & valid])) if np.any(~naive & valid) else np.nan,
        # Precision/recall language here treats C<0 as the phenomenon the old naive rule tries to select.
        "naive_precision_for_negative_contribution": float((naive & negative).sum() / naive.sum()) if naive.sum() else np.nan,
        "naive_recall_of_negative_contribution": float((naive & negative).sum() / negative.sum()) if negative.sum() else np.nan,
        "naive_precision_for_confident_negative_contribution": float((naive & confident_negative).sum() / naive.sum()) if naive.sum() else np.nan,
        "naive_recall_of_confident_negative_contribution": float((naive & confident_negative).sum() / confident_negative.sum()) if confident_negative.sum() else np.nan,
    }
    return summary


def save_sample_csv(path: str, sample: Dict[str, np.ndarray], id_to_category: Dict[int, str]) -> None:
    rows = []
    n = len(sample["label"])
    for i in range(n):
        c = int(sample["label"][i])
        rows.append({
            "sample_index": i,
            "class_id": c,
            "category_name": id_to_category.get(c, str(c)),
            "margin_guided": float(sample["margin_guided"][i]),
            "margin_uniform": float(sample["margin_uniform"][i]),
            "guidance_contribution": float(sample["guidance_contribution"][i]),
            "true_logit_delta": float(sample["true_logit_delta"][i]),
            "rest_lse_delta": float(sample["rest_lse_delta"][i]),
            "pred_guided": int(sample["pred_guided"][i]),
            "pred_uniform": int(sample["pred_uniform"][i]),
            "correct_guided": int(sample["correct_guided"][i]),
            "correct_uniform": int(sample["correct_uniform"][i]),
            "rescue": int(sample["rescue"][i]),
            "suppression": int(sample["suppression"][i]),
        })
    write_csv(path, list(rows[0].keys()) if rows else [], rows)


def make_plots(rows: List[Dict], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[WARN] matplotlib unavailable; skipping plots: {}".format(exc))
        return

    os.makedirs(out_dir, exist_ok=True)
    contrib = class_metric_array(rows, "guidance_contribution_mean")

    for space in ["z0", "z1"]:
        gap = class_metric_array(rows, space + "_difficulty_gap_audio_minus_visual")
        mask = np.isfinite(gap) & np.isfinite(contrib)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(gap[mask], contrib[mask], alpha=0.75)
        ax.axhline(0.0, linewidth=1)
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("Audio difficulty - Visual difficulty ({})".format(space))
        ax.set_ylabel("Mean A->V guidance contribution")
        ax.set_title("Geometry gap vs decision-level guidance utility ({})".format(space))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "{}_gap_vs_guidance_contribution.png".format(space)), dpi=180)
        plt.close(fig)

    # Most helpful vs most harmful classes.
    valid_idx = np.where(np.isfinite(contrib))[0]
    if len(valid_idx):
        order = valid_idx[np.argsort(contrib[valid_idx])]
        take = np.concatenate([order[:10], order[-10:]]) if len(order) >= 20 else order
        names = [rows[int(i)].get("category_name", str(i)) for i in take]
        vals = contrib[take]
        fig_h = max(6, 0.32 * len(take))
        fig, ax = plt.subplots(figsize=(9, fig_h))
        y = np.arange(len(take))
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("Mean A->V guidance contribution")
        ax.set_title("Most harmful / helpful guidance classes")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "guidance_contribution_extreme_classes.png"), dpi=180)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-class A->V guidance contribution diagnostic")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--feature_root", type=str, required=True)
    p.add_argument("--meta_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--modality", type=str, default="audio-visual")
    p.add_argument("--num_classes", type=int, default=100)
    p.add_argument("--class_num_per_step", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--infer_batch_size", type=int, default=32)
    p.add_argument("--geometry_batch_size", type=int, default=64)
    p.add_argument("--geometry_split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--contribution_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--max_geometry_samples_per_class", type=int, default=0,
                   help="<=0 uses all samples. Recommended for the full-class diagnostic.")
    p.add_argument("--max_contribution_samples_per_class", type=int, default=0,
                   help="<=0 uses all samples from the contribution split.")
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--knn_chunk_size", type=int, default=512)
    p.add_argument("--knn_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--no_sample_csv", action="store_true")

    # Extra attributes commonly expected by the project's dataloader. They do not
    # change the diagnostic itself but make the Args object compatible.
    p.add_argument("--memory_size", type=int, default=500)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--exemplar_batch_size", type=int, default=128)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.class_num_per_step != args.num_classes:
        raise ValueError(
            "This script is intentionally a full-class diagnostic. Set "
            "--class_num_per_step equal to --num_classes (e.g. 100)."
        )
    setup_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.knn_device == "auto":
        knn_device = device
    elif args.knn_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--knn_device cuda requested but CUDA is unavailable")
        knn_device = torch.device("cuda:0")
    else:
        knn_device = torch.device("cpu")

    if args.checkpoint is None:
        args.checkpoint = "./save/{}/step_0_best_model.pkl".format(args.dataset)
    if args.output_dir is None:
        args.output_dir = "./save/diagnostics/{}/fullclass_guidance".format(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("Checkpoint not found: {}".format(args.checkpoint))

    # Import project modules only at runtime, so helper functions remain testable.
    # Importing the model module also makes whole-model torch.load pickle resolution explicit.
    from dataloader_ours import IcaAVELoader
    from model.audio_visual_model_incremental import IncreAudioVisualNet  # noqa: F401

    print("Loading checkpoint: {}".format(args.checkpoint))
    model = unwrap_model(safe_torch_load(args.checkpoint, map_location="cpu"))
    ensure_audio_visual_model(model)
    model = model.to(device)
    model.eval()

    # Verify classifier size before doing expensive work.
    if hasattr(model.classifier, "out_features") and int(model.classifier.out_features) != args.num_classes:
        raise RuntimeError(
            "Expected a {}-class full-class checkpoint, but classifier.out_features={}".format(
                args.num_classes, model.classifier.out_features
            )
        )

    print("Building geometry dataset: {}".format(args.geometry_split))
    geometry_set = IcaAVELoader(args=args, mode=args.geometry_split, modality=args.modality)
    geometry_set.set_incremental_step(0)

    print("Building contribution dataset: {}".format(args.contribution_split))
    contribution_set = IcaAVELoader(args=args, mode=args.contribution_split, modality=args.modality)
    contribution_set.set_incremental_step(0)

    category_encode_dict = getattr(contribution_set, "category_encode_dict", {})
    id_to_category = {int(v): str(k) for k, v in category_encode_dict.items()}

    max_forward_err = verify_guided_forward(
        model=model, dataset=contribution_set, device=device,
        batch_size=args.infer_batch_size, num_workers=args.num_workers, tol=1e-6,
    )
    print("Guided forward verification passed; max abs error = {:.3e}".format(max_forward_err))

    print("Extracting geometry bank...")
    geom_labels, banks = extract_geometry_bank(
        model=model,
        dataset=geometry_set,
        device=device,
        batch_size=args.geometry_batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        max_samples_per_class=args.max_geometry_samples_per_class,
    )
    unique, counts = np.unique(geom_labels, return_counts=True)
    print("Geometry bank: {} samples, {} classes, min/max per present class = {}/{}".format(
        len(geom_labels), len(unique), int(counts.min()), int(counts.max())
    ))
    if len(unique) != args.num_classes:
        print("[WARN] Geometry split contains only {} / {} classes".format(len(unique), args.num_classes))

    geometry = {}
    for key in ["z0_audio", "z0_visual_uniform", "z1_audio", "z1_visual_uniform"]:
        print("Computing geometry: {}".format(key))
        geometry[key] = geometry_by_class(
            features=banks[key],
            labels=geom_labels,
            num_classes=args.num_classes,
            knn_k=args.knn_k,
            knn_device=knn_device,
            knn_chunk_size=args.knn_chunk_size,
            eps=args.eps,
        )

    print("Extracting guided-vs-uniform decision contribution...")
    sample = extract_contribution(
        model=model,
        dataset=contribution_set,
        device=device,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        max_samples_per_class=args.max_contribution_samples_per_class,
    )

    rows = aggregate_contribution_by_class(sample, args.num_classes, id_to_category)
    attach_geometry(rows, "z0_audio", geometry["z0_audio"])
    attach_geometry(rows, "z0_visual", geometry["z0_visual_uniform"])
    attach_geometry(rows, "z1_audio", geometry["z1_audio"])
    attach_geometry(rows, "z1_visual", geometry["z1_visual_uniform"])
    add_groups_and_gaps(rows, "z0")
    add_groups_and_gaps(rows, "z1")

    # Ensure deterministic, readable column order.
    base_fields = [
        "class_id", "category_name", "support_test",
        "guided_margin_mean", "uniform_margin_mean",
        "guidance_contribution_mean", "guidance_contribution_std",
        "guidance_contribution_median", "guidance_contribution_ci95_low",
        "guidance_contribution_ci95_high", "guidance_contribution_sign",
        "negative_sample_fraction", "true_logit_delta_mean", "rest_lse_delta_mean",
        "guided_acc", "uniform_acc", "delta_acc_guided_minus_uniform",
        "rescue_rate", "suppression_rate", "rescue_minus_suppression_rate",
    ]
    geom_fields = []
    for prefix in ["z0_audio", "z0_visual", "z1_audio", "z1_visual"]:
        geom_fields += [
            prefix + "_support_geometry",
            prefix + "_intra_dispersion",
            prefix + "_nearest_centroid_distance",
            prefix + "_normalized_margin",
            prefix + "_difficulty",
            prefix + "_knn_purity",
        ]
    group_fields = []
    for space in ["z0", "z1"]:
        group_fields += [
            space + "_audio_group", space + "_visual_group",
            space + "_difficulty_gap_audio_minus_visual",
            space + "_naive_audio_hard_visual_easy",
        ]
    class_csv = os.path.join(args.output_dir, "class_guidance_diagnostic.csv")
    write_csv(class_csv, base_fields + geom_fields + group_fields, rows)

    cross_rows = cross_table(rows, "z0") + cross_table(rows, "z1")
    cross_csv = os.path.join(args.output_dir, "difficulty_cross_table.csv")
    write_csv(cross_csv, list(cross_rows[0].keys()), cross_rows)

    if not args.no_sample_csv:
        sample_csv = os.path.join(args.output_dir, "sample_guidance_contribution.csv")
        save_sample_csv(sample_csv, sample, id_to_category)
    else:
        sample_csv = None

    overall_guided_acc = float(np.mean(sample["correct_guided"]))
    overall_uniform_acc = float(np.mean(sample["correct_uniform"]))
    overall_contrib = float(np.mean(sample["guidance_contribution"]))
    overall_rescue = float(np.mean(sample["rescue"]))
    overall_suppression = float(np.mean(sample["suppression"]))

    summary = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "geometry_split": args.geometry_split,
        "contribution_split": args.contribution_split,
        "num_classes": args.num_classes,
        "num_geometry_samples": int(len(geom_labels)),
        "num_contribution_samples": int(len(sample["label"])),
        "guided_forward_max_abs_error": max_forward_err,
        "overall": {
            "guided_acc": overall_guided_acc,
            "uniform_acc": overall_uniform_acc,
            "delta_acc_guided_minus_uniform": overall_guided_acc - overall_uniform_acc,
            "mean_guidance_contribution": overall_contrib,
            "negative_sample_fraction": float(np.mean(sample["guidance_contribution"] < 0)),
            "rescue_rate": overall_rescue,
            "suppression_rate": overall_suppression,
            "rescue_minus_suppression_rate": overall_rescue - overall_suppression,
        },
        "z0": summarize_space(rows, "z0"),
        "z1": summarize_space(rows, "z1"),
        "interpretation": {
            "main_quantity": "C_AtoV = target-vs-rest margin(guided) - target-vs-rest margin(uniform)",
            "negative_means": "for this trained model, replacing uniform visual pooling with matched audio-guided pooling lowers the true-vs-rest margin",
            "naive_group": "audio hard + visual easy under exact 30/40/30 class ranking",
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    make_plots(rows, args.output_dir)

    print("\n=== Full-class guidance diagnostic ===")
    print("Guided acc:  {:.4f}".format(overall_guided_acc))
    print("Uniform acc: {:.4f}".format(overall_uniform_acc))
    print("Delta acc:   {:+.4f}".format(overall_guided_acc - overall_uniform_acc))
    print("Mean C_AtoV: {:+.6f}".format(overall_contrib))
    print("Rescue:      {:.4f}".format(overall_rescue))
    print("Suppression: {:.4f}".format(overall_suppression))
    for space in ["z0", "z1"]:
        s = summary[space]
        print("\n[{}] Spearman(diff_gap, C_AtoV) = {}".format(
            space, "nan" if not np.isfinite(s["spearman_difficulty_gap_vs_guidance_contribution"])
            else "{:+.4f}".format(s["spearman_difficulty_gap_vs_guidance_contribution"])
        ))
        print("[{}] naive group classes = {}".format(space, s["num_naive_audio_hard_visual_easy_classes"]))
        print("[{}] naive precision for C<0 = {}".format(
            space, "nan" if not np.isfinite(s["naive_precision_for_negative_contribution"])
            else "{:.4f}".format(s["naive_precision_for_negative_contribution"])
        ))
        print("[{}] naive recall of C<0 = {}".format(
            space, "nan" if not np.isfinite(s["naive_recall_of_negative_contribution"])
            else "{:.4f}".format(s["naive_recall_of_negative_contribution"])
        ))

    print("\nSaved:")
    print("  {}".format(class_csv))
    print("  {}".format(cross_csv))
    if sample_csv:
        print("  {}".format(sample_csv))
    print("  {}".format(summary_path))

    # Close HDF5-backed loaders if applicable.
    for ds in [geometry_set, contribution_set]:
        close_fn = getattr(ds, "close_visual_features_h5", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()