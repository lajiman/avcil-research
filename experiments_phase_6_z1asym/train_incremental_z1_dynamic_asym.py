"""
Dynamic asymmetric z1 replacement for AVCIL.

Controlled experiment:
  * replace AVCIL instance/class cross-modal z1 contrastive losses;
  * preserve current/replay CE, old-logit KD, attention-score KD, and replay;
  * do not use z2 features, z2 difficulty, or a z2 auxiliary loss.

Dynamic class-modality difficulty:
  H_geo = 1 - joint_percentile_rank(normalized_margin)
  H_ucom = 1 - joint_percentile_rank(prototype_space_UCoM)
  H = eta * H_geo + (1 - eta) * H_ucom

Action 1:
  purity-gated modality-specific prototypical self-repair, weighted by H * Q.

Action 2:
  teacher-purity-gated asymmetric non-target correction. The teacher is the
  easier modality for that class; no raw feature alignment is imposed.

Statistics are computed after every epoch by default, updated with EMA, and
used only from the following epoch. A per-step warm-up is enabled by default.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random
from itertools import cycle
import csv
import json

from tsne_plotter import make_tsne_plots_for_step


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def CE_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)
    return -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))


def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()


def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def compute_per_class_prf(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    y_true = y_true.long()
    y_pred = y_pred.long()

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        tp[c] = (true_c & pred_c).sum()
        fp[c] = ((~true_c) & pred_c).sum()
        fn[c] = (true_c & (~pred_c)).sum()

    support = tp + fn
    precision = torch.zeros(num_classes, dtype=torch.float32)
    recall = torch.zeros(num_classes, dtype=torch.float32)
    f1 = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        tp_c = tp[c].item()
        fp_c = fp[c].item()
        fn_c = fn[c].item()
        p = safe_div(tp_c, tp_c + fp_c)
        r = safe_div(tp_c, tp_c + fn_c)
        precision[c] = p
        recall[c] = r
        f1[c] = safe_div(2 * p * r, p + r)

    return {
        "tp": tp.cpu().numpy(),
        "fp": fp.cpu().numpy(),
        "fn": fn.cpu().numpy(),
        "support": support.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_csv_rows(csv_path, header, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================================
# Dynamic z1 statistics
# ============================================================================

def empty_z1_state(num_classes, feature_dim=768):
    """CPU-resident, non-trainable class/modality statistics."""
    zeros_c = lambda: torch.zeros(num_classes)
    return {
        "audio_prototypes": torch.zeros(num_classes, feature_dim),
        "visual_prototypes": torch.zeros(num_classes, feature_dim),
        # Geometry component: inverse percentile rank of normalized margin.
        "audio_geo_hardness": zeros_c(),
        "visual_geo_hardness": zeros_c(),
        # Decision component: inverse percentile rank of prototype UCoM.
        "audio_ucom_hardness": zeros_c(),
        "visual_ucom_hardness": zeros_c(),
        # Weighted combination used by the two z1 actions.
        "audio_hardness": zeros_c(),
        "visual_hardness": zeros_c(),
        # Raw diagnostics retained for interpretation.
        "audio_intra": zeros_c(),
        "visual_intra": zeros_c(),
        "audio_inter": zeros_c(),
        "visual_inter": zeros_c(),
        "audio_margin": zeros_c(),
        "visual_margin": zeros_c(),
        "audio_ucom": zeros_c(),
        "visual_ucom": zeros_c(),
        "audio_purity": zeros_c(),
        "visual_purity": zeros_c(),
        "valid": torch.zeros(num_classes, dtype=torch.bool),
        "support": torch.zeros(num_classes, dtype=torch.long),
    }


def expand_z1_state(state, num_classes, feature_dim=768):
    """Expand a saved state and tolerate earlier ablation-state formats."""
    if state is None:
        return empty_z1_state(num_classes, feature_dim)
    old_num = state["valid"].numel()
    if old_num > num_classes:
        raise ValueError("Cannot shrink z1 state from {} to {} classes".format(old_num, num_classes))
    expanded = empty_z1_state(num_classes, feature_dim)
    for key in expanded:
        if key in state:
            expanded[key][:old_num] = state[key][:old_num]
    return expanded


def load_previous_z1_state(args, step, num_classes):
    if step == 0:
        return empty_z1_state(num_classes)
    path = './save/{}/step_{}_best_z1_stats.pt'.format(args.dataset, step - 1)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Previous z1 statistics are missing: {}. "
            "Run earlier incremental steps with this training script.".format(path)
        )
    state = torch.load(path, map_location='cpu')
    return expand_z1_state(state, num_classes)


def state_to_device(state, target_device):
    return {key: value.to(target_device) for key, value in state.items()}


def percentile_rank(values):
    """Map larger values to larger empirical percentile ranks in [0, 1]."""
    values = values.float()
    n = values.numel()
    if n <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(n, dtype=values.dtype, device=values.device)
    return ranks / float(n - 1)


def _sample_per_class(audio_features, visual_features, labels, num_classes, max_per_class, seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    selected = []
    for c in range(num_classes):
        idx = torch.where(labels == c)[0]
        if idx.numel() == 0:
            continue
        if max_per_class > 0 and idx.numel() > max_per_class:
            perm = torch.randperm(idx.numel(), generator=generator)[:max_per_class]
            idx = idx[perm]
        selected.append(idx)
    if not selected:
        raise RuntimeError("No samples were collected for z1 statistics")
    selected = torch.cat(selected, dim=0)
    return audio_features[selected], visual_features[selected], labels[selected]


def _compute_prototypes(features, labels, num_classes):
    feature_dim = features.shape[1]
    prototypes = torch.zeros(num_classes, feature_dim, device=features.device)
    support = torch.zeros(num_classes, dtype=torch.long, device=features.device)
    observed = torch.zeros(num_classes, dtype=torch.bool, device=features.device)
    for c in range(num_classes):
        mask = labels == c
        count = int(mask.sum().item())
        if count == 0:
            continue
        prototypes[c] = F.normalize(features[mask].mean(dim=0), dim=0)
        support[c] = count
        observed[c] = True
    return prototypes, support, observed


def _ema_prototypes(previous, observed_proto, observed_mask, previous_valid, momentum):
    updated = previous.clone()
    for c in torch.where(observed_mask)[0].tolist():
        if bool(previous_valid[c]):
            mixed = momentum * previous[c] + (1.0 - momentum) * observed_proto[c]
            updated[c] = F.normalize(mixed, dim=0)
        else:
            updated[c] = observed_proto[c]
    return updated


def _ema_scalar(previous, observed_value, observed_mask, previous_valid, momentum):
    updated = previous.clone()
    for c in torch.where(observed_mask)[0].tolist():
        if bool(previous_valid[c]):
            updated[c] = momentum * previous[c] + (1.0 - momentum) * observed_value[c]
        else:
            updated[c] = observed_value[c]
    return updated


def _class_geometry(features, labels, prototypes, valid, num_classes, inter_topk):
    """Class geometry: intra dispersion, nearest-centroid distance, normalized margin."""
    intra = torch.zeros(num_classes, device=features.device)
    inter = torch.zeros(num_classes, device=features.device)
    margin = torch.zeros(num_classes, device=features.device)
    observed = torch.zeros(num_classes, dtype=torch.bool, device=features.device)

    valid_idx = torch.where(valid)[0]
    for c in range(num_classes):
        mask = labels == c
        if not mask.any() or not bool(valid[c]):
            continue
        competitors = valid_idx[valid_idx != c]
        if competitors.numel() == 0:
            continue

        # Cosine distance because features and prototypes are L2-normalized.
        intra_c = (1.0 - torch.mv(features[mask], prototypes[c])).mean()
        centroid_dist = 1.0 - torch.mv(prototypes[competitors], prototypes[c])
        k = min(inter_topk, centroid_dist.numel())
        inter_c = torch.topk(centroid_dist, k=k, largest=False).values.mean()

        intra[c] = intra_c
        inter[c] = inter_c
        margin[c] = inter_c / intra_c.clamp_min(EPS)
        observed[c] = True
    return intra, inter, margin, observed


def _class_ucom(features, labels, prototypes, valid, temperature, num_classes):
    """
    Prototype-space UCoM inspired by MNL:
        true-class logit - strongest non-target logit.
    Larger UCoM means easier / more reliable modality-specific evidence.
    """
    logits = torch.mm(features, prototypes.t()) / temperature
    logits[:, ~valid] = -1e4
    idx = torch.arange(labels.shape[0], device=features.device)
    true_logits = logits[idx, labels]
    non_target_logits = logits.clone()
    non_target_logits[idx, labels] = -1e4
    competitor_logits = non_target_logits.max(dim=1).values
    sample_ucom = true_logits - competitor_logits

    class_ucom = torch.zeros(num_classes, device=features.device)
    observed = torch.zeros(num_classes, dtype=torch.bool, device=features.device)
    if int(valid.sum().item()) < 2:
        return class_ucom, observed
    for c in range(num_classes):
        mask = labels == c
        if mask.any() and bool(valid[c]):
            class_ucom[c] = sample_ucom[mask].mean()
            observed[c] = True
    return class_ucom, observed


def _class_knn_purity(features, labels, num_classes, knn_k):
    """
    Chance-corrected local label purity.

    Purity is computed in each modality's normalized z1 space. Self-neighbors are
    excluded. For class c, the chance baseline is (N_c - 1) / (N - 1).
    """
    n = features.shape[0]
    purity = torch.zeros(num_classes, device=features.device)
    observed = torch.zeros(num_classes, dtype=torch.bool, device=features.device)
    if n <= 1:
        return purity, observed

    k = min(knn_k, n - 1)
    similarity = torch.mm(features, features.t())
    similarity.fill_diagonal_(-float('inf'))
    neighbor_idx = similarity.topk(k=k, dim=1, largest=True).indices
    neighbor_labels = labels[neighbor_idx]
    sample_purity = (neighbor_labels == labels.unsqueeze(1)).float().mean(dim=1)

    for c in range(num_classes):
        mask = labels == c
        n_c = int(mask.sum().item())
        if n_c <= 1:
            continue
        raw = sample_purity[mask].mean()
        chance = float(n_c - 1) / float(n - 1)
        corrected = (raw - chance) / max(1.0 - chance, EPS)
        purity[c] = corrected.clamp(0.0, 1.0)
        observed[c] = True
    return purity, observed


def _collect_z1_features(model, loaders):
    audio_list = []
    visual_list = []
    label_list = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for loader in loaders:
            if loader is None:
                continue
            for data, labels in loader:
                visual = data[0].to(device, non_blocking=True)
                audio = data[1].to(device, non_blocking=True)
                outputs = model(
                    visual=visual,
                    audio=audio,
                    return_dict=True,
                    out_analysis_features=True,
                )
                audio_list.append(outputs["z1_audio_norm"].detach().cpu())
                visual_list.append(outputs["z1_visual_norm"].detach().cpu())
                label_list.append(labels.detach().cpu().long())
    if was_training:
        model.train()
    if not audio_list:
        raise RuntimeError("The z1 statistics loaders yielded no samples")
    return torch.cat(audio_list), torch.cat(visual_list), torch.cat(label_list)


def update_z1_state(args, model, stats_loaders, state, step, epoch):
    num_classes = (step + 1) * args.class_num_per_step
    audio_cpu, visual_cpu, labels_cpu = _collect_z1_features(model, stats_loaders)
    audio_cpu, visual_cpu, labels_cpu = _sample_per_class(
        audio_cpu,
        visual_cpu,
        labels_cpu,
        num_classes=num_classes,
        max_per_class=args.z1_stats_max_per_class,
        seed=args.seed + step * 10000 + epoch,
    )

    audio = F.normalize(audio_cpu.to(device), dim=1)
    visual = F.normalize(visual_cpu.to(device), dim=1)
    labels = labels_cpu.to(device)
    previous = state_to_device(state, device)

    obs_audio_proto, support, observed = _compute_prototypes(audio, labels, num_classes)
    obs_visual_proto, _, _ = _compute_prototypes(visual, labels, num_classes)

    audio_proto = _ema_prototypes(
        previous["audio_prototypes"], obs_audio_proto, observed,
        previous["valid"], args.z1_stats_momentum,
    )
    visual_proto = _ema_prototypes(
        previous["visual_prototypes"], obs_visual_proto, observed,
        previous["valid"], args.z1_stats_momentum,
    )
    valid = previous["valid"] | observed

    # ------------------------------------------------------------------
    # Geometry difficulty: inverse rank of normalized class margin.
    # ------------------------------------------------------------------
    audio_intra_obs, audio_inter_obs, audio_margin_obs, audio_geo_observed = _class_geometry(
        audio, labels, audio_proto, valid, num_classes, args.z1_inter_topk
    )
    visual_intra_obs, visual_inter_obs, visual_margin_obs, visual_geo_observed = _class_geometry(
        visual, labels, visual_proto, valid, num_classes, args.z1_inter_topk
    )

    # ------------------------------------------------------------------
    # Decision difficulty: inverse rank of prototype-space UCoM.
    # UCoM = true prototype logit - strongest non-target prototype logit.
    # ------------------------------------------------------------------
    audio_ucom_obs, audio_ucom_observed = _class_ucom(
        audio, labels, audio_proto, valid, args.z1_temperature, num_classes
    )
    visual_ucom_obs, visual_ucom_observed = _class_ucom(
        visual, labels, visual_proto, valid, args.z1_temperature, num_classes
    )

    observed_for_rank = (
        audio_geo_observed & visual_geo_observed
        & audio_ucom_observed & visual_ucom_observed
    )
    rank_idx = torch.where(observed_for_rank)[0]

    audio_geo_h_obs = torch.zeros(num_classes, device=device)
    visual_geo_h_obs = torch.zeros(num_classes, device=device)
    audio_ucom_h_obs = torch.zeros(num_classes, device=device)
    visual_ucom_h_obs = torch.zeros(num_classes, device=device)
    audio_h_obs = torch.zeros(num_classes, device=device)
    visual_h_obs = torch.zeros(num_classes, device=device)

    if rank_idx.numel() > 0:
        # Larger normalized margin is easier; larger UCoM is also easier.
        # Rank audio and visual class-modality pairs JOINTLY rather than ranking
        # each modality separately. Separate ranking would erase a global modality
        # gap and make H_audio - H_visual unsuitable for asymmetric routing.
        joint_margin = torch.cat([
            audio_margin_obs[rank_idx],
            visual_margin_obs[rank_idx],
        ], dim=0)
        joint_margin_h = 1.0 - percentile_rank(joint_margin)
        n_rank = rank_idx.numel()
        audio_geo_h_obs[rank_idx] = joint_margin_h[:n_rank]
        visual_geo_h_obs[rank_idx] = joint_margin_h[n_rank:]

        joint_ucom = torch.cat([
            audio_ucom_obs[rank_idx],
            visual_ucom_obs[rank_idx],
        ], dim=0)
        joint_ucom_h = 1.0 - percentile_rank(joint_ucom)
        audio_ucom_h_obs[rank_idx] = joint_ucom_h[:n_rank]
        visual_ucom_h_obs[rank_idx] = joint_ucom_h[n_rank:]

        eta = args.z1_geo_weight
        audio_h_obs[rank_idx] = (
            eta * audio_geo_h_obs[rank_idx]
            + (1.0 - eta) * audio_ucom_h_obs[rank_idx]
        )
        visual_h_obs[rank_idx] = (
            eta * visual_geo_h_obs[rank_idx]
            + (1.0 - eta) * visual_ucom_h_obs[rank_idx]
        )

    audio_q_obs, audio_q_observed = _class_knn_purity(
        audio, labels, num_classes, args.z1_knn_k
    )
    visual_q_obs, visual_q_observed = _class_knn_purity(
        visual, labels, num_classes, args.z1_knn_k
    )

    m = args.z1_stats_momentum
    prev_valid = previous["valid"]
    audio_geo_h = _ema_scalar(previous["audio_geo_hardness"], audio_geo_h_obs, observed_for_rank, prev_valid, m)
    visual_geo_h = _ema_scalar(previous["visual_geo_hardness"], visual_geo_h_obs, observed_for_rank, prev_valid, m)
    audio_ucom_h = _ema_scalar(previous["audio_ucom_hardness"], audio_ucom_h_obs, observed_for_rank, prev_valid, m)
    visual_ucom_h = _ema_scalar(previous["visual_ucom_hardness"], visual_ucom_h_obs, observed_for_rank, prev_valid, m)
    audio_h = _ema_scalar(previous["audio_hardness"], audio_h_obs, observed_for_rank, prev_valid, m)
    visual_h = _ema_scalar(previous["visual_hardness"], visual_h_obs, observed_for_rank, prev_valid, m)

    audio_intra = _ema_scalar(previous["audio_intra"], audio_intra_obs, audio_geo_observed, prev_valid, m)
    visual_intra = _ema_scalar(previous["visual_intra"], visual_intra_obs, visual_geo_observed, prev_valid, m)
    audio_inter = _ema_scalar(previous["audio_inter"], audio_inter_obs, audio_geo_observed, prev_valid, m)
    visual_inter = _ema_scalar(previous["visual_inter"], visual_inter_obs, visual_geo_observed, prev_valid, m)
    audio_margin = _ema_scalar(previous["audio_margin"], audio_margin_obs, audio_geo_observed, prev_valid, m)
    visual_margin = _ema_scalar(previous["visual_margin"], visual_margin_obs, visual_geo_observed, prev_valid, m)
    audio_ucom = _ema_scalar(previous["audio_ucom"], audio_ucom_obs, audio_ucom_observed, prev_valid, m)
    visual_ucom = _ema_scalar(previous["visual_ucom"], visual_ucom_obs, visual_ucom_observed, prev_valid, m)
    audio_q = _ema_scalar(previous["audio_purity"], audio_q_obs, audio_q_observed, prev_valid, m)
    visual_q = _ema_scalar(previous["visual_purity"], visual_q_obs, visual_q_observed, prev_valid, m)

    updated = {
        "audio_prototypes": audio_proto.detach().cpu(),
        "visual_prototypes": visual_proto.detach().cpu(),
        "audio_geo_hardness": audio_geo_h.detach().cpu().clamp(0.0, 1.0),
        "visual_geo_hardness": visual_geo_h.detach().cpu().clamp(0.0, 1.0),
        "audio_ucom_hardness": audio_ucom_h.detach().cpu().clamp(0.0, 1.0),
        "visual_ucom_hardness": visual_ucom_h.detach().cpu().clamp(0.0, 1.0),
        "audio_hardness": audio_h.detach().cpu().clamp(0.0, 1.0),
        "visual_hardness": visual_h.detach().cpu().clamp(0.0, 1.0),
        "audio_intra": audio_intra.detach().cpu(),
        "visual_intra": visual_intra.detach().cpu(),
        "audio_inter": audio_inter.detach().cpu(),
        "visual_inter": visual_inter.detach().cpu(),
        "audio_margin": audio_margin.detach().cpu(),
        "visual_margin": visual_margin.detach().cpu(),
        "audio_ucom": audio_ucom.detach().cpu(),
        "visual_ucom": visual_ucom.detach().cpu(),
        "audio_purity": audio_q.detach().cpu().clamp(0.0, 1.0),
        "visual_purity": visual_q.detach().cpu().clamp(0.0, 1.0),
        "valid": valid.detach().cpu(),
        "support": support.detach().cpu(),
    }
    return updated


def append_z1_statistics_csv(args, state, step, epoch, id_to_category):
    path = './save/metrics/{}/z1_dynamic_statistics.csv'.format(args.dataset)
    header = [
        "step", "epoch", "class_id", "category_name", "support",
        "audio_intra", "audio_inter", "audio_margin", "audio_ucom",
        "audio_geo_hardness", "audio_ucom_hardness", "audio_hardness", "audio_purity",
        "visual_intra", "visual_inter", "visual_margin", "visual_ucom",
        "visual_geo_hardness", "visual_ucom_hardness", "visual_hardness", "visual_purity",
        "audio_self_weight", "visual_self_weight", "omega_audio_to_visual",
        "omega_visual_to_audio", "valid",
    ]
    rows = []
    num_classes = (step + 1) * args.class_num_per_step
    for c in range(num_classes):
        h_a = float(state["audio_hardness"][c])
        h_v = float(state["visual_hardness"][c])
        q_a = float(state["audio_purity"][c])
        q_v = float(state["visual_purity"][c])
        omega_a2v = (1.0 - h_a) * q_a * max(h_v - h_a - args.z1_dead_zone, 0.0)
        omega_v2a = (1.0 - h_v) * q_v * max(h_a - h_v - args.z1_dead_zone, 0.0)
        rows.append({
            "step": step,
            "epoch": epoch,
            "class_id": c,
            "category_name": id_to_category.get(c, "class_{}".format(c)),
            "support": int(state["support"][c]),
            "audio_intra": float(state["audio_intra"][c]),
            "audio_inter": float(state["audio_inter"][c]),
            "audio_margin": float(state["audio_margin"][c]),
            "audio_ucom": float(state["audio_ucom"][c]),
            "audio_geo_hardness": float(state["audio_geo_hardness"][c]),
            "audio_ucom_hardness": float(state["audio_ucom_hardness"][c]),
            "audio_hardness": h_a,
            "audio_purity": q_a,
            "visual_intra": float(state["visual_intra"][c]),
            "visual_inter": float(state["visual_inter"][c]),
            "visual_margin": float(state["visual_margin"][c]),
            "visual_ucom": float(state["visual_ucom"][c]),
            "visual_geo_hardness": float(state["visual_geo_hardness"][c]),
            "visual_ucom_hardness": float(state["visual_ucom_hardness"][c]),
            "visual_hardness": h_v,
            "visual_purity": q_v,
            "audio_self_weight": h_a * q_a,
            "visual_self_weight": h_v * q_v,
            "omega_audio_to_visual": omega_a2v,
            "omega_visual_to_audio": omega_v2a,
            "valid": int(bool(state["valid"][c])),
        })
    append_csv_rows(path, header, rows)


# ============================================================================
# New z1 losses
# ============================================================================

def _prototype_probabilities(features, prototypes, valid, temperature):
    logits = torch.mm(features, prototypes.t()) / temperature
    logits[:, ~valid] = -1e4
    return F.softmax(logits, dim=1), F.log_softmax(logits, dim=1)


def _weighted_mean(values, weights):
    denom = weights.sum()
    if float(denom.detach().item()) <= EPS:
        return values.sum() * 0.0
    return (values * weights).sum() / denom.clamp_min(EPS)


def z1_dynamic_losses(audio_feature, visual_feature, labels, state_device, args):
    """
    Action 1: hybrid-difficulty + purity gated modality-specific self-repair.
    Action 2: hybrid-difficulty + teacher-purity gated asymmetric correction.

    Both actions operate in modality-specific normalized z1 spaces. No z2 feature,
    z2 difficulty statistic, or z2 loss is used in this controlled experiment.
    """
    audio_feature = F.normalize(audio_feature, dim=1)
    visual_feature = F.normalize(visual_feature, dim=1)

    p_a, log_p_a = _prototype_probabilities(
        audio_feature,
        state_device["audio_prototypes"],
        state_device["valid"],
        args.z1_temperature,
    )
    p_v, log_p_v = _prototype_probabilities(
        visual_feature,
        state_device["visual_prototypes"],
        state_device["valid"],
        args.z1_temperature,
    )

    idx = torch.arange(labels.shape[0], device=labels.device)
    nll_a = -log_p_a[idx, labels]
    nll_v = -log_p_v[idx, labels]

    h_a = state_device["audio_hardness"][labels]
    h_v = state_device["visual_hardness"][labels]
    q_a = state_device["audio_purity"][labels]
    q_v = state_device["visual_purity"][labels]
    sample_valid = state_device["valid"][labels].float()

    # Action 1: only hard-and-structured classes receive strong own-space repair.
    self_weight_a = h_a * q_a * sample_valid
    self_weight_v = h_v * q_v * sample_valid
    self_values = torch.cat([nll_a, nll_v], dim=0)
    self_weights = torch.cat([self_weight_a, self_weight_v], dim=0)
    loss_self = _weighted_mean(self_values, self_weights)

    # Action 2 direction: teacher must be easy and pure. Student purity is not used.
    omega_a2v = (
        (1.0 - h_a) * q_a
        * F.relu(h_v - h_a - args.z1_dead_zone)
        * sample_valid
    )
    omega_v2a = (
        (1.0 - h_v) * q_v
        * F.relu(h_a - h_v - args.z1_dead_zone)
        * sample_valid
    )

    target_mask = F.one_hot(labels, num_classes=p_a.shape[1]).bool()

    def negative_correction(student_p, teacher_p):
        # Select error classes with detached probabilities so the student cannot
        # reduce the loss merely by changing the weighting distribution.
        excess = F.relu(student_p.detach() - teacher_p.detach())
        excess = excess.masked_fill(target_mask, 0.0)
        excess_sum = excess.sum(dim=1, keepdim=True)
        correction_weight = excess / excess_sum.clamp_min(EPS)
        per_sample = -(
            correction_weight
            * torch.log((1.0 - student_p).clamp_min(EPS))
        ).sum(dim=1)
        return torch.where(excess_sum.squeeze(1) > EPS, per_sample, torch.zeros_like(per_sample))

    loss_a2v_per_sample = negative_correction(p_v, p_a)
    loss_v2a_per_sample = negative_correction(p_a, p_v)
    asym_values = torch.cat([loss_a2v_per_sample, loss_v2a_per_sample], dim=0)
    asym_weights = torch.cat([omega_a2v, omega_v2a], dim=0)
    loss_asym = _weighted_mean(asym_values, asym_weights)

    diagnostics = {
        "self_weight_mean": float(self_weights.detach().mean().item()),
        "omega_a2v_mean": float(omega_a2v.detach().mean().item()),
        "omega_v2a_mean": float(omega_v2a.detach().mean().item()),
    }
    return loss_self, loss_asym, diagnostics


def train(args, step, train_data_set, val_data_set, exemplar_set, id_to_category):
    T = 2

    train_loader = DataLoader(
        train_data_set,
        batch_size=min(args.train_batch_size, train_data_set.__len__()),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data_set,
        batch_size=min(args.infer_batch_size, val_data_set.__len__()),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    stats_train_loader = DataLoader(
        train_data_set,
        batch_size=min(args.infer_batch_size, train_data_set.__len__()),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    step_out_class_num = (step + 1) * args.class_num_per_step
    exemplar_loader = None
    stats_exemplar_loader = None
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
    else:
        model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step - 1))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step - 1))

        exemplar_loader = DataLoader(
            exemplar_set,
            batch_size=min(args.exemplar_batch_size, exemplar_set.__len__()),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        stats_exemplar_loader = DataLoader(
            exemplar_set,
            batch_size=min(args.infer_batch_size, exemplar_set.__len__()),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
        last_step_out_class_num = step * args.class_num_per_step

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)

    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        old_model.eval()

    z1_state = load_previous_z1_state(args, step, step_out_class_num)
    z1_state_device = state_to_device(z1_state, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0

    for epoch in range(args.max_epoches):
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "logit_kd": 0.0,
            "attn_kd": 0.0,
            "z1_self": 0.0,
            "z1_asym": 0.0,
        }
        num_steps = 0
        model.train()
        if step == 0:
            iterator = tqdm(train_loader)
        else:
            iterator = tzip(train_loader, cycle(exemplar_loader))

        for samples in iterator:
            zero = torch.zeros((), device=device)
            loss_CE = zero
            loss_KD = zero
            attn_dist_loss = zero
            loss_self = zero
            loss_asym = zero

            if step == 0:
                data, labels = samples
                labels = labels.to(device)
                visual = data[0].to(device)
                audio = data[1].to(device)
                outputs = model(
                    visual=visual,
                    audio=audio,
                    return_dict=True,
                    out_analysis_features=True,
                )
                out = outputs["logits"]
                loss_CE = CE_loss(step_out_class_num, out, labels)
                all_labels = labels
                audio_feature = outputs["z1_audio_norm"]
                visual_feature = outputs["z1_visual_norm"]
                loss = loss_CE
            else:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(device)
                labels_local = (labels % args.class_num_per_step).to(device)

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device)

                data_batch_size = labels_local.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                total_visual = torch.cat((data[0], exemplar_data[0])).to(device)
                total_audio = torch.cat((data[1], exemplar_data[1])).to(device)
                all_labels = torch.cat((labels, exemplar_labels))

                outputs = model(
                    visual=total_visual,
                    audio=total_audio,
                    return_dict=True,
                    out_analysis_features=True,
                    out_attn_score=True,
                )
                out = outputs["logits"]
                audio_feature = outputs["z1_audio_norm"]
                visual_feature = outputs["z1_visual_norm"]
                spatial_attn_score = outputs["spatial_attn_score"]
                temporal_attn_score = outputs["temporal_attn_score"]

                with torch.no_grad():
                    old_outputs = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        return_dict=True,
                        out_attn_score=True,
                    )
                    old_out = old_outputs["logits"].detach()
                    old_spatial_attn_score = old_outputs["spatial_attn_score"].detach()
                    old_temporal_attn_score = old_outputs["temporal_attn_score"].detach()

                if args.attn_score_distil:
                    new_spatial = spatial_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(2, 3)
                    new_spatial = new_spatial.reshape(-1, new_spatial.shape[-1])
                    old_spatial = old_spatial_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(2, 3)
                    old_spatial = old_spatial.reshape(-1, old_spatial.shape[-1])

                    new_temporal = temporal_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(1, 2)
                    new_temporal = new_temporal.reshape(-1, new_temporal.shape[-1])
                    old_temporal = old_temporal_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(1, 2)
                    old_temporal = old_temporal.reshape(-1, old_temporal.shape[-1])

                    spatial_attn_dist_loss = F.kl_div(
                        new_spatial.clamp_min(EPS).log(), old_spatial,
                        reduction='sum'
                    ) / exemplar_data_batch_size
                    temporal_attn_dist_loss = F.kl_div(
                        new_temporal.clamp_min(EPS).log(), old_temporal,
                        reduction='sum'
                    ) / exemplar_data_batch_size
                    attn_dist_loss = (
                        args.lam * spatial_attn_dist_loss
                        + (1.0 - args.lam) * temporal_attn_dist_loss
                    )

                old_out = old_out[:, :last_step_out_class_num]
                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = CE_loss(args.class_num_per_step, curr_out, labels_local)
                prev_out = out[
                    data_batch_size:data_batch_size + exemplar_data_batch_size,
                    :last_step_out_class_num,
                ]
                loss_prev = CE_loss(last_step_out_class_num, prev_out, exemplar_labels)
                loss_CE = (
                    loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size
                ) / (data_batch_size + exemplar_data_batch_size)

                if args.dataset == 'AVE' and args.class_num_per_step == 4 and step == 1:
                    loss_CE = CE_loss(
                        args.class_num_per_step + last_step_out_class_num,
                        out,
                        all_labels,
                    )

                kd_terms = []
                for task_id in range(step):
                    start = task_id * args.class_num_per_step
                    end = (task_id + 1) * args.class_num_per_step
                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    kd_terms.append(
                        F.kl_div(output_log, soft_target, reduction='batchmean') * (T ** 2)
                    )
                loss_KD = torch.stack(kd_terms).sum() if kd_terms else zero
                loss = loss_CE + loss_KD + attn_dist_loss

            z1_active = (
                args.z1_dynamic_losses
                and (step > 0 or args.z1_apply_step0)
                and epoch >= args.z1_warmup_epochs
                and bool(z1_state_device["valid"][all_labels].all().item())
            )
            if z1_active:
                loss_self, loss_asym, _ = z1_dynamic_losses(
                    audio_feature,
                    visual_feature,
                    all_labels,
                    z1_state_device,
                    args,
                )
                loss = loss + args.lam_self * loss_self + args.lam_asym * loss_asym

            model.zero_grad()
            loss.backward()
            opt.step()

            totals["loss"] += float(loss.item())
            totals["ce"] += float(loss_CE.item())
            totals["logit_kd"] += float(loss_KD.item())
            totals["attn_kd"] += float(attn_dist_loss.item())
            totals["z1_self"] += float(loss_self.item())
            totals["z1_asym"] += float(loss_asym.item())
            num_steps += 1

        for key in totals:
            totals[key] /= max(num_steps, 1)
        train_loss_list.append(totals["loss"])
        print(
            'Epoch:{} train_loss:{:.5f} CE:{:.5f} logitKD:{:.5f} '
            'attnKD:{:.5f} z1Self:{:.5f} z1Asym:{:.5f}'.format(
                epoch,
                totals["loss"],
                totals["ce"],
                totals["logit_kd"],
                totals["attn_kd"],
                totals["z1_self"],
                totals["z1_asym"],
            ),
            flush=True,
        )

        # Estimate difficulty and purity with one deterministic pass over the
        # current training data and replay memory. This avoids duplicated replay
        # samples introduced by cycle(exemplar_loader).
        should_update_stats = (
            epoch == 0
            or (epoch + 1) % args.z1_stats_update_interval == 0
            or epoch == args.max_epoches - 1
        )
        if should_update_stats:
            z1_state = update_z1_state(
                args,
                model,
                [stats_train_loader, stats_exemplar_loader],
                z1_state,
                step,
                epoch,
            )
            z1_state_device = state_to_device(z1_state, device)
            append_z1_statistics_csv(args, z1_state, step, epoch, id_to_category)
            print(
                'Updated z1 statistics at step {}, epoch {}: valid={}/{}'.format(
                    step,
                    epoch,
                    int(z1_state["valid"][:step_out_class_num].sum().item()),
                    step_out_class_num,
                ),
                flush=True,
            )

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0].to(device)
                val_audio = val_data[1].to(device)
                val_out_logits = model(visual=val_visual, audio=val_audio)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print('Epoch:{} val_res:{:.6f} '.format(epoch, val_top1), flush=True)

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print('Saving best model and z1 statistics at Epoch {}'.format(epoch), flush=True)
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                model_to_save,
                './save/{}/step_{}_best_model.pkl'.format(args.dataset, step),
            )
            torch.save(
                z1_state,
                './save/{}/step_{}_best_z1_stats.pt'.format(args.dataset, step),
            )

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/train_loss_step_{}.png'.format(args.dataset, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/val_acc_step_{}.png'.format(args.dataset, step))
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, opt, epoch)


def detailed_test(args, step, test_data_set, task_best_acc_list,
                  metrics_root: str,
                  metrics_state: dict,
                  id_to_category: dict):
    """
    metrics_state:
      {
        "best_recall": { "0": 0.83, "1": 0.55, ... },
        "first_seen_step": { "0": 0, "1": 2, ... }
      }
    """
    print("=====================================")
    print("Start testing...")
    print("=====================================")

    model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)

    # ---- 收集 logits 与 labels（保持 dtype 正确）----
    all_logits_list = []
    all_labels_list = []

    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            test_visual = test_data[0].to(device)
            test_audio = test_data[1].to(device)

            logits = model(visual=test_visual, audio=test_audio)   # raw logits
            all_logits_list.append(logits.detach().cpu())
            all_labels_list.append(test_labels.detach().cpu().long())

    all_test_logits = torch.cat(all_logits_list, dim=0)            # (N, C_seen)
    all_test_labels = torch.cat(all_labels_list, dim=0).long()     # (N,)

    # ---- overall top-1 acc ----
    pred = all_test_logits.argmax(dim=1).long()
    overall_acc = (pred == all_test_labels).float().mean().item()
    print("Incremental step {} Testing res (overall acc): {:.6f}".format(step, overall_acc))

    # ---- per-task acc（与你原逻辑等价，但更快更稳）----
    K = args.class_num_per_step
    num_seen_classes = (step + 1) * K

    old_task_acc_list = []
    current_step_acc = None
    for i in range(step + 1):
        lo = i * K
        hi = (i + 1) * K
        mask = (all_test_labels >= lo) & (all_test_labels < hi)
        if mask.sum().item() == 0:
            i_acc = 0.0
        else:
            i_acc = (pred[mask] == all_test_labels[mask]).float().mean().item()

        if i == step:
            current_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)

    # ---- task-level forgetting（你的原定义：best_old_task - current_old_task）----
    if step > 0:
        forgetting = float(np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list)))
        print('task-level forgetting: {:.6f}'.format(forgetting))
        # 更新旧任务历史最好
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None

    # append 当前任务的“历史最好”（初始就是当前）
    task_best_acc_list.append(current_step_acc)

    # ======================================================================
    # Per-class metrics (CIL classic: evaluate only on seen classes)
    # ======================================================================
    stats = compute_per_class_prf(all_test_labels, pred, num_seen_classes)

    # state dicts
    best_f1 = metrics_state.get("best_f1", {})
    first_seen = metrics_state.get("first_seen_step", {})

    rows = []
    for c in range(num_seen_classes):
        c_str = str(c)

        # record first seen step (对 CIL shuffle 很有用)
        if c_str not in first_seen:
            first_seen[c_str] = step

        support_c = int(stats["support"][c])
        tp_c = int(stats["tp"][c])
        fp_c = int(stats["fp"][c])
        fn_c = int(stats["fn"][c])

        precision_c = float(stats["precision"][c])
        recall_c = float(stats["recall"][c])
        f1_c = float(stats["f1"][c])

        # per-class forgetting（基于 F1）
        # 约定：forget = best_before - current；若没有 best_before，则 forget=0
        best_before = float(best_f1[c_str]) if c_str in best_f1 else None
        forget_f1 = (best_before - f1_c) if best_before is not None else 0.0

        # update best_f1
        new_best = f1_c if best_before is None else max(best_before, f1_c)
        best_f1[c_str] = new_best

        forgetting_value = float(forgetting) if forgetting is not None else 0.0
        rows.append({
            "step": step,
            "class_id": c,
            "category_name": id_to_category.get(c, f"class_{c}"),
            "first_seen_step": int(first_seen[c_str]),
            "support": support_c,
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "best_f1": float(new_best),
            "forget_f1": float(forget_f1),  # 基于F1 的 forgetting
            "forgetting": forgetting_value,    # 基于 task-level acc 的 forgetting
            "overall_acc": float(overall_acc),          # 冗余存一下，做 step-level plot 更方便
        })

    header = [
        "step", "class_id", "category_name", "first_seen_step", "support",
        "tp", "fp", "fn", "precision", "recall", "f1",
        "best_f1", "forget_f1", "forgetting", "overall_acc"
    ]
    per_class_csv = os.path.join(metrics_root, "per_class_metrics.csv")
    append_csv_rows(per_class_csv, header, rows)

    # update state + dump
    metrics_state["best_f1"] = best_f1
    metrics_state["first_seen_step"] = first_seen
    save_json(metrics_state, os.path.join(metrics_root, "per_class_state.json"))

    return forgetting




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def dataset_type(s: str):
        if s in ['AVE', 'ksounds']:
            return s
        if 'VGGSound' in s:
            return s
        raise argparse.ArgumentTypeError(
            "dataset must be 'AVE', 'ksounds', or contain 'VGGSound'"
        )

    parser.add_argument('--dataset', type=dataset_type, default='AVE')
    parser.add_argument('--modality', type=str, default='audio-visual', choices=['audio-visual'])
    parser.add_argument('--feature_root', type=str, default="/mnt/data2/wpian/dataset/VGGSound")
    parser.add_argument('--meta_root', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=32)
    parser.add_argument('--exemplar_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=boolean_string, default=False)
    parser.add_argument('--milestones', type=int, default=[500], nargs='+')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Spatial/temporal attention-distillation interpolation.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class_num_per_step', type=int, default=7)
    parser.add_argument('--memory_size', type=int, default=340)
    parser.add_argument('--attn_score_distil', action='store_true', default=False)

    # New dynamic z1 replacement losses. The original instance/class contrastive
    # flags are intentionally absent from the proposed commands.
    parser.add_argument('--z1_dynamic_losses', action='store_true', default=False)
    parser.add_argument('--z1_apply_step0', action='store_true', default=False,
                        help='Off by default to match the supplied AVCIL code, where z1 losses start after step 0.')
    parser.add_argument('--lam_self', type=float, default=0.1)
    parser.add_argument('--lam_asym', type=float, default=1.0)
    parser.add_argument('--z1_temperature', type=float, default=0.1)
    parser.add_argument('--z1_dead_zone', type=float, default=0.1)
    parser.add_argument('--z1_knn_k', type=int, default=10)
    parser.add_argument('--z1_inter_topk', type=int, default=5,
                        help='Nearest prototype centroids used by normalized margin.')
    parser.add_argument('--z1_geo_weight', type=float, default=0.5,
                        help='eta in H = eta * H_geo + (1-eta) * H_ucom.')
    parser.add_argument('--z1_warmup_epochs', type=int, default=5)
    parser.add_argument('--z1_stats_update_interval', type=int, default=1)
    parser.add_argument('--z1_stats_momentum', type=float, default=0.9)
    parser.add_argument('--z1_stats_max_per_class', type=int, default=50)

    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--dump_tsne', action='store_true')
    parser.add_argument('--tsne_feature', type=str, default='logits',
                        choices=['audio', 'visual', 'joint_mean', 'joint_concat', 'logits'])
    parser.add_argument('--tsne_max_points_per_class', type=int, default=50)
    parser.add_argument('--tsne_out_root', type=str, default='./save/tsne')

    args = parser.parse_args()
    if args.z1_stats_update_interval < 1:
        raise ValueError('--z1_stats_update_interval must be at least 1')
    if not 0.0 <= args.z1_stats_momentum < 1.0:
        raise ValueError('--z1_stats_momentum must be in [0, 1)')
    if args.z1_knn_k < 1:
        raise ValueError('--z1_knn_k must be at least 1')
    if args.z1_inter_topk < 1:
        raise ValueError('--z1_inter_topk must be at least 1')
    if not 0.0 <= args.z1_geo_weight <= 1.0:
        raise ValueError('--z1_geo_weight must be in [0, 1]')
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step
    setup_seed(args.seed)
    print('Training start time: {}'.format(datetime.now()))

    train_set = IcaAVELoader(args=args, mode='train', modality=args.modality)
    val_set = IcaAVELoader(args=args, mode='val', modality=args.modality)
    test_set = IcaAVELoader(args=args, mode='test', modality=args.modality)
    exemplar_set = exemplarLoader(args=args, modality=args.modality)

    category_encode_dict = train_set.category_encode_dict
    id_to_category = {v: k for k, v in category_encode_dict.items()}

    ckpts_root = './save/{}/'.format(args.dataset)
    figs_root = './save/fig/{}/'.format(args.dataset)
    metrics_root = './save/metrics/{}/'.format(args.dataset)
    os.makedirs(ckpts_root, exist_ok=True)
    os.makedirs(figs_root, exist_ok=True)
    os.makedirs(metrics_root, exist_ok=True)

    for filename in ['per_class_metrics.csv', 'z1_dynamic_statistics.csv']:
        path = os.path.join(metrics_root, filename)
        if os.path.exists(path):
            os.remove(path)

    metrics_state = {
        'best_f1': {},
        'first_seen_step': {},
    }
    save_json(metrics_state, os.path.join(metrics_root, 'per_class_state.json'))

    task_best_acc_list = []
    step_forgetting_list = []

    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)
        exemplar_set._set_incremental_step_(step)

        print('Incremental step: {}'.format(step))
        if not args.test_only:
            train(
                args,
                step,
                train_set,
                val_set,
                exemplar_set,
                id_to_category,
            )

        step_forgetting = detailed_test(
            args=args,
            step=step,
            test_data_set=test_set,
            task_best_acc_list=task_best_acc_list,
            metrics_root=metrics_root,
            metrics_state=metrics_state,
            id_to_category=id_to_category,
        )
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)

        ckpt_path = './save/{}/step_{}_best_model.pkl'.format(args.dataset, step)
        if args.dump_tsne:
            print('Dumping t-SNE plots for step {}...'.format(step))
            out_root = os.path.join(args.tsne_out_root, args.dataset)
            make_tsne_plots_for_step(
                args=args,
                step=step,
                test_set=test_set,
                ckpt_path=ckpt_path,
                out_root=out_root,
                feature_type=args.tsne_feature,
                max_points_per_class=args.tsne_max_points_per_class,
            )

    mean_forgetting = np.mean(step_forgetting_list) if step_forgetting_list else 0.0
    print('Average Forgetting: {:.6f}'.format(mean_forgetting))

    if args.dataset != 'AVE':
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()
        exemplar_set.close_visual_features_h5()