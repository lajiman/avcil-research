"""
Train AV-CIL with class-conditional geometric guidance contribution and exact linear logit routing.

First-version experimental contract
-----------------------------------
1. Task 0 is trained exactly as the user's baseline (CE only in the supplied
   code path); the gate is disabled.
2. For task t >= geo_routing_start_step, the original AVCIL losses remain:
   replay CE, old-logit KD, z1 instance/class contrastive losses, and attention
   score distillation.
3. After a short per-task warm-up, detached class prototypes are constructed
   from z1 uniform-visual features of current training data plus old exemplars.
4. The linear classifier uses sample-and-class-specific routing between guided and
   uniform visual logit contributions; z1 contrastive learning and attention
   distillation retain the original full-strength guided branch.
5. No new gate/relation CL loss is added in this first complete experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from itertools import cycle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

from dataloader_ours import IcaAVELoader, exemplarLoader
from model.audio_visual_model_incremental_geometric_linear_routing import IncreAudioVisualNet
from tsne_plotter import make_tsne_plots_for_step


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Reproducibility and baseline losses
# ============================================================================

def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def boolean_string(value: str) -> bool:
    if value not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return value == "True"


def CE_loss(num_classes: int, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    targets = F.one_hot(label, num_classes=num_classes)
    return -torch.mean(
        torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1)
    )


def cal_contrastive_loss(
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    num_sample = score.shape[0]
    label = torch.arange(num_sample, device=score.device)
    return CE_loss(num_sample, score, label)


def class_contrastive_loss(
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    label: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    class_matrix = label.unsqueeze(0).repeat(label.shape[0], 1)
    class_matrix = (class_matrix == label.unsqueeze(-1)).float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature

    # Preserve the exact averaging convention in the supplied baseline code.
    return -torch.mean(
        torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)
    )


def top_1_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target.long()).float().mean().item()


def adjust_learning_rate(args, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    milestones = np.array(args.milestones) - 1
    if epoch in milestones:
        current_lr = optimizer.param_groups[0]["lr"]
        new_lr = current_lr * 0.1
        print("Reduce lr from {} to {}".format(current_lr, new_lr), flush=True)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


# ============================================================================
# I/O helpers
# ============================================================================

def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def save_json(obj: dict, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2)


def load_json(path: str, default: dict) -> dict:
    if os.path.exists(path):
        with open(path, "r") as handle:
            return json.load(handle)
    return default


def append_csv_rows(csv_path: str, header: Sequence[str], rows: Sequence[dict]) -> None:
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def torch_load_full(path: str, map_location="cpu"):
    """Load whole-model checkpoints across older/newer PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_model(model: nn.Module) -> IncreAudioVisualNet:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_whole_model(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(unwrap_model(model), path)


# ============================================================================
# Prototype construction
# ============================================================================

def _make_prototype_loader(args, dataset) -> Optional[DataLoader]:
    length = len(dataset)
    if length <= 0:
        return None
    batch_size = min(args.infer_batch_size, length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )


@torch.no_grad()
def refresh_geometric_prototypes(
    args,
    model: nn.Module,
    step: int,
    train_data_set,
    exemplar_set,
    reason: str,
) -> Dict[str, object]:
    """Rebuild detached z1 uniform-visual prototypes for all seen classes.

    Current classes are represented by the current task's full training set;
    old classes are represented by the exemplar set.  Missing classes fall back
    to the previous checkpoint's stored prototype, if available.
    """
    core = unwrap_model(model)
    num_seen_classes = (step + 1) * args.class_num_per_step
    if core.num_classes != num_seen_classes:
        raise RuntimeError(
            "Prototype refresh sees {} classes but model has {}".format(
                num_seen_classes, core.num_classes
            )
        )

    loaders: List[Tuple[str, DataLoader]] = []
    train_loader = _make_prototype_loader(args, train_data_set)
    if train_loader is not None:
        loaders.append(("current_train", train_loader))
    if step > 0:
        exemplar_loader = _make_prototype_loader(args, exemplar_set)
        if exemplar_loader is not None:
            loaders.append(("old_exemplars", exemplar_loader))

    if not loaders:
        raise RuntimeError("No data are available for prototype construction")

    was_training = model.training
    model.eval()

    sums = torch.zeros(num_seen_classes, 768, device=device, dtype=torch.float32)
    counts = torch.zeros(num_seen_classes, device=device, dtype=torch.float32)

    for loader_name, loader in loaders:
        print(
            "Refreshing geometric prototypes from {} ({})...".format(
                loader_name, reason
            ),
            flush=True,
        )
        for data, labels in tqdm(loader, leave=False):
            visual = data[0].to(device, non_blocking=True)
            audio = data[1].to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            # Prototype construction only needs the uniform visual endpoint;
            # avoid running the expensive audio-guided attention branch.
            visual_uniform = F.normalize(
                core.extract_uniform_visual_feature(visual),
                dim=1,
                eps=1e-12,
            )
            sums.index_add_(0, labels, visual_uniform.float())
            counts.index_add_(
                0,
                labels,
                torch.ones_like(labels, dtype=torch.float32),
            )

    prototypes = torch.zeros_like(sums)
    valid = counts > 0
    if valid.any():
        prototypes[valid] = F.normalize(
            sums[valid] / counts[valid].unsqueeze(1),
            dim=1,
            eps=1e-12,
        )

    # Robust fallback: a class can remain usable if the exemplar loader happens
    # not to expose it but a previous valid prototype is stored in the checkpoint.
    old_valid = core.geo_prototype_valid[:num_seen_classes].to(device)
    fallback = (~valid) & old_valid
    if fallback.any():
        prototypes[fallback] = core.geo_visual_prototypes[:num_seen_classes][
            fallback
        ].to(device)
        counts[fallback] = core.geo_prototype_counts[:num_seen_classes][fallback].to(
            device
        )
        valid[fallback] = True

    core.set_geometric_prototypes(
        prototypes=prototypes,
        valid_mask=valid,
        counts=counts,
    )

    missing = torch.where(~valid)[0].detach().cpu().tolist()
    stats = {
        "num_seen_classes": num_seen_classes,
        "num_valid_classes": int(valid.sum().item()),
        "missing_classes": missing,
        "min_count": float(counts[valid].min().item()) if valid.any() else 0.0,
        "max_count": float(counts[valid].max().item()) if valid.any() else 0.0,
        "mean_count": float(counts[valid].mean().item()) if valid.any() else 0.0,
    }
    print("Prototype refresh summary: {}".format(stats), flush=True)

    if was_training:
        model.train()
    else:
        model.eval()
    return stats


# ============================================================================
# Diagnostics
# ============================================================================

def target_vs_rest_benefit(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """B_y = logit_y - logsumexp_{k != y}(logit_k)."""
    labels = labels.long()
    target = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    mask = F.one_hot(labels, num_classes=logits.shape[1]).bool()
    others = logits.masked_fill(mask, float("-inf"))
    return target - torch.logsumexp(others, dim=1)


def approximate_rank(values: np.ndarray) -> np.ndarray:
    """Simple deterministic ranks; ties are rare for these continuous scores."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = approximate_rank(x)
    ry = approximate_rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    value = np.corrcoef(rx, ry)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def compute_per_class_prf(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> Dict[str, np.ndarray]:
    y_true = y_true.long()
    y_pred = y_pred.long()

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for class_id in range(num_classes):
        true_c = y_true == class_id
        pred_c = y_pred == class_id
        tp[class_id] = (true_c & pred_c).sum()
        fp[class_id] = ((~true_c) & pred_c).sum()
        fn[class_id] = (true_c & (~pred_c)).sum()

    support = tp + fn
    precision = torch.zeros(num_classes, dtype=torch.float32)
    recall = torch.zeros(num_classes, dtype=torch.float32)
    f1 = torch.zeros(num_classes, dtype=torch.float32)

    for class_id in range(num_classes):
        tp_c = tp[class_id].item()
        fp_c = fp[class_id].item()
        fn_c = fn[class_id].item()
        p = safe_div(tp_c, tp_c + fp_c)
        r = safe_div(tp_c, tp_c + fn_c)
        precision[class_id] = p
        recall[class_id] = r
        f1[class_id] = safe_div(2 * p * r, p + r)

    return {
        "tp": tp.numpy(),
        "fp": fp.numpy(),
        "fn": fn.numpy(),
        "support": support.numpy(),
        "precision": precision.numpy(),
        "recall": recall.numpy(),
        "f1": f1.numpy(),
    }


# ============================================================================
# Training
# ============================================================================

def train(args, step, train_data_set, val_data_set, exemplar_set) -> None:
    distill_temperature = args.distillation_temperature

    train_loader = DataLoader(
        train_data_set,
        batch_size=min(args.train_batch_size, len(train_data_set)),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data_set,
        batch_size=min(args.infer_batch_size, len(val_data_set)),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    step_out_class_num = (step + 1) * args.class_num_per_step
    checkpoint_dir = os.path.join("./save", args.dataset)
    checkpoint_path = os.path.join(
        checkpoint_dir, "step_{}_best_model.pkl".format(step)
    )

    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
        old_model = None
        exemplar_loader = None
        last_step_out_class_num = 0
    else:
        previous_path = os.path.join(
            checkpoint_dir, "step_{}_best_model.pkl".format(step - 1)
        )
        model = torch_load_full(previous_path, map_location="cpu")
        model.incremental_classifier(step_out_class_num)
        old_model = torch_load_full(previous_path, map_location="cpu")

        exemplar_loader = DataLoader(
            exemplar_set,
            batch_size=min(args.exemplar_batch_size, len(exemplar_set)),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        last_step_out_class_num = step * args.class_num_per_step

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if old_model is not None:
            old_model = nn.DataParallel(old_model)

    model = model.to(device)
    unwrap_model(model).set_geometric_routing_enabled(False)

    if old_model is not None:
        old_model = old_model.to(device)
        old_model.eval()
        for parameter in old_model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    routing_required = bool(
        args.geometric_linear_routing and step >= args.geo_routing_start_step
    )
    if routing_required and args.geo_routing_warmup_epochs >= args.max_epoches:
        raise ValueError(
            "geo_routing_warmup_epochs must be smaller than max_epoches "
            "for routed steps"
        )

    train_loss_list: List[float] = []
    val_acc_list: List[float] = []
    best_val_res = float("-inf")
    best_saved = False
    routing_has_started = False

    metrics_root = os.path.join("./save/metrics", args.dataset)
    epoch_csv = os.path.join(metrics_root, "routing_epoch_metrics.csv")
    epoch_header = [
        "step",
        "epoch",
        "routing_active",
        "prototype_valid_classes",
        "train_loss",
        "val_acc",
        "train_gate_all_mean",
        "train_gate_all_suppressed_frac",
        "train_gate_true_mean",
        "train_gate_true_suppressed_frac",
        "train_true_geo_mean",
        "val_gate_all_mean",
        "val_gate_all_suppressed_frac",
        "val_gate_true_mean",
        "val_gate_true_suppressed_frac",
        "val_true_geo_mean",
    ]

    for epoch in range(args.max_epoches):
        # Prototypes are piecewise fixed, but g_{i,c} is recomputed on every
        # forward from the current v_guided/v_uniform features.
        if routing_required:
            if epoch < args.geo_routing_warmup_epochs:
                unwrap_model(model).set_geometric_routing_enabled(False)
            else:
                refresh_due = (
                    epoch == args.geo_routing_warmup_epochs
                    or (
                        args.geo_prototype_refresh_epochs > 0
                        and epoch > args.geo_routing_warmup_epochs
                        and (
                            epoch - args.geo_routing_warmup_epochs
                        ) % args.geo_prototype_refresh_epochs
                        == 0
                    )
                )
                if refresh_due:
                    refresh_geometric_prototypes(
                        args=args,
                        model=model,
                        step=step,
                        train_data_set=train_data_set,
                        exemplar_set=exemplar_set,
                        reason="step{}_epoch{}".format(step, epoch),
                    )
                unwrap_model(model).set_geometric_routing_enabled(True)

                if not routing_has_started:
                    routing_has_started = True
                    # Warm-up checkpoints are not eligible for the method's best
                    # checkpoint; reset selection when routing first activates.
                    best_val_res = float("-inf")
                    best_saved = False
                    print(
                        "Geometric linear routing activated at step {}, epoch {}".format(
                            step, epoch
                        ),
                        flush=True,
                    )
        else:
            unwrap_model(model).set_geometric_routing_enabled(False)

        model.train()
        train_loss = 0.0
        num_steps = 0

        train_gate_all_sum = 0.0
        train_gate_all_count = 0
        train_gate_all_suppressed = 0
        train_gate_true_sum = 0.0
        train_gate_true_count = 0
        train_gate_true_suppressed = 0
        train_true_geo_sum = 0.0

        if step == 0:
            iterator = tqdm(train_loader)
        else:
            assert exemplar_loader is not None
            iterator = tzip(train_loader, cycle(exemplar_loader))

        for samples in iterator:
            if step == 0:
                data, labels = samples
                labels = labels.to(device, non_blocking=True).long()
                visual = data[0].to(device, non_blocking=True)
                audio = data[1].to(device, non_blocking=True)

                outputs = model(
                    visual=visual,
                    audio=audio,
                    out_feature_before_fusion=True,
                    return_dict=True,
                    out_gate_details=True,
                )
                logits = outputs["logits"]
                routing_labels = labels
                loss = CE_loss(step_out_class_num, logits, labels)

            else:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(device, non_blocking=True).long()
                labels_local = labels % args.class_num_per_step

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(
                    device, non_blocking=True
                ).long()

                data_batch_size = labels.shape[0]
                exemplar_batch_size = exemplar_labels.shape[0]
                routing_labels = torch.cat((labels, exemplar_labels), dim=0)

                total_visual = torch.cat((data[0], exemplar_data[0]), dim=0).to(
                    device, non_blocking=True
                )
                total_audio = torch.cat((data[1], exemplar_data[1]), dim=0).to(
                    device, non_blocking=True
                )

                outputs = model(
                    visual=total_visual,
                    audio=total_audio,
                    out_feature_before_fusion=True,
                    out_attn_score=True,
                    return_dict=True,
                    out_gate_details=True,
                )
                logits = outputs["logits"]
                audio_feature = outputs["audio_feature"]
                visual_feature = outputs["visual_feature"]
                spatial_attn_score = outputs["spatial_attn_score"]
                temporal_attn_score = outputs["temporal_attn_score"]

                assert old_model is not None
                with torch.no_grad():
                    old_outputs = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        out_attn_score=True,
                        return_dict=True,
                    )
                    old_out = old_outputs["logits"].detach()
                    old_spatial_attn_score = old_outputs[
                        "spatial_attn_score"
                    ].detach()
                    old_temporal_attn_score = old_outputs[
                        "temporal_attn_score"
                    ].detach()

                if args.instance_contrastive:
                    instance_contra_loss = cal_contrastive_loss(
                        audio_feature,
                        visual_feature,
                        temperature=args.instance_contrastive_temperature,
                    )

                if args.class_contrastive:
                    class_contra_loss = class_contrastive_loss(
                        audio_feature,
                        visual_feature,
                        routing_labels,
                        temperature=args.class_contrastive_temperature,
                    )

                if args.attn_score_distil:
                    exemplar_slice = slice(
                        data_batch_size,
                        data_batch_size + exemplar_batch_size,
                    )
                    exem_spatial = spatial_attn_score[exemplar_slice].transpose(2, 3)
                    exem_spatial = exem_spatial.reshape(-1, exem_spatial.shape[-1])
                    old_exem_spatial = old_spatial_attn_score[
                        exemplar_slice
                    ].transpose(2, 3)
                    old_exem_spatial = old_exem_spatial.reshape(
                        -1, old_exem_spatial.shape[-1]
                    )

                    exem_temporal = temporal_attn_score[exemplar_slice].transpose(1, 2)
                    exem_temporal = exem_temporal.reshape(
                        -1, exem_temporal.shape[-1]
                    )
                    old_exem_temporal = old_temporal_attn_score[
                        exemplar_slice
                    ].transpose(1, 2)
                    old_exem_temporal = old_exem_temporal.reshape(
                        -1, old_exem_temporal.shape[-1]
                    )

                    spatial_attn_dist_loss = F.kl_div(
                        exem_spatial.clamp_min(1e-12).log(),
                        old_exem_spatial,
                        reduction="sum",
                    ) / exemplar_batch_size
                    temporal_attn_dist_loss = F.kl_div(
                        exem_temporal.clamp_min(1e-12).log(),
                        old_exem_temporal,
                        reduction="sum",
                    ) / exemplar_batch_size

                old_out = old_out[:, :last_step_out_class_num]

                curr_out = logits[
                    :data_batch_size, last_step_out_class_num:
                ]
                loss_curr = CE_loss(
                    args.class_num_per_step, curr_out, labels_local
                )

                prev_out = logits[
                    data_batch_size : data_batch_size + exemplar_batch_size,
                    :last_step_out_class_num,
                ]
                loss_prev = CE_loss(
                    last_step_out_class_num, prev_out, exemplar_labels
                )

                loss_CE = (
                    loss_curr * data_batch_size
                    + loss_prev * exemplar_batch_size
                ) / (data_batch_size + exemplar_batch_size)

                if (
                    args.dataset == "AVE"
                    and args.class_num_per_step == 4
                    and step == 1
                ):
                    loss_CE = CE_loss(
                        args.class_num_per_step + last_step_out_class_num,
                        logits,
                        routing_labels,
                    )

                loss_KD = torch.zeros((), device=device)
                for old_task in range(step):
                    start = old_task * args.class_num_per_step
                    end = (old_task + 1) * args.class_num_per_step
                    soft_target = F.softmax(
                        old_out[:, start:end] / distill_temperature, dim=1
                    )
                    output_log = F.log_softmax(
                        logits[:, start:end] / distill_temperature, dim=1
                    )
                    loss_KD = loss_KD + F.kl_div(
                        output_log,
                        soft_target,
                        reduction="batchmean",
                    ) * (distill_temperature ** 2)

                loss = loss_CE + loss_KD
                if args.instance_contrastive:
                    loss = loss + args.lam_I * instance_contra_loss
                if args.class_contrastive:
                    loss = loss + args.lam_C * class_contra_loss
                if args.attn_score_distil:
                    loss = loss + args.lam * spatial_attn_dist_loss
                    loss = loss + (1.0 - args.lam) * temporal_attn_dist_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            num_steps += 1

            gate = outputs["geo_gate"].detach()
            geo = outputs["geo_contribution"].detach()
            true_gate = gate.gather(1, routing_labels.unsqueeze(1)).squeeze(1)
            true_geo = geo.gather(1, routing_labels.unsqueeze(1)).squeeze(1)

            train_gate_all_sum += float(gate.sum().item())
            train_gate_all_count += int(gate.numel())
            train_gate_all_suppressed += int((gate < 0.999).sum().item())
            train_gate_true_sum += float(true_gate.sum().item())
            train_gate_true_count += int(true_gate.numel())
            train_gate_true_suppressed += int(
                (true_gate < 0.999).sum().item()
            )
            train_true_geo_sum += float(true_geo.sum().item())

        if num_steps == 0:
            raise RuntimeError("Training loader produced zero batches")

        train_loss /= num_steps
        train_loss_list.append(train_loss)
        train_gate_all_mean = train_gate_all_sum / max(train_gate_all_count, 1)
        train_gate_all_suppressed_frac = (
            train_gate_all_suppressed / max(train_gate_all_count, 1)
        )
        train_gate_true_mean = train_gate_true_sum / max(train_gate_true_count, 1)
        train_gate_true_suppressed_frac = (
            train_gate_true_suppressed / max(train_gate_true_count, 1)
        )
        train_true_geo_mean = train_true_geo_sum / max(train_gate_true_count, 1)

        print(
            "Epoch:{} train_loss:{:.5f} gate_all:{:.5f} gate_true:{:.5f} "
            "true_suppressed:{:.5f} true_geo:{:.5f}".format(
                epoch,
                train_loss,
                train_gate_all_mean,
                train_gate_true_mean,
                train_gate_true_suppressed_frac,
                train_true_geo_mean,
            ),
            flush=True,
        )

        all_val_logits: List[torch.Tensor] = []
        all_val_labels: List[torch.Tensor] = []
        val_gate_all_sum = 0.0
        val_gate_all_count = 0
        val_gate_all_suppressed = 0
        val_gate_true_sum = 0.0
        val_gate_true_count = 0
        val_gate_true_suppressed = 0
        val_true_geo_sum = 0.0

        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader, leave=False):
                val_visual = val_data[0].to(device, non_blocking=True)
                val_audio = val_data[1].to(device, non_blocking=True)
                labels_device = val_labels.to(
                    device, non_blocking=True
                ).long()

                val_outputs = model(
                    visual=val_visual,
                    audio=val_audio,
                    return_dict=True,
                    out_gate_details=True,
                )
                all_val_logits.append(val_outputs["logits"].detach().cpu())
                all_val_labels.append(val_labels.detach().cpu().long())

                gate = val_outputs["geo_gate"].detach()
                geo = val_outputs["geo_contribution"].detach()
                true_gate = gate.gather(
                    1, labels_device.unsqueeze(1)
                ).squeeze(1)
                true_geo = geo.gather(
                    1, labels_device.unsqueeze(1)
                ).squeeze(1)

                val_gate_all_sum += float(gate.sum().item())
                val_gate_all_count += int(gate.numel())
                val_gate_all_suppressed += int((gate < 0.999).sum().item())
                val_gate_true_sum += float(true_gate.sum().item())
                val_gate_true_count += int(true_gate.numel())
                val_gate_true_suppressed += int(
                    (true_gate < 0.999).sum().item()
                )
                val_true_geo_sum += float(true_geo.sum().item())

        val_logits = torch.cat(all_val_logits, dim=0)
        val_labels = torch.cat(all_val_labels, dim=0)
        val_top1 = top_1_acc(val_logits, val_labels)
        val_acc_list.append(val_top1)

        val_gate_all_mean = val_gate_all_sum / max(val_gate_all_count, 1)
        val_gate_all_suppressed_frac = (
            val_gate_all_suppressed / max(val_gate_all_count, 1)
        )
        val_gate_true_mean = val_gate_true_sum / max(val_gate_true_count, 1)
        val_gate_true_suppressed_frac = (
            val_gate_true_suppressed / max(val_gate_true_count, 1)
        )
        val_true_geo_mean = val_true_geo_sum / max(val_gate_true_count, 1)

        routing_active = float(
            unwrap_model(model).geo_routing_enabled
            and unwrap_model(model).use_geometric_linear_routing
        )
        valid_classes = int(
            unwrap_model(model).geo_prototype_valid.sum().item()
        )

        print(
            "Epoch:{} val_res:{:.6f} gate_all:{:.5f} gate_true:{:.5f} "
            "true_suppressed:{:.5f} true_geo:{:.5f}".format(
                epoch,
                val_top1,
                val_gate_all_mean,
                val_gate_true_mean,
                val_gate_true_suppressed_frac,
                val_true_geo_mean,
            ),
            flush=True,
        )

        append_csv_rows(
            epoch_csv,
            epoch_header,
            [
                {
                    "step": step,
                    "epoch": epoch,
                    "routing_active": routing_active,
                    "prototype_valid_classes": valid_classes,
                    "train_loss": train_loss,
                    "val_acc": val_top1,
                    "train_gate_all_mean": train_gate_all_mean,
                    "train_gate_all_suppressed_frac": train_gate_all_suppressed_frac,
                    "train_gate_true_mean": train_gate_true_mean,
                    "train_gate_true_suppressed_frac": train_gate_true_suppressed_frac,
                    "train_true_geo_mean": train_true_geo_mean,
                    "val_gate_all_mean": val_gate_all_mean,
                    "val_gate_all_suppressed_frac": val_gate_all_suppressed_frac,
                    "val_gate_true_mean": val_gate_true_mean,
                    "val_gate_true_suppressed_frac": val_gate_true_suppressed_frac,
                    "val_true_geo_mean": val_true_geo_mean,
                }
            ],
        )

        save_eligible = (not routing_required) or routing_has_started
        if save_eligible and val_top1 > best_val_res:
            best_val_res = val_top1
            best_saved = True
            print("Saving best model at Epoch {}".format(epoch), flush=True)
            save_whole_model(model, checkpoint_path)

        figure_dir = os.path.join("./save/fig", args.dataset)
        os.makedirs(figure_dir, exist_ok=True)

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label="train_loss")
        plt.legend()
        plt.savefig(
            os.path.join(figure_dir, "train_loss_step_{}.png".format(step))
        )
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label="val_acc")
        plt.legend()
        plt.savefig(
            os.path.join(figure_dir, "val_acc_step_{}.png".format(step))
        )
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, optimizer, epoch)

    if not best_saved:
        raise RuntimeError(
            "No eligible best checkpoint was saved for step {}".format(step)
        )

    # Task 0 prediction remains pure AVCIL. Its best checkpoint receives a
    # detached prototype bank solely so step 1 can inherit old-class references.
    if step == 0 and args.geometric_linear_routing:
        best_model = torch_load_full(checkpoint_path, map_location="cpu").to(device)
        best_model.set_geometric_routing_enabled(False)
        refresh_geometric_prototypes(
            args=args,
            model=best_model,
            step=step,
            train_data_set=train_data_set,
            exemplar_set=exemplar_set,
            reason="step0_best_checkpoint_initialization",
        )
        best_model.set_geometric_routing_enabled(False)
        torch.save(best_model, checkpoint_path)
        del best_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Detailed testing and metric export
# ============================================================================

def detailed_test(
    args,
    step: int,
    test_data_set,
    task_best_acc_list: List[float],
    metrics_root: str,
    metrics_state: dict,
    id_to_category: Dict[int, str],
) -> Optional[float]:
    print("=====================================")
    print("Start testing...")
    print("=====================================")

    checkpoint_path = os.path.join(
        "./save", args.dataset, "step_{}_best_model.pkl".format(step)
    )
    model = torch_load_full(checkpoint_path, map_location="cpu").to(device)
    model.eval()

    test_loader = DataLoader(
        test_data_set,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    routed_logits_list: List[torch.Tensor] = []
    guided_logits_list: List[torch.Tensor] = []
    uniform_logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    true_gate_list: List[torch.Tensor] = []
    predicted_gate_list: List[torch.Tensor] = []
    true_geo_list: List[torch.Tensor] = []
    decision_guidance_list: List[torch.Tensor] = []

    gate_all_sum = 0.0
    gate_all_count = 0
    gate_all_suppressed = 0

    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            visual = test_data[0].to(device, non_blocking=True)
            audio = test_data[1].to(device, non_blocking=True)
            labels = test_labels.to(device, non_blocking=True).long()

            outputs = model(
                visual=visual,
                audio=audio,
                return_dict=True,
                out_gate_details=True,
            )
            routed_logits = outputs["logits"]
            guided_logits = outputs["guided_logits"]
            uniform_logits = outputs["uniform_logits"]
            gate = outputs["geo_gate"]
            geo = outputs["geo_contribution"]

            true_gate = gate.gather(1, labels.unsqueeze(1)).squeeze(1)
            true_geo = geo.gather(1, labels.unsqueeze(1)).squeeze(1)
            routed_pred = routed_logits.argmax(dim=1)
            predicted_gate = gate.gather(
                1, routed_pred.unsqueeze(1)
            ).squeeze(1)

            decision_guidance = target_vs_rest_benefit(
                guided_logits, labels
            ) - target_vs_rest_benefit(uniform_logits, labels)

            routed_logits_list.append(routed_logits.detach().cpu())
            guided_logits_list.append(guided_logits.detach().cpu())
            uniform_logits_list.append(uniform_logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
            true_gate_list.append(true_gate.detach().cpu())
            predicted_gate_list.append(predicted_gate.detach().cpu())
            true_geo_list.append(true_geo.detach().cpu())
            decision_guidance_list.append(decision_guidance.detach().cpu())

            gate_all_sum += float(gate.sum().item())
            gate_all_count += int(gate.numel())
            gate_all_suppressed += int((gate < 0.999).sum().item())

    all_logits = torch.cat(routed_logits_list, dim=0)
    all_guided_logits = torch.cat(guided_logits_list, dim=0)
    all_uniform_logits = torch.cat(uniform_logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0).long()
    all_true_gate = torch.cat(true_gate_list, dim=0)
    all_predicted_gate = torch.cat(predicted_gate_list, dim=0)
    all_true_geo = torch.cat(true_geo_list, dim=0)
    all_decision_guidance = torch.cat(decision_guidance_list, dim=0)

    pred = all_logits.argmax(dim=1).long()
    guided_pred = all_guided_logits.argmax(dim=1).long()
    uniform_pred = all_uniform_logits.argmax(dim=1).long()

    overall_acc = (pred == all_labels).float().mean().item()
    guided_acc = (guided_pred == all_labels).float().mean().item()
    uniform_acc = (uniform_pred == all_labels).float().mean().item()

    print(
        "Incremental step {} testing: routed={:.6f}, guided={:.6f}, "
        "uniform={:.6f}, delta(routed-guided)={:+.6f}".format(
            step,
            overall_acc,
            guided_acc,
            uniform_acc,
            overall_acc - guided_acc,
        ),
        flush=True,
    )

    sign_agreement = (
        torch.sign(all_true_geo) == torch.sign(all_decision_guidance)
    ).float().mean().item()
    spearman = safe_spearman(
        all_true_geo.numpy(), all_decision_guidance.numpy()
    )

    step_row = {
        "step": step,
        "routing_enabled": float(
            model.geo_routing_enabled
            and model.use_geometric_linear_routing
        ),
        "prototype_valid_classes": int(
            model.geo_prototype_valid.sum().item()
        ),
        "overall_acc_routed": overall_acc,
        "overall_acc_guided": guided_acc,
        "overall_acc_uniform": uniform_acc,
        "delta_routed_minus_guided": overall_acc - guided_acc,
        "gate_all_mean": gate_all_sum / max(gate_all_count, 1),
        "gate_all_suppressed_frac": gate_all_suppressed
        / max(gate_all_count, 1),
        "gate_true_mean": float(all_true_gate.mean().item()),
        "gate_true_suppressed_frac": float(
            (all_true_gate < 0.999).float().mean().item()
        ),
        "gate_predicted_class_mean": float(all_predicted_gate.mean().item()),
        "true_geo_contribution_mean": float(all_true_geo.mean().item()),
        "decision_guidance_mean": float(all_decision_guidance.mean().item()),
        "geo_decision_sign_agreement": sign_agreement,
        "geo_decision_spearman": spearman,
    }
    append_csv_rows(
        os.path.join(metrics_root, "routing_step_metrics.csv"),
        list(step_row.keys()),
        [step_row],
    )
    print("Routing diagnostics: {}".format(step_row), flush=True)

    # ------------------------ per-task forgetting ------------------------
    task_size = args.class_num_per_step
    old_task_acc_list: List[float] = []
    current_step_acc = 0.0
    for task_id in range(step + 1):
        lo = task_id * task_size
        hi = (task_id + 1) * task_size
        mask = (all_labels >= lo) & (all_labels < hi)
        task_acc = (
            (pred[mask] == all_labels[mask]).float().mean().item()
            if mask.any()
            else 0.0
        )
        if task_id == step:
            current_step_acc = task_acc
        else:
            old_task_acc_list.append(task_acc)

    if step > 0:
        forgetting = float(
            np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        )
        print("task-level forgetting: {:.6f}".format(forgetting), flush=True)
        for task_id in range(len(task_best_acc_list)):
            task_best_acc_list[task_id] = max(
                task_best_acc_list[task_id], old_task_acc_list[task_id]
            )
    else:
        forgetting = None
    task_best_acc_list.append(current_step_acc)

    # ------------------------ per-class metrics --------------------------
    num_seen_classes = (step + 1) * task_size
    stats = compute_per_class_prf(all_labels, pred, num_seen_classes)
    best_f1 = metrics_state.get("best_f1", {})
    first_seen = metrics_state.get("first_seen_step", {})

    per_class_rows: List[dict] = []
    routing_class_rows: List[dict] = []

    for class_id in range(num_seen_classes):
        class_key = str(class_id)
        if class_key not in first_seen:
            first_seen[class_key] = step

        support = int(stats["support"][class_id])
        precision = float(stats["precision"][class_id])
        recall = float(stats["recall"][class_id])
        f1_value = float(stats["f1"][class_id])

        best_before = float(best_f1[class_key]) if class_key in best_f1 else None
        forget_f1 = best_before - f1_value if best_before is not None else 0.0
        new_best = f1_value if best_before is None else max(best_before, f1_value)
        best_f1[class_key] = new_best

        per_class_rows.append(
            {
                "step": step,
                "class_id": class_id,
                "category_name": id_to_category.get(
                    class_id, "class_{}".format(class_id)
                ),
                "first_seen_step": int(first_seen[class_key]),
                "support": support,
                "tp": int(stats["tp"][class_id]),
                "fp": int(stats["fp"][class_id]),
                "fn": int(stats["fn"][class_id]),
                "precision": precision,
                "recall": recall,
                "f1": f1_value,
                "best_f1": float(new_best),
                "forget_f1": float(forget_f1),
                "forgetting": float(forgetting)
                if forgetting is not None
                else 0.0,
                "overall_acc": overall_acc,
            }
        )

        class_mask = all_labels == class_id
        if class_mask.any():
            class_geo = all_true_geo[class_mask]
            class_dec = all_decision_guidance[class_mask]
            class_gate = all_true_gate[class_mask]
            class_sign = (
                torch.sign(class_geo) == torch.sign(class_dec)
            ).float().mean().item()
            class_spearman = safe_spearman(
                class_geo.numpy(), class_dec.numpy()
            )
            class_routed_acc = (
                pred[class_mask] == all_labels[class_mask]
            ).float().mean().item()
            class_guided_acc = (
                guided_pred[class_mask] == all_labels[class_mask]
            ).float().mean().item()
            class_uniform_acc = (
                uniform_pred[class_mask] == all_labels[class_mask]
            ).float().mean().item()
        else:
            class_geo = torch.zeros(1)
            class_dec = torch.zeros(1)
            class_gate = torch.ones(1)
            class_sign = 0.0
            class_spearman = 0.0
            class_routed_acc = 0.0
            class_guided_acc = 0.0
            class_uniform_acc = 0.0

        routing_class_rows.append(
            {
                "step": step,
                "class_id": class_id,
                "category_name": id_to_category.get(
                    class_id, "class_{}".format(class_id)
                ),
                "support": support,
                "true_class_gate_mean": float(class_gate.mean().item()),
                "true_class_suppressed_frac": float(
                    (class_gate < 0.999).float().mean().item()
                ),
                "true_geo_contribution_mean": float(class_geo.mean().item()),
                "decision_guidance_mean": float(class_dec.mean().item()),
                "geo_decision_sign_agreement": class_sign,
                "geo_decision_spearman": class_spearman,
                "routed_acc": class_routed_acc,
                "guided_acc": class_guided_acc,
                "uniform_acc": class_uniform_acc,
                "delta_routed_minus_guided": class_routed_acc
                - class_guided_acc,
            }
        )

    per_class_header = [
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
    ]
    append_csv_rows(
        os.path.join(metrics_root, "per_class_metrics.csv"),
        per_class_header,
        per_class_rows,
    )
    append_csv_rows(
        os.path.join(metrics_root, "per_class_routing_metrics.csv"),
        list(routing_class_rows[0].keys()),
        routing_class_rows,
    )

    metrics_state["best_f1"] = best_f1
    metrics_state["first_seen_step"] = first_seen
    save_json(
        metrics_state, os.path.join(metrics_root, "per_class_state.json")
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return forgetting


# ============================================================================
# CLI
# ============================================================================

def dataset_type(value: str) -> str:
    if value in {"AVE", "ksounds"} or "VGGSound" in value:
        return value
    raise argparse.ArgumentTypeError(
        "dataset must be 'AVE', 'ksounds', or contain 'VGGSound'"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=dataset_type, default="AVE")
    parser.add_argument(
        "--modality",
        type=str,
        default="audio-visual",
        choices=["audio-visual"],
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        default="/mnt/data2/wpian/dataset/VGGSound",
    )
    parser.add_argument("--meta_root", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument("--exemplar_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epoches", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=28)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[500], nargs="+")

    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--lam_I", type=float, default=0.5)
    parser.add_argument("--lam_C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_num_per_step", type=int, default=7)
    parser.add_argument("--memory_size", type=int, default=340)

    parser.add_argument("--instance_contrastive", action="store_true", default=False)
    parser.add_argument("--class_contrastive", action="store_true", default=False)
    parser.add_argument("--attn_score_distil", action="store_true", default=False)
    parser.add_argument("--instance_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--class_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--distillation_temperature", type=float, default=2.0)

    # Class-conditional geometric linear routing.
    parser.add_argument("--geometric_linear_routing", action="store_true", default=False)
    parser.add_argument("--geo_routing_start_step", type=int, default=1)
    parser.add_argument("--geo_routing_warmup_epochs", type=int, default=20)
    parser.add_argument("--geo_prototype_refresh_epochs", type=int, default=20)
    parser.add_argument("--geo_prototype_temperature", type=float, default=0.1)
    parser.add_argument("--geo_gate_temperature", type=float, default=1.0)
    parser.add_argument("--geo_gate_min", type=float, default=0.0)

    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--dump_tsne", action="store_true")
    parser.add_argument(
        "--tsne_feature",
        type=str,
        default="logits",
        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"],
    )
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    return parser


def validate_args(args) -> None:
    if args.num_classes % args.class_num_per_step != 0:
        raise ValueError("num_classes must be divisible by class_num_per_step")
    if args.distillation_temperature <= 0:
        raise ValueError("distillation_temperature must be > 0")
    if args.geo_routing_start_step < 0:
        raise ValueError("geo_routing_start_step must be >= 0")
    if args.geo_routing_warmup_epochs < 0:
        raise ValueError("geo_routing_warmup_epochs must be >= 0")
    if args.geo_prototype_refresh_epochs < 0:
        raise ValueError("geo_prototype_refresh_epochs must be >= 0")
    if args.geo_prototype_temperature <= 0:
        raise ValueError("geo_prototype_temperature must be > 0")
    if args.geo_gate_temperature <= 0:
        raise ValueError("geo_gate_temperature must be > 0")
    if not 0.0 <= args.geo_gate_min < 1.0:
        raise ValueError("geo_gate_min must be in [0, 1)")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    print(args, flush=True)

    total_incremental_steps = args.num_classes // args.class_num_per_step
    setup_seed(args.seed)
    print("Training start time: {}".format(datetime.now()), flush=True)

    train_set = IcaAVELoader(args=args, mode="train", modality=args.modality)
    val_set = IcaAVELoader(args=args, mode="val", modality=args.modality)
    test_set = IcaAVELoader(args=args, mode="test", modality=args.modality)
    exemplar_set = exemplarLoader(args=args, modality=args.modality)

    category_encode_dict = train_set.category_encode_dict
    id_to_category = {value: key for key, value in category_encode_dict.items()}

    checkpoint_root = os.path.join("./save", args.dataset)
    figure_root = os.path.join("./save/fig", args.dataset)
    metrics_root = os.path.join("./save/metrics", args.dataset)
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(figure_root, exist_ok=True)
    os.makedirs(metrics_root, exist_ok=True)

    metrics_files = [
        "per_class_metrics.csv",
        "per_class_routing_metrics.csv",
        "routing_step_metrics.csv",
        "routing_epoch_metrics.csv",
        "per_class_state.json",
    ]
    if not args.test_only:
        for filename in metrics_files:
            path = os.path.join(metrics_root, filename)
            if os.path.exists(path):
                os.remove(path)
        metrics_state = {"best_f1": {}, "first_seen_step": {}}
        save_json(metrics_state, os.path.join(metrics_root, "per_class_state.json"))
    else:
        metrics_state = load_json(
            os.path.join(metrics_root, "per_class_state.json"),
            {"best_f1": {}, "first_seen_step": {}},
        )

    task_best_acc_list: List[float] = []
    step_forgetting_list: List[float] = []

    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)
        exemplar_set._set_incremental_step_(step)

        print("Incremental step: {}".format(step), flush=True)
        if not args.test_only:
            train(args, step, train_set, val_set, exemplar_set)

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

        checkpoint_path = os.path.join(
            checkpoint_root,
            "step_{}_best_model.pkl".format(step),
        )
        if args.dump_tsne:
            print("Dumping t-SNE plots for step {}...".format(step), flush=True)
            output_root = os.path.join(args.tsne_out_root, args.dataset)
            make_tsne_plots_for_step(
                args=args,
                step=step,
                test_set=test_set,
                ckpt_path=checkpoint_path,
                out_root=output_root,
                feature_type=args.tsne_feature,
                max_points_per_class=args.tsne_max_points_per_class,
            )

    mean_forgetting = (
        float(np.mean(step_forgetting_list)) if step_forgetting_list else 0.0
    )
    print("Average Forgetting: {:.6f}".format(mean_forgetting), flush=True)

    if args.dataset != "AVE":
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()
        exemplar_set.close_visual_features_h5()