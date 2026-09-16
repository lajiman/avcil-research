from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from itertools import cycle
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Portable import resolution: support both a self-contained experiment folder
# and the original layout where dataloader_ours.py/model live one directory up.
for candidate in (SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from dataloader_ours import IcaAVELoader, exemplarLoader
from model.audio_visual_model_incremental_oracle_gate import IncreAudioVisualNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def boolean_string(value: str) -> bool:
    if value not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return value == "True"


def ce_loss(num_classes, logits, labels):
    targets = F.one_hot(labels, num_classes=num_classes).float()
    return -torch.mean(
        torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1)
    )


def cal_contrastive_loss(feature_1, feature_2, temperature=0.1):
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    labels = torch.arange(score.shape[0], device=score.device)
    return ce_loss(score.shape[0], score, labels)


def class_contrastive_loss(
    feature_1, feature_2, labels, temperature=0.1
):
    class_matrix = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    return -torch.mean(
        torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)
    )


def adjust_learning_rate(args, optimizer, epoch):
    milestones = np.asarray(args.milestones) - 1
    if epoch in milestones:
        for group in optimizer.param_groups:
            group["lr"] *= 0.1
        print(f"Reduced learning rate at epoch {epoch}", flush=True)


def append_csv_rows(
    path: str, fieldnames: Iterable[str], rows: Iterable[Dict]
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames)
        )
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(payload: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_per_class_prf(y_true, y_pred, num_classes):
    y_true = y_true.long()
    y_pred = y_pred.long()
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for class_id in range(num_classes):
        true_c = y_true.eq(class_id)
        pred_c = y_pred.eq(class_id)
        tp[class_id] = (true_c & pred_c).sum()
        fp[class_id] = ((~true_c) & pred_c).sum()
        fn[class_id] = (true_c & (~pred_c)).sum()

    support = tp + fn
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)

    for class_id in range(num_classes):
        tp_c = int(tp[class_id])
        fp_c = int(fp[class_id])
        fn_c = int(fn[class_id])
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



def load_full_model(path: str, map_location):
    """Load a trusted locally produced full-model checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap(model: nn.Module) -> IncreAudioVisualNet:
    return model.module if isinstance(model, nn.DataParallel) else model


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().split())


def load_and_remap_gate_table(
    gate_json_path: str,
    category_encode_dict: Dict[str, int],
    num_classes: int,
) -> Tuple[torch.Tensor, Dict]:
    if not os.path.exists(gate_json_path):
        raise FileNotFoundError(
            f"Oracle gate table does not exist: {gate_json_path}. "
            "Run the full-class gate discovery job first."
        )

    with open(gate_json_path, "r") as handle:
        payload = json.load(handle)

    category_to_gate = payload.get("category_to_gate")
    if not isinstance(category_to_gate, dict):
        raise ValueError(
            "Gate JSON is missing category_to_gate"
        )

    source = {
        normalize_name(name): float(value)
        for name, value in category_to_gate.items()
    }
    gate_vector = torch.ones(num_classes, dtype=torch.float32)
    missing = []

    for category_name, local_id_raw in category_encode_dict.items():
        local_id = int(local_id_raw)
        key = normalize_name(category_name)
        if key not in source:
            missing.append(str(category_name))
        else:
            gate_vector[local_id] = source[key]

    if missing:
        raise KeyError(
            "Gate table is missing categories: "
            + ", ".join(missing[:20])
        )
    if not torch.isfinite(gate_vector).all():
        raise ValueError("Gate vector contains NaN or Inf")
    if (gate_vector < 0).any() or (gate_vector > 1).any():
        raise ValueError("Gate values must be in [0,1]")

    return gate_vector, payload


def apply_fixed_gate(model, gate_vector):
    model.set_oracle_gate_values(gate_vector, mode="fixed")


def make_model(args, step_out_class_num, gate_vector):
    model = IncreAudioVisualNet(args, step_out_class_num)
    apply_fixed_gate(model, gate_vector)
    return model


def train(
    args,
    step,
    train_data_set,
    val_data_set,
    exemplar_set,
    gate_vector,
):
    kd_temperature = 2.0

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

    if step == 0:
        model = make_model(
            args, step_out_class_num, gate_vector
        )
        old_model = None
        exemplar_loader = None
        last_step_out_class_num = 0
    else:
        previous_path = (
            f"./save/{args.dataset}/"
            f"step_{step - 1}_best_model.pkl"
        )
        model = load_full_model(previous_path, map_location="cpu")
        model.incremental_classifier(step_out_class_num)
        apply_fixed_gate(model, gate_vector)

        old_model = load_full_model(
            previous_path, map_location="cpu"
        )
        apply_fixed_gate(old_model, gate_vector)
        old_model.eval()

        exemplar_loader = DataLoader(
            exemplar_set,
            batch_size=min(
                args.exemplar_batch_size, len(exemplar_set)
            ),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        last_step_out_class_num = (
            step * args.class_num_per_step
        )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if old_model is not None:
            old_model = nn.DataParallel(old_model)

    model = model.to(device)
    if old_model is not None:
        old_model = old_model.to(device)
        old_model.eval()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics_root = f"./save/metrics/{args.dataset}"
    epoch_csv = os.path.join(
        metrics_root, "train_epoch_metrics.csv"
    )
    best_val = -1.0

    for epoch in range(args.max_epoches):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = 0

        if step == 0:
            iterator = tqdm(
                train_loader,
                desc=f"step {step} epoch {epoch}",
            )
        else:
            iterator = tzip(
                train_loader, cycle(exemplar_loader)
            )

        for samples in iterator:
            optimizer.zero_grad(set_to_none=True)

            if step == 0:
                data, labels_cpu = samples
                labels = labels_cpu.to(device).long()
                visual = data[0].to(device)
                audio = data[1].to(device)

                out, _, _ = model(
                    visual=visual,
                    audio=audio,
                    labels=labels,
                    out_feature_before_fusion=True,
                )
                loss = ce_loss(
                    step_out_class_num, out, labels
                )
                batch_labels_for_acc = labels
                batch_logits_for_acc = out

            else:
                curr, prev = samples
                data, labels_cpu = curr
                exemplar_data, exemplar_labels_cpu = prev

                labels = labels_cpu.to(device).long()
                labels_local = (
                    labels % args.class_num_per_step
                ).long()
                exemplar_labels = (
                    exemplar_labels_cpu.to(device).long()
                )

                current_bs = labels.shape[0]
                exemplar_bs = exemplar_labels.shape[0]
                total_labels = torch.cat(
                    (labels, exemplar_labels), dim=0
                )
                total_visual = torch.cat(
                    (data[0], exemplar_data[0]), dim=0
                ).to(device)
                total_audio = torch.cat(
                    (data[1], exemplar_data[1]), dim=0
                ).to(device)

                (
                    out,
                    audio_feature,
                    visual_feature,
                    spatial_attn,
                    temporal_attn,
                ) = model(
                    visual=total_visual,
                    audio=total_audio,
                    labels=total_labels,
                    out_feature_before_fusion=True,
                    out_attn_score=True,
                )

                with torch.no_grad():
                    (
                        old_out,
                        old_spatial_attn,
                        old_temporal_attn,
                    ) = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        labels=total_labels,
                        out_attn_score=True,
                    )
                    old_out = old_out.detach()[
                        :, :last_step_out_class_num
                    ]
                    old_spatial_attn = (
                        old_spatial_attn.detach()
                    )
                    old_temporal_attn = (
                        old_temporal_attn.detach()
                    )

                curr_out = out[
                    :current_bs,
                    last_step_out_class_num:,
                ]
                loss_curr = ce_loss(
                    args.class_num_per_step,
                    curr_out,
                    labels_local,
                )

                prev_out = out[
                    current_bs:current_bs + exemplar_bs,
                    :last_step_out_class_num,
                ]
                loss_prev = ce_loss(
                    last_step_out_class_num,
                    prev_out,
                    exemplar_labels,
                )
                loss_ce = (
                    loss_curr * current_bs
                    + loss_prev * exemplar_bs
                ) / (current_bs + exemplar_bs)

                if (
                    args.dataset == "AVE"
                    and args.class_num_per_step == 4
                    and step == 1
                ):
                    loss_ce = ce_loss(
                        args.class_num_per_step
                        + last_step_out_class_num,
                        out,
                        total_labels,
                    )

                kd_terms = []
                for task_id in range(step):
                    start = (
                        task_id * args.class_num_per_step
                    )
                    end = (
                        (task_id + 1)
                        * args.class_num_per_step
                    )
                    soft_target = F.softmax(
                        old_out[:, start:end]
                        / kd_temperature,
                        dim=1,
                    )
                    output_log = F.log_softmax(
                        out[:, start:end]
                        / kd_temperature,
                        dim=1,
                    )
                    kd_terms.append(
                        F.kl_div(
                            output_log,
                            soft_target,
                            reduction="batchmean",
                        )
                        * (kd_temperature ** 2)
                    )

                loss_kd = (
                    torch.stack(kd_terms).sum()
                    if kd_terms
                    else out.new_zeros(())
                )
                loss = loss_ce + loss_kd

                if args.instance_contrastive:
                    loss = (
                        loss
                        + args.lam_I
                        * cal_contrastive_loss(
                            audio_feature,
                            visual_feature,
                            args.instance_contrastive_temperature,
                        )
                    )

                if args.class_contrastive:
                    loss = (
                        loss
                        + args.lam_C
                        * class_contrastive_loss(
                            audio_feature,
                            visual_feature,
                            total_labels,
                            args.class_contrastive_temperature,
                        )
                    )

                if args.attn_score_distil:
                    exem_spatial = spatial_attn[
                        current_bs:current_bs + exemplar_bs
                    ]
                    old_exem_spatial = old_spatial_attn[
                        current_bs:current_bs + exemplar_bs
                    ]
                    exem_temporal = temporal_attn[
                        current_bs:current_bs + exemplar_bs
                    ]
                    old_exem_temporal = old_temporal_attn[
                        current_bs:current_bs + exemplar_bs
                    ]

                    exem_spatial = (
                        exem_spatial.transpose(2, 3)
                        .reshape(-1, exem_spatial.shape[2])
                    )
                    old_exem_spatial = (
                        old_exem_spatial.transpose(2, 3)
                        .reshape(-1, old_exem_spatial.shape[2])
                    )
                    exem_temporal = (
                        exem_temporal.transpose(1, 2)
                        .reshape(-1, exem_temporal.shape[1])
                    )
                    old_exem_temporal = (
                        old_exem_temporal.transpose(1, 2)
                        .reshape(
                            -1,
                            old_exem_temporal.shape[1],
                        )
                    )

                    spatial_dist = F.kl_div(
                        exem_spatial
                        .clamp_min(1e-8)
                        .log(),
                        old_exem_spatial,
                        reduction="sum",
                    ) / exemplar_bs
                    temporal_dist = F.kl_div(
                        exem_temporal
                        .clamp_min(1e-8)
                        .log(),
                        old_exem_temporal,
                        reduction="sum",
                    ) / exemplar_bs

                    loss = (
                        loss
                        + args.lam * spatial_dist
                        + (1.0 - args.lam)
                        * temporal_dist
                    )

                batch_labels_for_acc = total_labels
                batch_logits_for_acc = out

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_correct += int(
                batch_logits_for_acc.argmax(dim=1)
                .eq(batch_labels_for_acc)
                .sum()
                .item()
            )
            running_total += int(
                batch_labels_for_acc.numel()
            )
            num_batches += 1

        train_loss = running_loss / max(
            num_batches, 1
        )
        train_acc = running_correct / max(
            running_total, 1
        )

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_data, val_labels_cpu in tqdm(
                val_loader, desc="val", leave=False
            ):
                val_labels = (
                    val_labels_cpu.to(device).long()
                )
                val_logits = model(
                    visual=val_data[0].to(device),
                    audio=val_data[1].to(device),
                    labels=val_labels,
                )
                val_correct += int(
                    val_logits.argmax(dim=1)
                    .eq(val_labels)
                    .sum()
                    .item()
                )
                val_total += int(
                    val_labels.numel()
                )

        val_acc = val_correct / max(val_total, 1)
        row = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        append_csv_rows(
            epoch_csv, row.keys(), [row]
        )
        print(
            f"Step:{step} Epoch:{epoch} "
            f"train_loss:{train_loss:.6f} "
            f"train_acc:{train_acc:.6f} "
            f"val_acc:{val_acc:.6f}",
            flush=True,
        )

        if val_acc > best_val:
            best_val = val_acc
            save_path = (
                f"./save/{args.dataset}/"
                f"step_{step}_best_model.pkl"
            )
            torch.save(unwrap(model), save_path)
            print(
                f"Saving best model at epoch {epoch}",
                flush=True,
            )

        if args.lr_decay and step > 0:
            adjust_learning_rate(
                args, optimizer, epoch
            )


def detailed_test(
    args,
    step,
    test_data_set,
    task_best_acc_list,
    metrics_root,
    metrics_state,
    id_to_category,
    gate_vector,
):
    model = load_full_model(
        f"./save/{args.dataset}/"
        f"step_{step}_best_model.pkl",
        map_location=device,
    )
    apply_fixed_gate(model, gate_vector)
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_data_set,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for test_data, labels_cpu in tqdm(
            test_loader, desc="test"
        ):
            labels = labels_cpu.to(device).long()
            logits = model(
                visual=test_data[0].to(device),
                audio=test_data[1].to(device),
                labels=labels,
            )
            logits_list.append(
                logits.detach().cpu()
            )
            labels_list.append(
                labels_cpu.long()
            )

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    pred = all_logits.argmax(dim=1)
    overall_acc = float(
        pred.eq(all_labels).float().mean().item()
    )
    print(
        f"Incremental step {step} "
        f"oracle overall acc: {overall_acc:.6f}",
        flush=True,
    )

    classes_per_step = args.class_num_per_step
    num_seen = (
        (step + 1) * classes_per_step
    )
    old_task_accs = []
    current_task_acc = 0.0

    for task_id in range(step + 1):
        low = task_id * classes_per_step
        high = (
            (task_id + 1)
            * classes_per_step
        )
        mask = (
            (all_labels >= low)
            & (all_labels < high)
        )
        task_acc = (
            float(
                pred[mask]
                .eq(all_labels[mask])
                .float()
                .mean()
                .item()
            )
            if mask.any()
            else 0.0
        )
        if task_id == step:
            current_task_acc = task_acc
        else:
            old_task_accs.append(task_acc)

    if step > 0:
        forgetting = float(
            np.mean(
                np.asarray(task_best_acc_list)
                - np.asarray(old_task_accs)
            )
        )
        for index in range(
            len(task_best_acc_list)
        ):
            task_best_acc_list[index] = max(
                task_best_acc_list[index],
                old_task_accs[index],
            )
    else:
        forgetting = None

    task_best_acc_list.append(
        current_task_acc
    )

    stats = compute_per_class_prf(
        all_labels, pred, num_seen
    )
    best_f1 = metrics_state.get(
        "best_f1", {}
    )
    first_seen = metrics_state.get(
        "first_seen_step", {}
    )
    rows = []

    for class_id in range(num_seen):
        key = str(class_id)
        first_seen.setdefault(key, step)
        f1_value = float(
            stats["f1"][class_id]
        )
        best_before = (
            float(best_f1[key])
            if key in best_f1
            else None
        )
        forget_f1 = (
            best_before - f1_value
            if best_before is not None
            else 0.0
        )
        best_f1[key] = (
            f1_value
            if best_before is None
            else max(best_before, f1_value)
        )

        rows.append(
            {
                "step": step,
                "class_id": class_id,
                "category_name": id_to_category.get(
                    class_id,
                    f"class_{class_id}",
                ),
                "first_seen_step": int(
                    first_seen[key]
                ),
                "oracle_gate": float(
                    gate_vector[class_id].item()
                ),
                "support": int(
                    stats["support"][class_id]
                ),
                "tp": int(
                    stats["tp"][class_id]
                ),
                "fp": int(
                    stats["fp"][class_id]
                ),
                "fn": int(
                    stats["fn"][class_id]
                ),
                "precision": float(
                    stats["precision"][class_id]
                ),
                "recall": float(
                    stats["recall"][class_id]
                ),
                "f1": f1_value,
                "best_f1": float(
                    best_f1[key]
                ),
                "forget_f1": float(
                    forget_f1
                ),
                "forgetting": (
                    float(forgetting)
                    if forgetting is not None
                    else 0.0
                ),
                "overall_acc": overall_acc,
            }
        )

    if rows:
        append_csv_rows(
            os.path.join(
                metrics_root,
                "per_class_metrics.csv",
            ),
            rows[0].keys(),
            rows,
        )

    step_row = {
        "step": step,
        "num_seen_classes": num_seen,
        "overall_acc": overall_acc,
        "current_task_acc": current_task_acc,
        "old_task_mean_acc": (
            float(np.mean(old_task_accs))
            if old_task_accs
            else 0.0
        ),
        "forgetting": (
            float(forgetting)
            if forgetting is not None
            else 0.0
        ),
    }
    append_csv_rows(
        os.path.join(
            metrics_root, "step_metrics.csv"
        ),
        step_row.keys(),
        [step_row],
    )

    metrics_state["best_f1"] = best_f1
    metrics_state[
        "first_seen_step"
    ] = first_seen
    save_json(
        metrics_state,
        os.path.join(
            metrics_root,
            "per_class_state.json",
        ),
    )
    return forgetting


def build_parser():
    parser = argparse.ArgumentParser()

    def dataset_type(value: str):
        if (
            value in {"AVE", "ksounds"}
            or "VGGSound" in value
        ):
            return value
        raise argparse.ArgumentTypeError(
            "dataset must be AVE, ksounds, "
            "or contain VGGSound"
        )

    parser.add_argument(
        "--dataset",
        type=dataset_type,
        required=True,
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="audio-visual",
        choices=["audio-visual"],
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--meta_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--exemplar_batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_epoches",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--class_num_per_step",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr_decay",
        type=boolean_string,
        default=False,
    )
    parser.add_argument(
        "--milestones",
        type=int,
        default=[100],
        nargs="+",
    )
    parser.add_argument(
        "--lam", type=float, default=0.5
    )
    parser.add_argument(
        "--lam_I",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lam_C",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--instance_contrastive",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--class_contrastive",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--attn_score_distil",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--instance_contrastive_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--class_contrastive_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--oracle_gate_table",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--oracle_gate_mode",
        type=str,
        default="fixed",
    )
    parser.add_argument(
        "--oracle_gate_table_size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--oracle_gate_init",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--oracle_gate_min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--oracle_gate_max",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--z1_cm_projection_head",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--z1_cm_projection_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--z1_cm_projection_hidden_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--z1_cm_projection_type",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
    )
    return parser


def main(args):
    setup_seed(args.seed)
    args.oracle_gate_mode = "fixed"
    args.oracle_gate_table_size = (
        args.num_classes
    )

    print(args, flush=True)
    print(
        f"Training start time: {datetime.now()}",
        flush=True,
    )

    train_set = IcaAVELoader(
        args=args,
        mode="train",
        modality=args.modality,
    )
    val_set = IcaAVELoader(
        args=args,
        mode="val",
        modality=args.modality,
    )
    test_set = IcaAVELoader(
        args=args,
        mode="test",
        modality=args.modality,
    )
    exemplar_set = exemplarLoader(
        args=args,
        modality=args.modality,
    )

    category_encode_dict = (
        train_set.category_encode_dict
    )
    id_to_category = {
        int(class_id): str(category_name)
        for category_name, class_id
        in category_encode_dict.items()
    }
    gate_vector, gate_payload = (
        load_and_remap_gate_table(
            args.oracle_gate_table,
            category_encode_dict,
            args.num_classes,
        )
    )

    ckpt_root = f"./save/{args.dataset}"
    metrics_root = (
        f"./save/metrics/{args.dataset}"
    )
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(metrics_root, exist_ok=True)

    save_json(
        {
            "source_gate_table": os.path.abspath(
                args.oracle_gate_table
            ),
            "source_dataset": gate_payload.get(
                "dataset"
            ),
            "dataset": args.dataset,
            "category_to_local_id": {
                str(name): int(class_id)
                for name, class_id
                in category_encode_dict.items()
            },
            "local_id_to_category": {
                str(class_id): name
                for class_id, name
                in id_to_category.items()
            },
            "local_id_to_gate": {
                str(class_id): float(
                    gate_vector[class_id].item()
                )
                for class_id
                in range(args.num_classes)
            },
        },
        os.path.join(
            metrics_root,
            "remapped_oracle_gate_table.json",
        ),
    )

    if not args.test_only:
        for filename in [
            "per_class_metrics.csv",
            "step_metrics.csv",
            "train_epoch_metrics.csv",
        ]:
            path = os.path.join(
                metrics_root, filename
            )
            if os.path.exists(path):
                os.remove(path)

    metrics_state = {
        "best_f1": {},
        "first_seen_step": {},
    }
    save_json(
        metrics_state,
        os.path.join(
            metrics_root,
            "per_class_state.json",
        ),
    )

    task_best_acc_list: List[float] = []
    forgetting_list: List[float] = []
    total_steps = (
        args.num_classes
        // args.class_num_per_step
    )

    for step in range(total_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)
        exemplar_set._set_incremental_step_(
            step
        )

        print(
            f"Incremental step: {step}",
            flush=True,
        )
        if not args.test_only:
            train(
                args,
                step,
                train_set,
                val_set,
                exemplar_set,
                gate_vector,
            )

        forgetting = detailed_test(
            args,
            step,
            test_set,
            task_best_acc_list,
            metrics_root,
            metrics_state,
            id_to_category,
            gate_vector,
        )
        if forgetting is not None:
            forgetting_list.append(
                forgetting
            )

    mean_forgetting = (
        float(np.mean(forgetting_list))
        if forgetting_list
        else 0.0
    )
    print(
        f"Average Forgetting: "
        f"{mean_forgetting:.6f}",
        flush=True,
    )
    print(
        f"Training end time: {datetime.now()}",
        flush=True,
    )

    if args.dataset != "AVE":
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()
        exemplar_set.close_visual_features_h5()


if __name__ == "__main__":
    main(build_parser().parse_args())