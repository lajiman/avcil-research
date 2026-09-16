from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Portable import resolution: support both a self-contained experiment folder
# and the original layout where dataloader_ours.py/model live one directory up.
for candidate in (SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from dataloader_ours import IcaAVELoader
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


def ce_loss(num_classes: int, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = F.one_hot(labels, num_classes=num_classes).float()
    return -torch.mean(
        torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1)
    )


def instance_contrastive_loss(feature_1, feature_2, temperature):
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    labels = torch.arange(score.shape[0], device=score.device)
    return ce_loss(score.shape[0], score, labels)


def class_contrastive_loss(feature_1, feature_2, labels, temperature):
    class_matrix = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    return -torch.mean(
        torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)
    )


def append_csv_row(path: str, fieldnames: Iterable[str], row: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)



def load_full_model(path: str, map_location):
    """Load a trusted locally produced full-model checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap(model: nn.Module) -> IncreAudioVisualNet:
    return model.module if isinstance(model, nn.DataParallel) else model


def evaluate(model, loader, gate_active: bool) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for data, labels_cpu in tqdm(loader, desc="eval", leave=False):
            labels = labels_cpu.to(device, non_blocking=True).long()
            visual = data[0].to(device, non_blocking=True)
            audio = data[1].to(device, non_blocking=True)
            logits = model(
                visual=visual,
                audio=audio,
                labels=labels,
                use_oracle_gate=gate_active,
            )
            loss_sum += float(
                F.cross_entropy(logits, labels, reduction="sum").item()
            )
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        raise RuntimeError("Evaluation loader is empty")
    return correct / total, loss_sum / total


def build_optimizer(model: nn.Module, args):
    raw_model = unwrap(model)
    gate_param = raw_model.oracle_class_gate
    base_params = [
        p for p in raw_model.parameters()
        if p is not gate_param and p.requires_grad
    ]
    return torch.optim.Adam(
        [
            {
                "params": base_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "name": "network",
            },
            {
                "params": [gate_param],
                "lr": args.oracle_gate_lr,
                "weight_decay": 0.0,
                "name": "class_gate",
            },
        ]
    )


def export_gate_table(
    model: IncreAudioVisualNet,
    category_encode_dict: Dict[str, int],
    output_json: str,
    output_csv: str,
    args,
    best_epoch: int,
    best_val_acc: float,
    test_acc: float,
) -> None:
    gates = model.get_oracle_gate_values().detach().cpu().numpy()
    id_to_category = {
        int(class_id): str(category_name)
        for category_name, class_id in category_encode_dict.items()
    }
    missing_ids = [
        class_id for class_id in range(args.num_classes)
        if class_id not in id_to_category
    ]
    if missing_ids:
        raise RuntimeError(
            f"Missing category names for class IDs: {missing_ids[:10]}"
        )

    payload = {
        "format_version": 1,
        "oracle_type": "train_aware_fullclass_static_class_gate",
        "not_deployable": True,
        "label_conditioned": True,
        "dataset": args.dataset,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "meta_root": args.meta_root,
        "feature_root": args.feature_root,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "fullclass_test_acc": float(test_acc),
        "gate_parameterization": "direct_projected_scalar",
        "gate_bounds": [
            float(args.oracle_gate_min),
            float(args.oracle_gate_max),
        ],
        "gate_init": float(args.oracle_gate_init),
        "gate_lr": float(args.oracle_gate_lr),
        "category_to_gate": {
            id_to_category[class_id]: float(gates[class_id])
            for class_id in range(args.num_classes)
        },
        "class_id_to_category": {
            str(class_id): id_to_category[class_id]
            for class_id in range(args.num_classes)
        },
        "class_id_to_gate": {
            str(class_id): float(gates[class_id])
            for class_id in range(args.num_classes)
        },
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "category_name", "oracle_gate"],
        )
        writer.writeheader()
        for class_id in range(args.num_classes):
            writer.writerow(
                {
                    "class_id": class_id,
                    "category_name": id_to_category[class_id],
                    "oracle_gate": float(gates[class_id]),
                }
            )


def train_fullclass(args) -> None:
    if args.class_num_per_step != args.num_classes:
        raise ValueError(
            "Full-class gate discovery requires "
            "--class_num_per_step == --num_classes"
        )
    if not 0 <= args.oracle_gate_warmup_epochs < args.max_epoches:
        raise ValueError(
            "oracle_gate_warmup_epochs must be in [0, max_epoches)"
        )

    setup_seed(args.seed)
    args.oracle_gate_mode = "learnable"
    args.oracle_gate_table_size = args.num_classes

    print(args, flush=True)
    print(f"Training start time: {datetime.now()}", flush=True)

    train_set = IcaAVELoader(
        args=args, mode="train", modality=args.modality
    )
    val_set = IcaAVELoader(
        args=args, mode="val", modality=args.modality
    )
    test_set = IcaAVELoader(
        args=args, mode="test", modality=args.modality
    )
    train_set.set_incremental_step(0)
    val_set.set_incremental_step(0)
    test_set.set_incremental_step(0)

    train_loader = DataLoader(
        train_set,
        batch_size=min(args.train_batch_size, len(train_set)),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(args.infer_batch_size, len(val_set)),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=min(args.infer_batch_size, len(test_set)),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    model: nn.Module = IncreAudioVisualNet(args, args.num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = build_optimizer(model, args)

    save_root = os.path.join("./save", args.dataset)
    metrics_root = os.path.join("./save/metrics", args.dataset)
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(metrics_root, exist_ok=True)

    best_ckpt = os.path.join(
        save_root, "fullclass_oracle_gate_best_model.pkl"
    )
    epoch_csv = os.path.join(
        metrics_root, "fullclass_gate_epoch_metrics.csv"
    )
    if os.path.exists(epoch_csv):
        os.remove(epoch_csv)

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(args.max_epoches):
        gate_active = epoch >= args.oracle_gate_warmup_epochs
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = 0

        iterator = tqdm(train_loader, desc=f"epoch {epoch}")
        for data, labels_cpu in iterator:
            labels = labels_cpu.to(
                device, non_blocking=True
            ).long()
            visual = data[0].to(device, non_blocking=True)
            audio = data[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, audio_feature, visual_feature = model(
                visual=visual,
                audio=audio,
                labels=labels,
                use_oracle_gate=gate_active,
                out_feature_before_fusion=True,
            )
            loss = ce_loss(args.num_classes, logits, labels)

            if args.instance_contrastive:
                loss = loss + args.lam_I * instance_contrastive_loss(
                    audio_feature,
                    visual_feature,
                    args.instance_contrastive_temperature,
                )
            if args.class_contrastive:
                loss = loss + args.lam_C * class_contrastive_loss(
                    audio_feature,
                    visual_feature,
                    labels,
                    args.class_contrastive_temperature,
                )
            if (
                gate_active
                and args.oracle_gate_to_guided_reg > 0
            ):
                gates = unwrap(model).get_oracle_gate_values()
                loss = loss + (
                    args.oracle_gate_to_guided_reg
                    * torch.mean(1.0 - gates)
                )

            loss.backward()
            optimizer.step()
            unwrap(model).project_oracle_gate_()

            running_loss += float(loss.item())
            running_correct += int(
                logits.argmax(dim=1).eq(labels).sum().item()
            )
            running_total += int(labels.numel())
            num_batches += 1
            iterator.set_postfix(
                loss=f"{running_loss / num_batches:.4f}",
                acc=f"{running_correct / max(running_total, 1):.4f}",
            )

        train_loss = running_loss / max(num_batches, 1)
        train_acc = running_correct / max(running_total, 1)
        val_acc, val_loss = evaluate(
            model, val_loader, gate_active=gate_active
        )

        gates = (
            unwrap(model)
            .get_oracle_gate_values()
            .detach()
            .cpu()
        )
        row = {
            "epoch": epoch,
            "gate_active": int(gate_active),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "gate_mean": float(gates.mean().item()),
            "gate_std": float(gates.std(unbiased=False).item()),
            "gate_min": float(gates.min().item()),
            "gate_max": float(gates.max().item()),
            "gate_lt_0_95_frac": float(
                (gates < 0.95).float().mean().item()
            ),
            "gate_lt_0_80_frac": float(
                (gates < 0.80).float().mean().item()
            ),
            "gate_lt_0_50_frac": float(
                (gates < 0.50).float().mean().item()
            ),
        }
        append_csv_row(epoch_csv, row.keys(), row)

        print(
            "Epoch:{:03d} train_loss:{:.6f} train_acc:{:.6f} "
            "val_loss:{:.6f} val_acc:{:.6f} "
            "gate_mean:{:.4f} gate_min:{:.4f}".format(
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                row["gate_mean"],
                row["gate_min"],
            ),
            flush=True,
        )

        # Warmup checkpoints are never selected as the O1 gate result.
        if gate_active and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(unwrap(model), best_ckpt)
            print(
                f"Saving best oracle gate model at epoch {epoch}",
                flush=True,
            )

    if best_epoch < 0 or not os.path.exists(best_ckpt):
        raise RuntimeError("No gate-active checkpoint was saved")

    best_model: IncreAudioVisualNet = load_full_model(
        best_ckpt, map_location=device
    )
    best_model = best_model.to(device)
    best_model.eval()
    test_acc, test_loss = evaluate(
        best_model, test_loader, gate_active=True
    )
    print(
        f"Best epoch={best_epoch}, "
        f"val_acc={best_val_acc:.6f}, "
        f"fullclass_test_acc={test_acc:.6f}, "
        f"test_loss={test_loss:.6f}",
        flush=True,
    )

    output_json = (
        args.oracle_gate_table_out
        or os.path.join(save_root, "oracle_gate_table.json")
    )
    output_csv = os.path.splitext(output_json)[0] + ".csv"
    export_gate_table(
        best_model,
        train_set.category_encode_dict,
        output_json,
        output_csv,
        args,
        best_epoch,
        best_val_acc,
        test_acc,
    )
    print(f"Oracle gate table JSON: {output_json}", flush=True)
    print(f"Oracle gate table CSV:  {output_csv}", flush=True)
    print(f"Training end time: {datetime.now()}", flush=True)

    if args.dataset != "AVE":
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()


def build_parser():
    parser = argparse.ArgumentParser()

    def dataset_type(value: str):
        if value in {"AVE", "ksounds"} or "VGGSound" in value:
            return value
        raise argparse.ArgumentTypeError(
            "dataset must be AVE, ksounds, or contain VGGSound"
        )

    parser.add_argument("--dataset", type=dataset_type, required=True)
    parser.add_argument(
        "--modality",
        type=str,
        default="audio-visual",
        choices=["audio-visual"],
    )
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--meta_root", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epoches", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument(
        "--class_num_per_step", type=int, default=100
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

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
        "--instance_contrastive_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--class_contrastive_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument("--lam_I", type=float, default=0.1)
    parser.add_argument("--lam_C", type=float, default=1.0)

    parser.add_argument(
        "--oracle_gate_mode", type=str, default="learnable"
    )
    parser.add_argument(
        "--oracle_gate_table_size", type=int, default=100
    )
    parser.add_argument(
        "--oracle_gate_init", type=float, default=1.0
    )
    parser.add_argument(
        "--oracle_gate_min", type=float, default=0.0
    )
    parser.add_argument(
        "--oracle_gate_max", type=float, default=1.0
    )
    parser.add_argument(
        "--oracle_gate_lr", type=float, default=1e-2
    )
    parser.add_argument(
        "--oracle_gate_warmup_epochs", type=int, default=0
    )
    parser.add_argument(
        "--oracle_gate_to_guided_reg", type=float, default=0.0
    )
    parser.add_argument(
        "--oracle_gate_table_out", type=str, default=None
    )

    parser.add_argument(
        "--z1_cm_projection_head",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--z1_cm_projection_dim", type=int, default=768
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


if __name__ == "__main__":
    train_fullclass(build_parser().parse_args())