#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-class training script for the simple A->V guidance diagnostic.

Purpose
-------
Train all classes in ONE step (e.g. 100/100) using the same model and the same
z1 contrastive loss definitions as the supplied AVCIL baseline, but without
continual-learning-only losses (KD / attention distillation), because there is
no previous model in full-class training.

Training objective
------------------
    L = L_CE + lam_I * L_instance + lam_C * L_class

where L_instance and L_class are copied exactly from the supplied baseline.

Important
---------
This script intentionally requires:
    --class_num_per_step == --num_classes
so that it cannot accidentally become an incremental experiment.
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Match the project import convention used by the supplied baseline.
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader
from model.audio_visual_model_incremental import IncreAudioVisualNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def boolean_string(s: str) -> bool:
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string: {}".format(s))
    return s == "True"


def CE_loss(num_classes: int, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Exactly the CE implementation used by the supplied baseline."""
    targets = F.one_hot(label, num_classes=num_classes)
    loss = -torch.mean(
        torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1)
    )
    return loss


def cal_contrastive_loss(
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Exactly the instance contrastive implementation used by the baseline."""
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
    """Exactly the ACTIVE class-contrastive implementation in the baseline.

    Do not replace this with the commented alternative from the original code:
    the point of this diagnostic is to reproduce the representation learning
    actually used by the current baseline, not to redesign its loss.
    """
    class_matrix = label.unsqueeze(0)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = class_matrix == label.unsqueeze(-1)
    class_matrix = class_matrix.float()

    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss = -torch.mean(
        torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)
    )
    return loss


def top_1_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target.long()).float().mean().item()


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def safe_torch_load(path: str, map_location="cpu"):
    """Whole-model loading compatible with old PyTorch and PyTorch >= 2.6."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_loader(dataset, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    if len(dataset) <= 0:
        raise RuntimeError("Dataset is empty")
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        shuffle=train,
    )


@torch.no_grad()
def evaluate(model, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for data, labels in tqdm(loader, desc="eval", leave=False):
        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        logits = model(visual=visual, audio=audio)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

    if total == 0:
        raise RuntimeError("Evaluation loader produced zero samples")
    return correct / float(total)


def train_fullclass(args, train_set, val_set) -> str:
    train_loader = make_loader(
        train_set, args.train_batch_size, args.num_workers, train=True
    )
    val_loader = make_loader(
        val_set, args.infer_batch_size, args.num_workers, train=False
    )

    model = IncreAudioVisualNet(args, args.num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ckpt_dir = os.path.join(".", "save", args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "step_0_best_model.pkl")

    best_val = -1.0
    best_epoch = -1

    if args.lr_decay:
        # The supplied incremental baseline only applies its manual LR decay
        # when step > 0. Full-class training corresponds to step 0, so we do
        # NOT decay here. Keep the flag only for command-line compatibility.
        print(
            "[INFO] --lr_decay=True was provided, but no LR decay is applied: "
            "full-class training corresponds to baseline step 0.",
            flush=True,
        )

    for epoch in range(args.max_epoches):
        model.train()

        sum_total = 0.0
        sum_ce = 0.0
        sum_i = 0.0
        sum_c = 0.0
        num_batches = 0

        iterator = tqdm(train_loader, desc="epoch {:03d}".format(epoch))
        for data, labels in iterator:
            visual = data[0].to(device, non_blocking=True)
            audio = data[1].to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            out, audio_feature, visual_feature = model(
                visual=visual,
                audio=audio,
                out_feature_before_fusion=True,
            )

            loss_ce = CE_loss(args.num_classes, out, labels)
            loss_i = out.sum() * 0.0
            loss_c = out.sum() * 0.0

            if args.instance_contrastive:
                loss_i = cal_contrastive_loss(
                    audio_feature,
                    visual_feature,
                    temperature=args.instance_contrastive_temperature,
                )

            if args.class_contrastive:
                loss_c = class_contrastive_loss(
                    audio_feature,
                    visual_feature,
                    labels,
                    temperature=args.class_contrastive_temperature,
                )

            loss = loss_ce
            if args.instance_contrastive:
                loss = loss + args.lam_I * loss_i
            if args.class_contrastive:
                loss = loss + args.lam_C * loss_c

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sum_total += float(loss.item())
            sum_ce += float(loss_ce.item())
            sum_i += float(loss_i.item())
            sum_c += float(loss_c.item())
            num_batches += 1

            iterator.set_postfix(
                loss="{:.4f}".format(sum_total / num_batches),
                ce="{:.4f}".format(sum_ce / num_batches),
                inst="{:.4f}".format(sum_i / num_batches),
                cls="{:.4f}".format(sum_c / num_batches),
            )

        if num_batches == 0:
            raise RuntimeError(
                "Training loader produced zero batches. Check batch size/drop_last."
            )

        val_acc = evaluate(model, val_loader)
        print(
            "Epoch:{:03d} train_loss:{:.6f} CE:{:.6f} I:{:.6f} C:{:.6f} val_res:{:.6f}".format(
                epoch,
                sum_total / num_batches,
                sum_ce / num_batches,
                sum_i / num_batches,
                sum_c / num_batches,
                val_acc,
            ),
            flush=True,
        )

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            print(
                "Saving best model at Epoch {} (val_res={:.6f})".format(
                    epoch, val_acc
                ),
                flush=True,
            )
            torch.save(unwrap_model(model), ckpt_path)

    if best_epoch < 0 or not os.path.exists(ckpt_path):
        raise RuntimeError("No best checkpoint was saved")

    print(
        "Training finished. best_epoch={} best_val_res={:.6f}".format(
            best_epoch, best_val
        ),
        flush=True,
    )
    print("Best checkpoint: {}".format(ckpt_path), flush=True)
    return ckpt_path


def dataset_type(s: str):
    if s in ["AVE", "ksounds"]:
        return s
    if "VGGSound" in s:
        return s
    raise argparse.ArgumentTypeError(
        "dataset must be 'AVE', 'ksounds', or contain 'VGGSound'"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-step full-class CE + baseline z1 contrastive training"
    )

    parser.add_argument("--dataset", type=dataset_type, required=True)
    parser.add_argument(
        "--modality", type=str, default="audio-visual", choices=["audio-visual"]
    )
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--meta_root", type=str, default=None)

    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--infer_batch_size", type=int, default=32)
    # Kept for compatibility with project dataloader args; unused here.
    parser.add_argument("--exemplar_batch_size", type=int, default=128)
    parser.add_argument("--memory_size", type=int, default=500)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epoches", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--class_num_per_step", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[100], nargs="+")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--instance_contrastive", action="store_true", default=False)
    parser.add_argument("--class_contrastive", action="store_true", default=False)
    parser.add_argument("--instance_contrastive_temperature", type=float, default=0.05)
    parser.add_argument("--class_contrastive_temperature", type=float, default=0.05)
    parser.add_argument("--lam_I", type=float, default=0.1)
    parser.add_argument("--lam_C", type=float, default=1.0)

    return parser


def validate_args(args) -> None:
    if args.class_num_per_step != args.num_classes:
        raise ValueError(
            "This is intentionally a one-step full-class script: "
            "--class_num_per_step must equal --num_classes."
        )
    if args.num_classes < 2:
        raise ValueError("--num_classes must be >= 2")
    if args.train_batch_size <= 0 or args.infer_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.max_epoches <= 0:
        raise ValueError("--max_epoches must be positive")
    if args.instance_contrastive_temperature <= 0:
        raise ValueError("--instance_contrastive_temperature must be positive")
    if args.class_contrastive_temperature <= 0:
        raise ValueError("--class_contrastive_temperature must be positive")
    if args.lam_I < 0 or args.lam_C < 0:
        raise ValueError("contrastive loss weights must be non-negative")


def close_if_possible(dataset) -> None:
    close_fn = getattr(dataset, "close_visual_features_h5", None)
    if callable(close_fn):
        close_fn()


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    setup_seed(args.seed)

    print(args, flush=True)
    print("Device: {}".format(device), flush=True)
    print("Training start time: {}".format(datetime.now()), flush=True)
    print(
        "Objective: CE{}{}".format(
            " + {:.4g}*instance".format(args.lam_I)
            if args.instance_contrastive
            else "",
            " + {:.4g}*class".format(args.lam_C)
            if args.class_contrastive
            else "",
        ),
        flush=True,
    )

    train_set = IcaAVELoader(args=args, mode="train", modality=args.modality)
    val_set = IcaAVELoader(args=args, mode="val", modality=args.modality)
    test_set = IcaAVELoader(args=args, mode="test", modality=args.modality)

    # Full-class means the sole incremental index is 0 and it contains all classes.
    train_set.set_incremental_step(0)
    val_set.set_incremental_step(0)
    test_set.set_incremental_step(0)

    try:
        ckpt_path = train_fullclass(args, train_set, val_set)

        test_loader = make_loader(
            test_set, args.infer_batch_size, args.num_workers, train=False
        )
        best_model = safe_torch_load(ckpt_path, map_location="cpu")
        best_model = best_model.to(device)
        test_acc = evaluate(best_model, test_loader)
        print("Best-checkpoint test_res:{:.6f}".format(test_acc), flush=True)
    finally:
        if args.dataset != "AVE":
            close_if_possible(train_set)
            close_if_possible(val_set)
            close_if_possible(test_set)

    print("Training end time: {}".format(datetime.now()), flush=True)


if __name__ == "__main__":
    main()