"""Evaluation and per-class metrics, separated from the training method."""

import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


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
        f1[class_id] = safe_div(2.0 * p * r, p + r)

    return {
        "tp": tp.numpy(),
        "fp": fp.numpy(),
        "fn": fn.numpy(),
        "support": support.numpy(),
        "precision": precision.numpy(),
        "recall": recall.numpy(),
        "f1": f1.numpy(),
    }


def save_json(obj: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2)


def append_csv_rows(path: str, header: List[str], rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def detailed_test(
    args,
    step: int,
    test_data_set,
    task_best_acc_list: List[float],
    metrics_root: str,
    metrics_state: Dict[str, object],
    id_to_category: Dict[int, str],
    checkpoint_path: str,
    device: torch.device,
):
    print("=====================================")
    print("Start testing...")
    print("=====================================")

    model = torch.load(checkpoint_path)
    model.to(device)
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
        for test_data, test_labels in tqdm(test_loader):
            test_visual = test_data[0].to(device)
            test_audio = test_data[1].to(device)
            logits = model(visual=test_visual, audio=test_audio)
            logits_list.append(logits.detach().cpu())
            labels_list.append(test_labels.detach().cpu().long())

    all_logits = torch.cat(logits_list, dim=0)  # B * C --> E(poch) * C
    all_labels = torch.cat(labels_list, dim=0)
    pred = all_logits.argmax(dim=1).long()
    overall_acc = (pred == all_labels).float().mean().item()
    print("Incremental step {} Testing res (overall acc): {:.6f}".format(step, overall_acc))

    classes_per_step = args.class_num_per_step
    old_task_acc_list = []
    current_step_acc = None
    for task_id in range(step + 1):
        lo = task_id * classes_per_step
        hi = (task_id + 1) * classes_per_step
        mask = (all_labels >= lo) & (all_labels < hi)
        task_acc = 0.0 if mask.sum().item() == 0 else (
            (pred[mask] == all_labels[mask]).float().mean().item()
        )
        if task_id == step:
            current_step_acc = task_acc
        else:
            old_task_acc_list.append(task_acc)

    if step > 0:
        forgetting = float(
            np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        )
        print("task-level forgetting: {:.6f}".format(forgetting))
        for task_id in range(len(task_best_acc_list)):
            task_best_acc_list[task_id] = max(
                task_best_acc_list[task_id], old_task_acc_list[task_id]
            )
    else:
        forgetting = None
    task_best_acc_list.append(float(current_step_acc))

    num_seen_classes = (step + 1) * classes_per_step
    stats = compute_per_class_prf(all_labels, pred, num_seen_classes)
    best_f1 = metrics_state.get("best_f1", {})
    first_seen = metrics_state.get("first_seen_step", {})

    rows = []
    for class_id in range(num_seen_classes):
        key = str(class_id)
        if key not in first_seen:
            first_seen[key] = step

        f1_c = float(stats["f1"][class_id])
        best_before = float(best_f1[key]) if key in best_f1 else None
        forget_f1 = best_before - f1_c if best_before is not None else 0.0
        new_best = f1_c if best_before is None else max(best_before, f1_c)
        best_f1[key] = new_best

        rows.append({
            "step": step,
            "class_id": class_id,
            "category_name": id_to_category.get(class_id, "class_{}".format(class_id)),
            "first_seen_step": int(first_seen[key]),
            "support": int(stats["support"][class_id]),
            "tp": int(stats["tp"][class_id]),
            "fp": int(stats["fp"][class_id]),
            "fn": int(stats["fn"][class_id]),
            "precision": float(stats["precision"][class_id]),
            "recall": float(stats["recall"][class_id]),
            "f1": f1_c,
            "best_f1": float(new_best),
            "forget_f1": float(forget_f1),
            "forgetting": float(forgetting) if forgetting is not None else 0.0,
            "overall_acc": float(overall_acc),
        })

    header = [
        "step", "class_id", "category_name", "first_seen_step", "support",
        "tp", "fp", "fn", "precision", "recall", "f1", "best_f1",
        "forget_f1", "forgetting", "overall_acc",
    ]
    append_csv_rows(os.path.join(metrics_root, "per_class_metrics.csv"), header, rows)

    metrics_state["best_f1"] = best_f1
    metrics_state["first_seen_step"] = first_seen
    save_json(metrics_state, os.path.join(metrics_root, "per_class_state.json"))
    return forgetting
