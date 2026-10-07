"""Small, explicit CSV diagnostics for RD-CrossSDC experiments."""

import csv
import os
from typing import Dict, Iterable, List

import torch


def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_static_bank(   # step level
    path: str,
    step: int,
    counts: torch.Tensor,
    reliability_a: torch.Tensor,
    reliability_v: torch.Tensor,
    trust_a: torch.Tensor,
    trust_v: torch.Tensor,
    id_to_category: Dict[int, str],
    query_counts: torch.Tensor = None,
    prototype_policy: str = "memory",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        "step",
        "class_id",
        "category_name",
        "prototype_count",
        "reliability_a_from_v",
        "reliability_v_from_a",
        "trust_a_from_v",
        "trust_v_from_a",
    ]
    # A's existing CSV is unchanged. B/C distinguish the larger prototype
    # support from the reduced replay count used for Trust/shrinkage.
    if query_counts is not None:
        header.extend(["trust_query_count", "prototype_policy"])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for class_id in range(int(counts.numel())): # len(counts)
            row = {
                "step": step,
                "class_id": class_id,
                "category_name": id_to_category.get(class_id, "class_{}".format(class_id)), # will return "class_i" if there isn't the category_name
                "prototype_count": float(counts[class_id].item()),
                "reliability_a_from_v": float(reliability_a[class_id].item()),
                "reliability_v_from_a": float(reliability_v[class_id].item()),
                "trust_a_from_v": float(trust_a[class_id].item()),
                "trust_v_from_a": float(trust_v[class_id].item()),
            }
            if query_counts is not None:
                row["trust_query_count"] = float(query_counts[class_id].item())
                row["prototype_policy"] = prototype_policy
            writer.writerow(row)


def save_dynamic_weights(   # step level + epoch level
    path: str,
    step: int,
    epoch: int,
    snapshot: Dict[str, torch.Tensor],
    id_to_category: Dict[int, str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        "step",
        "epoch",
        "class_id",
        "category_name",
        "trust_a_from_v",
        "trust_v_from_a",
        "need_a_from_v",
        "need_v_from_a",
        "class_weight_a_from_v",    # the meaning of those indexes depends on the implementation
        "class_weight_v_from_a",
        "cmr_weight_a_from_v",
        "cmr_weight_v_from_a",
    ]
    num_classes = int(snapshot["trust_a_from_v"].numel())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for class_id in range(num_classes):
            row = {
                "step": step,
                "epoch": epoch,
                "class_id": class_id,
                "category_name": id_to_category.get(class_id, "class_{}".format(class_id)),
            }
            for key in header[4:]:
                row[key] = float(snapshot[key][class_id].item())
            writer.writerow(row)
