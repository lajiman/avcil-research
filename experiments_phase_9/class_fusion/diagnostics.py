"""Phase-8-style CSV diagnostics, now recording fusion decisions per class."""

import csv
from pathlib import Path

import torch

# P8 来源：experiments_phase_8_rdcrosssdc_modular/rd_crosssdc/diagnostics.py。

# [P8 逻辑沿用] 同名 CSV 追加函数；改用 pathlib 和显式 UTF-8 编码。
def append_csv_row(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# [P9 新增] 沿用 P8 的逐类 CSV 诊断形式，但字段改为可靠性/gate/eta/版本；
# 这些是融合状态，不是原 save_dynamic_weights 记录的 Trust/Need loss 权重。
def save_gate_snapshot(path, step, epoch, gate, version, id_to_category, snapshot=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["counts", "scored_counts", "reliability_a", "reliability_v", "valid", "eta", "previous_gate", "raw_gate"]
    header = ["step", "after_epoch", "active_from_epoch", "gate_version", "class_id", "category_name", *keys, "audio_gate", "visual_gate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for c, value in enumerate(gate.detach().cpu().tolist()):
            row = {"step": step, "after_epoch": epoch, "active_from_epoch": epoch + 1,
                   "gate_version": version, "class_id": c, "category_name": id_to_category.get(c, str(c)),
                   "audio_gate": value, "visual_gate": 1.0 - value}
            row.update({key: snapshot[key][c].item() if snapshot is not None else "" for key in keys})
            writer.writerow(row)


# [P9 新增] 与旧 gate CSV 分开，旧实验的列及路径保持不变。
def save_prototype_snapshot(path, step, epoch, diagnostics):
    """Record scalar decisions plus the current-space prototypes used for scoring."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["birth_count", "anchor_count", "history_used", "prior_strength_effective",
            "prototype_beta", "drift_audio_norm", "drift_visual_norm",
            "drift_audio_dispersion", "drift_visual_dispersion"]
    header = ["step", "after_epoch", "active_from_epoch", "class_id", *keys, "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for c, reason in enumerate(diagnostics["reason"]):
            row = {"step": step, "after_epoch": epoch, "active_from_epoch": epoch + 1,
                   "class_id": c, "reason": reason}
            row.update({key: diagnostics[key][c].item() for key in keys})
            writer.writerow(row)
    torch.save({"step": step, "after_epoch": epoch, "active_from_epoch": epoch + 1,
                **diagnostics}, path.with_suffix(".pt"))
