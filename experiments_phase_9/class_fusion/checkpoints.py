"""Explicit state-dict checkpoints: model weights and gates always travel together."""

from pathlib import Path
from types import SimpleNamespace
import hashlib

import torch

from model.audio_visual_model_incremental_class_fusion import ClassFusionAudioVisualNet
from .fusion_method import preserve_rng_state, unwrap_model

# [P9 新增：整个模块] 替代 P8/train_incremental_rd_crosssdc_modular.py 的整模型
# torch.save/torch.load 流程；显式保存 state_dict、gate、配置和观测状态，格式不兼容 P8。
# P8 指 experiments_phase_8_rdcrosssdc_modular。

def save_checkpoint(path, model, args, step, epoch, val_acc, fusion_state=None, cl_history=None):
    net = unwrap_model(model)
    payload = {
        "format_version": 1,
        "model_args": vars(args).copy(),
        "num_classes": net.num_classes,
        "state_dict": {key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
        "step": step, "epoch": epoch, "val_acc": val_acc,
        "fusion_state": fusion_state,
        "cl_history": cl_history,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, return_metadata=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Expected a phase-9 checkpoint, not a phase-8 pickled model")
    # Reconstructing a state-dict model must not advance training's RNG stream.
    with preserve_rng_state():
        model = ClassFusionAudioVisualNet(SimpleNamespace(**payload["model_args"]), payload["num_classes"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if return_metadata:
        metadata = {key: payload[key] for key in ("format_version", "step", "epoch", "val_acc", "num_classes")}
        metadata["checkpoint_name"] = Path(path).name
        metadata["checkpoint_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return model, metadata
    return model
