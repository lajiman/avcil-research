"""Small shared IO and matching helpers; no training state is modified."""

import csv
import hashlib
import json
import math
from pathlib import Path


DEFAULTS = {
    "dataset": "VGGSound_random_balance_crosssdc_z1_seed42", "modality": "audio-visual",
    "num_classes": 100, "class_num_per_step": 10, "max_epoches": 200, "memory_size": 500,
    "train_batch_size": 128, "exemplar_batch_size": 128, "infer_batch_size": 32,
    "num_workers": 0, "lr": 0.001, "weight_decay": 0.0001, "lr_decay": False,
    "milestones": [100], "lam": 0.5, "lam_I": 0.1, "lam_C": 1.0,
    "instance_contrastive": True, "class_contrastive": True, "attn_score_distil": True,
    "instance_contrastive_temperature": 0.05, "class_contrastive_temperature": 0.05,
    "fusion_mode": "periodic", "fusion_prototype_mode": "fresh",
    "fusion_update_rule": "sample_aware", "fusion_classifier": "linear",
    "fusion_hidden_dim": 768, "fusion_chunk_size": 16, "fusion_warmup_epochs": 40,
    "fusion_update_interval": 40, "fusion_eta_max": 0.5, "fusion_n_ref": 10.0,
    "fusion_temperature": 0.1, "fusion_min_samples": 2, "fusion_batch_size": 128,
    "fusion_update_first_step": False, "prototype_prior_strength": 10.0,
    "record_cl_history": True, "cl_history_interval": 40,
}

# All non-method training settings must match before a seed-wise comparison.
PROTOCOL_KEYS = (
    "dataset", "modality", "num_classes", "class_num_per_step", "max_epoches", "memory_size",
    "train_batch_size", "exemplar_batch_size", "infer_batch_size", "num_workers", "lr",
    "weight_decay", "lr_decay", "milestones", "lam", "lam_I", "lam_C", "instance_contrastive",
    "class_contrastive", "attn_score_distil", "instance_contrastive_temperature",
    "class_contrastive_temperature", "fusion_classifier", "fusion_hidden_dim", "fusion_chunk_size",
)
METHOD_KEYS = tuple(k for k in DEFAULTS if k.startswith("fusion_") or k == "prototype_prior_strength")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def scalar(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    text_keys = {"run_id", "reference_id", "sample_id", "video_id", "job_id", "method_id",
                 "protocol_id", "category_name", "method", "reason", "status", "events"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: v if k in text_keys or "path" in k else scalar(v) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def number(value, default=float("nan")):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_stat_fingerprint(path):
    path = Path(path).resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint(obj):
    return hashlib.sha256(json.dumps(clean_json(obj), sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def method_label(config):
    cfg = {**DEFAULTS, **config}
    if cfg["fusion_mode"] == "uniform":
        return "uniform"
    prefix = "prototype_bank" if cfg["fusion_prototype_mode"] == "history_bank" else "periodic"
    rule = cfg["fusion_update_rule"]
    return prefix + "_" + ("smooth" if prefix == "prototype_bank" and rule == "sample_aware" else rule)


def baseline_candidates(run, runs):
    return [candidate for candidate in runs
            if candidate["config"].get("fusion_mode") == "uniform"
            and candidate.get("protocol_id") == run.get("protocol_id")
            and candidate.get("seed") == run.get("seed")]


def resolve_baseline(run, runs):
    candidates = baseline_candidates(run, runs)
    return candidates[0] if len(candidates) == 1 else None
