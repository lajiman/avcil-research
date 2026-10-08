"""Read-only checkpoint inference for the phase-9 analysis pipeline.

One persistent process owns one CUDA device and one audio-feature dictionary.
All artifacts are written under results/cache; training outputs are never
modified. Gates are evaluated as saved or at fixed, predeclared controls.
Neither validation nor test labels are used to select or estimate a gate.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import zipfile
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments_phase_9.dataloader_ours import IcaAVELoader
from model.audio_visual_model_incremental_class_fusion import (
    ClassFusionAudioVisualNet, class_conditional_logits,
)


INFERENCE_SCHEMA_VERSION = 1
_META_NAMES = ("category_encode_dict.npy", "all_id_category_dict.npy", "all_classId_vid_dict.npy")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _hashed_stat(path):
    before = _stat(path)
    digest = _sha256(path)
    if _stat(path) != before:
        raise RuntimeError(f"Input changed while fingerprinting: {path}")
    return {**before, "sha256": digest}


def _assert_inputs_stable(provenance, checkpoint_before=None):
    """A transfer/training writer must not create a mixed-version result."""
    items = [provenance["checkpoint"], *provenance["metadata"], *provenance["features"]]
    for item in items:
        expected = {key: item[key] for key in ("path", "size", "mtime_ns")}
        if _stat(item["path"]) != expected:
            raise RuntimeError(f"Input changed during checkpoint analysis: {item['path']}")
    if checkpoint_before is not None:
        recorded = {key: provenance["checkpoint"][key] for key in ("path", "size", "mtime_ns")}
        if recorded != checkpoint_before:
            raise RuntimeError("Checkpoint changed between loading and fingerprinting")


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = Path(path).with_name(Path(path).name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _safe_json(value):
    """Keep summaries portable JSON, including invalid observed quantities."""
    if isinstance(value, torch.Tensor):
        return _safe_json(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _resolve(path, phase_root):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(phase_root) / path).resolve()


def _source_version():
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root / "model/audio_visual_model_incremental_class_fusion.py",
             root / "model/audio_visual_model_incremental.py", root / "experiments_phase_9/dataloader_ours.py"]
    return {str(path.relative_to(root)): _sha256(path) for path in files}


def _fingerprint(checkpoint, args, manifest, device):
    feature_root, meta_root = Path(args.feature_root), Path(args.meta_root)
    features = [feature_root / "audio_pretrained_feature/audio_pretrained_feature_dict.npy",
                feature_root / ("visual_pretrained_feature_dict.npy" if args.dataset == "AVE" else "visual_features.h5")]
    provenance = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "checkpoint": _hashed_stat(checkpoint),
        "metadata": [_hashed_stat(meta_root / name) for name in _META_NAMES],
        # Hashing a multi-hundred-GB feature file for every checkpoint is not
        # appropriate; path/size/mtime are an explicit cache assumption.
        "features": [_stat(path) for path in features],
        "source": _source_version(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "options": {key: manifest["options"].get(key) for key in
                    ("batch_size", "num_workers", "splits", "torch_threads")},
        "dataset": args.dataset,
    }
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, provenance


def _validate_payload(payload, job, run):
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Expected a phase-9 format_version=1 state-dict checkpoint")
    if not isinstance(payload.get("model_args"), dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint has no model_args/state_dict")
    args = payload["model_args"]
    if int(payload["step"]) != int(job["step"]):
        raise ValueError("Checkpoint step disagrees with its manifest job")
    width = int(args["class_num_per_step"])
    classes = int(payload["num_classes"])
    if width <= 0 or classes != (int(job["step"]) + 1) * width or classes > int(args["num_classes"]):
        raise ValueError("Checkpoint seen-class count does not match task/class order")
    # Relocation overrides intentionally do not require original absolute paths.
    defaults = {"fusion_prototype_mode": "fresh", "fusion_classifier": "linear"}
    for key in ("dataset", "seed", "class_num_per_step", "num_classes", "modality",
                "fusion_mode", "fusion_classifier", "fusion_update_rule", "fusion_prototype_mode",
                "lam", "lam_I", "lam_C", "instance_contrastive", "class_contrastive", "attn_score_distil",
                "instance_contrastive_temperature", "class_contrastive_temperature", "memory_size",
                "fusion_temperature", "fusion_eta_max", "fusion_n_ref", "fusion_min_samples"):
        if key in run.get("config", {}) and run["config"][key] != args.get(key, defaults.get(key)):
            raise ValueError(f"Checkpoint/config mismatch for {key}")
    gate = payload["state_dict"].get("fusion_gate")
    if not isinstance(gate, torch.Tensor) or gate.shape != (classes,) or not torch.isfinite(gate).all():
        raise ValueError("Missing or nonfinite/mis-sized checkpoint gate")
    if ((gate < 0) | (gate > 1)).any():
        raise ValueError("Checkpoint gate is outside [0, 1]")
    return SimpleNamespace(**args)


class AudioFeatureCache:
    """Keep exactly one feature root resident per worker, shared across splits."""

    def __init__(self):
        self.key = None
        self.features = None

    def get(self, feature_root):
        path = Path(feature_root) / "audio_pretrained_feature/audio_pretrained_feature_dict.npy"
        key = _stat(path)
        if key != self.key:
            self.features = None
            gc.collect()
            self.features = np.load(path, allow_pickle=True).item()
            self.key = key
        return self.features


def fixed_gate_logits(model, audio, visual):
    """Evaluate four declared gates from one branch extraction, without labels.

    The two extreme gates are shared-head ablations, not separately trained
    unimodal models. The visual feature remains audio-attention conditioned.
    MLP heads must use candidate-wise nonlinear fusion, never linear algebra.
    """
    gate = model.fusion_gate
    args = (audio, visual, model.classifier)
    if isinstance(model.classifier, nn.Linear):
        uniform = model.classifier(audio + visual)
        difference = F.linear(audio - visual, model.classifier.weight, bias=None)
        return {"logits_actual": uniform + difference * (2 * gate - 1),
                "logits_uniform": uniform, "logits_audio": uniform + difference,
                "logits_visual": uniform - difference, "d": difference}
    return {name: class_conditional_logits(*args, value, model.fusion_chunk_size)
            for name, value in (("logits_actual", gate), ("logits_uniform", torch.full_like(gate, 0.5)),
                                ("logits_audio", torch.ones_like(gate)), ("logits_visual", torch.zeros_like(gate)))}


@torch.inference_mode()
def infer_split(model, args, split, device, batch_size, num_workers, shared_audio):
    if split not in ("test", "val"):
        raise ValueError("Analysis inference supports only val and test; no training or replay datasets")
    dataset = IcaAVELoader(args, split, args.modality, (model.num_classes // args.class_num_per_step) - 1,
                           shared_audio_features=shared_audio)
    try:
        ids = [str(item) for item in dataset.all_current_data_vids]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{split} needs a nonempty, unique sample cohort")
        expected_ids = [str(item) for c in range(model.num_classes) for item in dict.fromkeys(dataset._get_class_vids(c))]
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
        columns = {}
        max_error = 0.0
        model.eval()
        for batch_index, (data, labels) in enumerate(loader):
            visual, audio = data[0].to(device), data[1].to(device)
            a, v, _, _ = model.extract_branch_features(visual, audio)
            logits = fixed_gate_logits(model, a, v)
            if batch_index == 0:
                expected = model(visual=visual, audio=audio)
                torch.testing.assert_close(logits["logits_actual"], expected, rtol=2e-5, atol=2e-5)
                max_error = float((logits["logits_actual"] - expected).abs().max())
            logits.update(labels=labels, audio_norm=a.norm(dim=1), visual_norm=v.norm(dim=1))
            for name, tensor in logits.items():
                if name != "labels" and not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite {name} in {split} batch {batch_index}")
                columns.setdefault(name, []).append(tensor.detach().cpu().numpy())
        arrays = {key: np.concatenate(value, axis=0) for key, value in columns.items()}
        arrays.update(ids=np.asarray(ids, dtype=str), gate=model.fusion_gate.cpu().numpy().copy(),
                      class_ids=np.arange(model.num_classes, dtype=np.int64))
        arrays["labels"] = arrays["labels"].astype(np.int64, copy=False)
        if arrays["labels"].shape != (len(ids),) or (arrays["labels"] < 0).any() or (arrays["labels"] >= model.num_classes).any():
            raise ValueError("Inference sample alignment or label range is invalid")
        diagnostics = {"num_samples": len(ids), "num_classes": model.num_classes,
                       "expected_unique_samples": len(set(expected_ids)),
                       "missing_feature_pairs": len(set(expected_ids) - set(ids)),
                       "forward_max_absolute_error": max_error,
                       "actual_accuracy": float(np.mean(arrays["logits_actual"].argmax(1) == arrays["labels"])),
                       "cohort_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()}
        return arrays, diagnostics
    finally:
        dataset.close_visual_features_h5()


def _npz(path, arrays):
    temporary = Path(path).with_name(Path(path).name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _cache_valid(directory, fingerprint, splits):
    try:
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("fingerprint") != fingerprint or meta.get("fingerprint") != fingerprint:
            return False
        for split in splits:
            with np.load(directory / f"{split}.npz", allow_pickle=False) as data:
                n, c = len(data["ids"]), int(meta["num_classes"])
                if data["ids"].dtype.kind != "U" or len(set(data["ids"].tolist())) != n or data["labels"].shape != (n,):
                    return False
                if not np.array_equal(data["class_ids"], np.arange(c)) or data["gate"].shape != (c,):
                    return False
                for name in ("actual", "uniform", "audio", "visual"):
                    logits = data["logits_" + name]
                    if logits.shape != (n, c) or not np.isfinite(logits).all():
                        return False
        return True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _bank_summary(bank):
    if not isinstance(bank, dict):
        return None
    anchors = bank.get("anchors", {})
    return _safe_json({"schema_version": bank.get("schema_version"), "prepared_step": bank.get("prepared_step"),
                       "num_classes": bank.get("num_classes"), "tensor_nbytes": _tensor_bytes(bank),
                       "birth_count": bank.get("birth_count"), "anchor_count": len(anchors.get("sample_ids", [])),
                       "valid_anchor_count": int(anchors["valid"].sum()) if "valid" in anchors else None,
                       "sources": bank.get("sources", [])})


def _vectors(container, length, prefix="", sample=False):
    """Select explicit class/sample fields; never dump embeddings or scores."""
    result = {}
    for key, value in container.items():
        is_sample = key.startswith("sample_") or key == "query_valid"
        if isinstance(value, dict):
            result.update(_vectors(value, length, prefix + key + "_", sample=sample))
        elif isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel() == length and is_sample == sample:
            result[prefix + key] = value.cpu().tolist()
    return result


def export_checkpoint_history(directory, payload):
    """Export embedded observation-only memory diagnostics, never test metrics."""
    history = payload.get("cl_history")
    if not isinstance(history, dict) or not isinstance(history.get("reference"), dict) or not isinstance(history.get("observation"), dict):
        return {"present": False}
    reference, observation = history["reference"], history["observation"]
    if reference.get("reference_id") != observation.get("reference_id"):
        raise ValueError("Checkpoint CL reference and observation IDs disagree")
    if int(observation["step"]) != int(payload["step"]) or int(observation["epoch"]) != int(payload["epoch"]):
        raise ValueError("Checkpoint CL observation step/epoch does not describe saved model")
    state = payload.get("state_dict", {})
    if "fusion_gate" in state and not torch.equal(observation["audio_gate"], state["fusion_gate"]):
        raise ValueError("Checkpoint CL observation gate does not describe saved model")
    if "fusion_version" in state and int(observation["gate_version"]) != int(state["fusion_version"]):
        raise ValueError("Checkpoint CL observation gate version does not describe saved model")
    nclass = int(reference["num_old_classes"])
    context = {"scope": "fixed_old_training_memory", "step": int(payload["step"]),
               "epoch": int(payload["epoch"]), "reference_id": reference["reference_id"]}
    fields = {"memory_count": reference["memory_counts"].tolist(), "teacher_gate": reference["teacher_gate"].tolist(),
              "student_gate": observation["audio_gate"][:nclass].tolist()}
    fields.update(_vectors(reference["teacher_reliability"], nclass, "teacher_"))
    fields.update(_vectors(reference["teacher_classification"], nclass, "teacher_classification_"))
    fields.update(_vectors(observation, nclass, "student_"))
    _csv(directory / "history_classes.csv", [{**context, "class_id": c, **{key: values[c] for key, values in fields.items()}}
                                               for c in range(nclass)])
    mask = torch.as_tensor(reference["included_mask"], dtype=torch.bool).tolist()
    labels = torch.as_tensor(reference["sample_labels"]).tolist()
    ids = reference["sample_ids"]
    if len(ids) != len(mask) or len(labels) != len(mask):
        raise ValueError("Checkpoint history sample IDs/mask/labels do not align")
    if len(set(ids)) != len(ids) or any(label < 0 or label >= nclass for label in labels):
        raise ValueError("Checkpoint history must contain unique old-class memory IDs only")
    cohort = [(str(vid), int(label)) for vid, label, included in zip(ids, labels, mask) if included]
    fields = _vectors(reference["teacher_reliability"], len(cohort), "teacher_", sample=True)
    fields.update(_vectors(reference["teacher_classification"], len(cohort), "teacher_classification_", sample=True))
    fields.update(_vectors(observation, len(cohort), "student_", sample=True))
    _csv(directory / "history_samples.csv", [{**context, "id": vid, "label": label,
                                             **{key: values[index] for key, values in fields.items()}}
                                            for index, (vid, label) in enumerate(cohort)])
    return {"present": True, **context, "num_old_classes": nclass, "num_reference_samples": len(ids),
            "num_included_samples": len(cohort), "teacher_source": _safe_json(reference.get("teacher_source"))}


def run_worker(manifest_path, rank=0, world_size=1, device="cpu"):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid manifest version or worker rank/world size")
    options = manifest["options"]
    splits = options.get("splits", ["test", "val"])
    if not splits or len(splits) != len(set(splits)) or any(split not in ("test", "val") for split in splits):
        raise ValueError("splits must be a unique nonempty subset of test,val")
    if int(options.get("batch_size", 32)) < 1 or int(options.get("num_workers", 0)) < 0:
        raise ValueError("Invalid inference batch size or worker count")
    threads = int(options.get("torch_threads", 6))
    if threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(threads)
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() <= rank:
            raise ValueError(f"Worker {rank} needs a visible CUDA device at index {rank}")
        target = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(target)
    elif device == "cpu":
        target = torch.device("cpu")
    else:
        raise ValueError("device must be cpu or cuda")
    output = Path(manifest["output_root"]).resolve()
    runs = {run["run_id"]: run for run in manifest["runs"]}
    if len(runs) != len(manifest["runs"]) or len({job["job_id"] for job in manifest["jobs"]}) != len(manifest["jobs"]):
        raise ValueError("Manifest run and job IDs must be unique")
    jobs = [job for index, job in enumerate(manifest["jobs"]) if index % world_size == rank]
    audio_cache = AudioFeatureCache()
    summary = {"rank": rank, "world_size": world_size, "device": str(target), "jobs": [], "failed": 0}
    for job in jobs:
        job_id = job["job_id"]
        if not job_id or Path(job_id).name != job_id or job_id in (".", "..") or "\\" in job_id:
            raise ValueError("job_id must be a safe single directory name")
        directory = output / "cache" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        record = {key: job[key] for key in ("job_id", "run_id", "step", "kind", "checkpoint")}
        record["invocation_id"] = manifest.get("invocation_id")
        started = time.monotonic()
        model = payload = arrays = None
        try:
            run = runs[job["run_id"]]
            checkpoint = Path(job["checkpoint"]).resolve()
            checkpoint_before = _stat(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if _stat(checkpoint) != checkpoint_before:
                raise RuntimeError("Checkpoint changed while loading; retry after transfer/training finishes")
            args = _validate_payload(payload, job, run)
            args.feature_root = str(_resolve(options.get("feature_root") or args.feature_root, manifest["phase_root"]))
            args.meta_root = str(_resolve(options.get("meta_root") or args.meta_root, manifest["phase_root"]))
            fingerprint, provenance = _fingerprint(checkpoint, args, manifest, target)
            _assert_inputs_stable(provenance, checkpoint_before)
            record["fingerprint"] = fingerprint
            if not options.get("force", False) and _cache_valid(directory, fingerprint, splits):
                _assert_inputs_stable(provenance, checkpoint_before)
                record.update(status="complete", cached=True)
            else:
                _json(directory / "status.json", {**record, "status": "running", "splits": splits})
                model = ClassFusionAudioVisualNet(args, int(payload["num_classes"]))
                model.load_state_dict(payload["state_dict"], strict=True)
                model.to(target).eval().requires_grad_(False)
                diagnostics = {}
                shared = None  # Release the previous root before replacing the worker cache.
                shared = audio_cache.get(args.feature_root)
                for split in splits:
                    arrays, diagnostics[split] = infer_split(model, args, split, target,
                        int(options.get("batch_size", 32)), int(options.get("num_workers", 0)), shared)
                    if split == "val" and payload.get("val_acc") is not None:
                        diagnostics[split]["accuracy_minus_saved_val_acc"] = diagnostics[split]["actual_accuracy"] - float(payload["val_acc"])
                    _npz(directory / f"{split}.npz", arrays)
                    arrays = None
                history = export_checkpoint_history(directory, payload)
                _assert_inputs_stable(provenance, checkpoint_before)
                metadata = {**record, "checkpoint_epoch": int(payload["epoch"]),
                            "num_classes": int(payload["num_classes"]), "val_acc": payload.get("val_acc"),
                            "gate_version": int(model.fusion_version), "gate": model.fusion_gate.cpu().tolist(),
                            "model_args": payload["model_args"], "checkpoint_sha256": provenance["checkpoint"]["sha256"],
                            "resolved_feature_root": args.feature_root, "resolved_meta_root": args.meta_root,
                            "provenance": provenance, "split_diagnostics": diagnostics,
                            "prototype_bank_summary": _bank_summary(payload.get("prototype_bank")), "cl_history": history,
                            "controls": {"uniform": 0.5, "audio": 1.0, "visual": 0.0,
                                         "note": "fixed shared-head controls; visual retains audio-guided attention"}}
                _json(directory / "checkpoint.json", _safe_json(metadata))
                record.update(status="complete", cached=False)
            # Refresh invocation identity after validation even on a cache hit.
            # Reports can then reject old complete results if this worker never
            # started or was interrupted before verifying the current inputs.
            _json(directory / "status.json", {**record, "splits": splits, "elapsed_seconds": time.monotonic() - started})
            print(f"[worker {rank}] {job_id}: {record['status']} cached={record['cached']}", flush=True)
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            summary["failed"] += 1
            _json(directory / "status.json", {**record, "elapsed_seconds": time.monotonic() - started})
            print(f"[worker {rank}] {job_id}: {record['error']}", flush=True)
        finally:
            model = payload = arrays = shared = None
            gc.collect()
            if target.type == "cuda":
                torch.cuda.empty_cache()
        summary["jobs"].append(record)
        _json(output / f"worker_{rank}.json", summary)
    _json(output / f"worker_{rank}.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    result = run_worker(args.manifest, args.rank, args.world_size, args.device)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
