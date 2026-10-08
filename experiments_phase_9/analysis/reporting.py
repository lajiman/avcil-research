"""Descriptive prediction analysis; never loads or modifies training checkpoints.

Fractions are used for accuracy/F1; *_pp columns are percentage points. Seeds
are experimental repetitions. Classes and steps must not be treated as seeds.
"""

from collections import defaultdict
import html
import json
import math
from pathlib import Path

import numpy as np

from .common import read_csv, read_json, write_csv, write_json, resolve_baseline

VARIANTS = ("actual", "uniform", "audio", "visual")
GROUPS = {
    "validation_difficulty": ("hard", "middle", "easy"),
    "validation_modality_gap": ("low", "middle", "high"),
    "validation_complementarity": ("low", "middle", "high"),
}


def _ratio(n, d):
    return float(n / d) if d else float("nan")


def _core(run, job):
    return {**{k: run.get(k, "") for k in
               ("run_id", "method", "method_id", "protocol_id", "seed")},
            "step": int(job["step"]), "kind": job["kind"], "job_id": job["job_id"]}


def prediction_statistics(labels, predictions, num_classes, old_classes):
    """Macro-F1 uses all seen candidates, as in the original evaluator."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    matrix = np.bincount(labels * num_classes + predictions,
                        minlength=num_classes ** 2).reshape(num_classes, num_classes)
    support, predicted, tp = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    recall = np.divide(tp, support, out=np.zeros(num_classes, float), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes, float), where=predicted > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(num_classes, float),
                   where=support + predicted > 0)
    old, correct = labels < old_classes, labels == predictions
    old_new = int(((predictions >= old_classes) & old).sum())
    new_old = int(((predictions < old_classes) & ~old).sum())
    summary = {
        "support": len(labels), "accuracy": float(correct.mean()), "macro_f1": float(f1.mean()),
        "old_support": int(old.sum()), "new_support": int((~old).sum()),
        "old_accuracy": _ratio(correct[old].sum(), old.sum()),
        "new_accuracy": _ratio(correct[~old].sum(), (~old).sum()),
        "old_to_new": old_new, "new_to_old": new_old,
        "old_to_new_rate": _ratio(old_new, old.sum()), "new_to_old_rate": _ratio(new_old, (~old).sum()),
    }
    rows = [{"class_id": c, "support": int(support[c]), "tp": int(tp[c]),
             "fp": int(predicted[c] - tp[c]), "fn": int(support[c] - tp[c]),
             "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c]),
             "group": "old" if c < old_classes else "new"} for c in range(num_classes)]
    return summary, rows, matrix


def _load_cache(root, job, split, expected_invocation=None):
    directory = root / "cache" / job["job_id"]
    status = read_json(directory / "status.json", {})
    if status.get("status") != "complete":
        raise ValueError("cache status is not complete")
    if expected_invocation is not None and status.get("invocation_id") != expected_invocation:
        raise ValueError("cache was not verified in this pipeline invocation")
    if split not in status.get("splits", []):
        raise ValueError("requested split absent from cache status: " + split)
    for key in ("job_id", "run_id", "step", "kind"):
        if status.get(key) != job[key]:
            raise ValueError("cache identity mismatch: " + key)
    metadata = read_json(directory / "checkpoint.json", {})
    if not status.get("fingerprint") or metadata.get("fingerprint") != status["fingerprint"]:
        raise ValueError("cache/checkpoint fingerprint absent or inconsistent")
    with np.load(directory / (split + ".npz"), allow_pickle=False) as stored:
        result = {k: stored[k] for k in stored.files}
    ids, labels, gate = result["ids"], result["labels"], result["gate"]
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(set(ids.tolist())) != len(ids):
        raise ValueError("sample IDs must be a unique string vector")
    if labels.shape != ids.shape or labels.dtype.kind not in "iu" or len(labels) == 0:
        raise ValueError("labels must be a nonempty integer vector matching IDs")
    if (gate.ndim != 1 or not len(gate) or not np.isfinite(gate).all()
            or np.any(gate < 0) or np.any(gate > 1)):
        raise ValueError("invalid gate vector")
    c = len(gate)
    if metadata.get("step", job["step"]) != job["step"] or metadata.get("num_classes", c) != c:
        raise ValueError("checkpoint metadata class count/step mismatch")
    if not np.array_equal(result["class_ids"], np.arange(c)):
        raise ValueError("candidate columns must be contiguous class IDs")
    if np.any(labels < 0) or np.any(labels >= c):
        raise ValueError("labels outside candidate classes")
    for variant in VARIANTS:
        logits = result["logits_" + variant]
        if logits.shape != (len(ids), c) or not np.isfinite(logits).all():
            raise ValueError("invalid/nonfinite logits_" + variant)
    order = np.argsort(ids, kind="stable")
    for key in ("ids", "labels") + tuple("logits_" + v for v in VARIANTS):
        result[key] = result[key][order]
    result["metadata"] = metadata
    return result


def _aligned(left, right):
    for key in ("ids", "labels", "class_ids"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError("paired caches disagree on " + key + "; comparison skipped")


def _pair_rows(cache, method, comparator, core, comparison):
    labels = cache["labels"]
    left, right = method == labels, comparator == labels
    corrected, broken, changed = left & ~right, ~left & right, method != comparator
    rows = []
    for c in [-1] + list(range(len(cache["class_ids"]))):
        mask = np.ones(len(labels), bool) if c == -1 else labels == c
        rows.append({**core, "comparison": comparison, "class_id": c, "support": int(mask.sum()),
                     "corrected": int((corrected & mask).sum()), "broken": int((broken & mask).sum()),
                     "both_correct": int((left & right & mask).sum()),
                     "both_wrong": int((~left & ~right & mask).sum()),
                     "changed_predictions": int((changed & mask).sum()),
                     "delta_accuracy_pp": 100 * _ratio((corrected & mask).sum() - (broken & mask).sum(), mask.sum())})
    samples = [{**core, "comparison": comparison, "sample_id": str(cache["ids"][i]),
                "label": int(labels[i]), "method_prediction": int(method[i]),
                "comparator_prediction": int(comparator[i]),
                "change": "corrected" if corrected[i] else "broken" if broken[i] else "changed_wrong"}
               for i in np.flatnonzero(changed)]
    return rows, samples


def validation_groups(cache):
    """Groups use baseline validation only; ties stay together, possibly emptying groups."""
    y = cache["labels"]
    correct = {v: cache["logits_" + v].argmax(1) == y for v in VARIANTS}
    measurements = []
    for c in cache["class_ids"]:
        mask = y == c
        if not mask.any():
            continue
        a, v = correct["audio"][mask], correct["visual"][mask]
        measurements.append({"class_id": int(c), "validation_support": int(mask.sum()),
                             "validation_accuracy": float(correct["uniform"][mask].mean()),
                             "audio_accuracy": float(a.mean()), "visual_accuracy": float(v.mean()),
                             "modality_gap": abs(float(a.mean() - v.mean())),
                             "audio_only_correct": float((a & ~v).mean()),
                             "visual_only_correct": float((v & ~a).mean()),
                             "both_correct": float((a & v).mean()), "both_wrong": float((~a & ~v).mean()),
                             "exclusive_correct_rate": float((a ^ v).mean()),
                             "complementarity": min(float((a & ~v).mean()), float((v & ~a).mean()))})
    rows = []
    for axis, metric in (("validation_difficulty", "validation_accuracy"),
                         ("validation_modality_gap", "modality_gap"),
                         ("validation_complementarity", "complementarity")):
        if not measurements:
            continue
        lo, hi = np.quantile([r[metric] for r in measurements], [1 / 3, 2 / 3])
        for row in measurements:
            value = row[metric]
            group = GROUPS[axis][0 if value <= lo else 1 if value <= hi else 2]
            rows.append({**row, "axis": axis, "group": group, "grouping_value": value,
                         "lower_threshold": float(lo), "upper_threshold": float(hi),
                         "group_source": "baseline_validation_fusion_ablation"})
    return rows


def _group_metrics(cache, baseline, members, core):
    y = cache["labels"]
    pred, base_pred = cache["logits_actual"].argmax(1), baseline["logits_uniform"].argmax(1)
    _, mc, _ = prediction_statistics(y, pred, len(cache["class_ids"]), 0)
    _, bc, _ = prediction_statistics(y, base_pred, len(cache["class_ids"]), 0)
    output = []
    for axis, names in GROUPS.items():
        for name in names:
            cs = [r["class_id"] for r in members if r["axis"] == axis and r["group"] == name]
            mask = np.isin(y, cs)
            acc = _ratio((pred[mask] == y[mask]).sum(), mask.sum())
            base_acc = _ratio((base_pred[mask] == y[mask]).sum(), mask.sum())
            f1 = float(np.mean([mc[c]["f1"] for c in cs])) if cs else float("nan")
            base_f1 = float(np.mean([bc[c]["f1"] for c in cs])) if cs else float("nan")
            output.append({**core, "axis": axis, "group": name, "class_count": len(cs),
                           "support": int(mask.sum()), "accuracy": acc, "baseline_accuracy": base_acc,
                           "delta_accuracy_pp": 100 * (acc - base_acc), "macro_f1": f1,
                           "baseline_macro_f1": base_f1, "delta_macro_f1_pp": 100 * (f1 - base_f1),
                           "corrected": int(((pred == y) & (base_pred != y) & mask).sum()),
                           "broken": int(((pred != y) & (base_pred == y) & mask).sum()),
                           "group_source": "baseline_validation_fusion_ablation"})
    return output


def _group_seed_summary(rows):
    fields = ("method_id", "protocol_id", "kind", "step", "axis", "group")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda x: str(x[0])):
        seeds = defaultdict(list)
        for row in values:
            seeds[str(row["seed"])].append(row)
        good = [v[0] for v in seeds.values() if len(v) == 1 and math.isfinite(v[0]["delta_accuracy_pp"])]
        delta = np.array([r["delta_accuracy_pp"] for r in good])
        output.append({**dict(zip(fields, key)), "n_seeds": len(good),
                       "ambiguous_seeds_excluded": sum(len(v) > 1 for v in seeds.values()),
                       "delta_accuracy_mean_pp": float(delta.mean()) if len(delta) else float("nan"),
                       "delta_accuracy_sd_pp": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
                       "improved_seeds": int((delta > 0).sum()), "degraded_seeds": int((delta < 0).sum()),
                       "unchanged_seeds": int((delta == 0).sum()),
                       "seed_deltas_pp": json.dumps({str(r["seed"]): r["delta_accuracy_pp"] for r in good}, sort_keys=True)})
    return output


def _mechanism_seed_summary(rows):
    fields = ("method_id", "protocol_id", "kind", "step", "metric")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda x: str(x[0])):
        seeds = defaultdict(list)
        for row in values:
            seeds[str(row["seed"])].append(row)
        unique = [v[0] for v in seeds.values() if len(v) == 1]
        for effect in ("total_delta_pp", "fixed_model_gate_effect_pp", "trained_parameters_difference_pp"):
            good = [r for r in unique if math.isfinite(r[effect])]
            array = np.array([r[effect] for r in good])
            result.append({**dict(zip(fields, key)), "effect": effect, "n_seeds": len(good),
                           "ambiguous_seeds_excluded": sum(len(v) > 1 for v in seeds.values()),
                           "mean_pp": float(array.mean()) if len(array) else float("nan"),
                           "sd_pp": float(array.std(ddof=1)) if len(array) > 1 else float("nan"),
                           "improved_seeds": int((array > 0).sum()), "degraded_seeds": int((array < 0).sum()),
                           "seed_effects_pp": json.dumps({str(r["seed"]): r[effect] for r in good}, sort_keys=True)})
    return result


def _reproduction_audit(run, job, test, validation, core):
    """Compare only the same checkpoint/evaluation semantics; last is not best."""
    rows, warnings = [], []
    def compare(split, metric, observed, expected):
        if expected is None:
            rows.append({**core, "split": split, "metric": metric, "status": "skip",
                         "observed": observed, "expected": "", "absolute_difference": "",
                         "tolerance": 1e-6, "detail": "Original reference metric unavailable"})
            return
        try:
            difference = abs(float(observed) - float(expected))
        except (ValueError, TypeError):
            difference = float("nan")
        status = "pass" if math.isfinite(difference) and difference <= 1e-6 else "fail"
        rows.append({**core, "split": split, "metric": metric, "status": status,
                     "observed": observed, "expected": expected, "absolute_difference": difference,
                     "tolerance": 1e-6, "detail": "Full candidate set, saved checkpoint"})
        if status == "fail":
            warnings.append(job["job_id"] + ": reproduction mismatch for " + split + "/" + metric)
    if test is not None and job["kind"] == "best":
        directory = run.get("metrics_dir")
        saved = {}
        if directory:
            try:
                saved = read_json(Path(directory) / f"step_{job['step']}_test.json", {})
            except (OSError, ValueError) as exc:
                warnings.append(job["job_id"] + ": cannot read original test metrics: " + str(exc))
        summary = prediction_statistics(test["labels"], test["logits_actual"].argmax(1),
                                        len(test["class_ids"]), 0)[0]
        compare("test", "accuracy", summary["accuracy"], saved.get("overall_acc"))
        compare("test", "macro_f1", summary["macro_f1"], saved.get("macro_f1"))
        compare("test", "gate_version", test["metadata"].get("gate_version"), saved.get("gate_version"))
    if validation is not None:
        accuracy = float((validation["logits_actual"].argmax(1) == validation["labels"]).mean())
        compare("val", "accuracy", accuracy, validation["metadata"].get("val_acc"))
    return rows, warnings


def _coverage(cache, split, core):
    """Show feature filtering and class coverage before interpreting accuracy."""
    diagnostics = cache["metadata"].get("split_diagnostics", {}).get(split, {})
    observed = len(cache["labels"])
    expected = diagnostics.get("expected_unique_samples")
    recorded = diagnostics.get("num_samples")
    missing_pairs = diagnostics.get("missing_feature_pairs")
    missing_classes = sorted(set(cache["class_ids"].tolist()) - set(cache["labels"].tolist()))
    partial = bool(missing_classes or (missing_pairs is not None and missing_pairs > 0)
                   or (expected is not None and observed != expected)
                   or (recorded is not None and observed != recorded))
    status = "partial" if partial else "complete" if expected is not None and missing_pairs is not None else "unverified"
    row = {**core, "split": split, "scope": split, "status": status,
           "observed_samples": observed, "expected_unique_samples": expected,
           "recorded_num_samples": recorded, "missing_feature_pairs": missing_pairs,
           "candidate_classes": len(cache["class_ids"]), "observed_classes": len(np.unique(cache["labels"])),
           "missing_class_ids": json.dumps(missing_classes)}
    warnings = []
    if partial:
        warnings.append(core["job_id"] + "/" + split + ": incomplete evaluation cohort; observed=" +
                        str(observed) + ", expected=" + str(expected) + ", missing feature pairs=" +
                        str(missing_pairs) + ", absent class IDs=" + str(missing_classes))
    elif status == "unverified":
        warnings.append(core["job_id"] + "/" + split + ": expected sample coverage unavailable in cache metadata")
    return row, warnings


def build_report(manifest):
    """Write analysis outputs; missing/invalid/ambiguous inputs stay explicit."""
    root = Path(manifest["output_root"])
    tables = root / "tables"
    root.mkdir(parents=True, exist_ok=True)
    runs = manifest.get("runs", [])
    by_run = {run["run_id"]: run for run in runs}
    warnings, caches, job_index = [], {}, defaultdict(list)
    data = {name: [] for name in (
        "inference_summary", "inference_classes", "inference_confusion",
        "inference_pairs", "inference_sample_changes", "inference_class_deltas", "mechanism_decomposition",
        "validation_group_membership", "validation_group_metrics", "inference_reproduction_audit",
        "inference_coverage")}
    options = manifest.get("options", {})
    offline = options.get("mode") == "offline" or options.get("inference_mode") == "offline"
    jobs = [] if offline else manifest.get("jobs", [])
    if offline:
        warnings.append("Offline mode: prediction caches intentionally ignored; only existing logs/CSV are analyzed.")
    for job in jobs:
        run = by_run.get(job["run_id"])
        if run is None:
            warnings.append("Unknown run for " + job["job_id"])
            continue
        job_index[(job["run_id"], int(job["step"]), job["kind"])].append(job)
        directory = root / "cache" / job["job_id"]
        if not (directory / "status.json").is_file():
            warnings.append("No inference cache: " + job["job_id"])
            continue
        requested_splits = options.get("splits", ["test", "val"])
        for split in ("test", "val"):
            if split not in requested_splits:
                continue
            if split == "val" and not (directory / "val.npz").is_file():
                warnings.append(job["job_id"] + "/val: requested validation prediction cache missing")
                data["inference_coverage"].append({**_core(run, job), "split": split, "scope": split, "status": "missing"})
                continue
            try:
                cached = _load_cache(root, job, split, manifest.get("invocation_id"))
                caches[(job["job_id"], split)] = cached
                coverage, coverage_warnings = _coverage(cached, split, _core(run, job))
                data["inference_coverage"].append(coverage)
                warnings.extend(coverage_warnings)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                caches.pop((job["job_id"], split), None)
                warnings.append(job["job_id"] + "/" + split + ": " + str(exc))
                data["inference_coverage"].append({**_core(run, job), "split": split, "scope": split,
                                                   "status": "invalid", "detail": str(exc)})
        cache = caches.get((job["job_id"], "test"))
        audit_rows, audit_warnings = _reproduction_audit(
            run, job, cache, caches.get((job["job_id"], "val")), _core(run, job))
        data["inference_reproduction_audit"].extend(audit_rows)
        warnings.extend(audit_warnings)
        if cache is None:
            continue
        core, c = _core(run, job), len(cache["class_ids"])
        per_step = int(run.get("config", {}).get("class_num_per_step", c // (int(job["step"]) + 1)))
        old = int(job["step"]) * per_step
        if c != old + per_step:
            warnings.append(job["job_id"] + ": class count disagrees with step/config; metrics skipped")
            caches.pop((job["job_id"], "test"), None)
            continue
        for variant in VARIANTS:
            summary, classes, matrix = prediction_statistics(
                cache["labels"], cache["logits_" + variant].argmax(1), c, old)
            data["inference_summary"].append({
                **core, "split": "test", "scope": "test", "variant": variant, **summary,
                "checkpoint_epoch": cache["metadata"].get("checkpoint_epoch", ""),
                "gate_version": cache["metadata"].get("gate_version", ""),
                "gate_mean_abs_deviation": float(np.abs(cache["gate"] - 0.5).mean())})
            data["inference_classes"].extend(
                {**core, "variant": variant, **row, "class_age": int(job["step"]) - row["class_id"] // per_step,
                 "audio_gate": float(cache["gate"][row["class_id"]])}
                for row in classes)
            data["inference_confusion"].extend(
                {**core, "variant": variant, "true_class": int(t), "predicted_class": int(p), "count": int(matrix[t, p])}
                for t, p in zip(*np.nonzero(matrix)))
        pair, changes = _pair_rows(cache, cache["logits_actual"].argmax(1), cache["logits_uniform"].argmax(1),
                                   core, "same_model_actual_minus_uniform")
        data["inference_pairs"].extend(pair)
        data["inference_sample_changes"].extend(changes)

    # Pair only after loading all caches, independent of manifest ordering.
    for job in jobs:
        run, cache = by_run.get(job["run_id"]), caches.get((job["job_id"], "test"))
        if cache is None or run is None or run.get("config", {}).get("fusion_mode") == "uniform":
            continue
        core = _core(run, job)
        try:
            base_run = resolve_baseline(run, runs)
            if base_run is None:
                raise ValueError("independent uniform baseline missing or ambiguous")
            matches = job_index[(base_run["run_id"], int(job["step"]), job["kind"])]
            if len(matches) != 1:
                raise ValueError("baseline checkpoint match missing or ambiguous")
            base_job = matches[0]
            baseline = caches.get((base_job["job_id"], "test"))
            if baseline is None:
                raise ValueError("baseline test cache unavailable")
            _aligned(cache, baseline)
            if not np.all(baseline["gate"] == 0.5) or not np.allclose(
                    baseline["logits_actual"], baseline["logits_uniform"], rtol=1e-5, atol=1e-6):
                raise ValueError("uniform baseline cache does not have uniform behavior")
        except ValueError as exc:
            warnings.append(job["job_id"] + ": " + str(exc))
            continue
        core = {**core, "baseline_run_id": base_run["run_id"], "baseline_job_id": base_job["job_id"]}
        pred, own_uniform = cache["logits_actual"].argmax(1), cache["logits_uniform"].argmax(1)
        base_pred = baseline["logits_uniform"].argmax(1)
        pair, changes = _pair_rows(cache, pred, base_pred, core, "method_actual_minus_baseline_uniform")
        data["inference_pairs"].extend(pair)
        data["inference_sample_changes"].extend(changes)
        c = len(cache["class_ids"])
        per_step = int(run.get("config", {}).get("class_num_per_step", c // (int(job["step"]) + 1)))
        stats = [prediction_statistics(cache["labels"], p, c, int(job["step"]) * per_step)[0]
                 for p in (pred, own_uniform, base_pred)]
        _, method_class, _ = prediction_statistics(cache["labels"], pred, c, int(job["step"]) * per_step)
        _, base_class, _ = prediction_statistics(cache["labels"], base_pred, c, int(job["step"]) * per_step)
        for mc, bc in zip(method_class, base_class):
            data["inference_class_deltas"].append({
                **core, "class_id": mc["class_id"], "support": mc["support"], "group": mc["group"],
                "class_age": int(job["step"]) - mc["class_id"] // per_step,
                "recall": mc["recall"], "baseline_recall": bc["recall"],
                "delta_recall_pp": 100 * (mc["recall"] - bc["recall"]),
                "f1": mc["f1"], "baseline_f1": bc["f1"], "delta_f1_pp": 100 * (mc["f1"] - bc["f1"]),
                "fp": mc["fp"], "baseline_fp": bc["fp"], "delta_fp": mc["fp"] - bc["fp"]})
        for metric in ("accuracy", "macro_f1", "old_accuracy", "new_accuracy"):
            dd, d0, u0 = [s[metric] for s in stats]
            data["mechanism_decomposition"].append({
                **core, "metric": metric, "dynamic_actual": dd, "dynamic_uniform": d0, "baseline_uniform": u0,
                "total_delta_pp": 100 * (dd - u0), "fixed_model_gate_effect_pp": 100 * (dd - d0),
                "trained_parameters_difference_pp": 100 * (d0 - u0),
                "identity_residual_pp": 100 * ((dd - u0) - ((dd - d0) + (d0 - u0)))})
        base_val = caches.get((base_job["job_id"], "val"))
        if base_val is None:
            warnings.append(job["job_id"] + ": baseline validation cache absent; validation-defined groups omitted")
            continue
        if not np.array_equal(base_val["class_ids"], baseline["class_ids"]):
            warnings.append(job["job_id"] + ": validation/test candidates differ; groups omitted")
            continue
        if np.intersect1d(base_val["ids"], baseline["ids"]).size:
            warnings.append(job["job_id"] + ": validation/test share sample IDs; groups omitted")
            continue
        if not np.all(base_val["gate"] == 0.5) or not np.allclose(
                base_val["logits_actual"], base_val["logits_uniform"], rtol=1e-5, atol=1e-6):
            warnings.append(job["job_id"] + ": baseline validation cache is not uniform; groups omitted")
            continue
        if len(np.unique(base_val["labels"])) != len(base_val["class_ids"]):
            warnings.append(job["job_id"] + ": some classes have no validation samples; those classes are ungrouped")
        members = validation_groups(base_val)
        data["validation_group_membership"].extend({**core, **row} for row in members)
        data["validation_group_metrics"].extend(_group_metrics(cache, baseline, members, core))

    data["validation_group_seed_summary"] = _group_seed_summary(data["validation_group_metrics"])
    data["mechanism_seed_summary"] = _mechanism_seed_summary(data["mechanism_decomposition"])
    for name, rows in data.items():
        # Empty results still have a readable CSV header, not a blank file.
        output = [{k: "" if isinstance(v, float) and not math.isfinite(v) else v
                   for k, v in row.items()} for row in rows]
        write_csv(tables / (name + ".csv"), output, None if rows else ["run_id", "step", "kind"])
    flagged = [r for r in read_csv(tables / "audit.csv") if r.get("status") in ("warn", "fail")]
    warnings = list(dict.fromkeys(warnings))
    figures = []
    if manifest.get("options", {}).get("plots", True):
        try:
            figures = _plots(root, data, warnings)
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append("Figure generation failed; numerical tables remain valid: " + str(exc))
    _write_report(root, runs, caches, data, flagged, warnings, figures)
    result = {"test_caches": sum(split == "test" for _, split in caches),
              "cache_splits": len(caches), "warnings": warnings,
              "coverage_issues": sum(r["status"] != "complete" for r in data["inference_coverage"]),
              "reproduction_failures": sum(r["status"] == "fail" for r in data["inference_reproduction_audit"]),
              "offline_flags": len(flagged), "figures": figures,
              "tables": {name: len(rows) for name, rows in data.items()}, "report": str(root / "report.md")}
    write_json(root / "report_summary.json", result)
    return result


def _table(rows, fields, limit=30):
    if not rows:
        return "No rows available."
    def cell(v):
        if isinstance(v, float):
            v = f"{v:.5g}" if math.isfinite(v) else ""
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(
        ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"] +
        ["| " + " | ".join(cell(r.get(k, "")) for k in fields) + " |" for r in rows[:limit]])


def _write_report(root, runs, caches, data, flagged, warnings, figures):
    tables = root / "tables"
    sections = [
        "# Phase 9 experiment analysis",
        f"Runs discovered: {len(runs)}. Complete test caches: {sum(s == 'test' for _, s in caches)}. "
        f"Offline audit flags: {len(flagged)}. Analysis warnings: {len(warnings)}.",
        "## Run overview",
        _table(read_csv(tables / "run_summary.csv"), [
            "run_id", "seed", "complete", "observed_steps", "average_incremental_accuracy",
            "final_accuracy", "final_old_accuracy", "final_new_accuracy"]),
        "## Method and paired-seed summaries",
        _table(read_csv(tables / "method_summary.csv"), [
            "method", "protocol_id", "metric", "n_seeds", "mean", "sd", "paired_n_seeds",
            "delta_mean_pp", "improved_seeds"], 60),
        "## Fixed-model fusion decomposition",
        "For each matching seed, protocol, task and checkpoint kind: DD-U0 = (DD-D0) + (D0-U0). "
        "DD uses dynamic-model parameters and saved gates; D0 uses those parameters with equal fusion; "
        "U0 is the independently trained uniform baseline. The second term includes all parameter and "
        "training-trajectory differences; it is not a pure causal representation effect. Best and last stay separate.",
        _table(data["mechanism_decomposition"], [
            "run_id", "step", "kind", "metric", "total_delta_pp", "fixed_model_gate_effect_pp",
            "trained_parameters_difference_pp"], 40),
        _table(data["mechanism_seed_summary"], [
            "method_id", "kind", "step", "metric", "effect", "n_seeds", "mean_pp", "sd_pp", "improved_seeds"], 40),
        "## Validation-defined subgroup results",
        "Groups are frozen from the matched uniform baseline validation predictions before test outcomes: "
        "difficulty uses equal-fusion accuracy; modality gap uses the absolute audio/visual ablation accuracy gap; "
        "complementarity is min(audio-only correct, visual-only correct), equal to oracle accuracy minus "
        "the better branch accuracy. The exclusive-correct fraction is retained separately. All terciles are reported, "
        "including empty groups due to ties. No test-baseline difficulty grouping or significance testing is used.",
        _table(data["validation_group_seed_summary"], [
            "method_id", "kind", "step", "axis", "group", "n_seeds", "delta_accuracy_mean_pp",
            "delta_accuracy_sd_pp", "improved_seeds"], 40),
        "## Interpretation and coverage limits",
        "- Accuracy/F1 are fractions; _pp denotes percentage points. Steps are zero-based; class age is tasks since introduction.\n"
        "- Old/new accuracy uses all seen candidates. Old-to-new/new-to-old rates divide by the true old/new population.\n"
        "- Audio/visual ablations retain two-input feature extraction, including audio-guided visual attention; these are not independent unimodal models.\n"
        "- Macro-F1 includes all seen classes. Group F1 retains false positives from all test samples. Empty populations are blank, not zero.\n"
        "- Sample-change tables list changed predictions only; pair tables include corrected/broken/both-correct/both-wrong counts over all samples. Class -1 denotes all samples. Confusion CSV stores nonzero cells.\n"
        "- Validation already selects checkpoints. Groups are exploratory, not held-out confirmatory tests; train/validation source-video overlap can further reduce independence.\n"
        "- Seeds, not classes or steps, are experimental repetitions. Three seeds support descriptive means/SD and sign consistency, not strong significance claims.\n"
        "- Validation quantile membership may differ across seeds. Group means describe the defined stratum, not necessarily the same classes across seeds.\n"
        "- Missing, failed or ambiguous caches produce no invented metrics. Offline records still support analysis with inference disabled.\n"
        "- Inference summary/confusion tables describe test predictions. Coverage tables list each requested split's expected/observed samples and missing classes; partial feature coverage is not a complete test cohort.\n"
        "- Display tables are abbreviated for readability; linked CSVs contain every row, including decreases and empty groups.",
        "## Offline audit flags",
        _table(flagged, ["run_id", "check", "status", "count", "detail"], 60),
        "## Checkpoint reproduction audit",
        "Best test predictions are compared with the original test JSON; each checkpoint's validation "
        "accuracy is compared with its saved validation score (tolerance 1e-6). Last checkpoints are "
        "not compared to best-model test records. Missing original metrics are explicitly skipped.",
        _table(data["inference_reproduction_audit"], [
            "run_id", "step", "kind", "split", "metric", "status", "observed", "expected", "absolute_difference"], 40),
        "## Evaluation cohort coverage",
        _table(data["inference_coverage"], [
            "run_id", "step", "kind", "split", "status", "observed_samples", "expected_unique_samples",
            "missing_feature_pairs", "missing_class_ids"], 40),
        "## Analysis warnings",
        "\n".join("- " + w.replace("\n", " ") for w in warnings) if warnings else "No cache/report warnings.",
        "## Output files",
        "\n".join(f"- [{p.name}](tables/{p.name})" for p in sorted(tables.glob("*.csv"))),
    ]
    if figures:
        sections.extend(["## Figures", "\n".join(
            f"![{Path(p).stem}]({p})" for p in figures if p.endswith(".png"))])
    markdown = "\n\n".join(sections) + "\n"
    (root / "report.md").write_text(markdown, encoding="utf-8")
    links = "".join('<li><a href="tables/' + html.escape(p.name, quote=True) + '">' +
                    html.escape(p.name) + "</a></li>" for p in sorted(tables.glob("*.csv")))
    images = "".join('<img alt="' + html.escape(Path(p).stem, quote=True) + '" src="' +
                     html.escape(p, quote=True) + '">' for p in figures if p.endswith(".png"))
    # Escaped source is readable offline without a CDN or a Markdown dependency.
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 9 analysis</title>' \
               '<style>body{max-width:1200px;margin:2em auto;font:15px system-ui;padding:1em}' \
               'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:1em}img{max-width:100%}</style>' \
               '<h1>Phase 9 analysis</h1><ul>' + links + '</ul><pre>' + html.escape(markdown) + '</pre>' + images + '</html>'
    (root / "report.html").write_text(document, encoding="utf-8")


def _plots(root, data, warnings):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        warnings.append("Matplotlib unavailable: figures skipped, all numerical tables retained.")
        return []
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            p = directory / (name + "." + suffix)
            fig.savefig(p, dpi=160)
            files.append(p.relative_to(root).as_posix())
        plt.close(fig)
    cohorts = defaultdict(list)
    for r in read_csv(root / "tables" / "steps.csv"):
        try:
            cohorts[(r["protocol_id"], r["run_id"])].append((int(r["step"]), float(r["overall_acc"])))
        except (KeyError, ValueError, TypeError):
            continue
    for i, protocol in enumerate(sorted({k[0] for k in cohorts})):
        fig, ax = plt.subplots(figsize=(9, 5))
        for (p, run), values in sorted(cohorts.items()):
            if p == protocol:
                values.sort()
                ax.plot([s + 1 for s, _ in values], [100 * a for _, a in values], marker=".", label=run)
        ax.set(xlabel="Incremental task (1-based)", ylabel="Test accuracy (%)", title="Per-seed curves | " + str(protocol)[:16])
        ax.legend(fontsize=6)
        save(fig, "accuracy_protocol_" + str(i))
    actual = [r for r in data["inference_summary"] if r["variant"] == "actual"]
    if actual:
        fig, ax = plt.subplots(figsize=(9, 5))
        for run, kind in sorted({(r["run_id"], r["kind"]) for r in actual}):
            rows = sorted([r for r in actual if (r["run_id"], r["kind"]) == (run, kind)], key=lambda r: r["step"])
            ax.plot([r["step"] + 1 for r in rows], [r["gate_mean_abs_deviation"] for r in rows],
                    marker=".", label=run + "/" + kind)
        ax.set(xlabel="Incremental task (1-based)", ylabel="Mean |gate - 0.5|", title="Gate strength of evaluated checkpoints")
        ax.legend(fontsize=6)
        save(fig, "gate_deviation")
    heatmaps = defaultdict(list)
    for row in data["inference_class_deltas"]:
        heatmaps[(row["protocol_id"], row["kind"], row["step"])].append(row)
    # One final-observed-task heatmap per protocol/kind; each seed remains a row.
    latest = {}
    for protocol, kind, step in heatmaps:
        latest[(protocol, kind)] = max(latest.get((protocol, kind), -1), step)
    for i, ((protocol, kind), step) in enumerate(sorted(latest.items())):
        rows = heatmaps[(protocol, kind, step)]
        ids = sorted({r["run_id"] for r in rows})
        cs = sorted({r["class_id"] for r in rows})
        matrix = np.full((len(ids), len(cs)), np.nan)
        for r in rows:
            matrix[ids.index(r["run_id"]), cs.index(r["class_id"])] = r["delta_recall_pp"]
        bound = max(1.0, float(np.nanmax(np.abs(matrix))))
        fig, ax = plt.subplots(figsize=(12, max(3, 0.3 * len(ids))))
        plot = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_yticks(range(len(ids)), ids, fontsize=6)
        ticks = np.arange(0, len(cs), max(1, len(cs) // 15))
        ax.set_xticks(ticks, [cs[j] for j in ticks])
        ax.set(xlabel="Class ID", title=f"Per-seed recall difference | {kind}, task {step + 1}")
        fig.colorbar(plot, ax=ax, label="Recall difference (pp)")
        save(fig, "class_recall_delta_" + str(i))
    groups = defaultdict(list)
    for row in data["validation_group_seed_summary"]:
        if row["n_seeds"]:
            groups[(row["method_id"], row["protocol_id"], row["kind"], row["step"])].append(row)
    latest = {}
    for method, protocol, kind, step in groups:
        key = (method, protocol, kind)
        latest[key] = max(latest.get(key, -1), step)
    for i, (key, step) in enumerate(sorted(latest.items())):
        rows = groups[(*key, step)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, (axis, names) in zip(axes, GROUPS.items()):
            selected = {r["group"]: r for r in rows if r["axis"] == axis}
            means = [selected[name]["delta_accuracy_mean_pp"] if name in selected else np.nan for name in names]
            errors = [selected[name]["delta_accuracy_sd_pp"] if name in selected and selected[name]["n_seeds"] > 1 else 0 for name in names]
            ax.bar(range(3), means, yerr=errors, color="#4875a8", capsize=3)
            ax.set_xticks(range(3), [name + "\nn=" + str(selected.get(name, {}).get("n_seeds", 0)) for name in names])
            ax.axhline(0, color="#444", linewidth=0.8)
            ax.set_title(axis.replace("validation_", ""), fontsize=9)
        axes[0].set_ylabel("Test accuracy difference (pp), mean ± seed SD")
        fig.suptitle(f"{key[0]} / {key[2]} / task {step + 1}", fontsize=10)
        save(fig, "validation_group_delta_" + str(i))
    return files
