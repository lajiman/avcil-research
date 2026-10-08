"""Read-only audits and seed-paired descriptive statistics for saved phase-9 runs.

No model is selected using test results. All accuracies are fractions; fields
ending in ``_pp`` are percentage-point differences. Memory prototype probes
are explicitly separated from test classification metrics. This module uses
only the standard library and never loads a checkpoint or executes log text.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from .common import read_csv, read_json, write_csv, write_json, number, resolve_baseline


IDENTITY = ("run_id", "method", "method_id", "protocol_id", "seed")
RUN_METRICS = ("average_incremental_accuracy", "final_accuracy", "final_macro_f1",
               "final_forgetting", "average_forgetting", "final_old_accuracy", "final_new_accuracy")


def finite(value):
    return math.isfinite(number(value))


def _integer(value, minimum=0):
    """Reject missing, fractional and boolean identifiers before int conversion."""
    return not isinstance(value, bool) and finite(value) and number(value).is_integer() and number(value) >= minimum


def mean(values):
    values = [number(v) for v in values if finite(v)]
    return statistics.mean(values) if values else None


def sd(values):
    values = [number(v) for v in values if finite(v)]
    return statistics.stdev(values) if len(values) > 1 else None


def close(a, b, tolerance=2e-6):
    return finite(a) and finite(b) and math.isclose(number(a), number(b), rel_tol=tolerance, abs_tol=tolerance)


def truth(value):
    return value is True or str(value).lower() in ("true", "1")


def identity(run):
    return {key: run.get(key) for key in IDENTITY}


def parse_namespace(text):
    """Accept only Namespace(k=<Python literal>, ...), never eval a log line."""
    tree = ast.parse(text.strip(), mode="eval").body
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != "Namespace" or tree.args:
        raise ValueError("Expected a literal argparse Namespace")
    result = {}
    for keyword in tree.keywords:
        if keyword.arg is None or keyword.arg in result:
            raise ValueError("Expanded or duplicate Namespace keyword")
        result[keyword.arg] = ast.literal_eval(keyword.value)
    return result


def parse_log(path):
    """Stream large tqdm logs, retaining the small structured summaries only."""
    result = {"namespace": None, "tests": [], "averages": [], "epochs": [], "traceback": False}
    if not path or not Path(path).is_file():
        return result
    pattern = r"Incremental step (\d+) Testing res \(overall acc\): ([\deE.+-]+)"
    epoch_pattern = r"Step (\d+) epoch (\d+): loss=([^,\s]+), val_acc=([^,\s]+), gate_version=(\d+)"
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Namespace("):
                try:
                    result["namespace"] = parse_namespace(line)
                except (ValueError, SyntaxError):
                    result["namespace_error"] = "Unsafe or malformed Namespace ignored"
            match = re.search(pattern, line)
            if match:
                result["tests"].append((int(match[1]), float(match[2])))
            match = re.search(epoch_pattern, line)
            if match:
                result["epochs"].append({"step": int(match[1]), "epoch": int(match[2]),
                                         "train_loss": number(match[3]), "val_acc": number(match[4]),
                                         "gate_version": int(match[5])})
            match = re.search(r"Average incremental accuracy: ([^;\s]+); average forgetting: (\S+)", line)
            if match:
                result["averages"].append((number(match[1]), number(match[2])))
            result["traceback"] |= "Traceback (most recent call last)" in line
    return result


def expected_updates(config, step):
    if config.get("fusion_mode", "periodic") == "uniform" or (step == 0 and not config.get("fusion_update_first_step", False)):
        return []
    start = int(config.get("fusion_warmup_epochs", 40))
    interval = int(config.get("fusion_update_interval", 40))
    maximum = int(config.get("max_epoches", 200))
    return list(range(start, maximum, interval)) if interval > 0 else []


def expected_loss(row, config):
    if int(row["step"]) == 0:
        return number(row.get("ce"))
    value = number(row.get("ce")) + number(row.get("kd"))
    for flag, key, coefficient, default in (("instance_contrastive", "instance", "lam_I", .1),
                                            ("class_contrastive", "class", "lam_C", 1.)):
        if config.get(flag, True):
            value += number(config.get(coefficient, default)) * number(row.get(key))
    if config.get("attn_score_distil", True):
        weight = number(config.get("lam", .5))
        value += weight * number(row.get("spatial")) + (1 - weight) * number(row.get("temporal"))
    return value


def _accuracy(rows):
    if not rows or any(not finite(r.get(k)) for r in rows for k in ("tp", "support")):
        return None
    support = sum(number(r["support"]) for r in rows)
    return sum(number(r["tp"]) for r in rows) / support if support > 0 else None


def _unique(rows, keys):
    counts = Counter(tuple(row.get(k) for k in keys) for row in rows)
    # Ambiguous duplicates are excluded, not silently averaged or overwritten.
    return [row for row in rows if counts[tuple(row.get(k) for k in keys)] == 1], sum(n for n in counts.values() if n > 1)


def _quantile(values, q):
    values = sorted(v for v in values if finite(v))
    if not values:
        return None
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _tertiles(rows, key):
    # Ties stay together; equal values may legitimately yield empty bins.
    values = [r[key] for r in rows if finite(r.get(key))]
    q1, q2 = _quantile(values, 1 / 3), _quantile(values, 2 / 3)
    return {r["class_id"]: ("low" if r[key] <= q1 else "middle" if r[key] <= q2 else "high")
            for r in rows if finite(r.get(key))} if values else {}


def analyze(manifest):
    """Write reproducible small tables below output_root; never modify training files."""
    output = Path(manifest["output_root"])
    tables = defaultdict(list)
    warnings = []
    runs = manifest["runs"]
    cache = {}

    def audit(run, check, status, count=0, detail="", source=""):
        tables["audit"].append({**identity(run), "check": check, "status": status,
                                "count": count, "detail": detail, "source": str(source)})
        if status in ("warn", "fail"):
            warnings.append({"run_id": run["run_id"], "check": check, "status": status, "detail": detail})

    def csv(run, path):
        if not path.is_file():
            return []
        try:
            return read_csv(path)
        except (ValueError, OSError, UnicodeError) as exc:
            audit(run, "csv_read", "fail", detail=f"{type(exc).__name__}: {exc}", source=path)
            return []

    def js(run, path, default=None):
        if not path.is_file():
            return default
        try:
            return read_json(path, default)
        except (ValueError, OSError, UnicodeError) as exc:
            audit(run, "json_read", "fail", detail=f"{type(exc).__name__}: {exc}", source=path)
            return default

    for run in runs:
        core, config = identity(run), run["config"]
        root = Path(run["metrics_dir"])
        expected_steps = int(run["num_steps"])
        classes_per_step = int(config.get("class_num_per_step", 10))
        max_epochs = int(config.get("max_epoches", 200))
        log = parse_log(run.get("log_path"))
        if log.get("namespace_error"):
            audit(run, "log_namespace", "warn", detail=log["namespace_error"])
        disagreements = []
        if log["namespace"] is not None:
            disagreements = [k for k, v in log["namespace"].items() if k in config and config[k] != v]
        audit(run, "log_config", "fail" if disagreements else "pass" if log["namespace"] else "warn",
              len(disagreements), "mismatched keys: " + ", ".join(disagreements) if disagreements else "No config mismatch; missing Namespace is not a pass" if not log["namespace"] else "Literal Namespace agrees with config")
        tables["log_summary"].append({**core, "log_path": run.get("log_path"), "log_present": bool(run.get("log_path") and Path(run["log_path"]).is_file()),
                                      "namespace_present": log["namespace"] is not None, "epoch_lines": len(log["epochs"]),
                                      "test_lines": len(log["tests"]), "test_steps": json.dumps([s for s, _ in log["tests"]]),
                                      "summary_lines": len(log["averages"]), "traceback": log["traceback"],
                                      "logged_average_incremental_accuracy": log["averages"][-1][0] if log["averages"] else None,
                                      "logged_average_forgetting": log["averages"][-1][1] if log["averages"] else None})
        audit(run, "log_traceback", "fail" if log["traceback"] else "pass", int(log["traceback"]), "Traceback text detected" if log["traceback"] else "No traceback text")
        duplicate_log_steps = sum(n - 1 for n in Counter(s for s, _ in log["tests"]).values() if n > 1)
        duplicate_log_epochs = sum(n - 1 for n in Counter((r["step"], r["epoch"]) for r in log["epochs"]).values() if n > 1)
        audit(run, "duplicate_log_records", "warn" if duplicate_log_steps or duplicate_log_epochs else "pass",
              duplicate_log_steps + duplicate_log_epochs, "Repeated test/epoch lines may indicate appended or restarted logs")

        epochs, duplicate_epochs = _unique(csv(run, root / "class_fusion/epoch_summary.csv"), ("step", "epoch"))
        audit(run, "duplicate_epochs", "fail" if duplicate_epochs else "pass", duplicate_epochs)
        invalid_epochs = [r for r in epochs if not all(finite(r.get(k)) for k in ("step", "epoch", "gate_version", "val_acc", "train_loss", "ce", "kd", "instance", "class", "spatial", "temporal"))]
        audit(run, "finite_epoch_metrics", "fail" if invalid_epochs else "pass", len(invalid_epochs), "Rows with absent/nonfinite required values")
        invalid_epoch_ids = [r for r in epochs if all(finite(r.get(k)) for k in ("step", "epoch", "gate_version"))
                             and not (_integer(r["step"]) and _integer(r["epoch"], 1) and _integer(r["gate_version"]))]
        audit(run, "integer_epoch_identifiers", "fail" if invalid_epoch_ids else "pass", len(invalid_epoch_ids),
              "Step and gate_version must be nonnegative integers; epoch must be a positive integer")
        excluded_epoch_rows = {id(r) for r in invalid_epochs + invalid_epoch_ids}
        valid_epochs = [r for r in epochs if id(r) not in excluded_epoch_rows]
        loss_fail = disabled_fail = 0
        best_epochs, epoch_by_step = {}, defaultdict(list)
        for row in valid_epochs:
            step, epoch = int(row["step"]), int(row["epoch"])
            if not (0 <= step < expected_steps and 1 <= epoch <= max_epochs):
                audit(run, "epoch_bounds", "fail", 1, f"step={step}, epoch={epoch} outside configured range")
                continue
            calculated = expected_loss(row, config)
            error = number(row["train_loss"]) - calculated
            loss_fail += not close(row["train_loss"], calculated)
            absent_terms = ("kd", "instance", "class", "spatial", "temporal") if step == 0 else tuple(
                k for flag, keys in (("instance_contrastive", ("instance",)), ("class_contrastive", ("class",)), ("attn_score_distil", ("spatial", "temporal")))
                if not config.get(flag, True) for k in keys)
            disabled_fail += any(abs(number(row[k])) > 2e-6 for k in absent_terms)
            tables["epoch_losses"].append({**core, **row, "recomputed_train_loss": calculated, "loss_formula_error": error})
            epoch_by_step[step].append(row)
        for step, values in epoch_by_step.items():
            best_epochs[step] = sorted(values, key=lambda row: (-number(row["val_acc"]), int(row["epoch"])))[0]
        audit(run, "loss_formula", "fail" if loss_fail else "pass" if valid_epochs else "warn", loss_fail, "Unweighted CSV terms recombined with actual config coefficients")
        audit(run, "inactive_loss_terms", "fail" if disabled_fail else "pass", disabled_fail, "First task CE only; disabled losses must be zero")
        missing_epochs = sum(len(set(range(1, max_epochs + 1)) - {int(r["epoch"]) for r in epoch_by_step[s]}) for s in range(expected_steps))
        audit(run, "epoch_completeness", "warn" if missing_epochs else "pass", missing_epochs, "Missing expected step/epoch rows (possible partial transfer/run)")

        classes, duplicate_classes = _unique(csv(run, root / "per_class_metrics.csv"), ("step", "class_id"))
        audit(run, "duplicate_class_metrics", "fail" if duplicate_classes else "pass", duplicate_classes, "Ambiguous rows excluded from aggregates")
        class_by_step = defaultdict(list)
        class_errors = 0
        for row in classes:
            if not all(finite(row.get(k)) for k in ("step", "class_id", "tp", "fp", "fn", "support", "recall", "precision", "f1")):
                class_errors += 1
                continue
            step, class_id = int(row["step"]), int(row["class_id"])
            if not (0 <= step < expected_steps and 0 <= class_id < (step + 1) * classes_per_step):
                class_errors += 1
                continue
            support, tp, fp, fn = (number(row[k]) for k in ("support", "tp", "fp", "fn"))
            recall = tp / support if support > 0 else 0.
            precision = tp / (tp + fp) if tp + fp > 0 else 0.
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.
            if min(support, tp, fp, fn) < 0 or not close(support, tp + fn) or not all(close(row[k], v) for k, v in (("recall", recall), ("precision", precision), ("f1", f1))):
                class_errors += 1
                continue
            normalized = {**core, **row, "scope": "test", "class_age": step - class_id // classes_per_step,
                          "group": "new" if class_id // classes_per_step == step else "old"}
            tables["classes"].append(normalized)
            class_by_step[step].append(normalized)
        audit(run, "class_metric_identities", "fail" if class_errors else "pass", class_errors, "Check finite TP/FP/FN/support and precision/recall/F1; invalid rows excluded")

        snapshots, gate_rows = defaultdict(dict), []
        gate_failures = Counter()
        for path in sorted((root / "class_fusion").glob("*_gates.csv")):
            rows, duplicates = _unique(csv(run, path), ("step", "after_epoch", "class_id"))
            gate_failures["duplicate_gate_rows"] += duplicates
            for row in rows:
                if not all(finite(row.get(k)) for k in ("step", "after_epoch", "active_from_epoch", "gate_version", "class_id", "audio_gate", "visual_gate")):
                    gate_failures["gate_nonfinite"] += 1
                    continue
                step, after, c = int(row["step"]), int(row["after_epoch"]), int(row["class_id"])
                g = number(row["audio_gate"])
                gate_failures["gate_bounds"] += not (0 <= g <= 1 and close(g + number(row["visual_gate"]), 1.))
                gate_failures["gate_active_epoch"] += int(row["active_from_epoch"]) != after + 1
                snapshots[(step, after)][c] = row
                if after:
                    valid = truth(row.get("valid"))
                    p, raw, eta, n = (number(row.get(k)) for k in ("previous_gate", "raw_gate", "eta", "counts"))
                    expected_eta = 1. if config.get("fusion_update_rule") == "direct" else number(config.get("fusion_eta_max", .5))
                    if config.get("fusion_update_rule") == "sample_aware":
                        expected_eta *= n / (n + number(config.get("fusion_n_ref", 10)))
                    if not valid:
                        expected_eta = 0.
                    gate_failures["gate_eta"] += not close(eta, expected_eta)
                    gate_failures["gate_update_formula"] += not close(g, p + eta * (raw - p))
                    if valid:
                        ra, rv = number(row.get("reliability_a")), number(row.get("reliability_v"))
                        gate_failures["gate_raw_ratio"] += not close(raw, ra / max(ra + rv, 1e-12))
                        gate_failures["gate_valid_counts"] += not (close(row.get("counts"), row.get("scored_counts")) and n >= int(config.get("fusion_min_samples", 2)))
                    else:
                        gate_failures["invalid_gate_hold"] += not (close(g, p) and close(raw, p))
                    normalized = {**core, **row, "scope": "all_seen_training_reference", "gate_change": g - p,
                                  "abs_gate_change": abs(g - p), "abs_gate_deviation": abs(g - .5),
                                  "reliability_gap": abs(number(row.get("reliability_a")) - number(row.get("reliability_v")))}
                    gate_rows.append(normalized)
                    tables["gate_updates"].append(normalized)
        for step in range(expected_steps):
            actual = sorted(after for s, after in snapshots if s == step and after)
            expected = expected_updates(config, step)
            if actual != expected:
                audit(run, "gate_schedule", "warn", len(set(actual) ^ set(expected)), f"step={step}; expected={expected}; found={actual}")
            else:
                audit(run, "gate_schedule", "pass", detail=f"step={step}; updates={actual}")
            previous = snapshots.get((step, 0), {})
            gate_failures["initial_gate_class_coverage"] += set(previous) != set(range((step + 1) * classes_per_step))
            if step == 0 or config.get("fusion_mode") == "uniform":
                gate_failures["initial_uniform_gate"] += any(not close(r["audio_gate"], .5) or int(r["gate_version"]) != 0 for r in previous.values())
            for after in actual:
                current = snapshots[(step, after)]
                expected_n = (step + 1) * classes_per_step
                gate_failures["gate_class_coverage"] += set(current) != set(range(expected_n))
                for c, row in current.items():
                    if c in previous:
                        gate_failures["gate_previous_continuity"] += not close(row["previous_gate"], previous[c]["audio_gate"])
                        gate_failures["gate_version_continuity"] += int(row["gate_version"]) != int(previous[c]["gate_version"]) + 1
                previous = current
                selected = [r for r in gate_rows if int(r["step"]) == step and int(r["after_epoch"]) == after]
                tables["gate_update_summary"].append({**core, "step": step, "after_epoch": after, "active_from_epoch": after + 1,
                    "classes": len(selected), "valid_classes": sum(truth(r["valid"]) for r in selected),
                    **{f"mean_{key}": mean(r[key] for r in selected) for key in ("eta", "counts", "abs_gate_change", "abs_gate_deviation", "reliability_gap")}})
            for row in epoch_by_step[step]:
                available = [after for s, after in snapshots if s == step and after < int(row["epoch"])]
                active = snapshots[(step, max(available))] if available else {}
                if active:
                    values = [number(r["audio_gate"]) for r in active.values()]
                    gate_failures["epoch_gate_version"] += int(row["gate_version"]) != int(next(iter(active.values()))["gate_version"])
                    gate_failures["epoch_gate_statistics"] += not all(close(row.get(k), v) for k, v in (("gate_min", min(values)), ("gate_max", max(values)), ("gate_mean", mean(values))))
        for check, errors in sorted(gate_failures.items()):
            audit(run, check, "fail" if errors else "pass", errors)

        step_rows, tests = [], {}
        test_errors = 0
        summary = js(run, root / "summary.json", {}) or {}
        summary_steps, duplicate_summary = _unique(summary.get("steps", []), ("step",))
        summary_map = {int(r["step"]): r for r in summary_steps if finite(r.get("step"))}
        audit(run, "duplicate_summary_steps", "fail" if duplicate_summary else "pass", duplicate_summary)
        previous_task_best = {}
        forgetting_history_complete = True
        for step in range(expected_steps):
            result = js(run, root / f"step_{step}_test.json")
            if not result:
                forgetting_history_complete = False
                continue
            tests[step] = result
            values, best = class_by_step[step], best_epochs.get(step, {})
            gate = result.get("audio_gates", [])
            coverage = {int(r["class_id"]) for r in values} == set(range((step + 1) * classes_per_step))
            if not coverage:
                forgetting_history_complete = False
                audit(run, "test_class_coverage", "fail", detail=f"step={step}; expected={(step + 1) * classes_per_step}; rows={len(values)}")
            overall = _accuracy(values) if coverage else None
            old = _accuracy([r for r in values if r["group"] == "old"]) if coverage else None
            new = _accuracy([r for r in values if r["group"] == "new"]) if coverage else None
            macro = mean(r["f1"] for r in values) if coverage else None
            checks = [close(result.get("overall_acc"), overall), close(result.get("macro_f1"), macro)]
            task_accuracies = {task: _accuracy([r for r in values if int(r["class_id"]) // classes_per_step == task]) for task in range(step + 1)} if coverage else {}
            can_check_forgetting = step > 0 and forgetting_history_complete and coverage and all(task in previous_task_best and finite(task_accuracies[task]) for task in range(step))
            expected_forgetting = mean(previous_task_best[task] - task_accuracies[task] for task in range(step)) if can_check_forgetting else None
            if can_check_forgetting:
                consistent = close(result.get("forgetting"), expected_forgetting)
                checks.append(consistent)
                audit(run, "task_forgetting_formula", "pass" if consistent else "fail", int(not consistent),
                      f"step={step}; recomputed={expected_forgetting}; historical best before current step; negative values retained")
            for task, task_accuracy in task_accuracies.items():
                tables["task_metrics"].append({**core, "step": step, "task": task, "scope": "test", "accuracy": task_accuracy,
                    "best_before": previous_task_best.get(task), "forgetting": previous_task_best[task] - task_accuracy if task in previous_task_best and finite(task_accuracy) else None})
                if finite(task_accuracy):
                    previous_task_best[task] = max(previous_task_best.get(task, task_accuracy), task_accuracy)
            if step in summary_map:
                checks += [close(result.get(key), summary_map[step].get(key)) for key in ("overall_acc", "macro_f1", "gate_version")]
                if step:
                    checks.append(close(result.get("forgetting"), summary_map[step].get("forgetting")))
            else:
                checks.append(False)
            test_errors += not all(checks)
            available = [after for s, after in snapshots if s == step and after < int(best.get("epoch", 0))]
            active = snapshots[(step, max(available))] if available else {}
            for row in values:
                c = int(row["class_id"])
                estimate = active.get(c, {})
                row.update(best_audio_gate=gate[c] if c < len(gate) else None,
                           best_gate_version=result.get("gate_version"), best_epoch=best.get("epoch"),
                           gate_estimate_after_epoch=max(available) if available else None,
                           gate_estimate_reliability_a=estimate.get("reliability_a"),
                           gate_estimate_reliability_v=estimate.get("reliability_v"),
                           gate_estimate_valid=estimate.get("valid"))
            gate_match = len(active) == len(gate) == (step + 1) * classes_per_step and all(close(active.get(c, {}).get("audio_gate"), g) for c, g in enumerate(gate))
            if active:
                gate_match &= int(result.get("gate_version", -1)) == int(next(iter(active.values()))["gate_version"])
            audit(run, "test_best_gate", "pass" if gate_match else "warn", detail=f"step={step}; earliest maximum validation epoch={best.get('epoch')}; test version={result.get('gate_version')}; inferred snapshot after={max(available) if available else None}; PT confirmation separate")
            if step:
                previous = tests.get(step - 1, {})
                initial = snapshots.get((step, 0), {})
                inherited = previous.get("audio_gates", [])
                valid_inheritance = len(inherited) == step * classes_per_step and len(initial) == (step + 1) * classes_per_step
                valid_inheritance &= all(close(initial.get(c, {}).get("audio_gate"), g) for c, g in enumerate(inherited))
                valid_inheritance &= all(close(initial.get(c, {}).get("audio_gate"), .5) for c in range(step * classes_per_step, (step + 1) * classes_per_step))
                if initial:
                    valid_inheritance &= int(next(iter(initial.values()))["gate_version"]) == int(previous.get("gate_version", -1))
                audit(run, "initial_gate_inheritance", "pass" if valid_inheritance else "warn", detail=f"step={step}; old gate/version from previous best; new=.5")
            normalized = {**core, "step": step, "scope": "test", "overall_acc": result.get("overall_acc"),
                          "macro_f1": result.get("macro_f1"), "forgetting": result.get("forgetting"),
                          "old_accuracy": old, "new_accuracy": new, "test_support": sum(number(r["support"]) for r in values) if coverage else None,
                          "best_epoch": best.get("epoch"), "best_val_acc": best.get("val_acc"), "gate_version": result.get("gate_version"),
                          "gate_mean_abs_deviation": mean(abs(number(g) - .5) for g in gate), "complete_class_rows": coverage}
            step_rows.append(normalized)
            tables["steps"].append(normalized)
            for group in ("old", "new"):
                selected = [r for r in values if r["group"] == group]
                tables["new_old_metrics"].append({**core, "step": step, "group": group, "scope": "test", "class_count": len(selected),
                    "support": sum(number(r["support"]) for r in selected) if selected else None,
                    "tp": sum(number(r["tp"]) for r in selected) if selected else None, "accuracy": _accuracy(selected) if coverage else None})
        audit(run, "test_summary_class_consistency", "fail" if test_errors else "pass" if step_rows else "warn", test_errors)
        complete = len(step_rows) == expected_steps and all(finite(r["overall_acc"]) and r["complete_class_rows"] for r in step_rows) and not duplicate_classes and not test_errors
        values = {"average_incremental_accuracy": mean(r["overall_acc"] for r in step_rows),
                  "average_forgetting": mean(r["forgetting"] for r in step_rows if r["step"] > 0)}
        last = next((r for r in step_rows if r["step"] == expected_steps - 1), {})
        values.update({"final_accuracy": last.get("overall_acc"), "final_macro_f1": last.get("macro_f1"), "final_forgetting": last.get("forgetting"),
                       "final_old_accuracy": last.get("old_accuracy"), "final_new_accuracy": last.get("new_accuracy")})
        if complete:
            mismatch = [key for key in ("average_incremental_accuracy", "average_forgetting") if not close(values[key], summary.get(key)) and not (values[key] is None and summary.get(key) is None)]
            audit(run, "summary_averages", "fail" if mismatch else "pass", len(mismatch), ",".join(mismatch))
            if log["averages"]:
                logged = log["averages"][-1]
                mismatch = [key for key, value in zip(("average_incremental_accuracy", "average_forgetting"), logged)
                            if not close(values[key], value) and not (values[key] is None and not finite(value))]
                audit(run, "log_summary_averages", "fail" if mismatch else "pass", len(mismatch), ",".join(mismatch))
            else:
                audit(run, "log_summary_averages", "warn", detail="No final summary line; results may be complete but log was not transferred in full")
        log_test_map = dict(log["tests"])
        mismatches = [step for step, result in tests.items() if step in log_test_map and not close(result.get("overall_acc"), log_test_map[step])]
        audit(run, "log_test_consistency", "fail" if mismatches else "pass" if log_test_map else "warn", len(mismatches), f"mismatched steps={mismatches}")
        audit(run, "result_completeness", "pass" if complete else "warn", expected_steps - len(step_rows), "AIA/final aggregate emitted only for complete, unambiguous test trajectories")
        run_row = {**core, "expected_steps": expected_steps, "observed_steps": len(step_rows), "complete": complete,
                   **{key: value if complete else None for key, value in values.items()}, "run_dir": run["run_dir"], "metrics_dir": str(root)}
        tables["run_summary"].append(run_row)

        history, dup_history = _unique(csv(run, root / "cl_history/class_history.csv"), ("step", "epoch", "class_id"))
        audit(run, "duplicate_cl_history", "fail" if dup_history else "pass", dup_history)
        by_observation, start_history = defaultdict(list), []
        for row in history:
            if not all(finite(row.get(k)) for k in ("step", "epoch", "class_id")):
                continue
            step, epoch = int(row["step"]), int(row["epoch"])
            by_observation[(step, epoch)].append(row)
            if epoch == 0 and "task_start" in str(row.get("events", "")):
                start_history.append(row)
            if epoch == best_epochs.get(step, {}).get("epoch"):
                tables["cl_history_best"].append({**core, **row, "scope": "fixed_old_memory_and_old_classes",
                    "is_selected_best_epoch": True, "source": "CSV matched to earliest validation maximum"})
        for (step, epoch), rows in by_observation.items():
            comparable = [r for r in rows if truth(r.get("comparable"))]
            keys = ("teacher_R_a", "teacher_R_v", "student_R_a", "student_R_v", "signed_drop_a", "signed_drop_v", "drop_asymmetry",
                    "teacher_probe_accuracy_a", "teacher_probe_accuracy_v", "student_probe_accuracy_a", "student_probe_accuracy_v",
                    "student_audio_only_correct_rate", "student_visual_only_correct_rate", "student_both_wrong_rate")
            observation = {**core, "step": step, "epoch": epoch, "events": rows[0].get("events"),
                "scope": "fixed_old_memory_and_old_classes", "classes": len(rows), "comparable_classes": len(comparable),
                "memory_count": sum(number(r["memory_count"]) for r in rows) if all(finite(r.get("memory_count")) for r in rows) else None,
                **{f"mean_{key}": mean(r.get(key) for r in comparable) for key in keys}}
            for role in ("teacher", "student"):
                valid = [r for r in rows if truth(r.get(f"{role}_classification_valid")) and finite(r.get(f"{role}_classification_support"))]
                support = sum(number(r[f"{role}_classification_support"]) for r in valid)
                observation[f"{role}_classification_support"] = support if valid else None
                for metric in ("old_only_accuracy", "all_seen_accuracy", "old_to_new_rate"):
                    key = f"{role}_{metric}"
                    observation[f"weighted_{key}"] = sum(number(r[key]) * number(r[f"{role}_classification_support"]) for r in valid) / support if support and all(finite(r.get(key)) for r in valid) else None
            tables["cl_history_summary"].append(observation)
        if config.get("record_cl_history", True):
            for step in range(1, expected_steps):
                selected = by_observation.get((step, best_epochs.get(step, {}).get("epoch")), [])
                coverage = {int(r["class_id"]) for r in selected} == set(range(step * classes_per_step))
                audit(run, "cl_history_selected_best_coverage", "pass" if coverage else "warn", detail=f"step={step}; selected best epoch={best_epochs.get(step, {}).get('epoch')}; observed old classes={len(selected)}")
        for path in sorted((root / "prototype_bank").glob("*.csv")):
            raw_prototypes = csv(run, path)
            prototype_rows = [r for r in raw_prototypes if all(_integer(r.get(key)) for key in ("step", "after_epoch", "class_id"))
                              and int(r["step"]) < expected_steps and int(r["after_epoch"]) <= max_epochs
                              and int(r["class_id"]) < (int(r["step"]) + 1) * classes_per_step]
            invalid_count = len(raw_prototypes) - len(prototype_rows)
            audit(run, "prototype_row_identifiers", "fail" if invalid_count else "pass", invalid_count,
                  "Missing/nonfinite/fractional/out-of-range step, after_epoch or class_id excluded", source=path)
            prototype_rows, duplicate_count = _unique(prototype_rows, ("step", "after_epoch", "class_id"))
            audit(run, "duplicate_prototype_rows", "fail" if duplicate_count else "pass", duplicate_count, source=path)
            prototype_groups = defaultdict(list)
            for row in prototype_rows:
                tables["prototype_bank"].append({**core, **row, "scope": "training_prototype_diagnostics", "source_file": path.name})
                prototype_groups[(int(row["step"]), int(row["after_epoch"]))].append(row)
            if len(prototype_groups) > 1:
                audit(run, "prototype_snapshot_identity", "fail", len(prototype_groups),
                      "File contains multiple step/epoch identities; each cohort summarized separately", source=path)
            for (step, after), prototype_rows in prototype_groups.items():
                gate_snapshot = snapshots.get((step, after), {})
                tables["prototype_bank_summary"].append({**core, "step": step, "after_epoch": after, "scope": "training_prototype_diagnostics",
                    "classes": len(prototype_rows), "history_used_classes": sum(truth(r.get("history_used")) for r in prototype_rows),
                    "history_used_and_gate_valid_classes": sum(truth(r.get("history_used")) and truth(gate_snapshot.get(int(r["class_id"]), {}).get("valid")) for r in prototype_rows),
                    **{f"mean_{key}": mean(r.get(key) for r in prototype_rows) for key in ("birth_count", "anchor_count", "prior_strength_effective", "prototype_beta", "drift_audio_norm", "drift_visual_norm", "drift_audio_dispersion", "drift_visual_dispersion")},
                    "reason_counts": json.dumps(dict(Counter(str(r.get("reason")) for r in prototype_rows)), sort_keys=True)})
        cache[run["run_id"]] = {"summary": run_row, "steps": step_rows, "classes": class_by_step,
                                "start_history": start_history, "metrics_dir": root, "first_epoch": epoch_by_step.get(0, [])}

    _paired_analysis(runs, cache, tables, audit)
    for name in ("log_summary", "run_summary", "steps", "classes", "epoch_losses", "audit", "gate_updates", "gate_update_summary",
                 "cl_history_summary", "cl_history_best", "class_deltas", "paired_metrics", "method_summary", "new_old_metrics",
                 "subgroup_membership", "subgroup_metrics", "prototype_bank", "prototype_bank_summary", "task_metrics"):
        write_csv(output / "tables" / f"{name}.csv", tables[name], fieldnames=None if tables[name] else [*IDENTITY, "status"])
    result = {"schema_version": 1, "counts": {key: len(value) for key, value in tables.items()}, "warnings": warnings,
              "notes": ["Accuracy values are fractions; _pp are percentage-point differences.",
                        "Incomplete/ambiguous trajectories have no primary run aggregates.",
                        "Seed is the independent replication unit; class-step rows are descriptive repeated measurements.",
                        "Memory probes are not test accuracy. Quantile groups use uniform task-start memory only.",
                        "No checkpoint, gate, epoch, or hyperparameter is selected using test results."]}
    write_json(output / "offline_summary.json", result)
    return result


def _paired_analysis(runs, cache, tables, audit):
    """Pair within protocol and seed; refuse an ambiguous/missing baseline."""
    for run in runs:
        if run["config"].get("fusion_mode") == "uniform":
            continue
        core, data = identity(run), cache[run["run_id"]]
        baseline = resolve_baseline(run, runs)
        if baseline is None:
            audit(run, "matched_baseline", "warn", detail="No unique uniform baseline with same protocol and seed; paired comparisons skipped")
            continue
        base = cache[baseline["run_id"]]
        pairing = {"baseline_run_id": baseline["run_id"], "baseline_method": baseline["method"]}
        audit(run, "matched_baseline", "pass", detail=baseline["run_id"])
        # Only the first task is expected to match. Later inherited models differ.
        a0, b0 = data["first_epoch"], base["first_epoch"]
        b0map = {int(r["epoch"]): r for r in b0}
        first_differences = sum(any(not close(r.get(k), b0map[int(r["epoch"])].get(k), 1e-5) for k in ("train_loss", "val_acc"))
                                for r in a0 if int(r["epoch"]) in b0map)
        audit(run, "matched_first_task", "warn" if first_differences else "pass" if a0 and b0 else "warn", first_differences,
              "Same-seed first-task epoch losses/validation; GPU nondeterminism or differing code versions may require investigation")
        for metric in RUN_METRICS:
            value, reference = data["summary"].get(metric), base["summary"].get(metric)
            if finite(value) and finite(reference):
                tables["paired_metrics"].append({**core, **pairing, "metric": metric, "value": value, "baseline_value": reference,
                    "delta": value - reference, "delta_pp": 100 * (value - reference),
                    "improvement_direction": "lower" if "forgetting" in metric else "higher"})
        baseline_start = defaultdict(list)
        for row in base["start_history"]:
            if not truth(row.get("comparable")):
                continue
            ra, rv = number(row.get("teacher_R_a")), number(row.get("teacher_R_v"))
            audio_only, visual_only = number(row.get("teacher_audio_only_correct_rate")), number(row.get("teacher_visual_only_correct_rate"))
            baseline_start[int(row["step"])].append({"class_id": int(row["class_id"]),
                "modality_gap": abs(ra - rv), "complementarity": min(audio_only, visual_only),
                "reference_difficulty": 1 - number(row.get("teacher_old_only_accuracy")) if truth(row.get("teacher_classification_valid")) else float("nan")})
        for step, rows in data["classes"].items():
            baseline_rows = {int(r["class_id"]): r for r in base["classes"].get(step, [])}
            deltas = []
            for row in rows:
                b = baseline_rows.get(int(row["class_id"]))
                if not b or not close(row["support"], b["support"]):
                    continue
                delta = {**core, **pairing, "step": step, "class_id": row["class_id"], "category_name": row.get("category_name"),
                    "class_age": row["class_age"], "group": row["group"], "scope": "test", "support": row["support"], "tp": row["tp"], "baseline_tp": b["tp"],
                    "recall": row["recall"], "baseline_recall": b["recall"], "delta_recall_pp": 100 * (row["recall"] - b["recall"]),
                    "f1": row["f1"], "baseline_f1": b["f1"], "delta_f1_pp": 100 * (row["f1"] - b["f1"])}
                deltas.append(delta)
                tables["class_deltas"].append(delta)
            assignments = {"class_age": {int(r["class_id"]): str(r["class_age"]) for r in deltas},
                           "new_old": {int(r["class_id"]): r["group"] for r in deltas}}
            definitions = baseline_start[step]
            for variable in ("modality_gap", "complementarity", "reference_difficulty"):
                assignments[variable] = _tertiles(definitions, variable)
            # IDs include checkpoint provenance, so compare actual cohort rather than hash.
            try:
                ref_a = read_json(data["metrics_dir"] / "cl_history" / f"step_{step}_reference.json", {})
                ref_b = read_json(base["metrics_dir"] / "cl_history" / f"step_{step}_reference.json", {})
            except (ValueError, OSError, UnicodeError) as exc:
                ref_a, ref_b = {}, {}
                audit(run, "reference_json_read", "warn", detail=f"step={step}; {type(exc).__name__}: {exc}")
            same_reference = bool(ref_a and ref_b) and all(ref_a.get(k) == ref_b.get(k) for k in ("sample_ids", "labels", "candidate_class_ids", "included"))
            for variable, mapping in assignments.items():
                selected = defaultdict(list)
                for delta in deltas:
                    c = int(delta["class_id"])
                    if c not in mapping:
                        continue
                    group = mapping[c]
                    selected[group].append(delta)
                    definition = next((r for r in definitions if r["class_id"] == c), {})
                    tables["subgroup_membership"].append({**core, **pairing, "step": step, "class_id": c, "variable": variable,
                        "bin": group, "baseline_reference_value": definition.get(variable), "class_age": delta["class_age"],
                        "definition_scope": "class_identity" if variable in ("class_age", "new_old") else "uniform_task_start_old_memory",
                        "reference_cohort_matches": same_reference, "exploratory": True})
                # Emit empty tertiles to avoid concealing tied/missing groups.
                groups = ("low", "middle", "high") if variable in ("modality_gap", "complementarity", "reference_difficulty") else sorted(selected)
                for group in groups:
                    members = selected[group]
                    support = sum(number(r["support"]) for r in members)
                    value = sum(number(r["tp"]) for r in members) / support if support else None
                    reference = sum(number(r["baseline_tp"]) for r in members) / support if support else None
                    tables["subgroup_metrics"].append({**core, **pairing, "step": step, "variable": variable, "bin": group,
                        "scope": "test_outcome", "definition_scope": "class_identity" if variable in ("class_age", "new_old") else "uniform_task_start_old_memory",
                        "classes": len(members), "support": support, "accuracy": value, "baseline_accuracy": reference,
                        "delta_accuracy_pp": 100 * (value - reference) if support else None, "mean_delta_f1_pp": mean(r["delta_f1_pp"] for r in members),
                        "improved_classes": sum(r["delta_recall_pp"] > 1e-5 for r in members), "worse_classes": sum(r["delta_recall_pp"] < -1e-5 for r in members),
                        "reference_cohort_matches": same_reference, "exploratory": True})
    summary_groups = defaultdict(list)
    for row in tables["run_summary"]:
        summary_groups[(row["method_id"], row["protocol_id"])].append(row)
    for (method_id, protocol), rows in summary_groups.items():
        for metric in RUN_METRICS:
            valid = [r for r in rows if finite(r.get(metric))]
            paired = [r for r in tables["paired_metrics"] if r["method_id"] == method_id and r["protocol_id"] == protocol and r["metric"] == metric]
            seed_counts = Counter(r["seed"] for r in valid)
            # Repeated runs with the same seed are not independent replications.
            valid = [r for r in valid if seed_counts[r["seed"]] == 1]
            paired_counts = Counter(r["seed"] for r in paired)
            paired = [r for r in paired if paired_counts[r["seed"]] == 1]
            values, differences = [r[metric] for r in valid], [r["delta"] for r in paired]
            tables["method_summary"].append({"method": rows[0]["method"], "method_id": method_id, "protocol_id": protocol, "metric": metric,
                "n_seeds": len(valid), "mean": mean(values), "sd": sd(values), "baseline_method": paired[0]["baseline_method"] if paired else None,
                "paired_n_seeds": len(paired), "delta_mean": mean(differences), "delta_sd": sd(differences),
                "delta_mean_pp": 100 * mean(differences) if differences else None,
                "improved_seeds": sum(r["delta"] < 0 if "forgetting" in metric else r["delta"] > 0 for r in paired),
                "seed_deltas": json.dumps({str(r["seed"]): r["delta_pp"] for r in paired}, sort_keys=True)})
