"""Synthetic prediction tests independent of GPU/checkpoint availability."""

import copy
from pathlib import Path

import numpy as np
import pytest

from experiments_phase_9.analysis.common import read_csv, write_json
from experiments_phase_9.analysis.reporting import (
    build_report, prediction_statistics, validation_groups, _group_seed_summary)


def logits(predictions, classes=4):
    return np.eye(classes, dtype=np.float32)[predictions] * 3


def cache(root, job, predictions, uniform=None, val=None, gate=0.5):
    path = root / "cache" / job["job_id"]
    path.mkdir(parents=True)
    write_json(path / "status.json", {**job, "status": "complete", "fingerprint": "synthetic",
                                      "splits": ["test", "val"] if val is not None else ["test"]})
    write_json(path / "checkpoint.json", {"fingerprint": "synthetic", "checkpoint_epoch": 80,
                                         "step": job["step"], "num_classes": 4, "gate_version": 2,
                                         "split_diagnostics": {
                                             split: {"expected_unique_samples": 8, "num_samples": 8,
                                                     "missing_feature_pairs": 0}
                                             for split in (["test", "val"] if val is not None else ["test"])}})
    labels = np.repeat(np.arange(4), 2)
    arrays = {"ids": np.array(["test_" + str(i) for i in range(8)]), "labels": labels,
              "class_ids": np.arange(4), "gate": np.full(4, gate, np.float32),
              "logits_actual": logits(predictions), "logits_uniform": logits(uniform if uniform is not None else predictions),
              "logits_audio": logits(predictions), "logits_visual": logits(predictions)}
    np.savez_compressed(path / "test.npz", **arrays)
    if val is not None:
        arrays.update(ids=np.array(["val_" + str(i) for i in range(8)]), logits_uniform=logits(val),
                      logits_actual=logits(val),
                      logits_audio=logits([0, 0, 1, 1, 2, 3, 3, 3]),
                      logits_visual=logits([1, 1, 1, 2, 2, 2, 3, 3]))
        np.savez_compressed(path / "val.npz", **arrays)


@pytest.fixture
def prediction_fixture(tmp_path):
    runs = [
        {"run_id": "baseline", "method": "uniform", "method_id": "u", "protocol_id": "p", "seed": 42,
         "config": {"fusion_mode": "uniform", "class_num_per_step": 2}},
        {"run_id": "dynamic", "method": "periodic", "method_id": "d", "protocol_id": "p", "seed": 42,
         "config": {"fusion_mode": "periodic", "class_num_per_step": 2}},
    ]
    jobs = [{"job_id": run["run_id"] + "_best_1", "run_id": run["run_id"], "step": 1, "kind": "best"} for run in runs]
    cache(tmp_path, jobs[0], [0, 1, 1, 2, 2, 3, 3, 0], val=[1, 1, 1, 2, 2, 2, 3, 3])
    cache(tmp_path, jobs[1], [0, 0, 1, 1, 2, 2, 3, 3],
          uniform=[0, 0, 1, 1, 2, 3, 3, 0], gate=0.8)
    return {"output_root": str(tmp_path), "runs": runs, "jobs": jobs,
            "options": {"plots": False, "mode": "all"}}


def test_confusion_new_old_uses_full_candidates():
    stats, classes, matrix = prediction_statistics(
        np.array([0, 0, 1, 1, 2, 2, 3, 3]), np.array([0, 2, 1, 3, 0, 2, 1, 3]), 4, 2)
    assert stats["accuracy"] == 0.5
    assert stats["old_to_new"] == stats["new_to_old"] == 2
    assert stats["old_to_new_rate"] == stats["new_to_old_rate"] == 0.5
    assert sum(r["fp"] for r in classes) == 4
    assert matrix.sum() == 8


def test_report_exact_decomposition_and_all_groups(prediction_fixture):
    result = build_report(prediction_fixture)
    root = Path(prediction_fixture["output_root"])
    rows = read_csv(root / "tables/mechanism_decomposition.csv")
    acc = next(r for r in rows if r["metric"] == "accuracy")
    assert acc["total_delta_pp"] == 50
    assert acc["fixed_model_gate_effect_pp"] == 25
    assert acc["trained_parameters_difference_pp"] == 25
    assert acc["identity_residual_pp"] == 0
    assert result["test_caches"] == 2
    groups = read_csv(root / "tables/validation_group_metrics.csv")
    assert len(groups) == 9
    difficult = [r for r in groups if r["axis"] == "validation_difficulty"]
    assert {r["group"] for r in difficult} == {"hard", "middle", "easy"}
    assert next(r for r in difficult if r["group"] == "easy")["class_count"] == 0
    assert next(r for r in difficult if r["group"] == "easy")["accuracy"] == ""
    pairs = read_csv(root / "tables/inference_pairs.csv")
    overall = next(r for r in pairs if r["comparison"] == "method_actual_minus_baseline_uniform" and r["class_id"] == -1)
    assert overall["corrected"] == 4 and overall["broken"] == 0
    assert len(read_csv(root / "tables/inference_class_deltas.csv")) == 4
    assert (root / "report.md").is_file() and (root / "report.html").is_file()


def test_groups_depend_only_on_baseline_validation(prediction_fixture):
    build_report(prediction_fixture)
    root = Path(prediction_fixture["output_root"])
    before = read_csv(root / "tables/validation_group_membership.csv")
    target = root / "cache/dynamic_best_1/test.npz"
    with np.load(target) as saved:
        arrays = dict(saved)
    arrays["logits_actual"] = logits([1, 1, 2, 2, 3, 3, 0, 0])
    np.savez_compressed(target, **arrays)
    build_report(prediction_fixture)
    assert before == read_csv(root / "tables/validation_group_membership.csv")


def test_mismatched_cohort_rejected_not_intersected(prediction_fixture):
    root = Path(prediction_fixture["output_root"])
    target = root / "cache/dynamic_best_1/test.npz"
    with np.load(target) as saved:
        arrays = dict(saved)
    arrays["ids"] = np.array(["wrong_" + str(i) for i in range(8)])
    np.savez_compressed(target, **arrays)
    result = build_report(prediction_fixture)
    assert any("disagree on ids" in warning for warning in result["warnings"])
    assert not read_csv(root / "tables/mechanism_decomposition.csv")


def test_ambiguous_baseline_and_missing_val_not_fabricated(prediction_fixture):
    root = Path(prediction_fixture["output_root"])
    (root / "cache/baseline_best_1/val.npz").unlink()
    result = build_report(prediction_fixture)
    assert any("validation cache absent" in warning for warning in result["warnings"])
    assert result["tables"]["validation_group_metrics"] == 0
    duplicate = copy.deepcopy(prediction_fixture["runs"][0])
    duplicate["run_id"] = "baseline_duplicate"
    prediction_fixture["runs"].append(duplicate)
    result = build_report(prediction_fixture)
    assert result["tables"]["mechanism_decomposition"] == 0
    assert any("ambiguous" in warning for warning in result["warnings"])


def test_offline_ignores_stale_caches_and_html_is_escaped(prediction_fixture):
    prediction_fixture["options"]["mode"] = "offline"
    result = build_report(prediction_fixture)
    assert result["test_caches"] == 0
    assert result["tables"]["mechanism_decomposition"] == 0
    assert any("Offline mode" in w for w in result["warnings"])
    # Run IDs are untrusted data: a crafted warning cannot become active HTML.
    prediction_fixture["options"]["mode"] = "all"
    prediction_fixture["jobs"].append({"job_id": "<script>alert(1)</script>", "run_id": "unknown", "step": 0, "kind": "last"})
    build_report(prediction_fixture)
    html = (Path(prediction_fixture["output_root"]) / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_no_inference_jobs_still_writes_report(tmp_path):
    result = build_report({"output_root": str(tmp_path), "runs": [], "jobs": [], "options": {"plots": False}})
    assert result["test_caches"] == 0
    assert (tmp_path / "report.md").is_file()


def test_current_invocation_and_requested_splits_required(prediction_fixture):
    prediction_fixture["invocation_id"] = "new-run"
    result = build_report(prediction_fixture)
    assert result["test_caches"] == 0
    assert any("not verified" in w for w in result["warnings"])
    prediction_fixture.pop("invocation_id")
    prediction_fixture["options"]["splits"] = ["test"]
    result = build_report(prediction_fixture)
    assert result["test_caches"] == 2
    assert result["tables"]["validation_group_metrics"] == 0
    assert any("validation cache absent" in w for w in result["warnings"])


def test_complementarity_measures_oracle_gain_not_just_disagreement():
    arrays = {"class_ids": np.arange(2), "labels": np.array([0, 0, 1, 1]),
              "logits_actual": logits([0, 0, 1, 1], 2),
              "logits_uniform": logits([0, 0, 1, 1], 2),
              "logits_audio": logits([0, 0, 1, 0], 2),
              "logits_visual": logits([1, 1, 0, 1], 2)}
    rows = validation_groups(arrays)
    c0 = next(r for r in rows if r["class_id"] == 0)
    assert c0["exclusive_correct_rate"] == 1
    assert c0["complementarity"] == 0  # Audio dominates; visual offers no extra success.
    c1 = next(r for r in rows if r["class_id"] == 1)
    assert c1["complementarity"] == 0.5  # Disjoint successes give genuine oracle headroom.


def test_reproduction_audit_flags_mismatch_and_last_not_compared(prediction_fixture):
    root = Path(prediction_fixture["output_root"])
    metrics = root / "original_metrics"
    prediction_fixture["runs"][1]["metrics_dir"] = str(metrics)
    write_json(metrics / "step_1_test.json", {"overall_acc": 1.0, "macro_f1": 1.0, "gate_version": 2})
    result = build_report(prediction_fixture)
    assert result["reproduction_failures"] == 0
    write_json(metrics / "step_1_test.json", {"overall_acc": 0.2, "macro_f1": 1.0, "gate_version": 2})
    result = build_report(prediction_fixture)
    assert result["reproduction_failures"] == 1
    assert any("reproduction mismatch" in w for w in result["warnings"])
    job = {**prediction_fixture["jobs"][1], "kind": "last", "job_id": "dynamic_last_1"}
    prediction_fixture["jobs"] = [job]
    cache(root, job, [0, 0, 1, 1, 2, 2, 3, 3], gate=0.8)
    result = build_report(prediction_fixture)
    assert result["reproduction_failures"] == 0
    assert not read_csv(root / "tables/inference_reproduction_audit.csv")


def test_seed_summary_does_not_treat_duplicate_runs_as_independent():
    core = {"method_id": "d", "protocol_id": "p", "kind": "best", "step": 1,
            "axis": "validation_difficulty", "group": "hard"}
    rows = [{**core, "seed": seed, "delta_accuracy_pp": delta}
            for seed, delta in [(42, 2), (43, -2), (44, 10), (44, 12)]]
    summary = _group_seed_summary(rows)[0]
    assert summary["n_seeds"] == 2 and summary["ambiguous_seeds_excluded"] == 1
    assert summary["delta_accuracy_mean_pp"] == 0
    assert summary["delta_accuracy_sd_pp"] == pytest.approx(np.sqrt(8))
    assert summary["improved_seeds"] == summary["degraded_seeds"] == 1


def test_optional_figures_export_png_and_pdf(prediction_fixture):
    pytest.importorskip("matplotlib")
    prediction_fixture["options"]["plots"] = True
    result = build_report(prediction_fixture)
    assert "figures/gate_deviation.png" in result["figures"]
    assert "figures/class_recall_delta_0.pdf" in result["figures"]
    assert "figures/validation_group_delta_0.png" in result["figures"]
    assert not any("Figure generation failed" in w for w in result["warnings"])


def test_partial_feature_and_missing_class_coverage_reported(prediction_fixture):
    root = Path(prediction_fixture["output_root"])
    target = root / "cache/dynamic_best_1/test.npz"
    with np.load(target) as saved:
        arrays = dict(saved)
    for key in ("ids", "labels") + tuple("logits_" + v for v in ("actual", "uniform", "audio", "visual")):
        arrays[key] = arrays[key][:6]  # Entire class 3 disappears through feature filtering.
    np.savez_compressed(target, **arrays)
    result = build_report(prediction_fixture)
    assert result["coverage_issues"] > 0
    coverage = read_csv(root / "tables/inference_coverage.csv")
    row = next(r for r in coverage if r["run_id"] == "dynamic" and r["split"] == "test")
    assert row["status"] == "partial" and row["observed_samples"] == 6
    assert row["expected_unique_samples"] == 8 and row["missing_class_ids"] == "[3]"
    assert any("incomplete evaluation cohort" in w for w in result["warnings"])
    assert any("requested validation prediction cache missing" in w for w in result["warnings"])
