"""Small real HDF5/checkpoint tests for offline analysis, without training."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments_phase_9.analysis import inference
from experiments_phase_9.analysis.inference import (
    AudioFeatureCache, _sha256, export_checkpoint_history, fixed_gate_logits, run_worker,
)
from experiments_phase_9.class_fusion.checkpoints import save_checkpoint
from experiments_phase_9.dataloader_ours import IcaAVELoader
from experiments_phase_9.tests.test_training import make_fixture
from experiments_phase_9.train_incremental_fusion_modular import build_parser
from model.audio_visual_model_incremental_class_fusion import ClassFusionAudioVisualNet, class_conditional_logits


def make_analysis_fixture(root, head="linear"):
    features, meta = make_fixture(root)
    args = build_parser().parse_args(["--feature_root", str(features), "--meta_root", str(meta),
        "--num_classes", "6", "--class_num_per_step", "2", "--fusion_classifier", head,
        "--fusion_hidden_dim", "9", "--fusion_chunk_size", "2", "--device", "cpu"])
    torch.manual_seed(18)
    model = ClassFusionAudioVisualNet(args, 4).eval()
    model.set_fusion_gate(torch.tensor([0.12, 0.85, 0.4, 0.65]))
    checkpoint = root / "save/run/step_1_best_model.pt"
    save_checkpoint(checkpoint, model, args, 1, 7, 0.25)
    output = root / "results"
    manifest = {"schema_version": 1, "phase_root": str(root), "output_root": str(output),
        "save_root": str(root / "save"), "logs_root": str(root / "logs"),
        "options": {"batch_size": 3, "num_workers": 0, "splits": ["test", "val"],
                    "feature_root": None, "meta_root": None, "force": False, "torch_threads": 2},
        "runs": [{"run_id": "run", "config": vars(args)}],
        "jobs": [{"job_id": "run_step_1_best", "run_id": "run", "checkpoint": str(checkpoint), "step": 1, "kind": "best"}]}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return args, model, manifest, path


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_checkpoint_inference_matches_forward_fixed_controls_and_split_boundary(tmp_path, head):
    args, model, manifest, path = make_analysis_fixture(tmp_path, head)
    # No train/replay metadata is available to the analysis process.
    for filename in ("all_id_category_dict.npy", "all_classId_vid_dict.npy"):
        split_data = np.load(Path(args.meta_root) / filename, allow_pickle=True).item()
        del split_data["train"]
        np.save(Path(args.meta_root) / filename, split_data)
    checkpoint = manifest["jobs"][0]["checkpoint"]
    before = _sha256(checkpoint)
    result = run_worker(path)
    assert result["failed"] == 0
    assert not result["jobs"][0]["cached"]
    assert _sha256(checkpoint) == before
    directory = Path(manifest["output_root"]) / "cache/run_step_1_best"
    with np.load(directory / "test.npz", allow_pickle=False) as saved:
        assert saved["ids"].dtype.kind == "U"
        assert saved["ids"].tolist() == [f"test_{c}_0" for c in range(4)]
        assert saved["labels"].tolist() == list(range(4))
        assert saved["class_ids"].tolist() == list(range(4))
        dataset = IcaAVELoader(args, "test", incremental_step=1)
        try:
            for index in range(len(dataset)):
                (visual, audio), _ = dataset[index]
                with torch.inference_mode():
                    a, v, _, _ = model.extract_branch_features(visual[None], audio[None])
                    expected = model(visual=visual[None], audio=audio[None])
                    np.testing.assert_allclose(saved["logits_actual"][index], expected.numpy()[0], rtol=2e-5, atol=2e-5)
                    for name, gate in (("uniform", 0.5), ("audio", 1.0), ("visual", 0.0)):
                        expected = class_conditional_logits(a, v, model.classifier, torch.full((4,), gate), 2)
                        np.testing.assert_allclose(saved["logits_" + name][index], expected.numpy()[0], rtol=2e-5, atol=2e-5)
            if head == "linear":
                np.testing.assert_allclose(saved["logits_actual"], saved["logits_uniform"] + saved["d"] * (2 * saved["gate"] - 1))
            else:
                assert "d" not in saved
        finally:
            dataset.close_visual_features_h5()
    metadata = json.loads((directory / "checkpoint.json").read_text())
    assert metadata["checkpoint_epoch"] == 7 and metadata["gate_version"] == 1
    assert metadata["checkpoint_sha256"] == before
    assert metadata["split_diagnostics"]["test"]["missing_feature_pairs"] == 0


def test_resume_invalidation_and_one_audio_dictionary_across_jobs(tmp_path, monkeypatch):
    args, model, manifest, path = make_analysis_fixture(tmp_path)
    first_job = manifest["jobs"][0]
    manifest["jobs"].append({**first_job, "job_id": "run_step_1_copy", "kind": "last"})
    path.write_text(json.dumps(manifest))
    loaded = []
    original_load = np.load

    def load_count(file, *args, **kwargs):
        if str(file).endswith("audio_pretrained_feature_dict.npy"):
            loaded.append(str(file))
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", load_count)
    first = run_worker(path)
    assert first["failed"] == 0 and len(loaded) == 1
    second = run_worker(path)
    assert all(item["cached"] for item in second["jobs"]) and len(loaded) == 1
    manifest["invocation_id"] = "new-invocation"
    path.write_text(json.dumps(manifest))
    resumed = run_worker(path)
    assert all(item["cached"] for item in resumed["jobs"])
    status = json.loads((Path(manifest["output_root"]) / "cache/run_step_1_best/status.json").read_text())
    assert status["invocation_id"] == "new-invocation"
    feature_path = Path(args.feature_root) / "visual_features.h5"
    stat = feature_path.stat()
    os.utime(feature_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    third = run_worker(path)
    assert not any(item["cached"] for item in third["jobs"])
    assert third["jobs"][0]["fingerprint"] != first["jobs"][0]["fingerprint"]
    directory = Path(manifest["output_root"]) / "cache/run_step_1_best"
    # Half-complete work is never counted as a successful cached result.
    status = json.loads((directory / "status.json").read_text())
    status["status"] = "failed"
    (directory / "status.json").write_text(json.dumps(status))
    fourth = run_worker(path)
    assert not fourth["jobs"][0]["cached"] and fourth["jobs"][1]["cached"]
    # A truncated npz also triggers recomputation rather than silent resume.
    (directory / "test.npz").write_bytes(b"broken")
    fifth = run_worker(path)
    assert not fifth["jobs"][0]["cached"]


def test_bad_checkpoint_fails_but_next_job_completes_and_no_training_mutation(tmp_path):
    args, model, manifest, path = make_analysis_fixture(tmp_path)
    valid = manifest["jobs"][0]
    invalid = tmp_path / "save/run/bad.pt"
    payload = torch.load(valid["checkpoint"], weights_only=True)
    payload["step"] = 2
    torch.save(payload, invalid)
    manifest["jobs"].insert(0, {**valid, "job_id": "invalid", "checkpoint": str(invalid)})
    path.write_text(json.dumps(manifest))
    before = {str(item): _sha256(item) for item in (invalid, Path(valid["checkpoint"]))}
    result = run_worker(path)
    assert result["failed"] == 1
    assert result["jobs"][0]["status"] == "failed" and "step disagrees" in result["jobs"][0]["error"]
    assert result["jobs"][1]["status"] == "complete"
    assert before == {str(item): _sha256(item) for item in (invalid, Path(valid["checkpoint"]))}


def test_manifest_configuration_mismatch_and_rank_partition(tmp_path):
    args, model, manifest, path = make_analysis_fixture(tmp_path)
    manifest["runs"][0]["config"]["lam_I"] = 99
    path.write_text(json.dumps(manifest))
    result = run_worker(path)
    assert result["failed"] == 1 and "lam_I" in result["jobs"][0]["error"]
    empty = run_worker(path, rank=1, world_size=2)
    assert empty["jobs"] == [] and empty["failed"] == 0


def test_history_exports_excluded_cohort_and_provenance(tmp_path):
    reference = {"reference_id": "ref", "num_old_classes": 2, "memory_counts": torch.tensor([2, 1]),
        "teacher_gate": torch.tensor([0.3, 0.7]), "included_mask": torch.tensor([True, False, True]),
        "sample_labels": torch.tensor([0, 0, 1]), "sample_ids": ["old_0", "excluded", "old_1"],
        "teacher_reliability": {"reliability_a": torch.tensor([0.8, 0.5]), "sample_probability_a": torch.tensor([0.8, 0.5])},
        "teacher_classification": {"sample_prediction_all_seen": torch.tensor([0, 1])},
        "teacher_source": {"checkpoint_sha256": "teacher"}}
    observation = {"reference_id": "ref", "step": 1, "epoch": 7, "audio_gate": torch.tensor([0.4, 0.6, 0.5, 0.5]),
                   "reliability": {"reliability_a": torch.tensor([0.7, 0.3]), "sample_probability_a": torch.tensor([0.7, 0.3])}}
    summary = export_checkpoint_history(tmp_path, {"step": 1, "epoch": 7, "cl_history": {"reference": reference, "observation": observation}})
    assert summary["scope"] == "fixed_old_training_memory" and summary["num_included_samples"] == 2
    content = (tmp_path / "history_samples.csv").read_text()
    assert "old_0" in content and "old_1" in content and "excluded" not in content
    assert "student_reliability_sample_probability_a" in content
    observation["epoch"] = 8
    with pytest.raises(ValueError, match="step/epoch"):
        export_checkpoint_history(tmp_path, {"step": 1, "epoch": 7, "cl_history": {"reference": reference, "observation": observation}})


def test_mutating_input_during_inference_cannot_become_complete_cache(tmp_path, monkeypatch):
    args, model, manifest, path = make_analysis_fixture(tmp_path)
    first = manifest["jobs"][0]
    manifest["jobs"].append({**first, "job_id": "second_stable"})
    path.write_text(json.dumps(manifest))
    original = inference.infer_split
    mutated = False

    def mutate_after_first(*args_, **kwargs):
        nonlocal mutated
        result = original(*args_, **kwargs)
        if not mutated:
            feature = Path(args.feature_root) / "visual_features.h5"
            stat = feature.stat()
            os.utime(feature, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            mutated = True
        return result

    monkeypatch.setattr(inference, "infer_split", mutate_after_first)
    result = run_worker(path)
    assert result["failed"] == 1
    assert "Input changed during" in result["jobs"][0]["error"]
    assert result["jobs"][1]["status"] == "complete"


def test_checkpoint_changed_while_loading_is_rejected(tmp_path, monkeypatch):
    args, model, manifest, path = make_analysis_fixture(tmp_path)
    original_load = torch.load

    def changed_load(file, *args_, **kwargs):
        payload = original_load(file, *args_, **kwargs)
        stat = Path(file).stat()
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return payload

    monkeypatch.setattr(torch, "load", changed_load)
    result = run_worker(path)
    assert result["failed"] == 1 and "Checkpoint changed while loading" in result["jobs"][0]["error"]
