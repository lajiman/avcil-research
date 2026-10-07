import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments_phase_9.class_fusion.cl_history import (
    CLHistoryRecorder, collect_memory_outputs, memory_classification, old_class_reliability, should_record_cl_history,
)
from experiments_phase_9.train_incremental_fusion_modular import build_parser, main
from test_training import make_fixture


class MemoryPairs(Dataset):
    mode = "train"

    def __init__(self):
        self.exemplar_vids_set = ["c0_a", "c0_b", "c1_a", "c1_b"]
        self.labels = torch.tensor([0, 0, 1, 1])
        self.audio = torch.zeros(4, 768)
        self.audio[:2, 0], self.audio[2:, 1] = 1, 1
        self.visual = self.audio.clone()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.visual[index], self.audio[index]), self.labels[index]


class ToyBranches(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes, self.fusion_chunk_size = num_classes, 2
        self.register_buffer("fusion_gate", torch.full((num_classes,), 0.5))
        self.register_buffer("fusion_version", torch.zeros((), dtype=torch.long))
        self.classifier = nn.Linear(768, num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.collapse_audio = False
        self.fail = False

    def extract_branch_features(self, visual, audio):
        if self.fail:
            raise RuntimeError("fixture extraction error")
        if self.collapse_audio:
            audio = torch.zeros_like(audio)
            audio[:, 0] = 1
        return audio, visual, None, None


def make_recorder(tmp_path, dataset=None):
    args = build_parser().parse_args(["--class_num_per_step", "2", "--fusion_batch_size", "2"])
    teacher, memory = ToyBranches(), dataset if dataset is not None else MemoryPairs()
    recorder = CLHistoryRecorder(args, 1, teacher, memory,
                                 {"checkpoint_name": "step_0_best_model.pt", "step": 0, "epoch": 4},
                                 tmp_path, {0: "zero", 1: "one"}, torch.device("cpu"))
    return recorder, teacher, memory


def test_asymmetric_branch_degradation_is_observed_without_changing_gate(tmp_path):
    recorder, teacher, _ = make_recorder(tmp_path)
    student = copy.deepcopy(teacher).train()
    before = recorder.observe(student, 0, ["task_start"])
    torch.testing.assert_close(before["observation"]["drop_a"], torch.zeros(2))
    student.collapse_audio = True
    gate = student.fusion_gate.clone()
    state = recorder.observe(student, 40, ["periodic"])
    observation = state["observation"]
    assert observation["comparable"].all()
    assert (observation["drop_a"] > 0.49).all()
    torch.testing.assert_close(observation["drop_v"], torch.zeros(2))
    torch.testing.assert_close(student.fusion_gate, gate)
    assert student.training and int(student.fusion_version) == 0
    persisted = torch.load(tmp_path / "step_1_epoch_40_history.pt", weights_only=True)
    torch.testing.assert_close(persisted["drop_asymmetry"], observation["drop_asymmetry"])
    assert persisted["reference_id"] == recorder.reference["reference_id"]
    assert recorder.reference["sample_ids"] == ["c0_a", "c0_b", "c1_a", "c1_b"]
    assert all("audio" != key and "visual" != key for key in recorder.reference)


def test_saved_scores_margins_and_paired_errors_explain_asymmetric_drop(tmp_path):
    recorder, teacher, memory = make_recorder(tmp_path)
    student = copy.deepcopy(teacher)
    student.collapse_audio = True
    observation = recorder.observe(student, 40, ["periodic"])["observation"]
    probe = observation["reliability"]
    for branch in ("a", "v"):
        scores = probe["sample_scores_" + branch]
        target = scores.gather(1, memory.labels[:, None]).squeeze(1)
        competitors = torch.stack([scores[i, 1-int(label)] for i, label in enumerate(memory.labels)])
        torch.testing.assert_close(scores.softmax(1).gather(1, memory.labels[:, None]).squeeze(1),
                                   probe["sample_probability_" + branch])
        torch.testing.assert_close(target - competitors, probe["sample_top1_margin_" + branch])
        torch.testing.assert_close(probe["sample_log_odds_" + branch].sigmoid(), probe["sample_probability_" + branch])
    # Audio collapse reduces both classes' confidence but only flips class 1's
    # predictions (the deterministic tie-break picks class 0).
    torch.testing.assert_close(probe["accuracy_a"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(probe["accuracy_v"], torch.ones(2))
    pairs = probe["paired_correctness"]
    torch.testing.assert_close(pairs["both_correct_rate"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(pairs["visual_only_correct_rate"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(sum(pairs[key] for key in ("both_correct_rate", "audio_only_correct_rate",
                                                        "visual_only_correct_rate", "both_wrong_rate")), torch.ones(2))
    transitions = observation["prediction_transitions"]
    torch.testing.assert_close(transitions["forgotten_rate_a"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(transitions["forgotten_rate_v"], torch.zeros(2))
    assert transitions["sample_forgotten_a"].tolist() == [0, 0, 1, 1]
    assert observation["schema_version"] == 2


def test_recovery_is_preserved_instead_of_clipped_away(tmp_path):
    args = build_parser().parse_args(["--class_num_per_step", "2"])
    teacher, memory = ToyBranches(), MemoryPairs()
    teacher.collapse_audio = True
    recorder = CLHistoryRecorder(args, 1, teacher, memory, {}, tmp_path, {}, torch.device("cpu"))
    student = copy.deepcopy(teacher)
    student.collapse_audio = False
    observation = recorder.observe(student, 40, ["periodic"])["observation"]
    assert (observation["signed_drop_a"] < 0).all()
    torch.testing.assert_close(observation["drop_a"], torch.zeros(2))
    torch.testing.assert_close(observation["prediction_transitions"]["recovered_rate_a"], torch.tensor([0.0, 1.0]))


def test_actual_fusion_logits_and_missing_predictions_round_trip(tmp_path):
    labels = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[3, 2, 10, 0], [4, 1, -2, 0], [1, 5, 0, 0], [torch.nan, 3, 1, 0]])
    result = memory_classification(logits, labels, 2)
    torch.save(result, tmp_path / "classification.pt")
    saved = torch.load(tmp_path / "classification.pt", weights_only=True)
    torch.testing.assert_close(saved["sample_logits"], logits, equal_nan=True)
    assert saved["sample_prediction_all_seen"].tolist() == [2, 0, 1, -1]
    assert saved["sample_prediction_old_only"].tolist() == [0, 0, 1, -1]
    assert saved["classification_valid"].tolist() == [True, False]
    assert saved["old_to_new_rate"][0] == 0.5
    assert torch.isnan(saved["all_seen_accuracy"][1])


def test_new_class_logits_do_not_contaminate_old_class_reliability(tmp_path):
    recorder, _, _ = make_recorder(tmp_path)
    student = ToyBranches(4)
    with torch.no_grad():
        student.classifier.bias[2] = 100
    observed = recorder.observe(student, 40, ["periodic"])["observation"]
    torch.testing.assert_close(observed["drop_a"], torch.zeros(2))
    torch.testing.assert_close(observed["drop_v"], torch.zeros(2))
    assert observed["reliability"]["candidate_mask"].tolist() == [True, True]
    torch.testing.assert_close(observed["classification"]["old_to_new_rate"], torch.ones(2))
    assert observed["classification"]["confusion_matrix"].shape == (2, 4)


def test_own_prototypes_do_not_treat_a_coordinate_rotation_as_forgetting():
    data = MemoryPairs()
    kwargs = dict(labels=data.labels, num_classes=2, candidates=torch.tensor([True, True]), temperature=0.1, min_samples=2)
    old = old_class_reliability(data.audio, data.visual, **kwargs)
    rotated = old_class_reliability(data.audio.roll(7, 1), data.visual.roll(19, 1), **kwargs)
    torch.testing.assert_close(old["reliability_a"], rotated["reliability_a"])
    torch.testing.assert_close(old["reliability_v"], rotated["reliability_v"])


def test_invalid_reference_changes_are_flagged_not_silently_dropped(tmp_path):
    recorder, student, memory = make_recorder(tmp_path)
    memory.audio[0] = torch.nan
    observed = recorder.observe(student, 40, ["periodic"])["observation"]
    assert not observed["comparable"].any()
    assert torch.isnan(observed["drop_a"]).all()
    assert observed["reliability"]["reason"] == "invalid_current_reference_features"
    # A same-class video replacement cannot slip through a labels-only check.
    memory.exemplar_vids_set[1] = "different_video_same_class"
    with pytest.raises(ValueError, match="cohort/order"):
        recorder.observe(student, 80, ["periodic"])


def test_teacher_exclusions_are_frozen_and_singletons_remain_unscorable(tmp_path):
    memory = MemoryPairs()
    memory.audio[0] = 0
    recorder, teacher, _ = make_recorder(tmp_path, memory)
    assert recorder.reference["included_mask"].tolist() == [False, True, True, True]
    assert recorder.reference["sample_row_indices"] == [1, 2, 3]
    memory.audio[0, 0] = 1
    observation = recorder.observe(teacher, 40, ["periodic"])["observation"]
    assert observation["reliability"]["counts"].tolist() == [1, 2]
    assert observation["comparable"].tolist() == [False, True]
    assert torch.isnan(observation["signed_drop_a"][0])
    assert observation["reliability"]["sample_scores_a"].shape == (3, 2)
    assert torch.isnan(observation["reliability"]["sample_scores_a"][0]).all()
    assert observation["reliability"]["sample_prediction_a"][0] == -1
    assert torch.isnan(observation["prediction_transitions"]["sample_forgotten_a"][0])


def test_collector_restores_rng_and_modes_on_success_and_failure():
    model, dataset = ToyBranches().train(), MemoryPairs()
    model.classifier.eval()  # Preserve mixed submodule modes too.
    cpu, python, numpy = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    collect_memory_outputs(model, dataset, 2, 0, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), cpu) and random.getstate() == python
    np.testing.assert_array_equal(np.random.get_state()[1], numpy[1])
    assert model.training and not model.classifier.training
    model.fail = True
    with pytest.raises(RuntimeError, match="fixture extraction"):
        collect_memory_outputs(model, dataset, 2, 0, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), cpu) and random.getstate() == python
    assert model.training and not model.classifier.training


def test_history_schedule_records_uniform_control_best_and_final():
    args = build_parser().parse_args(["--fusion_mode", "uniform"])
    assert should_record_cl_history(args, 1, 40)
    assert should_record_cl_history(args, 1, 200)
    assert should_record_cl_history(args, 1, 3, is_best=True)
    assert not should_record_cl_history(args, 1, 3)
    assert not should_record_cl_history(args, 0, 40)
    args.record_cl_history = False
    assert not should_record_cl_history(args, 1, 200, is_best=True)


@pytest.mark.parametrize("fusion_mode", ["periodic", "uniform"])
def test_recording_on_off_leaves_training_parameters_and_gates_identical(tmp_path, fusion_mode):
    features, meta = make_fixture(tmp_path)
    output = tmp_path / "outputs"
    base = ["--feature_root", str(features), "--meta_root", str(meta), "--output_root", str(output),
            "--num_classes", "4", "--class_num_per_step", "2", "--memory_size", "8",
            "--max_epoches", "3", "--fusion_warmup_epochs", "1", "--fusion_update_interval", "1",
            "--train_batch_size", "4", "--exemplar_batch_size", "4", "--infer_batch_size", "4", "--device", "cpu",
            "--fusion_mode", fusion_mode]
    main(base + ["--experiment_name", "observed"])
    main(base + ["--experiment_name", "unobserved", "--no_record_cl_history"])
    for step in (0, 1):
        for which in ("best", "last"):
            observed = torch.load(output / f"observed/step_{step}_{which}_model.pt", weights_only=True)
            unobserved = torch.load(output / f"unobserved/step_{step}_{which}_model.pt", weights_only=True)
            for key in observed["state_dict"]:
                torch.testing.assert_close(observed["state_dict"][key], unobserved["state_dict"][key], rtol=0, atol=0)
            assert observed["epoch"] == unobserved["epoch"] and observed["val_acc"] == unobserved["val_acc"]
            if fusion_mode == "uniform":
                gate = observed["state_dict"]["fusion_gate"]
                torch.testing.assert_close(gate, torch.full_like(gate, 0.5), rtol=0, atol=0)
                assert int(observed["state_dict"]["fusion_version"]) == 0
            assert unobserved["cl_history"] is None
            if step == 1:
                history = observed["cl_history"]
                assert history["observation"]["epoch"] == observed["epoch"]
                assert history["observation"]["gate_version"] == int(observed["state_dict"]["fusion_version"])
                assert len(history["reference"]["teacher_source"]["checkpoint_sha256"]) == 64
                assert history["observation"]["reference_id"] == history["reference"]["reference_id"]
    assert not (output / "metrics/unobserved/cl_history").exists()
    with (output / "metrics/observed/cl_history/step_1_reference.json").open() as handle:
        assert json.load(handle)["observation_only"] is True
