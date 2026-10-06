import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from experiments_phase_9.class_fusion.checkpoints import load_model, save_checkpoint
from experiments_phase_9.class_fusion.exact_losses import avcil_loss
from experiments_phase_9.class_fusion.fusion_method import (
    FusionReferenceDataset, ReliabilityBank, build_reliability_bank, should_update_gate, update_class_gates,
)
from experiments_phase_9.train_incremental_fusion_modular import build_parser, validate_args
from experiments_phase_8_rdcrosssdc_modular.rd_crosssdc import exact_losses as original_losses
from model.audio_visual_model_incremental import IncreAudioVisualNet
from model.audio_visual_model_incremental_class_fusion import ClassFusionAudioVisualNet, class_conditional_logits


def args_for_test():
    return build_parser().parse_args([])


def test_uniform_matches_original_features_logits_and_parameter_gradients():
    torch.manual_seed(9)
    args = args_for_test()
    original = IncreAudioVisualNet(args, 4)
    model = ClassFusionAudioVisualNet(args, 4)
    model.load_state_dict(original.state_dict(), strict=False)
    audio, visual = torch.randn(4, 768), torch.randn(4, 16, 768)
    old = original(audio=audio, visual=visual, out_feature_before_fusion=True, out_attn_score=True)
    new = model(audio=audio, visual=visual, out_feature_before_fusion=True, out_attn_score=True)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    old[0].square().mean().backward()
    new[0].square().mean().backward()
    for (name, p), (new_name, q) in zip(original.named_parameters(), model.named_parameters()):
        assert name == new_name
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert not model.fusion_gate.requires_grad


@pytest.mark.parametrize("nonlinear", [False, True])
def test_candidate_fusion_matches_explicit_class_features_and_gradients(nonlinear):
    torch.manual_seed(3)
    a, v = torch.randn(3, 7, requires_grad=True), torch.randn(3, 7, requires_grad=True)
    head = nn.Sequential(nn.Linear(7, 9), nn.Tanh(), nn.Linear(9, 5)) if nonlinear else nn.Linear(7, 5)
    gate = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    actual = class_conditional_logits(a, v, head, gate, chunk_size=2)
    expected = torch.stack([head(2*g*a + 2*(1-g)*v)[:, c] for c, g in enumerate(gate)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    tensors = [a, v, *head.parameters()]
    actual_grads = torch.autograd.grad(actual.square().sum(), tensors, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), tensors)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_expansion_checkpoint_and_teacher_isolation(tmp_path, head):
    args = args_for_test()
    args.fusion_classifier, args.fusion_hidden_dim = head, 11
    teacher = ClassFusionAudioVisualNet(args, 2)
    teacher.set_fusion_gate(torch.tensor([0.2, 0.8]))
    student = copy.deepcopy(teacher)
    student.incremental_classifier(4)
    torch.testing.assert_close(student.fusion_gate, torch.tensor([0.2, 0.8, 0.5, 0.5]))
    student.set_fusion_gate(torch.tensor([0.1, 0.9, 0.3, 0.7]))
    torch.testing.assert_close(teacher.fusion_gate, torch.tensor([0.2, 0.8]))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, student, args, step=1, epoch=81, val_acc=0.6)
    rng = torch.get_rng_state().clone()
    restored = load_model(path)
    assert torch.equal(torch.get_rng_state(), rng)
    audio, visual = torch.randn(2, 768), torch.randn(2, 16, 768)
    torch.testing.assert_close(restored(visual, audio), student(visual, audio), rtol=0, atol=0)
    assert int(restored.fusion_version) == int(student.fusion_version)


class FeaturePairs(Dataset):
    mode = "train"

    def __init__(self, audio, visual, labels, vids=None):
        self.audio, self.visual, self.labels = audio, visual, labels
        self.all_current_data_vids = vids or list(range(len(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.visual[index], self.audio[index]), self.labels[index]


class IdentityBranches(nn.Module):
    def extract_branch_features(self, visual, audio):
        return audio, visual, None, None


def make_reference():
    a = torch.zeros(7, 768)
    v = torch.zeros_like(a)
    a[0:3, 0], a[3:6, 1], a[6, 2] = 1, 1, 1
    v[0:6, 0], v[6, 2] = 1, 1
    return FeaturePairs(a, v, torch.tensor([0, 0, 0, 1, 1, 1, 2]))


def test_loo_probabilities_against_independent_per_query_reference():
    data = make_reference()
    net = IdentityBranches().train()
    rng = torch.get_rng_state().clone()
    py_state, np_state = random.getstate(), np.random.get_state()
    bank = build_reliability_bank(net, data, 4, 2, 0, torch.device("cpu"), temperature=0.5)
    assert net.training and torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == py_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert bank.counts.tolist() == [3, 3, 1, 0]
    assert bank.valid.tolist() == [True, True, False, False]
    for features, result in ((data.audio, bank.reliability_a), (data.visual, bank.reliability_v)):
        for c in (0, 1):
            probabilities = []
            for i in torch.where(data.labels == c)[0]:
                scores = []
                for k in range(3):
                    members = [j for j in range(len(data)) if data.labels[j] == k and j != i]
                    proto = F.normalize(features[members].mean(0), dim=0)
                    scores.append((F.normalize(features[i], dim=0) @ proto) / 0.5)
                probabilities.append(torch.softmax(torch.stack(scores), dim=0)[c])
            torch.testing.assert_close(result[c], torch.stack(probabilities).mean())
    assert (bank.reliability_a[:2] > bank.reliability_v[:2]).all()


def test_missing_degenerate_data_retains_class_history_and_restores_mode_on_error():
    data = make_reference()
    data.audio[0] = float("nan")
    data.visual[1] = 0
    net = IdentityBranches().train()
    bank = build_reliability_bank(net, data, 4, 2, 0, torch.device("cpu"))
    assert bank.counts.tolist() == [1, 3, 1, 0]
    result = update_class_gates(torch.tensor([0.8, 0.3, 0.6, 0.9]), bank)
    torch.testing.assert_close(result["gate"][[0, 2, 3]], torch.tensor([0.8, 0.6, 0.9]))
    assert torch.isfinite(result["gate"]).all()
    data.labels[0] = 8
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="seen-class"):
        build_reliability_bank(net, data, 4, 2, 0, torch.device("cpu"))
    assert net.training and torch.equal(torch.get_rng_state(), state)


def test_unique_reference_and_train_only_boundary():
    data = make_reference()
    data.all_current_data_vids = ["a", "a", "b", "c", "d", "e", "f"]
    reference = FusionReferenceDataset(data, data)
    assert len(reference) == 6
    data.mode = "test"
    with pytest.raises(ValueError, match="training"):
        FusionReferenceDataset(data)


def test_memory_damps_updates_toward_history_not_toward_half():
    counts = torch.tensor([50, 10, 5, 1])
    bank = ReliabilityBank(counts, counts, torch.full((4,), 0.2), torch.full((4,), 0.8), torch.tensor([True, True, True, False]))
    old = torch.tensor([0.8, 0.8, 0.8, 0.9])
    scaled = update_class_gates(old, bank, "sample_aware", eta_max=0.5, n_ref=10)
    fixed = update_class_gates(old, bank, "fixed", eta_max=0.5)
    assert scaled["eta"][0] > scaled["eta"][1] > scaled["eta"][2] > 0
    torch.testing.assert_close(scaled["gate"][:3], old[:3] + (0.5 * counts[:3] / (counts[:3] + 10)) * (0.2 - old[:3]))
    torch.testing.assert_close(fixed["gate"][:3], torch.full((3,), 0.5))
    assert scaled["gate"][3] == old[3]


def test_update_schedule_has_warmup_and_never_updates_after_last_epoch():
    args = args_for_test()
    assert [e for e in range(201) if should_update_gate(args, 1, e)] == [40, 80, 120, 160]
    assert not any(should_update_gate(args, 0, e) for e in range(201))
    args.fusion_update_first_step = True
    assert should_update_gate(args, 0, 40)
    args.fusion_mode = "uniform"
    assert not any(should_update_gate(args, 1, e) for e in range(201))


def test_original_avcil_objective_including_ce_slices_kd_and_attention():
    torch.manual_seed(7)
    args = args_for_test()
    args.class_num_per_step = 2
    model, old = ClassFusionAudioVisualNet(args, 6), ClassFusionAudioVisualNet(args, 4)
    a, v = torch.randn(6, 768), torch.randn(6, 16, 768)
    output = model(v, a, out_feature_before_fusion=True, out_attn_score=True)
    with torch.no_grad():
        teacher = old(v, a, out_feature_before_fusion=True, out_attn_score=True)
    labels = torch.tensor([4, 5, 4, 0, 1, 3])
    loss, parts = avcil_loss(args, 2, output, labels, 3, teacher)
    logits, audio, visual, spatial, temporal = output
    expected_ce = (original_losses.ce_loss(2, logits[:3, 4:], labels[:3] % 2)
                   + original_losses.ce_loss(4, logits[3:, :4], labels[3:])) / 2
    expected_kd = sum(F.kl_div(F.log_softmax(logits[:, i:i+2] / 2, dim=1),
                             F.softmax(teacher[0][:, i:i+2] / 2, dim=1), reduction="batchmean") * 4 for i in (0, 2))
    expected = expected_ce + expected_kd
    expected = expected + args.lam_I * original_losses.cal_contrastive_loss(audio, visual, args.instance_contrastive_temperature)
    expected = expected + args.lam_C * original_losses.class_contrastive_loss(audio, visual, labels, args.class_contrastive_temperature)
    s, s_old = spatial[3:].transpose(2, 3), teacher[3][3:].transpose(2, 3)
    t, t_old = temporal[3:].transpose(1, 2), teacher[4][3:].transpose(1, 2)
    expected = expected + args.lam * F.kl_div(s.reshape(-1, s.shape[-1]).log(), s_old.reshape(-1, s_old.shape[-1]), reduction="sum") / 3
    expected = expected + (1-args.lam) * F.kl_div(t.reshape(-1, t.shape[-1]).log(), t_old.reshape(-1, t_old.shape[-1]), reduction="sum") / 3
    torch.testing.assert_close(parts["ce"], expected_ce)
    torch.testing.assert_close(parts["kd"], expected_kd)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(model.audio_proj.weight.grad).all()


@pytest.mark.parametrize("option,value", [("fusion_eta_max", "nan"), ("fusion_temperature", "0"), ("fusion_update_interval", "0"), ("fusion_min_samples", "1")])
def test_invalid_arguments_fail_before_training(option, value):
    parser = build_parser()
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--" + option, value]))
