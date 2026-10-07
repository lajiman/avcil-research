"""Independent numerical and lifecycle checks for the optional historical bank."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from experiments_phase_9.class_fusion.fusion_method import (
    ReliabilityBank, build_reliability_bank, update_class_gates,
)
from experiments_phase_9.class_fusion.prototype_bank import (
    build_bank_reliability, prepare_prototype_bank,
)


CPU = torch.device("cpu")


class Branches(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.child = nn.Dropout()
        self.fail = fail

    def extract_branch_features(self, visual, audio):
        if self.fail:
            random.random()
            np.random.random()
            torch.rand(2)
            raise RuntimeError("deliberate extraction failure")
        return audio, visual, None, None


class Pairs(Dataset):
    mode = "train"

    def __init__(self, features, ids):
        self.features, self.sample_ids = features, list(ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        audio, visual, label = self.features[self.sample_ids[index]]
        return (visual, audio), label


class Replay:
    """The full-class accessor records (and can forbid) old-class rereads."""

    mode = "train"

    def __init__(self, features, groups):
        self.features = features
        self.exemplar_class_vids_set = copy.deepcopy(groups)
        self.exemplar_vids_set = [vid for group in groups for vid in group]
        self.category_encode_dict = {str(c): c for _, _, c in features.values()}
        self.all_id_category_dict = {vid: str(row[2]) for vid, row in features.items()}
        self.class_reads, self.feature_reads = [], []
        self.forbid_classes = set()

    def _valid_class_vids(self, class_id):
        self.class_reads.append(class_id)
        if class_id in self.forbid_classes:
            raise AssertionError("A discarded old-class training set was reopened")
        return [vid for vid, row in self.features.items() if row[2] == class_id]

    def _read(self, vid):
        self.feature_reads.append(vid)
        audio, visual, label = self.features[vid]
        return (visual, audio), label


def make_features(seed, classes=3, per_class=6):
    rng = torch.Generator().manual_seed(seed)
    features = {}
    for c in range(classes):
        for i in range(per_class):
            a, v = torch.zeros(768), torch.zeros(768)
            a[:9] = torch.randn(9, generator=rng)
            v[:9] = torch.randn(9, generator=rng)
            a[c] += 1.0
            v[(c + 1) % 3] += 0.7
            features[f"train_{c}_{i}"] = (a, v, c)
    return features


def prepare_two_old_classes(birth, groups):
    first = prepare_prototype_bank(
        None, Branches(), Replay(birth, groups[:1]), 1, 1, 2, 0, CPU,
        {"sha256": "teacher-0", "step": 0, "epoch": 17, "num_classes": 1},
    )
    replay = Replay(birth, groups)
    bank = prepare_prototype_bank(first, Branches(), replay, 2, 1, 2, 0, CPU,
                                  {"sha256": "teacher-1", "step": 1, "epoch": 11, "num_classes": 2})
    return replay, bank


def prepared_fixture():
    birth, current = make_features(13), make_features(29)
    groups = [[f"train_{c}_{i}" for i in range(3)] for c in range(2)]
    replay, bank = prepare_two_old_classes(birth, groups)
    ids = replay.exemplar_vids_set + [f"train_2_{i}" for i in range(4)]
    return birth, Pairs(current, ids), replay, bank


def assert_nested_equal(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


def oracle_probabilities(birth, reference, classes, old_classes, strength, temperature):
    """Enumerate sample lists, explicitly excluding a query from all three sets.

    This does not use implementation sums, correction helpers, or saved-bank
    fields. It therefore checks the intended estimator rather than its code.
    """
    results, full_prototypes = [], []
    for modality in (0, 1):
        def unit(features, ids):
            return torch.stack([F.normalize(features[vid][modality].double(), dim=0) for vid in ids])

        def prototype(c, query=None):
            complete_current = [vid for vid in reference.sample_ids if reference.features[vid][2] == c]
            present = [vid for vid in complete_current if vid != query]
            mean_now = unit(reference.features, present).mean(0)
            if c >= old_classes:
                return F.normalize(mean_now, dim=0)
            complete_birth = [vid for vid, row in birth.items() if row[2] == c]
            birth_members = [vid for vid in complete_birth if vid != query]
            historical_mean = unit(birth, birth_members).mean(0)
            matching_old_mean = unit(birth, present).mean(0)
            drift = mean_now - matching_old_mean
            # A repeated scan is not new data; the discarded population sets
            # the capped prior strength, and is unchanged by leaving one out.
            k = min(strength, len(complete_birth) - len(complete_current))
            beta = len(present) / (len(present) + k)
            return F.normalize(beta * mean_now + (1 - beta) * (historical_mean + drift), dim=0)

        full = [prototype(c) for c in range(classes)]
        probabilities = []
        for vid in reference.sample_ids:
            label = reference.features[vid][2]
            query = F.normalize(reference.features[vid][modality].double(), dim=0)
            candidates = [prototype(c, vid) if c == label else full[c] for c in range(classes)]
            scores = torch.stack([query @ p / temperature for p in candidates])
            probabilities.append(scores.softmax(0)[label])
        results.append(torch.stack(probabilities))
        full_prototypes.append(torch.stack(full))
    return results, full_prototypes


def assert_matches_oracle(result, birth, reference, strength=10, temperature=0.4):
    expected, prototypes = oracle_probabilities(birth, reference, 3, 2, strength, temperature)
    labels = torch.tensor([reference.features[vid][2] for vid in reference.sample_ids])
    for actual, probabilities in zip((result.reliability_a, result.reliability_v), expected):
        means = torch.stack([probabilities[labels == c].mean() for c in range(3)])
        torch.testing.assert_close(actual.double(), means, rtol=2e-5, atol=2e-6)
    return expected, prototypes


@pytest.mark.parametrize("batch_size", [1, 4])
def test_drift_bank_reliability_matches_independent_per_query_loo(batch_size):
    birth, reference, _, bank = prepared_fixture()
    result, diagnostics = build_bank_reliability(
        Branches(), reference, bank, 3, batch_size, 0, CPU, temperature=0.4,
    )
    _, prototypes = assert_matches_oracle(result, birth, reference)
    assert result.counts.tolist() == result.scored_counts.tolist() == [3, 3, 4]
    assert result.valid.tolist() == [True, True, True]
    assert diagnostics["history_used"].tolist() == [True, True, False]
    torch.testing.assert_close(diagnostics["prototype_beta"][:2], torch.full((2,), 0.5))
    for name, expected in zip(("prototype_audio", "prototype_visual"), prototypes):
        torch.testing.assert_close(diagnostics[name].double(), expected, rtol=2e-5, atol=2e-6)


def test_query_is_excluded_from_birth_prior_and_drift_not_just_current_mean():
    birth, reference, replay, bank = prepared_fixture()
    first, _ = build_bank_reliability(Branches(), reference, bank, 3, 3, 0, CPU, temperature=0.4)
    first_query_probs, _ = assert_matches_oracle(first, birth, reference)
    # Change ONLY the historical feature of query 0. A correct query-0 LOO
    # target excludes it both from the full birth class and matched anchors.
    # Other queries legitimately still use this point in their historical sets.
    changed = copy.deepcopy(birth)
    changed[reference.sample_ids[0]] = (-birth[reference.sample_ids[0]][0],
                                       -birth[reference.sample_ids[0]][1], 0)
    _, changed_bank = prepare_two_old_classes(changed, replay.exemplar_class_vids_set)
    second, _ = build_bank_reliability(Branches(), reference, changed_bank, 3, 3, 0, CPU, temperature=0.4)
    second_query_probs, _ = assert_matches_oracle(second, changed, reference)
    for original, modified in zip(first_query_probs, second_query_probs):
        torch.testing.assert_close(original[0], modified[0], rtol=0, atol=1e-14)
        assert not torch.allclose(original[1:3], modified[1:3], atol=1e-7, rtol=0)


def test_repeated_scans_do_not_accumulate_evidence_or_mutate_saved_bank():
    _, reference, _, bank = prepared_fixture()
    before = copy.deepcopy(bank)
    net = Branches().train()
    net.child.eval()
    rng = torch.get_rng_state().clone()
    first, first_diagnostics = build_bank_reliability(net, reference, bank, 3, 2, 0, CPU)
    second, second_diagnostics = build_bank_reliability(net, reference, bank, 3, 2, 0, CPU)
    assert_nested_equal(bank, before)
    assert_nested_equal(vars(first), vars(second))
    assert_nested_equal(first_diagnostics, second_diagnostics)
    assert first.counts.tolist() == [3, 3, 4]
    assert net.training and not net.child.training
    assert torch.equal(torch.get_rng_state(), rng)


def test_prepare_prunes_anchors_preserves_birth_statistics_and_never_reopens_older_data():
    birth = make_features(41)
    first_replay = Replay(birth, [[f"train_0_{i}" for i in range(4)]])
    first_metadata = {"sha256": "snapshot-0", "step": 0, "epoch": 12, "num_classes": 1}
    first = prepare_prototype_bank(None, Branches(), first_replay, 1, 1, 3, 0, CPU, first_metadata)
    before = copy.deepcopy(first)
    assert first_replay.class_reads == [0]
    assert set(first["anchors"]["sample_ids"]) == set(first_replay.exemplar_vids_set)
    second_replay = Replay(birth, [["train_0_0", "train_0_1"], ["train_1_0", "train_1_1"]])
    second_replay.forbid_classes = {0, 2}
    second_metadata = {"sha256": "snapshot-1", "step": 1, "epoch": 9, "num_classes": 2}
    second = prepare_prototype_bank(first, Branches(), second_replay, 2, 1, 3, 0, CPU, second_metadata)
    assert_nested_equal(first, before)
    assert second_replay.class_reads == [1]
    assert all(vid.startswith("train_1_") for vid in second_replay.feature_reads)
    assert set(second["anchors"]["sample_ids"]) == set(second_replay.exemplar_vids_set)
    assert second["birth_count"].tolist() == [6, 6]
    assert second["prepared_step"] == 2 and second["num_classes"] == 2
    for key in ("birth_sum_audio", "birth_sum_visual", "birth_count", "birth_step"):
        torch.testing.assert_close(second[key][:1], first[key], rtol=0, atol=0)
        assert second[key].device.type == "cpu"
    assert "snapshot-0" in str(second["sources"]) and "snapshot-1" in str(second["sources"])
    for vid in ("train_0_0", "train_0_1"):
        i, j = first["anchors"]["sample_ids"].index(vid), second["anchors"]["sample_ids"].index(vid)
        for key in ("audio", "visual"):
            torch.testing.assert_close(second["anchors"][key][j], first["anchors"][key][i], rtol=0, atol=0)


@pytest.mark.parametrize("fallback", ["zero_strength", "invalid_anchors", "missing_anchors"])
def test_no_usable_history_reproduces_fresh_prototypes(fallback):
    _, reference, _, bank = prepared_fixture()
    strength = 10
    if fallback == "zero_strength":
        strength = 0
    elif fallback == "invalid_anchors":
        bank["anchors"]["valid"].fill_(False)
    else:
        bank["anchors"]["sample_ids"] = []
        for key in ("labels", "valid", "audio", "visual"):
            bank["anchors"][key] = bank["anchors"][key][:0]
    fresh = build_reliability_bank(Branches(), reference, 3, 2, 0, CPU, temperature=0.4)
    actual, diagnostics = build_bank_reliability(
        Branches(), reference, bank, 3, 2, 0, CPU, temperature=0.4, prior_strength=strength,
    )
    for key in vars(fresh):
        torch.testing.assert_close(getattr(actual, key), getattr(fresh, key), rtol=0, atol=0)
    assert not diagnostics["history_used"].any()


def test_history_cannot_make_a_current_singleton_valid_for_gate_update():
    _, reference, _, bank = prepared_fixture()
    audio, visual, label = reference.features["train_0_0"]
    reference.features["train_0_0"] = (torch.full_like(audio, float("nan")), visual, label)
    audio, visual, label = reference.features["train_0_1"]
    reference.features["train_0_1"] = (audio, torch.zeros_like(visual), label)
    result, _ = build_bank_reliability(Branches(), reference, bank, 3, 2, 0, CPU)
    assert result.counts.tolist() == [1, 3, 4]
    assert result.valid.tolist() == [False, True, True]
    assert int(result.scored_counts[0]) == 0
    previous = torch.tensor([0.8, 0.5, 0.5])
    update = update_class_gates(previous, result, "direct")
    assert update["gate"][0] == previous[0]
    assert torch.isfinite(update["gate"]).all()


@pytest.mark.parametrize("cancellation", ["full_class", "loo_target"])
def test_history_does_not_rescue_a_degenerate_current_prototype(cancellation):
    _, reference, _, bank = prepared_fixture()
    e1, e2 = torch.zeros(768), torch.zeros(768)
    e1[0], e2[1] = 1, 1
    vectors = [e1, -e1] if cancellation == "full_class" else [e1, e2, -e2]
    if cancellation == "full_class":
        reference.sample_ids.remove("train_0_2")
    for i, vector in enumerate(vectors):
        reference.features[f"train_0_{i}"] = (vector.clone(), vector.clone(), 0)
    fresh = build_reliability_bank(Branches(), reference, 3, 2, 0, CPU)
    actual, _ = build_bank_reliability(Branches(), reference, bank, 3, 2, 0, CPU)
    assert not fresh.valid[0]
    assert not actual.valid[0]
    assert actual.scored_counts[0] == fresh.scored_counts[0]


@pytest.mark.parametrize("operation", ["prepare", "estimate"])
def test_failed_extraction_restores_every_mode_and_all_rng_states(operation):
    _, reference, replay, bank = prepared_fixture()
    net = Branches(fail=True).train()
    net.child.eval()
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="deliberate extraction failure"):
        if operation == "prepare":
            prepare_prototype_bank(None, net, replay, 1, 2, 2, 0, CPU,
                                   {"sha256": "fails", "step": 0, "num_classes": 2})
        else:
            build_bank_reliability(net, reference, bank, 3, 2, 0, CPU)
    assert net.training and not net.child.training
    assert random.getstate() == python_state
    current_np_state = np.random.get_state()
    assert current_np_state[0] == numpy_state[0]
    np.testing.assert_array_equal(current_np_state[1], numpy_state[1])
    assert current_np_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_direct_gate_updates_only_valid_classes_without_count_damping():
    counts = torch.tensor([2, 50, 1, 0])
    bank = ReliabilityBank(counts, counts, torch.tensor([0.2, 0.7, 0.9, 0.0]),
                           torch.tensor([0.8, 0.3, 0.1, 0.0]), torch.tensor([True, True, False, False]))
    previous = torch.tensor([0.8, 0.1, 0.6, 0.9])
    update = update_class_gates(previous, bank, "direct")
    torch.testing.assert_close(update["gate"], torch.tensor([0.2, 0.7, 0.6, 0.9]), rtol=0, atol=1e-7)
    torch.testing.assert_close(update["eta"], torch.tensor([1.0, 1.0, 0.0, 0.0]))
