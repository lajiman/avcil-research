"""Mathematical and data-boundary checks for persistent B/C prototype banks.

These use tiny, ID-addressable feature tables: expected prototypes can be
calculated independently of the bank implementation and without a GPU or HDF5.
"""

from copy import deepcopy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rd_crosssdc.prototype_history import (
    IndexedReplayDataset,
    load_task_bank,
    save_task_bank,
)


def make_args(policy="pre_shrink", **overrides):
    values = dict(
        dataset="synthetic_train",
        modality="audio-visual",
        num_classes=8,
        class_num_per_step=2,
        num_workers=0,
        exemplar_batch_size=3,
        infer_batch_size=3,
        train_batch_size=3,
        seed=7,
        rd_mode="adaptive_crosssdc_cmr",
        rd_prototype_policy=policy,
        rd_margin_temperature=0.4,
        rd_trust_shrinkage_beta=0.0,
        rd_history_folds=2,
        rd_history_mass_cap=50.0,
        rd_history_decay=0.9,
        rd_history_error_scale=0.25,
        rd_history_min_anchors=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FeatureTableModel(nn.Module):
    """Each dataset item addresses a deterministic normalized feature pair."""

    def __init__(self, audio, visual):
        super().__init__()
        self.register_buffer("audio_table", audio.clone())
        self.register_buffer("visual_table", visual.clone())
        self.child = nn.Dropout(0.2)

    def forward(self, visual, audio, out_feature_before_fusion=False):
        indices = audio[:, 0].long()
        a = self.audio_table[indices]
        v = self.visual_table[indices]
        logits = a.new_zeros((len(indices), 8))
        if out_feature_before_fusion:
            return logits, a, v
        return logits


class TinyDataset(Dataset):
    def __init__(self, records, *, replay=False, mode="train"):
        self.records = list(records)
        ids = [record[0] for record in records]
        if replay:
            self.exemplar_vids_set = ids
        else:
            self.mode = mode
            self.all_current_data_vids = ids
        self.category_encode_dict = {"class{}".format(label): label for label in range(8)}
        self.all_id_category_dict = {vid: "class{}".format(label)
                                     for vid, _, label in records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        _, feature_index, label = self.records[index]
        feature = torch.tensor([feature_index], dtype=torch.float32)
        return (feature, feature), label


class PersistentPrototypeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.records = []
        audio, visual = [], []
        for label in range(8):
            for sample in range(8):
                self.records.append(("class{}_sample{}".format(label, sample),
                                     len(self.records), label))
                # Distinct branches and within-class variance make accidental
                # self-inclusion and incorrect averaging visible in the margins.
                angle = 0.55 * label + 0.04 * sample
                audio.append([np.cos(angle), np.sin(angle), 0.2 + 0.02 * sample])
                visual.append([np.cos(angle + 0.11), np.sin(angle + 0.11),
                               0.1 + 0.03 * sample])
        self.audio = F.normalize(torch.tensor(audio, dtype=torch.float32), dim=1)
        self.visual = F.normalize(torch.tensor(visual, dtype=torch.float32), dim=1)
        self.model = FeatureTableModel(self.audio, self.visual)
        self.device = torch.device("cpu")

    def tearDown(self):
        self.tmp.cleanup()

    def data(self, counts, *, replay=False, mode="train"):
        selected = [record for label, count in counts.items()
                    for record in self.records if record[2] == label
                    and int(record[0].rsplit("sample", 1)[1]) < count]
        return TinyDataset(selected, replay=replay, mode=mode)

    def save(self, args, step, train, replay=None, previous_bank=None,
             model=None, tag="default"):
        model = self.model if model is None else model
        checkpoint = self.root / "{}_step{}_model.pt".format(tag, step)
        path = self.root / "{}_step{}_bank.pt".format(tag, step)
        torch.save(model.state_dict(), checkpoint)
        artifact = save_task_bank(
            args, step, model, str(checkpoint), train, replay,
            self.device, previous_bank=previous_bank, path=str(path),
        )
        return checkpoint, path, artifact

    def load(self, args, step, replay, checkpoint, path, model=None):
        return load_task_bank(
            args, step, self.model if model is None else model,
            str(checkpoint), replay, self.device, str(path),
        )

    def first_bank(self, args=None, *, supports=None, queries=None, tag="default"):
        args = make_args() if args is None else args
        supports = {0: 6, 1: 4} if supports is None else supports
        queries = {0: 2, 1: 1} if queries is None else queries
        replay = self.data(queries, replay=True)
        checkpoint, path, artifact = self.save(args, 0, self.data(supports), tag=tag)
        bank = self.load(args, 1, replay, checkpoint, path)
        return bank, replay, checkpoint, path, artifact

    def shifted_model(self, angle=0.2):
        rotation = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0],
                                 [0, 0, 1]], dtype=torch.float32)
        return FeatureTableModel(self.audio @ rotation.T, self.visual @ rotation)

    def second_bank(self, args, *, tag="default", model=None):
        bank, replay, _, _, _ = self.first_bank(
            args, supports={0: 8, 1: 8}, queries={0: 6, 1: 6}, tag=tag,
        )
        model = self.shifted_model() if model is None else model
        checkpoint, path, artifact = self.save(
            args, 1, self.data({2: 8, 3: 8}), replay, bank, model, tag=tag,
        )
        queries = self.data({0: 3, 1: 3, 2: 6, 3: 6}, replay=True)
        loaded = self.load(args, 2, queries, checkpoint, path, model=model)
        return loaded, queries, model, artifact

    def manual_margin(self, query, labels, paired_teacher, sums, temperature):
        rows = []
        for q, label, teacher in zip(query, labels, paired_teacher):
            label = int(label)
            target = F.normalize(sums[label] - teacher.detach(), dim=0)
            negatives = [q @ F.normalize(sums[c], dim=0) / temperature
                         for c in range(len(sums)) if c != label]
            rows.append(q @ target / temperature -
                        torch.logsumexp(torch.stack(negatives), dim=0))
        return torch.stack(rows)

    def test_b_support_counts_and_trust_query_counts_are_distinct(self):
        args = make_args(rd_trust_shrinkage_beta=3.0)
        bank, replay, _, _, _ = self.first_bank(args)
        torch.testing.assert_close(bank.counts.cpu(), torch.tensor([6.0, 4.0]))
        torch.testing.assert_close(bank.query_counts.cpu(), torch.tensor([2.0, 1.0]))
        expected_audio = torch.stack([self.audio[:6].sum(0), self.audio[8:12].sum(0)])
        expected_visual = torch.stack([self.visual[:6].sum(0), self.visual[8:12].sum(0)])
        torch.testing.assert_close(bank.audio_sums.cpu(), expected_audio)
        torch.testing.assert_close(bank.visual_sums.cpu(), expected_visual)
        indices = torch.tensor([record[1] for record in replay.records])
        labels = torch.tensor([record[2] for record in replay.records])
        margin = self.manual_margin(self.audio[indices], labels, self.visual[indices],
                                    expected_visual, args.rd_margin_temperature)
        reliability = torch.stack([torch.sigmoid(margin[labels == c]).mean()
                                   for c in range(2)])
        trust = ((reliability - 0.5) / 0.5).clamp(0, 1)
        query_counts = torch.tensor([2.0, 1.0])
        expected_trust = (query_counts * trust + 3.0 * trust.mean()) / (query_counts + 3.0)
        torch.testing.assert_close(bank.reliability_a_from_v.cpu(), reliability)
        torch.testing.assert_close(bank.trust_a_from_v.cpu(), expected_trust)

    def test_b_margin_is_exact_loo_and_only_query_receives_gradients(self):
        args = make_args()
        bank, replay, _, _, _ = self.first_bank(args)
        order = torch.tensor([2, 0, 2, 1])  # shuffled and repeated replay entries
        feature_indices = torch.tensor([r[1] for r in replay.records])[order]
        labels = torch.tensor([r[2] for r in replay.records])[order]
        query = self.visual[feature_indices].clone().requires_grad_()
        paired = self.audio[feature_indices].clone().requires_grad_()
        expected_query = query.detach().clone().requires_grad_()
        actual = bank.cross_modal_margin(query, labels, paired,
                                         args.rd_margin_temperature, "audio", order)
        expected = self.manual_margin(expected_query, labels, paired,
                                      bank.audio_sums, args.rd_margin_temperature)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(query.grad, expected_query.grad)
        self.assertGreater(float(query.grad.abs().sum()), 0.0)
        self.assertIsNone(paired.grad)
        self.assertFalse(bank.audio_sums.requires_grad)

    def test_c_without_retired_history_is_exactly_b(self):
        b, _, _, _, _ = self.first_bank(make_args(), tag="b")
        c, _, _, _, _ = self.first_bank(make_args("historical"), tag="c")
        self.assertTrue(torch.equal(b.audio_prototypes, c.audio_prototypes))
        self.assertTrue(torch.equal(b.visual_prototypes, c.visual_prototypes))
        self.assertTrue(torch.equal(b.trust_a_from_v, c.trust_a_from_v))
        for modality, query, positive in [
            ("audio", b.query_visual, b.query_audio),
            ("visual", b.query_audio, b.query_visual),
        ]:
            ids = torch.arange(len(b.query_labels))
            bm = b.cross_modal_margin(query, b.query_labels, positive, 0.4, modality, ids)
            cm = c.cross_modal_margin(query, c.query_labels, positive, 0.4, modality, ids)
            self.assertTrue(torch.equal(bm, cm))

    def test_c_zero_history_cap_matches_b_after_actual_retirement(self):
        b, _, _, _ = self.second_bank(make_args(), tag="b")
        c, _, _, artifact = self.second_bank(
            make_args("historical", rd_history_mass_cap=0.0), tag="c",
        )
        self.assertEqual(artifact["retired_unique_counts"].tolist(), [2, 2, 0, 0])
        self.assertEqual(float(artifact["history_mass_a"].sum()), 0.0)
        self.assertEqual(float(artifact["history_mass_v"].sum()), 0.0)
        self.assertTrue(torch.equal(b.trust_a_from_v, c.trust_a_from_v))
        self.assertTrue(torch.equal(b.trust_v_from_a, c.trust_v_from_a))
        ids = torch.arange(len(b.query_ids))
        bm = b.cross_modal_margin(b.query_audio, b.query_labels, b.query_visual,
                                   0.4, "visual", ids)
        cm = c.cross_modal_margin(c.query_audio, c.query_labels, c.query_visual,
                                   0.4, "visual", ids)
        self.assertTrue(torch.equal(bm, cm))

    def test_c_repeated_replay_is_not_counted_as_new_historical_evidence(self):
        args = make_args("historical")
        bank, replay, model, previous = self.second_bank(args)
        _, _, artifact = self.save(args, 2, self.data({4: 8, 5: 8}), replay,
                                    bank, model)
        self.assertEqual(previous["retired_unique_counts"].tolist(), [2, 2, 0, 0])
        self.assertEqual(artifact["retired_unique_counts"].tolist(), [5, 5, 2, 2, 0, 0])
        self.assertEqual(len(artifact["retired_ids"]), 14)
        self.assertEqual(len(set(artifact["retired_ids"])), 14)
        self.assertFalse(set(artifact["retired_ids"]) & set(artifact["support_ids"]))
        # Class 1 retains two fold-0 anchors; there is no further model drift.
        # Confidence therefore equals decay, and only THREE newly retired
        # samples are added to the prior effective mass, not all five retirees.
        expected_mass = (previous["history_mass_a"][1, 1] + 3) * args.rd_history_decay
        torch.testing.assert_close(artifact["history_mass_a"][1, 1], expected_mass)
        self.assertLess(float(artifact["history_mass_a"][1, 1]), 5.0)

    def test_c_fold_history_excludes_query_anchor_from_shift_and_confidence(self):
        args = make_args("historical")
        bank, replay, _, _, _ = self.first_bank(
            args, supports={0: 8, 1: 8}, queries={0: 6, 1: 6},
        )
        model = self.shifted_model()
        _, _, regular = self.save(args, 1, self.data({2: 8, 3: 8}), replay,
                                    bank, model, tag="regular")
        altered = deepcopy(model)
        # class1_sample0 is in fold 0 for this fixed seed. Its feature may
        # affect the exact current support, but never fold-0 history migration.
        index = bank.query_ids.index("class1_sample0")
        self.assertEqual(int(bank.query_folds[index]), 0)
        altered.audio_table[8] = F.normalize(torch.tensor([0.1, 0.3, 1.0]), dim=0)
        altered.visual_table[8] = F.normalize(torch.tensor([0.4, 0.1, 1.0]), dim=0)
        _, _, perturbed = self.save(args, 1, self.data({2: 8, 3: 8}), replay,
                                      bank, altered, tag="perturbed")
        for suffix in ["a", "v"]:
            for prefix in ["history_mean_", "history_mass_", "shift_", "residual_", "confidence_"]:
                self.assertTrue(torch.equal(regular[prefix + suffix][0],
                                            perturbed[prefix + suffix][0]), prefix + suffix)
            self.assertFalse(torch.equal(regular["shift_" + suffix][1, 1],
                                         perturbed["shift_" + suffix][1, 1]))
        self.assertFalse(torch.equal(regular["sums_a"], perturbed["sums_a"]))

    def test_c_transport_matches_retired_sum_plus_fold_held_out_shift(self):
        args = make_args("historical")
        bank, replay, _, _, _ = self.first_bank(
            args, supports={0: 8, 1: 8}, queries={0: 6, 1: 6},
        )
        model = self.shifted_model()
        _, _, artifact = self.save(args, 1, self.data({2: 8, 3: 8}), replay,
                                    bank, model)
        for cls in [0, 1]:
            for fold in [0, 1]:
                selected = [record[1] for record, f in zip(replay.records, bank.query_folds)
                            if record[2] == cls and int(f) != fold]
                self.assertEqual(int(artifact["anchor_counts"][fold, cls]), len(selected))
                if len(selected) < args.rd_history_min_anchors:
                    self.assertEqual(float(artifact["history_mass_a"][fold, cls]), 0.0)
                    continue
                deltas = model.audio_table[selected] - self.audio[selected]
                # Independent explicit leave-one-anchor-out prediction errors.
                errors = []
                for index in range(len(selected)):
                    other = torch.cat([deltas[:index], deltas[index + 1:]])
                    errors.append((deltas[index] - other.mean(0)).square().sum())
                expected_error = torch.stack(errors).mean()
                expected_shift = deltas.mean(0)
                expected_mean = self.audio[cls * 8 + 6:cls * 8 + 8].mean(0) + expected_shift
                expected_mass = 2 * args.rd_history_decay * torch.exp(
                    -expected_error / args.rd_history_error_scale ** 2)
                torch.testing.assert_close(artifact["residual_a"][fold, cls], expected_error)
                torch.testing.assert_close(artifact["history_mean_a"][fold, cls], expected_mean)
                torch.testing.assert_close(artifact["history_mass_a"][fold, cls], expected_mass)

    def test_c_fold_exclusion_survives_two_historical_transitions(self):
        args = make_args("historical")
        first, first_replay, _, _, _ = self.first_bank(
            args, supports={0: 8, 1: 8}, queries={0: 6, 1: 6},
        )
        regular = self.shifted_model()
        altered = deepcopy(regular)
        altered.audio_table[8] = F.normalize(torch.tensor([0.1, 0.3, 1.0]), dim=0)
        altered.visual_table[8] = F.normalize(torch.tensor([0.4, 0.1, 1.0]), dim=0)
        # Keep the perturbed fold-0 query alive across both transitions. Its
        # effect must cancel from newly retired sums and remain excluded from
        # the entire history, including its earlier migration/confidence.
        second_replay = self.data({0: 6, 1: 5, 2: 6, 3: 6}, replay=True)
        final_model = self.shifted_model(0.3)
        histories = []
        for tag, middle_model in [("regular", regular), ("altered", altered)]:
            checkpoint, path, _ = self.save(args, 1, self.data({2: 8, 3: 8}),
                                              first_replay, first, middle_model, tag=tag)
            second = self.load(args, 2, second_replay, checkpoint, path, model=middle_model)
            _, _, artifact = self.save(args, 2, self.data({4: 8, 5: 8}),
                                        second_replay, second, final_model, tag=tag)
            histories.append(artifact)
        self.assertGreater(float(histories[0]["history_mass_a"][0, 1]), 0)
        for suffix in ["a", "v"]:
            for prefix in ["history_mean_", "history_mass_"]:
                torch.testing.assert_close(histories[0][prefix + suffix][0],
                                           histories[1][prefix + suffix][0])
            self.assertFalse(torch.allclose(histories[0]["history_mean_" + suffix][1, 1],
                                            histories[1]["history_mean_" + suffix][1, 1]))

    def test_c_entire_candidate_bank_uses_the_query_fold(self):
        bank, _, _, artifact = self.second_bank(make_args("historical"))
        order = torch.tensor([0, 3, 5, 7, 12, 2])
        labels = bank.query_labels[order]
        query = bank.query_audio[order].clone().requires_grad_()
        paired = bank.query_visual[order]
        actual = bank.cross_modal_margin(query, labels, paired, 0.4, "visual", order)
        expected = []
        for row, query_index in enumerate(order):
            fold = int(bank.query_folds[query_index])
            sums = artifact["sums_v"] + artifact["history_mass_v"][fold, :, None] * artifact["history_mean_v"][fold]
            expected.append(self.manual_margin(query[row:row + 1], labels[row:row + 1],
                                               paired[row:row + 1], sums, 0.4)[0])
        torch.testing.assert_close(actual, torch.stack(expected))
        actual.sum().backward()
        self.assertGreater(float(query.grad.abs().sum()), 0.0)

    def test_index_wrapper_preserves_data_and_tracks_actual_dataset_index(self):
        replay = self.data({0: 2, 1: 2}, replay=True)
        indexed = IndexedReplayDataset(replay)
        self.assertEqual(len(indexed), len(replay))
        for index in [3, 0, 3, 1]:
            data, label, returned_index = indexed[index]
            expected_data, expected_label = replay[index]
            self.assertEqual(returned_index, index)
            self.assertEqual(label, expected_label)
            torch.testing.assert_close(data[0], expected_data[0])
            torch.testing.assert_close(data[1], expected_data[1])

    def test_scans_preserve_rng_and_individual_module_modes(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        self.model.train()
        self.model.child.eval()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        self.first_bank()
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.child.training)

    def test_changed_checkpoint_is_rejected(self):
        args = make_args()
        _, replay, checkpoint, path, _ = self.first_bank(args)
        with checkpoint.open("ab") as stream:
            stream.write(b"checkpoint changed after bank construction")
        with self.assertRaises((ValueError, RuntimeError)):
            self.load(args, 1, replay, checkpoint, path)

    def test_nontraining_source_is_rejected(self):
        for mode in ["val", "test"]:
            with self.subTest(mode=mode), self.assertRaises((ValueError, RuntimeError)):
                self.save(make_args(), 0, self.data({0: 3, 1: 3}, mode=mode), tag=mode)

    def test_duplicate_support_ids_are_rejected(self):
        train = self.data({0: 3, 1: 3})
        train.records.append(train.records[0])
        train.all_current_data_vids.append(train.all_current_data_vids[0])
        with self.assertRaises((ValueError, RuntimeError)):
            self.save(make_args(), 0, train)

    def test_duplicate_replay_queries_are_rejected(self):
        args = make_args()
        _, replay, checkpoint, path, _ = self.first_bank(args)
        replay.records.append(replay.records[0])
        replay.exemplar_vids_set.append(replay.exemplar_vids_set[0])
        with self.assertRaises((ValueError, RuntimeError)):
            self.load(args, 1, replay, checkpoint, path)

    def test_replay_must_be_subset_of_saved_exact_support(self):
        args = make_args()
        _, _, checkpoint, path, _ = self.first_bank(args)
        outside_support = self.data({0: 7, 1: 1}, replay=True)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load(args, 1, outside_support, checkpoint, path)

    def test_changed_model_with_unchanged_checkpoint_path_is_rejected(self):
        args = make_args()
        _, replay, checkpoint, path, _ = self.first_bank(args)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load(args, 1, replay, checkpoint, path, model=self.shifted_model())

    def test_query_index_and_teacher_pair_are_verified(self):
        bank, _, _, _, _ = self.first_bank()
        with self.assertRaises(ValueError):
            bank.cross_modal_margin(bank.query_audio[:1], bank.query_labels[:1],
                                     bank.query_visual[:1], 0.4, "visual", None)
        with self.assertRaises(ValueError):
            bank.cross_modal_margin(bank.query_audio[:1], bank.query_labels[:1],
                                     bank.query_visual[:1], 0.4, "visual", torch.tensor([2]))
        with self.assertRaises(ValueError):
            bank.cross_modal_margin(bank.query_audio[:1], bank.query_labels[:1],
                                     bank.query_visual[1:2], 0.4, "visual", torch.tensor([0]))

    def test_invalid_feature_failure_restores_modes_and_rng(self):
        self.model.audio_table[0] = 0
        self.model.train()
        self.model.child.eval()
        before = torch.get_rng_state().clone()
        with self.assertRaises((ValueError, FloatingPointError)):
            self.save(make_args(), 0, self.data({0: 3, 1: 3}))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.child.training)

    def test_singleton_exact_support_is_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.first_bank(supports={0: 1, 1: 3}, queries={0: 1, 1: 1})

    def test_artifact_is_weights_only_readable_and_contains_no_feature_table(self):
        _, _, _, path, _ = self.first_bank()
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        self.assertIsInstance(artifact, dict)
        self.assertTrue(path.with_suffix(".json").is_file())
        # IDs and class sums are sufficient. Persisting a row for every exact
        # support embedding would silently introduce a second feature memory.
        for name in ["support_audio", "support_visual", "query_audio", "query_visual"]:
            self.assertNotIn(name, artifact)


if __name__ == "__main__":
    unittest.main()
