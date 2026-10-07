"""CPU integration of persistent prototypes with the unchanged phase-8 losses.

The network is deliberately tiny, but optimizer steps, checkpoint selection,
memory indices, CMR/Trust, history transport and bank serialization are real.
No feature files or GPU are required.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_incremental_rd_crosssdc_modular as training


class TinyAVModel(nn.Module):
    """Original tuple API, normalized branch features and trainable attention."""

    last_created = None

    def __init__(self, args, step_out_class_num):
        super().__init__()
        self.audio_proj = nn.Linear(8, 8)
        self.visual_proj = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, step_out_class_num)
        TinyAVModel.last_created = self

    def incremental_classifier(self, num_classes):
        previous = self.classifier
        self.classifier = nn.Linear(8, num_classes)
        with torch.no_grad():
            self.classifier.weight[:previous.out_features].copy_(previous.weight)
            self.classifier.bias[:previous.out_features].copy_(previous.bias)

    def forward(self, *, visual, audio, out_feature_before_fusion=False,
                out_attn_score=False):
        a = F.normalize(self.audio_proj(audio), dim=1)
        v = F.normalize(self.visual_proj(visual), dim=1)
        logits = self.classifier(a + v)
        if not out_feature_before_fusion:
            return logits
        if not out_attn_score:
            return logits, a, v
        spatial = (a + v).reshape(-1, 2, 2, 2).softmax(dim=2)
        temporal = (a[:, :4] + v[:, :4]).reshape(-1, 2, 2).softmax(dim=1)
        return logits, a, v, spatial, temporal


class SyntheticData(Dataset):
    """Training-only metadata and deterministic paired features for six classes."""

    def __init__(self, classes, per_class, *, replay=False, mode="train"):
        self.category_encode_dict = {"class_{}".format(c): c for c in range(6)}
        self.all_id_category_dict = {
            "c{}_s{}".format(c, i): "class_{}".format(c)
            for c in range(6) for i in range(12)
        }
        self.ids = ["c{}_s{}".format(c, i) for c in classes for i in range(per_class)]
        if replay:
            self.exemplar_vids_set = self.ids
        else:
            self.mode = mode
            self.all_current_data_vids = self.ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        vid = self.ids[index]
        label = self.category_encode_dict[self.all_id_category_dict[vid]]
        sample = int(vid.split("_s")[1])
        # Local generator keeps data reads from disturbing the trainer's RNG.
        generator = torch.Generator().manual_seed(100 * label + sample)
        a = torch.eye(8)[label] + 0.1 * torch.randn(8, generator=generator)
        v = torch.eye(8)[label] + 0.1 * torch.randn(8, generator=generator)
        return (v, a), label


class PrototypeTrainingTests(unittest.TestCase):
    def parse(self, policy="historical", *extra):
        parser = training.build_parser()
        args = parser.parse_args([
            "--meta_root", "unused", "--dataset", "VGGSound_synthetic",
            "--cross_sdc", "--rd_mode", "adaptive_crosssdc_cmr",
            "--num_classes", "6", "--class_num_per_step", "2",
            "--memory_size", "16", "--num_workers", "0",
            "--train_batch_size", "8", "--exemplar_batch_size", "16",
            "--infer_batch_size", "12", "--max_epoches", "2",
            "--lr", "0.01", "--instance_contrastive", "--class_contrastive",
            "--attn_score_distil", "--log_loss_components",
            "--rd_prototype_policy", policy, *extra,
        ])
        training.validate_args(parser, args)
        return args

    def temporary_training(self, directory):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(training, "device", torch.device("cpu")))
        stack.enter_context(mock.patch.object(training.torch.cuda, "device_count", return_value=0))
        stack.enter_context(mock.patch.object(training, "IncreAudioVisualNet", TinyAVModel))
        stack.enter_context(mock.patch.object(training, "checkpoint_path", side_effect=lambda a, s: str(Path(directory, "step_{}_best.pkl".format(s)))))
        stack.enter_context(mock.patch.object(training, "prototype_bank_path", side_effect=lambda a, s: str(Path(directory, "step_{}_bank.pt".format(s)))))
        stack.enter_context(mock.patch.object(training, "metrics_dir", return_value=str(Path(directory, "metrics"))))
        stack.enter_context(mock.patch.object(training, "figure_dir", return_value=directory))
        # Plotting/progress are presentation only; losses and all I/O stay real.
        stack.enter_context(mock.patch.object(training, "plt"))
        stack.enter_context(mock.patch.object(training, "tqdm", side_effect=lambda x: x))
        stack.enter_context(mock.patch.object(training, "tzip", side_effect=lambda *x: zip(*x)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_b_and_c_train_three_tasks_with_reduced_memory(self):
        for policy in ("pre_shrink", "historical"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                args = self.parse(policy)
                training.setup_seed(args.seed)
                banks, artifacts = [], []
                with self.temporary_training(directory):
                    for step in range(3):
                        train = SyntheticData(range(2 * step, 2 * step + 2), 12)
                        val = SyntheticData(range(2 * step + 2), 2, mode="val")
                        replay = SyntheticData(range(2 * step), 8 if step == 1 else 4, replay=True)
                        previous = None if step == 0 else torch.load(
                            training.checkpoint_path(args, step - 1), weights_only=False)
                        bank = training.train(args, step, train, val, replay, {})
                        banks.append(bank)
                        artifacts.append(training.export_best_prototype_bank(
                            args, step, train, replay, bank))
                        saved = torch.load(training.checkpoint_path(args, step), weights_only=False)
                        self.assertTrue(all(torch.isfinite(p).all() for p in saved.parameters()))
                        if previous is not None:
                            self.assertFalse(torch.equal(previous.audio_proj.weight, saved.audio_proj.weight))
                self.assertIsNone(banks[0])
                self.assertEqual(banks[1].counts.tolist(), [12, 12])
                self.assertEqual(banks[1].query_counts.tolist(), [8, 8])
                self.assertEqual(banks[2].counts.tolist(), [8, 8, 12, 12])
                self.assertEqual(banks[2].query_counts.tolist(), [4, 4, 4, 4])
                if policy == "historical":
                    self.assertGreater(artifacts[1]["history_mass_a"].sum().item(), 0)
                    self.assertGreater(banks[2].artifact["history_mass_a"].sum().item(), 0)
                    self.assertEqual(artifacts[2]["retired_unique_counts"].tolist(), [8, 8, 8, 8, 0, 0])
                else:
                    self.assertEqual(artifacts[2]["history_mass_a"].sum().item(), 0)
                with open(Path(directory, "metrics", "rd_crosssdc", "epoch_summary.csv"), newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 6)
                for row in rows:
                    for name in ("train_loss", "cross_sdc_i", "cross_sdc_c", "cmr"):
                        self.assertTrue(torch.isfinite(torch.tensor(float(row[name]))), (policy, name, row))
                with open(Path(directory, "metrics", "rd_crosssdc", "loss_components.csv"), newline="") as handle:
                    components = list(csv.DictReader(handle))
                enabled = {row["component"] for row in components if row["step"] == "2" and row["enabled"] == "1"}
                self.assertEqual(enabled, {"ce", "kd", "instance_contrastive", "class_contrastive",
                                           "cross_sdc_i", "cross_sdc_c", "cmr", "attn_spatial", "attn_temporal"})
                self.assertLess(max(float(row["reconstruction_max_abs_residual"]) for row in components), 1e-5)

    def test_export_reloads_selected_best_instead_of_final_epoch(self):
        args = self.parse("pre_shrink")
        training.setup_seed(args.seed)
        train = SyntheticData(range(2), 12)
        replay = SyntheticData([], 0, replay=True)
        val = SyntheticData(range(2), 2, mode="val")
        with tempfile.TemporaryDirectory() as directory, self.temporary_training(directory):
            # The second epoch updates the student but cannot replace epoch 0.
            with mock.patch.object(training, "top_1_acc", side_effect=[1.0, 0.0]):
                training.train(args, 0, train, val, replay, {})
            saved = torch.load(training.checkpoint_path(args, 0), weights_only=False)
            final = TinyAVModel.last_created
            self.assertFalse(torch.equal(saved.audio_proj.weight, final.audio_proj.weight))
            with mock.patch.object(training, "save_task_bank", wraps=training.save_task_bank) as export:
                training.export_best_prototype_bank(args, 0, train, replay, None)
            actual = export.call_args.kwargs["best_model"]
            torch.testing.assert_close(actual.audio_proj.weight, saved.audio_proj.weight, rtol=0, atol=0)

    def test_default_a_still_uses_original_memory_builder(self):
        parser = training.build_parser()
        defaults = parser.parse_args(["--meta_root", "unused"])
        self.assertEqual(defaults.rd_prototype_policy, "memory")
        args = self.parse("memory")
        with tempfile.TemporaryDirectory() as directory, self.temporary_training(directory), \
                mock.patch.object(training, "load_task_bank") as load, \
                mock.patch.object(training, "save_task_bank") as save, \
                mock.patch.object(training, "build_old_teacher_prototype_bank", wraps=training.build_old_teacher_prototype_bank) as build:
            teacher = TinyAVModel(args, 2)
            bank, controller = training._prepare_optional_rd_state(
                args, 1, teacher, SyntheticData(range(2), 4, replay=True), {})
            self.assertEqual(bank.counts.tolist(), [4, 4])
            self.assertIsNotNone(controller)
            build.assert_called_once()
            load.assert_not_called()
            save.assert_not_called()

    def test_test_only_never_loads_or_exports_prototypes(self):
        args = self.parse("historical", "--test_only")
        fake_dataset = mock.MagicMock()
        fake_dataset.category_encode_dict = {"class_0": 0}
        with tempfile.TemporaryDirectory() as directory, self.temporary_training(directory), \
                mock.patch.object(training, "build_parser") as parser_factory, \
                mock.patch.object(training, "IcaAVELoader", return_value=fake_dataset), \
                mock.patch.object(training, "exemplarLoader", return_value=fake_dataset), \
                mock.patch.object(training, "train") as train, \
                mock.patch.object(training, "load_task_bank") as load, \
                mock.patch.object(training, "save_task_bank") as save, \
                mock.patch.object(training, "detailed_test", return_value=0.0) as test:
            parser_factory.return_value.parse_args.return_value = args
            # main() has a legacy relative output mkdir; confine it to the fixture.
            original_cwd = os.getcwd()
            try:
                os.chdir(directory)
                training.main()
            finally:
                os.chdir(original_cwd)
            self.assertEqual(test.call_count, 3)
            train.assert_not_called()
            load.assert_not_called()
            save.assert_not_called()

    def test_existing_experiment_is_rejected_before_data_or_output_changes(self):
        args = self.parse("historical")
        parser = training.build_parser()
        with tempfile.TemporaryDirectory() as directory, self.temporary_training(directory), \
                mock.patch.object(training, "build_parser", return_value=parser), \
                mock.patch.object(parser, "parse_args", return_value=args), \
                mock.patch.object(training, "IcaAVELoader") as load_data, \
                redirect_stderr(io.StringIO()):
            existing = Path(directory, "step_0_best.pkl")
            existing.write_bytes(b"existing experiment must survive")
            with self.assertRaises(SystemExit):
                training.main()
            load_data.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"existing experiment must survive")

    def test_invalid_bank_cli_configuration_fails_before_training(self):
        invalid = [
            ("pre_shrink", ["--rd_mode", "crosssdc", "--lam_cmr", "0"]),
            ("pre_shrink", ["--class_num_per_step", "1"]),
            ("pre_shrink", ["--num_classes", "7"]),
            ("pre_shrink", ["--memory_size", "3"]),
            ("pre_shrink", ["--memory_size", "4"]),  # Would fail final snapshot with 1 support/class.
            ("pre_shrink", ["--max_epoches", "0"]),
            ("historical", ["--rd_history_folds", "1"]),
            ("historical", ["--rd_history_min_anchors", "1"]),
            ("historical", ["--rd_history_mass_cap", "-1"]),
            ("historical", ["--rd_history_mass_cap", "nan"]),
            ("historical", ["--rd_history_decay", "1.1"]),
            ("historical", ["--rd_history_error_scale", "0"]),
        ]
        for policy, extra in invalid:
            with self.subTest(policy=policy, extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse(policy, *extra)
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.parse("historical", "--rd_history_mass_cap", "0").rd_history_mass_cap, 0)


if __name__ == "__main__":
    unittest.main()
