"""CLI-to-controller coverage for opt-in paper reliability weights on CPU."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_incremental_rd_crosssdc_modular as training
from summarize_logs import namespace


class TinyReplay(Dataset):
    """Three classes, two identical exemplars each, with mismatched modalities."""

    audio = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    visual = torch.tensor([[0., 1.], [1., 0.], [-1., 0.]])

    def __len__(self):
        return 6

    def __getitem__(self, index):
        label = index // 2
        return (self.visual[label], self.audio[label]), label


class FrozenIdentityTeacher(torch.nn.Module):
    def forward(self, *, visual, audio, out_feature_before_fusion):
        assert out_feature_before_fusion
        return torch.zeros(len(audio), 3, device=audio.device), audio, visual


class WeightCliTests(unittest.TestCase):
    def parse(self, *extra):
        parser = training.build_parser()
        args = parser.parse_args([
            "--meta_root", "unused", "--cross_sdc",
            "--rd_mode", "adaptive_crosssdc_cmr", *extra,
        ])
        training.validate_args(parser, args)
        return args

    def test_existing_defaults_and_alpha_one_remain_valid(self):
        defaults = self.parse()
        self.assertFalse(defaults.rd_disable_weight_clipping)
        self.assertEqual(defaults.rd_trust_offset, 0.05)
        self.assertEqual(defaults.rd_trust_shrinkage_beta, 10.0)
        self.assertEqual(defaults.rd_trust_gamma, 1.0)
        self.assertEqual(defaults.rd_class_weight_alpha, 0.5)
        self.assertFalse(self.parse(
            "--rd_class_weight_alpha", "1",
        ).rd_disable_weight_clipping)

    def test_opt_in_alpha_must_be_in_half_open_interval(self):
        for alpha in ["0", "0.8", "0.999"]:
            with self.subTest(alpha=alpha):
                self.assertTrue(self.parse(
                    "--rd_disable_weight_clipping", "--rd_class_weight_alpha", alpha,
                ).rd_disable_weight_clipping)
        for alpha in ["-0.1", "1", "1.1", "nan", "inf"]:
            with self.subTest(alpha=alpha), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse("--rd_disable_weight_clipping", "--rd_class_weight_alpha", alpha)
                self.assertEqual(error.exception.code, 2)

    def test_opt_in_cannot_silently_do_nothing_in_other_modes(self):
        for mode, coefficient in [("crosssdc", "0"), ("crosssdc_cmr", "0.1")]:
            with self.subTest(mode=mode), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse("--rd_disable_weight_clipping", "--rd_mode", mode,
                               "--lam_cmr", coefficient)
                self.assertEqual(error.exception.code, 2)

    def test_legacy_bounds_are_validated_only_when_used(self):
        invalid_bounds = ["--rd_weight_min", "-1", "--rd_weight_max", "-2"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(*invalid_bounds)
        self.assertTrue(self.parse(
            "--rd_disable_weight_clipping", *invalid_bounds,
        ).rd_disable_weight_clipping)

    def test_paper_cli_reaches_real_bank_and_both_cmr_weight_vectors(self):
        args = self.parse(
            "--rd_disable_weight_clipping", "--rd_trust_offset", "0",
            "--rd_trust_shrinkage_beta", "0", "--rd_trust_gamma", "1",
            "--rd_class_weight_alpha", "0.8", "--num_workers", "0",
            "--class_num_per_step", "3", "--exemplar_batch_size", "3",
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(training, "device", torch.device("cpu")), \
                    mock.patch.object(training, "metrics_dir", return_value=directory), \
                    mock.patch.object(training, "build_old_teacher_prototype_bank",
                                      wraps=training.build_old_teacher_prototype_bank) as build:
                bank, controller = training._prepare_optional_rd_state(
                    args, 1, FrozenIdentityTeacher(), TinyReplay(), {},
                )
            self.assertEqual(build.call_args.kwargs["trust_shrinkage_beta"], 0)
            self.assertTrue(Path(directory, "rd_crosssdc", "step_1_static_trust.csv").is_file())

        self.assertFalse(controller.clip_weights)
        for reliability, trust, weights in [
            (bank.reliability_a_from_v, bank.trust_a_from_v, controller.cmr_weight_a),
            (bank.reliability_v_from_a, bank.trust_v_from_a, controller.cmr_weight_v),
        ]:
            expected_trust = ((reliability - 1 / 3) / (1 - 1 / 3)).clamp(0, 1)
            torch.testing.assert_close(trust, expected_trust)
            expected_weights = 0.2 + 0.8 * expected_trust / expected_trust.mean()
            torch.testing.assert_close(weights, expected_weights)
            self.assertTrue((trust == 0).any())
            torch.testing.assert_close(weights[trust == 0], torch.full_like(weights[trust == 0], 0.2))
            self.assertLess(weights.min().item(), args.rd_weight_min)
            self.assertGreater(weights.max().item(), args.rd_weight_max)
            self.assertFalse(weights.requires_grad)

        # The ordinary printed Namespace records the switch for log summaries.
        recorded = namespace(str(args))
        self.assertIs(recorded["rd_disable_weight_clipping"], True)
        self.assertEqual(recorded["rd_trust_offset"], 0)


if __name__ == "__main__":
    unittest.main()
