"""Regression coverage for legacy and opt-in unclipped class weighting."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rd_crosssdc.rd_method import (
    AdaptiveWeightController,
    MarginTerms,
    cmr_loss,
    normalize_class_weights,
)


class NormalizeClassWeightTests(unittest.TestCase):
    def normalize(self, scores, alpha=0.8, **kwargs):
        return normalize_class_weights(
            torch.as_tensor(scores, dtype=torch.float32),
            alpha=alpha,
            min_weight=0.5,
            max_weight=2.0,
            **kwargs,
        )

    def test_default_preserves_legacy_clipping(self):
        # Recorded from the original four-pass clip/renormalize algorithm.
        expected = torch.tensor(
            [0.528301835, 0.528301835, 0.935849130, 2.007547140]
        )
        actual = self.normalize([0.0, 0.1, 0.3, 1.0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        self.assertTrue(torch.equal(
            actual,
            self.normalize([0.0, 0.1, 0.3, 1.0], clip_weights=True),
        ))

    def test_default_still_accepts_legacy_alpha_one(self):
        actual = self.normalize([0.0, 0.1, 0.9], alpha=1.0)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(actual.min().item(), 0)

    def test_unclipped_formula_preserves_uniform_baseline(self):
        # R / mean(R) = [0, .3, 2.7], then .2 + .8 * relative.
        actual = self.normalize([0.0, 0.1, 0.9], clip_weights=False)
        torch.testing.assert_close(actual, torch.tensor([0.2, 0.44, 2.36]))
        self.assertAlmostEqual(actual[0].item(), 1.0 - 0.8)
        self.assertAlmostEqual(actual.mean().item(), 1.0)

    def test_unclipped_weights_ignore_legacy_min_and_max(self):
        actual = normalize_class_weights(
            torch.tensor([0.0, 0.1, 0.9]),
            alpha=0.8,
            min_weight=0.9,
            max_weight=1.1,
            clip_weights=False,
        )
        torch.testing.assert_close(actual, torch.tensor([0.2, 0.44, 2.36]))

    def test_tiny_positive_reliabilities_keep_their_ratios(self):
        actual = self.normalize([0.0, 1e-20, 9e-20], clip_weights=False)
        torch.testing.assert_close(actual, torch.tensor([0.2, 0.44, 2.36]))

    def test_large_finite_priorities_do_not_overflow_the_mean(self):
        actual = self.normalize([1e38, 2e38, 3e38], clip_weights=False)
        torch.testing.assert_close(actual, torch.tensor([0.6, 1.0, 1.4]))

    def test_all_zero_reliabilities_fall_back_to_uniform_weights(self):
        actual = self.normalize([0.0, 0.0, 0.0], clip_weights=False)
        self.assertTrue(torch.equal(actual, torch.ones(3)))

    def test_alpha_zero_returns_uniform_weights(self):
        actual = self.normalize([0.0, 0.1, 0.9], alpha=0.0, clip_weights=False)
        self.assertTrue(torch.equal(actual, torch.ones(3)))

    def test_unclipped_mode_rejects_invalid_alpha(self):
        for alpha in [-0.1, 1.0, 1.1, float("nan"), float("inf")]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                self.normalize([0.0, 0.1, 0.9], alpha=alpha, clip_weights=False)

    def test_unclipped_mode_rejects_nonfinite_priorities(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                self.normalize([0.0, value, 0.9], clip_weights=False)

    def test_weights_are_detached_and_input_is_unchanged(self):
        scores = torch.tensor([0.0, 0.1, 0.9], requires_grad=True)
        original = scores.detach().clone()
        actual = self.normalize(scores, clip_weights=False)
        self.assertFalse(actual.requires_grad)
        self.assertIsNone(actual.grad_fn)
        self.assertTrue(torch.equal(scores, original))


class AdaptiveWeightControllerTests(unittest.TestCase):
    def make_controller(self, **overrides):
        options = dict(
            trust_a_from_v=torch.tensor([0.0, 0.1, 0.9], requires_grad=True),
            trust_v_from_a=torch.tensor([0.8, 0.2, 0.0], requires_grad=True),
            alpha=0.8,
            trust_offset=0.0,
            trust_gamma=1.0,
            need_delta=0.05,
            need_eta=0.0,
            ema_momentum=0.9,
            min_weight=0.5,
            max_weight=2.0,
        )
        options.update(overrides)
        return AdaptiveWeightController(**options)

    def test_default_controller_preserves_original_weights(self):
        controller = self.make_controller(trust_offset=0.05)
        explicit = self.make_controller(trust_offset=0.05, clip_weights=True)
        torch.testing.assert_close(
            controller.class_weight_a,
            torch.tensor([0.499989986, 0.510382712, 1.989627242]),
            rtol=0,
            atol=1e-7,
        )
        self.assertTrue(torch.equal(
            controller.class_weight_a, explicit.class_weight_a
        ))
        self.assertTrue(torch.equal(
            controller.class_weight_v, explicit.class_weight_v
        ))

    def test_unclipped_controller_applies_formula_to_both_directions(self):
        controller = self.make_controller(clip_weights=False)
        torch.testing.assert_close(
            controller.class_weight_a, torch.tensor([0.2, 0.44, 2.36])
        )
        torch.testing.assert_close(
            controller.class_weight_v, torch.tensor([2.12, 0.68, 0.2])
        )
        for weight in [controller.cmr_weight_a, controller.cmr_weight_v]:
            self.assertFalse(weight.requires_grad)
        self.assertTrue(torch.equal(
            controller.cmr_weight_a, controller.class_weight_a
        ))
        self.assertTrue(torch.equal(
            controller.cmr_weight_v, controller.class_weight_v
        ))

    def test_trust_need_route_keeps_zero_trust_at_uniform_baseline(self):
        controller = self.make_controller(clip_weights=False, need_eta=1.0)
        actual = controller._trust_need_weight(
            controller.trust_a, torch.tensor([0.4, 0.45, 0.95])
        )
        # Priorities are [0, .05, .9]; the zero-trust class retains only .2.
        torch.testing.assert_close(
            actual, torch.tensor([0.2, 0.2 + 0.8 * 3 / 19, 0.2 + 0.8 * 54 / 19])
        )

    def test_tiny_trust_is_not_replaced_by_a_priority_floor(self):
        controller = self.make_controller(
            clip_weights=False,
            trust_a_from_v=torch.tensor([0.0, 1e-20, 9e-20]),
        )
        expected = torch.tensor([0.2, 0.44, 2.36])
        torch.testing.assert_close(controller.class_weight_a, expected)
        # eta=0 means the need route must agree with trust-only weighting.
        torch.testing.assert_close(
            controller._trust_need_weight(controller.trust_a, torch.ones(3)),
            expected,
        )

    def test_cmr_loss_uses_unclipped_weights_in_both_directions(self):
        controller = self.make_controller(clip_weights=False)
        labels = torch.tensor([0, 2, 2, 1])
        violation_a = torch.tensor([1.0, 3.0, 2.0, 4.0], requires_grad=True)
        violation_v = torch.tensor([4.0, 1.0, 5.0, 2.0], requires_grad=True)
        zeros = torch.zeros(4)
        terms = MarginTerms(
            ref_a_from_v=zeros,
            cur_a_from_v=zeros,
            ref_v_from_a=zeros,
            cur_v_from_a=zeros,
            deficit_a_from_v=violation_a.detach(),
            deficit_v_from_a=violation_v.detach(),
            violation_a_from_v=violation_a,
            violation_v_from_a=violation_v,
        )
        loss, stats = cmr_loss(
            terms, labels, controller.cmr_weight_a, controller.cmr_weight_v
        )
        # Sample weights: a=[.2,2.36,2.36,.44], v=[2.12,.2,.2,.68].
        self.assertAlmostEqual(stats.loss_a_from_v.item(), 13.76 / 5.36, places=6)
        self.assertAlmostEqual(stats.loss_v_from_a.item(), 11.04 / 3.2, places=6)
        self.assertAlmostEqual(loss.item(), (13.76 / 5.36 + 11.04 / 3.2) / 2, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(violation_a.grad).all())
        self.assertTrue(torch.isfinite(violation_v.grad).all())


if __name__ == "__main__":
    unittest.main()
