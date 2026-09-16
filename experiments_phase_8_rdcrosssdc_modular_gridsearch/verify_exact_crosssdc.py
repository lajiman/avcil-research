"""Verify that the modular exact losses reproduce the original formulas."""

import torch
from torch.nn import functional as F

from rd_crosssdc.exact_losses import (
    cal_contrastive_loss,
    cross_sdc_class_contrastive_loss,
    cross_sdc_z1_loss,
)


def original_ce_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)
    return -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))


def original_instance(feature_1, feature_2, temperature):
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    labels = torch.arange(score.shape[0]).to(score.device)
    return original_ce_loss(score.shape[0], score, labels)


def original_class(feature_1, feature_2, labels, temperature):
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    log_prob = F.log_softmax(score, dim=-1)
    num_pos = positive_mask.sum(dim=1).clamp_min(1.0)
    return -((log_prob * positive_mask).sum(dim=1) / num_pos).mean()


def main():
    torch.manual_seed(42)
    batch_size = 12
    dim = 32
    temperature = 0.05

    cur_a = F.normalize(torch.randn(batch_size, dim), dim=1)
    cur_v = F.normalize(torch.randn(batch_size, dim), dim=1)
    old_a = F.normalize(torch.randn(batch_size, dim), dim=1)
    old_v = F.normalize(torch.randn(batch_size, dim), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    expected_i = 0.5 * (
        original_instance(cur_a, old_v, temperature)
        + original_instance(old_a, cur_v, temperature)
    )
    expected_c = 0.5 * (
        original_class(cur_a, old_v, labels, temperature)
        + original_class(old_a, cur_v, labels, temperature)
    )

    actual_i, actual_c = cross_sdc_z1_loss(
        cur_audio=cur_a,
        cur_visual=cur_v,
        old_audio=old_a,
        old_visual=old_v,
        labels=labels,
        temperature=temperature,
    )

    assert torch.equal(expected_i, actual_i), (expected_i, actual_i)
    assert torch.equal(expected_c, actual_c), (expected_c, actual_c)
    assert torch.equal(
        original_instance(cur_a, old_v, temperature),
        cal_contrastive_loss(cur_a, old_v, temperature),
    )
    assert torch.equal(
        original_class(cur_a, old_v, labels, temperature),
        cross_sdc_class_contrastive_loss(cur_a, old_v, labels, temperature),
    )
    print("Exact CrossSDC formula check passed.")


if __name__ == "__main__":
    main()
