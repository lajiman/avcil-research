"""AVCIL and CrossSDC losses.

The functions in the first two sections are direct extractions from the user's
working AVCIL/CrossSDC script.  Do not rewrite them with alternative PyTorch
reductions when exact CrossSDC comparability is required.
"""

from typing import Dict, Tuple

import torch
from torch.nn import functional as F


# =============================================================================
# 1. Exact losses from the original AVCIL training script
# =============================================================================

def CE_loss(num_classes: int, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Original one-hot cross-entropy implementation."""
    targets = F.one_hot(label, num_classes=num_classes) # B*C 
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))   # (B,C) --logsoftmax in C--> (B,C) --sum--> (B,)
    return loss


# Lowercase alias used by the modular trainer; it is the same function object.
ce_loss = CE_loss


def cal_contrastive_loss(   # instance level contrastive loss
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Original instance contrastive loss."""
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature    # (B, B)
    num_sample = score.shape[0]
    label = torch.arange(num_sample).to(score.device)
    return CE_loss(num_sample, score, label)


def class_contrastive_loss( # class level contrastive loss
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    label: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Original AVCIL current-current class contrastive loss."""
    class_matrix = label.unsqueeze(0)   # (B,) --> (1, B)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = class_matrix == label.unsqueeze(-1)
    class_matrix = class_matrix.float()

    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss = -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)) # same class, rather than same instance
    # actually here is 1/B, rather than 1/|Pi|
    # that's why author have the notation below:
    ###################################################################################################
    # As the author mentioned,
    # You can also use the following implementation, which is more consistent with Equation (7) in our paper (and also the standard InfoNCE), 
    # but you may need to further adjust the hyperparameters lam_I and lam_C to get optimal performance.
    # loss = -torch.mean(
    #     (torch.sum(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)) / torch.sum(class_matrix, dim=-1))
    ###################################################################################################
    return loss


# =============================================================================
# 2. Exact CrossSDC losses from the working CrossSDC script
# =============================================================================

def cross_sdc_class_contrastive_loss(   # similar as class_contrastive_loss. Here normalize using the number of each anchor, more close to the standard InfoNCE.
    feature_1: torch.Tensor,            # apart of this, it is the same as class_contrastive_loss. Here does not show the character of "cross", but it is the same as the original implementation. The "cross" is in the function name, which means that the two features are from different modalities.
    feature_2: torch.Tensor,            # the charactor of "cross" depends on the inputs
    label: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Original positive-count-normalized CrossSDC-C."""
    positive_mask = (label.unsqueeze(1) == label.unsqueeze(0)).float()  # using the broadcast mechanism, the same as unsqueeze --> repeat --> ==

    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    log_prob = F.log_softmax(score, dim=-1)

    num_pos = positive_mask.sum(dim=1).clamp_min(1.0)
    loss = -((log_prob * positive_mask).sum(dim=1) / num_pos).mean()
    return loss


def cross_sdc_instance_loss(
    cur_audio: torch.Tensor,
    cur_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Exact CrossSDC-I used by the original implementation."""
    inst_cur_old = cal_contrastive_loss(cur_audio, old_visual, temperature=temperature)
    inst_old_cur = cal_contrastive_loss(old_audio, cur_visual, temperature=temperature)
    return 0.5 * (inst_cur_old + inst_old_cur)


def cross_sdc_class_loss(
    cur_audio: torch.Tensor, 
    cur_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Exact CrossSDC_C used by the original implementation."""
    cls_cur_old = cross_sdc_class_contrastive_loss(cur_audio, old_visual, labels, temperature=temperature)
    cls_old_cur = cross_sdc_class_contrastive_loss(old_audio, old_visual, labels, temperature=temperature)
    return 0.5 * (cls_cur_old + cls_old_cur)


def cross_sdc_z1_loss(
    cur_audio: torch.Tensor,
    cur_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact original CrossSDC-I + CrossSDC-C execution path."""
    # The body below is deliberately kept in the same form as the working
    # CrossSDC script rather than routed through a rewritten reduction.
    # very clear, with no weighting.
    inst_cur_old = cal_contrastive_loss(
        cur_audio, old_visual, temperature=temperature
    )
    inst_old_cur = cal_contrastive_loss(
        old_audio, cur_visual, temperature=temperature
    )
    loss_inst = 0.5 * (inst_cur_old + inst_old_cur)

    cls_cur_old = cross_sdc_class_contrastive_loss(
        cur_audio, old_visual, labels, temperature=temperature
    )
    cls_old_cur = cross_sdc_class_contrastive_loss(
        old_audio, cur_visual, labels, temperature=temperature
    )
    loss_cls = 0.5 * (cls_cur_old + cls_old_cur)

    return loss_inst, loss_cls


# =============================================================================
# 3. Minimal extension needed only by the adaptive CrossSDC-C experiment
# =============================================================================

# TODO: maybe we need a weighted_cross_sdc_instance_loss() function. 
# Apart of this, this document is very clear.

def _cross_sdc_class_per_anchor(
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-anchor form of the original CrossSDC-C formula.

    Averaging the returned vector with ``.mean()`` is exactly the original
    ``cross_sdc_class_contrastive_loss``.  Exposing the vector is the minimum
    change needed for class-wise weighting.
    """
    # The same as the original CrossSDC-C implementation, but we return the per-anchor loss vector and the positive count vector.
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    log_prob = F.log_softmax(score, dim=-1)
    num_pos = positive_mask.sum(dim=1).clamp_min(1.0)
    per_anchor = -((log_prob * positive_mask).sum(dim=1) / num_pos) # in the original implementation, the mean() is used to get the final loss. Here we return the per-anchor loss vector and the positive count vector.
    return per_anchor, num_pos


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(1e-12)


def weighted_cross_sdc_class_loss(
    cur_audio: torch.Tensor,
    cur_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    class_weight_a_from_v: torch.Tensor,
    class_weight_v_from_a: torch.Tensor,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Trust×Need weighted CrossSDC-C using the original per-anchor formula.

    The two temporal orientations remain identical to the working CrossSDC
    implementation:
      current audio -> old visual
      old audio     -> current visual
    """
    per_a, num_pos = _cross_sdc_class_per_anchor(
        cur_audio, old_visual, labels, temperature
    )
    per_v, _ = _cross_sdc_class_per_anchor(
        old_audio, cur_visual, labels, temperature
    )

    # The weights are selected based on the labels, and then detached to prevent backpropagation through them.
    # The detach() is important because we don't want to learn the weights during backpropagation.
    sample_weight_a = class_weight_a_from_v.index_select(0, labels).detach()    # here the detach() is a important charactor of our algorithm, means that we should not learn the weight
    sample_weight_v = class_weight_v_from_a.index_select(0, labels).detach()    # we should aware that "weighted" does not require detach(). It is our algorithm that requires detach()

    loss_a = _weighted_mean(per_a, sample_weight_a)
    loss_v = _weighted_mean(per_v, sample_weight_v)
    loss = 0.5 * (loss_a + loss_v)

    stats = {
        "loss_a_from_v": loss_a.detach(),
        "loss_v_from_a": loss_v.detach(),
        "mean_positive_count": num_pos.float().mean().detach(),
        "singleton_ratio": (num_pos <= 1.0).float().mean().detach(),
    }
    return loss, stats
