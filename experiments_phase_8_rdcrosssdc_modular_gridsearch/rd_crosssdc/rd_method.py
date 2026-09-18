"""Cross-Modal Margin Retention and adaptive class weighting.

This module is imported by the training script but is called only when the
selected experiment mode actually enables CMR.  The pure CrossSDC control never
builds a prototype bank and never creates a zero-weight CMR graph.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cmr_penalties import cmr_penalty_per_sample


@dataclass
class TeacherPrototypeBank:
    audio_sums: torch.Tensor
    visual_sums: torch.Tensor
    audio_prototypes: torch.Tensor
    visual_prototypes: torch.Tensor
    counts: torch.Tensor
    reliability_a_from_v: Optional[torch.Tensor] = None # optional because the bank can be built without computing trust
    reliability_v_from_a: Optional[torch.Tensor] = None
    trust_a_from_v: Optional[torch.Tensor] = None
    trust_v_from_a: Optional[torch.Tensor] = None


@dataclass
class MarginTerms:
    ref_a_from_v: torch.Tensor
    cur_a_from_v: torch.Tensor
    ref_v_from_a: torch.Tensor
    cur_v_from_a: torch.Tensor
    deficit_a_from_v: torch.Tensor
    deficit_v_from_a: torch.Tensor
    violation_a_from_v: torch.Tensor
    violation_v_from_a: torch.Tensor


@dataclass
class CMRStats:
    loss_a_from_v: torch.Tensor
    loss_v_from_a: torch.Tensor
    active_a_from_v: torch.Tensor
    active_v_from_a: torch.Tensor
    mean_deficit_a_from_v: torch.Tensor
    mean_deficit_v_from_a: torch.Tensor


@dataclass
class _RNGState:
    python_state: object
    numpy_state: tuple
    torch_cpu_state: torch.Tensor
    torch_cuda_state: Optional[object]


def capture_rng_state() -> _RNGState:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return _RNGState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_state=cuda_state,
    )


def restore_rng_state(state: _RNGState) -> None:
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_cpu_state)
    if torch.cuda.is_available() and state.torch_cuda_state is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda_state)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(1e-12)


def normalize_class_weights(
    raw_scores: torch.Tensor,
    alpha: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    """Convert non-negative priorities into detached mean-one weights."""
    """
    The process of the algorithm is as follows:
    1. calculate Reliability R, R -> T(Trust)
    2. calculate sample deficit D, D -> epoch class mean -> EMA -> Need N
    3. calculate priority P = T^gamma * N^eta

    So this function is right after we get the priority P, and we need to normalize it to get the final weight W.
    """
    raw_scores = raw_scores.detach().float().clamp_min(0.0) # here raw score is the priority P. We want the weight W to be detached
    if raw_scores.numel() == 0:
        return raw_scores

    mean_score = raw_scores.mean()
    if (not torch.isfinite(mean_score).item()) or mean_score.item() <= 1e-12:
        relative = torch.ones_like(raw_scores)
    else:
        relative = raw_scores / mean_score.clamp_min(1e-12)

    weights = (1.0 - alpha) + alpha * relative
    # Clip and re-normalize. Repeating prevents re-normalization from moving a
    # clipped endpoint noticeably outside the requested interval.

    '''
    TODO: According to the math of the algorithm, we can also put this part before "weights = (1.0 - alpha) + alpha * relative"
    It can be r' = Rmin + (1-Rmin) * r, or r' = Rmin + (Rmax-Rmin) * r
    However, it can be repetive with the "weights = (1.0 - alpha) + alpha * relative"?
    '''
    for _ in range(4):  # because after normalize, weights might be over the clamp range again, so we do multiple time. However, doing multiple time might make every class similar.
        weights = weights.clamp(min=min_weight, max=max_weight)
        weights = weights / weights.mean().clamp_min(1e-12)

    return weights.detach()


def _cross_modal_margin(
    query: torch.Tensor,    # (B, D)
    labels: torch.Tensor,   # (B,)
    prototypes: torch.Tensor,   # (C, D)
    prototype_sums: torch.Tensor,   # (C, D)
    prototype_counts: torch.Tensor,   # (C,)
    positive_teacher_features: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Target-vs-rest log-odds margin with a leave-one-out target prototype."""
    # Mathematically we do not need to leave-one. But it's ok to preserve this design.
    if prototypes.shape[0] < 2:
        raise ValueError("CMR requires at least two old classes")

    labels = labels.long()
    scores = torch.mm(query, prototypes.transpose(0, 1)) / temperature  # compute the Score. (B, C)

    target_global_proto = prototypes.index_select(0, labels)
    target_counts = prototype_counts.index_select(0, labels)
    # prototype_sums is sum zi, and positive_teacher_features is the zi of the current sample. So we can get the leave-one-out prototype by (sum zi - zi) / (count - 1)
    loo_sums = prototype_sums.index_select(0, labels) - positive_teacher_features.detach()
    # Mathematically, the leave-one-out prototype is (sum zi - zi) / (count - 1).
    # However, since prototype=norm(mean zi), use (count -1) or not does not matter.
    # We know that the prototypes are normalized. If we substitude with the leave-one-out prototypes, is it still normalized? The answer is yes, normalization means ||mu||2 = 1
    loo_proto = F.normalize(loo_sums, dim=1, eps=1e-12)  
    use_loo = target_counts > 1.0
    target_proto = torch.where(use_loo.unsqueeze(1), loo_proto, target_global_proto)

    target_scores = torch.sum(query * target_proto, dim=1) / temperature    # inner product between the query and the target prototype
    scores = scores.scatter(1, labels.unsqueeze(1), target_scores.unsqueeze(1)) # replace the target score with the leave-one-out target score. 

    target_mask = F.one_hot(labels, num_classes=prototypes.shape[0]).bool()
    negative_scores = scores.masked_fill(target_mask, float("-inf"))
    negative_logsumexp = torch.logsumexp(negative_scores, dim=1)
    return target_scores - negative_logsumexp


# frozen teacher
@torch.no_grad()
def build_old_teacher_prototype_bank(
    old_model,
    exemplar_set,
    num_old_classes: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    margin_temperature: float,
    compute_trust: bool,
    trust_shrinkage_beta: float,
) -> TeacherPrototypeBank:
    """Build old-class prototypes from replay memory only.

    This is sufficient for the current E0/E1/E2 design because CMR uses the
    old-only label space.  No current-task data are passed through the teacher.
    """
    old_model.eval()
    loader = DataLoader(
        exemplar_set,
        batch_size=min(batch_size, len(exemplar_set)),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    all_audio = []
    all_visual = []
    all_labels = []

    for data, labels in tqdm(loader, desc="Build old teacher prototype bank"):
        visual = data[0].to(device)
        audio = data[1].to(device)
        labels = labels.to(device).long()

        outputs = old_model(
            visual=visual,
            audio=audio,
            out_feature_before_fusion=True,
        )
        # Original tuple interface: logits, audio_feature, visual_feature.
        # TODO: Check the output interface of the model
        _, audio_feature, visual_feature = outputs
        all_audio.append(audio_feature.detach())
        all_visual.append(visual_feature.detach())
        all_labels.append(labels)

    old_audio = torch.cat(all_audio, dim=0) # for whole memory
    old_visual = torch.cat(all_visual, dim=0)
    labels = torch.cat(all_labels, dim=0)

    counts = torch.zeros(num_old_classes, device=device, dtype=torch.float32)   # count number of each class
    audio_sums = torch.zeros(   # get the prototype_sum in _cross_modal_margin() function
        num_old_classes, old_audio.shape[1], device=device, dtype=old_audio.dtype
    )
    visual_sums = torch.zeros(
        num_old_classes, old_visual.shape[1], device=device, dtype=old_visual.dtype
    )
    ones = torch.ones_like(labels, dtype=torch.float32)
    counts.index_add_(0, labels, ones)  # count number of each class, counts[c] = number of samples in class c
    audio_sums.index_add_(0, labels, old_audio) # S_c^A = sum_{i in class c} z_i^A
    visual_sums.index_add_(0, labels, old_visual) # S_c^V = sum_{i in class c} z_i^V

    if torch.any(counts <= 0):
        missing = torch.nonzero(counts <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError("Missing replay exemplars for old classes: {}".format(missing))

    audio_prototypes = F.normalize(audio_sums, dim=1, eps=1e-12)    # Because we normalize the prototype, we do not need to divide by the counts here.
    visual_prototypes = F.normalize(visual_sums, dim=1, eps=1e-12)

    bank = TeacherPrototypeBank(
        audio_sums=audio_sums,
        visual_sums=visual_sums,
        audio_prototypes=audio_prototypes,
        visual_prototypes=visual_prototypes,
        counts=counts,
    )

    if not compute_trust:
        return bank

    margin_a = _cross_modal_margin( # compute the B_i here
        query=old_audio,
        labels=labels,
        prototypes=visual_prototypes,
        prototype_sums=visual_sums,
        prototype_counts=counts,
        positive_teacher_features=old_visual,
        temperature=margin_temperature,
    )
    margin_v = _cross_modal_margin(
        query=old_visual,
        labels=labels,
        prototypes=audio_prototypes,
        prototype_sums=audio_sums,
        prototype_counts=counts,
        positive_teacher_features=old_audio,
        temperature=margin_temperature,
    )

    prob_a = torch.sigmoid(margin_a)
    prob_v = torch.sigmoid(margin_v)
    reliability_a = torch.zeros(num_old_classes, device=device)
    reliability_v = torch.zeros(num_old_classes, device=device)
    reliability_a.index_add_(0, labels, prob_a)
    reliability_v.index_add_(0, labels, prob_v)
    reliability_a = reliability_a / counts.clamp_min(1.0)   # compute the R_c here
    reliability_v = reliability_v / counts.clamp_min(1.0)

    chance = 1.0 / float(num_old_classes)
    trust_a = ((reliability_a - chance) / (1.0 - chance)).clamp(min=0.0, max=1.0)
    trust_v = ((reliability_v - chance) / (1.0 - chance)).clamp(min=0.0, max=1.0)

    beta = float(trust_shrinkage_beta)  # shrinkage is a method to reduce the variance of the trust estimation. Just a trick to stabilize the trust estimation.
    if beta > 0:
        trust_a = (counts / (counts + beta)) * trust_a + (beta / (counts + beta)) * trust_a.mean()
        trust_v = (counts / (counts + beta)) * trust_v + (beta / (counts + beta)) * trust_v.mean()

    bank.reliability_a_from_v = reliability_a.detach()
    bank.reliability_v_from_a = reliability_v.detach()
    bank.trust_a_from_v = trust_a.detach()
    bank.trust_v_from_a = trust_v.detach()
    return bank


def compute_margin_terms(
    current_audio: torch.Tensor,
    current_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    bank: TeacherPrototypeBank,
    temperature: float,
    tolerance: float,
) -> MarginTerms:
    """Compute old-only reference/current margins and one-sided deficits."""
    # Compute the [Bref - Bcur]+, and the [Bref - Bcur - tolerance]+ 
    ref_a = _cross_modal_margin(
        old_audio,
        labels,
        bank.visual_prototypes,
        bank.visual_sums,
        bank.counts,
        old_visual,
        temperature,
    ).detach()
    cur_a = _cross_modal_margin(
        current_audio,
        labels,
        bank.visual_prototypes,
        bank.visual_sums,
        bank.counts,
        old_visual,
        temperature,
    )

    ref_v = _cross_modal_margin(
        old_visual,
        labels,
        bank.audio_prototypes,
        bank.audio_sums,
        bank.counts,
        old_audio,
        temperature,
    ).detach()
    cur_v = _cross_modal_margin(
        current_visual,
        labels,
        bank.audio_prototypes,
        bank.audio_sums,
        bank.counts,
        old_audio,
        temperature,
    )

    deficit_a = F.relu(ref_a - cur_a)
    deficit_v = F.relu(ref_v - cur_v)
    violation_a = F.relu(ref_a - cur_a - tolerance)
    violation_v = F.relu(ref_v - cur_v - tolerance)

    return MarginTerms(
        ref_a_from_v=ref_a,
        cur_a_from_v=cur_a,
        ref_v_from_a=ref_v,
        cur_v_from_a=cur_v,
        deficit_a_from_v=deficit_a,
        deficit_v_from_a=deficit_v,
        violation_a_from_v=violation_a,
        violation_v_from_a=violation_v,
    )


# def cmr_loss(
#     terms: MarginTerms,
#     labels: torch.Tensor,
#     class_weight_a_from_v: torch.Tensor,
#     class_weight_v_from_a: torch.Tensor,
# ) -> Tuple[torch.Tensor, CMRStats]:
#     # compute the loss and output the stats
#     sample_weight_a = class_weight_a_from_v.index_select(0, labels).detach()
#     sample_weight_v = class_weight_v_from_a.index_select(0, labels).detach()

#     loss_a = _weighted_mean(terms.violation_a_from_v, sample_weight_a)
#     loss_v = _weighted_mean(terms.violation_v_from_a, sample_weight_v)
#     loss = 0.5 * (loss_a + loss_v)

#     stats = CMRStats(
#         loss_a_from_v=loss_a.detach(),
#         loss_v_from_a=loss_v.detach(),
#         active_a_from_v=(terms.violation_a_from_v > 0).float().mean().detach(),
#         active_v_from_a=(terms.violation_v_from_a > 0).float().mean().detach(),
#         mean_deficit_a_from_v=terms.deficit_a_from_v.mean().detach(),
#         mean_deficit_v_from_a=terms.deficit_v_from_a.mean().detach(),
#     )
#     return loss, stats


def cmr_loss(
    terms: MarginTerms,
    labels: torch.Tensor,
    class_weight_a_from_v: torch.Tensor,
    class_weight_v_from_a: torch.Tensor,
    *,
    penalty: str = "hinge",
    penalty_scale: float = 1.0,
    tolerance: float = 0.0,
) -> Tuple[torch.Tensor, CMRStats]:
    # compute the loss and output the stats
    """Trust-weighted mean of the selected per-sample margin objective.

    For non-hinge variants, pass the same tolerance used by
    compute_margin_terms().

    active_* continues to mean retention-constraint violation ratio,
    not gradient-active fraction for every penalty.
    """
    sample_weight_a = (
        class_weight_a_from_v.index_select(0, labels).detach()
    )
    sample_weight_v = (
        class_weight_v_from_a.index_select(0, labels).detach()
    )

    if penalty == "hinge":
        # Preserve the original hinge objective and reduction exactly.
        per_a = terms.violation_a_from_v
        per_v = terms.violation_v_from_a

    else:
        per_a = cmr_penalty_per_sample(
            current_margin=terms.cur_a_from_v,
            reference_margin=terms.ref_a_from_v,
            penalty=penalty,
            tolerance=tolerance,
            scale=penalty_scale,
        )

        per_v = cmr_penalty_per_sample(
            current_margin=terms.cur_v_from_a,
            reference_margin=terms.ref_v_from_a,
            penalty=penalty,
            tolerance=tolerance,
            scale=penalty_scale,
        )

    loss_a = _weighted_mean(per_a, sample_weight_a)
    loss_v = _weighted_mean(per_v, sample_weight_v)
    loss = 0.5 * (loss_a + loss_v)

    if penalty != "hinge" and not torch.isfinite(loss).item():
        raise FloatingPointError(
            "Non-finite CMR loss after weighted reduction"
        )

    stats = CMRStats(
        loss_a_from_v=loss_a.detach(),
        loss_v_from_a=loss_v.detach(),

        # Same diagnostic definition for all penalty choices.
        active_a_from_v=(terms.violation_a_from_v > 0).float().mean().detach(),
        active_v_from_a=(terms.violation_v_from_a > 0).float().mean().detach(),

        mean_deficit_a_from_v=(terms.deficit_a_from_v.mean().detach()),
        mean_deficit_v_from_a=(terms.deficit_v_from_a.mean().detach()),
    )

    return loss, stats


class AdaptiveWeightController:
    """Epoch-level Trust×Need controller for the adaptive experiment.

    - Trust is fixed within an incremental step.
    - Need is collected during an epoch and updates weights only after the epoch.
    - All output class weights are detached and mean-normalized.
    """

    def __init__(
        self,
        trust_a_from_v: torch.Tensor,
        trust_v_from_a: torch.Tensor,
        alpha: float,
        trust_offset: float,
        trust_gamma: float,
        need_delta: float,
        need_eta: float,
        ema_momentum: float,
        min_weight: float,
        max_weight: float,
    ):
        self.trust_a = trust_a_from_v.detach()  # trust is fixed within an incremental step, so we detach it to prevent backpropagation through it
        self.trust_v = trust_v_from_a.detach()
        self.alpha = alpha
        self.trust_offset = trust_offset
        self.trust_gamma = trust_gamma
        self.need_delta = need_delta
        self.need_eta = need_eta
        self.ema_momentum = ema_momentum
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.num_classes = int(self.trust_a.numel())
        device = self.trust_a.device
        self.need_ema_a = torch.zeros(self.num_classes, device=device)
        self.need_ema_v = torch.zeros(self.num_classes, device=device)
        self.need_steps_a = torch.zeros(self.num_classes, device=device, dtype=torch.long)
        self.need_steps_v = torch.zeros(self.num_classes, device=device, dtype=torch.long)
        self.need_a = torch.zeros(self.num_classes, device=device)
        self.need_v = torch.zeros(self.num_classes, device=device)

        # Match the original E2 design: epoch 0 already uses Trust×(delta)^eta
        # weights. Need becomes data-dependent after the first epoch update.
        self.class_weight_a = self._trust_only_weight(self.trust_a)
        self.class_weight_v = self._trust_only_weight(self.trust_v)
        # In the replacement formulation, CMR is the class-level RD term.
        # Therefore CMR, rather than CrossSDC-C, receives Trust x Need weights.
        self.cmr_weight_a = self.class_weight_a
        self.cmr_weight_v = self.class_weight_v
        self.begin_epoch()

    # so far this function is not used
    def _trust_only_weight(self, trust: torch.Tensor) -> torch.Tensor:
        priority = ((trust + self.trust_offset).clamp_min(1e-12)).pow(self.trust_gamma)
        return normalize_class_weights(
            raw_scores=priority,
            alpha=self.alpha,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
        )

    def _trust_need_weight(self, trust: torch.Tensor, need: torch.Tensor) -> torch.Tensor:
        priority = (
            (trust + self.trust_offset).clamp_min(1e-12).pow(self.trust_gamma)
            * (need + self.need_delta).clamp_min(1e-12).pow(self.need_eta)
        )
        return normalize_class_weights(
            raw_scores=priority,
            alpha=self.alpha,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
        )

    def begin_epoch(self) -> None:
        device = self.trust_a.device
        self.epoch_sum_a = torch.zeros(self.num_classes, device=device)
        self.epoch_sum_v = torch.zeros(self.num_classes, device=device)
        self.epoch_count = torch.zeros(self.num_classes, device=device)

    @torch.no_grad()
    def accumulate(self, labels: torch.Tensor, terms: MarginTerms) -> None:
        labels = labels.long()
        ones = torch.ones_like(labels, dtype=torch.float32)
        self.epoch_sum_a.index_add_(0, labels, terms.deficit_a_from_v.detach())
        self.epoch_sum_v.index_add_(0, labels, terms.deficit_v_from_a.detach())
        self.epoch_count.index_add_(0, labels, ones)

    @torch.no_grad()
    def end_epoch(self) -> None:
        present = self.epoch_count > 0
        mean_a = torch.zeros_like(self.epoch_sum_a)
        mean_v = torch.zeros_like(self.epoch_sum_v)
        mean_a[present] = self.epoch_sum_a[present] / self.epoch_count[present]
        mean_v[present] = self.epoch_sum_v[present] / self.epoch_count[present]

        momentum = self.ema_momentum

        self.need_ema_a[present] = (
            momentum * self.need_ema_a[present] + (1.0 - momentum) * mean_a[present]
        )
        self.need_ema_v[present] = (
            momentum * self.need_ema_v[present] + (1.0 - momentum) * mean_v[present]
        )
        self.need_steps_a[present] += 1
        self.need_steps_v[present] += 1

        # TODO: Here use the ema bias correction to get the final need. This is a normal trick for EMA. 
        # TODO: We can also warmup for several epochs
        steps_a = self.need_steps_a[present].float()
        steps_v = self.need_steps_v[present].float()
        correction_a = 1.0 - torch.pow(torch.full_like(steps_a, momentum), steps_a)
        correction_v = 1.0 - torch.pow(torch.full_like(steps_v, momentum), steps_v)
        self.need_a[present] = self.need_ema_a[present] / correction_a.clamp_min(1e-12)
        self.need_v[present] = self.need_ema_v[present] / correction_v.clamp_min(1e-12)

        # self.class_weight_a = self._trust_need_weight(self.trust_a, self.need_a)
        # self.class_weight_v = self._trust_need_weight(self.trust_v, self.need_v)
        # self.cmr_weight_a = self.class_weight_a
        # self.cmr_weight_v = self.class_weight_v

        # Because now in weight calculation we only use the trust, do not need to update.
        self.begin_epoch()

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            "trust_a_from_v": self.trust_a.detach().cpu(),
            "trust_v_from_a": self.trust_v.detach().cpu(),
            "need_a_from_v": self.need_a.detach().cpu(),
            "need_v_from_a": self.need_v.detach().cpu(),
            "class_weight_a_from_v": self.class_weight_a.detach().cpu(),
            "class_weight_v_from_a": self.class_weight_v.detach().cpu(),
            "cmr_weight_a_from_v": self.cmr_weight_a.detach().cpu(),
            "cmr_weight_v_from_a": self.cmr_weight_v.detach().cpu(),
        }
