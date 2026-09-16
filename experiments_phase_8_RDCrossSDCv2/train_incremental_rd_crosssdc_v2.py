import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random
from itertools import cycle
import csv
import json

from tsne_plotter import make_tsne_plots_for_step

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def CE_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    return loss

def cal_contrastive_loss(feature_1, feature_2, temperature=0.1):
    # (BS, BS)
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    num_sample = score.shape[0]
    label = torch.arange(num_sample).to(score.device)

    loss = CE_loss(num_sample, score, label)
    return loss

def class_contrastive_loss(feature_1, feature_2, label, temperature=0.1):
    class_matrix = label.unsqueeze(0)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = class_matrix == label.unsqueeze(-1)
    # (BS, BS)
    class_matrix = class_matrix.float()
    # (BS, BS)
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss = -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))

    ###################################################################################################
    # You can also use the following implementation, which is more consistent with Equation (7) in our paper, 
    # but you may need to further adjust the hyperparameters lam_I and lam_C to get optimal performance.
    # loss = -torch.mean(
    #     (torch.sum(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)) / torch.sum(class_matrix, dim=-1))
    ###################################################################################################

    return loss



# ============================================================================
# RD-CrossSDC v2 helpers
#
# v2 separates three roles:
#   1) persistent temporal instance alignment (uniform CrossSDC-I),
#   2) persistent class-level alignment whose normalized class weights follow
#      teacher Trust x current retention Need,
#   3) a small one-sided Cross-Modal Margin Retention (CMR) guardrail.
# ============================================================================

def _rd_weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12):
    """Weighted mean with a scale-invariant denominator."""
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(eps)


def _rd_normalize_class_weights(
    raw_scores: torch.Tensor,
    alpha: float,
    min_weight: float,
    max_weight: float,
    eps: float = 1e-12,
):
    """
    Convert non-negative class priorities into mean-one weights.

    alpha controls redistribution only. The global loss coefficients control
    regularization strength. Clipping is followed by re-normalization so that
    mean(weight)=1 remains true.
    """
    raw_scores = raw_scores.detach().float().clamp_min(0.0)
    if raw_scores.numel() == 0:
        return raw_scores

    raw_mean = raw_scores.mean()
    if (not torch.isfinite(raw_mean).item()) or raw_mean.item() <= eps:
        relative = torch.ones_like(raw_scores)
    else:
        relative = raw_scores / raw_mean.clamp_min(eps)

    weights = (1.0 - alpha) + alpha * relative
    # Iterate because a final normalization can otherwise move a clipped value
    # slightly outside the requested interval.
    for _ in range(4):
        weights = weights.clamp(min=min_weight, max=max_weight)
        weights = weights / weights.mean().clamp_min(eps)
    return weights.detach()


def _rd_make_trust_need_weights(args, trust: torch.Tensor, need: torch.Tensor):
    """Mean-one class weights from chance-corrected teacher trust and EMA need."""
    priority = (
        (trust + args.rd_trust_offset).clamp_min(1e-12).pow(args.rd_trust_gamma)
        * (need + args.rd_need_delta).clamp_min(1e-12).pow(args.rd_need_eta)
    )
    return _rd_normalize_class_weights(
        raw_scores=priority,
        alpha=args.rd_class_weight_alpha,
        min_weight=args.rd_weight_min,
        max_weight=args.rd_weight_max,
    )


def _rd_make_trust_only_weights(args, trust: torch.Tensor):
    """Trust-only weights used by CMR; the hinge already supplies sample-level need."""
    return _rd_normalize_class_weights(
        raw_scores=trust + args.rd_trust_offset,
        alpha=args.rd_class_weight_alpha,
        min_weight=args.rd_weight_min,
        max_weight=args.rd_weight_max,
    )


def _rd_base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _rd_capture_rng_state():
    """Capture RNG states so step-level diagnostics do not alter training shuffles."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _rd_restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _rd_extract_teacher_features(args, old_model, visual, audio):
    """
    Return normalized teacher audio, guided visual, and trust-estimation visual features.

    When rd_trust_visual_space=uniform, trust is estimated from an audio-independent
    uniformly pooled visual representation after the frozen teacher's visual_proj.
    The actual CrossSDC and margin losses still act on the guided z1 visual feature.
    """
    need_uniform = args.rd_trust_visual_space == 'uniform'
    outputs = old_model(
        visual=visual,
        audio=audio,
        out_feature_before_fusion=True,
        out_analysis_features=need_uniform,
        return_dict=True,
    )
    audio_feature = outputs['audio_feature'].detach()
    guided_visual = outputs['visual_feature'].detach()

    if need_uniform:
        base = _rd_base_model(old_model)
        uniform_raw = outputs['z0_visual_uniform_raw']
        uniform_visual = F.relu(base.visual_proj(uniform_raw))
        trust_visual = F.normalize(uniform_visual, dim=1, eps=1e-12).detach()
    else:
        trust_visual = guided_visual

    return audio_feature, guided_visual, trust_visual


def _rd_cross_modal_margin(
    query: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_sums: torch.Tensor,
    prototype_counts: torch.Tensor,
    positive_teacher_features: torch.Tensor,
    temperature: float,
):
    """Target-vs-rest prototype log-odds margin with a leave-one-out positive."""
    if temperature <= 0:
        raise ValueError('RD-CrossSDC temperatures must be positive')
    if prototypes.shape[0] < 2:
        raise ValueError('Cross-modal margin requires at least two candidate classes')

    labels = labels.long()
    scores = torch.mm(query, prototypes.transpose(0, 1)) / temperature

    target_global_proto = prototypes.index_select(0, labels)
    target_counts = prototype_counts.index_select(0, labels)
    loo_sums = prototype_sums.index_select(0, labels) - positive_teacher_features.detach()
    loo_proto = F.normalize(loo_sums, dim=1, eps=1e-12)
    use_loo = target_counts > 1.0
    target_proto = torch.where(use_loo.unsqueeze(1), loo_proto, target_global_proto)

    target_scores = torch.sum(query * target_proto, dim=1) / temperature
    scores = scores.scatter(1, labels.unsqueeze(1), target_scores.unsqueeze(1))

    target_mask = F.one_hot(labels, num_classes=prototypes.shape[0]).bool()
    negative_scores = scores.masked_fill(target_mask, float('-inf'))
    negative_logsumexp = torch.logsumexp(negative_scores, dim=1)
    return target_scores - negative_logsumexp


@torch.no_grad()
def _rd_build_teacher_prototype_bank(
    args,
    old_model,
    train_data_set,
    exemplar_set,
    num_old_classes: int,
    num_seen_classes: int,
):
    """
    Build frozen teacher prototype banks once per incremental step.

    Guided audio/visual prototypes are used by CMR. A separate audio-independent
    visual bank is optionally built for teacher-trust estimation.
    """
    old_model.eval()

    audio_sums = None
    visual_sums = None
    trust_visual_sums = None
    counts = torch.zeros(num_seen_classes, dtype=torch.float32, device=device)

    sources = [
        ('memory', exemplar_set, args.exemplar_batch_size),
        ('current', train_data_set, args.train_batch_size),
    ]

    for source_name, dataset, requested_batch_size in sources:
        dataset_len = len(dataset)
        if dataset_len <= 0:
            continue

        loader = DataLoader(
            dataset,
            batch_size=min(requested_batch_size, dataset_len),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        for data, labels in tqdm(loader, desc='RDv2 prototypes ({})'.format(source_name), leave=False):
            labels = labels.to(device, non_blocking=True).long()
            visual = data[0].to(device, non_blocking=True)
            audio = data[1].to(device, non_blocking=True)

            if torch.any(labels < 0) or torch.any(labels >= num_seen_classes):
                raise RuntimeError(
                    'Prototype labels must lie in [0, {}). Got min={}, max={}.'.format(
                        num_seen_classes, int(labels.min().item()), int(labels.max().item())
                    )
                )
            if source_name == 'memory' and torch.any(labels >= num_old_classes):
                raise RuntimeError(
                    'exemplar_set unexpectedly contains a current/new class label; '
                    'expected labels < {}.'.format(num_old_classes)
                )
            if source_name == 'current' and torch.any(labels < num_old_classes):
                raise RuntimeError(
                    'train_data_set unexpectedly contains an old class label; '
                    'expected labels >= {}.'.format(num_old_classes)
                )

            audio_feature, guided_visual, trust_visual = _rd_extract_teacher_features(
                args, old_model, visual, audio
            )

            if audio_sums is None:
                dim = audio_feature.shape[1]
                audio_sums = torch.zeros(num_seen_classes, dim, dtype=audio_feature.dtype, device=device)
                visual_sums = torch.zeros(num_seen_classes, dim, dtype=guided_visual.dtype, device=device)
                trust_visual_sums = torch.zeros(
                    num_seen_classes, dim, dtype=trust_visual.dtype, device=device
                )

            audio_sums.index_add_(0, labels, audio_feature)
            visual_sums.index_add_(0, labels, guided_visual)
            trust_visual_sums.index_add_(0, labels, trust_visual)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    if audio_sums is None:
        raise RuntimeError('Could not build RD-CrossSDC v2 prototype bank')

    missing = torch.nonzero(counts <= 0, as_tuple=False).flatten()
    if missing.numel() > 0:
        raise RuntimeError(
            'RD-CrossSDC prototype bank is missing seen classes: {}'.format(
                missing.detach().cpu().tolist()
            )
        )

    audio_means = audio_sums / counts.unsqueeze(1)
    visual_means = visual_sums / counts.unsqueeze(1)
    trust_visual_means = trust_visual_sums / counts.unsqueeze(1)

    return {
        'audio_prototypes': F.normalize(audio_means, dim=1, eps=1e-12).detach(),
        'visual_prototypes': F.normalize(visual_means, dim=1, eps=1e-12).detach(),
        'trust_visual_prototypes': F.normalize(
            trust_visual_means, dim=1, eps=1e-12
        ).detach(),
        'audio_sums': audio_sums.detach(),
        'visual_sums': visual_sums.detach(),
        'trust_visual_sums': trust_visual_sums.detach(),
        'counts': counts.detach(),
    }


@torch.no_grad()
def _rd_compute_teacher_trust(
    args,
    old_model,
    exemplar_set,
    prototype_bank,
    num_old_classes: int,
):
    """
    Estimate static direction-specific teacher trust over old classes.

    Raw reliability is converted to above-chance trust and then shrunk toward the
    class mean when a class has few replay exemplars.
    """
    exemplar_len = len(exemplar_set)
    if exemplar_len <= 0:
        raise RuntimeError('RD-CrossSDC v2 requires replay exemplars after step 0')

    loader = DataLoader(
        exemplar_set,
        batch_size=min(args.exemplar_batch_size, exemplar_len),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    rel_sum_a_from_v = torch.zeros(num_old_classes, dtype=torch.float32, device=device)
    rel_sum_v_from_a = torch.zeros(num_old_classes, dtype=torch.float32, device=device)
    rel_count = torch.zeros(num_old_classes, dtype=torch.float32, device=device)

    old_audio_prototypes = prototype_bank['audio_prototypes'][:num_old_classes]
    old_audio_sums = prototype_bank['audio_sums'][:num_old_classes]
    old_trust_visual_prototypes = prototype_bank['trust_visual_prototypes'][:num_old_classes]
    old_trust_visual_sums = prototype_bank['trust_visual_sums'][:num_old_classes]
    old_counts = prototype_bank['counts'][:num_old_classes]

    for data, labels in tqdm(loader, desc='RDv2 teacher trust', leave=False):
        labels = labels.to(device, non_blocking=True).long()
        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)

        if torch.any(labels < 0) or torch.any(labels >= num_old_classes):
            raise RuntimeError('Teacher-trust estimation received a non-old label')

        old_audio, _, trust_visual = _rd_extract_teacher_features(
            args, old_model, visual, audio
        )

        margin_a_from_v = _rd_cross_modal_margin(
            query=old_audio,
            labels=labels,
            prototypes=old_trust_visual_prototypes,
            prototype_sums=old_trust_visual_sums,
            prototype_counts=old_counts,
            positive_teacher_features=trust_visual,
            temperature=args.rd_margin_temperature,
        )
        margin_v_from_a = _rd_cross_modal_margin(
            query=trust_visual,
            labels=labels,
            prototypes=old_audio_prototypes,
            prototype_sums=old_audio_sums,
            prototype_counts=old_counts,
            positive_teacher_features=old_audio,
            temperature=args.rd_margin_temperature,
        )

        rel_sum_a_from_v.index_add_(0, labels, torch.sigmoid(margin_a_from_v).float())
        rel_sum_v_from_a.index_add_(0, labels, torch.sigmoid(margin_v_from_a).float())
        rel_count.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    missing = torch.nonzero(rel_count <= 0, as_tuple=False).flatten()
    if missing.numel() > 0:
        raise RuntimeError(
            'No replay samples for old classes: {}'.format(missing.detach().cpu().tolist())
        )

    reliability_a_from_v = rel_sum_a_from_v / rel_count
    reliability_v_from_a = rel_sum_v_from_a / rel_count

    chance = 1.0 / float(num_old_classes)
    chance_denominator = max(1.0 - chance, 1e-12)
    trust_a_from_v = ((reliability_a_from_v - chance) / chance_denominator).clamp(0.0, 1.0)
    trust_v_from_a = ((reliability_v_from_a - chance) / chance_denominator).clamp(0.0, 1.0)

    def shrink(trust):
        if args.rd_trust_shrink_beta <= 0:
            return trust
        coefficient = rel_count / (rel_count + args.rd_trust_shrink_beta)
        global_mean = trust.mean()
        return coefficient * trust + (1.0 - coefficient) * global_mean

    trust_a_from_v_shrunk = shrink(trust_a_from_v).detach()
    trust_v_from_a_shrunk = shrink(trust_v_from_a).detach()

    zeros = torch.zeros(num_old_classes, dtype=torch.float32, device=device)
    zero_steps = torch.zeros(num_old_classes, dtype=torch.long, device=device)

    state = {
        'reliability_a_from_v': reliability_a_from_v.detach(),
        'reliability_v_from_a': reliability_v_from_a.detach(),
        'chance_level': chance,
        'trust_a_from_v': trust_a_from_v.detach(),
        'trust_v_from_a': trust_v_from_a.detach(),
        'trust_a_from_v_shrunk': trust_a_from_v_shrunk,
        'trust_v_from_a_shrunk': trust_v_from_a_shrunk,
        'trust_weight_a_from_v': _rd_make_trust_only_weights(args, trust_a_from_v_shrunk),
        'trust_weight_v_from_a': _rd_make_trust_only_weights(args, trust_v_from_a_shrunk),
        'need_ema_a_from_v': zeros.clone(),
        'need_ema_v_from_a': zeros.clone(),
        'need_steps_a_from_v': zero_steps.clone(),
        'need_steps_v_from_a': zero_steps.clone(),
        'need_a_from_v': zeros.clone(),
        'need_v_from_a': zeros.clone(),
    }
    state['class_weight_a_from_v'] = _rd_make_trust_need_weights(
        args, trust_a_from_v_shrunk, state['need_a_from_v']
    )
    state['class_weight_v_from_a'] = _rd_make_trust_need_weights(
        args, trust_v_from_a_shrunk, state['need_v_from_a']
    )
    return state


@torch.no_grad()
def _rd_update_need_ema_from_epoch(
    args,
    rd_state,
    deficit_sum_a_from_v: torch.Tensor,
    deficit_sum_v_from_a: torch.Tensor,
    deficit_count: torch.Tensor,
):
    """
    Update detached per-class Need once per epoch.

    This is more stable than updating from sparse within-batch class counts and
    avoids an immediate feedback loop in which one batch changes the weights of
    the next batch in the same epoch.
    """
    present = deficit_count > 0
    if not present.any():
        return

    epoch_mean_a = torch.zeros_like(deficit_sum_a_from_v)
    epoch_mean_v = torch.zeros_like(deficit_sum_v_from_a)
    epoch_mean_a[present] = deficit_sum_a_from_v[present] / deficit_count[present]
    epoch_mean_v[present] = deficit_sum_v_from_a[present] / deficit_count[present]

    def update_direction(epoch_mean, ema_key, steps_key, need_key):
        momentum = args.rd_need_ema_momentum
        rd_state[ema_key][present] = (
            momentum * rd_state[ema_key][present]
            + (1.0 - momentum) * epoch_mean[present]
        )
        rd_state[steps_key][present] += 1
        steps = rd_state[steps_key][present].float()
        correction = 1.0 - torch.pow(torch.full_like(steps, momentum), steps)
        rd_state[need_key][present] = (
            rd_state[ema_key][present] / correction.clamp_min(1e-12)
        )

    update_direction(
        epoch_mean_a,
        'need_ema_a_from_v',
        'need_steps_a_from_v',
        'need_a_from_v',
    )
    update_direction(
        epoch_mean_v,
        'need_ema_v_from_a',
        'need_steps_v_from_a',
        'need_v_from_a',
    )

    rd_state['class_weight_a_from_v'] = _rd_make_trust_need_weights(
        args, rd_state['trust_a_from_v_shrunk'], rd_state['need_a_from_v']
    )
    rd_state['class_weight_v_from_a'] = _rd_make_trust_need_weights(
        args, rd_state['trust_v_from_a_shrunk'], rd_state['need_v_from_a']
    )


def _rd_crosssdc_instance(
    current_audio: torch.Tensor,
    current_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    temperature: float,
    orientation: str,
):
    """Uniform temporal CrossSDC-I. Legacy orientation reproduces the direct baseline."""
    batch_size = current_audio.shape[0]
    if batch_size <= 1:
        zero = (current_audio.sum() + current_visual.sum()) * 0.0
        return zero, {'loss_a_from_v': zero.detach(), 'loss_v_from_a': zero.detach()}

    targets = torch.arange(batch_size, device=current_audio.device)
    score_a_from_v = torch.mm(current_audio, old_visual.transpose(0, 1)) / temperature

    if orientation == 'legacy':
        score_v_from_a = torch.mm(old_audio, current_visual.transpose(0, 1)) / temperature
    elif orientation == 'current_query':
        score_v_from_a = torch.mm(current_visual, old_audio.transpose(0, 1)) / temperature
    else:
        raise ValueError('Unknown temporal orientation: {}'.format(orientation))

    loss_a_from_v = F.cross_entropy(score_a_from_v, targets)
    loss_v_from_a = F.cross_entropy(score_v_from_a, targets)
    return 0.5 * (loss_a_from_v + loss_v_from_a), {
        'loss_a_from_v': loss_a_from_v.detach(),
        'loss_v_from_a': loss_v_from_a.detach(),
    }


def _rd_per_anchor_class_loss(
    query: torch.Tensor,
    keys: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    normalization: str,
):
    """Per-anchor supervised cross-modal contrastive loss."""
    score = torch.mm(query, keys.transpose(0, 1)) / temperature
    class_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
    log_prob = F.log_softmax(score, dim=1)
    positive_count = class_mask.sum(dim=1)

    if normalization == 'legacy':
        # Original AVCIL class_contrastive_loss scaling (not the direct CrossSDC-C baseline).
        per_anchor = -torch.mean(log_prob * class_mask, dim=1)
    elif normalization == 'positive_mean':
        # Equation-style positive averaging used by the direct CrossSDC baseline.
        per_anchor = -torch.sum(log_prob * class_mask, dim=1) / positive_count.clamp_min(1.0)
    else:
        raise ValueError('Unknown class-loss normalization: {}'.format(normalization))

    return per_anchor, positive_count


def _rd_weighted_crosssdc_class(
    current_audio: torch.Tensor,
    current_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    weight_a_from_v: torch.Tensor,
    weight_v_from_a: torch.Tensor,
    temperature: float,
    orientation: str,
    normalization: str,
):
    """Persistent Trust x Need weighted CrossSDC-C on replay exemplars."""
    if current_audio.shape[0] <= 1:
        zero = (current_audio.sum() + current_visual.sum()) * 0.0
        return zero, {
            'loss_a_from_v': zero.detach(),
            'loss_v_from_a': zero.detach(),
            'mean_positive_count': zero.detach(),
            'singleton_ratio': zero.detach(),
        }

    per_a, positive_count = _rd_per_anchor_class_loss(
        current_audio, old_visual, labels, temperature, normalization
    )
    if orientation == 'legacy':
        per_v, _ = _rd_per_anchor_class_loss(
            old_audio, current_visual, labels, temperature, normalization
        )
    elif orientation == 'current_query':
        per_v, _ = _rd_per_anchor_class_loss(
            current_visual, old_audio, labels, temperature, normalization
        )
    else:
        raise ValueError('Unknown temporal orientation: {}'.format(orientation))

    sample_weight_a = weight_a_from_v.index_select(0, labels).detach()
    sample_weight_v = weight_v_from_a.index_select(0, labels).detach()
    loss_a = _rd_weighted_mean(per_a, sample_weight_a)
    loss_v = _rd_weighted_mean(per_v, sample_weight_v)

    return 0.5 * (loss_a + loss_v), {
        'loss_a_from_v': loss_a.detach(),
        'loss_v_from_a': loss_v.detach(),
        'mean_positive_count': positive_count.float().mean().detach(),
        'singleton_ratio': (positive_count <= 1.0).float().mean().detach(),
    }


def _rd_compute_margin_terms(
    args,
    current_audio: torch.Tensor,
    current_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank,
    num_old_classes: int,
):
    """
    Compute reference/current margins, raw deficits, CMR violations, and
    new-class competition pressure.

    The selected rd_margin_label_space controls the loss. New-class pressure is
    always measured as B_ref(old-only) - B_ref(all-seen) for diagnostics.
    """
    num_seen_classes = int(prototype_bank['counts'].shape[0])

    def margins_for_limit(limit: int):
        audio_prototypes = prototype_bank['audio_prototypes'][:limit]
        visual_prototypes = prototype_bank['visual_prototypes'][:limit]
        audio_sums = prototype_bank['audio_sums'][:limit]
        visual_sums = prototype_bank['visual_sums'][:limit]
        counts = prototype_bank['counts'][:limit]

        ref_a = _rd_cross_modal_margin(
            old_audio, labels, visual_prototypes, visual_sums, counts,
            old_visual, args.rd_margin_temperature
        ).detach()
        cur_a = _rd_cross_modal_margin(
            current_audio, labels, visual_prototypes, visual_sums, counts,
            old_visual, args.rd_margin_temperature
        )
        ref_v = _rd_cross_modal_margin(
            old_visual, labels, audio_prototypes, audio_sums, counts,
            old_audio, args.rd_margin_temperature
        ).detach()
        cur_v = _rd_cross_modal_margin(
            current_visual, labels, audio_prototypes, audio_sums, counts,
            old_audio, args.rd_margin_temperature
        )
        return ref_a, cur_a, ref_v, cur_v

    if args.rd_margin_label_space == 'old_only':
        selected_limit = num_old_classes
    elif args.rd_margin_label_space == 'all_seen':
        selected_limit = num_seen_classes
    else:
        raise ValueError('Unknown margin label space: {}'.format(args.rd_margin_label_space))

    ref_a, cur_a, ref_v, cur_v = margins_for_limit(selected_limit)

    if selected_limit == num_old_classes:
        ref_old_a, ref_old_v = ref_a, ref_v
    else:
        ref_old_a, _, ref_old_v, _ = margins_for_limit(num_old_classes)

    if selected_limit == num_seen_classes:
        ref_all_a, ref_all_v = ref_a, ref_v
    else:
        ref_all_a, _, ref_all_v, _ = margins_for_limit(num_seen_classes)

    signed_deficit_a = ref_a - cur_a
    signed_deficit_v = ref_v - cur_v
    deficit_a = F.relu(signed_deficit_a)
    deficit_v = F.relu(signed_deficit_v)
    violation_a = F.relu(signed_deficit_a - args.rd_margin_tolerance)
    violation_v = F.relu(signed_deficit_v - args.rd_margin_tolerance)

    return {
        'ref_a_from_v': ref_a,
        'cur_a_from_v': cur_a,
        'ref_v_from_a': ref_v,
        'cur_v_from_a': cur_v,
        'deficit_a_from_v': deficit_a,
        'deficit_v_from_a': deficit_v,
        'violation_a_from_v': violation_a,
        'violation_v_from_a': violation_v,
        'new_pressure_a_from_v': (ref_old_a - ref_all_a).detach(),
        'new_pressure_v_from_a': (ref_old_v - ref_all_v).detach(),
    }

def _rd_margin_retention_from_terms(
    terms,
    labels: torch.Tensor,
    trust_weight_a_from_v: torch.Tensor,
    trust_weight_v_from_a: torch.Tensor,
):
    """Trust-weighted one-sided CMR guardrail."""
    sample_weight_a = trust_weight_a_from_v.index_select(0, labels).detach()
    sample_weight_v = trust_weight_v_from_a.index_select(0, labels).detach()
    loss_a = _rd_weighted_mean(terms['violation_a_from_v'], sample_weight_a)
    loss_v = _rd_weighted_mean(terms['violation_v_from_a'], sample_weight_v)
    return 0.5 * (loss_a + loss_v), {
        'loss_a_from_v': loss_a.detach(),
        'loss_v_from_a': loss_v.detach(),
        'active_a_from_v': (terms['violation_a_from_v'] > 0).float().mean().detach(),
        'active_v_from_a': (terms['violation_v_from_a'] > 0).float().mean().detach(),
        'mean_deficit_a_from_v': terms['deficit_a_from_v'].mean().detach(),
        'mean_deficit_v_from_a': terms['deficit_v_from_a'].mean().detach(),
        'mean_new_pressure_a_from_v': terms['new_pressure_a_from_v'].mean().detach(),
        'mean_new_pressure_v_from_a': terms['new_pressure_v_from_a'].mean().detach(),
    }


def _rd_save_static_statistics(
    args,
    step,
    num_old_classes,
    prototype_bank,
    rd_state,
    metrics_root,
    id_to_category,
):
    out_dir = os.path.join(metrics_root, 'rd_crosssdc_v2')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'step_{}_static_trust.csv'.format(step))
    header = [
        'step', 'class_id', 'category_name', 'prototype_count', 'chance_level',
        'reliability_a_from_v', 'reliability_v_from_a',
        'trust_a_from_v', 'trust_v_from_a',
        'trust_a_from_v_shrunk', 'trust_v_from_a_shrunk',
        'initial_class_weight_a_from_v', 'initial_class_weight_v_from_a',
        'cmr_trust_weight_a_from_v', 'cmr_trust_weight_v_from_a',
    ]
    rows = []
    for c in range(num_old_classes):
        rows.append({
            'step': step,
            'class_id': c,
            'category_name': id_to_category.get(c, 'class_{}'.format(c)),
            'prototype_count': int(prototype_bank['counts'][c].item()),
            'chance_level': float(rd_state['chance_level']),
            'reliability_a_from_v': float(rd_state['reliability_a_from_v'][c].item()),
            'reliability_v_from_a': float(rd_state['reliability_v_from_a'][c].item()),
            'trust_a_from_v': float(rd_state['trust_a_from_v'][c].item()),
            'trust_v_from_a': float(rd_state['trust_v_from_a'][c].item()),
            'trust_a_from_v_shrunk': float(rd_state['trust_a_from_v_shrunk'][c].item()),
            'trust_v_from_a_shrunk': float(rd_state['trust_v_from_a_shrunk'][c].item()),
            'initial_class_weight_a_from_v': float(rd_state['class_weight_a_from_v'][c].item()),
            'initial_class_weight_v_from_a': float(rd_state['class_weight_v_from_a'][c].item()),
            'cmr_trust_weight_a_from_v': float(rd_state['trust_weight_a_from_v'][c].item()),
            'cmr_trust_weight_v_from_a': float(rd_state['trust_weight_v_from_a'][c].item()),
        })
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    config = {
        'step': step,
        'rd_lambda_i': args.rd_lambda_i,
        'rd_lambda_c': args.rd_lambda_c,
        'rd_lambda_m': args.rd_lambda_m,
        'rd_instance_temperature': args.rd_instance_temperature,
        'rd_class_temperature': args.rd_class_temperature,
        'rd_margin_temperature': args.rd_margin_temperature,
        'rd_temporal_orientation': args.rd_temporal_orientation,
        'rd_class_loss_normalization': args.rd_class_loss_normalization,
        'rd_margin_label_space': args.rd_margin_label_space,
        'rd_trust_visual_space': args.rd_trust_visual_space,
        'rd_class_weight_alpha': args.rd_class_weight_alpha,
        'rd_trust_gamma': args.rd_trust_gamma,
        'rd_need_eta': args.rd_need_eta,
        'rd_need_delta': args.rd_need_delta,
        'rd_need_ema_momentum': args.rd_need_ema_momentum,
        'rd_trust_offset': args.rd_trust_offset,
        'rd_trust_shrink_beta': args.rd_trust_shrink_beta,
        'rd_weight_min': args.rd_weight_min,
        'rd_weight_max': args.rd_weight_max,
        'rd_margin_tolerance': args.rd_margin_tolerance,
        'leave_one_out_positive_prototype': True,
    }
    save_json(config, os.path.join(out_dir, 'step_{}_config.json'.format(step)))


def _rd_save_dynamic_snapshot(
    step,
    epoch,
    num_old_classes,
    rd_state,
    metrics_root,
    id_to_category,
):
    out_dir = os.path.join(metrics_root, 'rd_crosssdc_v2')
    csv_path = os.path.join(out_dir, 'step_{}_dynamic_weights.csv'.format(step))
    header = [
        'step', 'epoch', 'class_id', 'category_name',
        'trust_a_from_v_shrunk', 'trust_v_from_a_shrunk',
        'need_a_from_v', 'need_v_from_a',
        'class_weight_a_from_v', 'class_weight_v_from_a',
    ]
    rows = []
    for c in range(num_old_classes):
        rows.append({
            'step': step,
            'epoch': epoch,
            'class_id': c,
            'category_name': id_to_category.get(c, 'class_{}'.format(c)),
            'trust_a_from_v_shrunk': float(rd_state['trust_a_from_v_shrunk'][c].item()),
            'trust_v_from_a_shrunk': float(rd_state['trust_v_from_a_shrunk'][c].item()),
            'need_a_from_v': float(rd_state['need_a_from_v'][c].item()),
            'need_v_from_a': float(rd_state['need_v_from_a'][c].item()),
            'class_weight_a_from_v': float(rd_state['class_weight_a_from_v'][c].item()),
            'class_weight_v_from_a': float(rd_state['class_weight_v_from_a'][c].item()),
        })
    append_csv_rows(csv_path, header, rows)


def _rd_append_epoch_summary(metrics_root, row):
    out_dir = os.path.join(metrics_root, 'rd_crosssdc_v2')
    csv_path = os.path.join(out_dir, 'epoch_summary.csv')
    header = [
        'step', 'epoch', 'train_loss', 'val_acc', 'val_old_acc', 'val_new_acc',
        'cross_i_raw', 'cross_i_a_from_v', 'cross_i_v_from_a',
        'cross_c_raw', 'cross_c_a_from_v', 'cross_c_v_from_a',
        'cmr_raw', 'cmr_a_from_v', 'cmr_v_from_a',
        'cross_i_weighted', 'cross_c_weighted', 'cmr_weighted',
        'active_a_from_v', 'active_v_from_a',
        'mean_deficit_a_from_v', 'mean_deficit_v_from_a',
        'mean_new_pressure_a_from_v', 'mean_new_pressure_v_from_a',
        'mean_need_a_from_v', 'mean_need_v_from_a',
        'weight_min_a_from_v', 'weight_max_a_from_v',
        'weight_min_v_from_a', 'weight_max_v_from_a',
        'mean_positive_count', 'singleton_anchor_ratio',
    ]
    append_csv_rows(csv_path, header, [row])


def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups: 
            param_group['lr'] = new_lr


def safe_div(a, b):
    return a / b if b > 0 else 0.0

def compute_per_class_prf(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    """
    y_true/y_pred: (N,) long
    returns dict of numpy arrays: tp, fp, fn, support, precision, recall, f1
    """
    y_true = y_true.long()
    y_pred = y_pred.long()

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    # 向量化实现：逐类统计（对 C<=500 量级足够快且清晰）
    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        tp[c] = (true_c & pred_c).sum()
        fp[c] = ((~true_c) & pred_c).sum()
        fn[c] = (true_c & (~pred_c)).sum()

    support = tp + fn

    precision = torch.zeros(num_classes, dtype=torch.float32)
    recall    = torch.zeros(num_classes, dtype=torch.float32)
    f1        = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        tp_c = tp[c].item()
        fp_c = fp[c].item()
        fn_c = fn[c].item()
        p = safe_div(tp_c, tp_c + fp_c)
        r = safe_div(tp_c, tp_c + fn_c)
        precision[c] = p
        recall[c] = r
        f1[c] = safe_div(2 * p * r, p + r)

    return {
        "tp": tp.cpu().numpy(),
        "fp": fp.cpu().numpy(),
        "fn": fn.cpu().numpy(),
        "support": support.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
    }

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_csv_rows(csv_path, header, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def train(args, step, train_data_set, val_data_set, exemplar_set,
          metrics_root=None, id_to_category=None):
    T = 2

    train_loader = DataLoader(
        train_data_set,
        batch_size=min(args.train_batch_size, train_data_set.__len__()),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data_set,
        batch_size=min(args.infer_batch_size, val_data_set.__len__()),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    step_out_class_num = (step + 1) * args.class_num_per_step
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
        old_model = None
        exemplar_loader = None
        last_step_out_class_num = 0
    else:
        model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.output_name, step - 1))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.output_name, step - 1))

        exemplar_loader = DataLoader(
            exemplar_set,
            batch_size=min(args.exemplar_batch_size, exemplar_set.__len__()),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        last_step_out_class_num = step * args.class_num_per_step

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)

    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        old_model.eval()
        for parameter in old_model.parameters():
            parameter.requires_grad_(False)

    rd_prototype_bank = None
    rd_state = None
    if step > 0 and args.rd_crosssdc_v2:
        print('Building RD-CrossSDC v2 teacher banks...', flush=True)
        # Extra prototype/trust DataLoaders must not change the RNG sequence used
        # by the real shuffled training loaders. This makes alpha=0, lambda_m=0
        # a strict implementation control against direct CrossSDC.
        rd_rng_state = _rd_capture_rng_state()
        try:
            rd_prototype_bank = _rd_build_teacher_prototype_bank(
                args=args,
                old_model=old_model,
                train_data_set=train_data_set,
                exemplar_set=exemplar_set,
                num_old_classes=last_step_out_class_num,
                num_seen_classes=step_out_class_num,
            )
            rd_state = _rd_compute_teacher_trust(
                args=args,
                old_model=old_model,
                exemplar_set=exemplar_set,
                prototype_bank=rd_prototype_bank,
                num_old_classes=last_step_out_class_num,
            )
        finally:
            _rd_restore_rng_state(rd_rng_state)
        if metrics_root is not None:
            _rd_save_static_statistics(
                args=args,
                step=step,
                num_old_classes=last_step_out_class_num,
                prototype_bank=rd_prototype_bank,
                rd_state=rd_state,
                metrics_root=metrics_root,
                id_to_category=id_to_category or {},
            )
        print(
            'Initial RDv2 class weights A<-V min/mean/max: {:.4f}/{:.4f}/{:.4f}'.format(
                rd_state['class_weight_a_from_v'].min().item(),
                rd_state['class_weight_a_from_v'].mean().item(),
                rd_state['class_weight_a_from_v'].max().item(),
            ),
            flush=True,
        )
        print(
            'Initial RDv2 class weights V<-A min/mean/max: {:.4f}/{:.4f}/{:.4f}'.format(
                rd_state['class_weight_v_from_a'].min().item(),
                rd_state['class_weight_v_from_a'].mean().item(),
                rd_state['class_weight_v_from_a'].max().item(),
            ),
            flush=True,
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()

        rd_epoch_cross_i = 0.0
        rd_epoch_cross_i_a = 0.0
        rd_epoch_cross_i_v = 0.0
        rd_epoch_cross_c = 0.0
        rd_epoch_cross_c_a = 0.0
        rd_epoch_cross_c_v = 0.0
        rd_epoch_margin = 0.0
        rd_epoch_margin_a = 0.0
        rd_epoch_margin_v = 0.0
        rd_epoch_active_a = 0.0
        rd_epoch_active_v = 0.0
        rd_epoch_deficit_a = 0.0
        rd_epoch_deficit_v = 0.0
        rd_epoch_new_pressure_a = 0.0
        rd_epoch_new_pressure_v = 0.0
        rd_epoch_positive_count = 0.0
        rd_epoch_singleton_ratio = 0.0

        if step > 0 and args.rd_crosssdc_v2:
            rd_deficit_sum_a = torch.zeros(last_step_out_class_num, device=device)
            rd_deficit_sum_v = torch.zeros(last_step_out_class_num, device=device)
            rd_deficit_count = torch.zeros(last_step_out_class_num, device=device)

        if step == 0:
            iterator = tqdm(train_loader)
        else:
            iterator = tzip(train_loader, cycle(exemplar_loader))

        for samples in iterator:
            if step == 0:
                data, labels = samples
                labels = labels.to(device)
                visual = data[0].to(device)
                audio = data[1].to(device)
                out, audio_feature, visual_feature = model(
                    visual=visual,
                    audio=audio,
                    out_feature_before_fusion=True,
                )
                loss = CE_loss(step_out_class_num, out, labels)
            else:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(device)
                labels_ = (labels % args.class_num_per_step).to(device)

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device).long()

                data_batch_size = labels_.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                total_visual = torch.cat((data[0], exemplar_data[0])).to(device)
                total_audio = torch.cat((data[1], exemplar_data[1])).to(device)

                current_outputs = model(
                    visual=total_visual,
                    audio=total_audio,
                    out_feature_before_fusion=True,
                    out_attn_score=True,
                    return_dict=True,
                )
                out = current_outputs['logits']
                audio_feature = current_outputs['audio_feature']
                visual_feature = current_outputs['visual_feature']
                spatial_attn_score = current_outputs['spatial_attn_score']
                temporal_attn_score = current_outputs['temporal_attn_score']

                with torch.no_grad():
                    old_outputs = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        out_feature_before_fusion=True,
                        out_attn_score=True,
                        return_dict=True,
                    )
                    old_out = old_outputs['logits'].detach()
                    old_audio_feature = old_outputs['audio_feature'].detach()
                    old_visual_feature = old_outputs['visual_feature'].detach()
                    old_spatial_attn_score = old_outputs['spatial_attn_score'].detach()
                    old_temporal_attn_score = old_outputs['temporal_attn_score'].detach()

                if args.instance_contrastive:
                    instance_contra_loss = cal_contrastive_loss(
                        audio_feature,
                        visual_feature,
                        temperature=args.instance_contrastive_temperature,
                    )

                if args.class_contrastive:
                    all_labels = torch.cat((labels, exemplar_labels))
                    class_contra_loss = class_contrastive_loss(
                        audio_feature,
                        visual_feature,
                        all_labels,
                        temperature=args.class_contrastive_temperature,
                    )

                if args.attn_score_distil:
                    exem_spatial_attn_score = spatial_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(2, 3)
                    exem_spatial_attn_score = exem_spatial_attn_score.reshape(
                        -1, exem_spatial_attn_score.shape[-1]
                    )
                    exem_old_spatial_attn_score = old_spatial_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(2, 3)
                    exem_old_spatial_attn_score = exem_old_spatial_attn_score.reshape(
                        -1, exem_old_spatial_attn_score.shape[-1]
                    )
                    exem_temporal_attn_score = temporal_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(1, 2)
                    exem_temporal_attn_score = exem_temporal_attn_score.reshape(
                        -1, exem_temporal_attn_score.shape[-1]
                    )
                    exem_old_temporal_attn_score = old_temporal_attn_score[
                        data_batch_size:data_batch_size + exemplar_data_batch_size
                    ].transpose(1, 2)
                    exem_old_temporal_attn_score = exem_old_temporal_attn_score.reshape(
                        -1, exem_old_temporal_attn_score.shape[-1]
                    )

                    spatial_attn_dist_loss = F.kl_div(
                        exem_spatial_attn_score.log(),
                        exem_old_spatial_attn_score,
                        reduction='sum',
                    ) / exemplar_data_batch_size
                    temporal_attn_dist_loss = F.kl_div(
                        exem_temporal_attn_score.log(),
                        exem_old_temporal_attn_score,
                        reduction='sum',
                    ) / exemplar_data_batch_size

                old_out = old_out[:, :last_step_out_class_num]
                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = CE_loss(args.class_num_per_step, curr_out, labels_)
                prev_out = out[
                    data_batch_size:data_batch_size + exemplar_data_batch_size,
                    :last_step_out_class_num,
                ]
                loss_prev = CE_loss(last_step_out_class_num, prev_out, exemplar_labels)
                loss_CE = (
                    loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size
                ) / (data_batch_size + exemplar_data_batch_size)

                if args.dataset == 'AVE' and args.class_num_per_step == 4 and step == 1:
                    loss_CE = CE_loss(
                        args.class_num_per_step + last_step_out_class_num,
                        out,
                        torch.cat((labels, exemplar_labels)),
                    )

                loss_KD = torch.zeros(step).to(device)
                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step
                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(
                        output_log, soft_target, reduction='batchmean'
                    ) * (T ** 2)
                loss_KD = loss_KD.sum()
                loss = loss_CE + loss_KD

                if args.instance_contrastive:
                    loss += args.lam_I * instance_contra_loss
                if args.class_contrastive:
                    loss += args.lam_C * class_contra_loss
                if args.attn_score_distil:
                    loss += (
                        args.lam * spatial_attn_dist_loss
                        + (1.0 - args.lam) * temporal_attn_dist_loss
                    )

                if args.rd_crosssdc_v2:
                    exemplar_slice = slice(
                        data_batch_size, data_batch_size + exemplar_data_batch_size
                    )
                    current_exemplar_audio = audio_feature[exemplar_slice]
                    current_exemplar_visual = visual_feature[exemplar_slice]
                    old_exemplar_audio = old_audio_feature[exemplar_slice]
                    old_exemplar_visual = old_visual_feature[exemplar_slice]

                    # Class weights are detached and fixed throughout this epoch.
                    # Current deficits are accumulated and update Need only after the epoch.
                    cross_i_loss, cross_i_stats = _rd_crosssdc_instance(
                        current_audio=current_exemplar_audio,
                        current_visual=current_exemplar_visual,
                        old_audio=old_exemplar_audio,
                        old_visual=old_exemplar_visual,
                        temperature=args.rd_instance_temperature,
                        orientation=args.rd_temporal_orientation,
                    )
                    cross_c_loss, cross_c_stats = _rd_weighted_crosssdc_class(
                        current_audio=current_exemplar_audio,
                        current_visual=current_exemplar_visual,
                        old_audio=old_exemplar_audio,
                        old_visual=old_exemplar_visual,
                        labels=exemplar_labels,
                        weight_a_from_v=rd_state['class_weight_a_from_v'],
                        weight_v_from_a=rd_state['class_weight_v_from_a'],
                        temperature=args.rd_class_temperature,
                        orientation=args.rd_temporal_orientation,
                        normalization=args.rd_class_loss_normalization,
                    )
                    margin_terms = _rd_compute_margin_terms(
                        args=args,
                        current_audio=current_exemplar_audio,
                        current_visual=current_exemplar_visual,
                        old_audio=old_exemplar_audio,
                        old_visual=old_exemplar_visual,
                        labels=exemplar_labels,
                        prototype_bank=rd_prototype_bank,
                        num_old_classes=last_step_out_class_num,
                    )
                    margin_loss, margin_stats = _rd_margin_retention_from_terms(
                        terms=margin_terms,
                        labels=exemplar_labels,
                        trust_weight_a_from_v=rd_state['trust_weight_a_from_v'],
                        trust_weight_v_from_a=rd_state['trust_weight_v_from_a'],
                    )

                    loss += args.rd_lambda_i * cross_i_loss
                    loss += args.rd_lambda_c * cross_c_loss
                    loss += args.rd_lambda_m * margin_loss

                    rd_epoch_cross_i += float(cross_i_loss.detach().item())
                    rd_epoch_cross_i_a += float(cross_i_stats['loss_a_from_v'].item())
                    rd_epoch_cross_i_v += float(cross_i_stats['loss_v_from_a'].item())
                    rd_epoch_cross_c += float(cross_c_loss.detach().item())
                    rd_epoch_cross_c_a += float(cross_c_stats['loss_a_from_v'].item())
                    rd_epoch_cross_c_v += float(cross_c_stats['loss_v_from_a'].item())
                    rd_epoch_margin += float(margin_loss.detach().item())
                    rd_epoch_margin_a += float(margin_stats['loss_a_from_v'].item())
                    rd_epoch_margin_v += float(margin_stats['loss_v_from_a'].item())
                    rd_epoch_active_a += float(margin_stats['active_a_from_v'].item())
                    rd_epoch_active_v += float(margin_stats['active_v_from_a'].item())
                    rd_epoch_deficit_a += float(margin_stats['mean_deficit_a_from_v'].item())
                    rd_epoch_deficit_v += float(margin_stats['mean_deficit_v_from_a'].item())
                    rd_epoch_new_pressure_a += float(
                        margin_stats['mean_new_pressure_a_from_v'].item()
                    )
                    rd_epoch_new_pressure_v += float(
                        margin_stats['mean_new_pressure_v_from_a'].item()
                    )
                    rd_epoch_positive_count += float(
                        cross_c_stats['mean_positive_count'].item()
                    )
                    rd_epoch_singleton_ratio += float(
                        cross_c_stats['singleton_ratio'].item()
                    )

                    rd_deficit_sum_a.index_add_(
                        0, exemplar_labels, margin_terms['deficit_a_from_v'].detach().float()
                    )
                    rd_deficit_sum_v.index_add_(
                        0, exemplar_labels, margin_terms['deficit_v_from_a'].detach().float()
                    )
                    rd_deficit_count.index_add_(
                        0,
                        exemplar_labels,
                        torch.ones_like(exemplar_labels, dtype=torch.float32),
                    )

            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1

        train_loss /= num_steps
        train_loss_list.append(train_loss)

        if step > 0 and args.rd_crosssdc_v2:
            _rd_update_need_ema_from_epoch(
                args=args,
                rd_state=rd_state,
                deficit_sum_a_from_v=rd_deficit_sum_a,
                deficit_sum_v_from_a=rd_deficit_sum_v,
                deficit_count=rd_deficit_count,
            )

            avg_cross_i = rd_epoch_cross_i / num_steps
            avg_cross_i_a = rd_epoch_cross_i_a / num_steps
            avg_cross_i_v = rd_epoch_cross_i_v / num_steps
            avg_cross_c = rd_epoch_cross_c / num_steps
            avg_cross_c_a = rd_epoch_cross_c_a / num_steps
            avg_cross_c_v = rd_epoch_cross_c_v / num_steps
            avg_margin = rd_epoch_margin / num_steps
            avg_margin_a = rd_epoch_margin_a / num_steps
            avg_margin_v = rd_epoch_margin_v / num_steps
            avg_active_a = rd_epoch_active_a / num_steps
            avg_active_v = rd_epoch_active_v / num_steps
            avg_deficit_a = rd_epoch_deficit_a / num_steps
            avg_deficit_v = rd_epoch_deficit_v / num_steps
            avg_new_pressure_a = rd_epoch_new_pressure_a / num_steps
            avg_new_pressure_v = rd_epoch_new_pressure_v / num_steps
            avg_positive_count = rd_epoch_positive_count / num_steps
            avg_singleton_ratio = rd_epoch_singleton_ratio / num_steps
            print(
                'Epoch:{} train_loss:{:.5f} '
                'RD-I:{:.5f}(x{:.3g}={:.5f}) '
                'RD-C:{:.5f}(x{:.3g}={:.5f}) '
                'RD-M:{:.5f}(x{:.3g}={:.5f}) '
                'active(A<-V/V<-A):{:.3f}/{:.3f} '
                'need:{:.4f}/{:.4f} singleton:{:.3f}'.format(
                    epoch,
                    train_loss,
                    avg_cross_i, args.rd_lambda_i, args.rd_lambda_i * avg_cross_i,
                    avg_cross_c, args.rd_lambda_c, args.rd_lambda_c * avg_cross_c,
                    avg_margin, args.rd_lambda_m, args.rd_lambda_m * avg_margin,
                    avg_active_a, avg_active_v,
                    rd_state['need_a_from_v'].mean().item(),
                    rd_state['need_v_from_a'].mean().item(),
                    avg_singleton_ratio,
                ),
                flush=True,
            )
        else:
            avg_cross_i = avg_cross_i_a = avg_cross_i_v = 0.0
            avg_cross_c = avg_cross_c_a = avg_cross_c_v = 0.0
            avg_margin = avg_margin_a = avg_margin_v = 0.0
            avg_active_a = avg_active_v = 0.0
            avg_deficit_a = avg_deficit_v = 0.0
            avg_new_pressure_a = avg_new_pressure_v = 0.0
            avg_positive_count = avg_singleton_ratio = 0.0
            print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0].to(device)
                val_audio = val_data[1].to(device)
                if torch.cuda.device_count() > 1:
                    val_out_logits = model.module.forward(visual=val_visual, audio=val_audio)
                else:
                    val_out_logits = model(visual=val_visual, audio=val_audio)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)

        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        val_pred = all_val_out_logits.argmax(dim=1)
        if step > 0:
            old_mask = all_val_labels < last_step_out_class_num
            new_mask = ~old_mask
            val_old_acc = (
                (val_pred[old_mask] == all_val_labels[old_mask]).float().mean().item()
                if old_mask.any() else 0.0
            )
            val_new_acc = (
                (val_pred[new_mask] == all_val_labels[new_mask]).float().mean().item()
                if new_mask.any() else 0.0
            )
            print(
                'Epoch:{} val_res:{:.6f} val_old:{:.6f} val_new:{:.6f}'.format(
                    epoch, val_top1, val_old_acc, val_new_acc
                ),
                flush=True,
            )
        else:
            val_old_acc = 0.0
            val_new_acc = val_top1
            print('Epoch:{} val_res:{:.6f}'.format(epoch, val_top1), flush=True)

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(
                    model.module,
                    './save/{}/step_{}_best_model.pkl'.format(args.output_name, step),
                )
            else:
                torch.save(
                    model,
                    './save/{}/step_{}_best_model.pkl'.format(args.output_name, step),
                )

        if step > 0 and args.rd_crosssdc_v2 and metrics_root is not None:
            _rd_append_epoch_summary(
                metrics_root,
                {
                    'step': step,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_acc': val_top1,
                    'val_old_acc': val_old_acc,
                    'val_new_acc': val_new_acc,
                    'cross_i_raw': avg_cross_i,
                    'cross_i_a_from_v': avg_cross_i_a,
                    'cross_i_v_from_a': avg_cross_i_v,
                    'cross_c_raw': avg_cross_c,
                    'cross_c_a_from_v': avg_cross_c_a,
                    'cross_c_v_from_a': avg_cross_c_v,
                    'cmr_raw': avg_margin,
                    'cmr_a_from_v': avg_margin_a,
                    'cmr_v_from_a': avg_margin_v,
                    'cross_i_weighted': args.rd_lambda_i * avg_cross_i,
                    'cross_c_weighted': args.rd_lambda_c * avg_cross_c,
                    'cmr_weighted': args.rd_lambda_m * avg_margin,
                    'active_a_from_v': avg_active_a,
                    'active_v_from_a': avg_active_v,
                    'mean_deficit_a_from_v': avg_deficit_a,
                    'mean_deficit_v_from_a': avg_deficit_v,
                    'mean_new_pressure_a_from_v': avg_new_pressure_a,
                    'mean_new_pressure_v_from_a': avg_new_pressure_v,
                    'mean_need_a_from_v': rd_state['need_a_from_v'].mean().item(),
                    'mean_need_v_from_a': rd_state['need_v_from_a'].mean().item(),
                    'weight_min_a_from_v': rd_state['class_weight_a_from_v'].min().item(),
                    'weight_max_a_from_v': rd_state['class_weight_a_from_v'].max().item(),
                    'weight_min_v_from_a': rd_state['class_weight_v_from_a'].min().item(),
                    'weight_max_v_from_a': rd_state['class_weight_v_from_a'].max().item(),
                    'mean_positive_count': avg_positive_count,
                    'singleton_anchor_ratio': avg_singleton_ratio,
                },
            )
            if epoch % args.rd_diag_interval == 0 or epoch == args.max_epoches - 1:
                _rd_save_dynamic_snapshot(
                    step=step,
                    epoch=epoch,
                    num_old_classes=last_step_out_class_num,
                    rd_state=rd_state,
                    metrics_root=metrics_root,
                    id_to_category=id_to_category or {},
                )

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/train_loss_step_{}.png'.format(args.output_name, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/val_acc_step_{}.png'.format(args.output_name, step))
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, opt, epoch)


# def detailed_test(args, step, test_data_set, task_best_acc_list):
#     print("=====================================")
#     print("Start testing...")
#     print("=====================================")

#     model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.output_name, step))
#     model.to(device)

#     test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
#                              pin_memory=True, drop_last=False, shuffle=False)
    
#     all_test_out_logits = torch.Tensor([])
#     all_test_labels = torch.Tensor([])
#     model.eval()
#     with torch.no_grad():
#         for test_data, test_labels in tqdm(test_loader):
#             test_visual = test_data[0]
#             test_audio = test_data[1]
#             test_visual = test_visual.to(device)
#             test_audio = test_audio.to(device)
#             test_out_logits = model(visual=test_visual, audio=test_audio)
#             test_out_logits = F.softmax(test_out_logits, dim=-1).detach().cpu()
#             all_test_out_logits = torch.cat((all_test_out_logits, test_out_logits), dim=0)
#             all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
#     test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
#     print("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
    
#     old_task_acc_list = []
#     for i in range(step+1):
#         step_class_list = range(i*args.class_num_per_step, (i+1)*args.class_num_per_step)
#         step_class_idxs = []
#         for c in step_class_list:
#             idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
#             step_class_idxs += idxs
#         step_class_idxs = np.array(step_class_idxs)
#         i_labels = torch.Tensor(all_test_labels.numpy()[step_class_idxs])
#         i_logits = torch.Tensor(all_test_out_logits.numpy()[step_class_idxs])
#         i_acc = top_1_acc(i_logits, i_labels)
#         if i == step:
#             curren_step_acc = i_acc
#         else:
#             old_task_acc_list.append(i_acc)
#     if step > 0:
#         forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
#         print('forgetting: {:.6f}'.format(forgetting))
#         for i in range(len(task_best_acc_list)):
#             task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
#     else:
#         forgetting = None
#     task_best_acc_list.append(curren_step_acc)

#     return forgetting


def detailed_test(args, step, test_data_set, task_best_acc_list,
                  metrics_root: str,
                  metrics_state: dict,
                  id_to_category: dict):
    """
    metrics_state:
      {
        "best_recall": { "0": 0.83, "1": 0.55, ... },
        "first_seen_step": { "0": 0, "1": 2, ... }
      }
    """
    print("=====================================")
    print("Start testing...")
    print("=====================================")

    model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.output_name, step))
    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)

    # ---- 收集 logits 与 labels（保持 dtype 正确）----
    all_logits_list = []
    all_labels_list = []

    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            test_visual = test_data[0].to(device)
            test_audio = test_data[1].to(device)

            logits = model(visual=test_visual, audio=test_audio)   # raw logits
            all_logits_list.append(logits.detach().cpu())
            all_labels_list.append(test_labels.detach().cpu().long())

    all_test_logits = torch.cat(all_logits_list, dim=0)            # (N, C_seen)
    all_test_labels = torch.cat(all_labels_list, dim=0).long()     # (N,)

    # ---- overall top-1 acc ----
    pred = all_test_logits.argmax(dim=1).long()
    overall_acc = (pred == all_test_labels).float().mean().item()
    print("Incremental step {} Testing res (overall acc): {:.6f}".format(step, overall_acc))

    # ---- per-task acc（与你原逻辑等价，但更快更稳）----
    K = args.class_num_per_step
    num_seen_classes = (step + 1) * K

    old_task_acc_list = []
    current_step_acc = None
    for i in range(step + 1):
        lo = i * K
        hi = (i + 1) * K
        mask = (all_test_labels >= lo) & (all_test_labels < hi)
        if mask.sum().item() == 0:
            i_acc = 0.0
        else:
            i_acc = (pred[mask] == all_test_labels[mask]).float().mean().item()

        if i == step:
            current_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)

    # ---- task-level forgetting（你的原定义：best_old_task - current_old_task）----
    if step > 0:
        forgetting = float(np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list)))
        print('task-level forgetting: {:.6f}'.format(forgetting))
        # 更新旧任务历史最好
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None

    # append 当前任务的“历史最好”（初始就是当前）
    task_best_acc_list.append(current_step_acc)

    # ======================================================================
    # Per-class metrics (CIL classic: evaluate only on seen classes)
    # ======================================================================
    stats = compute_per_class_prf(all_test_labels, pred, num_seen_classes)

    # state dicts
    best_f1 = metrics_state.get("best_f1", {})
    first_seen = metrics_state.get("first_seen_step", {})

    rows = []
    for c in range(num_seen_classes):
        c_str = str(c)

        # record first seen step (对 CIL shuffle 很有用)
        if c_str not in first_seen:
            first_seen[c_str] = step

        support_c = int(stats["support"][c])
        tp_c = int(stats["tp"][c])
        fp_c = int(stats["fp"][c])
        fn_c = int(stats["fn"][c])

        precision_c = float(stats["precision"][c])
        recall_c = float(stats["recall"][c])
        f1_c = float(stats["f1"][c])

        # per-class forgetting（基于 F1）
        # 约定：forget = best_before - current；若没有 best_before，则 forget=0
        best_before = float(best_f1[c_str]) if c_str in best_f1 else None
        forget_f1 = (best_before - f1_c) if best_before is not None else 0.0

        # update best_f1
        new_best = f1_c if best_before is None else max(best_before, f1_c)
        best_f1[c_str] = new_best

        forgetting_value = float(forgetting) if forgetting is not None else 0.0
        rows.append({
            "step": step,
            "class_id": c,
            "category_name": id_to_category.get(c, f"class_{c}"),
            "first_seen_step": int(first_seen[c_str]),
            "support": support_c,
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "best_f1": float(new_best),
            "forget_f1": float(forget_f1),  # 基于F1 的 forgetting
            "forgetting": forgetting_value,    # 基于 task-level acc 的 forgetting
            "overall_acc": float(overall_acc),          # 冗余存一下，做 step-level plot 更方便
        })

    header = [
        "step", "class_id", "category_name", "first_seen_step", "support",
        "tp", "fp", "fn", "precision", "recall", "f1",
        "best_f1", "forget_f1", "forgetting", "overall_acc"
    ]
    per_class_csv = os.path.join(metrics_root, "per_class_metrics.csv")
    append_csv_rows(per_class_csv, header, rows)

    # update state + dump
    metrics_state["best_f1"] = best_f1
    metrics_state["first_seen_step"] = first_seen
    save_json(metrics_state, os.path.join(metrics_root, "per_class_state.json"))

    return forgetting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def dataset_type(s: str):
        if s in ['AVE', 'ksounds']:
            return s
        if 'VGGSound' in s:
            return s
        raise argparse.ArgumentTypeError(
            "dataset must be 'AVE', 'ksounds', or contain 'VGGSound'"
        )
    parser.add_argument('--dataset', type=dataset_type, default='AVE')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Output/checkpoint directory name. Defaults to --dataset; does not affect data loading.')
    parser.add_argument('--modality', type=str, default='audio-visual', choices=['audio-visual'])
    parser.add_argument('--feature_root', type=str, default="/mnt/data2/wpian/dataset/VGGSound", help='Root dir for feature files: visual_features.h5, audio_pretrained_feature_dict.npy, etc.')
    parser.add_argument('--meta_root', type=str, default=None, help='Root dir for metadata dicts: all_id_category_dict.npy, category_encode_dict.npy, all_classId_vid_dict.npy, etc.')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=32)
    parser.add_argument('--exemplar_batch_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")
    
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--lam_I', type=float, default=0.5)
    parser.add_argument('--lam_C', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--class_num_per_step', type=int, default=7)

    parser.add_argument('--memory_size', type=int, default=340)

    parser.add_argument('--instance_contrastive', action='store_true', default=False)
    parser.add_argument('--class_contrastive', action='store_true', default=False)
    parser.add_argument('--attn_score_distil', action='store_true', default=False)

    # RD-CrossSDC v2: persistent CrossSDC-I + Trust x Need CrossSDC-C + CMR guardrail.
    parser.add_argument(
        '--rd_crosssdc_v2', '--rd_crosssdc',
        dest='rd_crosssdc_v2', action='store_true', default=False,
    )
    parser.add_argument('--rd_lambda_i', type=float, default=0.1,
                        help='Global coefficient for uniform temporal CrossSDC-I.')
    parser.add_argument('--rd_lambda_c', type=float, default=0.3,
                        help='Global coefficient for Trust x Need weighted CrossSDC-C.')
    parser.add_argument('--rd_lambda_m', type=float, default=0.1,
                        help='Global coefficient for the one-sided CMR guardrail.')
    parser.add_argument('--rd_instance_temperature', type=float, default=0.05)
    parser.add_argument('--rd_class_temperature', type=float, default=0.05)
    parser.add_argument('--rd_margin_temperature', type=float, default=0.1)

    parser.add_argument('--rd_class_weight_alpha', type=float, default=0.5,
                        help='0=uniform CrossSDC-C; 1=fully Trust x Need relative weighting.')
    parser.add_argument('--rd_trust_gamma', type=float, default=1.0)
    parser.add_argument('--rd_need_eta', type=float, default=0.5)
    parser.add_argument('--rd_need_delta', type=float, default=0.05)
    parser.add_argument('--rd_need_ema_momentum', type=float, default=0.9)
    parser.add_argument('--rd_trust_offset', type=float, default=0.05,
                        help='Small floor after chance correction; prevents zeroing a class.')
    parser.add_argument('--rd_trust_shrink_beta', type=float, default=10.0,
                        help='Count-aware shrinkage strength toward mean teacher trust.')
    parser.add_argument('--rd_weight_min', type=float, default=0.5)
    parser.add_argument('--rd_weight_max', type=float, default=2.0)
    parser.add_argument('--rd_margin_tolerance', type=float, default=0.0)

    parser.add_argument('--rd_temporal_orientation', type=str, default='legacy',
                        choices=['legacy', 'current_query'],
                        help='legacy reproduces direct CrossSDC: curA->oldV and oldA->curV.')
    parser.add_argument('--rd_class_loss_normalization', type=str, default='positive_mean',
                        choices=['legacy', 'positive_mean'],
                        help='positive_mean implements the stated CrossSDC-C formula and matches the direct CrossSDC loss scale.')
    parser.add_argument('--rd_margin_label_space', type=str, default='old_only',
                        choices=['old_only', 'all_seen'])
    parser.add_argument('--rd_trust_visual_space', type=str, default='guided',
                        choices=['guided', 'uniform'],
                        help='guided is the conservative default; uniform removes audio-guided attention from trust estimation.')
    parser.add_argument('--rd_diag_interval', type=int, default=10)

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.1)

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dump_tsne", action="store_true", help="If set, dump t-SNE plots for each step")
    parser.add_argument("--tsne_feature", type=str, default="logits",
                        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"])
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    

    args = parser.parse_args()

    if not (0.0 <= args.rd_class_weight_alpha <= 1.0):
        parser.error('--rd_class_weight_alpha must lie in [0, 1]')
    if args.rd_instance_temperature <= 0 or args.rd_class_temperature <= 0 or args.rd_margin_temperature <= 0:
        parser.error('All RD-CrossSDC temperatures must be positive')
    if args.rd_lambda_i < 0 or args.rd_lambda_c < 0 or args.rd_lambda_m < 0:
        parser.error('RD-CrossSDC loss coefficients must be non-negative')
    if args.rd_trust_gamma < 0 or args.rd_need_eta < 0:
        parser.error('--rd_trust_gamma and --rd_need_eta must be non-negative')
    if args.rd_need_delta < 0 or args.rd_trust_offset < 0:
        parser.error('--rd_need_delta and --rd_trust_offset must be non-negative')
    if not (0.0 <= args.rd_need_ema_momentum < 1.0):
        parser.error('--rd_need_ema_momentum must lie in [0, 1)')
    if args.rd_trust_shrink_beta < 0:
        parser.error('--rd_trust_shrink_beta must be non-negative')
    if args.rd_weight_min <= 0 or args.rd_weight_max < args.rd_weight_min:
        parser.error('Require 0 < rd_weight_min <= rd_weight_max')
    if args.rd_margin_tolerance < 0:
        parser.error('--rd_margin_tolerance must be non-negative')
    if args.rd_diag_interval <= 0:
        parser.error('--rd_diag_interval must be positive')

    args.output_name = args.experiment_name if args.experiment_name else args.dataset

    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step
    setup_seed(args.seed)

    print('Training start time: {}'.format(datetime.now()))

    train_set = IcaAVELoader(args=args, mode='train', modality=args.modality)
    val_set = IcaAVELoader(args=args, mode='val', modality=args.modality)
    test_set = IcaAVELoader(args=args, mode='test', modality=args.modality)
    exemplar_set = exemplarLoader(args=args, modality=args.modality)

    category_encode_dict = train_set.category_encode_dict
    id_to_category = {v: k for k, v in category_encode_dict.items()}

    ckpts_root = './save/{}/'.format(args.output_name)
    figs_root = './save/fig/{}/'.format(args.output_name)

    # NEW: metrics root (paper-plot friendly outputs)
    metrics_root = './save/metrics/{}/'.format(args.output_name)

    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    if not os.path.exists(figs_root):
        os.makedirs(figs_root)
    if not os.path.exists(metrics_root):
        os.makedirs(metrics_root)

    per_class_csv = os.path.join(metrics_root, "per_class_metrics.csv")
    if os.path.exists(per_class_csv):
        os.remove(per_class_csv)

    metrics_state = {
        "best_recall": {},
        "first_seen_step": {},
    }
    save_json(metrics_state, os.path.join(metrics_root, "per_class_state.json"))

    task_best_acc_list = []
    step_forgetting_list = []

    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)
        exemplar_set._set_incremental_step_(step)

        print('Incremental step: {}'.format(step))

        if args.test_only == False:
            train(
                args=args,
                step=step,
                train_data_set=train_set,
                val_data_set=val_set,
                exemplar_set=exemplar_set,
                metrics_root=metrics_root,
                id_to_category=id_to_category,
            )

        step_forgetting = detailed_test(
            args=args,
            step=step,
            test_data_set=test_set,
            task_best_acc_list=task_best_acc_list,
            metrics_root=metrics_root,
            metrics_state=metrics_state,
            id_to_category=id_to_category
        )
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)

        ckpt_path = './save/{}/step_{}_best_model.pkl'.format(args.output_name, step)

        if args.dump_tsne:
            print("Dumping t-SNE plots for step {}...".format(step))
            out_root = os.path.join(args.tsne_out_root, args.output_name)
            make_tsne_plots_for_step(
                args=args,
                step=step,
                test_set=test_set,               # 注意：此时 test_set 已经 set_incremental_step(step)
                ckpt_path=ckpt_path,
                out_root=out_root,
                feature_type=args.tsne_feature,
                max_points_per_class=args.tsne_max_points_per_class,
            )

    Mean_forgetting = np.mean(step_forgetting_list) if len(step_forgetting_list) > 0 else 0.0
    print('Average Forgetting: {:.6f}'.format(Mean_forgetting))

    if args.dataset != 'AVE':
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()
        exemplar_set.close_visual_features_h5()