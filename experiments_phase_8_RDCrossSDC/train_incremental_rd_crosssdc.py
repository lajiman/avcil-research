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
# RD-CrossSDC v1 helpers
# ============================================================================

def _rd_weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12):
    """Weighted mean whose denominator includes every sample, including zero-hinge samples."""
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(eps)


def _rd_cross_modal_margin(
    query: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_sums: torch.Tensor,
    prototype_counts: torch.Tensor,
    positive_teacher_features: torch.Tensor,
    temperature: float,
):
    """
    Target-vs-rest cross-modal prototype margin.

    For an old exemplar, its own teacher feature is removed from the positive-class
    prototype whenever that class contains at least two prototype samples. This
    leave-one-out positive prevents the class-level margin from collapsing into a
    paired-instance matching objective.

    Args:
        query:
            Normalized student/reference feature, shape (B, D).
        labels:
            Global class labels, shape (B,).
        prototypes:
            Normalized prototype bank, shape (C, D).
        prototype_sums:
            Sum of normalized teacher features before final prototype normalization,
            shape (C, D).
        prototype_counts:
            Number of samples used for each prototype, shape (C,).
        positive_teacher_features:
            The paired normalized teacher feature from the prototype modality,
            shape (B, D). Used only for the leave-one-out positive prototype.
        temperature:
            Prototype-score temperature.
    """
    if temperature <= 0:
        raise ValueError('RD-CrossSDC temperatures must be positive')
    if prototypes.shape[0] < 2:
        raise ValueError('Cross-modal margin requires at least two candidate classes')

    labels = labels.long()
    scores = torch.mm(query, prototypes.transpose(0, 1)) / temperature

    # Use a leave-one-out target prototype for old exemplar anchors. Negative
    # prototypes remain the full class means.
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
    Build a frozen teacher prototype bank once per incremental step.

    Old-class prototypes use replay exemplars. New-class prototypes use current-step
    training samples passed through the frozen old model. New-class prototypes are
    competitors only; they never provide positive distillation targets.
    """
    old_model.eval()

    audio_sums = None
    visual_sums = None
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

        for data, labels in tqdm(loader, desc='RD prototypes ({})'.format(source_name), leave=False):
            labels = labels.to(device, non_blocking=True).long()
            visual = data[0].to(device, non_blocking=True)
            audio = data[1].to(device, non_blocking=True)

            if torch.any(labels < 0) or torch.any(labels >= num_seen_classes):
                raise RuntimeError(
                    'Prototype labels must lie in [0, {}). Got min={}, max={}.'.format(
                        num_seen_classes, int(labels.min().item()), int(labels.max().item())
                    )
                )

            # These checks match the training code's assumption that train_data_set
            # contains only current classes and exemplar_set contains only old classes.
            if source_name == 'memory' and torch.any(labels >= num_old_classes):
                raise RuntimeError(
                    'exemplar_set unexpectedly contains a current/new class label. '
                    'RD-CrossSDC assumes replay labels are < {}.'.format(num_old_classes)
                )
            if source_name == 'current' and torch.any(labels < num_old_classes):
                raise RuntimeError(
                    'train_data_set unexpectedly contains an old class label. '
                    'RD-CrossSDC assumes current-step labels are >= {}.'.format(num_old_classes)
                )

            outputs = old_model(
                visual=visual,
                audio=audio,
                out_feature_before_fusion=True,
                return_dict=True,
            )
            audio_feature = outputs['audio_feature'].detach()
            visual_feature = outputs['visual_feature'].detach()

            if audio_sums is None:
                audio_sums = torch.zeros(
                    num_seen_classes,
                    audio_feature.shape[1],
                    dtype=audio_feature.dtype,
                    device=device,
                )
                visual_sums = torch.zeros(
                    num_seen_classes,
                    visual_feature.shape[1],
                    dtype=visual_feature.dtype,
                    device=device,
                )

            audio_sums.index_add_(0, labels, audio_feature)
            visual_sums.index_add_(0, labels, visual_feature)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    if audio_sums is None or visual_sums is None:
        raise RuntimeError('Could not build RD-CrossSDC prototype bank: no samples were found')

    missing = torch.nonzero(counts <= 0, as_tuple=False).flatten()
    if missing.numel() > 0:
        raise RuntimeError(
            'RD-CrossSDC prototype bank is missing seen classes: {}'.format(
                missing.detach().cpu().tolist()
            )
        )

    audio_means = audio_sums / counts.unsqueeze(1)
    visual_means = visual_sums / counts.unsqueeze(1)

    return {
        'audio_prototypes': F.normalize(audio_means, dim=1, eps=1e-12).detach(),
        'visual_prototypes': F.normalize(visual_means, dim=1, eps=1e-12).detach(),
        'audio_sums': audio_sums.detach(),
        'visual_sums': visual_sums.detach(),
        'counts': counts.detach(),
    }


@torch.no_grad()
def _rd_compute_reliability_and_weights(
    args,
    old_model,
    exemplar_set,
    prototype_bank,
    num_old_classes: int,
):
    """
    Compute direction-specific teacher reliability over the old-class label space.

    A<-V reliability: old audio queries retrieve their labels from old visual prototypes.
    V<-A reliability: old visual queries retrieve their labels from old audio prototypes.
    """
    exemplar_len = len(exemplar_set)
    if exemplar_len <= 0:
        raise RuntimeError('RD-CrossSDC requires a non-empty exemplar set after step 0')

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
    old_visual_prototypes = prototype_bank['visual_prototypes'][:num_old_classes]
    old_audio_sums = prototype_bank['audio_sums'][:num_old_classes]
    old_visual_sums = prototype_bank['visual_sums'][:num_old_classes]
    old_counts = prototype_bank['counts'][:num_old_classes]

    for data, labels in tqdm(loader, desc='RD reliability', leave=False):
        labels = labels.to(device, non_blocking=True).long()
        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)

        if torch.any(labels < 0) or torch.any(labels >= num_old_classes):
            raise RuntimeError('Reliability estimation received a non-old exemplar label')

        outputs = old_model(
            visual=visual,
            audio=audio,
            out_feature_before_fusion=True,
            return_dict=True,
        )
        old_audio = outputs['audio_feature'].detach()
        old_visual = outputs['visual_feature'].detach()

        margin_a_from_v = _rd_cross_modal_margin(
            query=old_audio,
            labels=labels,
            prototypes=old_visual_prototypes,
            prototype_sums=old_visual_sums,
            prototype_counts=old_counts,
            positive_teacher_features=old_visual,
            temperature=args.rd_margin_temperature,
        )
        margin_v_from_a = _rd_cross_modal_margin(
            query=old_visual,
            labels=labels,
            prototypes=old_audio_prototypes,
            prototype_sums=old_audio_sums,
            prototype_counts=old_counts,
            positive_teacher_features=old_audio,
            temperature=args.rd_margin_temperature,
        )

        # sigmoid(target-vs-rest log-odds) equals the target softmax probability.
        prob_a_from_v = torch.sigmoid(margin_a_from_v).float()
        prob_v_from_a = torch.sigmoid(margin_v_from_a).float()

        rel_sum_a_from_v.index_add_(0, labels, prob_a_from_v)
        rel_sum_v_from_a.index_add_(0, labels, prob_v_from_a)
        rel_count.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    missing = torch.nonzero(rel_count <= 0, as_tuple=False).flatten()
    if missing.numel() > 0:
        raise RuntimeError(
            'No replay samples were available to estimate reliability for old classes: {}'.format(
                missing.detach().cpu().tolist()
            )
        )

    reliability_a_from_v = rel_sum_a_from_v / rel_count
    reliability_v_from_a = rel_sum_v_from_a / rel_count

    def make_normalized_weights(reliability):
        stabilized = (
            args.rd_reliability_floor
            + (1.0 - args.rd_reliability_floor) * reliability
        )
        relative = stabilized / stabilized.mean().clamp_min(1e-12)
        weights = (1.0 - args.rd_weight_alpha) + args.rd_weight_alpha * relative
        # Re-normalize after interpolation to make mean(weight)=1 exactly up to FP error.
        return (weights / weights.mean().clamp_min(1e-12)).detach()

    return {
        'reliability_a_from_v': reliability_a_from_v.detach(),
        'reliability_v_from_a': reliability_v_from_a.detach(),
        'weight_a_from_v': make_normalized_weights(reliability_a_from_v),
        'weight_v_from_a': make_normalized_weights(reliability_v_from_a),
    }


def _rd_weighted_crosssdc_instance(
    current_audio: torch.Tensor,
    current_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    weight_a_from_v: torch.Tensor,
    weight_v_from_a: torch.Tensor,
    temperature: float,
):
    """Reliability-weighted temporal CrossSDC-I on replay exemplars only."""
    batch_size = labels.shape[0]
    if batch_size <= 1:
        zero = (current_audio.sum() + current_visual.sum()) * 0.0
        return zero, zero.detach(), zero.detach()

    targets = torch.arange(batch_size, device=labels.device)

    score_a_from_v = torch.mm(current_audio, old_visual.transpose(0, 1)) / temperature
    score_v_from_a = torch.mm(current_visual, old_audio.transpose(0, 1)) / temperature

    per_sample_a_from_v = F.cross_entropy(score_a_from_v, targets, reduction='none')
    per_sample_v_from_a = F.cross_entropy(score_v_from_a, targets, reduction='none')

    sample_weight_a_from_v = weight_a_from_v.index_select(0, labels).detach()
    sample_weight_v_from_a = weight_v_from_a.index_select(0, labels).detach()

    loss_a_from_v = _rd_weighted_mean(per_sample_a_from_v, sample_weight_a_from_v)
    loss_v_from_a = _rd_weighted_mean(per_sample_v_from_a, sample_weight_v_from_a)

    return 0.5 * (loss_a_from_v + loss_v_from_a), loss_a_from_v, loss_v_from_a


def _rd_cross_modal_margin_retention(
    args,
    current_audio: torch.Tensor,
    current_visual: torch.Tensor,
    old_audio: torch.Tensor,
    old_visual: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank,
    weight_a_from_v: torch.Tensor,
    weight_v_from_a: torch.Tensor,
):
    """One-sided, reliability-weighted expanded-label cross-modal margin retention."""
    ref_margin_a_from_v = _rd_cross_modal_margin(
        query=old_audio,
        labels=labels,
        prototypes=prototype_bank['visual_prototypes'],
        prototype_sums=prototype_bank['visual_sums'],
        prototype_counts=prototype_bank['counts'],
        positive_teacher_features=old_visual,
        temperature=args.rd_margin_temperature,
    ).detach()
    cur_margin_a_from_v = _rd_cross_modal_margin(
        query=current_audio,
        labels=labels,
        prototypes=prototype_bank['visual_prototypes'],
        prototype_sums=prototype_bank['visual_sums'],
        prototype_counts=prototype_bank['counts'],
        positive_teacher_features=old_visual,
        temperature=args.rd_margin_temperature,
    )

    ref_margin_v_from_a = _rd_cross_modal_margin(
        query=old_visual,
        labels=labels,
        prototypes=prototype_bank['audio_prototypes'],
        prototype_sums=prototype_bank['audio_sums'],
        prototype_counts=prototype_bank['counts'],
        positive_teacher_features=old_audio,
        temperature=args.rd_margin_temperature,
    ).detach()
    cur_margin_v_from_a = _rd_cross_modal_margin(
        query=current_visual,
        labels=labels,
        prototypes=prototype_bank['audio_prototypes'],
        prototype_sums=prototype_bank['audio_sums'],
        prototype_counts=prototype_bank['counts'],
        positive_teacher_features=old_audio,
        temperature=args.rd_margin_temperature,
    )

    violation_a_from_v = F.relu(
        ref_margin_a_from_v - cur_margin_a_from_v - args.rd_margin_tolerance
    )
    violation_v_from_a = F.relu(
        ref_margin_v_from_a - cur_margin_v_from_a - args.rd_margin_tolerance
    )

    sample_weight_a_from_v = weight_a_from_v.index_select(0, labels).detach()
    sample_weight_v_from_a = weight_v_from_a.index_select(0, labels).detach()

    loss_a_from_v = _rd_weighted_mean(violation_a_from_v, sample_weight_a_from_v)
    loss_v_from_a = _rd_weighted_mean(violation_v_from_a, sample_weight_v_from_a)
    loss = 0.5 * (loss_a_from_v + loss_v_from_a)

    stats = {
        'loss_a_from_v': loss_a_from_v.detach(),
        'loss_v_from_a': loss_v_from_a.detach(),
        'active_a_from_v': (violation_a_from_v > 0).float().mean().detach(),
        'active_v_from_a': (violation_v_from_a > 0).float().mean().detach(),
        'ref_margin_a_from_v': ref_margin_a_from_v.mean().detach(),
        'cur_margin_a_from_v': cur_margin_a_from_v.mean().detach(),
        'ref_margin_v_from_a': ref_margin_v_from_a.mean().detach(),
        'cur_margin_v_from_a': cur_margin_v_from_a.mean().detach(),
    }
    return loss, stats


def _rd_save_step_statistics(
    args,
    step: int,
    num_old_classes: int,
    prototype_bank,
    rd_state,
    metrics_root: str,
    id_to_category: dict,
):
    out_dir = os.path.join(metrics_root, 'rd_crosssdc')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'step_{}_class_weights.csv'.format(step))
    header = [
        'step', 'class_id', 'category_name', 'prototype_count',
        'reliability_a_from_v', 'reliability_v_from_a',
        'weight_a_from_v', 'weight_v_from_a',
    ]

    rows = []
    for c in range(num_old_classes):
        rows.append({
            'step': step,
            'class_id': c,
            'category_name': id_to_category.get(c, 'class_{}'.format(c)),
            'prototype_count': int(prototype_bank['counts'][c].item()),
            'reliability_a_from_v': float(rd_state['reliability_a_from_v'][c].item()),
            'reliability_v_from_a': float(rd_state['reliability_v_from_a'][c].item()),
            'weight_a_from_v': float(rd_state['weight_a_from_v'][c].item()),
            'weight_v_from_a': float(rd_state['weight_v_from_a'][c].item()),
        })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    config = {
        'step': step,
        'rd_lambda_i': args.rd_lambda_i,
        'rd_lambda_m': args.rd_lambda_m,
        'rd_instance_temperature': args.rd_instance_temperature,
        'rd_margin_temperature': args.rd_margin_temperature,
        'rd_weight_alpha': args.rd_weight_alpha,
        'rd_reliability_floor': args.rd_reliability_floor,
        'rd_margin_tolerance': args.rd_margin_tolerance,
        'leave_one_out_positive_prototype': True,
        'class_weight_mean_a_from_v': float(rd_state['weight_a_from_v'].mean().item()),
        'class_weight_mean_v_from_a': float(rd_state['weight_v_from_a'].mean().item()),
    }
    save_json(config, os.path.join(out_dir, 'step_{}_config.json'.format(step)))

    print(
        'RD weights A<-V min/mean/max: {:.4f}/{:.4f}/{:.4f}'.format(
            rd_state['weight_a_from_v'].min().item(),
            rd_state['weight_a_from_v'].mean().item(),
            rd_state['weight_a_from_v'].max().item(),
        ),
        flush=True,
    )
    print(
        'RD weights V<-A min/mean/max: {:.4f}/{:.4f}/{:.4f}'.format(
            rd_state['weight_v_from_a'].min().item(),
            rd_state['weight_v_from_a'].mean().item(),
            rd_state['weight_v_from_a'].max().item(),
        ),
        flush=True,
    )

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

    train_loader = DataLoader(train_data_set, batch_size=min(args.train_batch_size, train_data_set.__len__()), num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=min(args.infer_batch_size, val_data_set.__len__()), num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)

    step_out_class_num = (step + 1) * args.class_num_per_step
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
        old_model = None
        exemplar_loader = None
        last_step_out_class_num = 0
    else:
        model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.output_name, step-1))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.output_name, step-1))

        exemplar_loader = DataLoader(exemplar_set, batch_size=min(args.exemplar_batch_size, exemplar_set.__len__()), num_workers=args.num_workers,
                                     pin_memory=True, drop_last=True, shuffle=True)

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

    # ------------------------------------------------------------------
    # RD-CrossSDC state is fixed within an incremental step because both
    # the teacher and the step data are fixed. Step 0 remains pure AVCIL.
    # ------------------------------------------------------------------
    rd_prototype_bank = None
    rd_state = None
    if step > 0 and args.rd_crosssdc:
        print('Building RD-CrossSDC teacher prototype bank...', flush=True)
        rd_prototype_bank = _rd_build_teacher_prototype_bank(
            args=args,
            old_model=old_model,
            train_data_set=train_data_set,
            exemplar_set=exemplar_set,
            num_old_classes=last_step_out_class_num,
            num_seen_classes=step_out_class_num,
        )
        rd_state = _rd_compute_reliability_and_weights(
            args=args,
            old_model=old_model,
            exemplar_set=exemplar_set,
            prototype_bank=rd_prototype_bank,
            num_old_classes=last_step_out_class_num,
        )
        if metrics_root is not None:
            _rd_save_step_statistics(
                args=args,
                step=step,
                num_old_classes=last_step_out_class_num,
                prototype_bank=rd_prototype_bank,
                rd_state=rd_state,
                metrics_root=metrics_root,
                id_to_category=id_to_category or {},
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
        rd_epoch_margin = 0.0
        rd_epoch_active_a_from_v = 0.0
        rd_epoch_active_v_from_a = 0.0

        if step == 0:
            iterator = tqdm(train_loader)
        else:
            iterator = tzip(train_loader, cycle(exemplar_loader))

        for samples in iterator:
            if step == 0:
                data, labels = samples
                labels = labels.to(device)
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                out, audio_feature, visual_feature = model(visual=visual, audio=audio, out_feature_before_fusion=True)
                loss = CE_loss(step_out_class_num, out, labels)
            else:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(device)
                labels_ = labels % args.class_num_per_step
                labels_ = labels_.to(device)

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device).long()

                data_batch_size = labels_.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                visual = data[0]
                audio = data[1]
                exemplar_visual = exemplar_data[0]
                exemplar_audio = exemplar_data[1]
                total_visual = torch.cat((visual, exemplar_visual)).to(device)
                total_audio = torch.cat((audio, exemplar_audio)).to(device)

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
                    instance_contra_loss = cal_contrastive_loss(audio_feature, visual_feature, temperature=args.instance_contrastive_temperature)

                if args.class_contrastive:
                    all_labels = torch.cat((labels, exemplar_labels))
                    class_contra_loss = class_contrastive_loss(audio_feature, visual_feature, all_labels, temperature=args.class_contrastive_temperature)

                if args.attn_score_distil:
                    exem_spatial_attn_score = spatial_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(2, 3)
                    exem_spatial_attn_score = exem_spatial_attn_score.reshape(-1, exem_spatial_attn_score.shape[-1])

                    exem_old_spatial_attn_score = old_spatial_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(2, 3)
                    exem_old_spatial_attn_score = exem_old_spatial_attn_score.reshape(-1, exem_old_spatial_attn_score.shape[-1])

                    exem_temporal_attn_score = temporal_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(1, 2)
                    exem_temporal_attn_score = exem_temporal_attn_score.reshape(-1, exem_temporal_attn_score.shape[-1])

                    exem_old_temporal_attn_score = old_temporal_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(1, 2)
                    exem_old_temporal_attn_score = exem_old_temporal_attn_score.reshape(-1, exem_old_temporal_attn_score.shape[-1])

                    spatial_attn_dist_loss = F.kl_div(exem_spatial_attn_score.log(), exem_old_spatial_attn_score, reduction='sum') / exemplar_data_batch_size
                    temporal_attn_dist_loss = F.kl_div(exem_temporal_attn_score.log(), exem_old_temporal_attn_score, reduction='sum') / exemplar_data_batch_size

                old_out = old_out[:, :last_step_out_class_num]

                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = CE_loss(args.class_num_per_step, curr_out, labels_)

                prev_out = out[data_batch_size:data_batch_size+exemplar_data_batch_size, :last_step_out_class_num]
                loss_prev = CE_loss(last_step_out_class_num, prev_out, exemplar_labels)

                loss_CE = (loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size) / (data_batch_size + exemplar_data_batch_size)

                if args.dataset == 'AVE' and args.class_num_per_step == 4 and step == 1:
                    loss_CE = CE_loss(args.class_num_per_step + last_step_out_class_num, out, torch.cat((labels, exemplar_labels)))

                loss_KD = torch.zeros(step).to(device)

                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step

                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                loss_KD = loss_KD.sum()
                loss = loss_CE + loss_KD

                if args.instance_contrastive:
                    loss += args.lam_I * instance_contra_loss
                if args.class_contrastive:
                    loss += args.lam_C * class_contra_loss
                if args.attn_score_distil:
                    loss += args.lam * spatial_attn_dist_loss + (1 - args.lam) * temporal_attn_dist_loss

                if args.rd_crosssdc:
                    exemplar_slice = slice(data_batch_size, data_batch_size + exemplar_data_batch_size)
                    current_exemplar_audio = audio_feature[exemplar_slice]
                    current_exemplar_visual = visual_feature[exemplar_slice]
                    old_exemplar_audio = old_audio_feature[exemplar_slice]
                    old_exemplar_visual = old_visual_feature[exemplar_slice]

                    rd_cross_i_loss, _, _ = _rd_weighted_crosssdc_instance(
                        current_audio=current_exemplar_audio,
                        current_visual=current_exemplar_visual,
                        old_audio=old_exemplar_audio,
                        old_visual=old_exemplar_visual,
                        labels=exemplar_labels,
                        weight_a_from_v=rd_state['weight_a_from_v'],
                        weight_v_from_a=rd_state['weight_v_from_a'],
                        temperature=args.rd_instance_temperature,
                    )
                    rd_margin_loss, rd_margin_stats = _rd_cross_modal_margin_retention(
                        args=args,
                        current_audio=current_exemplar_audio,
                        current_visual=current_exemplar_visual,
                        old_audio=old_exemplar_audio,
                        old_visual=old_exemplar_visual,
                        labels=exemplar_labels,
                        prototype_bank=rd_prototype_bank,
                        weight_a_from_v=rd_state['weight_a_from_v'],
                        weight_v_from_a=rd_state['weight_v_from_a'],
                    )

                    # The normalized weights decide allocation. The two lambda values
                    # independently decide the global strength of each added objective.
                    loss += args.rd_lambda_i * rd_cross_i_loss
                    loss += args.rd_lambda_m * rd_margin_loss

                    rd_epoch_cross_i += float(rd_cross_i_loss.detach().item())
                    rd_epoch_margin += float(rd_margin_loss.detach().item())
                    rd_epoch_active_a_from_v += float(rd_margin_stats['active_a_from_v'].item())
                    rd_epoch_active_v_from_a += float(rd_margin_stats['active_v_from_a'].item())

            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1

        train_loss /= num_steps
        train_loss_list.append(train_loss)
        if step > 0 and args.rd_crosssdc:
            print(
                'Epoch:{} train_loss:{:.5f} RD-I:{:.5f} RD-M:{:.5f} active(A<-V/V<-A):{:.3f}/{:.3f}'.format(
                    epoch,
                    train_loss,
                    rd_epoch_cross_i / num_steps,
                    rd_epoch_margin / num_steps,
                    rd_epoch_active_a_from_v / num_steps,
                    rd_epoch_active_v_from_a / num_steps,
                ),
                flush=True,
            )
        else:
            print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0]
                val_audio = val_data[1]
                val_visual = val_visual.to(device)
                val_audio = val_audio.to(device)
                if torch.cuda.device_count() > 1:
                    val_out_logits = model.module.forward(visual=val_visual, audio=val_audio)
                else:
                    val_out_logits = model(visual=val_visual, audio=val_audio)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print('Epoch:{} val_res:{:.6f} '.format(epoch, val_top1), flush=True)

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(model.module, './save/{}/step_{}_best_model.pkl'.format(args.output_name, step))
            else:
                torch.save(model, './save/{}/step_{}_best_model.pkl'.format(args.output_name, step))

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

    # RD-CrossSDC v1. These losses are applied only for step > 0 and only to replay exemplars.
    parser.add_argument('--rd_crosssdc', action='store_true', default=False)
    parser.add_argument('--rd_lambda_i', type=float, default=0.1,
                        help='Global coefficient for reliability-weighted temporal CrossSDC-I.')
    parser.add_argument('--rd_lambda_m', type=float, default=0.1,
                        help='Global coefficient for cross-modal margin retention.')
    parser.add_argument('--rd_instance_temperature', type=float, default=0.05,
                        help='Temperature for current-modality versus old-other-modality instance matching.')
    parser.add_argument('--rd_margin_temperature', type=float, default=0.1,
                        help='Temperature used by prototype reliability and target-vs-rest margins.')
    parser.add_argument('--rd_weight_alpha', type=float, default=0.5,
                        help='Interpolation from uniform weights (0) to fully reliability-adaptive weights (1).')
    parser.add_argument('--rd_reliability_floor', type=float, default=0.2,
                        help='Floor applied before mean-normalizing class reliability.')
    parser.add_argument('--rd_margin_tolerance', type=float, default=0.0,
                        help='Allowed target-vs-rest log-odds decrease before the one-sided hinge activates.')

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.1)

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dump_tsne", action="store_true", help="If set, dump t-SNE plots for each step")
    parser.add_argument("--tsne_feature", type=str, default="logits",
                        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"])
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    

    args = parser.parse_args()

    if not (0.0 <= args.rd_weight_alpha <= 1.0):
        parser.error('--rd_weight_alpha must lie in [0, 1]')
    if not (0.0 <= args.rd_reliability_floor <= 1.0):
        parser.error('--rd_reliability_floor must lie in [0, 1]')
    if args.rd_instance_temperature <= 0 or args.rd_margin_temperature <= 0:
        parser.error('RD-CrossSDC temperatures must be positive')
    if args.rd_lambda_i < 0 or args.rd_lambda_m < 0:
        parser.error('RD-CrossSDC loss coefficients must be non-negative')
    if args.rd_margin_tolerance < 0:
        parser.error('--rd_margin_tolerance must be non-negative')

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