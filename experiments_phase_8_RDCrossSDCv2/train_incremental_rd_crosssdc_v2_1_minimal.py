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



def cross_sdc_class_contrastive_loss(feature_1, feature_2, label, temperature=0.05):
    """
    Equation-(9)-style class-aware cross-modal contrastive loss.

    Unlike the baseline class_contrastive_loss above, this implementation
    explicitly averages over the number of positive samples for each anchor,
    matching the CrossSDC formulation more closely.

    feature_1, feature_2 are expected to be L2-normalized z1 features.
    """
    positive_mask = (label.unsqueeze(1) == label.unsqueeze(0)).float()

    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    log_prob = F.log_softmax(score, dim=-1)

    num_pos = positive_mask.sum(dim=1).clamp_min(1.0)
    loss = -((log_prob * positive_mask).sum(dim=1) / num_pos).mean()
    return loss


def cross_sdc_z1_loss(cur_audio, cur_visual,
                      old_audio, old_visual,
                      labels, temperature=0.05):
    """
    CrossSDC-like cross-task / cross-modal loss for this AVCIL baseline.

    The baseline already applies current-current z1 instance/class contrastive
    learning to the concatenated current+exemplar batch. Therefore this adds
    only the missing memory cross-task terms:

        current audio  -> old visual
        old audio      -> current visual

    All inputs here are exemplar z1 features. The old-model features are
    detached/frozen.
    """
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


# ============================================================================
# Minimal RD-CrossSDC v2.1 additions
#
# Design goal:
#   - preserve the supplied CrossSDC implementation exactly when the new
#     options are disabled;
#   - test whether CMR can replace CrossSDC-C without the static Trust x Need
#     weighting used in v1/v2;
#   - optionally stop standard CrossSDC-I from treating same-class, non-paired
#     examples as negatives.
# ============================================================================

def _rd_capture_rng_state():
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


def cross_sdc_instance_loss_same_class_ignored(
    feature_1,
    feature_2,
    labels,
    temperature=0.05,
):
    """
    Paired temporal InfoNCE that ignores same-class, non-paired samples.

    For anchor i, the admissible denominator contains:
      - its paired positive j=i;
      - all samples from different classes.

    Samples j != i with y_j == y_i are neither positives nor negatives. This
    removes the direct class-level conflict between standard CrossSDC-I and a
    prototype/class retention objective.
    """
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    batch_size = score.shape[0]
    targets = torch.arange(batch_size, device=score.device)

    same_class = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    paired = torch.eye(batch_size, dtype=torch.bool, device=score.device)
    allowed = (~same_class) | paired
    masked_score = score.masked_fill(~allowed, float('-inf'))

    # Use the diagonal log probability directly. This avoids 0 * (-inf) in
    # the supplied one-hot CE implementation while preserving row-wise InfoNCE.
    log_prob = F.log_softmax(masked_score, dim=1)
    return -log_prob[targets, targets].mean()


def cross_sdc_instance_loss_configurable(
    cur_audio,
    cur_visual,
    old_audio,
    old_visual,
    labels,
    temperature,
    same_class_mode,
):
    if same_class_mode == 'negative':
        # Exact supplied CrossSDC-I implementation and legacy orientation.
        inst_cur_old = cal_contrastive_loss(
            cur_audio, old_visual, temperature=temperature
        )
        inst_old_cur = cal_contrastive_loss(
            old_audio, cur_visual, temperature=temperature
        )
    elif same_class_mode == 'ignore':
        inst_cur_old = cross_sdc_instance_loss_same_class_ignored(
            cur_audio, old_visual, labels, temperature=temperature
        )
        inst_old_cur = cross_sdc_instance_loss_same_class_ignored(
            old_audio, cur_visual, labels, temperature=temperature
        )
    else:
        raise ValueError(
            "cross_sdc_i_same_class_mode must be 'negative' or 'ignore'"
        )
    return 0.5 * (inst_cur_old + inst_old_cur)


@torch.no_grad()
def build_old_teacher_prototype_bank(
    old_model,
    exemplar_set,
    num_old_classes,
    batch_size,
    num_workers,
):
    """Build normalized old-audio and old-visual class prototypes on replay."""
    exemplar_len = exemplar_set.__len__()
    if exemplar_len <= 0:
        raise RuntimeError('CMR requires a non-empty exemplar set after step 0')

    loader = DataLoader(
        exemplar_set,
        batch_size=min(batch_size, exemplar_len),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    audio_sums = None
    visual_sums = None
    counts = torch.zeros(num_old_classes, dtype=torch.float32, device=device)

    old_model.eval()
    for exemplar_data, exemplar_labels in loader:
        visual = exemplar_data[0].to(device)
        audio = exemplar_data[1].to(device)
        labels = exemplar_labels.to(device).long()

        _, old_audio, old_visual = old_model(
            visual=visual,
            audio=audio,
            out_feature_before_fusion=True,
        )
        old_audio = old_audio.detach()
        old_visual = old_visual.detach()

        if audio_sums is None:
            feature_dim = old_audio.shape[1]
            audio_sums = torch.zeros(
                num_old_classes, feature_dim,
                dtype=old_audio.dtype, device=device,
            )
            visual_sums = torch.zeros(
                num_old_classes, feature_dim,
                dtype=old_visual.dtype, device=device,
            )

        audio_sums.index_add_(0, labels, old_audio)
        visual_sums.index_add_(0, labels, old_visual)
        counts.index_add_(
            0, labels, torch.ones_like(labels, dtype=torch.float32)
        )

    if audio_sums is None:
        raise RuntimeError('Failed to construct CMR teacher prototypes')
    missing = torch.nonzero(counts <= 0, as_tuple=False).flatten()
    if missing.numel() > 0:
        raise RuntimeError(
            'CMR prototype bank is missing old classes: {}'.format(
                missing.detach().cpu().tolist()
            )
        )

    audio_means = audio_sums / counts.unsqueeze(1)
    visual_means = visual_sums / counts.unsqueeze(1)
    return {
        'audio_prototypes': F.normalize(audio_means, dim=1).detach(),
        'visual_prototypes': F.normalize(visual_means, dim=1).detach(),
        'audio_sums': audio_sums.detach(),
        'visual_sums': visual_sums.detach(),
        'counts': counts.detach(),
    }


def _cross_modal_target_vs_rest_margin(
    query,
    labels,
    prototypes,
    prototype_sums,
    prototype_counts,
    paired_old_key,
    temperature,
):
    """Prototype target-vs-rest log-odds with a leave-one-out positive."""
    if prototypes.shape[0] < 2:
        raise ValueError('CMR requires at least two old classes')

    scores = torch.mm(query, prototypes.transpose(0, 1)) / temperature

    full_target_proto = prototypes.index_select(0, labels)
    target_count = prototype_counts.index_select(0, labels)
    loo_sum = prototype_sums.index_select(0, labels) - paired_old_key.detach()
    loo_proto = F.normalize(loo_sum, dim=1)
    target_proto = torch.where(
        (target_count > 1.0).unsqueeze(1), loo_proto, full_target_proto
    )
    target_score = torch.sum(query * target_proto, dim=1) / temperature

    scores = scores.scatter(1, labels.unsqueeze(1), target_score.unsqueeze(1))
    target_mask = F.one_hot(labels, num_classes=prototypes.shape[0]).bool()
    negative_scores = scores.masked_fill(target_mask, float('-inf'))
    return target_score - torch.logsumexp(negative_scores, dim=1)


def cross_modal_margin_retention_loss(
    cur_audio,
    cur_visual,
    old_audio,
    old_visual,
    labels,
    prototype_bank,
    temperature=0.1,
    mode='smooth',
    reserve=0.05,
    beta=0.05,
):
    """
    Uniform, direction-symmetric CMR over old classes only.

    hard:
        [B_ref + reserve - B_cur]_+

    smooth:
        beta * softplus((B_ref + reserve - B_cur) / beta)

    The smooth reserve version keeps a bounded, non-abrupt gradient near the
    teacher margin while still allowing the current representation to improve.
    """
    ref_a = _cross_modal_target_vs_rest_margin(
        old_audio, labels,
        prototype_bank['visual_prototypes'],
        prototype_bank['visual_sums'],
        prototype_bank['counts'],
        old_visual,
        temperature,
    ).detach()
    cur_a = _cross_modal_target_vs_rest_margin(
        cur_audio, labels,
        prototype_bank['visual_prototypes'],
        prototype_bank['visual_sums'],
        prototype_bank['counts'],
        old_visual,
        temperature,
    )

    ref_v = _cross_modal_target_vs_rest_margin(
        old_visual, labels,
        prototype_bank['audio_prototypes'],
        prototype_bank['audio_sums'],
        prototype_bank['counts'],
        old_audio,
        temperature,
    ).detach()
    cur_v = _cross_modal_target_vs_rest_margin(
        cur_visual, labels,
        prototype_bank['audio_prototypes'],
        prototype_bank['audio_sums'],
        prototype_bank['counts'],
        old_audio,
        temperature,
    )

    gap_a = ref_a + reserve - cur_a
    gap_v = ref_v + reserve - cur_v

    if mode == 'hard':
        loss_a = F.relu(gap_a).mean()
        loss_v = F.relu(gap_v).mean()
    elif mode == 'smooth':
        if beta <= 0:
            raise ValueError('cmr_beta must be positive for smooth CMR')
        loss_a = (beta * F.softplus(gap_a / beta)).mean()
        loss_v = (beta * F.softplus(gap_v / beta)).mean()
    else:
        raise ValueError("cmr_mode must be 'hard' or 'smooth'")

    loss = 0.5 * (loss_a + loss_v)
    stats = {
        'loss_a': loss_a.detach(),
        'loss_v': loss_v.detach(),
        'active_a': (gap_a > 0).float().mean().detach(),
        'active_v': (gap_v > 0).float().mean().detach(),
        'signed_gap_a': gap_a.mean().detach(),
        'signed_gap_v': gap_v.mean().detach(),
    }
    return loss, stats


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


def train(args, step, train_data_set, val_data_set, exemplar_set):
    T = 2

    train_loader = DataLoader(train_data_set, batch_size=min(args.train_batch_size, train_data_set.__len__()), num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=min(args.infer_batch_size, val_data_set.__len__()), num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)
    
    step_out_class_num = (step + 1) * args.class_num_per_step
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
    else:
        model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step-1))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step-1))

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
        # old_model = old_model.to('cpu')
        old_model.eval()

    cmr_prototype_bank = None
    if step > 0 and args.cmr and args.lam_cmr > 0:
        # Diagnostic prototype passes must not alter the random sequence used by
        # the actual shuffled training loaders or exemplar sampling.
        cmr_rng_state = _rd_capture_rng_state()
        try:
            cmr_prototype_bank = build_old_teacher_prototype_bank(
                old_model=old_model,
                exemplar_set=exemplar_set,
                num_old_classes=last_step_out_class_num,
                batch_size=args.exemplar_batch_size,
                num_workers=args.num_workers,
            )
        finally:
            _rd_restore_rng_state(cmr_rng_state)
        print(
            'Built CMR old-class prototype bank: {} classes'.format(
                last_step_out_class_num
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
        cross_sdc_inst_sum = 0.0
        cross_sdc_cls_sum = 0.0
        cmr_loss_sum = 0.0
        cmr_loss_a_sum = 0.0
        cmr_loss_v_sum = 0.0
        cmr_active_a_sum = 0.0
        cmr_active_v_sum = 0.0
        model.train()
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
                # CE_loss = CE_loss(step_out_class_num, out, labels)
                # loss = CE_loss
                loss = CE_loss(step_out_class_num, out, labels)
            else:
                curr, prev = samples
                data, labels = curr
                # labels = labels % ((step_out_class_num - 1) - (last_step_out_class_num - 1))
                labels = labels.to(device)
                labels_ = labels % args.class_num_per_step
                labels_ = labels_.to(device)

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device)

                data_batch_size = labels_.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                visual = data[0]
                audio = data[1]
                exemplar_visual = exemplar_data[0]
                exemplar_audio = exemplar_data[1]
                total_visual = torch.cat((visual, exemplar_visual))
                total_audio = torch.cat((audio, exemplar_audio))
                total_visual = total_visual.to(device)
                total_audio = total_audio.to(device)
                out, audio_feature, visual_feature, spatial_attn_score, temporal_attn_score = model(visual=total_visual, audio=total_audio, out_feature_before_fusion=True, out_attn_score=True)
                with torch.no_grad():
                    old_out, old_audio_feature, old_visual_feature, old_spatial_attn_score, old_temporal_attn_score = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        out_feature_before_fusion=True,
                        out_attn_score=True,
                    )
                    old_out = old_out.detach()
                    old_audio_feature = old_audio_feature.detach()
                    old_visual_feature = old_visual_feature.detach()
                    old_spatial_attn_score = old_spatial_attn_score.detach()
                    old_temporal_attn_score = old_temporal_attn_score.detach()
                
                if args.instance_contrastive:
                    instance_contra_loss = cal_contrastive_loss(audio_feature, visual_feature, temperature=args.instance_contrastive_temperature)
                
                if args.class_contrastive:
                    all_labels = torch.cat((labels, exemplar_labels))
                    class_contra_loss = class_contrastive_loss(audio_feature, visual_feature, all_labels, temperature=args.class_contrastive_temperature)
                
                # ---------------------------------------------------------
                # CrossSDC-like z1 cross-task relation preservation.
                # Only replay exemplars use current-old cross-task pairs.
                # Step 0 never reaches this branch, so Task 1 is unchanged.
                # ---------------------------------------------------------
                if args.cross_sdc:
                    exem_start = data_batch_size
                    exem_end = data_batch_size + exemplar_data_batch_size

                    cur_exem_audio_feature = audio_feature[exem_start:exem_end]
                    cur_exem_visual_feature = visual_feature[exem_start:exem_end]
                    old_exem_audio_feature = old_audio_feature[exem_start:exem_end]
                    old_exem_visual_feature = old_visual_feature[exem_start:exem_end]

                    if (
                        args.cross_sdc_i_same_class_mode == 'negative'
                        and args.lam_cross_sdc_c > 0
                    ):
                        # Exact supplied CrossSDC control path.
                        cross_sdc_inst_loss, cross_sdc_cls_loss = cross_sdc_z1_loss(
                            cur_audio=cur_exem_audio_feature,
                            cur_visual=cur_exem_visual_feature,
                            old_audio=old_exem_audio_feature,
                            old_visual=old_exem_visual_feature,
                            labels=exemplar_labels,
                            temperature=args.cross_sdc_temperature,
                        )
                    else:
                        cross_sdc_inst_loss = cross_sdc_instance_loss_configurable(
                            cur_audio=cur_exem_audio_feature,
                            cur_visual=cur_exem_visual_feature,
                            old_audio=old_exem_audio_feature,
                            old_visual=old_exem_visual_feature,
                            labels=exemplar_labels,
                            temperature=args.cross_sdc_temperature,
                            same_class_mode=args.cross_sdc_i_same_class_mode,
                        )
                        if args.lam_cross_sdc_c > 0:
                            cls_cur_old = cross_sdc_class_contrastive_loss(
                                cur_exem_audio_feature, old_exem_visual_feature,
                                exemplar_labels, temperature=args.cross_sdc_temperature,
                            )
                            cls_old_cur = cross_sdc_class_contrastive_loss(
                                old_exem_audio_feature, cur_exem_visual_feature,
                                exemplar_labels, temperature=args.cross_sdc_temperature,
                            )
                            cross_sdc_cls_loss = 0.5 * (cls_cur_old + cls_old_cur)
                        else:
                            cross_sdc_cls_loss = cross_sdc_inst_loss.detach() * 0.0

                    if args.cmr and args.lam_cmr > 0:
                        cmr_loss, cmr_stats = cross_modal_margin_retention_loss(
                            cur_audio=cur_exem_audio_feature,
                            cur_visual=cur_exem_visual_feature,
                            old_audio=old_exem_audio_feature,
                            old_visual=old_exem_visual_feature,
                            labels=exemplar_labels,
                            prototype_bank=cmr_prototype_bank,
                            temperature=args.cmr_temperature,
                            mode=args.cmr_mode,
                            reserve=args.cmr_reserve,
                            beta=args.cmr_beta,
                        )
                    else:
                        cmr_loss = cross_sdc_inst_loss.detach() * 0.0
                        cmr_stats = {
                            'loss_a': cmr_loss, 'loss_v': cmr_loss,
                            'active_a': cmr_loss, 'active_v': cmr_loss,
                        }

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

                old_out = old_out[:,:last_step_out_class_num]
                
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
                if args.cross_sdc:
                    loss += (
                        args.lam_cross_sdc_i * cross_sdc_inst_loss
                        + args.lam_cross_sdc_c * cross_sdc_cls_loss
                    )
                if args.cmr and args.lam_cmr > 0:
                    loss += args.lam_cmr * cmr_loss
                if args.attn_score_distil:
                    loss += args.lam * spatial_attn_dist_loss + (1 - args.lam) * temporal_attn_dist_loss

                if args.cross_sdc:
                    cross_sdc_inst_sum += cross_sdc_inst_loss.item()
                    cross_sdc_cls_sum += cross_sdc_cls_loss.item()
                if args.cmr and args.lam_cmr > 0:
                    cmr_loss_sum += cmr_loss.item()
                    cmr_loss_a_sum += cmr_stats['loss_a'].item()
                    cmr_loss_v_sum += cmr_stats['loss_v'].item()
                    cmr_active_a_sum += cmr_stats['active_a'].item()
                    cmr_active_v_sum += cmr_stats['active_v'].item()
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)
        if step > 0 and args.cross_sdc:
            avg_cross_inst = cross_sdc_inst_sum / max(num_steps, 1)
            avg_cross_cls = cross_sdc_cls_sum / max(num_steps, 1)
            print(
                'Epoch:{} cross_sdc_inst:{:.5f} cross_sdc_cls:{:.5f} '
                'weighted_cross_sdc:{:.5f}'.format(
                    epoch,
                    avg_cross_inst,
                    avg_cross_cls,
                    args.lam_cross_sdc_i * avg_cross_inst
                    + args.lam_cross_sdc_c * avg_cross_cls,
                ),
                flush=True,
            )
        if step > 0 and args.cmr and args.lam_cmr > 0:
            avg_cmr = cmr_loss_sum / max(num_steps, 1)
            print(
                'Epoch:{} cmr:{:.6f} weighted_cmr:{:.6f} '
                'cmr_A:{:.6f} cmr_V:{:.6f} active_A/V:{:.3f}/{:.3f}'.format(
                    epoch,
                    avg_cmr,
                    args.lam_cmr * avg_cmr,
                    cmr_loss_a_sum / max(num_steps, 1),
                    cmr_loss_v_sum / max(num_steps, 1),
                    cmr_active_a_sum / max(num_steps, 1),
                    cmr_active_v_sum / max(num_steps, 1),
                ),
                flush=True,
            )

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
                torch.save(model.module, './save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
            else:
                torch.save(model, './save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
        
        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/train_loss_step_{}.png'.format(args.dataset, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/val_acc_step_{}.png'.format(args.dataset, step))
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, opt, epoch)


# def detailed_test(args, step, test_data_set, task_best_acc_list):
#     print("=====================================")
#     print("Start testing...")
#     print("=====================================")

#     model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
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

    model = torch.load('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step))
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

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.1)

    # CrossSDC-like z1 cross-task / cross-modal relation preservation.
    parser.add_argument('--cross_sdc', action='store_true', default=False)
    parser.add_argument('--cross_sdc_temperature', type=float, default=0.05)
    parser.add_argument('--lam_cross_sdc_i', type=float, default=0.1)
    parser.add_argument('--lam_cross_sdc_c', type=float, default=0.3)
    parser.add_argument(
        '--cross_sdc_i_same_class_mode',
        type=str,
        default='negative',
        choices=['negative', 'ignore'],
        help=(
            "negative reproduces supplied CrossSDC-I; ignore removes "
            "same-class non-paired samples from its denominator."
        ),
    )

    parser.add_argument('--cmr', action='store_true', default=False)
    parser.add_argument('--lam_cmr', type=float, default=0.1)
    parser.add_argument('--cmr_temperature', type=float, default=0.1)
    parser.add_argument(
        '--cmr_mode', type=str, default='smooth', choices=['hard', 'smooth']
    )
    parser.add_argument(
        '--cmr_reserve', type=float, default=0.05,
        help='Required margin reserve above the frozen teacher margin.'
    )
    parser.add_argument(
        '--cmr_beta', type=float, default=0.05,
        help='Softplus transition scale for smooth CMR.'
    )

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dump_tsne", action="store_true", help="If set, dump t-SNE plots for each step")
    parser.add_argument("--tsne_feature", type=str, default="logits",
                        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"])
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    

    args = parser.parse_args()
    if args.cross_sdc_temperature <= 0 or args.cmr_temperature <= 0:
        parser.error('Contrastive and CMR temperatures must be positive')
    if args.lam_cross_sdc_i < 0 or args.lam_cross_sdc_c < 0 or args.lam_cmr < 0:
        parser.error('Loss coefficients must be non-negative')
    if args.cmr_reserve < 0:
        parser.error('--cmr_reserve must be non-negative')
    if args.cmr_mode == 'smooth' and args.cmr_beta <= 0:
        parser.error('--cmr_beta must be positive for smooth CMR')
    if args.cmr and not args.cross_sdc:
        parser.error('--cmr currently requires --cross_sdc')
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

    ckpts_root = './save/{}/'.format(args.dataset)
    figs_root = './save/fig/{}/'.format(args.dataset)

    # NEW: metrics root (paper-plot friendly outputs)
    metrics_root = './save/metrics/{}/'.format(args.dataset)

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
            train(args, step, train_set, val_set, exemplar_set)

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

        ckpt_path = './save/{}/step_{}_best_model.pkl'.format(args.dataset, step)

        if args.dump_tsne:
            print("Dumping t-SNE plots for step {}...".format(step))
            out_root = os.path.join(args.tsne_out_root, args.dataset)
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
