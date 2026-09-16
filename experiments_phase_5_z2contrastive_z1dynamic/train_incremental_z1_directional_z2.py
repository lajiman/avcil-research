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


# ======================================================================
# Difficulty-aware directional z1 contrastive learning
# ======================================================================

def _percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """
    Convert a 1-D tensor to [0, 1] percentile ranks.

    The operation is used only to build detached difficulty weights, so a
    simple deterministic rank is preferable to a differentiable surrogate.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1-D tensor")
    n = values.numel()
    if n <= 1:
        return torch.full_like(values, 0.5)

    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.linspace(
        0.0, 1.0, steps=n, device=values.device, dtype=values.dtype
    )
    return ranks


@torch.no_grad()
def _class_geometry_difficulty(
    features: torch.Tensor,
    labels: torch.Tensor,
    knn_k: int = 10,
    eps: float = 1e-8,
):
    """
    Estimate a detached class-wise geometry difficulty in the current
    current-task + replay mini-batch.

    Four components are converted to percentile difficulty and averaged:
      1) larger intra-class dispersion        -> harder
      2) smaller nearest-centroid distance    -> harder
      3) smaller normalized margin            -> harder
      4) smaller sample-level kNN purity      -> harder

    Returns
    -------
    unique_labels: (C,)
    difficulty:    (C,), values in [0, 1]
    """
    z = F.normalize(features.detach(), dim=1)
    labels = labels.detach().long().to(z.device)
    unique_labels, inverse = labels.unique(sorted=True, return_inverse=True)
    class_num = unique_labels.numel()
    sample_num = z.shape[0]

    if class_num <= 1:
        return unique_labels, torch.full(
            (class_num,), 0.5, device=z.device, dtype=z.dtype
        )

    centroids = []
    counts = []
    intra = []
    for class_index in range(class_num):
        mask = inverse.eq(class_index)
        class_features = z[mask]
        centroid = F.normalize(class_features.mean(dim=0, keepdim=True), dim=1)[0]
        centroids.append(centroid)
        counts.append(int(mask.sum().item()))
        intra.append((1.0 - class_features @ centroid).mean())

    centroids = torch.stack(centroids, dim=0)
    counts_tensor = torch.tensor(counts, device=z.device)
    intra = torch.stack(intra)

    # A singleton has zero dispersion only because no within-class estimate is
    # available. Replace it by the median non-singleton dispersion rather than
    # incorrectly treating the class as maximally easy.
    multi_mask = counts_tensor > 1
    if multi_mask.any():
        fallback_intra = intra[multi_mask].median()
        intra = torch.where(multi_mask, intra, fallback_intra)

    centroid_distance = 1.0 - centroids @ centroids.t()
    centroid_distance.fill_diagonal_(float("inf"))
    nearest_distance = centroid_distance.min(dim=1).values
    normalized_margin = nearest_distance / intra.clamp_min(eps)

    # Sample-level kNN purity, then aggregate within each class.
    if sample_num <= 1:
        class_purity = torch.ones(class_num, device=z.device, dtype=z.dtype)
    else:
        similarity = z @ z.t()
        similarity.fill_diagonal_(-float("inf"))
        k = min(max(int(knn_k), 1), sample_num - 1)
        neighbor_index = torch.topk(similarity, k=k, dim=1).indices
        neighbor_labels = labels[neighbor_index]
        sample_purity = neighbor_labels.eq(labels.unsqueeze(1)).float().mean(dim=1)
        class_purity = torch.stack(
            [sample_purity[inverse.eq(ci)].mean() for ci in range(class_num)]
        )

    difficulty_score = torch.stack(
        [
            _percentile_rank(intra),
            1.0 - _percentile_rank(nearest_distance),
            1.0 - _percentile_rank(normalized_margin),
            1.0 - _percentile_rank(class_purity),
        ],
        dim=0,
    ).mean(dim=0)

    # Match the analysis convention: the composite score is itself converted
    # to a final percentile.
    difficulty = _percentile_rank(difficulty_score)
    return unique_labels, difficulty


@torch.no_grad()
def build_directional_z1_weights(
    audio_feature: torch.Tensor,
    visual_feature: torch.Tensor,
    labels: torch.Tensor,
    ema_state: dict,
    ema_momentum: float = 0.9,
    knn_k: int = 10,
    weight_floor: float = 0.25,
    gap_power: float = 1.0,
    eps: float = 1e-8,
):
    """
    Build sample weights for the two directed transfers:

        audio <- visual
        visual <- audio

    For class c:
        raw_w[a<-v] = floor + |d_a-d_v|^p * d_a * (1-d_v)
        raw_w[v<-a] = floor + |d_a-d_v|^p * d_v * (1-d_a)

    The first factor measures mismatch confidence, the second marks the target
    as difficult, and the third requires the source modality to be reliable.

    We normalize the two directions jointly so their average weight is one.
    This keeps the overall z1-loss scale close to the original baseline while
    redistributing gradient direction and class emphasis.
    """
    labels = labels.long().to(audio_feature.device)

    audio_classes, audio_difficulty = _class_geometry_difficulty(
        audio_feature, labels, knn_k=knn_k, eps=eps
    )
    visual_classes, visual_difficulty = _class_geometry_difficulty(
        visual_feature, labels, knn_k=knn_k, eps=eps
    )

    if not torch.equal(audio_classes, visual_classes):
        raise RuntimeError("audio and visual class sets are inconsistent")

    momentum = float(ema_momentum)
    momentum = min(max(momentum, 0.0), 0.9999)

    audio_ema = []
    visual_ema = []
    for class_id, da, dv in zip(
        audio_classes.tolist(), audio_difficulty.tolist(), visual_difficulty.tolist()
    ):
        old = ema_state.get(int(class_id))
        if old is None:
            new_da, new_dv = float(da), float(dv)
        else:
            new_da = momentum * old[0] + (1.0 - momentum) * float(da)
            new_dv = momentum * old[1] + (1.0 - momentum) * float(dv)
        ema_state[int(class_id)] = (new_da, new_dv)
        audio_ema.append(new_da)
        visual_ema.append(new_dv)

    audio_ema = torch.tensor(
        audio_ema, device=audio_feature.device, dtype=audio_feature.dtype
    )
    visual_ema = torch.tensor(
        visual_ema, device=visual_feature.device, dtype=visual_feature.dtype
    )

    class_to_position = {
        int(class_id): position for position, class_id in enumerate(audio_classes.tolist())
    }
    sample_position = torch.tensor(
        [class_to_position[int(class_id)] for class_id in labels.tolist()],
        device=labels.device,
        dtype=torch.long,
    )

    sample_da = audio_ema[sample_position]
    sample_dv = visual_ema[sample_position]
    mismatch = (sample_da - sample_dv).abs().pow(float(gap_power))

    weight_audio_from_visual = (
        float(weight_floor) + mismatch * sample_da * (1.0 - sample_dv)
    )
    weight_visual_from_audio = (
        float(weight_floor) + mismatch * sample_dv * (1.0 - sample_da)
    )

    # Joint normalization: E[(w_a<-v + w_v<-a)/2] = 1.
    mean_weight = 0.5 * (
        weight_audio_from_visual.mean() + weight_visual_from_audio.mean()
    )
    mean_weight = mean_weight.clamp_min(eps)
    weight_audio_from_visual = weight_audio_from_visual / mean_weight
    weight_visual_from_audio = weight_visual_from_audio / mean_weight

    stats = {
        "audio_difficulty": sample_da.mean().item(),
        "visual_difficulty": sample_dv.mean().item(),
        "signed_gap": (sample_da - sample_dv).mean().item(),
        "abs_gap": mismatch.mean().item(),
        "weight_audio_from_visual": weight_audio_from_visual.mean().item(),
        "weight_visual_from_audio": weight_visual_from_audio.mean().item(),
    }
    return weight_audio_from_visual, weight_visual_from_audio, stats


def directional_instance_contrastive_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    anchor_weight: torch.Tensor,
    temperature: float = 0.1,
):
    """
    Instance matching with a detached teacher.

    Rows are student anchors. Columns are teacher candidates. The identity
    target matches the paired audio/visual sample.
    """
    student = F.normalize(student_feature, dim=1)
    teacher = F.normalize(teacher_feature.detach(), dim=1)
    logits = student @ teacher.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_per_anchor = F.cross_entropy(logits, targets, reduction="none")
    return (loss_per_anchor * anchor_weight.detach()).mean()


def directional_class_contrastive_loss(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    labels: torch.Tensor,
    anchor_weight: torch.Tensor,
    temperature: float = 0.1,
):
    """
    Class-level cross-modal contrastive loss with a detached teacher.

    The per-anchor scaling follows the original implementation:
        -mean_j log p(j|i) * 1[y_j=y_i]
    so lam_C remains comparable to the user's existing experiments.
    """
    student = F.normalize(student_feature, dim=1)
    teacher = F.normalize(teacher_feature.detach(), dim=1)
    labels = labels.long().to(student.device)

    same_class = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
    logits = student @ teacher.t() / temperature
    log_probability = F.log_softmax(logits, dim=1)
    loss_per_anchor = -(log_probability * same_class).mean(dim=1)
    return (loss_per_anchor * anchor_weight.detach()).mean()


def directional_z1_contrastive_losses(
    audio_feature: torch.Tensor,
    visual_feature: torch.Tensor,
    labels: torch.Tensor,
    args,
    ema_state: dict,
):
    """
    Return directionally weighted instance/class z1 losses and logging stats.

    audio <- visual:
        audio is the student; visual is detached and protected.

    visual <- audio:
        visual is the student; audio is detached and protected.
    """
    weight_a_from_v, weight_v_from_a, stats = build_directional_z1_weights(
        audio_feature=audio_feature,
        visual_feature=visual_feature,
        labels=labels,
        ema_state=ema_state,
        ema_momentum=args.directional_difficulty_ema,
        knn_k=args.directional_difficulty_knn_k,
        weight_floor=args.directional_weight_floor,
        gap_power=args.directional_gap_power,
    )

    instance_loss = audio_feature.sum() * 0.0
    class_loss = audio_feature.sum() * 0.0

    if args.instance_contrastive:
        loss_a_from_v = directional_instance_contrastive_loss(
            student_feature=audio_feature,
            teacher_feature=visual_feature,
            anchor_weight=weight_a_from_v,
            temperature=args.instance_contrastive_temperature,
        )
        loss_v_from_a = directional_instance_contrastive_loss(
            student_feature=visual_feature,
            teacher_feature=audio_feature,
            anchor_weight=weight_v_from_a,
            temperature=args.instance_contrastive_temperature,
        )
        instance_loss = 0.5 * (loss_a_from_v + loss_v_from_a)

    if args.class_contrastive:
        loss_a_from_v = directional_class_contrastive_loss(
            student_feature=audio_feature,
            teacher_feature=visual_feature,
            labels=labels,
            anchor_weight=weight_a_from_v,
            temperature=args.class_contrastive_temperature,
        )
        loss_v_from_a = directional_class_contrastive_loss(
            student_feature=visual_feature,
            teacher_feature=audio_feature,
            labels=labels,
            anchor_weight=weight_v_from_a,
            temperature=args.class_contrastive_temperature,
        )
        class_loss = 0.5 * (loss_a_from_v + loss_v_from_a)

    return instance_loss, class_loss, stats



def z2_similar_contrastive_loss(
    features,
    labels,
    temperature=0.1,
    hard_topk=5,
    hard_negative_weight=2.0,
    eps=1e-8,
):
    """
    Geometry-guided z2 supervised contrastive loss.

    features: (N, D), z2_fusion features. They may already be normalized;
              this function normalizes again for safety.
    labels:   (N,), global class ids.

    Main idea:
      - positives: samples from the same class in current batch + exemplar memory.
      - negatives: samples from different classes.
      - hard negatives: samples whose class centroid is among top-k nearest
        class centroids in the current mini-batch representation space.
        These negatives receive larger denominator weights.

    This uses no extra stored samples, so it is memory-friendly for CIL.
    """
    if features is None or labels is None:
        raise ValueError("features and labels must not be None")

    device = features.device
    labels = labels.long().to(device)
    z = F.normalize(features, dim=1)    # 归一化到单位向量
    n = z.shape[0]

    if n <= 1:
        return z.new_tensor(0.0)

    # pairwise logits
    logits = torch.matmul(z, z.t()) / temperature   # n * n similarity matrix
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()   # 数值稳定技巧，利用softmax的平移不变性。值得学习

    eye = torch.eye(n, device=device, dtype=torch.bool) # n * n identity matrix
    same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))    # n * n positive matrix (same matrix)
    pos_mask = same_class & (~eye)  # 在 same matrix 的基础上去掉自己本身

    pos_count = pos_mask.sum(dim=1) 
    valid_anchor = pos_count > 0    # class without more than 2 samples can not be recognized as anchor
    if valid_anchor.sum().item() == 0:
        # No class has at least two samples in this mini-batch.
        # Return a graph-connected zero to avoid breaking backward.
        return z.sum() * 0.0

    # ---------------------------------------------------------
    # Dynamic similar-class mining from in-batch class centroids
    # ---------------------------------------------------------
    unique_labels, inv = labels.unique(sorted=True, return_inverse=True)    # inv have the same dimension as label, inv[i] 是第 i 个 label 在 unique_labels 中的位置
    cnum = unique_labels.shape[0]

    if cnum <= 1 or hard_topk <= 0 or hard_negative_weight <= 1.0:
        pair_weight = torch.ones((n, n), device=device)
    else:
        centroids = []
        for ci in range(cnum):
            centroids.append(z[inv == ci].mean(dim=0))
        centroids = F.normalize(torch.stack(centroids, dim=0), dim=1)   # calculate in-batch centroids

        class_sim = torch.matmul(centroids, centroids.t())  # m * m centroids similarity matrix
        class_sim.fill_diagonal_(-1e9)
        k = min(int(hard_topk), cnum - 1)
        _, nn_idx = torch.topk(class_sim, k=k, dim=1)
        # for example:
        # class_sim =
        #     [[-1e9, 0.80, 0.20, 0.50],
        #     [0.80, -1e9, 0.60, 0.10],
        #     [0.20, 0.60, -1e9, 0.90],
        #     [0.50, 0.10, 0.90, -1e9]]
        # hard_topk = 2
        # nn_idx =
        #     [[1, 3],
        #     [0, 2],
        #     [3, 1],
        #     [2, 0]]

        similar_class = torch.zeros((cnum, cnum), device=device, dtype=torch.bool)
        similar_class.scatter_(1, nn_idx, True) # m * m hard mask matrix

        # map m * m -> n * n
        anchor_class = inv.unsqueeze(1).expand(n, n)    # 第 i 个sample 属于哪一类
        sample_class = inv.unsqueeze(0).expand(n, n)    # 第 j 个sample 属于哪一类
        hard_negative = similar_class[anchor_class, sample_class]   # m * m -> n * n
        hard_negative = hard_negative & (~same_class) & (~eye)  # 排除同类样本和自己 (hard_negative[i, j] = similar_class[class_i, class_j] and class_i != class_j and i != j)

        pair_weight = torch.ones((n, n), device=device)
        pair_weight = pair_weight + (float(hard_negative_weight) - 1.0) * hard_negative.float() # top-k, higher weight

    # remove self pairs from denominator
    pair_weight = pair_weight.masked_fill(eye, 0.0)

    exp_logits = torch.exp(logits) * pair_weight
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)  # sum_k weight(i, k) exp(sim(i, k) / τ)
    log_prob = logits - torch.log(denom)    # log_prob(i, j) = sim(i, j) / τ - log sum_k weight(i, k) exp(sim(i, k) / τ), that is log(exp(sim(i, j) / τ) /sum_k weight(i, k) exp(sim(i, k) / τ) )

    loss_per_anchor = -(pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1).float()
    loss = loss_per_anchor[valid_anchor].mean()
    return loss


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

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0

    # One EMA dictionary per incremental step. It is updated from the current
    # task + exemplar mini-batches and never receives gradients.
    directional_difficulty_ema_state = {}

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        directional_stat_sums = {
            "audio_difficulty": 0.0,
            "visual_difficulty": 0.0,
            "signed_gap": 0.0,
            "abs_gap": 0.0,
            "weight_audio_from_visual": 0.0,
            "weight_visual_from_audio": 0.0,
        }
        directional_stat_count = 0
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
                out, z2_feature, audio_feature, visual_feature = model(visual=visual, audio=audio, out_features=True, out_features_norm=True, out_feature_before_fusion=True)
                # CE_loss = CE_loss(step_out_class_num, out, labels)
                # loss = CE_loss
                loss = CE_loss(step_out_class_num, out, labels)
                if args.z2_contrastive and args.z2_contrastive_on_step0:
                    z2_contra_loss = z2_similar_contrastive_loss(
                        z2_feature, labels,
                        temperature=args.z2_contrastive_temperature,
                        hard_topk=args.z2_hard_topk,
                        hard_negative_weight=args.z2_hard_negative_weight,
                    )
                    loss += args.lam_Z2 * z2_contra_loss
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
                out, z2_feature, audio_feature, visual_feature, spatial_attn_score, temporal_attn_score = model(visual=total_visual, audio=total_audio, out_features=True, out_features_norm=True, out_feature_before_fusion=True, out_attn_score=True)
                with torch.no_grad():
                    old_out, old_spatial_attn_score, old_temporal_attn_score = old_model(visual=total_visual, audio=total_audio, out_attn_score=True)
                    old_out = old_out.detach()
                    old_spatial_attn_score = old_spatial_attn_score.detach()
                    old_temporal_attn_score = old_temporal_attn_score.detach()
                
                all_labels = torch.cat((labels, exemplar_labels))

                use_directional_z1 = (
                    args.directional_z1_contrastive
                    and step >= args.directional_z1_start_step
                    and (args.instance_contrastive or args.class_contrastive)
                )

                if use_directional_z1:
                    instance_contra_loss, class_contra_loss, directional_stats = (
                        directional_z1_contrastive_losses(
                            audio_feature=audio_feature,
                            visual_feature=visual_feature,
                            labels=all_labels,
                            args=args,
                            ema_state=directional_difficulty_ema_state,
                        )
                    )
                    for key in directional_stat_sums:
                        directional_stat_sums[key] += directional_stats[key]
                    directional_stat_count += 1
                else:
                    if args.instance_contrastive:
                        instance_contra_loss = cal_contrastive_loss(
                            audio_feature,
                            visual_feature,
                            temperature=args.instance_contrastive_temperature,
                        )

                    if args.class_contrastive:
                        class_contra_loss = class_contrastive_loss(
                            audio_feature,
                            visual_feature,
                            all_labels,
                            temperature=args.class_contrastive_temperature,
                        )

                if args.z2_contrastive and step >= args.z2_contrastive_start_step:
                    z2_contra_loss = z2_similar_contrastive_loss(
                        z2_feature, all_labels,
                        temperature=args.z2_contrastive_temperature,
                        hard_topk=args.z2_hard_topk,
                        hard_negative_weight=args.z2_hard_negative_weight,
                    )
                
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
                if args.z2_contrastive and step >= args.z2_contrastive_start_step:
                    loss += args.lam_Z2 * z2_contra_loss
                if args.attn_score_distil:
                    loss += args.lam * spatial_attn_dist_loss + (1 - args.lam) * temporal_attn_dist_loss
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        train_log = 'Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss)
        if directional_stat_count > 0:
            train_log += (
                ' d_a:{:.3f} d_v:{:.3f} |gap|:{:.3f}'
                ' w_a<-v:{:.3f} w_v<-a:{:.3f}'
            ).format(
                directional_stat_sums["audio_difficulty"] / directional_stat_count,
                directional_stat_sums["visual_difficulty"] / directional_stat_count,
                directional_stat_sums["abs_gap"] / directional_stat_count,
                directional_stat_sums["weight_audio_from_visual"] / directional_stat_count,
                directional_stat_sums["weight_visual_from_audio"] / directional_stat_count,
            )
        print(train_log, flush=True)

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
    parser.add_argument('--lam_Z2', type=float, default=0.1, help='Weight for geometry-guided z2 contrastive loss')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--class_num_per_step', type=int, default=7)

    parser.add_argument('--memory_size', type=int, default=340)

    parser.add_argument('--instance_contrastive', action='store_true', default=False)
    parser.add_argument('--class_contrastive', action='store_true', default=False)
    parser.add_argument('--attn_score_distil', action='store_true', default=False)
    parser.add_argument('--z2_contrastive', action='store_true', default=False, help='Enable geometry-guided z2 contrastive replay')
    parser.add_argument('--z2_contrastive_on_step0', action='store_true', default=False, help='Also apply z2 contrastive loss at step 0')
    parser.add_argument('--z2_contrastive_start_step', type=int, default=1, help='First incremental step to apply z2 contrastive loss')

    parser.add_argument(
        '--directional_z1_contrastive',
        action='store_true',
        default=False,
        help='Replace the original z1 contrastive losses by difficulty-aware directional losses with detached teachers',
    )
    parser.add_argument(
        '--directional_z1_start_step',
        type=int,
        default=1,
        help='First incremental step to use directional z1 contrastive learning',
    )
    parser.add_argument(
        '--directional_weight_floor',
        type=float,
        default=0.25,
        help='Residual raw weight shared by the two directions before joint normalization',
    )
    parser.add_argument(
        '--directional_gap_power',
        type=float,
        default=1.0,
        help='Exponent applied to |audio difficulty - visual difficulty|',
    )
    parser.add_argument(
        '--directional_difficulty_knn_k',
        type=int,
        default=10,
        help='k used by the mini-batch class-geometry difficulty estimator',
    )
    parser.add_argument(
        '--directional_difficulty_ema',
        type=float,
        default=0.9,
        help='EMA momentum for class-wise audio/visual difficulty within an incremental step',
    )

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--z2_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--z2_hard_topk', type=int, default=5, help='Number of dynamic similar classes used as hard negatives')
    parser.add_argument('--z2_hard_negative_weight', type=float, default=2.0, help='Denominator weight for samples from similar classes')

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dump_tsne", action="store_true", help="If set, dump t-SNE plots for each step")
    parser.add_argument("--tsne_feature", type=str, default="logits",
                        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"])
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    

    args = parser.parse_args()
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