import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental_continual_guidance_v3 import IncreAudioVisualNet
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


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_full_model(path, map_location=None):
    """Load a trusted full-model checkpoint across old and new PyTorch."""
    try:
        return torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=map_location)


def _per_sample_attention_kl(student_prob, teacher_prob, eps=1e-8):
    """KL(teacher || student), reduced over every non-batch dimension."""
    student_prob = student_prob.clamp_min(eps)
    teacher_prob = teacher_prob.clamp_min(eps)
    kl = teacher_prob * (teacher_prob.log() - student_prob.log())
    return kl.reshape(kl.shape[0], -1).sum(dim=1)


def weighted_attention_distillation(
    student_spatial,
    teacher_spatial,
    student_temporal,
    teacher_temporal,
    sample_weights,
):
    spatial_kl = _per_sample_attention_kl(student_spatial, teacher_spatial)
    temporal_kl = _per_sample_attention_kl(student_temporal, teacher_temporal)

    sample_weights = sample_weights.to(spatial_kl.device, spatial_kl.dtype)
    denominator = sample_weights.sum().clamp_min(1e-8)
    spatial_loss = (sample_weights * spatial_kl).sum() / denominator
    temporal_loss = (sample_weights * temporal_kl).sum() / denominator
    return spatial_loss, temporal_loss


def _append_capped_features(bank, labels, outputs, max_per_class):
    keys = {
        "audio": "z1_audio_norm",
        "visual": "z1_visual_uniform_norm",
        "z2": "z2_guided_reference_norm",
    }
    labels = labels.detach().cpu().long()

    for row_idx, class_id in enumerate(labels.tolist()):
        class_bank = bank.setdefault(
            class_id, {"audio": [], "visual": [], "z2": []}
        )
        if len(class_bank["audio"]) >= max_per_class:
            continue
        for short_key, output_key in keys.items():
            class_bank[short_key].append(
                outputs[output_key][row_idx].detach().cpu()
            )


def collect_guidance_feature_bank(
    model,
    train_data_set,
    exemplar_set,
    step,
    args,
):
    """
    Collect a class-balanced feature bank from current training data and old
    exemplars. Visual reliability is measured on the uniform z1 branch, so it
    is independent of the current audio-guided gate.
    """
    core_model = unwrap_model(model)
    was_training = core_model.training
    core_model.eval()

    bank = {}
    datasets = [train_data_set]
    if step > 0 and exemplar_set is not None and exemplar_set.__len__() > 0:
        datasets.append(exemplar_set)

    generator = torch.Generator()
    generator.manual_seed(args.seed + 1009 * step)

    with torch.no_grad():
        for dataset in datasets:
            loader = DataLoader(
                dataset,
                batch_size=min(args.infer_batch_size, dataset.__len__()),
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                shuffle=True,
                generator=generator,
            )
            for data, labels in loader:
                visual = data[0].to(device)
                audio = data[1].to(device)
                outputs = core_model(
                    visual=visual,
                    audio=audio,
                    return_dict=True,
                    out_analysis_features=True,
                )
                _append_capped_features(
                    bank,
                    labels,
                    outputs,
                    args.guidance_max_samples_per_class,
                )

    if was_training:
        core_model.train()

    audio_rows, visual_rows, z2_rows, labels_rows = [], [], [], []
    num_seen_classes = (step + 1) * args.class_num_per_step
    for class_id in range(num_seen_classes):
        class_bank = bank.get(class_id)
        if class_bank is None or len(class_bank["audio"]) == 0:
            continue
        count = len(class_bank["audio"])
        audio_rows.extend(class_bank["audio"])
        visual_rows.extend(class_bank["visual"])
        z2_rows.extend(class_bank["z2"])
        labels_rows.extend([class_id] * count)

    if len(labels_rows) == 0:
        raise RuntimeError("No samples were collected for guidance statistics")

    return {
        "audio": torch.stack(audio_rows, dim=0),
        "visual": torch.stack(visual_rows, dim=0),
        "z2": torch.stack(z2_rows, dim=0),
        "labels": torch.tensor(labels_rows, dtype=torch.long),
    }


def compute_class_geometry(
    features,
    labels,
    num_classes,
    knn_k,
    eps,
    compute_purity=True,
    margin_clip=100.0,
    purity_alpha=1.0,
):
    """
    Class geometry in a normalized cosine space:

        S_c = mean_i [1 - z_i^T mu_c]
        B_c = min_{c' != c} [1 - mu_c^T mu_c']
        M_c = B_c / (S_c + eps)
        D_c = S_c / (B_c + eps)

    Standard kNN purity uses the same k for every sample, up to N-1. A
    symmetric Beta(alpha, alpha) posterior mean is used for class purity:

        P_c = (same-neighbour count + alpha) /
              (total-neighbour count + 2 alpha).

    This prevents accidental zero purity from creating an infinite log ratio.
    """
    features = F.normalize(features.float(), dim=1)
    labels = labels.long()
    feat_device = features.device

    centroids = torch.zeros(
        num_classes, features.shape[1], device=feat_device, dtype=features.dtype
    )
    dispersion = torch.zeros(num_classes, device=feat_device)
    counts = torch.zeros(num_classes, device=feat_device)
    valid = torch.zeros(num_classes, device=feat_device, dtype=torch.bool)

    class_indices = []
    for class_id in range(num_classes):
        idx = torch.where(labels == class_id)[0]
        class_indices.append(idx)
        if idx.numel() == 0:
            continue
        valid[class_id] = True
        counts[class_id] = float(idx.numel())
        class_features = features.index_select(0, idx)
        centroid = F.normalize(class_features.mean(dim=0), dim=0)
        centroids[class_id] = centroid
        dispersion[class_id] = (
            1.0 - torch.matmul(class_features, centroid)
        ).mean()

    inter_distance = torch.zeros(num_classes, device=feat_device)
    valid_ids = torch.where(valid)[0]
    if valid_ids.numel() > 1:
        valid_centroids = centroids.index_select(0, valid_ids)
        centroid_distance = 1.0 - torch.matmul(
            valid_centroids, valid_centroids.t()
        )
        centroid_distance.fill_diagonal_(float("inf"))
        inter_distance.index_copy_(
            0, valid_ids, centroid_distance.min(dim=1).values
        )
    elif valid_ids.numel() == 1:
        inter_distance[valid_ids[0]] = 1.0

    margin_raw = inter_distance / (dispersion + eps)
    margin = margin_raw.clamp(min=0.0, max=margin_clip)
    difficulty = dispersion / (inter_distance + eps)

    sample_purity = torch.zeros(features.shape[0], device=feat_device)
    class_purity = torch.zeros(num_classes, device=feat_device)
    same_counts = torch.zeros(num_classes, device=feat_device)
    neighbour_counts = torch.zeros(num_classes, device=feat_device)

    if compute_purity and features.shape[0] > 1:
        k_eff = min(int(knn_k), features.shape[0] - 1)
        similarity = torch.matmul(features, features.t())
        similarity.fill_diagonal_(-float("inf"))
        neighbor_ids = similarity.topk(k_eff, dim=1).indices
        neighbor_labels = labels.index_select(
            0, neighbor_ids.reshape(-1)
        ).view_as(neighbor_ids)
        sample_purity = (
            neighbor_labels == labels.unsqueeze(1)
        ).float().mean(dim=1)

        for class_id, idx in enumerate(class_indices):
            if idx.numel() == 0:
                continue
            same = (
                neighbor_labels.index_select(0, idx) == class_id
            ).float().sum()
            total = float(idx.numel() * k_eff)
            same_counts[class_id] = same
            neighbour_counts[class_id] = total
            class_purity[class_id] = (
                same + purity_alpha
            ) / (total + 2.0 * purity_alpha)
    else:
        class_purity[valid] = 1.0

    return {
        "margin": margin,
        "margin_raw": margin_raw,
        "difficulty": difficulty,
        "purity": class_purity,
        "sample_purity": sample_purity,
        "same_counts": same_counts,
        "neighbour_counts": neighbour_counts,
        "counts": counts,
        "valid": valid,
        "class_indices": class_indices,
    }


def _bootstrap_evidence(
    audio_features,
    visual_features,
    labels,
    audio_sample_purity,
    visual_sample_purity,
    num_classes,
    num_bootstraps,
    eps,
    seed,
    purity_alpha=1.0,
    knn_k=5,
):
    """
    Approximate Var(E_hat_c) by stratified nonparametric bootstrap.

    Centroid geometry is recomputed for every draw. Local sample purity is
    computed once on the full bank and then re-aggregated with the same Beta
    smoothing. E_hat is not clipped before variance estimation.
    """
    if num_bootstraps <= 1:
        return torch.zeros(num_classes, device=audio_features.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    class_indices = [
        torch.where(labels == class_id)[0] for class_id in range(num_classes)
    ]
    all_evidence = []

    for _ in range(num_bootstraps):
        sampled_indices = []
        sampled_labels = []
        for class_id, idx in enumerate(class_indices):
            if idx.numel() == 0:
                continue
            draw_cpu = torch.randint(
                low=0,
                high=idx.numel(),
                size=(idx.numel(),),
                generator=generator,
            )
            draw = draw_cpu.to(idx.device)
            sampled_indices.append(idx.index_select(0, draw))
            sampled_labels.append(
                torch.full(
                    (idx.numel(),),
                    class_id,
                    device=labels.device,
                    dtype=torch.long,
                )
            )

        sampled_indices = torch.cat(sampled_indices, dim=0)
        sampled_labels = torch.cat(sampled_labels, dim=0)
        sampled_audio = audio_features.index_select(0, sampled_indices)
        sampled_visual = visual_features.index_select(0, sampled_indices)

        audio_geo = compute_class_geometry(
            sampled_audio,
            sampled_labels,
            num_classes,
            knn_k=1,
            eps=eps,
            compute_purity=False,
        )
        visual_geo = compute_class_geometry(
            sampled_visual,
            sampled_labels,
            num_classes,
            knn_k=1,
            eps=eps,
            compute_purity=False,
        )

        audio_purity = torch.zeros(num_classes, device=labels.device)
        visual_purity = torch.zeros(num_classes, device=labels.device)
        k_eff = min(int(knn_k), labels.numel() - 1)

        for class_id in range(num_classes):
            mask = sampled_labels == class_id
            n = int(mask.sum().item())
            if n == 0:
                continue
            original_ids = sampled_indices[mask]

            a_success = (
                audio_sample_purity.index_select(0, original_ids).sum()
                * float(k_eff)
            )
            v_success = (
                visual_sample_purity.index_select(0, original_ids).sum()
                * float(k_eff)
            )
            total = float(n * k_eff)
            audio_purity[class_id] = (
                a_success + purity_alpha
            ) / (total + 2.0 * purity_alpha)
            visual_purity[class_id] = (
                v_success + purity_alpha
            ) / (total + 2.0 * purity_alpha)

        audio_reliability = audio_geo["margin"] * audio_purity
        visual_reliability = visual_geo["margin"] * visual_purity
        evidence = torch.relu(
            torch.log(
                (visual_reliability + eps)
                / (audio_reliability + eps)
            )
        )
        all_evidence.append(evidence)

    stacked = torch.stack(all_evidence, dim=0)
    return stacked.var(dim=0, unbiased=False)



def guidance_prior_path(args, step):
    return os.path.join(
        './save/{}/'.format(args.dataset),
        'step_{}_guidance_prior.pt'.format(step),
    )


def _new_empty_guidance_state(num_classes, state_device):
    return {
        'energy': torch.zeros(num_classes, device=state_device),
        'variance': torch.ones(num_classes, device=state_device),
        'z2_difficulty': torch.zeros(num_classes, device=state_device),
        'initialized': torch.zeros(
            num_classes, device=state_device, dtype=torch.bool
        ),
    }


def initialize_step_guidance(
    args, model, step, num_seen_classes, activate_forward
):
    """
    Load and freeze the consolidated posterior of step t-1.

    The returned tensors are never mutated during step t. Every periodic
    refresh is recomputed from this same fixed prior, so epochs inside one
    incremental step are not treated as independent Kalman time steps.
    """
    core_model = unwrap_model(model)
    state_device = core_model.guidance_energy.device
    fixed_prior = _new_empty_guidance_state(num_seen_classes, state_device)

    if step > 0:
        expected_old_classes = step * args.class_num_per_step
        prior_file = guidance_prior_path(args, step - 1)
        if not os.path.exists(prior_file):
            raise FileNotFoundError(
                'Missing consolidated guidance prior: {}. '
                'Run the previous incremental step with the v3 training '
                'script before starting step {}.'.format(prior_file, step)
            )

        payload = torch.load(prior_file, map_location=state_device)
        if int(payload.get('step', -1)) != step - 1:
            raise ValueError(
                'Guidance prior {} belongs to step {}, expected step {}.'.format(
                    prior_file, payload.get('step'), step - 1
                )
            )
        if int(payload.get('num_classes', -1)) != expected_old_classes:
            raise ValueError(
                'Guidance prior {} has {} classes, expected {}.'.format(
                    prior_file,
                    payload.get('num_classes'),
                    expected_old_classes,
                )
            )

        for key in ('energy', 'variance', 'z2_difficulty', 'initialized'):
            value = payload[key].to(state_device)
            if value.ndim != 1 or value.numel() != expected_old_classes:
                raise ValueError(
                    '{} in {} has shape {}, expected ({},).'.format(
                        key, prior_file, tuple(value.shape), expected_old_classes
                    )
                )
            fixed_prior[key][:expected_old_classes] = value

        fixed_prior['variance'][:expected_old_classes].clamp_min_(
            args.guidance_eps
        )
        fixed_prior['initialized'] = fixed_prior['initialized'].bool()

    # Before observing the current step, old classes use the consolidated
    # prior and receive full raw-attention distillation. New classes start
    # from E=0 (g=1) and have no teacher attention.
    initial_prior_weight = fixed_prior['initialized'].float()
    core_model.set_guidance_state(
        energy=fixed_prior['energy'],
        variance=fixed_prior['variance'],
        z2_difficulty=fixed_prior['z2_difficulty'],
        initialized=fixed_prior['initialized'],
        prior_weight=initial_prior_weight,
    )

    core_model.set_guidance_forward_active(activate_forward)

    return {
        key: value.detach().clone()
        for key, value in fixed_prior.items()
    }


def estimate_guidance_observation(
    args,
    model,
    step,
    train_data_set,
    exemplar_set,
):
    """Estimate current-step evidence and its measurement uncertainty."""
    num_seen_classes = (step + 1) * args.class_num_per_step
    feature_bank = collect_guidance_feature_bank(
        model=model,
        train_data_set=train_data_set,
        exemplar_set=exemplar_set,
        step=step,
        args=args,
    )

    audio_features = feature_bank['audio'].to(device)
    visual_features = feature_bank['visual'].to(device)
    z2_features = feature_bank['z2'].to(device)
    labels = feature_bank['labels'].to(device)

    geo_kwargs = dict(
        num_classes=num_seen_classes,
        knn_k=args.guidance_knn_k,
        eps=args.guidance_eps,
        margin_clip=args.guidance_margin_clip,
        purity_alpha=args.guidance_purity_alpha,
    )
    audio_geo = compute_class_geometry(
        audio_features, labels, compute_purity=True, **geo_kwargs
    )
    visual_geo = compute_class_geometry(
        visual_features, labels, compute_purity=True, **geo_kwargs
    )
    z2_geo = compute_class_geometry(
        z2_features, labels, compute_purity=False, **geo_kwargs
    )

    audio_reliability = audio_geo['margin'] * audio_geo['purity']
    visual_reliability = visual_geo['margin'] * visual_geo['purity']
    evidence_raw = torch.relu(
        torch.log(
            (visual_reliability + args.guidance_eps)
            / (audio_reliability + args.guidance_eps)
        )
    )

    # The resampling indices are deliberately fixed for all refresh epochs in
    # one step. Changes in the estimate then reflect representation evolution,
    # rather than changing Monte-Carlo draws.
    bootstrap_var = _bootstrap_evidence(
        audio_features=audio_features,
        visual_features=visual_features,
        labels=labels,
        audio_sample_purity=audio_geo['sample_purity'],
        visual_sample_purity=visual_geo['sample_purity'],
        num_classes=num_seen_classes,
        num_bootstraps=args.guidance_bootstrap_samples,
        eps=args.guidance_eps,
        seed=args.seed + 7919 * (step + 1),
        purity_alpha=args.guidance_purity_alpha,
        knn_k=args.guidance_knn_k,
    )
    counts = torch.minimum(
        audio_geo['counts'], visual_geo['counts']
    ).clamp_min(1.0)
    measurement_var = (
        bootstrap_var + args.guidance_r0 / counts
    ).clamp_min(args.guidance_eps)

    observation = {
        'evidence_raw': evidence_raw,
        'measurement_variance': measurement_var,
        'z2_difficulty': z2_geo['difficulty'].detach().clone(),
        'valid': audio_geo['valid'] & visual_geo['valid'] & z2_geo['valid'],
        'counts': counts,
        'audio_margin': audio_geo['margin'],
        'visual_margin': visual_geo['margin'],
        'audio_purity': audio_geo['purity'],
        'visual_purity': visual_geo['purity'],
        'audio_reliability': audio_reliability,
        'visual_reliability': visual_reliability,
    }

    del audio_features, visual_features, z2_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return observation


def posterior_from_fixed_prior(args, fixed_prior, observation):
    """
    Compute one step-level posterior from a fixed previous-step prior.

    Repeated calls in different epochs do *not* feed the prior call's
    posterior into the next call. This avoids repeated counting of the same
    incremental-step data.
    """
    old_energy = fixed_prior['energy']
    old_variance = fixed_prior['variance'].clamp_min(args.guidance_eps)
    old_z2 = fixed_prior['z2_difficulty']
    initialized = fixed_prior['initialized'].bool()

    current_z2 = observation['z2_difficulty']
    evidence_raw = observation['evidence_raw']
    measurement_var = observation['measurement_variance'].clamp_min(
        args.guidance_eps
    )
    valid = observation['valid']

    posterior_energy = old_energy.clone()
    posterior_variance = old_variance.clone()
    posterior_z2 = old_z2.clone()
    posterior_initialized = initialized.clone()
    prior_weight = initialized.float().clone()
    kalman_gain = torch.zeros_like(old_energy)
    process_variance = torch.zeros_like(old_energy)
    z2_log_change = torch.zeros_like(old_energy)

    for class_id in range(old_energy.numel()):
        if not bool(valid[class_id].item()):
            continue

        current_evidence = evidence_raw[class_id].clamp(
            min=0.0, max=args.guidance_energy_max
        )
        posterior_z2[class_id] = current_z2[class_id]

        # Classes first introduced in this step have no previous-step prior.
        # Each refresh re-estimates them directly from the current model.
        if not bool(initialized[class_id].item()):
            posterior_energy[class_id] = current_evidence
            posterior_variance[class_id] = measurement_var[class_id]
            posterior_initialized[class_id] = True
            prior_weight[class_id] = 0.0
            kalman_gain[class_id] = 1.0
            continue

        change = torch.abs(
            torch.log(
                (current_z2[class_id] + args.guidance_eps)
                / (old_z2[class_id] + args.guidance_eps)
            )
        ).clamp(max=args.guidance_z2_log_change_max)
        q_value = args.guidance_q0 + args.guidance_q1 * change
        p_minus = (old_variance[class_id] + q_value).clamp_min(
            args.guidance_eps
        )
        k_value = p_minus / (p_minus + measurement_var[class_id])

        posterior_energy[class_id] = (
            (1.0 - k_value) * old_energy[class_id]
            + k_value * current_evidence
        ).clamp(min=0.0, max=args.guidance_energy_max)
        posterior_variance[class_id] = (
            (1.0 - k_value) * p_minus
        ).clamp_min(args.guidance_eps)
        prior_weight[class_id] = 1.0 - k_value
        kalman_gain[class_id] = k_value
        process_variance[class_id] = q_value
        z2_log_change[class_id] = change

    state = {
        'energy': posterior_energy,
        'variance': posterior_variance,
        'z2_difficulty': posterior_z2,
        'initialized': posterior_initialized,
        'prior_weight': prior_weight,
    }
    diagnostics = {
        'kalman_gain': kalman_gain,
        'process_variance': process_variance,
        'z2_log_change': z2_log_change,
    }
    return state, diagnostics


def apply_active_guidance_state(model, state):
    core_model = unwrap_model(model)
    core_model.set_guidance_state(
        energy=state['energy'],
        variance=state['variance'],
        z2_difficulty=state['z2_difficulty'],
        initialized=state['initialized'],
        prior_weight=state['prior_weight'],
    )


def write_guidance_diagnostics(
    args,
    step,
    epoch,
    phase,
    fixed_prior,
    observation,
    state,
    diagnostics,
):
    num_seen_classes = state['energy'].numel()
    gates = torch.exp(-state['energy']).clamp(
        min=args.guidance_gate_min, max=1.0
    )
    rows = []
    for class_id in range(num_seen_classes):
        rows.append({
            'step': step,
            'epoch': epoch,
            'phase': phase,
            'class_id': class_id,
            'count': float(observation['counts'][class_id].item()),
            'prior_initialized': int(
                fixed_prior['initialized'][class_id].item()
            ),
            'prior_energy': float(fixed_prior['energy'][class_id].item()),
            'prior_variance': float(fixed_prior['variance'][class_id].item()),
            'prior_z2_difficulty': float(
                fixed_prior['z2_difficulty'][class_id].item()
            ),
            'audio_margin': float(
                observation['audio_margin'][class_id].item()
            ),
            'visual_margin': float(
                observation['visual_margin'][class_id].item()
            ),
            'audio_purity': float(
                observation['audio_purity'][class_id].item()
            ),
            'visual_purity': float(
                observation['visual_purity'][class_id].item()
            ),
            'audio_reliability': float(
                observation['audio_reliability'][class_id].item()
            ),
            'visual_reliability': float(
                observation['visual_reliability'][class_id].item()
            ),
            'evidence_raw': float(
                observation['evidence_raw'][class_id].item()
            ),
            'measurement_variance': float(
                observation['measurement_variance'][class_id].item()
            ),
            'z2_difficulty': float(
                observation['z2_difficulty'][class_id].item()
            ),
            'z2_log_change': float(
                diagnostics['z2_log_change'][class_id].item()
            ),
            'process_variance': float(
                diagnostics['process_variance'][class_id].item()
            ),
            'kalman_gain': float(
                diagnostics['kalman_gain'][class_id].item()
            ),
            'prior_weight': float(state['prior_weight'][class_id].item()),
            'posterior_energy': float(state['energy'][class_id].item()),
            'posterior_variance': float(state['variance'][class_id].item()),
            'gate': float(gates[class_id].item()),
            'posterior_initialized': int(
                state['initialized'][class_id].item()
            ),
        })

    metrics_root = './save/metrics/{}/'.format(args.dataset)
    csv_path = os.path.join(
        metrics_root, 'guidance_updates_step_{}.csv'.format(step)
    )
    append_csv_rows(csv_path, list(rows[0].keys()), rows)

    safe_phase = phase.replace(' ', '_')
    save_json(
        {
            'step': step,
            'epoch': epoch,
            'phase': phase,
            'gate_mean': float(gates.mean().item()),
            'gate_min': float(gates.min().item()),
            'gate_max': float(gates.max().item()),
            'classes': rows,
        },
        os.path.join(
            metrics_root,
            'guidance_{}_step_{}_epoch_{}.json'.format(
                safe_phase, step, epoch
            ),
        ),
    )


def refresh_active_guidance_state(
    args,
    model,
    step,
    epoch,
    train_data_set,
    exemplar_set,
    fixed_prior,
):
    observation = estimate_guidance_observation(
        args=args,
        model=model,
        step=step,
        train_data_set=train_data_set,
        exemplar_set=exemplar_set,
    )
    state, diagnostics = posterior_from_fixed_prior(
        args=args,
        fixed_prior=fixed_prior,
        observation=observation,
    )
    apply_active_guidance_state(model, state)
    write_guidance_diagnostics(
        args=args,
        step=step,
        epoch=epoch,
        phase='refresh',
        fixed_prior=fixed_prior,
        observation=observation,
        state=state,
        diagnostics=diagnostics,
    )

    gates = unwrap_model(model).class_guidance_gates()
    print(
        'Guidance refresh step {} epoch {}: gate mean={:.6f}, '
        'min={:.6f}, max={:.6f}, E mean={:.6f}, K mean={:.6f}'.format(
            step,
            epoch,
            gates.mean().item(),
            gates.min().item(),
            gates.max().item(),
            state['energy'].mean().item(),
            diagnostics['kalman_gain'].mean().item(),
        ),
        flush=True,
    )


def consolidate_best_checkpoint_prior(
    args,
    step,
    best_epoch,
    best_model_path,
    train_data_set,
    exemplar_set,
    fixed_prior,
):
    """
    Re-estimate the step-t posterior from the exact best checkpoint and save it
    separately for step t+1. The best model itself is not modified, so its
    reported validation/test result remains tied to the gate actually used at
    that epoch.
    """
    best_model = load_full_model(best_model_path, map_location=device)
    best_model = best_model.to(device)
    best_model.eval()
    fixed_prior = {
        key: value.to(device) for key, value in fixed_prior.items()
    }

    observation = estimate_guidance_observation(
        args=args,
        model=best_model,
        step=step,
        train_data_set=train_data_set,
        exemplar_set=exemplar_set,
    )
    state, diagnostics = posterior_from_fixed_prior(
        args=args,
        fixed_prior=fixed_prior,
        observation=observation,
    )

    prior_file = guidance_prior_path(args, step)
    payload = {
        'version': 4,
        'step': step,
        'best_epoch': int(best_epoch),
        'num_classes': int((step + 1) * args.class_num_per_step),
        'energy': state['energy'].detach().cpu(),
        'variance': state['variance'].detach().cpu(),
        'z2_difficulty': state['z2_difficulty'].detach().cpu(),
        'initialized': state['initialized'].detach().cpu(),
    }
    torch.save(payload, prior_file)

    write_guidance_diagnostics(
        args=args,
        step=step,
        epoch=best_epoch,
        phase='best_prior',
        fixed_prior=fixed_prior,
        observation=observation,
        state=state,
        diagnostics=diagnostics,
    )

    gates = torch.exp(-state['energy']).clamp(
        min=args.guidance_gate_min, max=1.0
    )
    print(
        'Saved consolidated guidance prior from best epoch {} to {} '
        '(gate mean={:.6f}, min={:.6f}, max={:.6f})'.format(
            best_epoch,
            prior_file,
            gates.mean().item(),
            gates.min().item(),
            gates.max().item(),
        ),
        flush=True,
    )

    del best_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(args, step, train_data_set, val_data_set, exemplar_set):
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
        previous_best = './save/{}/step_{}_best_model.pkl'.format(
            args.dataset, step - 1
        )
        model = load_full_model(previous_best)
        model.incremental_classifier(step_out_class_num)
        old_model = load_full_model(previous_best)

        exemplar_loader = DataLoader(
            exemplar_set,
            batch_size=min(
                args.exemplar_batch_size, exemplar_set.__len__()
            ),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        last_step_out_class_num = step * args.class_num_per_step

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if old_model is not None:
            old_model = nn.DataParallel(old_model)

    model = model.to(device)
    if old_model is not None:
        old_model = old_model.to(device)
        old_model.eval()

    guidance_training_active = (
        args.continual_guidance and step >= args.guidance_start_step
    )

    fixed_prior = None
    if args.continual_guidance:
        fixed_prior = initialize_step_guidance(
            args=args,
            model=model,
            step=step,
            num_seen_classes=step_out_class_num,
            activate_forward=guidance_training_active,
        )
        gates_now = unwrap_model(model).class_guidance_gates().detach()
        print(
            'Initial class gates at step {}: active={}, mean={:.6f}, '
            'min={:.6f}, max={:.6f}'.format(
                step,
                guidance_training_active,
                gates_now.mean().item(),
                gates_now.min().item(),
                gates_now.max().item(),
            ),
            flush=True,
        )

        guidance_csv = os.path.join(
            './save/metrics/{}/'.format(args.dataset),
            'guidance_updates_step_{}.csv'.format(step),
        )
        if os.path.exists(guidance_csv):
            os.remove(guidance_csv)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loss_list = []
    val_acc_list = []
    best_val_res = -float('inf')
    best_epoch = None
    best_model_path = './save/{}/step_{}_best_model.pkl'.format(
        args.dataset, step
    )

    for epoch in range(args.max_epoches):
        should_refresh = (
            guidance_training_active
            and epoch >= args.guidance_warmup_epochs
            and (
                (epoch - args.guidance_warmup_epochs)
                % args.guidance_refresh_interval
                == 0
            )
        )
        if should_refresh:
            refresh_active_guidance_state(
                args=args,
                model=model,
                step=step,
                epoch=epoch,
                train_data_set=train_data_set,
                exemplar_set=exemplar_set,
                fixed_prior=fixed_prior,
            )

        train_loss = 0.0
        num_steps = 0
        model.train()
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
                labels_ = labels % args.class_num_per_step

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device)

                data_batch_size = labels_.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                visual = data[0]
                audio = data[1]
                exemplar_visual = exemplar_data[0]
                exemplar_audio = exemplar_data[1]
                total_visual = torch.cat((visual, exemplar_visual)).to(device)
                total_audio = torch.cat((audio, exemplar_audio)).to(device)

                (
                    out,
                    audio_feature,
                    visual_feature,
                    spatial_attn_score,
                    temporal_attn_score,
                ) = model(
                    visual=total_visual,
                    audio=total_audio,
                    out_feature_before_fusion=True,
                    out_attn_score=True,
                )

                with torch.no_grad():
                    (
                        old_out,
                        old_spatial_attn_score,
                        old_temporal_attn_score,
                    ) = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        out_attn_score=True,
                    )
                    old_out = old_out.detach()
                    old_spatial_attn_score = old_spatial_attn_score.detach()
                    old_temporal_attn_score = old_temporal_attn_score.detach()

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
                    exem_slice = slice(
                        data_batch_size,
                        data_batch_size + exemplar_data_batch_size,
                    )
                    if guidance_training_active:
                        prior_weights = unwrap_model(
                            model
                        ).prior_weights_for_labels(exemplar_labels)
                    else:
                        prior_weights = torch.ones(
                            exemplar_data_batch_size, device=device
                        )

                    (
                        spatial_attn_dist_loss,
                        temporal_attn_dist_loss,
                    ) = weighted_attention_distillation(
                        student_spatial=spatial_attn_score[exem_slice],
                        teacher_spatial=old_spatial_attn_score[exem_slice],
                        student_temporal=temporal_attn_score[exem_slice],
                        teacher_temporal=old_temporal_attn_score[exem_slice],
                        sample_weights=prior_weights,
                    )

                old_out = old_out[:, :last_step_out_class_num]

                curr_out = out[
                    :data_batch_size, last_step_out_class_num:
                ]
                loss_curr = CE_loss(
                    args.class_num_per_step, curr_out, labels_
                )

                prev_out = out[
                    data_batch_size:
                    data_batch_size + exemplar_data_batch_size,
                    :last_step_out_class_num,
                ]
                loss_prev = CE_loss(
                    last_step_out_class_num, prev_out, exemplar_labels
                )

                loss_CE = (
                    loss_curr * data_batch_size
                    + loss_prev * exemplar_data_batch_size
                ) / (data_batch_size + exemplar_data_batch_size)

                if (
                    args.dataset == 'AVE'
                    and args.class_num_per_step == 4
                    and step == 1
                ):
                    loss_CE = CE_loss(
                        args.class_num_per_step + last_step_out_class_num,
                        out,
                        torch.cat((labels, exemplar_labels)),
                    )

                loss_KD = torch.zeros(step, device=device)
                for task_id in range(step):
                    start = task_id * args.class_num_per_step
                    end = (task_id + 1) * args.class_num_per_step
                    soft_target = F.softmax(
                        old_out[:, start:end] / T, dim=1
                    )
                    output_log = F.log_softmax(
                        out[:, start:end] / T, dim=1
                    )
                    loss_KD[task_id] = F.kl_div(
                        output_log, soft_target, reduction='batchmean'
                    ) * (T ** 2)
                loss = loss_CE + loss_KD.sum()

                if args.instance_contrastive:
                    loss += args.lam_I * instance_contra_loss
                if args.class_contrastive:
                    loss += args.lam_C * class_contra_loss
                if args.attn_score_distil:
                    loss += (
                        args.lam * spatial_attn_dist_loss
                        + (1 - args.lam) * temporal_attn_dist_loss
                    )

            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1

        if num_steps == 0:
            raise RuntimeError('Training loader produced zero optimization steps')
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print(
            'Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss),
            flush=True,
        )

        all_val_out_logits = []
        all_val_labels = []
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0].to(device)
                val_audio = val_data[1].to(device)
                val_out_logits = model(
                    visual=val_visual, audio=val_audio
                )
                all_val_out_logits.append(
                    F.softmax(val_out_logits, dim=-1).detach().cpu()
                )
                all_val_labels.append(val_labels.detach().cpu())

        all_val_out_logits = torch.cat(all_val_out_logits, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print(
            'Epoch:{} val_res:{:.6f} '.format(epoch, val_top1),
            flush=True,
        )

        # Save the true validation-best checkpoint over the whole training
        # trajectory. In particular, step 0 is an exact baseline run, and
        # warm-up checkpoints at later steps are not discarded.
        if val_top1 > best_val_res:
            best_val_res = val_top1
            best_epoch = epoch
            print(
                'Saving best model at Epoch {}'.format(epoch), flush=True
            )
            torch.save(unwrap_model(model), best_model_path)

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig(
            './save/fig/{}/train_loss_step_{}.png'.format(
                args.dataset, step
            )
        )
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig(
            './save/fig/{}/val_acc_step_{}.png'.format(args.dataset, step)
        )
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, opt, epoch)

    if best_epoch is None or not os.path.exists(best_model_path):
        raise RuntimeError(
            'No best checkpoint was saved at step {}.'.format(step)
        )

    if args.continual_guidance:
        # Consolidation loads the best checkpoint. Release the final-epoch
        # student/teacher first so large models are not resident on the GPU
        # simultaneously. The frozen prior is small and can be moved back.
        fixed_prior_for_consolidation = {
            key: value.detach().cpu()
            for key, value in fixed_prior.items()
        }
        del opt
        del model
        if old_model is not None:
            del old_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        consolidate_best_checkpoint_prior(
            args=args,
            step=step,
            best_epoch=best_epoch,
            best_model_path=best_model_path,
            train_data_set=train_data_set,
            exemplar_set=exemplar_set,
            fixed_prior=fixed_prior_for_consolidation,
        )

    return best_epoch, best_val_res


# def detailed_test(args, step, test_data_set, task_best_acc_list):
#     print("=====================================")
#     print("Start testing...")
#     print("=====================================")

#     model = load_full_model('./save/{}/step_{}_best_model.pkl'.format(args.dataset, step), map_location=device)
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

    model = load_full_model(
        './save/{}/step_{}_best_model.pkl'.format(args.dataset, step),
        map_location=device,
    )
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

    # Explicit class-wise continual audio-to-visual guidance prior.
    parser.add_argument('--continual_guidance', action='store_true', default=False)
    parser.add_argument(
        '--guidance_start_step', type=int, default=1,
        help=(
            'First incremental step that applies guidance during optimization. '
            'The default 1 leaves step 0 as an exact baseline training run, '
            'while still consolidating its best-checkpoint statistics as the '
            'prior for step 1.'
        ),
    )
    parser.add_argument('--guidance_warmup_epochs', type=int, default=20)
    parser.add_argument('--guidance_refresh_interval', type=int, default=20,
                        help='Epoch interval for within-step re-estimation from the fixed previous-step prior.')
    parser.add_argument('--guidance_max_samples_per_class', type=int, default=64)
    parser.add_argument('--guidance_knn_k', type=int, default=5)
    parser.add_argument('--guidance_bootstrap_samples', type=int, default=8)
    parser.add_argument('--guidance_q0', type=float, default=0.01,
                        help='Base process variance of the step-wise guidance state.')
    parser.add_argument('--guidance_q1', type=float, default=0.10,
                        help='Process-variance scale for z2 log-difficulty change.')
    parser.add_argument('--guidance_r0', type=float, default=0.10,
                        help='Finite-sample floor in measurement variance R_c.')
    parser.add_argument('--guidance_eps', type=float, default=1e-6)
    parser.add_argument('--guidance_margin_clip', type=float, default=100.0)
    parser.add_argument('--guidance_energy_max', type=float, default=6.0)
    parser.add_argument('--guidance_z2_log_change_max', type=float, default=5.0,
                        help='Numerical cap on the z2 log-difficulty change used in process variance.')
    parser.add_argument('--guidance_gate_min', type=float, default=0.0)
    parser.add_argument('--guidance_purity_alpha', type=float, default=1.0,
                        help='Beta smoothing strength for class kNN purity.')

    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.1)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.1)

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dump_tsne", action="store_true", help="If set, dump t-SNE plots for each step")
    parser.add_argument("--tsne_feature", type=str, default="logits",
                        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"])
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    

    args = parser.parse_args()
    print(args)

    if args.continual_guidance:
        if args.guidance_start_step < 1:
            parser.error(
                '--guidance_start_step must be at least 1 so step 0 remains '
                'an exact baseline run'
            )
        if args.guidance_warmup_epochs < 0:
            parser.error('--guidance_warmup_epochs must be non-negative')
        if args.guidance_warmup_epochs >= args.max_epoches:
            parser.error('--guidance_warmup_epochs must be smaller than --max_epoches')
        if args.guidance_refresh_interval <= 0:
            parser.error('--guidance_refresh_interval must be positive')
        if args.guidance_bootstrap_samples < 1:
            parser.error('--guidance_bootstrap_samples must be at least 1')
        if args.guidance_max_samples_per_class < 2:
            parser.error('--guidance_max_samples_per_class must be at least 2')
        if args.guidance_knn_k < 1:
            parser.error('--guidance_knn_k must be positive')
        if args.guidance_q0 < 0 or args.guidance_q1 < 0:
            parser.error('--guidance_q0 and --guidance_q1 must be non-negative')
        if args.guidance_r0 <= 0:
            parser.error('--guidance_r0 must be positive')
        if not (0.0 <= args.guidance_gate_min <= 1.0):
            parser.error('--guidance_gate_min must be in [0, 1]')

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