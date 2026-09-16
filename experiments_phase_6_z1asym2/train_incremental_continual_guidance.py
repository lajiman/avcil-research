import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental_continual_guidance import IncreAudioVisualNet
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
        "z2": "z2_fusion_norm",
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
):
    """
    Normalized-margin geometry:
        M_c = B_c / (S_c + eps)
        D_c = S_c / (B_c + eps)

    where S_c is mean cosine distance to the class centroid and B_c is the
    nearest-centroid cosine distance. If requested, local kNN purity is also
    returned for each sample and class.
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
        nearest = centroid_distance.min(dim=1).values
        inter_distance.index_copy_(0, valid_ids, nearest)
    elif valid_ids.numel() == 1:
        inter_distance[valid_ids[0]] = 1.0

    margin = inter_distance / (dispersion + eps)
    margin = margin.clamp(min=0.0, max=margin_clip)
    difficulty = dispersion / (inter_distance + eps)

    sample_purity = torch.zeros(features.shape[0], device=feat_device)
    class_purity = torch.zeros(num_classes, device=feat_device)

    if compute_purity and features.shape[0] > 1:
        max_available_same_class = int(
            max((idx.numel() - 1 for idx in class_indices), default=0)
        )
        max_k = min(knn_k, max_available_same_class, features.shape[0] - 1)
        if max_k > 0:
            similarity = torch.matmul(features, features.t())
            similarity.fill_diagonal_(-float("inf"))
            neighbor_ids = similarity.topk(max_k, dim=1).indices
            neighbor_labels = labels.index_select(0, neighbor_ids.reshape(-1)).view(
                neighbor_ids.shape
            )

            for class_id, idx in enumerate(class_indices):
                if idx.numel() == 0:
                    continue
                local_k = min(knn_k, int(idx.numel()) - 1, features.shape[0] - 1)
                if local_k <= 0:
                    class_purity[class_id] = 0.0
                    continue
                same = (
                    neighbor_labels.index_select(0, idx)[:, :local_k]
                    == class_id
                ).float()
                values = same.mean(dim=1)
                sample_purity.index_copy_(0, idx, values)
                class_purity[class_id] = values.mean()

    return {
        "margin": margin,
        "difficulty": difficulty,
        "purity": class_purity,
        "sample_purity": sample_purity,
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
    energy_max,
    seed,
):
    """Bootstrap only the class aggregates; the kNN graph is computed once."""
    if num_bootstraps <= 1:
        return torch.zeros(num_classes, device=audio_features.device)

    generator = torch.Generator()
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
            draw = torch.randint(
                low=0,
                high=idx.numel(),
                size=(idx.numel(),),
                generator=generator,
            ).to(idx.device)
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
        for class_id in range(num_classes):
            mask = sampled_labels == class_id
            if not mask.any():
                continue
            original_ids = sampled_indices[mask]
            audio_purity[class_id] = audio_sample_purity.index_select(
                0, original_ids
            ).mean()
            visual_purity[class_id] = visual_sample_purity.index_select(
                0, original_ids
            ).mean()

        audio_reliability = audio_geo["margin"] * audio_purity
        visual_reliability = visual_geo["margin"] * visual_purity
        evidence = torch.relu(
            torch.log(
                (visual_reliability + eps) / (audio_reliability + eps)
            )
        ).clamp(max=energy_max)
        all_evidence.append(evidence)

    stacked = torch.stack(all_evidence, dim=0)
    return stacked.var(dim=0, unbiased=False)


def update_continual_guidance_state(
    args,
    model,
    step,
    train_data_set,
    exemplar_set,
):
    """
    One class-wise Kalman update per incremental step.

        E_hat_c^t = [log(R_v,c^t / R_a,c^t)]_+
        Q_c^t     = q0 + q1 * |log(D2_c^t / D2_c^{t-1})|
        K_c^t     = (P_c^{t-1} + Q_c^t) /
                    (P_c^{t-1} + Q_c^t + R_c^t)
        E_c^t     = (1-K_c^t) E_c^{t-1} + K_c^t E_hat_c^t
        g_c^t     = exp(-E_c^t)

    New classes have no prior and are initialized directly from E_hat.
    """
    core_model = unwrap_model(model)
    num_seen_classes = (step + 1) * args.class_num_per_step

    feature_bank = collect_guidance_feature_bank(
        model=model,
        train_data_set=train_data_set,
        exemplar_set=exemplar_set,
        step=step,
        args=args,
    )

    stats_device = device
    audio_features = feature_bank["audio"].to(stats_device)
    visual_features = feature_bank["visual"].to(stats_device)
    z2_features = feature_bank["z2"].to(stats_device)
    labels = feature_bank["labels"].to(stats_device)

    audio_geo = compute_class_geometry(
        audio_features,
        labels,
        num_seen_classes,
        args.guidance_knn_k,
        args.guidance_eps,
        compute_purity=True,
        margin_clip=args.guidance_margin_clip,
    )
    visual_geo = compute_class_geometry(
        visual_features,
        labels,
        num_seen_classes,
        args.guidance_knn_k,
        args.guidance_eps,
        compute_purity=True,
        margin_clip=args.guidance_margin_clip,
    )
    z2_geo = compute_class_geometry(
        z2_features,
        labels,
        num_seen_classes,
        args.guidance_knn_k,
        args.guidance_eps,
        compute_purity=False,
        margin_clip=args.guidance_margin_clip,
    )

    audio_reliability = audio_geo["margin"] * audio_geo["purity"]
    visual_reliability = visual_geo["margin"] * visual_geo["purity"]
    evidence = torch.relu(
        torch.log(
            (visual_reliability + args.guidance_eps)
            / (audio_reliability + args.guidance_eps)
        )
    ).clamp(max=args.guidance_energy_max)

    bootstrap_var = _bootstrap_evidence(
        audio_features=audio_features,
        visual_features=visual_features,
        labels=labels,
        audio_sample_purity=audio_geo["sample_purity"],
        visual_sample_purity=visual_geo["sample_purity"],
        num_classes=num_seen_classes,
        num_bootstraps=args.guidance_bootstrap_samples,
        eps=args.guidance_eps,
        energy_max=args.guidance_energy_max,
        seed=args.seed + 7919 * (step + 1),
    )
    counts = torch.minimum(audio_geo["counts"], visual_geo["counts"]).clamp_min(1.0)
    measurement_var = bootstrap_var + args.guidance_r0 / counts

    old_energy = core_model.guidance_energy[:num_seen_classes].detach().clone()
    old_variance = core_model.guidance_variance[:num_seen_classes].detach().clone()
    old_z2 = core_model.guidance_z2_difficulty[:num_seen_classes].detach().clone()
    initialized = core_model.guidance_initialized[:num_seen_classes].detach().clone()

    new_energy = old_energy.clone()
    new_variance = old_variance.clone()
    new_z2 = z2_geo["difficulty"].detach().clone()
    new_initialized = initialized.clone()
    prior_weight = torch.zeros_like(new_energy)

    valid = audio_geo["valid"] & visual_geo["valid"] & z2_geo["valid"]
    for class_id in range(num_seen_classes):
        if not bool(valid[class_id].item()):
            # No current evidence: preserve the old state and its distillation.
            prior_weight[class_id] = 1.0 if initialized[class_id] else 0.0
            continue

        if not bool(initialized[class_id].item()):
            new_energy[class_id] = evidence[class_id]
            new_variance[class_id] = measurement_var[class_id]
            new_initialized[class_id] = True
            prior_weight[class_id] = 0.0
            continue

        z2_log_change = torch.abs(
            torch.log(
                (new_z2[class_id] + args.guidance_eps)
                / (old_z2[class_id] + args.guidance_eps)
            )
        )
        process_var = args.guidance_q0 + args.guidance_q1 * z2_log_change
        prior_var = old_variance[class_id] + process_var
        kalman_gain = prior_var / (prior_var + measurement_var[class_id])

        new_energy[class_id] = (
            (1.0 - kalman_gain) * old_energy[class_id]
            + kalman_gain * evidence[class_id]
        )
        new_variance[class_id] = (1.0 - kalman_gain) * prior_var
        prior_weight[class_id] = 1.0 - kalman_gain

    full_energy = core_model.guidance_energy.detach().clone()
    full_variance = core_model.guidance_variance.detach().clone()
    full_z2 = core_model.guidance_z2_difficulty.detach().clone()
    full_initialized = core_model.guidance_initialized.detach().clone()
    full_prior_weight = core_model.guidance_prior_weight.detach().clone()

    full_energy[:num_seen_classes] = new_energy
    full_variance[:num_seen_classes] = new_variance
    full_z2[:num_seen_classes] = new_z2
    full_initialized[:num_seen_classes] = new_initialized
    full_prior_weight[:num_seen_classes] = prior_weight

    core_model.set_guidance_state(
        energy=full_energy,
        variance=full_variance,
        z2_difficulty=full_z2,
        initialized=full_initialized,
        prior_weight=full_prior_weight,
    )

    class_gates = torch.exp(-new_energy).clamp(
        min=args.guidance_gate_min, max=1.0
    )
    metrics_root = './save/metrics/{}/'.format(args.dataset)
    rows = []
    for class_id in range(num_seen_classes):
        rows.append({
            "step": step,
            "class_id": class_id,
            "count": float(counts[class_id].item()),
            "audio_margin": float(audio_geo["margin"][class_id].item()),
            "visual_margin": float(visual_geo["margin"][class_id].item()),
            "audio_purity": float(audio_geo["purity"][class_id].item()),
            "visual_purity": float(visual_geo["purity"][class_id].item()),
            "audio_reliability": float(audio_reliability[class_id].item()),
            "visual_reliability": float(visual_reliability[class_id].item()),
            "evidence": float(evidence[class_id].item()),
            "measurement_variance": float(measurement_var[class_id].item()),
            "z2_difficulty": float(new_z2[class_id].item()),
            "posterior_energy": float(new_energy[class_id].item()),
            "posterior_variance": float(new_variance[class_id].item()),
            "prior_weight": float(prior_weight[class_id].item()),
            "gate": float(class_gates[class_id].item()),
        })

    guidance_csv = os.path.join(metrics_root, "guidance_state_step_{}.csv".format(step))
    if os.path.exists(guidance_csv):
        os.remove(guidance_csv)
    append_csv_rows(guidance_csv, list(rows[0].keys()), rows)
    save_json(
        {
            "step": step,
            "formula": "Kalman class-wise explicit audio-to-visual guidance prior",
            "rows": rows,
        },
        os.path.join(metrics_root, "guidance_state_step_{}.json".format(step)),
    )

    print(
        "Guidance update at step {}: mean_gate={:.4f}, min_gate={:.4f}, "
        "max_gate={:.4f}, mean_prior_weight={:.4f}".format(
            step,
            class_gates.mean().item(),
            class_gates.min().item(),
            class_gates.max().item(),
            prior_weight.mean().item(),
        ),
        flush=True,
    )

    del audio_features, visual_features, z2_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    guidance_updated = not args.continual_guidance
    for epoch in range(args.max_epoches):
        if (
            args.continual_guidance
            and not guidance_updated
            and epoch >= args.guidance_warmup_epochs
        ):
            update_continual_guidance_state(
                args=args,
                model=model,
                step=step,
                train_data_set=train_data_set,
                exemplar_set=exemplar_set,
            )
            guidance_updated = True
            # Force model selection to happen after the method has been activated.
            best_val_res = -1.0

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
                    old_out, old_spatial_attn_score, old_temporal_attn_score = old_model(visual=total_visual, audio=total_audio, out_attn_score=True)
                    old_out = old_out.detach()
                    old_spatial_attn_score = old_spatial_attn_score.detach()
                    old_temporal_attn_score = old_temporal_attn_score.detach()
                
                if args.instance_contrastive:
                    instance_contra_loss = cal_contrastive_loss(audio_feature, visual_feature, temperature=args.instance_contrastive_temperature)
                
                if args.class_contrastive:
                    all_labels = torch.cat((labels, exemplar_labels))
                    class_contra_loss = class_contrastive_loss(audio_feature, visual_feature, all_labels, temperature=args.class_contrastive_temperature)
                
                if args.attn_score_distil:
                    exem_slice = slice(
                        data_batch_size,
                        data_batch_size + exemplar_data_batch_size,
                    )
                    exem_spatial_attn_score = spatial_attn_score[exem_slice]
                    exem_old_spatial_attn_score = old_spatial_attn_score[exem_slice]
                    exem_temporal_attn_score = temporal_attn_score[exem_slice]
                    exem_old_temporal_attn_score = old_temporal_attn_score[exem_slice]

                    if args.continual_guidance:
                        prior_weights = unwrap_model(model).prior_weights_for_labels(
                            exemplar_labels
                        )
                    else:
                        prior_weights = torch.ones(
                            exemplar_data_batch_size, device=device
                        )

                    spatial_attn_dist_loss, temporal_attn_dist_loss = (
                        weighted_attention_distillation(
                            student_spatial=exem_spatial_attn_score,
                            teacher_spatial=exem_old_spatial_attn_score,
                            student_temporal=exem_temporal_attn_score,
                            teacher_temporal=exem_old_temporal_attn_score,
                            sample_weights=prior_weights,
                        )
                    )

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
                if args.attn_score_distil:
                    loss += args.lam * spatial_attn_dist_loss + (1 - args.lam) * temporal_attn_dist_loss
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
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

    # Explicit class-wise continual audio-to-visual guidance prior.
    parser.add_argument('--continual_guidance', action='store_true', default=False)
    parser.add_argument('--guidance_warmup_epochs', type=int, default=20)
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
    parser.add_argument('--guidance_gate_min', type=float, default=0.0)
    parser.add_argument('--guidance_routing_temperature', type=float, default=1.0)

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