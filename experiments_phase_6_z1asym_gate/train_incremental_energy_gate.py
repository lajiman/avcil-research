import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental_energy_gate import IncreAudioVisualNet
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
    """Load a trusted full-model checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _append_capped_energy_features(bank, labels, outputs, max_per_class):
    labels = labels.detach().cpu().long()
    for row_idx, class_id in enumerate(labels.tolist()):
        class_bank = bank.setdefault(class_id, {"audio": [], "visual": []})
        if len(class_bank["audio"]) >= max_per_class:
            continue
        class_bank["audio"].append(
            outputs["z1_audio_norm"][row_idx].detach().cpu()
        )
        class_bank["visual"].append(
            outputs["z1_visual_uniform_norm"][row_idx].detach().cpu()
        )


def collect_energy_feature_bank(model, train_data_set, exemplar_set, step, args):
    """
    Collect current-step z1 geometry from current training samples and the
    existing AVCIL exemplar memory. The visual branch is uniform and therefore
    independent of audio guidance and of the current energy gate.
    """
    core_model = unwrap_model(model)
    was_training = core_model.training
    core_model.eval()

    bank = {}
    datasets = [train_data_set]
    if step > 0 and exemplar_set is not None and exemplar_set.__len__() > 0:
        datasets.append(exemplar_set)

    generator = torch.Generator()
    generator.manual_seed(args.seed + 1009 * (step + 1))

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
                outputs = core_model(
                    visual=data[0].to(device),
                    audio=data[1].to(device),
                    return_dict=True,
                    out_analysis_features=True,
                )
                _append_capped_energy_features(
                    bank=bank,
                    labels=labels,
                    outputs=outputs,
                    max_per_class=args.energy_max_samples_per_class,
                )

    if was_training:
        core_model.train()

    audio_rows, visual_rows, label_rows = [], [], []
    num_seen_classes = (step + 1) * args.class_num_per_step
    for class_id in range(num_seen_classes):
        class_bank = bank.get(class_id)
        if class_bank is None or len(class_bank["audio"]) == 0:
            continue
        count = len(class_bank["audio"])
        audio_rows.extend(class_bank["audio"])
        visual_rows.extend(class_bank["visual"])
        label_rows.extend([class_id] * count)

    if not label_rows:
        raise RuntimeError("No samples were collected for energy estimation")

    return {
        "audio": torch.stack(audio_rows, dim=0),
        "visual": torch.stack(visual_rows, dim=0),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def compute_energy_class_geometry(
    features,
    labels,
    num_classes,
    knn_k,
    eps,
    margin_clip,
    purity_alpha,
):
    """
    Geometry in normalized cosine space:

        S_c = mean_i [1 - z_i^T mu_c]
        B_c = min_{c' != c} [1 - mu_c^T mu_c']
        M_c = B_c / (S_c + eps)
        P_c = smoothed global-kNN purity
        reliability_c = M_c P_c.
    """
    features = F.normalize(features.float(), dim=1)
    labels = labels.long()
    feat_device = features.device

    centroids = torch.zeros(
        num_classes, features.shape[1], device=feat_device, dtype=features.dtype
    )
    dispersion = torch.zeros(num_classes, device=feat_device)
    inter_distance = torch.zeros(num_classes, device=feat_device)
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

    purity = torch.zeros(num_classes, device=feat_device)
    if features.shape[0] > 1:
        k_eff = min(int(knn_k), features.shape[0] - 1)
        similarity = torch.matmul(features, features.t())
        similarity.fill_diagonal_(-float("inf"))
        neighbor_ids = similarity.topk(k_eff, dim=1).indices
        neighbor_labels = labels.index_select(
            0, neighbor_ids.reshape(-1)
        ).view_as(neighbor_ids)

        for class_id, idx in enumerate(class_indices):
            if idx.numel() == 0:
                continue
            same = (
                neighbor_labels.index_select(0, idx) == class_id
            ).float().sum()
            total = float(idx.numel() * k_eff)
            purity[class_id] = (
                same + purity_alpha
            ) / (total + 2.0 * purity_alpha)
    else:
        purity[valid] = 1.0

    return {
        "margin": margin,
        "purity": purity,
        "reliability": margin * purity,
        "dispersion": dispersion,
        "inter_distance": inter_distance,
        "counts": counts,
        "valid": valid,
    }


def estimate_current_energy(args, model, step, train_data_set, exemplar_set):
    """Compute E_hat directly from the current model; no prior or recursion."""
    num_seen_classes = (step + 1) * args.class_num_per_step
    bank = collect_energy_feature_bank(
        model=model,
        train_data_set=train_data_set,
        exemplar_set=exemplar_set,
        step=step,
        args=args,
    )
    audio = bank["audio"].to(device)
    visual = bank["visual"].to(device)
    labels = bank["labels"].to(device)

    kwargs = dict(
        labels=labels,
        num_classes=num_seen_classes,
        knn_k=args.energy_knn_k,
        eps=args.energy_eps,
        margin_clip=args.energy_margin_clip,
        purity_alpha=args.energy_purity_alpha,
    )
    audio_geo = compute_energy_class_geometry(features=audio, **kwargs)
    visual_geo = compute_energy_class_geometry(features=visual, **kwargs)

    valid = audio_geo["valid"] & visual_geo["valid"]
    energy_raw = torch.relu(
        torch.log(
            (visual_geo["reliability"] + args.energy_eps)
            / (audio_geo["reliability"] + args.energy_eps)
        )
    )
    energy = torch.zeros(num_seen_classes, device=device)
    energy[valid] = energy_raw[valid].clamp(
        min=0.0, max=args.energy_max
    )

    result = {
        "energy": energy,
        "energy_raw": energy_raw,
        "gate": torch.exp(-energy).clamp(
            min=args.energy_gate_min, max=1.0
        ),
        "valid": valid,
        "counts": torch.minimum(
            audio_geo["counts"], visual_geo["counts"]
        ),
        "audio_margin": audio_geo["margin"],
        "visual_margin": visual_geo["margin"],
        "audio_purity": audio_geo["purity"],
        "visual_purity": visual_geo["purity"],
        "audio_reliability": audio_geo["reliability"],
        "visual_reliability": visual_geo["reliability"],
    }

    del audio, visual, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_energy_diagnostics(args, step, epoch, result):
    rows = []
    num_seen_classes = result["energy"].numel()
    for class_id in range(num_seen_classes):
        rows.append({
            "step": step,
            "epoch": epoch,
            "class_id": class_id,
            "valid": int(result["valid"][class_id].item()),
            "count": float(result["counts"][class_id].item()),
            "audio_margin": float(result["audio_margin"][class_id].item()),
            "visual_margin": float(result["visual_margin"][class_id].item()),
            "audio_purity": float(result["audio_purity"][class_id].item()),
            "visual_purity": float(result["visual_purity"][class_id].item()),
            "audio_reliability": float(
                result["audio_reliability"][class_id].item()
            ),
            "visual_reliability": float(
                result["visual_reliability"][class_id].item()
            ),
            "energy_raw": float(result["energy_raw"][class_id].item()),
            "energy": float(result["energy"][class_id].item()),
            "gate": float(result["gate"][class_id].item()),
        })

    metrics_root = './save/metrics/{}/'.format(args.dataset)
    csv_path = os.path.join(
        metrics_root, 'energy_updates_step_{}.csv'.format(step)
    )
    append_csv_rows(csv_path, list(rows[0].keys()), rows)
    save_json(
        {
            "step": step,
            "epoch": epoch,
            "gate_mean": float(result["gate"].mean().item()),
            "gate_min": float(result["gate"].min().item()),
            "gate_max": float(result["gate"].max().item()),
            "classes": rows,
        },
        os.path.join(
            metrics_root,
            'energy_step_{}_epoch_{}.json'.format(step, epoch),
        ),
    )


def refresh_current_energy(
    args, model, step, epoch, train_data_set, exemplar_set
):
    result = estimate_current_energy(
        args=args,
        model=model,
        step=step,
        train_data_set=train_data_set,
        exemplar_set=exemplar_set,
    )
    core_model = unwrap_model(model)
    core_model.set_class_energy(result["energy"], activate=True)
    write_energy_diagnostics(args, step, epoch, result)
    print(
        "Energy refresh step {} epoch {}: gate mean={:.6f}, "
        "min={:.6f}, max={:.6f}, energy mean={:.6f}".format(
            step,
            epoch,
            result["gate"].mean().item(),
            result["gate"].min().item(),
            result["gate"].max().item(),
            result["energy"].mean().item(),
        ),
        flush=True,
    )
    return result


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
    else:
        previous_path = './save/{}/step_{}_best_model.pkl'.format(
            args.dataset, step - 1
        )
        model = load_full_model(previous_path, map_location='cpu')
        model.incremental_classifier(step_out_class_num)
        old_model = load_full_model(previous_path, map_location='cpu')

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

    # No class energy, gate, or other state is carried into the new step.
    # Every step starts with the exact AVCIL forward and re-estimates E_hat
    # from its own current representation after warm-up.
    unwrap_model(model).reset_energy_state()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)

    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        old_model.eval()

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0
    energy_activated = False
    best_model_path = './save/{}/step_{}_best_model.pkl'.format(
        args.dataset, step
    )
    energy_csv = './save/metrics/{}/energy_updates_step_{}.csv'.format(
        args.dataset, step
    )
    if os.path.exists(energy_csv):
        os.remove(energy_csv)

    for epoch in range(args.max_epoches):
        should_refresh = (
            args.energy_gate
            and step >= args.energy_start_step
            and epoch >= args.energy_warmup_epochs
            and (
                (epoch - args.energy_warmup_epochs)
                % args.energy_refresh_interval
                == 0
            )
        )
        if should_refresh:
            refresh_current_energy(
                args=args,
                model=model,
                step=step,
                epoch=epoch,
                train_data_set=train_data_set,
                exemplar_set=exemplar_set,
            )
            if not energy_activated:
                # The requested experiment evaluates the energy-gated model,
                # not the pre-gate warm-up checkpoint.
                energy_activated = True
                best_val_res = -1.0
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                print(
                    "Energy gate activated; best-checkpoint selection restarts.",
                    flush=True,
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
                # Preserve the original AVCIL step-0 objective exactly.
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

                total_visual = torch.cat(
                    (data[0], exemplar_data[0]), dim=0
                ).to(device)
                total_audio = torch.cat(
                    (data[1], exemplar_data[1]), dim=0
                ).to(device)

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

                # Both original z1 losses are preserved without modification.
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

                # Preserve the original raw AVCIL attention distillation.
                if args.attn_score_distil:
                    begin = data_batch_size
                    end = data_batch_size + exemplar_data_batch_size

                    exem_spatial = spatial_attn_score[begin:end].transpose(2, 3)
                    exem_spatial = exem_spatial.reshape(
                        -1, exem_spatial.shape[-1]
                    )
                    old_exem_spatial = old_spatial_attn_score[begin:end].transpose(
                        2, 3
                    )
                    old_exem_spatial = old_exem_spatial.reshape(
                        -1, old_exem_spatial.shape[-1]
                    )

                    exem_temporal = temporal_attn_score[begin:end].transpose(1, 2)
                    exem_temporal = exem_temporal.reshape(
                        -1, exem_temporal.shape[-1]
                    )
                    old_exem_temporal = old_temporal_attn_score[
                        begin:end
                    ].transpose(1, 2)
                    old_exem_temporal = old_exem_temporal.reshape(
                        -1, old_exem_temporal.shape[-1]
                    )

                    spatial_attn_dist_loss = F.kl_div(
                        exem_spatial.log(),
                        old_exem_spatial,
                        reduction='sum',
                    ) / exemplar_data_batch_size
                    temporal_attn_dist_loss = F.kl_div(
                        exem_temporal.log(),
                        old_exem_temporal,
                        reduction='sum',
                    ) / exemplar_data_batch_size

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
                        + (1.0 - args.lam) * temporal_attn_dist_loss
                    )

            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1

        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print(
            'Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss),
            flush=True,
        )

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0].to(device)
                val_audio = val_data[1].to(device)
                if torch.cuda.device_count() > 1:
                    val_out_logits = model.module.forward(
                        visual=val_visual, audio=val_audio
                    )
                else:
                    val_out_logits = model(
                        visual=val_visual, audio=val_audio
                    )
                val_out_logits = F.softmax(
                    val_out_logits, dim=-1
                ).detach().cpu()
                all_val_out_logits = torch.cat(
                    (all_val_out_logits, val_out_logits), dim=0
                )
                all_val_labels = torch.cat(
                    (all_val_labels, val_labels), dim=0
                )

        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print(
            'Epoch:{} val_res:{:.6f} '.format(epoch, val_top1),
            flush=True,
        )

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(model.module, best_model_path)
            else:
                torch.save(model, best_model_path)

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
            './save/fig/{}/val_acc_step_{}.png'.format(
                args.dataset, step
            )
        )
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

    parser.add_argument('--energy_gate', action='store_true', default=False)
    parser.add_argument('--energy_start_step', type=int, default=0)
    parser.add_argument('--energy_warmup_epochs', type=int, default=20)
    parser.add_argument('--energy_refresh_interval', type=int, default=20)
    parser.add_argument('--energy_max_samples_per_class', type=int, default=64)
    parser.add_argument('--energy_knn_k', type=int, default=5)
    parser.add_argument('--energy_eps', type=float, default=1e-6)
    parser.add_argument('--energy_margin_clip', type=float, default=100.0)
    parser.add_argument('--energy_max', type=float, default=6.0)
    parser.add_argument('--energy_gate_min', type=float, default=0.0)
    parser.add_argument('--energy_purity_alpha', type=float, default=1.0)

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dump_tsne", action="store_true", help="If set, dump t-SNE plots for each step")
    parser.add_argument("--tsne_feature", type=str, default="logits",
                        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"])
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    

    args = parser.parse_args()
    if args.energy_gate:
        if args.energy_start_step < 0:
            parser.error('--energy_start_step must be non-negative')
        if args.energy_warmup_epochs < 0:
            parser.error('--energy_warmup_epochs must be non-negative')
        if args.energy_warmup_epochs >= args.max_epoches:
            parser.error('--energy_warmup_epochs must be smaller than --max_epoches')
        if args.energy_refresh_interval <= 0:
            parser.error('--energy_refresh_interval must be positive')
        if args.energy_max_samples_per_class < 2:
            parser.error('--energy_max_samples_per_class must be at least 2')
        if args.energy_knn_k < 1:
            parser.error('--energy_knn_k must be positive')
        if args.energy_eps <= 0:
            parser.error('--energy_eps must be positive')
        if args.energy_margin_clip <= 0:
            parser.error('--energy_margin_clip must be positive')
        if args.energy_max < 0:
            parser.error('--energy_max must be non-negative')
        if not (0.0 <= args.energy_gate_min <= 1.0):
            parser.error('--energy_gate_min must be in [0, 1]')
        if args.energy_purity_alpha <= 0:
            parser.error('--energy_purity_alpha must be positive')
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