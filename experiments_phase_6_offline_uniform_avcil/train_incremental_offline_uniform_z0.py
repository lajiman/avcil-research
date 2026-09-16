import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental_offline_uniform import IncreAudioVisualNetOfflineUniform
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



# ======================================================================
# Offline z0 modality difficulty
# ======================================================================

def _difficulty_state_from_rank(
    scores,
    easy_ratio=0.30,
    hard_ratio=0.30,
):
    """
    Split classes by normalized-margin rank.

    Larger normalized margin means easier:
      top ``easy_ratio``  -> Easy
      bottom ``hard_ratio`` -> Hard
      remaining classes -> Medium
    """
    scores = np.asarray(scores, dtype=np.float64)
    num_classes = int(scores.shape[0])
    if num_classes <= 0:
        raise ValueError("scores must contain at least one class")
    if not (0.0 < easy_ratio < 1.0):
        raise ValueError("offline_easy_ratio must be in (0, 1)")
    if not (0.0 < hard_ratio < 1.0):
        raise ValueError("offline_hard_ratio must be in (0, 1)")
    if easy_ratio + hard_ratio >= 1.0:
        raise ValueError(
            "offline_easy_ratio + offline_hard_ratio must be < 1"
        )

    num_easy = max(1, int(round(num_classes * easy_ratio)))
    num_hard = max(1, int(round(num_classes * hard_ratio)))
    if num_easy + num_hard >= num_classes:
        raise ValueError("Easy and Hard groups leave no Medium classes")

    # Deterministic tie handling: score first, then class id.
    descending = sorted(
        range(num_classes),
        key=lambda class_id: (-scores[class_id], class_id),
    )
    easy_ids = set(descending[:num_easy])
    hard_ids = set(descending[-num_hard:])

    states = []
    for class_id in range(num_classes):
        if class_id in easy_ids:
            states.append("E")
        elif class_id in hard_ids:
            states.append("H")
        else:
            states.append("M")
    return states


@torch.no_grad()
def compute_offline_z0_difficulty(
    args,
    train_data_set,
    id_to_category,
):
    """
    Compute intrinsic audio/visual difficulty directly from the training z0
    features before any model is initialized or trained.

    z0 audio:
        the fixed input audio embedding.

    z0 visual:
        uniform average over all temporal and spatial visual tokens, with no
        audio-guided attention.

    For each modality and class c:

        intra(c) = mean_i [1 - cos(z_i, centroid_c)]
        inter(c) = min_{c' != c} [1 - cos(centroid_c, centroid_c')]
        margin(c) = inter(c) / (intra(c) + eps)

    Larger margin means easier.  Classes are ranked globally over all 100
    training classes using a 30/40/30 Easy/Medium/Hard split by default.
    """
    num_classes = int(args.num_classes)
    feature_dim = 768
    eps = float(args.offline_margin_eps)

    audio_sum = torch.zeros(
        num_classes, feature_dim, dtype=torch.float64, device=device
    )
    visual_sum = torch.zeros_like(audio_sum)
    counts = torch.zeros(
        num_classes, dtype=torch.float64, device=device
    )

    print(
        "Computing offline z0 difficulty from TRAIN data only...",
        flush=True,
    )

    total_steps = num_classes // args.class_num_per_step
    for difficulty_step in range(total_steps):
        train_data_set.set_incremental_step(difficulty_step)
        loader = DataLoader(
            train_data_set,
            batch_size=min(
                args.offline_difficulty_batch_size,
                train_data_set.__len__(),
            ),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        for data, labels in tqdm(
            loader,
            desc="z0 difficulty step {}".format(difficulty_step),
        ):
            labels = labels.to(device=device, dtype=torch.long)
            visual = data[0].to(device=device, dtype=torch.float32)
            audio = data[1].to(device=device, dtype=torch.float32)

            if audio.ndim != 2 or audio.shape[-1] != feature_dim:
                raise ValueError(
                    "Expected audio z0 shape (B, 768), got {}".format(
                        tuple(audio.shape)
                    )
                )

            visual_4d = visual.reshape(
                visual.shape[0], 8, -1, feature_dim
            )
            visual_uniform = visual_4d.mean(dim=(1, 2))

            audio_norm = F.normalize(audio, dim=1).to(torch.float64)
            visual_norm = F.normalize(
                visual_uniform, dim=1
            ).to(torch.float64)

            audio_sum.index_add_(0, labels, audio_norm)
            visual_sum.index_add_(0, labels, visual_norm)
            counts.index_add_(
                0,
                labels,
                torch.ones(
                    labels.shape[0],
                    dtype=torch.float64,
                    device=device,
                ),
            )

    missing = torch.where(counts <= 0)[0].detach().cpu().tolist()
    if missing:
        raise RuntimeError(
            "No training z0 samples were collected for class ids: {}".format(
                missing
            )
        )

    def geometry_from_sums(class_sum):
        centroid = F.normalize(class_sum, dim=1)

        # Since every sample was normalized before summation:
        # mean_i cos(z_i, centroid_c) = ||sum_i z_i|| / n_c.
        mean_cosine_to_centroid = (
            class_sum.norm(dim=1) / counts.clamp_min(1.0)
        ).clamp(min=-1.0, max=1.0)
        intra = (1.0 - mean_cosine_to_centroid).clamp_min(eps)

        centroid_distance = (
            1.0 - centroid @ centroid.transpose(0, 1)
        ).clamp_min(0.0)
        centroid_distance.fill_diagonal_(float("inf"))
        nearest_distance = centroid_distance.min(dim=1).values
        normalized_margin = nearest_distance / intra.clamp_min(eps)

        return (
            intra.detach().cpu().numpy(),
            nearest_distance.detach().cpu().numpy(),
            normalized_margin.detach().cpu().numpy(),
        )

    audio_intra, audio_nearest, audio_margin = geometry_from_sums(
        audio_sum
    )
    visual_intra, visual_nearest, visual_margin = geometry_from_sums(
        visual_sum
    )

    audio_state = _difficulty_state_from_rank(
        audio_margin,
        easy_ratio=args.offline_easy_ratio,
        hard_ratio=args.offline_hard_ratio,
    )
    visual_state = _difficulty_state_from_rank(
        visual_margin,
        easy_ratio=args.offline_easy_ratio,
        hard_ratio=args.offline_hard_ratio,
    )

    selected_ids = [
        class_id
        for class_id in range(num_classes)
        if audio_state[class_id] == "H"
        and visual_state[class_id] == "E"
    ]

    rows = []
    for class_id in range(num_classes):
        rows.append(
            {
                "class_id": class_id,
                "category_name": id_to_category.get(
                    class_id, "class_{}".format(class_id)
                ),
                "support": int(counts[class_id].item()),
                "audio_intra_dispersion": float(audio_intra[class_id]),
                "audio_nearest_centroid_distance": float(
                    audio_nearest[class_id]
                ),
                "audio_normalized_margin": float(
                    audio_margin[class_id]
                ),
                "audio_state": audio_state[class_id],
                "visual_intra_dispersion": float(
                    visual_intra[class_id]
                ),
                "visual_nearest_centroid_distance": float(
                    visual_nearest[class_id]
                ),
                "visual_normalized_margin": float(
                    visual_margin[class_id]
                ),
                "visual_state": visual_state[class_id],
                "offline_uniform_selected": int(
                    class_id in selected_ids
                ),
            }
        )

    contingency = {
        audio_group: {
            visual_group: sum(
                1
                for class_id in range(num_classes)
                if audio_state[class_id] == audio_group
                and visual_state[class_id] == visual_group
            )
            for visual_group in ["E", "M", "H"]
        }
        for audio_group in ["E", "M", "H"]
    }

    print("Offline z0 A/V state contingency: {}".format(contingency), flush=True)
    print(
        "Offline A:H / V:E classes: {}".format(
            [
                {
                    "class_id": class_id,
                    "category_name": id_to_category.get(
                        class_id, "class_{}".format(class_id)
                    ),
                    "audio_margin": float(audio_margin[class_id]),
                    "visual_margin": float(visual_margin[class_id]),
                }
                for class_id in selected_ids
            ]
        ),
        flush=True,
    )

    return selected_ids, rows, contingency


def save_offline_z0_outputs(
    args,
    rows,
    contingency,
    output_root,
):
    os.makedirs(output_root, exist_ok=True)

    csv_path = os.path.join(
        output_root, "offline_z0_modality_difficulty.csv"
    )
    header = [
        "class_id",
        "category_name",
        "support",
        "audio_intra_dispersion",
        "audio_nearest_centroid_distance",
        "audio_normalized_margin",
        "audio_state",
        "visual_intra_dispersion",
        "visual_nearest_centroid_distance",
        "visual_normalized_margin",
        "visual_state",
        "offline_uniform_selected",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    id_to_row = {int(row["class_id"]): row for row in rows}
    manifest = {
        "definition": {
            "data": "training split only",
            "audio_z0": "fixed audio input embedding",
            "visual_z0": (
                "uniform mean over temporal and spatial visual input tokens; "
                "independent of audio-guided attention"
            ),
            "distance": "cosine distance on L2-normalized samples/centroids",
            "normalized_margin": (
                "nearest-centroid distance / intra-class dispersion"
            ),
            "easy_ratio": float(args.offline_easy_ratio),
            "hard_ratio": float(args.offline_hard_ratio),
            "selection": "audio_state == H and visual_state == E",
        },
        "state_contingency": contingency,
        "offline_uniform_enabled": bool(args.offline_uniform),
        "uniform_feature_distil": bool(args.uniform_feature_distil),
        "lam_U": float(args.lam_U),
        "unguided_classes": [
            {
                "class_id": int(class_id),
                "category_name": id_to_row[class_id]["category_name"],
                "audio_normalized_margin": id_to_row[class_id][
                    "audio_normalized_margin"
                ],
                "visual_normalized_margin": id_to_row[class_id][
                    "visual_normalized_margin"
                ],
                "visual_branch": "uniform",
            }
            for class_id in args.offline_unguided_class_ids
        ],
    }
    save_json(
        manifest,
        os.path.join(output_root, "offline_uniform_gate.json"),
    )



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
        model = IncreAudioVisualNetOfflineUniform(args, step_out_class_num)
        old_model = None
        last_step_out_class_num = 0
        exemplar_loader = None
    else:
        previous_path = './save/{}/step_{}_best_model.pkl'.format(
            args.dataset, step - 1
        )
        model = torch.load(previous_path)
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load(previous_path)

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

    # The candidate class set is kept separate from whether routing is enabled.
    # This allows a matched Full+uniform-KD control to use the exact same KD
    # activation schedule without changing its all-guided forward path.
    has_old_uniform_target = (
        step > 0
        and any(
            class_id < last_step_out_class_num
            for class_id in args.offline_unguided_class_ids
        )
    )

    print(
        "Step {} routed uniform class ids among seen classes: {}".format(
            step,
            [
                class_id
                for class_id in args.offline_unguided_class_ids
                if class_id < step_out_class_num
            ],
        ),
        flush=True,
    )
    print(
        "Step {} uniform feature KD active: {}".format(
            step,
            bool(args.uniform_feature_distil and has_old_uniform_target),
        ),
        flush=True,
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loss_list = []
    val_acc_list = []
    best_val_res = -float("inf")

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        component_sums = {
            "ce": 0.0,
            "logit_kd": 0.0,
            "instance": 0.0,
            "class": 0.0,
            "attn": 0.0,
            "uniform_kd": 0.0,
        }

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

                outputs = model(
                    visual=visual,
                    audio=audio,
                    out_feature_before_fusion=True,
                    out_uniform_visual=True,
                    return_dict=True,
                )
                out = outputs["logits"]
                loss = CE_loss(step_out_class_num, out, labels)
                component_sums["ce"] += loss.item()

            else:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(device)
                labels_ = (labels % args.class_num_per_step).to(device)

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

                outputs = model(
                    visual=total_visual,
                    audio=total_audio,
                    out_feature_before_fusion=True,
                    out_attn_score=True,
                    out_uniform_visual=True,
                    return_dict=True,
                )

                out = outputs["logits"]
                audio_feature = outputs["audio_feature"]
                visual_feature = outputs["visual_feature"]
                uniform_visual_feature = outputs[
                    "uniform_visual_feature_raw"
                ]
                spatial_attn_score = outputs["spatial_attn_score"]
                temporal_attn_score = outputs["temporal_attn_score"]

                with torch.no_grad():
                    old_outputs = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        out_attn_score=True,
                        out_uniform_visual=True,
                        return_dict=True,
                    )
                    old_out = old_outputs["logits"].detach()
                    old_spatial_attn_score = old_outputs[
                        "spatial_attn_score"
                    ].detach()
                    old_temporal_attn_score = old_outputs[
                        "temporal_attn_score"
                    ].detach()
                    old_uniform_visual_feature = old_outputs[
                        "uniform_visual_feature_raw"
                    ].detach()

                instance_contra_loss = torch.zeros((), device=device)
                class_contra_loss = torch.zeros((), device=device)
                spatial_attn_dist_loss = torch.zeros((), device=device)
                temporal_attn_dist_loss = torch.zeros((), device=device)
                uniform_feature_dist_loss = torch.zeros((), device=device)

                # Keep the original Full AVCIL z1 contrastive pathway unchanged:
                # it continues to use the audio-guided visual z1 feature.
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

                # Preserve the original attention KD on all replay exemplars.
                if args.attn_score_distil:
                    begin = data_batch_size
                    end = data_batch_size + exemplar_data_batch_size

                    exem_spatial_attn_score = spatial_attn_score[
                        begin:end
                    ].transpose(2, 3)
                    exem_spatial_attn_score = exem_spatial_attn_score.reshape(
                        -1, exem_spatial_attn_score.shape[-1]
                    )

                    exem_old_spatial_attn_score = old_spatial_attn_score[
                        begin:end
                    ].transpose(2, 3)
                    exem_old_spatial_attn_score = (
                        exem_old_spatial_attn_score.reshape(
                            -1,
                            exem_old_spatial_attn_score.shape[-1],
                        )
                    )

                    exem_temporal_attn_score = temporal_attn_score[
                        begin:end
                    ].transpose(1, 2)
                    exem_temporal_attn_score = (
                        exem_temporal_attn_score.reshape(
                            -1,
                            exem_temporal_attn_score.shape[-1],
                        )
                    )

                    exem_old_temporal_attn_score = old_temporal_attn_score[
                        begin:end
                    ].transpose(1, 2)
                    exem_old_temporal_attn_score = (
                        exem_old_temporal_attn_score.reshape(
                            -1,
                            exem_old_temporal_attn_score.shape[-1],
                        )
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

                # New CL control for the newly used uniform branch.  It is
                # replay-only and starts once at least one unguided class is old.
                if (
                    args.uniform_feature_distil
                    and has_old_uniform_target
                ):
                    begin = data_batch_size
                    end = data_batch_size + exemplar_data_batch_size
                    uniform_feature_dist_loss = F.mse_loss(
                        uniform_visual_feature[begin:end],
                        old_uniform_visual_feature[begin:end],
                        reduction="mean",
                    )

                old_out = old_out[:, :last_step_out_class_num]

                curr_out = out[
                    :data_batch_size,
                    last_step_out_class_num:,
                ]
                loss_curr = CE_loss(
                    args.class_num_per_step,
                    curr_out,
                    labels_,
                )

                prev_out = out[
                    data_batch_size:
                    data_batch_size + exemplar_data_batch_size,
                    :last_step_out_class_num,
                ]
                loss_prev = CE_loss(
                    last_step_out_class_num,
                    prev_out,
                    exemplar_labels,
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
                        args.class_num_per_step
                        + last_step_out_class_num,
                        out,
                        torch.cat((labels, exemplar_labels)),
                    )

                loss_KD = torch.zeros(step, device=device)
                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step

                    soft_target = F.softmax(
                        old_out[:, start:end] / T,
                        dim=1,
                    )
                    output_log = F.log_softmax(
                        out[:, start:end] / T,
                        dim=1,
                    )
                    loss_KD[t] = F.kl_div(
                        output_log,
                        soft_target,
                        reduction='batchmean',
                    ) * (T ** 2)
                loss_KD = loss_KD.sum()

                loss = loss_CE + loss_KD

                if args.instance_contrastive:
                    loss = loss + args.lam_I * instance_contra_loss
                if args.class_contrastive:
                    loss = loss + args.lam_C * class_contra_loss

                attention_kd = torch.zeros((), device=device)
                if args.attn_score_distil:
                    attention_kd = (
                        args.lam * spatial_attn_dist_loss
                        + (1 - args.lam) * temporal_attn_dist_loss
                    )
                    loss = loss + attention_kd

                weighted_uniform_kd = torch.zeros((), device=device)
                if (
                    args.uniform_feature_distil
                    and has_old_uniform_target
                ):
                    weighted_uniform_kd = (
                        args.lam_U * uniform_feature_dist_loss
                    )
                    loss = loss + weighted_uniform_kd

                component_sums["ce"] += loss_CE.item()
                component_sums["logit_kd"] += loss_KD.item()
                component_sums["instance"] += (
                    args.lam_I * instance_contra_loss
                ).item() if args.instance_contrastive else 0.0
                component_sums["class"] += (
                    args.lam_C * class_contra_loss
                ).item() if args.class_contrastive else 0.0
                component_sums["attn"] += attention_kd.item()
                component_sums["uniform_kd"] += weighted_uniform_kd.item()

            model.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            num_steps += 1

        train_loss /= max(num_steps, 1)
        train_loss_list.append(train_loss)

        means = {
            key: value / max(num_steps, 1)
            for key, value in component_sums.items()
        }
        print(
            (
                'Epoch:{} train_loss:{:.5f} '
                'CE:{:.5f} logitKD:{:.5f} inst:{:.5f} '
                'class:{:.5f} attnKD:{:.5f} uniformKD:{:.5f}'
            ).format(
                epoch,
                train_loss,
                means["ce"],
                means["logit_kd"],
                means["instance"],
                means["class"],
                means["attn"],
                means["uniform_kd"],
            ),
            flush=True,
        )

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                val_visual = val_data[0].to(device)
                val_audio = val_data[1].to(device)
                val_out_logits = model(
                    visual=val_visual,
                    audio=val_audio,
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
            print(
                'Saving best model at Epoch {}'.format(epoch),
                flush=True,
            )
            if isinstance(model, nn.DataParallel):
                torch.save(
                    model.module,
                    './save/{}/step_{}_best_model.pkl'.format(
                        args.dataset, step
                    ),
                )
            else:
                torch.save(
                    model,
                    './save/{}/step_{}_best_model.pkl'.format(
                        args.dataset, step
                    ),
                )

        plt.figure()
        plt.plot(
            range(len(train_loss_list)),
            train_loss_list,
            label='train_loss',
        )
        plt.legend()
        plt.savefig(
            './save/fig/{}/train_loss_step_{}.png'.format(
                args.dataset, step
            )
        )
        plt.close()

        plt.figure()
        plt.plot(
            range(len(val_acc_list)),
            val_acc_list,
            label='val_acc',
        )
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

    # Offline class-wise guided/uniform visual routing.  Difficulty is
    # computed directly from all TRAIN z0 features before step-0 training.
    parser.add_argument('--offline_uniform', action='store_true', default=False)
    parser.add_argument('--offline_easy_ratio', type=float, default=0.30)
    parser.add_argument('--offline_hard_ratio', type=float, default=0.30)
    parser.add_argument('--offline_margin_eps', type=float, default=1e-8)
    parser.add_argument('--offline_difficulty_batch_size', type=int, default=128)

    # Preserve the newly used uniform visual z1 branch on replay exemplars.
    parser.add_argument('--uniform_feature_distil', action='store_true', default=False)
    parser.add_argument('--lam_U', type=float, default=1.0)

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

    # Offline means that all class difficulty states are computed once from
    # fixed TRAIN z0 features before any incremental optimization.  No
    # external difficulty table and no validation/test feature is used.
    if args.offline_uniform or args.uniform_feature_distil:
        (
            args.offline_unguided_class_ids,
            offline_z0_rows,
            offline_z0_contingency,
        ) = compute_offline_z0_difficulty(
            args=args,
            train_data_set=train_set,
            id_to_category=id_to_category,
        )
        if args.offline_uniform and not args.offline_unguided_class_ids:
            raise RuntimeError(
                "The z0 30/40/30 split selected no A:H / V:E class."
            )
        save_offline_z0_outputs(
            args=args,
            rows=offline_z0_rows,
            contingency=offline_z0_contingency,
            output_root=ckpts_root,
        )
    else:
        args.offline_unguided_class_ids = []

    print(
        "Resolved offline uniform classes from z0: {}".format(
            [
                {
                    "class_id": class_id,
                    "category_name": id_to_category[class_id],
                }
                for class_id in args.offline_unguided_class_ids
            ]
        ),
        flush=True,
    )

    # Re-seed so the additional deterministic z0 pass cannot alter model
    # initialization or minibatch shuffling relative to the baseline.
    setup_seed(args.seed)

    per_class_csv = os.path.join(metrics_root, "per_class_metrics.csv")
    if os.path.exists(per_class_csv):
        os.remove(per_class_csv)

    metrics_state = {
        "best_f1": {},
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