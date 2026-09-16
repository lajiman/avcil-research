import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_ours import IcaAVELoader, exemplarLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental_logit_guidance import IncreAudioVisualNet
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
    """Load trusted full-model checkpoints on both old and new PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def target_vs_rest_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    B(z,y) = z_y - logsumexp_{k != y} z_k
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, C)")
    if logits.shape[1] < 2:
        raise ValueError("target-vs-rest margin requires at least two classes")

    labels = labels.long()
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)

    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))
    rest_lse = torch.logsumexp(masked, dim=1)
    return true_logits - rest_lse


@torch.no_grad()
def estimate_guidance_contribution(
    model,
    dataset,
    class_ids,
    batch_size,
    num_workers,
    max_samples_per_class=64,
    sample_seed=0,
):
    """
    Estimate direct audio->visual guidance contribution for selected classes.

    For each labeled sample:
        C_i = B(logits_guided, y_i) - B(logits_uniform, y_i)

    No currently active gate is used in this measurement: the two endpoint
    logits are forced explicitly.

    Returns a dict keyed by class id with n/mean/std.
    """
    class_ids = [int(c) for c in class_ids]
    wanted = set(class_ids)
    if len(wanted) == 0:
        return {}

    core = unwrap_model(model)
    was_training = model.training
    model.eval()

    values = {c: [] for c in class_ids}
    generator = torch.Generator()
    generator.manual_seed(int(sample_seed))
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, dataset.__len__())),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        generator=generator,
    )

    for data, labels_cpu in loader:
        labels_cpu = labels_cpu.long()

        # Skip batches containing no target classes.
        keep_cpu = torch.zeros_like(labels_cpu, dtype=torch.bool)
        for c in class_ids:
            keep_cpu |= (labels_cpu == c)
        if not keep_cpu.any():
            continue

        visual = data[0][keep_cpu].to(device)
        audio = data[1][keep_cpu].to(device)
        labels = labels_cpu[keep_cpu].to(device)

        logits_g, logits_u = core.guided_uniform_logits(
            visual=visual, audio=audio
        )
        contribution = (
            target_vs_rest_margin(logits_g, labels)
            - target_vs_rest_margin(logits_u, labels)
        ).detach().cpu()

        labels_kept = labels.detach().cpu()
        for c in class_ids:
            if max_samples_per_class > 0:
                remaining = max_samples_per_class - len(values[c])
                if remaining <= 0:
                    continue
            else:
                remaining = None

            idx = torch.where(labels_kept == c)[0]
            if idx.numel() == 0:
                continue
            if remaining is not None:
                idx = idx[:remaining]
            values[c].extend(contribution[idx].tolist())

        if max_samples_per_class > 0 and all(
            len(values[c]) >= max_samples_per_class for c in class_ids
        ):
            break

    if was_training:
        model.train()

    result = {}
    for c in class_ids:
        arr = np.asarray(values[c], dtype=np.float64)
        if arr.size == 0:
            result[c] = {
                "n": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "stderr": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }
        else:
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            stderr = std / float(np.sqrt(arr.size)) if arr.size > 0 else float("nan")
            half = 1.96 * stderr if np.isfinite(stderr) else float("nan")
            result[c] = {
                "n": int(arr.size),
                "mean": mean,
                "std": std,
                "stderr": stderr,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            }
    return result


def write_guidance_rows(args, step, epoch, stats, old_gates, new_gates, phase):
    out_dir = os.path.join("./save/metrics", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "logit_guidance_updates.csv")

    header = [
        "step", "epoch", "phase", "class_id", "n",
        "contribution_mean", "contribution_std", "contribution_stderr",
        "contribution_ci95_low", "contribution_ci95_high",
        "threshold", "old_gate", "new_gate",
    ]
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for c in sorted(stats.keys()):
            s = stats[c]
            writer.writerow({
                "step": int(step),
                "epoch": int(epoch),
                "phase": phase,
                "class_id": int(c),
                "n": int(s["n"]),
                "contribution_mean": s["mean"],
                "contribution_std": s["std"],
                "contribution_stderr": s["stderr"],
                "contribution_ci95_low": s["ci95_low"],
                "contribution_ci95_high": s["ci95_high"],
                "threshold": float(args.guidance_threshold),
                "old_gate": float(old_gates[c]),
                "new_gate": float(new_gates[c]),
            })


@torch.no_grad()
def update_new_class_binary_gates(args, model, dataset, step, epoch):
    """
    Stage-B policy:
      * only classes introduced at the CURRENT incremental step are estimated;
      * old class gates are kept fixed;
      * C_c < threshold -> g_c = 0
      * C_c >= threshold -> g_c = 1

    This deliberately avoids any CL-specific old-class adaptation. That is
    reserved for the later contribution-distillation experiment.
    """
    core = unwrap_model(model)
    lo = step * args.class_num_per_step
    hi = (step + 1) * args.class_num_per_step
    class_ids = list(range(lo, hi))

    stats = estimate_guidance_contribution(
        model=model,
        dataset=dataset,
        class_ids=class_ids,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=args.guidance_max_samples_per_class,
        sample_seed=args.seed + 100003 * step + 1009 * epoch,
    )

    old_gates = core.get_class_gates().detach().cpu().numpy()
    new_gates = old_gates.copy()

    for c in class_ids:
        s = stats[c]
        if s["n"] < args.guidance_min_samples_per_class or not np.isfinite(s["mean"]):
            # Insufficient evidence: preserve the current gate.
            continue
        new_gates[c] = 0.0 if s["mean"] < args.guidance_threshold else 1.0

    core.set_class_gates(torch.tensor(new_gates, dtype=torch.float32))

    write_guidance_rows(
        args=args,
        step=step,
        epoch=epoch,
        stats=stats,
        old_gates=old_gates,
        new_gates=new_gates,
        phase="refresh_new_classes",
    )

    changed = int(np.sum(old_gates != new_gates))
    off = int(np.sum(new_gates[:hi] == 0))
    print(
        "Logit-guidance refresh step {} epoch {}: changed={}, "
        "off among seen classes={}/{}, new-class mean C={}".format(
            step,
            epoch,
            changed,
            off,
            hi,
            ", ".join(
                "{}:{:+.4f}".format(c, stats[c]["mean"])
                for c in class_ids
                if np.isfinite(stats[c]["mean"])
            ),
        ),
        flush=True,
    )


def guidance_state_path(args, step):
    return os.path.join(
        "./save", args.dataset, "step_{}_logit_guidance_state.pt".format(step)
    )


def save_guidance_state(args, step, gates, source):
    state = {
        "step": int(step),
        "num_classes": int(len(gates)),
        "gates": torch.as_tensor(gates, dtype=torch.float32).cpu(),
        "threshold": float(args.guidance_threshold),
        "policy": "binary: C_AtoV < threshold -> 0, else 1",
        "source": source,
    }
    torch.save(state, guidance_state_path(args, step))


def load_previous_guidance_state(args, step):
    if step <= 0:
        return None
    path = guidance_state_path(args, step - 1)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing previous guidance state: {}. "
            "Run the preceding incremental step with this script first.".format(path)
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    expected = step * args.class_num_per_step
    if int(state["num_classes"]) != expected:
        raise RuntimeError(
            "Previous guidance state has {} classes, expected {}".format(
                state["num_classes"], expected
            )
        )
    return state


@torch.no_grad()
def build_step0_guidance_state(args, train_data_set):
    """
    Step 0 remains a pure baseline model. After its best checkpoint has been
    selected, estimate C_AtoV on that frozen best model ONLY to prepare gates
    for step 1. The step-0 checkpoint itself is not modified.
    """
    ckpt = "./save/{}/step_0_best_model.pkl".format(args.dataset)
    model = load_full_model(ckpt, map_location=device).to(device)
    core = unwrap_model(model)
    core.set_logit_guidance_enabled(False)

    class_ids = list(range(args.class_num_per_step))
    stats = estimate_guidance_contribution(
        model=model,
        dataset=train_data_set,
        class_ids=class_ids,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=args.guidance_max_samples_per_class,
        sample_seed=args.seed + 7919,
    )

    gates = np.ones(args.class_num_per_step, dtype=np.float32)
    for c in class_ids:
        s = stats[c]
        if s["n"] >= args.guidance_min_samples_per_class and np.isfinite(s["mean"]):
            gates[c] = 0.0 if s["mean"] < args.guidance_threshold else 1.0

    old_gates = np.ones_like(gates)
    write_guidance_rows(
        args=args,
        step=0,
        epoch=-1,
        stats=stats,
        old_gates=old_gates,
        new_gates=gates,
        phase="step0_best_prepare_step1",
    )
    save_guidance_state(
        args=args,
        step=0,
        gates=gates,
        source="step0_best_checkpoint_contribution",
    )

    print(
        "Prepared step-0 logit-guidance state for step 1: "
        "off={}/{}".format(int(np.sum(gates == 0)), len(gates)),
        flush=True,
    )

    del model
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
        # Hard requirement: step 0 is EXACT baseline training.
        unwrap_model(model).set_logit_guidance_enabled(False)
    else:
        prev_ckpt = "./save/{}/step_{}_best_model.pkl".format(args.dataset, step - 1)
        model = load_full_model(prev_ckpt, map_location="cpu")
        old_model = load_full_model(prev_ckpt, map_location="cpu")

        # Apply the previous step's validated guidance state to the student.
        prev_state = load_previous_guidance_state(args, step)
        prev_gates = prev_state["gates"].float()

        # Apply the same previously established class policy to both student
        # and teacher. Otherwise logit KD would force the student back toward
        # ungated/guided teacher logits and directly oppose the gate itself.
        model.set_class_gates(prev_gates)
        model.incremental_classifier(step_out_class_num)
        model.set_logit_guidance_enabled(args.logit_guidance)

        old_model.set_class_gates(prev_gates)
        old_model.set_logit_guidance_enabled(args.logit_guidance)

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

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0

    for epoch in range(args.max_epoches):
        # Stage B: after baseline warm-up, periodically re-measure ONLY the
        # newly introduced classes. Old class gates remain fixed.
        if (
            args.logit_guidance
            and step >= args.guidance_start_step
            and epoch >= args.guidance_warmup_epochs
            and (
                (epoch - args.guidance_warmup_epochs)
                % args.guidance_refresh_interval == 0
            )
        ):
            update_new_class_binary_gates(
                args=args,
                model=model,
                dataset=train_data_set,
                step=step,
                epoch=epoch,
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

                # Exact baseline path because guidance is disabled.
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
                exemplar_labels = exemplar_labels.to(device)

                data_batch_size = labels_.shape[0]
                exemplar_data_batch_size = exemplar_labels.shape[0]

                visual = data[0]
                audio = data[1]
                exemplar_visual = exemplar_data[0]
                exemplar_audio = exemplar_data[1]

                total_visual = torch.cat((visual, exemplar_visual)).to(device)
                total_audio = torch.cat((audio, exemplar_audio)).to(device)

                out, audio_feature, visual_feature, spatial_attn_score, temporal_attn_score = model(
                    visual=total_visual,
                    audio=total_audio,
                    out_feature_before_fusion=True,
                    out_attn_score=True,
                )

                with torch.no_grad():
                    old_out, old_spatial_attn_score, old_temporal_attn_score = old_model(
                        visual=total_visual,
                        audio=total_audio,
                        out_attn_score=True,
                    )
                    old_out = old_out.detach()
                    old_spatial_attn_score = old_spatial_attn_score.detach()
                    old_temporal_attn_score = old_temporal_attn_score.detach()

                # Keep the two original z1 losses exactly as in Full AVCIL.
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

                # Keep original raw-attention distillation exactly unchanged.
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
                        reduction="sum",
                    ) / exemplar_data_batch_size

                    temporal_attn_dist_loss = F.kl_div(
                        exem_temporal_attn_score.log(),
                        exem_old_temporal_attn_score,
                        reduction="sum",
                    ) / exemplar_data_batch_size

                old_out = old_out[:, :last_step_out_class_num]

                curr_out = out[
                    :data_batch_size, last_step_out_class_num:
                ]
                loss_curr = CE_loss(
                    args.class_num_per_step, curr_out, labels_
                )

                prev_out = out[
                    data_batch_size:data_batch_size + exemplar_data_batch_size,
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
                    args.dataset == "AVE"
                    and args.class_num_per_step == 4
                    and step == 1
                ):
                    loss_CE = CE_loss(
                        args.class_num_per_step + last_step_out_class_num,
                        out,
                        torch.cat((labels, exemplar_labels)),
                    )

                loss_KD = torch.zeros(step, device=device)
                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step
                    soft_target = F.softmax(
                        old_out[:, start:end] / T, dim=1
                    )
                    output_log = F.log_softmax(
                        out[:, start:end] / T, dim=1
                    )
                    loss_KD[t] = F.kl_div(
                        output_log, soft_target, reduction="batchmean"
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
                        + (1 - args.lam) * temporal_attn_dist_loss
                    )

            model.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            num_steps += 1

        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print(
            "Epoch:{} train_loss:{:.5f}".format(epoch, train_loss),
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
            "Epoch:{} val_res:{:.6f} ".format(epoch, val_top1),
            flush=True,
        )

        # Do NOT reset best_val_res when a gate changes. A pre-gate warm-up
        # checkpoint is allowed to win; this keeps model selection fair.
        if val_top1 > best_val_res:
            best_val_res = val_top1
            print(
                "Saving best model at Epoch {}".format(epoch),
                flush=True,
            )
            if torch.cuda.device_count() > 1:
                torch.save(
                    model.module,
                    "./save/{}/step_{}_best_model.pkl".format(
                        args.dataset, step
                    ),
                )
            else:
                torch.save(
                    model,
                    "./save/{}/step_{}_best_model.pkl".format(
                        args.dataset, step
                    ),
                )

        plt.figure()
        plt.plot(
            range(len(train_loss_list)),
            train_loss_list,
            label="train_loss",
        )
        plt.legend()
        plt.savefig(
            "./save/fig/{}/train_loss_step_{}.png".format(
                args.dataset, step
            )
        )
        plt.close()

        plt.figure()
        plt.plot(
            range(len(val_acc_list)),
            val_acc_list,
            label="val_acc",
        )
        plt.legend()
        plt.savefig(
            "./save/fig/{}/val_acc_step_{}.png".format(
                args.dataset, step
            )
        )
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, opt, epoch)

    # ------------------------------------------------------------------
    # Cross-step state
    # ------------------------------------------------------------------
    if args.logit_guidance:
        if step == 0:
            # Pure baseline step 0; only AFTER model selection measure C_AtoV
            # to prepare the gate state for step 1.
            build_step0_guidance_state(args, train_data_set)
        else:
            # Use exactly the gates that belong to the selected best checkpoint.
            # No post-hoc gate overwrite is allowed.
            best_path = "./save/{}/step_{}_best_model.pkl".format(
                args.dataset, step
            )
            best_model = load_full_model(best_path, map_location="cpu")
            best_core = unwrap_model(best_model)
            gates = (
                best_core.get_class_gates()
                .detach()
                .cpu()
                .numpy()[:step_out_class_num]
            )
            save_guidance_state(
                args=args,
                step=step,
                gates=gates,
                source="selected_best_checkpoint_gate_buffer",
            )
            del best_model

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

    parser.add_argument("--logit_guidance", action="store_true", default=False)
    parser.add_argument("--guidance_start_step", type=int, default=1,
                        help="Step 0 is baseline when this is 1.")
    parser.add_argument("--guidance_warmup_epochs", type=int, default=20)
    parser.add_argument("--guidance_refresh_interval", type=int, default=20)
    parser.add_argument("--guidance_threshold", type=float, default=0.0,
                        help="C_AtoV below this value disables guidance.")
    parser.add_argument("--guidance_max_samples_per_class", type=int, default=64)
    parser.add_argument("--guidance_min_samples_per_class", type=int, default=16)

    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dump_tsne", action="store_true", help="If set, dump t-SNE plots for each step")
    parser.add_argument("--tsne_feature", type=str, default="logits",
                        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"])
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    

    args = parser.parse_args()

    if args.logit_guidance:
        if args.guidance_start_step < 1:
            raise ValueError(
                "For this Stage-B experiment guidance_start_step must be >= 1 "
                "so step 0 remains a pure baseline."
            )
        if args.guidance_refresh_interval <= 0:
            raise ValueError("guidance_refresh_interval must be > 0")
        if args.guidance_warmup_epochs < 0:
            raise ValueError("guidance_warmup_epochs must be >= 0")
        if args.guidance_warmup_epochs >= args.max_epoches:
            raise ValueError("guidance_warmup_epochs must be < max_epoches")
        if args.guidance_max_samples_per_class == 0:
            raise ValueError(
                "guidance_max_samples_per_class must be -1 (all) or > 0"
            )
        if args.guidance_min_samples_per_class <= 0:
            raise ValueError("guidance_min_samples_per_class must be > 0")

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