"""Modular AVCIL + CrossSDC + optional CMR experiments.

Design principles
-----------------
1. ``rd_mode=crosssdc`` follows the working CrossSDC training path:
   - original tuple model forward;
   - original CrossSDC formulas;
   - original loss-addition order;
   - no prototype bank, CMR graph, trust, or Need computation.
2. Optional modules are entered only when their mode enables them.
3. New method code is isolated in ``rd_crosssdc/`` so later changes can target a
   small module instead of rewriting the full training script.
"""

import argparse
import os
import random
import sys
from datetime import datetime
from itertools import cycle

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

from dataloader_ours import IcaAVELoader, exemplarLoader
from model.audio_visual_model_incremental import IncreAudioVisualNet
from tsne_plotter import make_tsne_plots_for_step

from rd_crosssdc.diagnostics import append_csv_row, save_dynamic_weights, save_static_bank
from rd_crosssdc.exact_losses import (
    cal_contrastive_loss,
    ce_loss,
    class_contrastive_loss,
    cross_sdc_instance_loss,
    cross_sdc_z1_loss,
    weighted_cross_sdc_class_loss,
)
from rd_crosssdc.metrics import detailed_test, save_json
from rd_crosssdc.rd_method import (
    AdaptiveWeightController,
    build_old_teacher_prototype_bank,
    capture_rng_state,
    cmr_loss,
    compute_margin_terms,
    restore_rng_state,
)
from rd_crosssdc.cmr_penalties import CMR_PENALTIES


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODE_CROSSSDC = "crosssdc"
MODE_CROSSSDC_CMR = "crosssdc_cmr"
MODE_ADAPTIVE = "adaptive_crosssdc_cmr"
VALID_MODES = (MODE_CROSSSDC, MODE_CROSSSDC_CMR, MODE_ADAPTIVE)


def setup_seed(seed):
    """Original seed setup from the working training script."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def boolean_string(value):
    if value not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return value == "True"


def run_name(args):
    """Output identifier; dataset remains unchanged for data/model semantics."""
    return args.experiment_name if args.experiment_name else args.dataset


def checkpoint_path(args, step):
    return "./save/{}/step_{}_best_model.pkl".format(run_name(args), step)


def figure_dir(args):
    return "./save/fig/{}/".format(run_name(args))


def metrics_dir(args):
    return "./save/metrics/{}/".format(run_name(args))


def uses_cmr(args):
    return args.rd_mode in (MODE_CROSSSDC_CMR, MODE_ADAPTIVE) and args.lam_cmr > 0


def uses_adaptive_weights(args):
    return args.rd_mode == MODE_ADAPTIVE


def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()


def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]["lr"]
        new_lr = current_lr * 0.1
        print("Reduce lr from {} to {}".format(current_lr, new_lr))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def _prepare_optional_rd_state(
    args,
    step,
    old_model,
    exemplar_set,
    id_to_category,
):
    """Build only the state required by the selected non-control mode."""
    if not uses_cmr(args):
        return None, None

    # Step-level diagnostic passes must not consume RNG used by training loaders.
    rng_state = capture_rng_state()
    try:
        bank = build_old_teacher_prototype_bank(
            old_model=old_model,
            exemplar_set=exemplar_set,
            num_old_classes=step * args.class_num_per_step,
            batch_size=args.exemplar_batch_size,
            num_workers=args.num_workers,
            device=device,
            margin_temperature=args.rd_margin_temperature,
            compute_trust=uses_adaptive_weights(args),
            trust_shrinkage_beta=args.rd_trust_shrinkage_beta,
        )
    finally:
        restore_rng_state(rng_state)

    if not uses_adaptive_weights(args):
        return bank, None

    if bank.trust_a_from_v is None or bank.trust_v_from_a is None:
        raise RuntimeError("Adaptive mode requires teacher trust")

    controller = AdaptiveWeightController(
        trust_a_from_v=bank.trust_a_from_v,
        trust_v_from_a=bank.trust_v_from_a,
        alpha=args.rd_class_weight_alpha,
        trust_offset=args.rd_trust_offset,
        trust_gamma=args.rd_trust_gamma,
        need_delta=args.rd_need_delta,
        need_eta=args.rd_need_eta,
        ema_momentum=args.rd_need_ema_momentum,
        min_weight=args.rd_weight_min,
        max_weight=args.rd_weight_max,
    )

    out_root = os.path.join(metrics_dir(args), "rd_crosssdc")
    save_static_bank(
        path=os.path.join(out_root, "step_{}_static_trust.csv".format(step)),
        step=step,
        counts=bank.counts.detach().cpu(),
        reliability_a=bank.reliability_a_from_v.detach().cpu(),
        reliability_v=bank.reliability_v_from_a.detach().cpu(),
        trust_a=bank.trust_a_from_v.detach().cpu(),
        trust_v=bank.trust_v_from_a.detach().cpu(),
        id_to_category=id_to_category,
    )
    return bank, controller


def train(args, step, train_data_set, val_data_set, exemplar_set, id_to_category):
    """Train one class-incremental step.

    The body intentionally preserves the working CrossSDC order.  New branches
    are visibly marked and are skipped entirely in pure CrossSDC mode.
    """
    # Exactly the same as original AVCIL implementation.
    distillation_temperature = 2

    train_loader = DataLoader(
        train_data_set,
        batch_size=min(args.train_batch_size, len(train_data_set)),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data_set,
        batch_size=min(args.infer_batch_size, len(val_data_set)),
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
        model = torch.load(checkpoint_path(args, step - 1))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load(checkpoint_path(args, step - 1))

        exemplar_loader = DataLoader(
            exemplar_set,
            batch_size=min(args.exemplar_batch_size, len(exemplar_set)),
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

    # Preserve original placement of optimizer construction.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Pure CrossSDC returns immediately here with (None, None); no optional
    # prototype/trust/CMR computation is executed.
    prototype_bank, adaptive_controller = _prepare_optional_rd_state(
        args=args,
        step=step,
        old_model=old_model,
        exemplar_set=exemplar_set,
        id_to_category=id_to_category,
    ) if step > 0 else (None, None)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0

    epoch_csv = os.path.join(metrics_dir(args), "rd_crosssdc", "epoch_summary.csv")
    epoch_header = [
        "step", "epoch", "rd_mode",
        "cmr_penalty", "cmr_scale", "cmr_tolerance",
        "train_loss",

        # Keep the remaining existing columns unchanged.
        "cross_sdc_i", "cross_sdc_c", "weighted_cross_sdc",
        "cmr", "weighted_cmr",
        "cmr_active_a_from_v", "cmr_active_v_from_a",
        "mean_deficit_a_from_v", "mean_deficit_v_from_a",
        "val_acc",
    ]

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0

        cross_i_sum = 0.0
        cross_c_sum = 0.0
        cmr_sum = 0.0
        cmr_active_a_sum = 0.0
        cmr_active_v_sum = 0.0
        deficit_a_sum = 0.0
        deficit_v_sum = 0.0

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
                out, _, _ = model(
                    visual=visual,
                    audio=audio,
                    out_feature_before_fusion=True,
                )
                loss = ce_loss(step_out_class_num, out, labels)

            else:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(device)
                local_labels = (labels % args.class_num_per_step).to(device)

                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(device).long()

                data_batch_size = local_labels.shape[0]
                exemplar_batch_size = exemplar_labels.shape[0]

                visual = data[0]
                audio = data[1]
                exemplar_visual = exemplar_data[0]
                exemplar_audio = exemplar_data[1]
                total_visual = torch.cat((visual, exemplar_visual)).to(device)
                total_audio = torch.cat((audio, exemplar_audio)).to(device)

                # Original tuple forward interface.
                out, audio_feature, visual_feature, spatial_attn_score, temporal_attn_score = model(
                    visual=total_visual,
                    audio=total_audio,
                    out_feature_before_fusion=True,
                    out_attn_score=True,
                )
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

                # -----------------------------------------------------------------
                # CrossSDC section
                # -----------------------------------------------------------------
                exemplar_start = data_batch_size
                exemplar_end = data_batch_size + exemplar_batch_size
                cur_exem_audio = audio_feature[exemplar_start:exemplar_end]
                cur_exem_visual = visual_feature[exemplar_start:exemplar_end]
                old_exem_audio = old_audio_feature[exemplar_start:exemplar_end]
                old_exem_visual = old_visual_feature[exemplar_start:exemplar_end]

                if uses_adaptive_weights(args):
                    # Instance term remains the exact original CrossSDC-I.
                    cross_sdc_inst_loss = cross_sdc_instance_loss(
                        cur_audio=cur_exem_audio,
                        cur_visual=cur_exem_visual,
                        old_audio=old_exem_audio,
                        old_visual=old_exem_visual,
                        temperature=args.cross_sdc_temperature,
                    )
                    # Class weights depend only on teacher Trust.
                    # Student margin deficits do not modulate CrossSDC-C weights.
                    cross_sdc_cls_loss, class_stats = weighted_cross_sdc_class_loss(
                        cur_audio=cur_exem_audio,
                        cur_visual=cur_exem_visual,
                        old_audio=old_exem_audio,
                        old_visual=old_exem_visual,
                        labels=exemplar_labels,
                        class_weight_a_from_v=adaptive_controller.class_weight_a,
                        class_weight_v_from_a=adaptive_controller.class_weight_v,
                        temperature=args.cross_sdc_temperature,
                    )
                else:
                    # Exact original working CrossSDC function.
                    cross_sdc_inst_loss, cross_sdc_cls_loss = cross_sdc_z1_loss(
                        cur_audio=cur_exem_audio,
                        cur_visual=cur_exem_visual,
                        old_audio=old_exem_audio,
                        old_visual=old_exem_visual,
                        labels=exemplar_labels,
                        temperature=args.cross_sdc_temperature,
                    )
                    class_stats = None

                # -----------------------------------------------------------------
                # Optional CMR section: completely skipped in pure CrossSDC mode.
                # -----------------------------------------------------------------
                if uses_cmr(args):
                    margin_terms = compute_margin_terms(
                        current_audio=cur_exem_audio,
                        current_visual=cur_exem_visual,
                        old_audio=old_exem_audio,
                        old_visual=old_exem_visual,
                        labels=exemplar_labels,
                        bank=prototype_bank,
                        temperature=args.rd_margin_temperature,
                        tolerance=args.rd_margin_tolerance,
                    )

                    if adaptive_controller is None:
                        num_old_classes = step * args.class_num_per_step
                        cmr_weight_a = torch.ones(num_old_classes, device=device)
                        cmr_weight_v = torch.ones(num_old_classes, device=device)
                    else:
                        cmr_weight_a = adaptive_controller.cmr_weight_a
                        cmr_weight_v = adaptive_controller.cmr_weight_v
                        adaptive_controller.accumulate(exemplar_labels, margin_terms)

                    # Current margin violations encode the need for correction.
                    # Class weights encode teacher Trust only.
                    current_cmr_loss, current_cmr_stats = cmr_loss(
                        terms=margin_terms,
                        labels=exemplar_labels,
                        class_weight_a_from_v=cmr_weight_a,
                        class_weight_v_from_a=cmr_weight_v,

                        penalty=args.rd_cmr_penalty,
                        penalty_scale=args.rd_cmr_scale,
                        tolerance=args.rd_margin_tolerance,
                    )
                else:
                    current_cmr_loss = None
                    current_cmr_stats = None

                if args.attn_score_distil:
                    exem_spatial_attn_score = spatial_attn_score[
                        data_batch_size:data_batch_size + exemplar_batch_size
                    ].transpose(2, 3)
                    exem_spatial_attn_score = exem_spatial_attn_score.reshape(
                        -1, exem_spatial_attn_score.shape[-1]
                    )

                    exem_old_spatial_attn_score = old_spatial_attn_score[
                        data_batch_size:data_batch_size + exemplar_batch_size
                    ].transpose(2, 3)
                    exem_old_spatial_attn_score = exem_old_spatial_attn_score.reshape(
                        -1, exem_old_spatial_attn_score.shape[-1]
                    )

                    exem_temporal_attn_score = temporal_attn_score[
                        data_batch_size:data_batch_size + exemplar_batch_size
                    ].transpose(1, 2)
                    exem_temporal_attn_score = exem_temporal_attn_score.reshape(
                        -1, exem_temporal_attn_score.shape[-1]
                    )

                    exem_old_temporal_attn_score = old_temporal_attn_score[
                        data_batch_size:data_batch_size + exemplar_batch_size
                    ].transpose(1, 2)
                    exem_old_temporal_attn_score = exem_old_temporal_attn_score.reshape(
                        -1, exem_old_temporal_attn_score.shape[-1]
                    )

                    spatial_attn_dist_loss = F.kl_div(
                        exem_spatial_attn_score.log(),
                        exem_old_spatial_attn_score,
                        reduction="sum",
                    ) / exemplar_batch_size
                    temporal_attn_dist_loss = F.kl_div(
                        exem_temporal_attn_score.log(),
                        exem_old_temporal_attn_score,
                        reduction="sum",
                    ) / exemplar_batch_size

                old_out = old_out[:, :last_step_out_class_num]
                curr_out = out[:data_batch_size, last_step_out_class_num:]
                loss_curr = ce_loss(args.class_num_per_step, curr_out, local_labels)

                prev_out = out[
                    data_batch_size:data_batch_size + exemplar_batch_size,
                    :last_step_out_class_num,
                ]
                loss_prev = ce_loss(last_step_out_class_num, prev_out, exemplar_labels)

                loss_CE = (
                    loss_curr * data_batch_size + loss_prev * exemplar_batch_size
                ) / (data_batch_size + exemplar_batch_size)

                if args.dataset == "AVE" and args.class_num_per_step == 4 and step == 1:
                    loss_CE = ce_loss(
                        args.class_num_per_step + last_step_out_class_num,
                        out,
                        torch.cat((labels, exemplar_labels)),
                    )

                # exactly the same as the original AVCIL implementation.
                loss_KD = torch.zeros(step).to(device)
                for task_id in range(step):
                    start = task_id * args.class_num_per_step
                    end = (task_id + 1) * args.class_num_per_step
                    soft_target = F.softmax(
                        old_out[:, start:end] / distillation_temperature, dim=1
                    )
                    output_log = F.log_softmax(
                        out[:, start:end] / distillation_temperature, dim=1
                    )
                    loss_KD[task_id] = F.kl_div(
                        output_log, soft_target, reduction="batchmean"
                    ) * (distillation_temperature ** 2)
                loss_KD = loss_KD.sum()

                # Preserve the working CrossSDC addition order.
                loss = loss_CE + loss_KD
                if args.instance_contrastive:
                    loss += args.lam_I * instance_contra_loss
                if args.class_contrastive:
                    loss += args.lam_C * class_contra_loss
                loss += (
                    args.lam_cross_sdc_i * cross_sdc_inst_loss
                    + args.lam_cross_sdc_c * cross_sdc_cls_loss
                )
                if uses_cmr(args):
                    loss += args.lam_cmr * current_cmr_loss
                if args.attn_score_distil:
                    loss += (
                        args.lam * spatial_attn_dist_loss
                        + (1.0 - args.lam) * temporal_attn_dist_loss
                    )

                cross_i_sum += cross_sdc_inst_loss.item()
                cross_c_sum += cross_sdc_cls_loss.item()
                if uses_cmr(args):
                    cmr_sum += current_cmr_loss.item()
                    cmr_active_a_sum += current_cmr_stats.active_a_from_v.item()
                    cmr_active_v_sum += current_cmr_stats.active_v_from_a.item()
                    deficit_a_sum += current_cmr_stats.mean_deficit_a_from_v.item()
                    deficit_v_sum += current_cmr_stats.mean_deficit_v_from_a.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_steps += 1

        train_loss /= max(num_steps, 1)
        train_loss_list.append(train_loss)
        print("Epoch:{} train_loss:{:.5f}".format(epoch, train_loss), flush=True)

        if step > 0:
            avg_cross_i = cross_i_sum / max(num_steps, 1)
            avg_cross_c = cross_c_sum / max(num_steps, 1)
            weighted_cross = (
                args.lam_cross_sdc_i * avg_cross_i
                + args.lam_cross_sdc_c * avg_cross_c
            )
            print(
                "Epoch:{} cross_sdc_inst:{:.5f} cross_sdc_cls:{:.5f} "
                "weighted_cross_sdc:{:.5f}".format(
                    epoch, avg_cross_i, avg_cross_c, weighted_cross
                ),
                flush=True,
            )
        else:
            avg_cross_i = 0.0
            avg_cross_c = 0.0
            weighted_cross = 0.0

        if step > 0 and uses_cmr(args):
            avg_cmr = cmr_sum / max(num_steps, 1)
            avg_active_a = cmr_active_a_sum / max(num_steps, 1)
            avg_active_v = cmr_active_v_sum / max(num_steps, 1)
            avg_deficit_a = deficit_a_sum / max(num_steps, 1)
            avg_deficit_v = deficit_v_sum / max(num_steps, 1)
            print(
                "Epoch:{} cmr:{:.6f} weighted_cmr:{:.6f} "
                "violation_ratio(A<-V/V<-A):{:.4f}/{:.4f} "
                "deficit(A<-V/V<-A):{:.6f}/{:.6f}".format(
                    epoch,
                    avg_cmr,
                    args.lam_cmr * avg_cmr,
                    avg_active_a,
                    avg_active_v,
                    avg_deficit_a,
                    avg_deficit_v,
                ),
                flush=True,
            )
        else:
            avg_cmr = 0.0
            avg_active_a = 0.0
            avg_active_v = 0.0
            avg_deficit_a = 0.0
            avg_deficit_v = 0.0

        # Update Need diagnostics after the full epoch.
        # Trust-only class weights remain fixed within the incremental step.
        if adaptive_controller is not None:
            adaptive_controller.end_epoch()
            snapshot = adaptive_controller.snapshot()
            dynamic_path = os.path.join(
                metrics_dir(args),
                "rd_crosssdc",
                "step_{}_epoch_{}_weights.csv".format(step, epoch),
            )
            save_dynamic_weights(
                path=dynamic_path,
                step=step,
                epoch=epoch,
                snapshot=snapshot,
                id_to_category=id_to_category,
            )

        # Original validation path.
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
                    val_out_logits = model(visual=val_visual, audio=val_audio)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat(
                    (all_val_out_logits, val_out_logits), dim=0
                )
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)

        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print("Epoch:{} val_res:{:.6f} ".format(epoch, val_top1), flush=True)

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print("Saving best model at Epoch {}".format(epoch), flush=True)
            if torch.cuda.device_count() > 1:
                torch.save(model.module, checkpoint_path(args, step))
            else:
                torch.save(model, checkpoint_path(args, step))

        append_csv_row(
            path=epoch_csv,
            fieldnames=epoch_header,
            row={
                "step": step,
                "epoch": epoch,
                "rd_mode": args.rd_mode,
                "cmr_penalty": args.rd_cmr_penalty,
                "cmr_scale": args.rd_cmr_scale,
                "cmr_tolerance": args.rd_margin_tolerance,
                "train_loss": train_loss,
                "cross_sdc_i": avg_cross_i,
                "cross_sdc_c": avg_cross_c,
                "weighted_cross_sdc": weighted_cross,
                "cmr": avg_cmr,
                "weighted_cmr": args.lam_cmr * avg_cmr,
                "cmr_active_a_from_v": avg_active_a,
                "cmr_active_v_from_a": avg_active_v,
                "mean_deficit_a_from_v": avg_deficit_a,
                "mean_deficit_v_from_a": avg_deficit_v,
                "val_acc": val_top1,
            },
        )

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label="train_loss")
        plt.legend()
        plt.savefig(
            os.path.join(figure_dir(args), "train_loss_step_{}.png".format(step))
        )
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label="val_acc")
        plt.legend()
        plt.savefig(
            os.path.join(figure_dir(args), "val_acc_step_{}.png".format(step))
        )
        plt.close()

        if args.lr_decay and step > 0:
            adjust_learning_rate(args, optimizer, epoch)


def dataset_type(value):
    if value in ["AVE", "ksounds"]:
        return value
    if "VGGSound" in value:
        return value
    raise argparse.ArgumentTypeError(
        "dataset must be 'AVE', 'ksounds', or contain 'VGGSound'"
    )


def build_parser():
    parser = argparse.ArgumentParser()

    # Original dataset/model/training arguments.
    parser.add_argument("--dataset", type=dataset_type, default="AVE")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--modality", type=str, default="audio-visual", choices=["audio-visual"]
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        default="/mnt/data2/wpian/dataset/VGGSound",
    )
    parser.add_argument("--meta_root", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument("--exemplar_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epoches", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=28)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[500], nargs="+")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--lam_I", type=float, default=0.5)
    parser.add_argument("--lam_C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_num_per_step", type=int, default=7)
    parser.add_argument("--memory_size", type=int, default=340)
    parser.add_argument("--instance_contrastive", action="store_true", default=False)
    parser.add_argument("--class_contrastive", action="store_true", default=False)
    parser.add_argument("--attn_score_distil", action="store_true", default=False)
    parser.add_argument("--instance_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--class_contrastive_temperature", type=float, default=0.1)

    # Original CrossSDC arguments and coefficients.
    parser.add_argument("--cross_sdc", action="store_true", default=False)
    parser.add_argument("--cross_sdc_temperature", type=float, default=0.05)
    parser.add_argument("--lam_cross_sdc_i", type=float, default=0.1)
    parser.add_argument("--lam_cross_sdc_c", type=float, default=0.3)

    # Three controlled experiment modes.
    parser.add_argument("--rd_mode", type=str, choices=VALID_MODES, default=MODE_CROSSSDC)
    parser.add_argument("--lam_cmr", type=float, default=0.1)
    parser.add_argument("--rd_margin_temperature", type=float, default=0.1)
    parser.add_argument("--rd_margin_tolerance", type=float, default=0.0)
    parser.add_argument("--rd_cmr_penalty",type=str,choices=CMR_PENALTIES,default="hinge",help="Per-sample CMR objective; weights are configured separately.",)
    parser.add_argument("--rd_cmr_scale",type=float,default=1.0,help=("Positive curvature scale for exp/softplus/log1p; ""unused by hinge/direct."),)

    # Adaptive mode only. These parameters do not execute in the other modes.
    parser.add_argument("--rd_class_weight_alpha", type=float, default=0.5)
    parser.add_argument("--rd_trust_offset", type=float, default=0.05)
    parser.add_argument("--rd_trust_gamma", type=float, default=1.0)
    parser.add_argument("--rd_need_delta", type=float, default=0.05)
    parser.add_argument("--rd_need_eta", type=float, default=0.5)
    parser.add_argument("--rd_need_ema_momentum", type=float, default=0.9)
    parser.add_argument("--rd_weight_min", type=float, default=0.5)
    parser.add_argument("--rd_weight_max", type=float, default=2.0)
    parser.add_argument("--rd_trust_shrinkage_beta", type=float, default=10.0)

    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--dump_tsne", action="store_true")
    parser.add_argument(
        "--tsne_feature",
        type=str,
        default="logits",
        choices=["audio", "visual", "joint_mean", "joint_concat", "logits"],
    )
    parser.add_argument("--tsne_max_points_per_class", type=int, default=50)
    parser.add_argument("--tsne_out_root", type=str, default="./save/tsne")
    return parser


def validate_args(parser, args):
    if not args.cross_sdc:
        parser.error("This script expects --cross_sdc for all three controlled modes")
    if args.cross_sdc_temperature <= 0 or args.rd_margin_temperature <= 0:
        parser.error("Temperatures must be positive")
    if args.rd_mode == MODE_CROSSSDC and args.lam_cmr != 0:
        parser.error("Pure crosssdc mode requires --lam_cmr 0")
    if args.rd_mode != MODE_CROSSSDC and args.lam_cmr <= 0:
        parser.error("CMR modes require --lam_cmr > 0")
    if not 0.0 <= args.rd_class_weight_alpha <= 1.0:
        parser.error("--rd_class_weight_alpha must lie in [0, 1]")
    if not 0.0 <= args.rd_need_ema_momentum < 1.0:
        parser.error("--rd_need_ema_momentum must lie in [0, 1)")
    if args.rd_weight_min <= 0 or args.rd_weight_max < args.rd_weight_min:
        parser.error("Invalid RD weight bounds")
    if (not np.isfinite(args.rd_margin_tolerance) or args.rd_margin_tolerance < 0):
        parser.error("--rd_margin_tolerance must be finite and non-negative")
    if (not np.isfinite(args.rd_cmr_scale) or args.rd_cmr_scale <= 0):
        parser.error("--rd_cmr_scale must be finite and positive")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step
    setup_seed(args.seed)
    print("Training start time: {}".format(datetime.now()))

    train_set = IcaAVELoader(args=args, mode="train", modality=args.modality)
    val_set = IcaAVELoader(args=args, mode="val", modality=args.modality)
    test_set = IcaAVELoader(args=args, mode="test", modality=args.modality)
    exemplar_set = exemplarLoader(args=args, modality=args.modality)

    id_to_category = {value: key for key, value in train_set.category_encode_dict.items()}

    os.makedirs("./save/{}/".format(run_name(args)), exist_ok=True)
    os.makedirs(figure_dir(args), exist_ok=True)
    os.makedirs(metrics_dir(args), exist_ok=True)

    per_class_csv = os.path.join(metrics_dir(args), "per_class_metrics.csv")
    if os.path.exists(per_class_csv):
        os.remove(per_class_csv)
    epoch_csv = os.path.join(metrics_dir(args), "rd_crosssdc", "epoch_summary.csv")
    if os.path.exists(epoch_csv):
        os.remove(epoch_csv)

    metrics_state = {"best_f1": {}, "first_seen_step": {}}
    save_json(metrics_state, os.path.join(metrics_dir(args), "per_class_state.json"))

    task_best_acc_list = []
    step_forgetting_list = []

    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)
        exemplar_set._set_incremental_step_(step)
        print("Incremental step: {}".format(step))

        if not args.test_only:
            train(
                args=args,
                step=step,
                train_data_set=train_set,
                val_data_set=val_set,
                exemplar_set=exemplar_set,
                id_to_category=id_to_category,
            )

        step_forgetting = detailed_test(
            args=args,
            step=step,
            test_data_set=test_set,
            task_best_acc_list=task_best_acc_list,
            metrics_root=metrics_dir(args),
            metrics_state=metrics_state,
            id_to_category=id_to_category,
            checkpoint_path=checkpoint_path(args, step),
            device=device,
        )
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)

        if args.dump_tsne:
            out_root = os.path.join(args.tsne_out_root, run_name(args))
            make_tsne_plots_for_step(
                args=args,
                step=step,
                test_set=test_set,
                ckpt_path=checkpoint_path(args, step),
                out_root=out_root,
                feature_type=args.tsne_feature,
                max_points_per_class=args.tsne_max_points_per_class,
            )

    mean_forgetting = (
        np.mean(step_forgetting_list) if len(step_forgetting_list) > 0 else 0.0
    )
    print("Average Forgetting: {:.6f}".format(mean_forgetting))

    if args.dataset != "AVE":
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()
        exemplar_set.close_visual_features_h5()


if __name__ == "__main__":
    main()
