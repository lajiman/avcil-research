"""Original AVCIL + periodic class-conditional modality fusion (phase 9).

Layout follows phase 8: a training entry point and a method package containing
exact_losses, fusion_method, diagnostics, metrics, and checkpoint handling.
"""

import argparse
from datetime import datetime
from itertools import cycle
import math
import os
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments_phase_9.dataloader_ours import IcaAVELoader, exemplarLoader
from experiments_phase_9.class_fusion.checkpoints import load_model, save_checkpoint
from experiments_phase_9.class_fusion.cl_history import CLHistoryRecorder, should_record_cl_history
from experiments_phase_9.class_fusion.diagnostics import append_csv_row, save_gate_snapshot
from experiments_phase_9.class_fusion.exact_losses import avcil_loss
from experiments_phase_9.class_fusion.fusion_method import (
    FusionReferenceDataset, build_reliability_bank, should_update_gate, unwrap_model, update_class_gates,
)
from experiments_phase_9.class_fusion.metrics import detailed_test, save_json
from model.audio_visual_model_incremental_class_fusion import ClassFusionAudioVisualNet

# 本文件的 P8 来源：experiments_phase_8_rdcrosssdc_modular/train_incremental_rd_crosssdc_modular.py。
# “原样沿用”仅指所标函数/代码块；“逻辑沿用”会注明改写点，不代表整个训练器相同。

# [P8 原样沿用] setup_seed 的可执行函数体保持一致。
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


# [P8 逻辑沿用] 同名布尔解析器；仍只接受 True/False，异常改为 argparse 参数错误。
def boolean_string(value):
    if value not in ("True", "False"):
        raise argparse.ArgumentTypeError("Expected True or False")
    return value == "True"


# [P8 逻辑沿用] 同名函数仍优先使用 experiment_name；默认名新增融合配置和 seed。
def run_name(args):
    return args.experiment_name or f"{args.dataset}_{args.fusion_mode}_{args.fusion_update_rule}_seed{args.seed}"


# [P8 逻辑沿用] 按实验/step 定位 checkpoint；改为可配置根目录、best/last 和 .pt 格式。
def checkpoint_path(args, step, which="best"):
    return Path(args.output_root) / run_name(args) / f"step_{step}_{which}_model.pt"


# [P8 逻辑沿用] 同名函数的 metrics/实验名布局；根目录改由 output_root 指定。
def metrics_dir(args):
    return Path(args.output_root) / "metrics" / run_name(args)


# [P8 逻辑沿用] 抽出 train 中重复的 DataLoader 设置，保留训练时 shuffle/drop_last。
# [P9 新增] 空数据检查及仅 CUDA 开启 pin_memory。
def make_loader(dataset, batch_size, workers, device, train=False):
    if not len(dataset):
        raise ValueError("Dataset is empty: check metadata, paired features and replay size")
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), num_workers=workers,
                      shuffle=train, drop_last=train, pin_memory=device.type == "cuda")


# [P8 逻辑沿用] 抽出 train 中的验证 top-1 评估；直接累计 logits.argmax 的正确数，
# 替代原先拼接 softmax 输出再调用 top_1_acc 的实现。
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct, count = 0, 0
    for data, labels in loader:
        logits = model(visual=data[0].to(device), audio=data[1].to(device))
        labels = labels.to(device)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return correct / count


# [P8 逻辑沿用] 保留 train 的增量框架：上一任务 best -> 扩类 -> 新数据+回放
# -> 原始 AVCIL loss -> 验证选 best；融合更新及 CL 观测见下方 P9 标记。
def train(args, step, train_data_set, val_data_set, exemplar_set, id_to_category, device):
    """Train one step; update gates only after validation/checkpoint selection."""
    train_loader = make_loader(train_data_set, args.train_batch_size, args.num_workers, device, train=True)
    val_loader = make_loader(val_data_set, args.infer_batch_size, args.num_workers, device)
    # [P8 逻辑沿用] 首任务初始化，后续任务继承 best 并保留冻结教师。
    # [P9 新增] 使用 ClassFusion 模型及包含 gate 的 state_dict checkpoint。
    if step == 0:
        model = ClassFusionAudioVisualNet(args, args.class_num_per_step)
        old_model, exemplar_loader = None, None
        teacher_metadata = None
    else:
        model = load_model(checkpoint_path(args, step - 1))
        model.incremental_classifier((step + 1) * args.class_num_per_step)
        old_model, teacher_metadata = load_model(checkpoint_path(args, step - 1), return_metadata=True)
        old_model = old_model.to(device)
        old_model.requires_grad_(False)
        old_model.eval()
        exemplar_loader = make_loader(exemplar_set, args.exemplar_batch_size, args.num_workers, device, train=True)
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        if old_model is not None:
            old_model = torch.nn.DataParallel(old_model)
            old_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    net = unwrap_model(model)
    # [P9 新增] 每个 step 固定统计参考集，保存初始 gate；不对应 P8 的 RD loss 状态。
    reference_set = None
    if args.fusion_mode == "periodic" and (step > 0 or args.fusion_update_first_step):
        reference_set = FusionReferenceDataset(train_data_set, exemplar_set if step > 0 else None)
    fusion_root = metrics_dir(args) / "class_fusion"
    save_gate_snapshot(fusion_root / f"step_{step}_initial_gates.csv", step, 0,
                       net.fusion_gate, int(net.fusion_version), id_to_category)
    # [P9 新增] CL 历史仅记录旧类双模态遗忘，不参与 loss 或 gate 更新。
    # Observation-only history uses old memory and an old-only label space.
    # It is deliberately separate from the all-seen reliability driving g_c.
    cl_recorder, cl_state = None, None
    if args.record_cl_history and step > 0:
        cl_recorder = CLHistoryRecorder(args, step, old_model, exemplar_set, teacher_metadata,
                                        metrics_dir(args) / "cl_history", id_to_category, device)
        cl_state = cl_recorder.observe(model, epoch=0, events=["task_start"])
    best_acc, last_snapshot = -float("inf"), None
    # [P8 逻辑沿用] epoch/batch 训练循环；此处 epoch 从 1 开始，P8 从 0 开始。
    for epoch in range(1, args.max_epoches + 1):
        model.train()
        if step == 0:
            batches = ((batch, None) for batch in train_loader)
        else:
            # [P8 逻辑沿用] 当前数据迭代一次，回放通过 cycle 循环配对。
            batches = zip(train_loader, cycle(exemplar_loader))
        sums = {key: 0.0 for key in ("train_loss", "ce", "kd", "instance", "class", "spatial", "temporal")}
        count = 0
        for current, replay in tqdm(batches, total=len(train_loader), desc=f"Step {step} epoch {epoch}"):
            # [P9 新增] 前向前释放上一批梯度，降低三进程共享 GPU 时的活跃张量峰值。
            optimizer.zero_grad(set_to_none=True)
            data, labels = current
            new_batch_size = labels.numel()
            visual, audio = data
            # [P8 逻辑沿用] batch 顺序固定为“新样本在前，回放在后”，供各 loss 切片。
            if replay is not None:
                replay_data, replay_labels = replay
                visual = torch.cat((visual, replay_data[0]))
                audio = torch.cat((audio, replay_data[1]))
                labels = torch.cat((labels, replay_labels))
            visual, audio, labels = visual.to(device), audio.to(device), labels.to(device)
            student = model(visual=visual, audio=audio, out_feature_before_fusion=True, out_attn_score=True)
            teacher = None
            if old_model is not None:
                with torch.no_grad():
                    teacher = old_model(visual=visual, audio=audio, out_feature_before_fusion=True, out_attn_score=True)
            # [P8 逻辑沿用] 原始 AVCIL loss 已抽入 exact_losses.avcil_loss；没有 RD/CrossSDC 项。
            loss, components = avcil_loss(args, step, student, labels, new_batch_size, teacher)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            sums["train_loss"] += loss.item()
            for key, value in components.items():
                sums[key] += value.item()
            count += 1
            # 不把上一批学生/教师输出及输入保留到下一批前向，loss/梯度计算顺序不变。
            del loss, components, student, teacher, visual, audio, labels, value
        # 释放 cycle 缓存的回放 batch 和最后一批引用，再进行验证/原型统计。
        # 保留 P8 cycle 的批次顺序，未重新采样回放，也未修改 batch size。
        del batches, current, replay, data
        if step > 0:
            del replay_data, replay_labels
        optimizer.zero_grad(set_to_none=True)
        averages = {key: value / count for key, value in sums.items()}
        val_acc = validate(model, val_loader, device)
        # [P9 新增] 各 loss 分项与 gate 的统一日志；CL 观测也在本轮 gate 更新之前记录。
        row = {"step": step, "epoch": epoch, "fusion_mode": args.fusion_mode,
               "fusion_update_rule": args.fusion_update_rule, "gate_version": int(net.fusion_version),
               "gate_min": float(net.fusion_gate.min()), "gate_max": float(net.fusion_gate.max()),
               "gate_mean": float(net.fusion_gate.mean()), **averages, "val_acc": val_acc}
        append_csv_row(fusion_root / "epoch_summary.csv", list(row), row)
        print(f"Step {step} epoch {epoch}: loss={averages['train_loss']:.6f}, val_acc={val_acc:.6f}, gate_version={int(net.fusion_version)}", flush=True)
        is_best = val_acc > best_acc
        if cl_recorder is not None and should_record_cl_history(args, step, epoch, is_best):
            events = []
            if is_best:
                events.append("best_checkpoint")
            if epoch == args.max_epoches:
                events.append("final_epoch")
            if epoch % args.cl_history_interval == 0:
                events.append("periodic")
            if should_update_gate(args, step, epoch):
                events.append("before_gate_update")
            cl_state = cl_recorder.observe(model, epoch, events)
            observed = cl_state["observation"]
            print(f"CL history epoch {epoch}: {int(observed['comparable'].sum())}/{step * args.class_num_per_step} comparable old classes", flush=True)
        # [P8 逻辑沿用] 验证准确率严格提高才保存 best；初始 best 改为 -inf，确保首轮可保存。
        if is_best:
            best_acc = val_acc
            save_checkpoint(checkpoint_path(args, step), model, args, step, epoch, val_acc, last_snapshot, cl_state)
        # [P9 新增] 另外保存 last，且 checkpoint 包含 gate/统计/观测状态。
        if epoch == args.max_epoches:
            save_checkpoint(checkpoint_path(args, step, "last"), model, args, step, epoch, val_acc, last_snapshot, cl_state)
        # [P8 逻辑沿用] 只在增量任务的 milestone 将 lr 乘 0.1；已适配从 1 开始的 epoch。
        if args.lr_decay and step > 0 and epoch in args.milestones:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        # [P9 新增] 重估可靠性、按样本量控制更新幅度，更新的 gate 从下一轮生效。
        # New gates take effect in the NEXT epoch, never retroactively in this
        # epoch's validation. No post-final-epoch update is permitted.
        if should_update_gate(args, step, epoch):
            bank = build_reliability_bank(model, reference_set, net.num_classes, args.fusion_batch_size,
                                          args.num_workers, device, args.fusion_temperature, args.fusion_min_samples)
            last_snapshot = update_class_gates(net.fusion_gate, bank, args.fusion_update_rule,
                                               args.fusion_eta_max, args.fusion_n_ref)
            if (last_snapshot["eta"] > 0).any():
                net.set_fusion_gate(last_snapshot["gate"])
            save_gate_snapshot(fusion_root / f"step_{step}_after_epoch_{epoch}_gates.csv", step, epoch,
                               net.fusion_gate, int(net.fusion_version), id_to_category, last_snapshot)
            print(f"Fusion update after epoch {epoch}: {int(bank.valid.sum())}/{net.num_classes} valid classes; effective from epoch {epoch + 1}", flush=True)

    # [P9 新增] 记录该进程一个 step 的 CUDA 分配器峰值，便于实测三个 seed 能否同卡。
    # 不包含其他进程或分配器之外的 CUDA context/驱动内存，不能代替整卡监控。
    if device.type == "cuda":
        memory = {"step": step,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                  "device_total_gib": torch.cuda.get_device_properties(device).total_memory / 2**30}
        append_csv_row(metrics_dir(args) / "resource_summary.csv", list(memory), memory)
        print(f"Step {step} CUDA memory: allocated peak={memory['peak_allocated_gib']:.3f} GiB, reserved peak={memory['peak_reserved_gib']:.3f} GiB", flush=True)


# [P8 逻辑沿用] 保留数据、优化器、原始 AVCIL loss 参数名称，默认值并非全部相同：
# 本实验默认打开三类可选 AVCIL 项，增加 --no_* 开关，并移除 RD/CrossSDC 参数。
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="VGGSound_random_balance_crosssdc_z1_seed42")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--modality", choices=["audio-visual"], default="audio-visual")
    parser.add_argument("--feature_root", default="../../../datasets/VGGSound")
    parser.add_argument("--meta_root", default=str(REPO_ROOT / "data2" / "balance"))
    parser.add_argument("--output_root", default=str(Path(__file__).resolve().parent / "save"))
    for name, default in (("train_batch_size", 128), ("infer_batch_size", 32), ("exemplar_batch_size", 128),
                          ("max_epoches", 200), ("num_classes", 100), ("class_num_per_step", 10),
                          ("memory_size", 500), ("num_workers", 0), ("seed", 42)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, nargs="+", default=[100])
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--lam_I", type=float, default=0.1)
    parser.add_argument("--lam_C", type=float, default=1.0)
    for name in ("instance_contrastive", "class_contrastive", "attn_score_distil"):
        parser.add_argument("--" + name, action="store_true", default=True)
        parser.add_argument("--no_" + name, dest=name, action="store_false")
    parser.add_argument("--instance_contrastive_temperature", type=float, default=0.05)
    parser.add_argument("--class_contrastive_temperature", type=float, default=0.05)
    # [P9 新增] 融合、周期更新和 CL 历史记录相关配置。
    parser.add_argument("--fusion_mode", choices=["uniform", "periodic"], default="periodic")
    parser.add_argument("--fusion_update_rule", choices=["fixed", "sample_aware"], default="sample_aware")
    parser.add_argument("--fusion_warmup_epochs", type=int, default=40)
    parser.add_argument("--fusion_update_interval", type=int, default=40)
    parser.add_argument("--fusion_eta_max", type=float, default=0.5)
    parser.add_argument("--fusion_n_ref", type=float, default=10.0)
    parser.add_argument("--fusion_temperature", type=float, default=0.1)
    parser.add_argument("--fusion_min_samples", type=int, default=2)
    parser.add_argument("--fusion_batch_size", type=int, default=128)
    parser.add_argument("--fusion_update_first_step", action="store_true")
    parser.add_argument("--fusion_classifier", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--fusion_hidden_dim", type=int, default=768)
    parser.add_argument("--fusion_chunk_size", type=int, default=16)
    parser.add_argument("--record_cl_history", action="store_true", default=True,
                        help="Record old-memory teacher/student retention diagnostics (default on)")
    parser.add_argument("--no_record_cl_history", dest="record_cl_history", action="store_false")
    parser.add_argument("--cl_history_interval", type=int, default=40,
                        help="Additional observation interval; task start, gate updates, best and final states are always recorded")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--test_only", action="store_true")
    return parser


# [P9 新增] 针对 phase 9 的参数约束，不沿用 P8 的 RD 参数校验。
def validate_args(parser, args):
    positive_ints = ("train_batch_size", "infer_batch_size", "exemplar_batch_size", "max_epoches", "num_classes",
                     "class_num_per_step", "fusion_update_interval", "fusion_warmup_epochs", "fusion_batch_size",
                     "fusion_hidden_dim", "fusion_chunk_size", "cl_history_interval")
    for key in positive_ints:
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.num_classes % args.class_num_per_step:
        parser.error("--num_classes must be divisible by --class_num_per_step")
    if args.num_workers < 0 or args.memory_size < 0 or args.fusion_min_samples < 2:
        parser.error("workers/memory must be nonnegative; fusion_min_samples must be at least 2")
    if args.num_classes > args.class_num_per_step and args.memory_size < args.num_classes - args.class_num_per_step:
        parser.error("The original AVCIL replay loss requires memory for at least one example per old class")
    for key in ("lr", "fusion_n_ref", "fusion_temperature", "instance_contrastive_temperature", "class_contrastive_temperature"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key} must be finite and positive")
    for key in ("weight_decay", "lam_I", "lam_C"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"--{key} must be finite and nonnegative")
    if not 0 <= args.lam <= 1 or not 0 < args.fusion_eta_max <= 1:
        parser.error("lam must be in [0,1]; fusion_eta_max must be in (0,1]")
    if not (args.dataset in ("AVE", "ksounds") or "VGGSound" in args.dataset):
        parser.error("dataset must be AVE, ksounds, or contain VGGSound")
    name = run_name(args)
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        parser.error("experiment_name must be a single directory name")


# [P8 逻辑沿用] main 中按 step 更新数据集/回放、训练、测试的调度框架。
# [P9 新增] 输出目录保护、配置落盘、test_only 独立输出及 finally 关闭数据句柄。
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    setup_seed(args.seed)
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        parser.error("CUDA requested but unavailable")
    print(args, flush=True)
    run_root = Path(args.output_root) / run_name(args)
    if not args.test_only and run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Use a new experiment_name; existing run: {run_root}")
    output_metrics = metrics_dir(args)
    if args.test_only:
        output_metrics /= "test_only_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        run_root.mkdir(parents=True, exist_ok=True)
        save_json(vars(args), str(run_root / "config.json"))
    datasets = []
    try:
        train_set = val_set = exemplar_set = None
        shared_audio_features = None
        if not args.test_only:
            train_set = IcaAVELoader(args, "train", args.modality)
            datasets.append(train_set)
            shared_audio_features = train_set.all_audio_pretrained_features
            val_set = IcaAVELoader(args, "val", args.modality, shared_audio_features=shared_audio_features)
            datasets.append(val_set)
            exemplar_set = exemplarLoader(args, args.modality, shared_audio_features=shared_audio_features)
            datasets.append(exemplar_set)
        test_set = IcaAVELoader(args, "test", args.modality, shared_audio_features=shared_audio_features)
        datasets.append(test_set)
        id_to_category = {int(value): key for key, value in test_set.category_encode_dict.items()}
        task_best, results = [], []
        metrics_state = {"best_f1": {}, "first_seen_step": {}}
        for step in range(args.num_classes // args.class_num_per_step):
            test_set.set_incremental_step(step)
            if not args.test_only:
                train_set.set_incremental_step(step)
                val_set.set_incremental_step(step)
                exemplar_set._set_incremental_step_(step)
                train(args, step, train_set, val_set, exemplar_set, id_to_category, device)
                # [P9 新增] 任务训练结束后模型/优化器已离开作用域，释放空闲 CUDA 缓存
                # 供同卡其他 seed 使用；不在每个 minibatch 清缓存，避免反复分配。
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            results.append(detailed_test(args, step, test_set, task_best, str(output_metrics), metrics_state,
                                         id_to_category, str(checkpoint_path(args, step)), device))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        forgetting = [item["forgetting"] for item in results if item["forgetting"] is not None]
        summary = {"steps": results, "average_incremental_accuracy": float(np.mean([r["overall_acc"] for r in results])),
                   "average_forgetting": float(np.mean(forgetting)) if forgetting else 0.0}
        save_json(summary, str(output_metrics / "summary.json"))
        print(f"Average incremental accuracy: {summary['average_incremental_accuracy']:.6f}; average forgetting: {summary['average_forgetting']:.6f}")
    finally:
        for dataset in datasets:
            dataset.close_visual_features_h5()


if __name__ == "__main__":
    main()
