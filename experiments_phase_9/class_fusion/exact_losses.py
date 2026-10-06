"""Original AVCIL losses, extracted unchanged from phase 8 (no CrossSDC/CMR)."""


import torch
from torch.nn import functional as F

# 标记说明：[P8 原样沿用] 函数计算代码一致；[P8 逻辑沿用] 公式保留、实现有重组；
# [P9 新增] 新接口或新行为。P8 指 experiments_phase_8_rdcrosssdc_modular。

# [P8 原样沿用] 来源：P8/rd_crosssdc/exact_losses.py::CE_loss。
def CE_loss(num_classes: int, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Original one-hot cross-entropy implementation."""
    targets = F.one_hot(label, num_classes=num_classes) # B*C
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))   # (B,C) --logsoftmax in C--> (B,C) --sum--> (B,)
    return loss


# [P8 原样沿用] 来源：P8/rd_crosssdc/exact_losses.py::cal_contrastive_loss。
# 保留 audio -> visual 的单向实例对比计算。
def cal_contrastive_loss(   # instance level contrastive loss
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Original instance contrastive loss."""
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature    # (B, B)
    num_sample = score.shape[0]
    label = torch.arange(num_sample).to(score.device)
    return CE_loss(num_sample, score, label)


# [P8 原样沿用] 来源：P8/rd_crosssdc/exact_losses.py::class_contrastive_loss。
# 保留原实现按整个 batch 求均值的方式，未改为按同类正样本数归一化。
def class_contrastive_loss( # class level contrastive loss
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    label: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Original AVCIL current-current class contrastive loss."""
    class_matrix = label.unsqueeze(0)   # (B,) --> (1, B)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = class_matrix == label.unsqueeze(-1)
    class_matrix = class_matrix.float()

    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss = -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)) # same class, rather than same instance
    # actually here is 1/B, rather than 1/|Pi|
    # that's why author have the notation below:
    ###################################################################################################
    # As the author mentioned,
    # You can also use the following implementation, which is more consistent with Equation (7) in our paper (and also the standard InfoNCE),
    # but you may need to further adjust the hyperparameters lam_I and lam_C to get optimal performance.
    # loss = -torch.mean(
    #     (torch.sum(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)) / torch.sum(class_matrix, dim=-1))
    ###################################################################################################
    return loss


ce_loss = CE_loss


# [P8 逻辑沿用] 来源：P8/train_incremental_rd_crosssdc_modular.py::train 中的
# AVCIL loss 计算块；这里抽成独立函数，移除了 CrossSDC/CMR，不能视为 P8 完整总 loss。
# [P9 新增] student/teacher 元组接口、输入检查和各项未加权 loss 的 parts 返回值。
def avcil_loss(args, step, student, labels, new_batch_size, teacher=None):
    """Original phase-8 AVCIL objective with all CrossSDC/CMR terms removed.

    The CE split, task-wise KD, contrastive reductions and attention axes are
    intentionally retained. Return unweighted components for diagnostics.
    """
    logits, audio, visual, spatial, temporal = student
    zero = logits.new_zeros(())
    parts = {key: zero for key in ("ce", "kd", "instance", "class", "spatial", "temporal")}
    # [P8 逻辑沿用] 首个任务只计算 CE；下面的 KD、对比和注意力项仅用于增量任务。
    if step == 0:
        parts["ce"] = ce_loss(args.class_num_per_step, logits, labels)
        return parts["ce"], parts
    if teacher is None:
        raise ValueError("Incremental AVCIL requires the frozen previous model")
    old_classes = step * args.class_num_per_step
    replay_size = labels.numel() - new_batch_size
    if replay_size < 1:
        raise ValueError("Incremental AVCIL requires nonempty replay")
    old_logits, _, _, old_spatial, old_temporal = teacher
    # [P8 逻辑沿用] 新样本只在新类输出上算 CE，回放只在旧类输出上算 CE，
    # 再按两组样本数加权；保留 AVE 在第二个任务的联合 CE 特例。
    new_loss = ce_loss(args.class_num_per_step, logits[:new_batch_size, old_classes:],
                       labels[:new_batch_size] % args.class_num_per_step)
    replay_loss = ce_loss(old_classes, logits[new_batch_size:, :old_classes], labels[new_batch_size:])
    parts["ce"] = (new_loss * new_batch_size + replay_loss * replay_size) / labels.numel()
    if args.dataset == "AVE" and args.class_num_per_step == 4 and step == 1:
        parts["ce"] = ce_loss(old_classes + args.class_num_per_step, logits, labels)
    # [P8 逻辑沿用] 对每个旧任务的输出切片分别做 logit KD，再求和；
    # 温度 T=2、补偿 T^2=4，输入包含新样本和回放样本。
    kd = torch.zeros(step, device=logits.device)
    for task in range(step):
        lo, hi = task * args.class_num_per_step, (task + 1) * args.class_num_per_step
        kd[task] = F.kl_div(F.log_softmax(logits[:, lo:hi] / 2.0, dim=1),
                            F.softmax(old_logits[:, lo:hi] / 2.0, dim=1), reduction="batchmean") * 4.0
    parts["kd"] = kd.sum()
    loss = parts["ce"] + parts["kd"]
    # [P8 逻辑沿用] 原始 L_i、L_c 及其系数 lam_I、lam_C；调用上方原样函数。
    if args.instance_contrastive:
        parts["instance"] = cal_contrastive_loss(audio, visual, args.instance_contrastive_temperature)
        loss = loss + args.lam_I * parts["instance"]
    if args.class_contrastive:
        parts["class"] = class_contrastive_loss(audio, visual, labels, args.class_contrastive_temperature)
        loss = loss + args.lam_C * parts["class"]
    # [P8 逻辑沿用] 仅在回放样本上蒸馏注意力；空间/时间轴变换和按回放数
    # 归一化的 KL 保持原公式，混合系数仍为 lam 和 1-lam。
    if args.attn_score_distil:
        s = spatial[new_batch_size:].transpose(2, 3)
        s_old = old_spatial[new_batch_size:].transpose(2, 3)
        t = temporal[new_batch_size:].transpose(1, 2)
        t_old = old_temporal[new_batch_size:].transpose(1, 2)
        parts["spatial"] = F.kl_div(s.reshape(-1, s.shape[-1]).log(),
                                   s_old.reshape(-1, s_old.shape[-1]), reduction="sum") / replay_size
        parts["temporal"] = F.kl_div(t.reshape(-1, t.shape[-1]).log(),
                                    t_old.reshape(-1, t_old.shape[-1]), reduction="sum") / replay_size
        loss = loss + (args.lam * parts["spatial"] + (1.0 - args.lam) * parts["temporal"])
    return loss, parts
