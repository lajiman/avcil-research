"""Same-modality LOO reliability and periodic class-gate updates.

This takes the role of phase 8's rd_method.py. Statistics never use test/val
data, teacher logits, the fused classifier, or gradients. A paused current
student in eval mode is the snapshot; no extra trainable teacher is needed.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# P8 来源：experiments_phase_8_rdcrosssdc_modular/rd_crosssdc/rd_method.py。
# 本模块整体为 P9 方法；只复用下方明确标出的统计构件，不沿用 P8 的 Trust/Need 权重或 CMR。

# [P8 逻辑沿用] capture_rng_state + restore_rng_state 的随机状态保存/恢复；
# 这里合并成 contextmanager，通过 finally 保证统计结束后恢复。
@contextmanager
def preserve_rng_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


# [P9 新增] 旧类回放 + 当前新类训练集的唯一视频参考集；P8 原型库只使用旧类回放。
class FusionReferenceDataset(Dataset):
    """One occurrence of each video from current train data and old replay.

    Construct once per step, outside the periodic update loop. In particular,
    repeated replay minibatches never increase the effective sample count.
    """
    def __init__(self, train_set, exemplar_set=None):
        if train_set.mode != "train":
            raise ValueError("Gate statistics require training data")
        self.sources = [train_set] + ([exemplar_set] if exemplar_set is not None else [])
        self.indices = []
        self.sample_ids = []
        seen = set()
        for source_id, source in enumerate(self.sources):
            if source.mode != "train":
                raise ValueError("Replay reference must come from training data")
            vids = getattr(source, "all_current_data_vids", None)
            if vids is None:
                vids = source.exemplar_vids_set
            for index, vid in enumerate(vids):
                if vid is not None and vid not in seen:
                    seen.add(vid)
                    self.indices.append((source_id, index))
                    self.sample_ids.append(str(vid))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source, item = self.indices[index]
        return self.sources[source][item]


# [P9 新增] 保存同模态可靠性、有效性和计数；不是 P8 的 TeacherPrototypeBank。
@dataclass
class ReliabilityBank:
    counts: torch.Tensor
    scored_counts: torch.Tensor
    reliability_a: torch.Tensor
    reliability_v: torch.Tensor
    valid: torch.Tensor


# [P9 新增] step 内的周期更新日程；默认首任务不更新，最终 epoch 后也不更新。
def should_update_gate(args, step, completed_epochs):
    return (
        args.fusion_mode == "periodic"
        and (step > 0 or args.fusion_update_first_step)
        and args.fusion_warmup_epochs <= completed_epochs < args.max_epoches
        and (completed_epochs - args.fusion_warmup_epochs) % args.fusion_update_interval == 0
    )


# [P9 新增] 当前学生、所有已见类、同模态 LOO 的可靠性估计；
# 与 P8 的冻结旧教师、旧类、跨模态可靠性不同。下面标注共享的数学构件。
@torch.no_grad()
def build_reliability_bank(model, reference_set, num_classes, batch_size,
                           num_workers, device, temperature=0.1, min_samples=2):
    """Two streaming passes: current prototypes, then LOO target probabilities.

    Holding all N x D features on the GPU is unnecessary. A class with an
    invalid prototype is excluded from the candidate set. Paired invalid/zero
    features are excluded from BOTH branches and from sample-count damping.
    """
    if temperature <= 0 or min_samples < 2:
        raise ValueError("Use a positive temperature and at least two samples for LOO")
    net = unwrap_model(model)
    modes = [(module, module.training) for module in model.modules()]
    with preserve_rng_state():
        model.eval()
        try:
            loader = DataLoader(reference_set, batch_size=batch_size, num_workers=num_workers,
                                shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
            sums_a = torch.zeros(num_classes, 768, device=device)
            sums_v = torch.zeros_like(sums_a)
            counts = torch.zeros(num_classes, device=device, dtype=torch.long)

            def batches():
                for data, labels in loader:
                    labels = labels.to(device=device, dtype=torch.long)
                    if ((labels < 0) | (labels >= num_classes)).any():
                        raise ValueError("Reference label lies outside the seen-class range")
                    a, v, _, _ = net.extract_branch_features(data[0].to(device), data[1].to(device))
                    valid = (torch.isfinite(a).all(1) & torch.isfinite(v).all(1)
                             & (a.norm(dim=1) > 1e-12) & (v.norm(dim=1) > 1e-12))
                    yield F.normalize(a[valid].float(), dim=1), F.normalize(v[valid].float(), dim=1), labels[valid]

            # [P8 逻辑沿用] build_old_teacher_prototype_bank 中按类 index_add_ 求和、
            # 计数及归一化原型的公式；P9 改为两次流式扫描并成对过滤无效特征。
            for a, v, labels in batches():
                sums_a.index_add_(0, labels, a)
                sums_v.index_add_(0, labels, v)
                counts.index_add_(0, labels, torch.ones_like(labels))
            candidate = (counts > 0) & (sums_a.norm(dim=1) > 1e-12) & (sums_v.norm(dim=1) > 1e-12)
            prototypes_a, prototypes_v = F.normalize(sums_a, dim=1), F.normalize(sums_v, dim=1)
            probability_a = torch.zeros(num_classes, device=device)
            probability_v = torch.zeros_like(probability_a)
            scored_counts = torch.zeros_like(counts)
            if candidate.sum() >= 2:
                for a, v, labels in batches():
                    # [P8 逻辑沿用] _cross_modal_margin 的 LOO 构造及替换目标类 score：
                    # 从目标类特征和减去当前样本，再归一化。这里 query/prototype 属于同模态，
                    # 且不足 min_samples 时跳过，未保留 P8 单样本退回全类原型的行为。
                    loo_a, loo_v = sums_a[labels] - a, sums_v[labels] - v
                    valid = (candidate[labels] & (counts[labels] >= min_samples)
                             & (loo_a.norm(dim=1) > 1e-12) & (loo_v.norm(dim=1) > 1e-12))
                    a, v, labels = a[valid], v[valid], labels[valid]
                    if not labels.numel():
                        continue
                    score_a = (a @ prototypes_a.T) / temperature
                    score_v = (v @ prototypes_v.T) / temperature
                    target_a = (a * F.normalize(loo_a[valid], dim=1)).sum(1) / temperature
                    target_v = (v * F.normalize(loo_v[valid], dim=1)).sum(1) / temperature
                    score_a.scatter_(1, labels[:, None], target_a[:, None])
                    score_v.scatter_(1, labels[:, None], target_v[:, None])
                    score_a.masked_fill_(~candidate[None, :], -torch.inf)
                    score_v.masked_fill_(~candidate[None, :], -torch.inf)
                    # [P8 逻辑沿用] 概率公式等价于 build_old_teacher_prototype_bank 的
                    # sigmoid(target - logsumexp(rest))；输入/候选集不同，不表示可靠性值相同。
                    p_a = torch.exp(target_a - torch.logsumexp(score_a, dim=1))
                    p_v = torch.exp(target_v - torch.logsumexp(score_v, dim=1))
                    probability_a.index_add_(0, labels, p_a)
                    probability_v.index_add_(0, labels, p_v)
                    scored_counts.index_add_(0, labels, torch.ones_like(labels))
            r_a = probability_a / scored_counts.clamp_min(1)
            r_v = probability_v / scored_counts.clamp_min(1)
            valid = ((scored_counts >= min_samples) & (scored_counts == counts)
                     & torch.isfinite(r_a) & torch.isfinite(r_v) & ((r_a + r_v) > 1e-12))
            return ReliabilityBank(*(x.detach().cpu() for x in (counts, scored_counts, r_a, r_v, valid)))
        finally:
            for module, training in modes:
                module.training = training


# [P9 新增] 用 R_a/(R_a+R_v) 生成类别 gate，并向该类历史值平滑；
# 样本量控制更新幅度 eta，不是 P8 normalize_class_weights 的跨类别归一化/收缩。
def update_class_gates(previous, bank, update_rule="sample_aware", eta_max=0.5, n_ref=10.0):
    """Smooth toward each class's new estimate, never toward a global mean."""
    if update_rule not in ("fixed", "sample_aware"):
        raise ValueError("Unknown gate update rule")
    if not 0 < eta_max <= 1 or n_ref <= 0:
        raise ValueError("eta_max must be in (0, 1], n_ref must be positive")
    previous = previous.detach().cpu().float()
    if previous.shape != bank.counts.shape:
        raise ValueError("Gate and reliability class orders differ")
    raw = bank.reliability_a / (bank.reliability_a + bank.reliability_v).clamp_min(1e-12)
    raw = torch.where(bank.valid, raw, previous)
    eta = torch.full_like(previous, eta_max)
    if update_rule == "sample_aware":
        eta *= bank.counts.float() / (bank.counts.float() + n_ref)
    eta = torch.where(bank.valid, eta, torch.zeros_like(eta))
    updated = previous + eta * (raw - previous)
    return {
        "previous_gate": previous, "raw_gate": raw, "gate": updated,
        "eta": eta, "counts": bank.counts, "scored_counts": bank.scored_counts,
        "reliability_a": bank.reliability_a, "reliability_v": bank.reliability_v,
        "valid": bank.valid,
    }
