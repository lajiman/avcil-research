"""Fixed birth prototypes with memory-anchor drift compensation.

The bank stores training-only summaries at a class's birth checkpoint. Later
updates re-encode retained memory, never discarded old training examples. A
class-average translation approximates representation drift; it does not
claim to align arbitrary rotations or nonlinear changes of feature space.

[P9 bank 新增：整个模块] 复用 fusion_method 的 RNG/mode 辅助接口与
ReliabilityBank 数据结构；固定 birth 统计、漂移补偿和历史 LOO 为新增算法，
不替换旧 fresh 路径，也不对应 P8 的 RD/CrossSDC loss。
"""

import copy
import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .fusion_method import ReliabilityBank, preserve_rng_state, unwrap_model


SCHEMA_VERSION = 1
FEATURE_DIM = 768
EPS = 1e-12


# [P9 bank 新增] 只包装刚结束任务的训练 ID；不会切换/重采样 exemplar 数据集。
class _BirthTrainingDataset(Dataset):
    def __init__(self, source, sample_ids, labels):
        self.source, self.sample_ids, self.labels = source, sample_ids, labels

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        data, label = self.source._read(self.sample_ids[index])
        if int(label) != self.labels[index]:
            raise ValueError("Birth training ID and class label disagree")
        return data, label


def _cpu_copy(value):
    """Keep checkpoint state compatible with torch.load(weights_only=True)."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Prototype provenance keys must be strings")
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Prototype provenance must contain tensors and primitive values")


def _validate_bank(bank):
    if not isinstance(bank, dict) or bank.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported prototype-bank schema")
    step, width, classes = bank["prepared_step"], bank["classes_per_step"], bank["num_classes"]
    if step < 1 or width < 1 or classes != step * width:
        raise ValueError("Prototype-bank step and class order disagree")
    for key in ("birth_sum_audio", "birth_sum_visual"):
        value = bank[key]
        if value.shape != (classes, FEATURE_DIM) or not torch.isfinite(value).all():
            raise ValueError("Invalid prototype birth sums")
    count, birth_step = bank["birth_count"], bank["birth_step"]
    if count.shape != (classes,) or count.dtype != torch.long or (count < 0).any():
        raise ValueError("Invalid prototype birth counts")
    if not torch.equal(birth_step.cpu(), torch.arange(classes) // width):
        raise ValueError("Prototype birth class order differs from incremental order")
    anchors = bank["anchors"]
    ids, labels, valid = anchors["sample_ids"], anchors["labels"], anchors["valid"]
    if len(ids) != len(set(ids)) or not all(isinstance(vid, str) for vid in ids):
        raise ValueError("Prototype anchors must have unique string sample IDs")
    if labels.shape != (len(ids),) or labels.dtype != torch.long or ((labels < 0) | (labels >= classes)).any():
        raise ValueError("Invalid prototype anchor class order")
    if valid.shape != labels.shape or valid.dtype != torch.bool:
        raise ValueError("Invalid prototype anchor validity mask")
    for key in ("audio", "visual"):
        value = anchors[key]
        if value.shape != (len(ids), FEATURE_DIM) or not torch.isfinite(value).all():
            raise ValueError("Invalid prototype anchor features")
        if valid.any() and (value[valid].norm(dim=1) <= EPS).any():
            raise ValueError("Valid prototype anchor has a zero feature")
    supported = torch.bincount(labels[valid].cpu(), minlength=classes)
    if (supported > count.cpu()).any():
        raise ValueError("Prototype anchors exceed the full birth sample count")
    return {vid: index for index, vid in enumerate(ids)}


def _unit_features(net, data, device):
    audio, visual, _, _ = net.extract_branch_features(data[0].to(device), data[1].to(device))
    audio, visual = audio.float(), visual.float()
    if audio.ndim != 2 or audio.shape != visual.shape or audio.shape[1] != FEATURE_DIM:
        raise ValueError("Prototype bank requires paired 768-dimensional branch features")
    valid = (torch.isfinite(audio).all(1) & torch.isfinite(visual).all(1)
             & (audio.norm(dim=1) > EPS) & (visual.norm(dim=1) > EPS))
    # [P9 沿用] 两个模态一起过滤；无效样本不能进入任一模态的原型或计数。
    result_a, result_v = torch.zeros_like(audio), torch.zeros_like(visual)
    result_a[valid] = F.normalize(audio[valid], dim=1)
    result_v[valid] = F.normalize(visual[valid], dim=1)
    return result_a, result_v, valid


@torch.no_grad()
def prepare_prototype_bank(previous_bank, teacher, exemplar_set, step,
                           classes_per_step, batch_size, num_workers, device,
                           teacher_metadata):
    """Prepare immutable birth summaries for the next incremental task.

    Call after the original replay selection and before training ``step``.
    Only classes from task ``step - 1`` are scanned in full, with that task's
    best model. Older summaries stay fixed and their anchors are only pruned.
    """
    if step < 1 or classes_per_step < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid prototype-bank preparation arguments")
    if exemplar_set.mode != "train":
        raise ValueError("Prototype banks require training data")
    classes, previous_classes = step * classes_per_step, (step - 1) * classes_per_step
    if previous_bank is None:
        if step != 1:
            raise ValueError("Missing old prototype bank; cannot silently reconstruct lost history")
        previous_lookup = {}
    else:
        previous_lookup = _validate_bank(previous_bank)
        if (previous_bank["prepared_step"] != step - 1
                or previous_bank["classes_per_step"] != classes_per_step
                or previous_bank["num_classes"] != previous_classes):
            raise ValueError("Prototype bank must advance one task at a time")
    if not isinstance(teacher_metadata, dict) or teacher_metadata.get("step") != step - 1:
        raise ValueError("Birth prototype teacher must come from the preceding task")
    if teacher_metadata.get("num_classes") != classes:
        raise ValueError("Birth teacher and prototype class order disagree")
    net = unwrap_model(teacher)
    if getattr(net, "num_classes", classes) != classes:
        raise ValueError("Birth teacher has the wrong number of classes")
    groups = exemplar_set.exemplar_class_vids_set
    if len(groups) != classes:
        raise ValueError("Replay class groups and prototype class order disagree")
    ids = [str(vid) for group in groups for vid in group]
    if ids != [str(vid) for vid in exemplar_set.exemplar_vids_set] or len(ids) != len(set(ids)):
        raise ValueError("Replay IDs must be unique and match their class groups")
    labels = torch.tensor([c for c, group in enumerate(groups) for _ in group], dtype=torch.long)
    for vid, label in zip(ids, labels.tolist()):
        metadata_label = int(exemplar_set.category_encode_dict[exemplar_set.all_id_category_dict[vid]])
        if metadata_label != label:
            raise ValueError("Replay ID and prototype anchor label disagree")

    bank = {
        "schema_version": SCHEMA_VERSION, "prepared_step": step,
        "classes_per_step": classes_per_step, "num_classes": classes,
        "birth_sum_audio": torch.zeros(classes, FEATURE_DIM),
        "birth_sum_visual": torch.zeros(classes, FEATURE_DIM),
        "birth_count": torch.zeros(classes, dtype=torch.long),
        "birth_step": torch.arange(classes) // classes_per_step,
        "anchors": {"sample_ids": ids, "labels": labels,
                    "valid": torch.zeros(len(ids), dtype=torch.bool),
                    "audio": torch.zeros(len(ids), FEATURE_DIM),
                    "visual": torch.zeros(len(ids), FEATURE_DIM)},
        "sources": copy.deepcopy(previous_bank["sources"]) if previous_bank is not None else [],
    }
    if previous_bank is not None:
        for key in ("birth_sum_audio", "birth_sum_visual", "birth_count"):
            bank[key][:previous_classes].copy_(previous_bank[key].cpu())
    selected = {vid: index for index, vid in enumerate(ids)}
    for index, (vid, label) in enumerate(zip(ids, labels.tolist())):
        if label >= previous_classes:
            continue
        if vid not in previous_lookup:
            raise ValueError("Retained old memory has no birth anchor; memory may only be pruned")
        old_index = previous_lookup[vid]
        old = previous_bank["anchors"]
        if int(old["labels"][old_index]) != label:
            raise ValueError("Old memory anchor changed class label")
        for key in ("valid", "audio", "visual"):
            bank["anchors"][key][index].copy_(old[key][old_index].cpu())

    modes = [(module, module.training) for module in teacher.modules()]
    with preserve_rng_state():
        teacher.eval()
        try:
            birth_ids, birth_labels, seen = [], [], set()
            for label in range(previous_classes, classes):
                for vid in exemplar_set._valid_class_vids(label):
                    vid = str(vid)
                    if vid in seen:
                        raise ValueError("Duplicate ID in birth training classes")
                    seen.add(vid)
                    birth_ids.append(vid)
                    birth_labels.append(label)
            if any(vid not in seen for vid, label in zip(ids, labels.tolist()) if label >= previous_classes):
                raise ValueError("New replay anchor is not in its birth training dataset")
            source = _BirthTrainingDataset(exemplar_set, birth_ids, birth_labels)
            loader = DataLoader(source, batch_size=batch_size, num_workers=num_workers,
                                shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
            offset = 0
            for data, target in loader:
                audio, visual, valid = _unit_features(net, data, device)
                audio, visual, valid = audio.cpu(), visual.cpu(), valid.cpu()
                target = target.long().cpu()
                bank["birth_sum_audio"].index_add_(0, target[valid], audio[valid])
                bank["birth_sum_visual"].index_add_(0, target[valid], visual[valid])
                bank["birth_count"].index_add_(0, target[valid], torch.ones_like(target[valid]))
                for local, vid in enumerate(birth_ids[offset:offset + len(target)]):
                    if vid in selected:
                        index = selected[vid]
                        bank["anchors"]["valid"][index] = valid[local]
                        bank["anchors"]["audio"][index] = audio[local]
                        bank["anchors"]["visual"][index] = visual[local]
                offset += len(target)
        finally:
            for module, training in modes:
                module.training = training
    bank["sources"].append(_cpu_copy(teacher_metadata))
    _validate_bank(bank)
    return bank


@torch.no_grad()
def build_bank_reliability(model, reference_set, prototype_bank, num_classes,
                           batch_size, num_workers, device, temperature=0.1,
                           min_samples=2, prior_strength=10.0):
    """Estimate reliability with drift-compensated prototypes and strict LOO.

    The historical mean is fixed at birth. ``prior_strength`` caps a heuristic
    historical pseudo-count, bounded by birth samples absent from this scan;
    it is not a claim that overlapping means are independent observations.
    Query contributions are excluded from the historical mean AND drift when
    constructing its target prototype. No new loss or learnable state is used.
    """
    if (temperature <= 0 or min_samples < 2 or batch_size < 1 or num_workers < 0
            or not math.isfinite(prior_strength) or prior_strength < 0):
        raise ValueError("Invalid prototype-bank reliability arguments")
    lookup = _validate_bank(prototype_bank)
    old_classes = prototype_bank["num_classes"]
    if num_classes != old_classes + prototype_bank["classes_per_step"]:
        raise ValueError("Prototype bank does not match the current incremental task")
    ids = [str(vid) for vid in reference_set.sample_ids]
    if len(ids) != len(reference_set) or len(ids) != len(set(ids)):
        raise ValueError("Prototype reference requires one unique ID per sample")
    if any(getattr(source, "mode", "train") != "train" for source in getattr(reference_set, "sources", [reference_set])):
        raise ValueError("Prototype reliability requires training-only references")
    net = unwrap_model(model)
    modes = [(module, module.training) for module in model.modules()]
    with preserve_rng_state():
        model.eval()
        try:
            return _build_bank_reliability(net, reference_set, prototype_bank, lookup,
                                           ids, num_classes, batch_size, num_workers,
                                           device, temperature, min_samples, prior_strength)
        finally:
            for module, training in modes:
                module.training = training


def _build_bank_reliability(net, reference_set, bank, lookup, ids, num_classes,
                            batch_size, num_workers, device, temperature,
                            min_samples, prior_strength):
    old_classes = bank["num_classes"]
    sums_a = torch.zeros(num_classes, FEATURE_DIM, device=device)
    sums_v = torch.zeros_like(sums_a)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    anchors_a, anchors_v = torch.zeros_like(sums_a), torch.zeros_like(sums_v)
    anchor_count = torch.zeros_like(counts)
    drift_sq_a, drift_sq_v = torch.zeros(num_classes, device=device), torch.zeros(num_classes, device=device)
    issues = [set() for _ in range(num_classes)]
    cached = []
    loader = DataLoader(reference_set, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
    offset = 0
    for data, labels in loader:
        labels = labels.to(device=device, dtype=torch.long)
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError("Reference label lies outside the seen-class range")
        audio, visual, valid = _unit_features(net, data, device)
        batch_ids = ids[offset:offset + len(labels)]
        offset += len(labels)
        # 标签匹配检查也覆盖当前特征无效的样本，不能借无效特征绕过身份校验。
        for vid, label in zip(batch_ids, labels.tolist()):
            if vid in lookup and int(bank["anchors"]["labels"][lookup[vid]]) != label:
                raise ValueError("Reference sample ID changed its prototype-bank label")
        current_ids = [vid for vid, usable in zip(batch_ids, valid.tolist()) if usable]
        audio, visual, labels = audio[valid], visual[valid], labels[valid]
        birth_a, birth_v = torch.zeros_like(audio), torch.zeros_like(visual)
        matched = torch.zeros(len(labels), dtype=torch.bool, device=device)
        for index, (vid, label) in enumerate(zip(current_ids, labels.tolist())):
            if label >= old_classes:
                continue
            if vid not in lookup:
                issues[label].add("missing_birth_anchor")
                continue
            old_index = lookup[vid]
            if not bool(bank["anchors"]["valid"][old_index]):
                issues[label].add("invalid_birth_anchor")
                continue
            matched[index] = True
            birth_a[index] = bank["anchors"]["audio"][old_index].to(device)
            birth_v[index] = bank["anchors"]["visual"][old_index].to(device)
        sums_a.index_add_(0, labels, audio)
        sums_v.index_add_(0, labels, visual)
        counts.index_add_(0, labels, torch.ones_like(labels))
        anchors_a.index_add_(0, labels[matched], birth_a[matched])
        anchors_v.index_add_(0, labels[matched], birth_v[matched])
        anchor_count.index_add_(0, labels[matched], torch.ones_like(labels[matched]))
        drift_sq_a.index_add_(0, labels[matched], (audio[matched] - birth_a[matched]).square().sum(1))
        drift_sq_v.index_add_(0, labels[matched], (visual[matched] - birth_v[matched]).square().sum(1))
        # [P9 bank 新增] 整体特征只缓存在 CPU；GPU 始终仅持有当前 batch 和类统计。
        cached.append(tuple(value.cpu() for value in (audio, visual, labels, birth_a, birth_v)))

    birth_count = torch.zeros_like(counts)
    birth_count[:old_classes] = bank["birth_count"].to(device)
    historical_a, historical_v = torch.zeros_like(sums_a), torch.zeros_like(sums_v)
    historical_a[:old_classes] = bank["birth_sum_audio"].to(device)
    historical_v[:old_classes] = bank["birth_sum_visual"].to(device)
    denom = counts.clamp_min(1).float()[:, None]
    mean_a, mean_v = sums_a / denom, sums_v / denom
    delta_a, delta_v = (sums_a - anchors_a) / denom, (sums_v - anchors_v) / denom
    k = torch.zeros(num_classes, device=device)
    reasons = ["new_class"] * num_classes
    for label in range(old_classes):
        if counts[label] == 0:
            reasons[label] = "no_current_samples"
        elif issues[label]:
            reasons[label] = "+".join(sorted(issues[label]))
        elif anchor_count[label] != counts[label] or birth_count[label] < counts[label]:
            reasons[label] = "anchor_count_mismatch"
        elif prior_strength == 0:
            reasons[label] = "prior_disabled"
        elif birth_count[label] == counts[label]:
            reasons[label] = "no_discarded_birth_samples"
        else:
            k[label] = min(prior_strength, int(birth_count[label] - counts[label]))
            reasons[label] = "history_used"
    history_used = k > 0
    beta = counts.float() / (counts.float() + k).clamp_min(1)
    beta = torch.where(counts > 0, beta, torch.ones_like(beta))
    full_a, full_v = sums_a.clone(), sums_v.clone()
    for total, mean, birth_sum, delta in ((full_a, mean_a, historical_a, delta_a),
                                          (full_v, mean_v, historical_v, delta_v)):
        compensated = birth_sum / birth_count.clamp_min(1).float()[:, None] + delta
        blended = beta[:, None] * mean + (1 - beta[:, None]) * compensated
        total[history_used] = blended[history_used]
    # [P9 沿用] 历史先验不挽救缺少有效当前方向的类；保留 fresh 的保守有效性条件。
    candidate = ((counts > 0) & torch.isfinite(sums_a).all(1) & torch.isfinite(sums_v).all(1)
                 & (sums_a.norm(dim=1) > EPS) & (sums_v.norm(dim=1) > EPS)
                 & torch.isfinite(full_a).all(1) & torch.isfinite(full_v).all(1)
                 & (full_a.norm(dim=1) > EPS) & (full_v.norm(dim=1) > EPS))
    prototypes_a, prototypes_v = F.normalize(full_a, dim=1), F.normalize(full_v, dim=1)
    probability_a, probability_v = torch.zeros(num_classes, device=device), torch.zeros(num_classes, device=device)
    scored_counts = torch.zeros_like(counts)
    if candidate.sum() >= 2:
        for cached_batch in cached:
            audio, visual, labels, birth_a, birth_v = (value.to(device) for value in cached_batch)
            loo_a, loo_v = sums_a[labels] - audio, sums_v[labels] - visual
            fresh_loo_valid = (torch.isfinite(loo_a).all(1) & torch.isfinite(loo_v).all(1)
                               & (loo_a.norm(dim=1) > EPS) & (loo_v.norm(dim=1) > EPS))
            using_history = history_used[labels] & (counts[labels] >= min_samples)
            if using_history.any():
                n_loo = (counts[labels] - 1).clamp_min(1).float()[:, None]
                beta_loo = n_loo / (n_loo + k[labels, None])
                n_birth_loo = (birth_count[labels] - 1).clamp_min(1).float()[:, None]
                # [P9 bank 新增] query 同时从当前均值、birth 均值及漂移锚点排除。
                for loo, anchor_sum, birth_sum, birth_query in (
                    (loo_a, anchors_a, historical_a, birth_a),
                    (loo_v, anchors_v, historical_v, birth_v),
                ):
                    delta_loo = (loo - (anchor_sum[labels] - birth_query)) / n_loo
                    old_loo = (birth_sum[labels] - birth_query) / n_birth_loo
                    blended = beta_loo * (loo / n_loo) + (1 - beta_loo) * (old_loo + delta_loo)
                    loo[using_history] = blended[using_history]
            valid = (candidate[labels] & fresh_loo_valid & (counts[labels] >= min_samples)
                     & torch.isfinite(loo_a).all(1) & torch.isfinite(loo_v).all(1)
                     & (loo_a.norm(dim=1) > EPS) & (loo_v.norm(dim=1) > EPS))
            audio, visual, labels = audio[valid], visual[valid], labels[valid]
            if not labels.numel():
                continue
            score_a = (audio @ prototypes_a.T) / temperature
            score_v = (visual @ prototypes_v.T) / temperature
            target_a = (audio * F.normalize(loo_a[valid], dim=1)).sum(1) / temperature
            target_v = (visual * F.normalize(loo_v[valid], dim=1)).sum(1) / temperature
            score_a.scatter_(1, labels[:, None], target_a[:, None])
            score_v.scatter_(1, labels[:, None], target_v[:, None])
            score_a.masked_fill_(~candidate[None, :], -torch.inf)
            score_v.masked_fill_(~candidate[None, :], -torch.inf)
            probability_a.index_add_(0, labels, torch.exp(target_a - torch.logsumexp(score_a, dim=1)))
            probability_v.index_add_(0, labels, torch.exp(target_v - torch.logsumexp(score_v, dim=1)))
            scored_counts.index_add_(0, labels, torch.ones_like(labels))
    reliability_a, reliability_v = probability_a / scored_counts.clamp_min(1), probability_v / scored_counts.clamp_min(1)
    valid = ((scored_counts >= min_samples) & (scored_counts == counts)
             & torch.isfinite(reliability_a) & torch.isfinite(reliability_v)
             & ((reliability_a + reliability_v) > EPS))
    reliable = ReliabilityBank(*(value.detach().cpu() for value in
        (counts, scored_counts, reliability_a, reliability_v, valid)))
    # 漂移诊断仅在全部当前样本都有有效 birth anchor 时有意义；其余类记 NaN。
    drift_valid = (counts > 0) & (anchor_count == counts) & (birth_count >= counts)
    def drift_metric(value):
        return torch.where(drift_valid, value, torch.full_like(value, float("nan"))).cpu()
    diagnostics = {
        "birth_count": birth_count.cpu(), "anchor_count": anchor_count.cpu(),
        "history_used": history_used.cpu(), "prior_strength_effective": k.cpu(),
        "prototype_beta": beta.cpu(), "reason": reasons,
        "drift_audio_norm": drift_metric(delta_a.norm(dim=1)),
        "drift_visual_norm": drift_metric(delta_v.norm(dim=1)),
        "drift_audio_dispersion": drift_metric((drift_sq_a / counts.clamp_min(1) - delta_a.square().sum(1)).clamp_min(0).sqrt()),
        "drift_visual_dispersion": drift_metric((drift_sq_v / counts.clamp_min(1) - delta_v.square().sum(1)).clamp_min(0).sqrt()),
        "prototype_audio": prototypes_a.cpu(), "prototype_visual": prototypes_v.cpu(),
    }
    return reliable, diagnostics
