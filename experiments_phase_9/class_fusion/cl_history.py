"""Observation-only CL history on a fixed per-task old-memory cohort.

Teacher and student each construct their OWN same-modality prototypes, using
the same retained videos and old-class candidates. Thus an orthogonal feature
rotation is not itself forgetting. No historical metric enters a loss or gate.
Raw input features and per-video hidden embeddings are not saved.
Old-only branch scores and actual fused logits are retained for later analysis.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.audio_visual_model_incremental_class_fusion import class_conditional_logits
from .diagnostics import append_csv_row
from .fusion_method import FusionReferenceDataset, preserve_rng_state, unwrap_model

# [P9 新增：整个模块] P8 没有这套固定旧类 memory 的双模态遗忘历史记录。
# 本模块的教师/学生配对观测、预测转移、旧新混淆及落盘均只用于分析，不进入 loss/gate。
# 局部 LOO/log-odds 公式与 P8 rd_method 的联系在下方标出，不代表继承了 CMR 算法。

HISTORY_SCHEMA_VERSION = 2
PAIR_METRICS = ("both_correct_rate", "audio_only_correct_rate", "visual_only_correct_rate",
                "both_wrong_rate", "prediction_disagreement_rate")


def class_mean(values, labels, num_classes, valid_classes):
    """Keep unsupported/incomparable classes as NaN, never as zero forgetting."""
    finite = torch.isfinite(values)
    sums = torch.zeros(num_classes)
    counts = torch.bincount(labels[finite], minlength=num_classes)
    sums.index_add_(0, labels[finite], values[finite].float())
    result = torch.full((num_classes,), torch.nan)
    valid = valid_classes & (counts > 0)
    result[valid] = (sums / counts.clamp_min(1))[valid]
    return result


@torch.no_grad()
def collect_memory_outputs(model, dataset, batch_size, num_workers, device):
    """One deterministic pass; restore modes/RNG even if feature extraction fails."""
    net = unwrap_model(model)
    modes = [(module, module.training) for module in model.modules()]
    with preserve_rng_state():
        model.eval()
        try:
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
            audio, visual, logits, labels = [], [], [], []
            for data, target in loader:
                a, v, _, _ = net.extract_branch_features(data[0].to(device), data[1].to(device))
                z = class_conditional_logits(a, v, net.classifier, net.fusion_gate, net.fusion_chunk_size)
                audio.append(a.detach().float().cpu())
                visual.append(v.detach().float().cpu())
                logits.append(z.detach().float().cpu())
                labels.append(target.detach().long().cpu())
            if not labels:
                raise ValueError("CL history requires nonempty old replay memory")
            return {"audio": torch.cat(audio), "visual": torch.cat(visual),
                    "logits": torch.cat(logits), "labels": torch.cat(labels)}
        finally:
            for module, training in modes:
                module.training = training


def paired_feature_valid(outputs):
    return (torch.isfinite(outputs["audio"]).all(1) & torch.isfinite(outputs["visual"]).all(1)
            & (outputs["audio"].norm(dim=1) > 1e-12) & (outputs["visual"].norm(dim=1) > 1e-12))


def old_class_reliability(audio, visual, labels, num_classes, candidates, temperature, min_samples):
    """LOO confidence with a FIXED candidate mask and a FIXED ordered cohort.

    A missing/nonfinite/zero current feature invalidates the snapshot instead
    of silently changing the cohort. An unusable candidate prototype also
    invalidates it globally because it would change every softmax denominator.
    Individual degenerate LOO queries invalidate the corresponding class only.
    """
    counts = torch.bincount(labels, minlength=num_classes)
    result = {"counts": counts, "candidate_mask": candidates.clone(),
              "valid": torch.zeros(num_classes, dtype=torch.bool),
              "scored_counts": torch.zeros(num_classes, dtype=torch.long),
              "reason": "ok"}
    for branch in ("a", "v"):
        result["reliability_" + branch] = torch.full((num_classes,), torch.nan)
        result["sample_probability_" + branch] = torch.full((labels.numel(),), torch.nan)
        result["sample_scores_" + branch] = torch.full((labels.numel(), num_classes), torch.nan)
        result["sample_prediction_" + branch] = torch.full((labels.numel(),), -1, dtype=torch.long)
        result["sample_top1_margin_" + branch] = torch.full((labels.numel(),), torch.nan)
        result["sample_log_odds_" + branch] = torch.full((labels.numel(),), torch.nan)
        result["accuracy_" + branch] = torch.full((num_classes,), torch.nan)
    result["paired_correctness"] = {key: torch.full((num_classes,), torch.nan) for key in PAIR_METRICS}
    result["query_valid"] = torch.zeros(labels.numel(), dtype=torch.bool)
    valid_features = (torch.isfinite(audio).all(1) & torch.isfinite(visual).all(1)
                      & (audio.norm(dim=1) > 1e-12) & (visual.norm(dim=1) > 1e-12))
    if not valid_features.all():
        result["reason"] = "invalid_current_reference_features"
        return result
    if candidates.sum() < 2:
        result["reason"] = "fewer_than_two_reference_classes"
        return result
    scores, query_valid = {}, counts[labels] >= min_samples
    for name, features in (("a", audio), ("v", visual)):
        u = F.normalize(features, dim=1)
        sums = torch.zeros(num_classes, features.shape[1])
        sums.index_add_(0, labels, u)
        if ((sums.norm(dim=1) <= 1e-12) & candidates).any():
            result["reason"] = "invalid_current_candidate_prototype"
            return result
        # [P8 逻辑沿用] rd_crosssdc/rd_method.py::_cross_modal_margin 中的
        # normalize(类特征和 - 当前样本) 及目标 score 替换公式。
        # [P9 新增] 此处限定固定旧类/固定视频，同一模型内部同模态打分，并显式记录无效项。
        prototype = F.normalize(sums, dim=1)
        loo = sums[labels] - u
        query_valid = query_valid & (loo.norm(dim=1) > 1e-12)
        score = (u @ prototype.T) / temperature
        score.scatter_(1, labels[:, None], ((u * F.normalize(loo, dim=1)).sum(1) / temperature)[:, None])
        score.masked_fill_(~candidates[None, :], -torch.inf)
        scores[name] = score
    scored_counts = torch.bincount(labels[query_valid], minlength=num_classes)
    valid = (counts >= min_samples) & (scored_counts == counts) & candidates
    result.update(valid=valid, scored_counts=scored_counts, query_valid=query_valid)
    for name, score in scores.items():
        probabilities = score.softmax(1).gather(1, labels[:, None]).squeeze(1)
        result["sample_probability_" + name][query_valid] = probabilities[query_valid]
        result["sample_scores_" + name][query_valid] = score[query_valid]
        result["sample_prediction_" + name][query_valid] = score[query_valid].argmax(1)
        target = score.gather(1, labels[:, None]).squeeze(1)
        competitors = score.clone().scatter_(1, labels[:, None], -torch.inf)
        result["sample_top1_margin_" + name][query_valid] = (target - competitors.max(1).values)[query_valid]
        # [P8 逻辑沿用] _cross_modal_margin 的 target - logsumexp(rest) 公式；
        # 此处仅保存同模态观测值，不计算教师-学生 CMR 惩罚。
        result["sample_log_odds_" + name][query_valid] = (target - torch.logsumexp(competitors, dim=1))[query_valid]
        correct = (result["sample_prediction_" + name] == labels).float()
        correct[~query_valid] = torch.nan
        result["accuracy_" + name] = class_mean(correct, labels, num_classes, valid)
        sums = torch.zeros(num_classes)
        sums.index_add_(0, labels[query_valid], probabilities[query_valid])
        result["reliability_" + name][valid] = (sums / counts.clamp_min(1))[valid]
    correct_a = result["sample_prediction_a"] == labels
    correct_v = result["sample_prediction_v"] == labels
    paired = {
        "both_correct_rate": correct_a & correct_v,
        "audio_only_correct_rate": correct_a & ~correct_v,
        "visual_only_correct_rate": ~correct_a & correct_v,
        "both_wrong_rate": ~correct_a & ~correct_v,
        "prediction_disagreement_rate": result["sample_prediction_a"] != result["sample_prediction_v"],
    }
    for key, value in paired.items():
        samples = value.float()
        samples[~query_valid] = torch.nan
        result["paired_correctness"][key] = class_mean(samples, labels, num_classes, valid)
    return result


def memory_classification(logits, labels, num_old_classes):
    """Separate within-old prediction from confusion with the new classes.

    These are memory diagnostics, never held-out accuracy estimates. Counts
    and a validity flag accompany metrics so invalid outputs cannot disappear
    unnoticed from the denominator.
    """
    counts = torch.bincount(labels, minlength=num_old_classes)
    finite = torch.isfinite(logits).all(1)
    support = torch.bincount(labels[finite], minlength=num_old_classes)
    confusion = torch.zeros(num_old_classes, logits.shape[1], dtype=torch.long)
    values = {name: torch.full((num_old_classes,), torch.nan)
              for name in ("old_only_accuracy", "all_seen_accuracy", "old_to_new_rate")}
    valid = (counts > 0) & (support == counts)
    sample_prediction_all = torch.full_like(labels, -1)
    sample_prediction_old = torch.full_like(labels, -1)
    if finite.any():
        targets, scores = labels[finite], logits[finite]
        pred = scores.argmax(1)
        old_pred = scores[:, :num_old_classes].argmax(1)
        sample_prediction_all[finite] = pred
        sample_prediction_old[finite] = old_pred
        confusion = torch.bincount(targets * logits.shape[1] + pred,
                                   minlength=num_old_classes * logits.shape[1]).reshape(num_old_classes, -1)
        for name, observed in (("old_only_accuracy", old_pred == targets),
                               ("all_seen_accuracy", pred == targets),
                               ("old_to_new_rate", pred >= num_old_classes)):
            sums = torch.zeros(num_old_classes)
            sums.index_add_(0, targets, observed.float())
            values[name][valid] = (sums / support.clamp_min(1))[valid]
    return {**values, "classification_valid": valid, "classification_support": support,
            "confusion_matrix": confusion, "sample_valid": finite,
            "sample_logits": logits.detach().cpu().clone(),
            "sample_prediction_all_seen": sample_prediction_all,
            "sample_prediction_old_only": sample_prediction_old}


def branch_prediction_transitions(baseline, current, labels, comparable):
    """Paired correct->wrong and wrong->correct changes of the LOO probes.

    These are probe transitions on replay memory, not test-set forgetting.
    Keep sample-level masks so confidence degradation can be distinguished
    from actual changes of the predicted old class.
    """
    mask = baseline["query_valid"] & current["query_valid"] & comparable[labels]
    result = {"sample_comparable": mask, "counts": torch.bincount(labels[mask], minlength=comparable.numel())}
    for branch in ("a", "v"):
        before_correct = baseline["sample_prediction_" + branch] == labels
        now_correct = current["sample_prediction_" + branch] == labels
        for name, value in (("forgotten", before_correct & ~now_correct),
                            ("recovered", ~before_correct & now_correct)):
            samples = value.float()
            samples[~mask] = torch.nan
            result["sample_" + name + "_" + branch] = samples
            result[name + "_rate_" + branch] = class_mean(samples, labels, comparable.numel(), comparable)
    return result


def should_record_cl_history(args, step, epoch, is_best=False):
    if not args.record_cl_history or step == 0:
        return False
    # Record the exact state of every selected checkpoint, not a stale periodic
    # measurement. Gate updates themselves are logged separately after this.
    from .fusion_method import should_update_gate
    return (is_best or epoch == args.max_epoches or epoch % args.cl_history_interval == 0
            or should_update_gate(args, step, epoch))


class CLHistoryRecorder:
    def __init__(self, args, step, teacher, exemplar_set, teacher_metadata, root, id_to_category, device):
        self.args, self.step, self.device = args, step, device
        self.num_old_classes = step * args.class_num_per_step
        self.reference_set = FusionReferenceDataset(exemplar_set)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.id_to_category = id_to_category
        outputs = self._collect(teacher)
        labels = outputs["labels"]
        if ((labels < 0) | (labels >= self.num_old_classes)).any():
            raise ValueError("CL reference must contain old-class training memory only")
        included = paired_feature_valid(outputs)
        selected_labels = labels[included]
        counts = torch.bincount(selected_labels, minlength=self.num_old_classes)
        candidates = counts > 0
        manifest = {
            "schema_version": HISTORY_SCHEMA_VERSION, "step": step,
            "num_old_classes": self.num_old_classes, "dataset": args.dataset,
            "seed": args.seed, "experiment_name": args.experiment_name,
            "observation_only": True,
            "sample_ids": self.reference_set.sample_ids.copy(), "labels": labels.tolist(),
            "included": included.tolist(), "candidate_class_ids": torch.where(candidates)[0].tolist(),
            "temperature": args.fusion_temperature, "min_samples": args.fusion_min_samples,
            "scope": "fixed_old_memory_and_old_classes",
            "prototype_policy": "each_model_rebuilds_own_same_modality_LOO_prototypes",
            "score_definition": "unit_feature_dot_unit_LOO_prototype_divided_by_temperature",
            "branch_score_columns": "global_old_class_ids_0_to_num_old_classes_minus_1",
            "sample_row_indices": torch.where(included)[0].tolist(),
            "teacher_source": teacher_metadata,
        }
        reference_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        self.reference = {
            **manifest, "reference_id": reference_id, "sample_labels": labels, "included_mask": included,
            "candidate_mask": candidates, "memory_counts": torch.bincount(labels, minlength=self.num_old_classes),
            "teacher_gate": unwrap_model(teacher).fusion_gate.detach().cpu().clone(),
            "teacher_gate_version": int(unwrap_model(teacher).fusion_version),
            "teacher_reliability": old_class_reliability(outputs["audio"][included], outputs["visual"][included],
                                                        selected_labels, self.num_old_classes, candidates,
                                                        args.fusion_temperature, args.fusion_min_samples),
            "teacher_classification": memory_classification(outputs["logits"][included], selected_labels, self.num_old_classes),
        }
        with (self.root / f"step_{step}_reference.json").open("w", encoding="utf-8") as handle:
            json.dump({**manifest, "reference_id": reference_id}, handle, ensure_ascii=False, indent=2)
        torch.save(self.reference, self.root / f"step_{step}_reference.pt")

    def _collect(self, model):
        # Indices alone are insufficient: replacing a video with another video
        # from the same class would leave the observed labels unchanged.
        current_ids = []
        for source_id, index in self.reference_set.indices:
            source = self.reference_set.sources[source_id]
            vids = getattr(source, "all_current_data_vids", None)
            if vids is None:
                vids = source.exemplar_vids_set
            if index >= len(vids):
                raise ValueError("CL reference cohort/order changed inside a task")
            current_ids.append(str(vids[index]))
        if current_ids != self.reference_set.sample_ids:
            raise ValueError("CL reference cohort/order changed inside a task")
        return collect_memory_outputs(model, self.reference_set, self.args.fusion_batch_size,
                                      self.args.num_workers, self.device)

    def observe(self, model, epoch, events):
        outputs = self._collect(model)
        if (self.reference_set.sample_ids != self.reference["sample_ids"]
                or not torch.equal(outputs["labels"], self.reference["sample_labels"])):
            raise ValueError("CL reference cohort/order changed inside a task")
        included = self.reference["included_mask"]
        labels = outputs["labels"][included]
        current = old_class_reliability(outputs["audio"][included], outputs["visual"][included], labels,
                                        self.num_old_classes, self.reference["candidate_mask"],
                                        self.args.fusion_temperature, self.args.fusion_min_samples)
        baseline = self.reference["teacher_reliability"]
        comparable = baseline["valid"] & current["valid"]
        signed_a = baseline["reliability_a"] - current["reliability_a"]
        signed_v = baseline["reliability_v"] - current["reliability_v"]
        signed_a[~comparable], signed_v[~comparable] = torch.nan, torch.nan
        net = unwrap_model(model)
        observed = {
            "schema_version": HISTORY_SCHEMA_VERSION, "reference_id": self.reference["reference_id"],
            "step": self.step, "epoch": epoch, "events": list(events),
            "num_old_classes": self.num_old_classes, "num_seen_classes": net.num_classes,
            "observation_only": True,
            "measurement_stage": "task_start_before_training" if epoch == 0 else "after_training_before_gate_update",
            "gate_version": int(net.fusion_version), "audio_gate": net.fusion_gate.detach().cpu().clone(),
            "reliability": current, "comparable": comparable,
            "signed_drop_a": signed_a, "signed_drop_v": signed_v,
            "drop_a": signed_a.clamp_min(0), "drop_v": signed_v.clamp_min(0),
            "drop_asymmetry": signed_a.clamp_min(0) - signed_v.clamp_min(0),
            "prediction_transitions": branch_prediction_transitions(baseline, current, labels, comparable),
            "classification": memory_classification(outputs["logits"][included], labels, self.num_old_classes),
        }
        self._write(observed)
        return {"reference": self.reference, "observation": observed}

    def _write(self, observed):
        torch.save(observed, self.root / f"step_{self.step}_epoch_{observed['epoch']}_history.pt")
        teacher, current = self.reference["teacher_reliability"], observed["reliability"]
        for c in range(self.num_old_classes):
            row = {
                "step": self.step, "epoch": observed["epoch"], "events": "+".join(observed["events"]),
                "reference_id": self.reference["reference_id"], "class_id": c,
                "category_name": self.id_to_category.get(c, str(c)),
                "introduced_step": c // self.args.class_num_per_step,
                "class_age_steps": self.step - c // self.args.class_num_per_step,
                "memory_count": int(self.reference["memory_counts"][c]),
                "reference_count": int(teacher["counts"][c]),
                "teacher_scored_count": int(teacher["scored_counts"][c]),
                "student_scored_count": int(current["scored_counts"][c]),
                "teacher_valid": bool(teacher["valid"][c]), "student_valid": bool(current["valid"][c]),
                "comparable": bool(observed["comparable"][c]),
                "teacher_status": teacher["reason"], "student_status": current["reason"],
                "teacher_R_a": float(teacher["reliability_a"][c]), "teacher_R_v": float(teacher["reliability_v"][c]),
                "student_R_a": float(current["reliability_a"][c]), "student_R_v": float(current["reliability_v"][c]),
                "signed_drop_a": float(observed["signed_drop_a"][c]), "signed_drop_v": float(observed["signed_drop_v"][c]),
                "drop_a": float(observed["drop_a"][c]), "drop_v": float(observed["drop_v"][c]),
                "drop_asymmetry": float(observed["drop_asymmetry"][c]),
                "teacher_audio_gate": float(self.reference["teacher_gate"][c]),
                "student_audio_gate": float(observed["audio_gate"][c]), "gate_version": observed["gate_version"],
            }
            for name in ("old_only_accuracy", "all_seen_accuracy", "old_to_new_rate", "classification_support", "classification_valid"):
                row["teacher_" + name] = self.reference["teacher_classification"][name][c].item()
                row["student_" + name] = observed["classification"][name][c].item()
            for name in ("a", "v"):
                row["teacher_probe_accuracy_" + name] = teacher["accuracy_" + name][c].item()
                row["student_probe_accuracy_" + name] = current["accuracy_" + name][c].item()
                for metric in ("forgotten_rate_", "recovered_rate_"):
                    row[metric + name] = observed["prediction_transitions"][metric + name][c].item()
            for name in PAIR_METRICS:
                row["teacher_" + name] = teacher["paired_correctness"][name][c].item()
                row["student_" + name] = current["paired_correctness"][name][c].item()
            append_csv_row(self.root / "class_history.csv", list(row), row)
