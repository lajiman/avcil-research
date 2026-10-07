"""Persistent teacher prototypes for the B and C experiments.

B saves exact statistics of H = current new training data + current replay,
under the *selected best* model, before the next memory reduction. C adds
retired-sample statistics transported into that same model's feature space.
Neither policy updates its bank while the next task is training.

C uses fixed ID folds for every historical transition. For a query in fold f,
both modalities and all candidate classes use history whose drift estimates
and confidence estimates have never used anchors in f. This is prototype-level
cross-fitting; it does not undo the model's earlier training on those samples.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dataloader_ours import h5_data_loader_kwargs
from loader_lifecycle import close_data_loader


SCHEMA_VERSION = 1
_EPS = 1e-12


def _policy(args):
    return str(args.rd_prototype_policy)


def _history_config(args):
    defaults = {
        "folds": 5,
        "seed": int(args.seed),
        "min_anchors": 2,
        "decay": 0.9,
        "error_scale": 0.25,
        "mass_cap": 50.0,
    }
    # B has no historical estimator. Unused C flags must neither alter B nor
    # prevent reloading its artifact; seed remains part of the run provenance.
    if _policy(args) != "historical":
        return defaults
    return {name: (int if name in ("folds", "min_anchors", "seed") else float)(
        getattr(args, "rd_history_" + name, value) if name != "seed" else value)
            for name, value in defaults.items()}


def _is_history(policy):
    return policy == "historical"


def _check_config(config):
    if config["folds"] < 2 or config["min_anchors"] < 2:
        raise ValueError("History needs at least two folds and two anchors")
    if not math.isfinite(config["decay"]) or not 0 <= config["decay"] <= 1:
        raise ValueError("History decay must lie in [0, 1]")
    if not math.isfinite(config["error_scale"]) or config["error_scale"] <= 0:
        raise ValueError("History error scale must be finite and positive")
    if not math.isfinite(config["mass_cap"]) or config["mass_cap"] < 0:
        raise ValueError("History mass cap must be finite and non-negative")


def _fold(vid, config):
    digest = hashlib.sha256((str(config["seed"]) + "\0" + vid).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") % config["folds"]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_sha256(model):
    """Bind the artifact to supplied model weights, not only a filename."""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Prototype models must have tensor-only state dictionaries")
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _evaluation_state(model):
    """Extra scans must not change training RNG or mixed module train modes."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _dataset_ids(dataset):
    mode = getattr(dataset, "mode", None)
    if mode is not None and mode != "train":
        raise ValueError("Prototype statistics accept training data only")
    if hasattr(dataset, "exemplar_vids_set"):
        ids = dataset.exemplar_vids_set
    elif mode == "train" and hasattr(dataset, "all_current_data_vids"):
        ids = dataset.all_current_data_vids
    else:
        raise TypeError("Expected the training or replay dataset interface")
    if len(ids) != len(dataset):
        raise ValueError("Dataset ID list and dataset length disagree")
    if any(not isinstance(vid, str) or not vid for vid in ids):
        raise ValueError("Every support sample must have a non-empty string ID")
    return list(ids)


def _class_order(dataset):
    mapping = dataset.category_encode_dict
    return sorted([[str(category), int(label)] for category, label in mapping.items()],
                  key=lambda pair: (pair[1], pair[0]))


def _dataset_labels(dataset, ids):
    try:
        return [int(dataset.category_encode_dict[dataset.all_id_category_dict[vid]])
                for vid in ids]
    except KeyError as error:
        raise ValueError("Support ID absent from training label metadata: {}".format(error)) from error


class IndexedReplayDataset(Dataset):
    """Keep replay order stable while returning its original query index."""

    def __init__(self, dataset):
        _dataset_ids(dataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index


class _SupportDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        dataset, source_index, _, expected_label = self.entries[index]
        data, label = dataset[source_index]
        if int(label) != expected_label:
            raise ValueError("Dataset label changed for {!r}".format(self.entries[index][2]))
        return data, label


def _support_entries(args, step, train_set, exemplar_set):
    classes_per_step = int(args.class_num_per_step)
    seen = min(int(args.num_classes), (step + 1) * classes_per_step)
    entries, manifest = [], {}
    sources = [(train_set, step * classes_per_step, seen)]
    if step > 0:
        if _class_order(train_set) != _class_order(exemplar_set):
            raise ValueError("Training and replay class mappings disagree")
        sources.append((exemplar_set, 0, step * classes_per_step))
    for dataset, low, high in sources:
        ids = _dataset_ids(dataset)
        labels = _dataset_labels(dataset, ids)
        for index, (vid, label) in enumerate(zip(ids, labels)):
            if not low <= label < high:
                raise ValueError("Out-of-task support sample {!r}, class {}".format(vid, label))
            if vid in manifest:
                if manifest[vid] != label:
                    raise ValueError("Conflicting labels for support ID {!r}".format(vid))
                raise ValueError("Duplicate support ID {!r}; each video must occur once".format(vid))
            manifest[vid] = label
            entries.append((dataset, index, vid, label))
    if not entries:
        raise ValueError("Cannot build a prototype bank from an empty support set")
    return entries, seen


def _extract(model, dataset, ids, labels, args, device):
    if not ids:
        raise ValueError("Cannot encode an empty prototype reference set")
    loader = DataLoader(
        dataset,
        batch_size=min(int(args.exemplar_batch_size), len(dataset)),
        shuffle=False,
        drop_last=False,
        **h5_data_loader_kwargs(int(args.num_workers), persistent_workers=int(args.num_workers) > 0),
    )
    audio_parts, visual_parts, offset = [], [], 0
    try:
        with _evaluation_state(model):
            for data, observed in loader:
                observed = observed.long().cpu()
                end = offset + len(observed)
                if observed.tolist() != labels[offset:end]:
                    raise ValueError("Reference loader changed ID/label ordering")
                output = model(visual=data[0].to(device), audio=data[1].to(device),
                               out_feature_before_fusion=True)
                if isinstance(output, dict):
                    audio, visual = output["audio_feature"], output["visual_feature"]
                else:
                    _, audio, visual = output
                if audio.ndim != 2 or visual.ndim != 2 or audio.shape != visual.shape:
                    raise ValueError("Expected paired branch features with shape (batch, dimension)")
                valid = torch.isfinite(audio).all(1) & torch.isfinite(visual).all(1)
                valid &= (audio.norm(dim=1) > _EPS) & (visual.norm(dim=1) > _EPS)
                if not bool(valid.all()):
                    bad = [ids[offset + i] for i in torch.where(~valid)[0].cpu().tolist()]
                    raise FloatingPointError("Invalid paired prototype features for IDs: {}".format(bad))
                for modality, feature in (("audio", audio), ("visual", visual)):
                    normalized = torch.isclose(feature.norm(dim=1), torch.ones_like(feature[:, 0]),
                                               atol=1e-4, rtol=1e-4)
                    if not bool(normalized.all()):
                        bad = [ids[offset + i] for i in torch.where(~normalized)[0].cpu().tolist()]
                        raise ValueError("{} branch must return normalized features: {}".format(modality, bad))
                audio_parts.append(audio.detach().float().cpu())
                visual_parts.append(visual.detach().float().cpu())
                offset = end
    finally:
        close_data_loader(loader)
    if offset != len(ids):
        raise ValueError("Prototype scan did not visit every support ID")
    return torch.cat(audio_parts), torch.cat(visual_parts)


def _sums(features, labels, num_classes):
    result = torch.zeros(num_classes, features.shape[1], dtype=features.dtype)
    result.index_add_(0, labels, features)
    return result


def _empty_history(folds, classes, dimension):
    return {
        "history_mean_a": torch.zeros(folds, classes, dimension),
        "history_mean_v": torch.zeros(folds, classes, dimension),
        "history_mass_a": torch.zeros(folds, classes),
        "history_mass_v": torch.zeros(folds, classes),
        "retired_unique_counts": torch.zeros(classes, dtype=torch.long),
        "retired_ids": [],
        "retired_labels": [],
        "anchor_counts": torch.zeros(folds, classes, dtype=torch.long),
        "shift_a": torch.zeros(folds, classes, dimension),
        "shift_v": torch.zeros(folds, classes, dimension),
        "residual_a": torch.zeros(folds, classes),
        "residual_v": torch.zeros(folds, classes),
        "confidence_a": torch.zeros(folds, classes),
        "confidence_v": torch.zeros(folds, classes),
    }


def _advance_history(history, previous_bank, ids, labels, audio, visual, config):
    """Migrate older retirees plus newly retired J, without reading J again."""
    previous = previous_bank.artifact
    old_classes = len(previous["counts"])
    current_index = {vid: index for index, vid in enumerate(ids)}
    query_ids = previous_bank.query_ids
    if not set(query_ids).issubset(current_index):
        raise ValueError("Current support dropped an anchor before bank construction")
    current_old_ids = {vid for vid, label in zip(ids, labels.tolist()) if label < old_classes}
    if current_old_ids != set(query_ids):
        raise ValueError("Current old-class support must be exactly the verified replay memory")
    positions = torch.tensor([current_index[vid] for vid in query_ids], dtype=torch.long)
    anchor_labels = previous_bank.query_labels.detach().cpu()
    if not torch.equal(labels.index_select(0, positions), anchor_labels):
        raise ValueError("Anchor labels changed between tasks")
    anchor_folds = previous_bank.query_folds.detach().cpu()
    query_set = set(query_ids)
    newly_retired = [(vid, label) for vid, label in
                     zip(previous["support_ids"], previous["support_labels"])
                     if vid not in query_set]
    retired_ids = list(previous["retired_ids"]) + [item[0] for item in newly_retired]
    retired_labels = list(previous["retired_labels"]) + [item[1] for item in newly_retired]
    if len(retired_ids) != len(set(retired_ids)) or set(retired_ids).intersection(ids):
        raise ValueError("Retired statistics overlap current support or were counted twice")
    history["retired_ids"], history["retired_labels"] = retired_ids, retired_labels
    history["retired_unique_counts"] = torch.bincount(
        torch.tensor(retired_labels, dtype=torch.long), minlength=len(history["retired_unique_counts"]))
    query_counts = torch.bincount(anchor_labels, minlength=old_classes).float()
    newly_counts = previous["counts"] - query_counts
    if bool((newly_counts < 0).any()):
        raise ValueError("Replay query count exceeds previous support count")
    expected_retired = previous["retired_unique_counts"] + newly_counts.long()
    if not torch.equal(history["retired_unique_counts"][:old_classes], expected_retired):
        raise ValueError("Retired ID ledger and support subtraction disagree")

    for modality, old_features, new_features in (
        ("a", previous_bank.query_audio.detach().cpu(), audio.index_select(0, positions)),
        ("v", previous_bank.query_visual.detach().cpu(), visual.index_select(0, positions)),
    ):
        exact_old = previous["sums_" + modality]
        newly_sums = exact_old - _sums(old_features, anchor_labels, old_classes)
        # No retired sample means exactly zero mass; discard summation roundoff.
        newly_sums[newly_counts == 0] = 0
        deltas = new_features - old_features
        for fold in range(config["folds"]):
            for cls in range(old_classes):
                mask = (anchor_labels == cls) & (anchor_folds != fold)
                n = int(mask.sum())
                history["anchor_counts"][fold, cls] = n
                prior_mass = previous["history_mass_" + modality][fold, cls]
                mass_pre = prior_mass + newly_counts[cls]
                if n < config["min_anchors"] or mass_pre <= 0:
                    continue
                delta = deltas[mask]
                shift = delta.mean(0)
                # Exact leave-one-anchor-out residual for the mean-shift model.
                residual = ((n / (n - 1)) * (delta - shift)).square().sum(1).mean()
                confidence = config["decay"] * torch.exp(-residual / config["error_scale"] ** 2)
                source_sum = (prior_mass * previous["history_mean_" + modality][fold, cls]
                              + newly_sums[cls])
                mean = source_sum / mass_pre + shift
                mass = mass_pre.clamp(max=config["mass_cap"]) * confidence
                if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(mass)):
                    raise FloatingPointError("Non-finite transported history for class {}".format(cls))
                history["history_mean_" + modality][fold, cls] = mean
                history["history_mass_" + modality][fold, cls] = mass
                history["shift_" + modality][fold, cls] = shift
                history["residual_" + modality][fold, cls] = residual
                history["confidence_" + modality][fold, cls] = confidence


def _atomic_save(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {key: artifact[key] for key in (
        "schema_version", "policy", "step", "num_classes", "checkpoint_sha256",
        "model_sha256", "parent_checkpoint_sha256", "history_config", "class_order")}
    summary.update({
        "support_count": len(artifact["support_ids"]),
        "support_counts": artifact["counts"].tolist(),
        "concentration_a": artifact["concentration_a"].tolist(),
        "concentration_v": artifact["concentration_v"].tolist(),
        "retired_unique_counts": artifact["retired_unique_counts"].tolist(),
        "history_mass_a": artifact["history_mass_a"].tolist(),
        "history_mass_v": artifact["history_mass_v"].tolist(),
        "anchor_counts": artifact["anchor_counts"].tolist(),
        "residual_a": artifact["residual_a"].tolist(),
        "residual_v": artifact["residual_v"].tolist(),
        "confidence_a": artifact["confidence_a"].tolist(),
        "confidence_v": artifact["confidence_v"].tolist(),
        "note": "Trust uses the next task's replay queries; support counts are not query counts.",
    })
    for destination, writer in (
        (path, lambda handle: torch.save(artifact, handle)),
        (path.with_suffix(".json"), lambda handle: handle.write(
            json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"))),
    ):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                             prefix=destination.name + ".", suffix=".tmp",
                                             delete=False) as handle:
                temporary = handle.name
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)


def save_task_bank(args, step, best_model, checkpoint_path, train_set, exemplar_set,
                   device, previous_bank=None, path=None):
    """Save B/C statistics at task end, before the next memory reduction.

    The caller must load ``best_model`` from ``checkpoint_path`` first. We store
    both the file hash and the actual model-state hash; the next task verifies
    both against its teacher. No last-epoch/student state is substituted here.
    """
    if path is None:
        raise ValueError("An explicit prototype artifact path is required")
    policy, config = _policy(args), _history_config(args)
    _check_config(config)
    if policy not in ("pre_shrink", "historical"):
        raise ValueError("Persistent banks support only pre_shrink (B) and historical (C)")
    if step > 0 and previous_bank is None:
        raise ValueError("A continued B/C run needs its previous verified teacher bank")
    if previous_bank is not None:
        if previous_bank.artifact["step"] != step - 1 or previous_bank.policy != policy:
            raise ValueError("Prototype history has an incompatible parent task/policy")
        if previous_bank.artifact["history_config"] != config:
            raise ValueError("Prototype history configuration changed within a run")
        if previous_bank.artifact["class_order"] != _class_order(train_set):
            raise ValueError("Prototype history class order changed within a run")
        if previous_bank.artifact["num_classes"] != step * int(args.class_num_per_step):
            raise ValueError("Prototype history has an incompatible class schedule")
    entries, seen = _support_entries(args, step, train_set, exemplar_set)
    ids, label_list = [entry[2] for entry in entries], [entry[3] for entry in entries]
    labels = torch.tensor(label_list, dtype=torch.long)
    if previous_bank is not None:
        current_old_ids = {vid for vid, label in zip(ids, label_list)
                           if label < step * int(args.class_num_per_step)}
        if current_old_ids != set(previous_bank.query_ids):
            raise ValueError("Current old-class support must equal the loaded replay memory")
    audio, visual = _extract(best_model, _SupportDataset(entries), ids, label_list, args, device)
    counts = torch.bincount(labels, minlength=seen).float()
    if bool((counts < 2).any()):
        raise ValueError("B/C require at least two exact support samples for every class")
    sums_a, sums_v = _sums(audio, labels, seen), _sums(visual, labels, seen)
    history = _empty_history(config["folds"], seen, audio.shape[1])
    if _is_history(policy) and previous_bank is not None:
        _advance_history(history, previous_bank, ids, labels, audio, visual, config)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "step": int(step),
        "num_classes": seen,
        "dataset": str(args.dataset),
        "class_num_per_step": int(args.class_num_per_step),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "model_sha256": _model_sha256(best_model),
        "parent_checkpoint_sha256": (previous_bank.artifact["checkpoint_sha256"]
                                     if previous_bank is not None else None),
        "class_order": _class_order(train_set),
        "history_config": config,
        "support_ids": ids,
        "support_labels": label_list,
        "support_folds": [_fold(vid, config) for vid in ids],
        "sums_a": sums_a,
        "sums_v": sums_v,
        "counts": counts,
        # Mean resultant length retains dispersion information lost by storing
        # only a unit prototype. This diagnostic does not alter B or C weights.
        "concentration_a": sums_a.norm(dim=1) / counts,
        "concentration_v": sums_v.norm(dim=1) / counts,
        **history,
    }
    _atomic_save(artifact, path)
    return artifact


class PersistentPrototypeBank:
    """Fixed exact support, optional fold history, and current replay queries."""

    def __init__(self, artifact, query_ids, query_labels, query_audio, query_visual,
                 query_folds, device):
        self.artifact, self.policy = artifact, artifact["policy"]
        self.query_ids = list(query_ids)
        self.query_labels = query_labels.to(device).detach()
        self.query_audio, self.query_visual = query_audio.to(device).detach(), query_visual.to(device).detach()
        self.query_folds = query_folds.to(device).detach()
        self.counts = artifact["counts"].to(device).detach()
        self.query_counts = torch.bincount(self.query_labels, minlength=len(self.counts)).float()
        self.audio_sums = artifact["sums_a"].to(device).detach()
        self.visual_sums = artifact["sums_v"].to(device).detach()
        self.audio_prototypes = F.normalize(self.audio_sums, dim=1, eps=_EPS)
        self.visual_prototypes = F.normalize(self.visual_sums, dim=1, eps=_EPS)
        self.reliability_a_from_v = self.reliability_v_from_a = None
        self.trust_a_from_v = self.trust_v_from_a = None
        self._fold_sums, self._fold_prototypes = {}, {}
        for name, exact in (("audio", self.audio_sums), ("visual", self.visual_sums)):
            key = "a" if name == "audio" else "v"
            history = (artifact["history_mass_" + key].unsqueeze(-1)
                       * artifact["history_mean_" + key]).to(device)
            if not _is_history(self.policy):
                history = torch.zeros_like(history)
            fold_sums = exact.unsqueeze(0) + history
            if bool((fold_sums.norm(dim=-1) <= _EPS).any()):
                raise ValueError("A prototype has zero norm after history transport")
            self._fold_sums[name] = fold_sums.detach()
            self._fold_prototypes[name] = F.normalize(fold_sums, dim=-1, eps=_EPS).detach()

    def cross_modal_margin(self, query, labels, positive_teacher_features, temperature,
                           modality, query_indices):
        """Use one query fold for *all* candidates, then exact target exclusion."""
        if modality not in ("audio", "visual"):
            raise ValueError("modality names the prototype branch: audio or visual")
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("Margin temperature must be finite and positive")
        if query_indices is None:
            raise ValueError("Persistent bank margins require replay query indices")
        indices = torch.as_tensor(query_indices, device=self.query_labels.device, dtype=torch.long)
        labels = labels.to(self.query_labels.device).long()
        if indices.ndim != 1 or labels.ndim != 1 or len(indices) != len(labels) or len(query) != len(labels):
            raise ValueError("Query indices, labels, and features must describe the same batch")
        if bool((indices < 0).any()) or bool((indices >= len(self.query_ids)).any()):
            raise IndexError("Replay query index is outside the verified teacher bank")
        if not torch.equal(self.query_labels.index_select(0, indices), labels):
            raise ValueError("Replay indices and labels disagree with the frozen bank")
        old_features = self.query_audio if modality == "audio" else self.query_visual
        expected = old_features.index_select(0, indices)
        if positive_teacher_features.shape != expected.shape or not torch.allclose(
                positive_teacher_features.detach(), expected, atol=1e-5, rtol=1e-4):
            raise ValueError("Replay features/indices do not match the frozen teacher")
        if len(self.counts) < 2:
            raise ValueError("CMR needs at least two candidate classes")
        folds = self.query_folds.index_select(0, indices)
        result = query.new_empty(len(query))
        for fold in torch.unique(folds).tolist():
            mask = folds == fold
            local_labels, local_query = labels[mask], query[mask]
            sums, prototypes = self._fold_sums[modality][fold], self._fold_prototypes[modality][fold]
            loo = sums.index_select(0, local_labels) - positive_teacher_features[mask].detach()
            if bool((loo.norm(dim=1) <= _EPS).any()):
                raise ValueError("Leave-one-out prototype has zero norm")
            target = (local_query * F.normalize(loo, dim=1, eps=_EPS)).sum(1) / temperature
            scores = local_query @ prototypes.T / temperature
            negative = scores.masked_fill(F.one_hot(local_labels, len(self.counts)).bool(), float("-inf"))
            result[mask] = target - torch.logsumexp(negative, dim=1)
        return result

    @torch.no_grad()
    def compute_trust(self, temperature, shrinkage_beta):
        indices = torch.arange(len(self.query_ids), device=self.query_labels.device)
        margin_a = self.cross_modal_margin(self.query_audio, self.query_labels, self.query_visual,
                                          temperature, "visual", indices)
        margin_v = self.cross_modal_margin(self.query_visual, self.query_labels, self.query_audio,
                                          temperature, "audio", indices)
        reliability, trust = [], []
        chance = 1.0 / len(self.counts)
        beta = float(shrinkage_beta)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("Trust shrinkage beta must be finite and non-negative")
        for margin in (margin_a, margin_v):
            sums = torch.zeros_like(self.query_counts)
            sums.index_add_(0, self.query_labels, torch.sigmoid(margin))
            # Query count is deliberately independent from prototype support.
            rel = sums / self.query_counts
            value = ((rel - chance) / (1.0 - chance)).clamp(0.0, 1.0)
            if beta > 0:
                value = ((self.query_counts / (self.query_counts + beta)) * value
                         + (beta / (self.query_counts + beta)) * value.mean())
            reliability.append(rel.detach())
            trust.append(value.detach())
        self.reliability_a_from_v, self.reliability_v_from_a = reliability
        self.trust_a_from_v, self.trust_v_from_a = trust


def load_task_bank(args, step, old_model, checkpoint_path, exemplar_set, device, path):
    """Verify the previous best artifact and score only the current replay M."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    config, policy = _history_config(args), _policy(args)
    _check_config(config)
    if policy not in ("pre_shrink", "historical"):
        raise ValueError("Only B/C policies load persistent prototype artifacts")
    required = {
        "schema_version", "policy", "step", "num_classes", "dataset", "class_num_per_step",
        "history_config", "class_order", "checkpoint_sha256", "model_sha256",
        "parent_checkpoint_sha256", "support_ids", "support_labels", "support_folds",
        "counts", "sums_a", "sums_v", "retired_ids", "retired_labels", "retired_unique_counts",
        "history_mean_a", "history_mean_v", "history_mass_a", "history_mass_v",
        "anchor_counts", "shift_a", "shift_v", "residual_a", "residual_v", "confidence_a", "confidence_v",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Incomplete or unsupported persistent prototype artifact")
    expected_classes = step * int(args.class_num_per_step)
    expected = {
        "schema_version": SCHEMA_VERSION, "policy": policy, "step": step - 1,
        "num_classes": expected_classes, "dataset": str(args.dataset),
        "class_num_per_step": int(args.class_num_per_step), "history_config": config,
        "class_order": _class_order(exemplar_set),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "model_sha256": _model_sha256(old_model),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError("Prototype artifact mismatch: {}".format(key))
    parent_hash = artifact["parent_checkpoint_sha256"]
    if step == 1:
        if parent_hash is not None:
            raise ValueError("First-task prototype artifact cannot have a parent checkpoint")
    elif (not isinstance(parent_hash, str) or len(parent_hash) != 64
          or any(character not in "0123456789abcdef" for character in parent_hash)):
        raise ValueError("Continued prototype artifact is missing its checkpoint lineage")
    ids = _dataset_ids(exemplar_set)
    if len(set(ids)) != len(ids):
        raise ValueError("Current replay contains duplicate query IDs")
    label_list = _dataset_labels(exemplar_set, ids)
    support_ids, support_labels = artifact["support_ids"], artifact["support_labels"]
    if len(support_ids) != len(set(support_ids)) or len(support_ids) != len(support_labels):
        raise ValueError("Saved support ID manifest is invalid")
    if any(not isinstance(vid, str) or not vid for vid in support_ids):
        raise ValueError("Saved support contains an invalid sample ID")
    if any(not isinstance(label, int) or not 0 <= label < expected_classes for label in support_labels):
        raise ValueError("Saved support labels are outside the teacher class domain")
    support = dict(zip(support_ids, support_labels))
    if artifact["support_folds"] != [_fold(vid, config) for vid in support_ids]:
        raise ValueError("Saved fold assignments do not match this run")
    for vid, label in zip(ids, label_list):
        if vid not in support or support[vid] != label or not 0 <= label < expected_classes:
            raise ValueError("Replay query {!r} is absent/mislabeled in saved support".format(vid))
    counts = torch.bincount(torch.tensor(support_labels), minlength=expected_classes).float()
    if counts.shape != (expected_classes,) or not torch.equal(counts, artifact["counts"]):
        raise ValueError("Saved support counts disagree with its ID manifest")
    if bool((counts < 2).any()):
        raise ValueError("B/C require at least two exact support samples per class")
    query_labels = torch.tensor(label_list, dtype=torch.long)
    if bool((torch.bincount(query_labels, minlength=expected_classes) == 0).any()):
        raise ValueError("Every old class needs at least one replay query")
    retired_ids = artifact["retired_ids"]
    if len(retired_ids) != len(set(retired_ids)) or set(retired_ids).intersection(support_ids):
        raise ValueError("Retired IDs overlap exact support or are duplicated")
    if any(not isinstance(vid, str) or not vid for vid in retired_ids):
        raise ValueError("Saved retired ledger contains an invalid sample ID")
    retired_labels = artifact["retired_labels"]
    if len(retired_ids) != len(retired_labels) or any(
            not isinstance(label, int) or not 0 <= label < expected_classes for label in retired_labels):
        raise ValueError("Saved retired ID/label manifest is invalid")
    retired_counts = torch.bincount(torch.tensor(retired_labels, dtype=torch.long), minlength=expected_classes)
    if not torch.equal(retired_counts, artifact["retired_unique_counts"]):
        raise ValueError("Saved retired counts disagree with the unique ID ledger")
    if not isinstance(artifact["sums_a"], torch.Tensor) or artifact["sums_a"].ndim != 2:
        raise ValueError("Saved exact prototype sums have invalid shape/type")
    dimension = artifact["sums_a"].shape[1]
    if dimension < 1:
        raise ValueError("Saved prototype feature dimension must be positive")
    for modality in ("a", "v"):
        if artifact["sums_" + modality].shape != (expected_classes, dimension):
            raise ValueError("Saved exact prototype sums have invalid shape")
        if artifact["history_mean_" + modality].shape != (config["folds"], expected_classes, dimension):
            raise ValueError("Saved historical means have invalid shape")
        mass = artifact["history_mass_" + modality]
        if mass.shape != (config["folds"], expected_classes) or bool((mass < 0).any()):
            raise ValueError("Saved historical masses have invalid shape or sign")
        if bool((mass > config["mass_cap"] + 1e-5).any()):
            raise ValueError("Saved historical mass exceeds its configured cap")
    for key, value in artifact.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError("Non-finite tensor in prototype artifact: {}".format(key))
    audio, visual = _extract(old_model, exemplar_set, ids, label_list, args, device)
    bank = PersistentPrototypeBank(artifact, ids, query_labels, audio, visual,
                                   torch.tensor([_fold(vid, config) for vid in ids]), device)
    if args.rd_mode == "adaptive_crosssdc_cmr":
        bank.compute_trust(float(args.rd_margin_temperature), float(args.rd_trust_shrinkage_beta))
    return bank
