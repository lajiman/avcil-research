import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from experiments_phase_9.class_fusion.checkpoints import load_model
from experiments_phase_9.class_fusion.fusion_method import build_reliability_bank, FusionReferenceDataset
from experiments_phase_9.dataloader_ours import IcaAVELoader, exemplarLoader
from experiments_phase_9.train_incremental_fusion_modular import build_parser, main


def make_fixture(root):
    features, meta = root / "features", root / "meta"
    (features / "audio_pretrained_feature").mkdir(parents=True)
    meta.mkdir()
    rng = np.random.default_rng(31)
    ids, classes, audio = {}, {}, {}
    with h5py.File(features / "visual_features.h5", "w") as h5:
        for split, size in (("train", 4), ("val", 1), ("test", 1)):
            ids[split], classes[split] = {}, {}
            for c in range(6):
                # Deliberately mix string and integer keys, as real metadata can.
                key = str(c) if c % 2 else c
                classes[split][key] = []
                for i in range(size):
                    vid = f"{split}_{c}_{i}"
                    ids[split][vid] = f"category_{c}"
                    classes[split][key].append(vid)
                    audio[vid] = rng.normal(size=768).astype(np.float32)
                    h5.create_dataset(vid, data=rng.normal(size=(16, 768)).astype(np.float32))
    np.save(features / "audio_pretrained_feature/audio_pretrained_feature_dict.npy", audio)
    np.save(meta / "all_id_category_dict.npy", ids)
    np.save(meta / "all_classId_vid_dict.npy", classes)
    np.save(meta / "category_encode_dict.npy", {f"category_{c}": c for c in range(6)})
    return features, meta


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_three_step_training_periodic_updates_shrinking_memory_and_test_only(tmp_path, head, monkeypatch):
    features, meta = make_fixture(tmp_path)
    audio_loads = []
    numpy_load = np.load

    def count_audio_loads(path, *args, **kwargs):
        if str(path).endswith("audio_pretrained_feature_dict.npy"):
            audio_loads.append(str(path))
        return numpy_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", count_audio_loads)
    output = tmp_path / "output"
    argv = ["--feature_root", str(features), "--meta_root", str(meta), "--output_root", str(output),
            "--experiment_name", "smoke", "--num_classes", "6", "--class_num_per_step", "2",
            "--memory_size", "8", "--train_batch_size", "4", "--exemplar_batch_size", "4", "--infer_batch_size", "4",
            "--max_epoches", "4", "--fusion_warmup_epochs", "1", "--fusion_update_interval", "2",
            "--fusion_batch_size", "4", "--num_workers", "0", "--device", "cpu",
            "--fusion_classifier", head, "--fusion_hidden_dim", "16"]
    main(argv)
    assert len(audio_loads) == 1  # All four datasets share one in-process dictionary.
    metrics = output / "metrics/smoke"
    history = read_csv(metrics / "class_fusion/epoch_summary.csv")
    assert len(history) == 12
    assert all(int(row["gate_version"]) == 0 for row in history if row["step"] == "0")
    for step in (1, 2):
        rows = [row for row in history if int(row["step"]) == step]
        assert int(rows[1]["gate_version"]) == int(rows[0]["gate_version"]) + 1
        assert int(rows[3]["gate_version"]) == int(rows[2]["gate_version"]) + 1
        assert not (metrics / f"class_fusion/step_{step}_after_epoch_4_gates.csv").exists()
        path = output / f"smoke/step_{step}_best_model.pt"
        payload = torch.load(path, weights_only=True)
        matching_row = rows[payload["epoch"] - 1]
        assert int(payload["state_dict"]["fusion_version"]) == int(matching_row["gate_version"])
        cl_state = payload["cl_history"]
        assert cl_state["observation"]["epoch"] == payload["epoch"]
        assert cl_state["observation"]["gate_version"] == int(matching_row["gate_version"])
        assert cl_state["reference"]["teacher_reliability"]["counts"].tolist() == [8 // (step * 2)] * (step * 2)
        assert cl_state["reference"]["candidate_class_ids"] == list(range(step * 2))
        assert all(label < step * 2 for label in cl_state["reference"]["labels"])
        assert cl_state["observation"]["reliability"]["counts"].tolist() == cl_state["reference"]["teacher_reliability"]["counts"].tolist()
        prior = load_model(output / f"smoke/step_{step-1}_best_model.pt")
        initial = read_csv(metrics / f"class_fusion/step_{step}_initial_gates.csv")
        np.testing.assert_allclose([float(row["audio_gate"]) for row in initial[:step*2]], prior.fusion_gate.numpy())
        assert all(float(row["audio_gate"]) == 0.5 for row in initial[step*2:])
    early = read_csv(metrics / "class_fusion/step_1_after_epoch_1_gates.csv")
    late = read_csv(metrics / "class_fusion/step_2_after_epoch_1_gates.csv")
    assert int(early[0]["counts"]) == 4 and int(late[0]["counts"]) == 2
    assert float(late[0]["eta"]) < float(early[0]["eta"])
    with (metrics / "summary.json").open() as handle:
        training_summary = json.load(handle)
    old_history = (metrics / "class_fusion/epoch_summary.csv").read_bytes()
    # Evaluation must not instantiate training datasets, rebuild replay, or
    # consult validation labels. Remove those splits to enforce the boundary.
    for filename in ("all_id_category_dict.npy", "all_classId_vid_dict.npy"):
        data = np.load(meta / filename, allow_pickle=True).item()
        np.save(meta / filename, {"test": data["test"]})
    main(argv + ["--test_only"])
    assert len(audio_loads) == 2  # A new invocation reads fresh data, with no global cache.
    evaluation_summary_path = next(metrics.glob("test_only_*/summary.json"))
    with evaluation_summary_path.open() as handle:
        assert json.load(handle) == training_summary
    assert (metrics / "class_fusion/epoch_summary.csv").read_bytes() == old_history


def test_hdf5_loader_and_statistics_with_spawn_worker(tmp_path):
    features, meta = make_fixture(tmp_path)
    args = build_parser().parse_args(["--feature_root", str(features), "--meta_root", str(meta), "--class_num_per_step", "2"])
    dataset = IcaAVELoader(args)
    try:
        # Open in the parent first: worker pickle must omit the open HDF5 file.
        first = dataset[0]
        loader = DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn")
        data, labels = next(iter(loader))
        torch.testing.assert_close(data[0][0], first[0][0])
        assert int(labels[0]) == first[1]
    finally:
        dataset.close_visual_features_h5()


def test_shared_audio_dictionary_keeps_dataset_splits_and_reference_values(tmp_path):
    features, meta = make_fixture(tmp_path)
    args = build_parser().parse_args(["--feature_root", str(features), "--meta_root", str(meta), "--class_num_per_step", "2"])
    train = IcaAVELoader(args)
    shared = train.all_audio_pretrained_features
    snapshot = {key: value.copy() for key, value in shared.items()}
    val = IcaAVELoader(args, "val", shared_audio_features=shared)
    test = IcaAVELoader(args, "test", shared_audio_features=shared)
    replay = exemplarLoader(args, shared_audio_features=shared)
    try:
        replay._set_incremental_step_(1)
        for dataset in (train, val, test, replay):
            assert dataset.all_audio_pretrained_features is shared
            vids = getattr(dataset, "all_current_data_vids", replay.exemplar_vids_set)
            for index, vid in enumerate(vids):
                assert vid.startswith(dataset.mode + "_")
                (_, audio), label = dataset[index]
                torch.testing.assert_close(audio, torch.from_numpy(snapshot[vid]), rtol=0, atol=0)
                assert label == int(vid.split("_")[1])
        for key, value in snapshot.items():
            np.testing.assert_array_equal(shared[key], value)
    finally:
        for dataset in (train, val, test, replay):
            dataset.close_visual_features_h5()


def test_test_data_cannot_change_training_or_gate_state(tmp_path, monkeypatch):
    """Perturb held-out features/labels; trained checkpoints must stay identical."""
    from experiments_phase_9 import train_incremental_fusion_modular as trainer

    features, meta = make_fixture(tmp_path)
    output = tmp_path / "output"
    observed_steps = []
    original_bank = trainer.build_reliability_bank

    def inspect_reference(model, reference, num_classes, *args, **kwargs):
        current, replay = reference.sources
        assert current.mode == replay.mode == "train"
        current_classes = set(current.current_step_class)
        assert current_classes == set(range(num_classes - 2, num_classes))
        assert set(reference.sample_ids) == set(current.all_current_data_vids + replay.exemplar_vids_set)
        assert len(reference.sample_ids) == len(set(reference.sample_ids))
        for source, index in reference.indices:
            dataset = reference.sources[source]
            vids = current.all_current_data_vids if source == 0 else replay.exemplar_vids_set
            vid = vids[index]
            assert vid.startswith("train_")
            label = dataset.category_encode_dict[dataset.all_id_category_dict[vid]]
            assert label in (current_classes if source == 0 else set(range(num_classes - 2)))
        observed_steps.append(num_classes)
        return original_bank(model, reference, num_classes, *args, **kwargs)

    monkeypatch.setattr(trainer, "build_reliability_bank", inspect_reference)
    argv = ["--feature_root", str(features), "--meta_root", str(meta), "--output_root", str(output),
            "--num_classes", "6", "--class_num_per_step", "2", "--memory_size", "8",
            "--train_batch_size", "4", "--exemplar_batch_size", "4", "--infer_batch_size", "4",
            "--max_epoches", "2", "--fusion_warmup_epochs", "1", "--fusion_update_interval", "1",
            "--fusion_batch_size", "4", "--num_workers", "0", "--device", "cpu", "--no_record_cl_history"]
    main(argv + ["--experiment_name", "before"])

    # Keep the global class encoding unchanged, but permute test labels within
    # each task and replace every test feature. Train/val bytes stay untouched.
    ids = np.load(meta / "all_id_category_dict.npy", allow_pickle=True).item()
    classes = np.load(meta / "all_classId_vid_dict.npy", allow_pickle=True).item()
    audio = np.load(features / "audio_pretrained_feature/audio_pretrained_feature_dict.npy", allow_pickle=True).item()
    classes["test"] = {str(c): [] for c in range(6)}
    with h5py.File(features / "visual_features.h5", "r+") as h5:
        for vid, category in ids["test"].items():
            new_class = int(category.rsplit("_", 1)[1]) ^ 1
            ids["test"][vid] = f"category_{new_class}"
            classes["test"][str(new_class)].append(vid)
            audio[vid] = -audio[vid] * 7
            h5[vid][...] = -h5[vid][...] * 3
    np.save(meta / "all_id_category_dict.npy", ids)
    np.save(meta / "all_classId_vid_dict.npy", classes)
    np.save(features / "audio_pretrained_feature/audio_pretrained_feature_dict.npy", audio)
    main(argv + ["--experiment_name", "after"])

    assert observed_steps == [4, 6, 4, 6]
    for step in range(3):
        for which in ("best", "last"):
            filename = f"step_{step}_{which}_model.pt"
            before = torch.load(output / "before" / filename, weights_only=True)
            after = torch.load(output / "after" / filename, weights_only=True)
            assert before["epoch"] == after["epoch"]
            assert before["val_acc"] == after["val_acc"]
            for key, value in before["state_dict"].items():
                torch.testing.assert_close(value, after["state_dict"][key], rtol=0, atol=0)
    for before in (output / "metrics/before/class_fusion").glob("*.csv"):
        after = output / "metrics/after/class_fusion" / before.name
        assert before.read_bytes() == after.read_bytes()
