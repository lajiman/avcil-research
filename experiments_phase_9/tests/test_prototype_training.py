"""End-to-end bank lifecycle, legacy equivalence and inference isolation."""

import json

import h5py
import numpy as np
import pytest
import torch

from experiments_phase_9 import train_incremental_fusion_modular as trainer
from experiments_phase_9.class_fusion.checkpoints import load_model, load_prototype_bank
from experiments_phase_9.tests.test_training import make_fixture, read_csv


def training_args(features, meta, output, name, rule="sample_aware", mode="history_bank"):
    return ["--feature_root", str(features), "--meta_root", str(meta), "--output_root", str(output),
            "--experiment_name", name, "--num_classes", "6", "--class_num_per_step", "2",
            "--memory_size", "8", "--train_batch_size", "4", "--exemplar_batch_size", "4",
            "--infer_batch_size", "4", "--max_epoches", "3", "--fusion_warmup_epochs", "1",
            "--fusion_update_interval", "1", "--fusion_batch_size", "4", "--num_workers", "0",
            "--device", "cpu", "--fusion_prototype_mode", mode, "--fusion_update_rule", rule,
            "--no_record_cl_history"]


@pytest.mark.parametrize("rule,head", [("sample_aware", "linear"), ("direct", "mlp")])
def test_bank_training_preserves_birth_state_and_test_needs_no_bank(tmp_path, monkeypatch, rule, head):
    features, meta = make_fixture(tmp_path)
    output = tmp_path / "out"
    argv = training_args(features, meta, output, "bank", rule) + [
        "--fusion_classifier", head, "--fusion_hidden_dim", "16", "--record_cl_history"]
    trainer.main(argv)
    run = output / "bank"
    metrics = output / "metrics/bank"
    assert load_prototype_bank(run / "step_0_best_model.pt") is None
    banks = [load_prototype_bank(run / f"step_{s}_best_model.pt") for s in (1, 2)]
    assert banks[0]["birth_count"].tolist() == [4, 4]
    assert banks[1]["birth_count"].tolist() == [4, 4, 4, 4]
    assert banks[1]["birth_step"].tolist() == [0, 0, 1, 1]
    assert banks[1]["prepared_step"] == 2
    # Repeated periodic updates never add pseudo-observations to fixed birth statistics.
    torch.testing.assert_close(banks[0]["birth_sum_audio"], banks[1]["birth_sum_audio"][:2], rtol=0, atol=0)
    torch.testing.assert_close(banks[0]["birth_sum_visual"], banks[1]["birth_sum_visual"][:2], rtol=0, atol=0)
    first_anchors = dict(zip(banks[0]["anchors"]["sample_ids"], banks[0]["anchors"]["audio"]))
    assert len(banks[1]["anchors"]["sample_ids"]) == 8
    for vid, label, vector in zip(banks[1]["anchors"]["sample_ids"],
                                  banks[1]["anchors"]["labels"], banks[1]["anchors"]["audio"]):
        assert vid.startswith("train_") and int(label) < 4
        if int(label) < 2:
            torch.testing.assert_close(vector, first_anchors[vid], rtol=0, atol=0)
    assert torch.bincount(banks[1]["anchors"]["labels"]).tolist() == [2, 2, 2, 2]
    for step in (1, 2):
        last_bank = load_prototype_bank(run / f"step_{step}_last_model.pt")
        torch.testing.assert_close(last_bank["birth_count"], banks[step - 1]["birth_count"])
        assert last_bank["anchors"]["sample_ids"] == banks[step - 1]["anchors"]["sample_ids"]
        prior = load_model(run / f"step_{step - 1}_best_model.pt")
        initial = read_csv(metrics / f"class_fusion/step_{step}_initial_gates.csv")
        np.testing.assert_allclose([float(row["audio_gate"]) for row in initial[:step*2]], prior.fusion_gate.numpy())
        for epoch in (1, 2):
            gates = read_csv(metrics / f"class_fusion/step_{step}_after_epoch_{epoch}_gates.csv")
            for row in gates:
                if row["valid"] == "True" and rule == "direct":
                    assert float(row["eta"]) == 1.0
                    assert float(row["audio_gate"]) == float(row["raw_gate"])
            snapshot = torch.load(metrics / f"prototype_bank/step_{step}_after_epoch_{epoch}.pt", weights_only=True)
            assert snapshot["active_from_epoch"] == epoch + 1
            assert snapshot["prototype_audio"].shape == ((step + 1) * 2, 768)
        assert not (metrics / f"prototype_bank/step_{step}_after_epoch_3.pt").exists()
        payload = torch.load(run / f"step_{step}_best_model.pt", weights_only=True)
        assert payload["cl_history"]["observation"]["epoch"] == payload["epoch"]
    second = read_csv(metrics / "prototype_bank/step_2_after_epoch_1.csv")
    assert all(int(row["birth_count"]) == 4 for row in second[:4])
    assert all(float(row["prior_strength_effective"]) == 2 for row in second[:4])
    assert all(row["history_used"] == "True" for row in second[:4])
    assert all(row["history_used"] == "False" for row in second[4:])

    before = json.loads((metrics / "summary.json").read_text())
    # Inference depends on saved gates, not training-only bank or training metadata.
    for path in run.glob("step_*_best_model.pt"):
        payload = torch.load(path, weights_only=True)
        payload.pop("prototype_bank", None)
        torch.save(payload, path)
    for filename in ("all_id_category_dict.npy", "all_classId_vid_dict.npy"):
        mapping = np.load(meta / filename, allow_pickle=True).item()
        np.save(meta / filename, {"test": mapping["test"]})

    def no_statistics(*args, **kwargs):
        pytest.fail("test_only attempted to build training statistics")

    monkeypatch.setattr(trainer, "prepare_prototype_bank", no_statistics)
    monkeypatch.setattr(trainer, "build_bank_reliability", no_statistics)
    trainer.main(argv + ["--test_only"])
    assert json.loads(next(metrics.glob("test_only_*/summary.json")).read_text()) == before


def test_zero_strength_bank_preserves_legacy_training_bit_for_bit(tmp_path):
    features, meta = make_fixture(tmp_path)
    output = tmp_path / "out"
    trainer.main(training_args(features, meta, output, "fresh", mode="fresh"))
    trainer.main(training_args(features, meta, output, "zero") + ["--prototype_prior_strength", "0"])
    for step in range(3):
        for which in ("best", "last"):
            a = torch.load(output / f"fresh/step_{step}_{which}_model.pt", weights_only=True)
            b = torch.load(output / f"zero/step_{step}_{which}_model.pt", weights_only=True)
            assert a["epoch"] == b["epoch"] and a["val_acc"] == b["val_acc"]
            for key in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
    before = output / "metrics/fresh/class_fusion"
    after = output / "metrics/zero/class_fusion"
    for csv in before.glob("*.csv"):
        assert csv.read_bytes() == (after / csv.name).read_bytes()


def test_test_features_and_labels_cannot_change_historical_bank_or_training(tmp_path, monkeypatch):
    features, meta = make_fixture(tmp_path)
    output = tmp_path / "out"
    observed = []
    estimate = trainer.build_bank_reliability

    def inspect_reference(model, reference, bank, num_classes, *args, **kwargs):
        current, replay = reference.sources
        assert current.mode == replay.mode == "train"
        assert set(reference.sample_ids) == set(current.all_current_data_vids + replay.exemplar_vids_set)
        assert set(bank["anchors"]["sample_ids"]) == set(replay.exemplar_vids_set)
        assert all(vid.startswith("train_") for vid in reference.sample_ids)
        assert current.current_step_class == list(range(num_classes - 2, num_classes))
        observed.append(num_classes)
        return estimate(model, reference, bank, num_classes, *args, **kwargs)

    monkeypatch.setattr(trainer, "build_bank_reliability", inspect_reference)
    trainer.main(training_args(features, meta, output, "before", rule="direct"))
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
    trainer.main(training_args(features, meta, output, "after", rule="direct"))
    assert observed == [4, 4, 6, 6] * 2
    for step in range(3):
        for which in ("best", "last"):
            a = torch.load(output / f"before/step_{step}_{which}_model.pt", weights_only=True)
            b = torch.load(output / f"after/step_{step}_{which}_model.pt", weights_only=True)
            assert (a["epoch"], a["val_acc"]) == (b["epoch"], b["val_acc"])
            for key, value in a["state_dict"].items():
                torch.testing.assert_close(value, b["state_dict"][key], rtol=0, atol=0)
            if step:
                for key in ("birth_count", "birth_step", "birth_sum_audio", "birth_sum_visual"):
                    torch.testing.assert_close(a["prototype_bank"][key], b["prototype_bank"][key], rtol=0, atol=0)
                anchors_a, anchors_b = a["prototype_bank"]["anchors"], b["prototype_bank"]["anchors"]
                assert anchors_a["sample_ids"] == anchors_b["sample_ids"]
                for key in ("labels", "valid", "audio", "visual"):
                    torch.testing.assert_close(anchors_a[key], anchors_b[key], rtol=0, atol=0)
    for folder in ("class_fusion", "prototype_bank"):
        for before in (output / f"metrics/before/{folder}").glob("*.csv"):
            assert before.read_bytes() == (output / f"metrics/after/{folder}" / before.name).read_bytes()


@pytest.mark.parametrize("options", [["--prototype_prior_strength", "nan"],
                                     ["--prototype_prior_strength", "-1"],
                                     ["--fusion_prototype_mode", "history_bank", "--fusion_mode", "uniform"]])
def test_bank_arguments_are_validated(options):
    parser = trainer.build_parser()
    with pytest.raises(SystemExit):
        trainer.validate_args(parser, parser.parse_args(options))
