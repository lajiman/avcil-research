"""Real-loader checks for process-local audio sharing across dataset views."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader_ours import IcaAVELoader, exemplarLoader


class SharedAudioFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="shared_audio_")
        self.root = Path(self.tempdir.name)
        self.meta = self.root / "metadata"
        self.meta.mkdir()
        audio_directory = self.root / "audio_pretrained_feature"
        audio_directory.mkdir()
        self.audio_path = audio_directory / "audio_pretrained_feature_dict.npy"
        self.audio = {}
        self.visual = {}
        self.by_class = {}
        self.categories = {}
        for split_index, split in enumerate(("train", "val", "test")):
            self.by_class[split] = {}
            self.categories[split] = {}
            for label in range(3):
                ids = ["{}_{}_{}".format(split, label, index) for index in range(2)]
                self.by_class[split][str(label)] = ids
                for index, vid in enumerate(ids):
                    value = split_index * 100 + label * 10 + index
                    self.audio[vid] = np.full((4,), value, dtype=np.float32)
                    self.visual[vid] = np.full((2, 4), value + 0.5, dtype=np.float32)
                    self.categories[split][vid] = "class{}".format(label)
        np.save(self.audio_path, self.audio)
        np.save(self.meta / "all_classId_vid_dict.npy", self.by_class)
        np.save(self.meta / "all_id_category_dict.npy", self.categories)
        np.save(self.meta / "category_encode_dict.npy",
                {"class{}".format(label): label for label in range(3)})
        with h5py.File(self.root / "visual_features.h5", "w") as store:
            for vid, feature in self.visual.items():
                store.create_dataset(vid, data=feature)
        self.args = SimpleNamespace(
            dataset="VGGSound_shared_audio_fixture", feature_root=str(self.root),
            meta_root=str(self.meta), class_num_per_step=1, memory_size=2,
        )
        self.datasets = []

    def tearDown(self):
        for dataset in self.datasets:
            dataset.close_visual_features_h5()
        self.tempdir.cleanup()

    def _view(self, mode, **kwargs):
        if mode == "replay":
            dataset = exemplarLoader(self.args, modality="audio-visual", **kwargs)
        else:
            dataset = IcaAVELoader(self.args, mode=mode, modality="audio-visual", **kwargs)
        self.datasets.append(dataset)
        return dataset

    def _audio_load_count(self, loader_mock):
        return sum(
            Path(call.args[0]) == self.audio_path
            for call in loader_mock.call_args_list
        )

    def _shared_views(self):
        train = self._view("train")
        shared = train.all_audio_pretrained_features
        return [train] + [self._view(mode, audio_features=shared)
                          for mode in ("val", "test", "replay")]

    def test_four_views_load_audio_once_and_return_original_features(self):
        with patch("dataloader_ours.np.load", wraps=np.load) as load:
            train, val, test, replay = self._shared_views()
            self.assertEqual(self._audio_load_count(load), 1)
        shared = train.all_audio_pretrained_features
        before = {vid: value.copy() for vid, value in shared.items()}
        for dataset in (train, val, test):
            dataset.set_incremental_step(1)
        replay._set_incremental_step_(1)

        for dataset in (train, val, test, replay):
            self.assertIs(dataset.all_audio_pretrained_features, shared)
            ids = (dataset.exemplar_vids_set if dataset is replay
                   else dataset.all_current_data_vids)
            dataset.preload_visual_features()
            for index, vid in enumerate(ids):
                (visual, audio), label = dataset[index]
                self.assertTrue(torch.equal(audio, torch.from_numpy(self.audio[vid])))
                self.assertTrue(torch.equal(visual, torch.from_numpy(self.visual[vid])))
                self.assertEqual(label, int(vid.split("_")[1]))
            dataset.clear_visual_features_cache()
        self.assertEqual(set(shared), set(before))
        for vid in shared:
            np.testing.assert_array_equal(shared[vid], before[vid])

    def test_shared_features_do_not_merge_split_or_replay_membership(self):
        train, val, test, replay = self._shared_views()
        for dataset in (train, val, test):
            dataset.set_incremental_step(1)
        replay._set_incremental_step_(1)
        self.assertEqual(set(train.all_current_data_vids), set(self.by_class["train"]["1"]))
        self.assertEqual(set(replay.exemplar_vids_set), set(self.by_class["train"]["0"]))
        for split, dataset in (("val", val), ("test", test)):
            expected = self.by_class[split]["0"] + self.by_class[split]["1"]
            self.assertEqual(set(dataset.all_current_data_vids), set(expected))
            self.assertEqual({dataset[index][1] for index in range(len(dataset))}, {0, 1})
        self.assertEqual({train[index][1] for index in range(len(train))}, {1})
        self.assertEqual({replay[index][1] for index in range(len(replay))}, {0})
        self.assertTrue(set(train.all_current_data_vids).isdisjoint(val.all_current_data_vids))
        self.assertTrue(set(train.all_current_data_vids).isdisjoint(test.all_current_data_vids))

    def test_legacy_constructor_still_loads_its_own_dictionary(self):
        with patch("dataloader_ours.np.load", wraps=np.load) as load:
            views = [self._view(mode) for mode in ("train", "val", "test", "replay")]
            self.assertEqual(self._audio_load_count(load), 4)
        for left, right in zip(views, views[1:]):
            self.assertIsNot(left.all_audio_pretrained_features, right.all_audio_pretrained_features)
        (_, audio), label = views[0][0]
        self.assertEqual(label, 0)
        self.assertTrue(torch.equal(audio, torch.from_numpy(self.audio["train_0_0"])))

    def test_explicit_empty_store_does_not_reload_audio(self):
        shared = {}
        with patch("dataloader_ours.np.load", wraps=np.load) as load:
            views = [self._view(mode, audio_features=shared)
                     for mode in ("train", "val", "test", "replay")]
            self.assertEqual(self._audio_load_count(load), 0)
        for dataset in views:
            self.assertIs(dataset.all_audio_pretrained_features, shared)


if __name__ == "__main__":
    unittest.main()
