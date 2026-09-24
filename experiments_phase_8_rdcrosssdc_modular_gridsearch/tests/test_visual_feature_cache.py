import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader_ours import _LazyVisualH5Mixin


class TinyCachedDataset(_LazyVisualH5Mixin, Dataset):
    def __init__(self, path, vids):
        self.args = SimpleNamespace(dataset="VGGSound_test")
        self.modality = "visual"
        self.mode = "test"
        self.incremental_step = 0
        self.visual_pretrained_feature_path = str(path)
        self.all_visual_pretrained_features = None
        self._visual_h5_owner_pid = None
        self._visual_feature_cache = None
        self._visual_feature_cache_vids = None
        self._visual_feature_cache_nbytes = 0
        self.vids = list(vids)

    def _current_visual_feature_vids(self):
        return self.vids

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        return torch.as_tensor(
            self._visual_feature(self.vids[index]), dtype=torch.float32
        )


class VisualFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="visual_cache_test_")
        self.path = Path(self.tempdir.name) / "features.h5"
        with h5py.File(self.path, "w") as store:
            for index in range(6):
                store.create_dataset(
                    "vid{}".format(index),
                    data=np.full((2, 3), index, dtype=np.float32),
                )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_preload_uses_cache_without_an_open_h5_handle(self):
        dataset = TinyCachedDataset(self.path, ["vid0", "vid2", "vid4"])
        expected_bytes = 3 * 2 * 3 * np.dtype(np.float32).itemsize

        self.assertEqual(dataset.preload_visual_features(), expected_bytes)
        self.assertIsNone(dataset.all_visual_pretrained_features)
        self.assertIsNone(dataset._visual_h5_owner_pid)
        self.assertTrue(torch.equal(dataset[1], torch.full((2, 3), 2.0)))

        dataset.clear_visual_features_cache()
        self.assertIsNone(dataset._visual_feature_cache)
        self.assertEqual(dataset._visual_feature_cache_nbytes, 0)

    @unittest.skipUnless(
        os.name == "posix" and multiprocessing.get_start_method() == "fork",
        "The production cache-sharing path uses Linux fork workers",
    )
    def test_fork_workers_read_cache_after_source_file_is_unavailable(self):
        dataset = TinyCachedDataset(
            self.path, ["vid0", "vid1", "vid2", "vid3", "vid4", "vid5"]
        )
        dataset.preload_visual_features()
        unavailable_path = self.path.with_suffix(".offline")
        self.path.rename(unavailable_path)
        try:
            loader = DataLoader(dataset, batch_size=2, num_workers=2)
            batches = list(loader)
        finally:
            unavailable_path.rename(self.path)

        values = torch.cat(batches, dim=0)[:, 0, 0].tolist()
        self.assertEqual(values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
