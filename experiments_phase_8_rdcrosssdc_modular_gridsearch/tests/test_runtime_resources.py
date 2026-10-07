"""Resource controls cannot silently use all CPUs or multiple visible GPUs."""
import csv
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_resources as resources


class RuntimeResourceTests(unittest.TestCase):
    def test_explicit_thread_budget_overrides_legacy_eight(self):
        with mock.patch.dict(os.environ, {"AVCIL_TORCH_THREADS": "4"}), \
                mock.patch.object(torch, "set_num_threads") as intra, \
                mock.patch.object(torch, "set_num_interop_threads") as inter:
            resources.configure_torch_threads()
        intra.assert_called_once_with(4)
        inter.assert_called_once_with(1)

    def test_default_thread_budget_and_no_cuda_limit_preserve_legacy(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(torch, "set_num_threads") as intra, \
                mock.patch.object(torch, "set_num_interop_threads"), \
                mock.patch.object(torch.cuda, "set_per_process_memory_fraction") as limit:
            resources.configure_torch_threads()
            resources.configure_cuda_budget(torch.device("cpu"))
        intra.assert_called_once_with(8)
        limit.assert_not_called()

    def test_gpu_budget_applies_to_one_visible_gpu_only(self):
        with mock.patch.dict(os.environ, {"AVCIL_CUDA_MEMORY_FRACTION": "0.3"}), \
                mock.patch.object(torch.cuda, "device_count", return_value=1), \
                mock.patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=100 * 2**30)), \
                mock.patch.object(torch.cuda, "set_per_process_memory_fraction") as limit:
            resources.configure_cuda_budget(torch.device("cuda:0"))
            limit.assert_called_once_with(0.3, torch.device("cuda:0"))
        with mock.patch.dict(os.environ, {"AVCIL_CUDA_MEMORY_FRACTION": "0.3"}), \
                mock.patch.object(torch.cuda, "device_count", return_value=2), \
                self.assertRaisesRegex(ValueError, "one visible GPU"):
            resources.configure_cuda_budget(torch.device("cuda:0"))

    def test_invalid_limits_fail_instead_of_silently_removing_budget(self):
        for text in ("0", "-1", "1.2", "nan", "inf"):
            with self.subTest(text=text), \
                    mock.patch.dict(os.environ, {"AVCIL_CUDA_MEMORY_FRACTION": text}), \
                    self.assertRaises(ValueError):
                resources.configure_cuda_budget(torch.device("cuda:0"))

    def test_recording_is_opt_in_and_keeps_phase_peaks_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "resources.csv")
            with mock.patch.dict(os.environ, {}, clear=True):
                resources.record_resource_phase(path, 1, "train", 0, torch.device("cpu"))
                self.assertFalse(path.exists())
            with mock.patch.dict(os.environ, {"AVCIL_RECORD_RESOURCES": "1"}), \
                    mock.patch.object(torch.cuda, "max_memory_allocated", side_effect=[2**30, 2**29]), \
                    mock.patch.object(torch.cuda, "max_memory_reserved", side_effect=[2 * 2**30, 2**30]), \
                    mock.patch.object(torch.cuda, "reset_peak_memory_stats") as reset:
                for phase in ("train_and_bank", "test"):
                    start = resources.start_resource_phase(torch.device("cuda:0"))
                    resources.record_resource_phase(path, 1, phase, start, torch.device("cuda:0"))
                self.assertEqual(reset.call_count, 2)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["phase"] for r in rows], ["train_and_bank", "test"])
            self.assertEqual([float(r["cuda_peak_allocated_gib"]) for r in rows], [1, 0.5])


if __name__ == "__main__":
    unittest.main()
