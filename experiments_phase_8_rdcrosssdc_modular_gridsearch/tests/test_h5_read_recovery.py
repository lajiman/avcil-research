"""Fault injection around real HDF5 files; no GPU or production features needed."""

from collections import defaultdict, deque
import errno
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader_ours import _LazyVisualH5Mixin


REAL_H5_FILE = h5py.File


class SmallVisualStore(_LazyVisualH5Mixin):
    def __init__(self, path, **retry_options):
        self.args = SimpleNamespace(dataset="VGGSound_test", **retry_options)
        self.modality = "visual"
        self.visual_pretrained_feature_path = str(path)
        self.all_visual_pretrained_features = None
        self._visual_h5_owner_pid = None
        self._visual_feature_cache = None
        self._visual_feature_cache_vids = None
        self._visual_feature_cache_nbytes = 0

    def _current_visual_feature_vids(self):
        return ["vid0", "vid1", "vid2"]


class FaultInjector:
    """Keep real file/dataset behavior, injecting only requested operation faults."""

    def __init__(self):
        self.failures = defaultdict(deque)
        self.events = []
        self.handles = []

    def inject(self, operation, vid, *errors):
        self.failures[operation, vid].extend(errors)

    def check(self, operation, vid=None):
        self.events.append((operation, vid))
        pending = self.failures[operation, vid]
        if pending:
            raise pending.popleft()

    def open(self, path, mode):
        self.check("open")
        handle = FileProxy(REAL_H5_FILE(path, mode), self)
        self.handles.append(handle)
        return handle


class FileProxy:
    def __init__(self, handle, faults):
        self.handle = handle
        self.faults = faults

    def __getitem__(self, vid):
        self.faults.check("lookup", vid)
        return DatasetProxy(self.handle[vid], vid, self.faults)

    def __contains__(self, vid):
        self.faults.check("contains", vid)
        return vid in self.handle

    def close(self):
        self.handle.close()
        self.faults.check("close")


class DatasetProxy:
    def __init__(self, dataset, vid, faults):
        self.dataset = dataset
        self.vid = vid
        self.faults = faults

    def __getitem__(self, selection):
        self.faults.check("read", self.vid)
        return self.dataset[selection]


class H5ReadRecoveryTests(unittest.TestCase):
    def setUp(self):
        tempdir = tempfile.TemporaryDirectory(prefix="h5_read_recovery_")
        self.addCleanup(tempdir.cleanup)
        self.path = Path(tempdir.name) / "features.h5"
        with REAL_H5_FILE(self.path, "w") as handle:
            for index in range(3):
                handle.create_dataset(
                    "vid{}".format(index),
                    data=np.full((2, 3), index, dtype=np.float32),
                )
        self.dataset = SmallVisualStore(self.path)
        self.addCleanup(self.dataset.close_visual_features_h5)
        self.faults = FaultInjector()
        sleep_patch = mock.patch("dataloader_ours.time.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        open_patch = mock.patch("dataloader_ours.h5py.File", side_effect=self.faults.open)
        self.open_file = open_patch.start()
        self.addCleanup(open_patch.stop)

    def assert_vid1(self, value):
        np.testing.assert_array_equal(value, np.ones((2, 3), dtype=np.float32))

    def test_transient_open_failure_reopens_the_same_file(self):
        self.faults.inject("open", None, OSError(errno.ENXIO, "No such device"))

        self.assert_vid1(self.dataset._visual_feature("vid1"))

        self.assertEqual(self.open_file.call_args_list, [
            mock.call(str(self.path), "r"), mock.call(str(self.path), "r")
        ])
        self.sleep.assert_called_once_with(2)

    def test_transient_lookup_read_and_contains_failures_reopen(self):
        cases = [
            ("lookup", KeyError("Unable to open object (errno = 6)")),
            ("read", OSError("Can't synchronously read data (errno = 6)")),
            ("contains", RuntimeError("Unable to check link (errno=5)")),
        ]
        for operation, error in cases:
            with self.subTest(operation=operation):
                self.dataset.close_visual_features_h5()
                previous_opens = self.open_file.call_count
                self.sleep.reset_mock()
                self.faults.inject(operation, "vid1", error)

                if operation == "contains":
                    self.assertTrue(self.dataset._has_visual_feature("vid1"))
                else:
                    self.assert_vid1(self.dataset._visual_feature("vid1"))

                self.assertEqual(self.open_file.call_count - previous_opens, 2)
                self.assertFalse(self.faults.handles[-2].handle.id.valid)
                self.sleep.assert_called_once_with(2)

    def test_supported_transient_errnos_retry(self):
        for code in (
            errno.ENXIO, errno.EIO, errno.ESTALE,
            errno.ETIMEDOUT, errno.EINTR, errno.EAGAIN,
        ):
            with self.subTest(errno=code):
                self.sleep.reset_mock()
                self.faults.inject("read", "vid1", OSError(code, "storage fault"))
                self.assert_vid1(self.dataset._visual_feature("vid1"))
                self.sleep.assert_called_once_with(2)

    def test_retry_exhaustion_is_bounded_and_reports_file_and_sample(self):
        errors = [OSError(errno.ENXIO, "No such device") for _ in range(6)]
        self.faults.inject("read", "vid1", *errors)

        with self.assertRaises(OSError) as raised:
            self.dataset._visual_feature("vid1")

        self.assertIn(str(self.path), str(raised.exception))
        self.assertIn("vid1", str(raised.exception))
        self.assertIs(raised.exception.__cause__, errors[-1])
        self.assertEqual(self.open_file.call_count, 6)
        self.assertEqual(self.sleep.call_args_list, [
            mock.call(2), mock.call(4), mock.call(8),
            mock.call(16), mock.call(30),
        ])
        self.assertIsNone(self.dataset.all_visual_pretrained_features)
        self.assertIsNone(self.dataset._visual_h5_owner_pid)
        self.assertTrue(all(not item.handle.id.valid for item in self.faults.handles))

    def test_retry_count_and_delay_are_configurable(self):
        self.dataset.args.h5_read_retries = 2
        self.dataset.args.h5_retry_delay = 0.25
        self.faults.inject("read", "vid1", *[
            OSError(errno.EIO, "I/O error") for _ in range(2)
        ])

        self.assert_vid1(self.dataset._visual_feature("vid1"))

        self.assertEqual(self.open_file.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [mock.call(0.25), mock.call(0.5)])

    def test_zero_retries_fails_after_one_attempt(self):
        self.dataset.args.h5_read_retries = 0
        self.faults.inject("read", "vid1", OSError(errno.EIO, "I/O error"))
        with self.assertRaises(OSError):
            self.dataset._visual_feature("vid1")
        self.assertEqual(self.open_file.call_count, 1)
        self.sleep.assert_not_called()

    def test_missing_file_and_invalid_hdf5_fail_without_retry(self):
        for invalid_file in ("missing.h5", "corrupt.h5"):
            with self.subTest(filename=invalid_file):
                bad_path = self.path.with_name(invalid_file)
                if invalid_file == "corrupt.h5":
                    bad_path.write_text("This is not an HDF5 file.")
                self.dataset.visual_pretrained_feature_path = str(bad_path)
                previous_opens = self.open_file.call_count
                with self.assertRaises(OSError):
                    self.dataset._visual_feature("vid1")
                self.assertEqual(self.open_file.call_count - previous_opens, 1)
                self.sleep.assert_not_called()

    def test_permission_failure_is_not_retried(self):
        self.faults.inject("open", None, PermissionError(errno.EACCES, "Permission denied"))
        with self.assertRaises(OSError):
            self.dataset._visual_feature("vid1")
        self.assertEqual(self.open_file.call_count, 1)
        self.sleep.assert_not_called()

    def test_missing_dataset_is_not_retried_or_replaced(self):
        with self.assertRaises(KeyError):
            self.dataset._visual_feature("missing_vid")
        self.assertFalse(self.dataset._has_visual_feature("missing_vid"))
        self.sleep.assert_not_called()
        self.assertNotIn(("read", "vid0"), self.faults.events)

    def test_preload_retries_only_failed_sample_and_preserves_prior_reads(self):
        self.faults.inject("read", "vid1", OSError(errno.ENXIO, "No such device"))

        self.assertEqual(self.dataset.preload_visual_features(), 3 * 2 * 3 * 4)

        reads = [vid for operation, vid in self.faults.events if operation == "read"]
        self.assertEqual(reads, ["vid0", "vid1", "vid1", "vid2"])
        self.assert_vid1(self.dataset._visual_feature("vid1"))
        self.assertIsNone(self.dataset.all_visual_pretrained_features)
        self.assertTrue(all(not item.handle.id.valid for item in self.faults.handles))

    def test_failed_preload_does_not_publish_partial_cache(self):
        self.dataset.args.h5_read_retries = 1
        self.faults.inject("read", "vid1", *[
            OSError(errno.ENXIO, "No such device") for _ in range(2)
        ])

        with self.assertRaises(OSError):
            self.dataset.preload_visual_features()

        reads = [vid for operation, vid in self.faults.events if operation == "read"]
        self.assertEqual(reads, ["vid0", "vid1", "vid1"])
        self.assertIsNone(self.dataset._visual_feature_cache)
        self.assertEqual(self.dataset._visual_feature_cache_nbytes, 0)
        self.assertIsNone(self.dataset.all_visual_pretrained_features)

    def test_failed_close_does_not_leave_a_stale_handle(self):
        self.dataset._visual_feature("vid1")
        self.faults.inject("close", None, OSError(errno.EIO, "close failed"))

        with self.assertRaises(OSError):
            self.dataset.close_visual_features_h5()

        self.assertIsNone(self.dataset.all_visual_pretrained_features)
        self.assertIsNone(self.dataset._visual_h5_owner_pid)
        self.assert_vid1(self.dataset._visual_feature("vid1"))
        self.assertEqual(self.open_file.call_count, 2)

    def test_failed_close_during_recovery_does_not_prevent_reopen(self):
        self.faults.inject("read", "vid1", OSError(errno.ENXIO, "No such device"))
        self.faults.inject("close", None, OSError(errno.EIO, "close failed"))

        self.assert_vid1(self.dataset._visual_feature("vid1"))

        self.assertEqual(self.open_file.call_count, 2)
        self.assertFalse(self.faults.handles[0].handle.id.valid)

    def test_cached_features_do_not_open_hdf5(self):
        cached = np.ones((2, 3), dtype=np.float32)
        self.dataset._visual_feature_cache = {"vid1": cached}
        self.assertIs(self.dataset._visual_feature("vid1"), cached)
        self.assertTrue(self.dataset._has_visual_feature("vid1"))
        self.assertFalse(self.dataset._has_visual_feature("missing_vid"))
        self.open_file.assert_not_called()
        self.sleep.assert_not_called()

    def test_ave_dictionary_does_not_open_hdf5(self):
        self.dataset.args.dataset = "AVE"
        feature = np.ones((2, 3), dtype=np.float32)
        self.dataset.all_visual_pretrained_features = {"vid1": feature}
        self.assertIs(self.dataset._visual_feature("vid1"), feature)
        self.assertTrue(self.dataset._has_visual_feature("vid1"))
        self.assertFalse(self.dataset._has_visual_feature("missing_vid"))
        self.assertEqual(self.dataset.preload_visual_features(), 0)
        self.open_file.assert_not_called()
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
