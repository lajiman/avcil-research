"""Exercise worker cleanup with real processes and retained exit callbacks.

These tests intentionally need neither PyTorch nor a GPU: the regression is
that retaining a stopped multiprocessing.Process can retain its pipe handles.
The small iterator stand-in supplies only DataLoader's shutdown boundary.
"""

import atexit
from contextlib import ExitStack
import errno
import multiprocessing
import os
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loader_lifecycle import close_data_loader


def _worker_main():
    pass


def _retained_worker_cleanup(worker):
    """Match the cleanup operations a retained PyTorch exit hook performs."""
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()


class _Iterator:
    def __init__(self, workers=(), shutdown_error=None):
        self._workers = list(workers)
        self.shutdown_calls = 0
        self.shutdown_error = shutdown_error

    def _shutdown_workers(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error
        for worker in self._workers:
            worker.join(timeout=5)


class _Loader:
    def __init__(self, iterator=None):
        self._iterator = iterator

    def __iter__(self):
        raise AssertionError("Cleanup must never start a loader")


class UnstartedLoaderTests(unittest.TestCase):
    def test_missing_loader_is_a_noop(self):
        close_data_loader(None)

    def test_cleanup_does_not_start_an_unstarted_loader(self):
        loader = _Loader()
        close_data_loader(loader)
        close_data_loader(loader)
        self.assertIsNone(loader._iterator)


@unittest.skipUnless(
    os.name == "posix" and "fork" in multiprocessing.get_all_start_methods(),
    "The retained POSIX process-descriptor regression requires fork",
)
class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.context = multiprocessing.get_context("fork")
        self.workers = []

    def tearDown(self):
        atexit.unregister(_retained_worker_cleanup)
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
            worker.close()

    def start_worker(self):
        worker = self.context.Process(target=_worker_main)
        worker.start()
        self.workers.append(worker)
        # This strong reference remains even after DataLoader drops its iterator.
        atexit.register(_retained_worker_cleanup, worker)
        return worker

    def assert_descriptor_closed(self, descriptor):
        with self.assertRaises(OSError) as raised:
            os.fstat(descriptor)
        self.assertEqual(raised.exception.errno, errno.EBADF)

    def test_stopped_worker_descriptors_close_despite_retained_callback(self):
        worker = self.start_worker()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        sentinel = worker.sentinel
        # Joining a process is insufficient while the Process remains referenced.
        os.fstat(sentinel)

        iterator = _Iterator([worker])
        loader = _Loader(iterator)
        close_data_loader(loader)

        self.assertIsNone(loader._iterator)
        self.assertEqual(iterator.shutdown_calls, 1)
        self.assert_descriptor_closed(sentinel)
        # Process.close() breaks this later callback; releasing only the stopped
        # Popen's handles must preserve the callback's join/is_alive operations.
        _retained_worker_cleanup(worker)

    def test_cleanup_is_idempotent(self):
        worker = self.start_worker()
        iterator = _Iterator([worker])
        loader = _Loader(iterator)
        sentinel = worker.sentinel

        close_data_loader(loader)
        close_data_loader(loader)

        self.assertEqual(iterator.shutdown_calls, 1)
        self.assertFalse(worker.is_alive())
        self.assert_descriptor_closed(sentinel)

    @unittest.skipUnless(os.path.isdir("/proc/self/fd"), "Requires Linux fd accounting")
    def test_repeated_stages_do_not_accumulate_process_descriptors(self):
        initial_count = len(os.listdir("/proc/self/fd"))
        for _ in range(12):
            worker = self.start_worker()
            loader = _Loader(_Iterator([worker]))
            close_data_loader(loader)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(os.listdir("/proc/self/fd")), initial_count)
        # All Process objects are still retained, including by their exit hooks.
        self.assertEqual(len(self.workers), 12)
        for worker in self.workers:
            _retained_worker_cleanup(worker)

    def test_exit_stack_cleans_other_loaders_after_shutdown_error(self):
        healthy_worker = self.start_worker()
        failing_worker = self.start_worker()
        # Model a shutdown exception after a worker has already exited.
        failing_worker.join(timeout=5)
        self.assertFalse(failing_worker.is_alive())
        healthy_sentinel = healthy_worker.sentinel
        failing_sentinel = failing_worker.sentinel
        healthy_iterator = _Iterator([healthy_worker])
        failing_iterator = _Iterator(
            [failing_worker], shutdown_error=RuntimeError("shutdown failed")
        )
        healthy_loader = _Loader(healthy_iterator)
        failing_loader = _Loader(failing_iterator)

        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            with ExitStack() as cleanup:
                cleanup.callback(close_data_loader, healthy_loader)
                cleanup.callback(close_data_loader, failing_loader)

        self.assertEqual(healthy_iterator.shutdown_calls, 1)
        self.assertEqual(failing_iterator.shutdown_calls, 1)
        self.assertIsNone(healthy_loader._iterator)
        # Preserve failed iterator state so callers may retry the shutdown.
        self.assertIs(failing_loader._iterator, failing_iterator)
        self.assert_descriptor_closed(healthy_sentinel)
        self.assert_descriptor_closed(failing_sentinel)
        failing_iterator.shutdown_error = None
        close_data_loader(failing_loader)
        self.assertIsNone(failing_loader._iterator)
        self.assertEqual(failing_iterator.shutdown_calls, 2)


if __name__ == "__main__":
    unittest.main()
