"""Exercise persistent HDF5 workers and their explicit cleanup on tiny data.

Run with the training environment's Python:
    python experiments_phase_8_rdcrosssdc_modular_gridsearch/tests/check_loader_resources.py

The default checks real CPU tensor loading and, separately, PyTorch's real
pin-memory thread/atexit lifecycle using string batches on a simulated CPU
pinning device. The latter does not test CUDA transfers or pinned allocation.
Add --cuda to also exercise actual pinned tensor batches when CUDA is available.
No training data or model is loaded. Linux /proc and fork support are required.
"""

import argparse
import atexit
import gc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader_ours import _LazyVisualH5Mixin  # noqa: E402
from loader_lifecycle import close_data_loader  # noqa: E402


SAMPLES = 8
STAGES = 3
EPOCHS = 3
WORKERS = 2


class TinyH5Dataset(_LazyVisualH5Mixin, Dataset):
    """Keep the production HDF5 opening logic, returning observable metadata."""

    def __init__(self, path, strings=False):
        self.args = SimpleNamespace(dataset="SYNTHETIC")
        self.visual_pretrained_feature_path = str(path)
        self.all_visual_pretrained_features = None
        self._visual_h5_owner_pid = None
        self.stage = 0
        self.opens = 0
        self.strings = strings
        self.fail_at = None

    def __len__(self):
        return SAMPLES

    def __getitem__(self, index):
        if index == self.fail_at:
            raise RuntimeError("intentional synthetic worker failure")
        previous = self.all_visual_pretrained_features
        store = self._visual_feature_store()
        if store is not previous:
            self.opens += 1
        record = [
            os.getpid(),
            self.opens,
            self.stage,
            int(store["values"][self.stage * SAMPLES + index]),
            store.id.id,
        ]
        if self.strings:
            return json.dumps(record)
        return torch.tensor(record, dtype=torch.int64)


def fd_count():
    gc.collect()
    return len(os.listdir("/proc/self/fd"))


def assert_fd_baseline(baseline, label):
    # Queue feeder threads may need a moment to finish closing their pipes.
    deadline = time.monotonic() + 3
    current = fd_count()
    while current > baseline and time.monotonic() < deadline:
        time.sleep(0.02)
        current = fd_count()
    assert current <= baseline, (
        f"{label}: parent descriptors grew from {baseline} to {current}"
    )
    return current


def assert_stopped(workers, pin_thread, original_children):
    for worker in workers:
        if not getattr(worker, "_closed", False):
            assert not worker.is_alive(), f"worker {worker.pid} still running"
            assert worker.exitcode is not None, f"worker {worker.pid} not reaped"
    if pin_thread is not None:
        assert not pin_thread.is_alive(), "pin-memory thread still running"
    remaining = {child.pid for child in multiprocessing.active_children()}
    assert remaining <= original_children, f"unexpected child processes: {remaining}"


def run_loader(dataset, mode, epochs=EPOCHS, outcome="complete"):
    """Close even after an early break or an exception, keeping handles to check."""
    kwargs = dict(
        batch_size=2,
        num_workers=WORKERS,
        persistent_workers=True,
        prefetch_factor=1,
        multiprocessing_context="fork",
        timeout=15,
        pin_memory=mode != "cpu",
    )
    if mode == "simulated_pin_thread":
        kwargs["pin_memory_device"] = "cpu"
    loader = DataLoader(dataset, **kwargs)
    original_children = {child.pid for child in multiprocessing.active_children()}
    workers = []
    pin_thread = None
    epoch_workers = []
    saw_failure = False
    # Only current_device is simulated. PyTorch starts and shuts down its actual
    # pinning thread, queues, workers, and persistent-worker atexit callbacks.
    device_context = (
        mock.patch.object(torch.cuda, "current_device", return_value=0)
        if mode == "simulated_pin_thread"
        else mock.patch.object(torch.cuda, "current_device", wraps=torch.cuda.current_device)
    )
    try:
        with device_context, mock.patch.object(
            atexit, "register", wraps=atexit.register
        ) as register:
            for _ in range(epochs):
                iterator = iter(loader)
                workers = list(iterator._workers)
                pin_thread = getattr(iterator, "_pin_memory_thread", None)
                assert (pin_thread is not None) == (mode != "cpu")
                observations = {}
                values = []
                for batch in iterator:
                    if mode == "cuda":
                        assert batch.is_pinned(), "CUDA batch did not use pinned memory"
                    records = (
                        [json.loads(item) for item in batch]
                        if dataset.strings
                        else batch.tolist()
                    )
                    for pid, opens, stage, value, h5_id in records:
                        assert opens == 1, f"worker {pid} reopened its HDF5 file"
                        assert stage == dataset.stage, "worker has stale stage state"
                        previous = observations.setdefault(pid, (opens, h5_id))
                        assert previous == (opens, h5_id), "HDF5 handle changed"
                        values.append(value)
                    if outcome == "early_exit":
                        break
                if outcome == "early_exit":
                    break
                assert sorted(values) == list(
                    range(dataset.stage * SAMPLES, (dataset.stage + 1) * SAMPLES)
                ), "worker read data from the wrong stage"
                assert len(observations) == WORKERS
                epoch_workers.append(observations)
                assert observations == epoch_workers[0], (
                    "worker PIDs or HDF5 handles changed between epochs"
                )
            if mode != "cpu":
                cleanups = [
                    call for call in register.call_args_list
                    if getattr(call.args[0], "__name__", "") == "_clean_up_worker"
                ]
                assert len(cleanups) == WORKERS, (
                    "the test did not exercise PyTorch's persistent pinning atexit path"
                )
    except RuntimeError as error:
        if outcome != "worker_error" or "intentional synthetic worker failure" not in str(error):
            raise
        saw_failure = True
    finally:
        # Retain the original iterator/Process/thread objects while confirming
        # explicit shutdown; descriptor checks happen after local refs are gone.
        close_data_loader(loader)
        close_data_loader(loader)
        assert_stopped(workers, pin_thread, original_children)
    assert saw_failure == (outcome == "worker_error")
    return [sorted(observations) for observations in epoch_workers]


def run_mode(path, mode):
    dataset = TinyH5Dataset(path, strings=mode == "simulated_pin_thread")
    initial = fd_count()
    # Warm up one-time PyTorch multiprocessing/IPC state before comparing stages.
    run_loader(dataset, mode, epochs=1)
    baseline = fd_count()
    stages = []
    for stage in range(STAGES):
        dataset.stage = stage
        workers = run_loader(dataset, mode)
        stages.append({"stage": stage, "worker_pids_by_epoch": workers})
        assert_fd_baseline(baseline, f"{mode}, stage {stage}")

    run_loader(dataset, mode, epochs=1, outcome="early_exit")
    assert_fd_baseline(baseline, f"{mode}, early exit")
    dataset.fail_at = 3
    try:
        run_loader(dataset, mode, epochs=1, outcome="worker_error")
    finally:
        dataset.fail_at = None
        dataset.close_visual_features_h5()
    final = assert_fd_baseline(baseline, f"{mode}, worker error")
    return {
        "mode": mode,
        "initial_fds": initial,
        "warmed_baseline_fds": baseline,
        "final_fds": final,
        "stages": stages,
        "early_exit_cleanup": "passed",
        "worker_error_cleanup": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also check actual pinned tensors")
    args = parser.parse_args()
    if not Path("/proc/self/fd").is_dir() or "fork" not in multiprocessing.get_all_start_methods():
        parser.error("this regression check requires Linux /proc and multiprocessing fork")
    # Do this before any worker forks. Changing the thread pool for the first
    # time in a pinning thread after fork can itself hang on some installations.
    torch.set_num_threads(1)
    close_data_loader(None)
    unused = DataLoader(TinyH5Dataset("never-opened.h5"), num_workers=WORKERS)
    close_data_loader(unused)
    close_data_loader(unused)
    modes = ["cpu", "simulated_pin_thread"]
    if args.cuda:
        if torch.cuda.is_available():
            modes.append("cuda")
        else:
            print("SKIP actual CUDA pinning: CUDA is unavailable.", flush=True)
    print(
        f"torch={torch.__version__}; num_workers={WORKERS}; "
        f"stages={STAGES}; epochs={EPOCHS}; persistent_workers=True",
        flush=True,
    )
    print(
        "The simulated pin-thread case uses strings and pin_memory_device='cpu'; "
        "it tests real thread/atexit cleanup, not CUDA allocation or transfer.",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="avcil_loader_resources_") as tempdir:
        path = Path(tempdir) / "tiny.h5"
        with h5py.File(path, "w") as store:
            store.create_dataset("values", data=np.arange(SAMPLES * STAGES))
        for mode in modes:
            print(json.dumps(run_mode(path, mode), indent=2), flush=True)
    print("PASS: persistent HDF5 reuse, stage updates, and loader resource cleanup.", flush=True)


if __name__ == "__main__":
    main()
