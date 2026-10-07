"""Opt-in resource limits/measurements for concurrent, independent GPU runs.

These controls never change batch size, precision, loss or replay order.
Absent AVCIL_* environment variables, existing training settings are retained.
"""
import csv
import math
import os
from pathlib import Path
import time

import torch


def configure_torch_threads():
    value = os.environ.get("AVCIL_TORCH_THREADS", "8")
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("AVCIL_TORCH_THREADS must be a positive integer") from error
    if count < 1:
        raise ValueError("AVCIL_TORCH_THREADS must be a positive integer")
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)


def configure_cuda_budget(device):
    value = os.environ.get("AVCIL_CUDA_MEMORY_FRACTION")
    if value is None:
        return
    fraction = float(value)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("AVCIL_CUDA_MEMORY_FRACTION must lie in (0, 1]")
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise ValueError("A CUDA allocator budget requires exactly one visible GPU per process")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    total = torch.cuda.get_device_properties(device).total_memory
    print("[Resources] PyTorch allocator limit {:.2f} GiB ({:.0%} of visible GPU); "
          "CUDA context/library allocations are additional.".format(total * fraction / 2**30, fraction),
          flush=True)


def resource_recording_enabled():
    return os.environ.get("AVCIL_RECORD_RESOURCES", "0") == "1"


def start_resource_phase(device):
    if resource_recording_enabled() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return time.monotonic()


def record_resource_phase(path, step, phase, started, device):
    """GPU phase peaks and host RSS; Linux RSS peak covers process lifetime."""
    if not resource_recording_enabled():
        return
    row = {"step": step, "phase": phase, "pid": os.getpid(),
           "elapsed_seconds": time.monotonic() - started,
           "torch_threads": torch.get_num_threads(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
           "host_rss_gib": "", "host_lifetime_peak_rss_gib": "",
           "cuda_peak_allocated_gib": "", "cuda_peak_reserved_gib": ""}
    status = Path("/proc/self/status")
    if status.is_file():
        fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
        for name, output in (("VmRSS", "host_rss_gib"), ("VmHWM", "host_lifetime_peak_rss_gib")):
            if name in fields:
                row[output] = int(fields[name].split()[0]) / 2**20
    if device.type == "cuda":
        row["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        row["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print("[Resources] {} step {}: GPU allocated peak={} GiB, reserved peak={} GiB; "
          "host RSS={} GiB".format(phase, step, row["cuda_peak_allocated_gib"],
                                    row["cuda_peak_reserved_gib"], row["host_rss_gib"]), flush=True)
