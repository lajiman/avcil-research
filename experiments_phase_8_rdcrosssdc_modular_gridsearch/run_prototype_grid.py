"""Run one B/C grid inside one Slurm GPU allocation, without a shell.

Each command file contains 18 runs ordered by tolerance, setting, then seed.
Three independent lanes execute indices i % 3; a completed lane immediately
starts its next run. CUDA_VISIBLE_DEVICES is inherited verbatim by every child.
This launcher limits PyTorch allocator use, not total device memory or CPU RAM.
Optionally --group 1..6 selects one setting/tolerance's three seeds after the
entire command file is validated, so Slurm arrays can give each group its own
wall-time budget without rerunning completed groups.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid


TRAINER = "train_incremental_rd_crosssdc_modular.py"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class GridRun:
    argv: tuple[str, ...]
    log_path: Path
    experiment: str
    policy: str
    setting: str
    tolerance: float
    seed: int


def _option(argv, name):
    if argv.count(name) != 1:
        raise ValueError(f"Expected exactly one {name}")
    index = argv.index(name)
    if index + 1 == len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"Missing value for {name}")
    return argv[index + 1]


def _replace_option(argv, name, value):
    if value is not None:
        _option(argv, name)
        argv[argv.index(name) + 1] = str(value)


def parse_grid(path, work_dir, *, feature_root=None, meta_root=None):
    """Parse a strict 18-run grid. Redirection is data, never shell execution."""
    work_dir = Path(work_dir).resolve()
    runs = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = shlex.split(line, posix=True)
        if len(tokens) < 6 or tokens[-3] != ">" or tokens[-1] != "2>&1":
            raise ValueError(f"Line {number}: require trailing > logfile 2>&1")
        argv, log_name = tokens[:-3], tokens[-2]
        if any(any(character in token for character in ";|&<>`\n\r") or "$(" in token
               for token in argv + [log_name]):
            raise ValueError(f"Line {number}: shell operators are forbidden")
        if argv[:3] not in (["python", "-u", TRAINER], ["python3", "-u", TRAINER]):
            raise ValueError(f"Line {number}: expected python -u {TRAINER}")
        options = [token for token in argv[3:] if token.startswith("--")]
        if len(options) != len(set(options)):
            raise ValueError(f"Line {number}: duplicate options")
        experiment = _option(argv, "--experiment_name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment):
            raise ValueError(f"Line {number}: unsafe or empty experiment name")
        policy = _option(argv, "--rd_prototype_policy")
        if policy not in ("pre_shrink", "historical"):
            raise ValueError(f"Line {number}: B/C policy required")
        if "--instance_contrastive" in argv or "--class_contrastive" in argv:
            raise ValueError(f"Line {number}: S3/S4 disable original L_i and L_c")
        coefficient = float(_option(argv, "--lam_cross_sdc_i"))
        if coefficient not in (0.0, 0.1):
            raise ValueError(f"Line {number}: expected S3=0 or S4=0.1 CrossSDC-I")
        if _option(argv, "--num_workers") != "0":
            raise ValueError(f"Line {number}: shared-GPU grid requires num_workers=0")
        if "--require_cuda" not in argv:
            raise ValueError(f"Line {number}: --require_cuda is required")
        _replace_option(argv, "--feature_root", feature_root)
        _replace_option(argv, "--meta_root", meta_root)
        argv[0] = sys.executable
        runs.append(GridRun(tuple(argv), (work_dir / log_name).resolve(), experiment,
                            policy, "s3" if coefficient == 0 else "s4",
                            float(_option(argv, "--rd_margin_tolerance")),
                            int(_option(argv, "--seed"))))
    expected = [(tol, setting, seed) for tol in (0.05, 0.1, 0.2)
                for setting in ("s3", "s4") for seed in (42, 43, 44)]
    observed = [(run.tolerance, run.setting, run.seed) for run in runs]
    if observed != expected:
        raise ValueError("Grid must contain 18 unique runs ordered tolerance (0.05,0.1,0.2), "
                         "setting (s3,s4), seed (42,43,44)")
    if len({run.policy for run in runs}) != 1:
        raise ValueError("One command file must contain only B or only C")
    if len({run.experiment for run in runs}) != len(runs) or len({run.log_path for run in runs}) != len(runs):
        raise ValueError("Experiment names and log files must be unique")
    return runs


def allocated_gpu(value):
    """Require the allocation's single opaque CUDA token, retaining its spelling."""
    if value is None or not value.strip() or "," in value or any(character.isspace() for character in value):
        raise ValueError("This job requires exactly one CUDA_VISIBLE_DEVICES token")
    if value in ("-1", "NoDevFiles"):
        raise ValueError("CUDA_VISIBLE_DEVICES contains no usable allocated GPU")
    return value


def select_group(runs, group):
    """Select a contiguous seed triplet from an already validated full grid."""
    if group is None:
        return runs
    if group not in range(1, 7) or len(runs) != 18:
        raise ValueError("Group must be 1..6 and selected from a full 18-run grid")
    return runs[3 * (group - 1):3 * group]


def preflight_outputs(runs, work_dir, status_file):
    """Validate the entire grid before creating files or starting any process."""
    if Path(status_file).exists():
        raise FileExistsError(f"Status file already exists: {status_file}")
    for run in runs:
        if run.log_path.exists():
            raise FileExistsError(f"Refusing to overwrite log: {run.log_path}")
        for directory in (Path(work_dir) / "save" / run.experiment,
                          Path(work_dir) / "save" / "metrics" / run.experiment,
                          Path(work_dir) / "save" / "fig" / run.experiment):
            if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
                raise FileExistsError(f"Existing experiment output: {directory}")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_status(path, state, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2)
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _signal_child(process, sig):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _stop_children(active, grace_seconds):
    """Only signal sessions started by this launcher; never touch unrelated jobs."""
    for process, _, _ in active.values():
        _signal_child(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    for process, _, _ in active.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _signal_child(process, getattr(signal, "SIGKILL", signal.SIGTERM))
    for process, _, _ in active.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_queue(runs, *, work_dir, gpu, status_file, runs_per_gpu=3,
              threads_per_run=4, cuda_memory_fraction=0.30, base_env=None,
              poll_seconds=0.2, grace_seconds=20.0, install_signal_handlers=True,
              group=None):
    """Run independent lanes; return nonzero and cancel the grid on any failure.

    This function is CPU-testable. main() enforces real Slurm/CUDA allocation.
    The caller supplies validated runs; no shell or CUDA import is used here.
    """
    gpu = allocated_gpu(gpu)
    if runs_per_gpu not in (1, 2, 3) or threads_per_run < 1:
        raise ValueError("runs_per_gpu must be 1..3 and threads_per_run positive")
    if not math.isfinite(cuda_memory_fraction) or cuda_memory_fraction <= 0 or runs_per_gpu * cuda_memory_fraction >= 1:
        raise ValueError("Require 0 < runs_per_gpu * cuda_memory_fraction < 1")
    work_dir = Path(work_dir).resolve()
    status_file = Path(status_file).resolve()
    preflight_outputs(runs, work_dir, status_file)
    env = dict(os.environ if base_env is None else base_env)
    if env.get("CUDA_VISIBLE_DEVICES", gpu) != gpu:
        raise ValueError("GPU token must match inherited CUDA_VISIBLE_DEVICES")
    # Preserve the allocation token verbatim, including GPU-UUID/MIG spelling.
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for name in THREAD_VARIABLES:
        env[name] = str(threads_per_run)
    env["AVCIL_TORCH_THREADS"] = str(threads_per_run)
    env["AVCIL_CUDA_MEMORY_FRACTION"] = str(cuda_memory_fraction)
    env["AVCIL_RECORD_RESOURCES"] = "1"
    state = {"started_at": _timestamp(), "ended_at": None, "status": "running",
             "slurm_job_id": env.get("SLURM_JOB_ID"), "gpu": gpu,
             "slurm_array_task_id": env.get("SLURM_ARRAY_TASK_ID"), "group": group,
             "runs_per_gpu": runs_per_gpu, "threads_per_run": threads_per_run,
             "cuda_memory_fraction": cuda_memory_fraction, "jobs": []}
    queues = [deque() for _ in range(runs_per_gpu)]
    for index, run in enumerate(runs):
        lane = index % runs_per_gpu
        queues[lane].append(index)
        state["jobs"].append({"experiment": run.experiment, "policy": run.policy,
                              "setting": run.setting, "tolerance": run.tolerance, "seed": run.seed,
                              "lane": lane, "gpu": gpu, "command": list(run.argv),
                              "log": str(run.log_path), "status": "pending", "pid": None,
                              "exit_code": None, "started_at": None, "ended_at": None})
    _write_status(status_file, state, exclusive=True)
    active = {}
    stop = {"signal": None}
    old_handlers = {}
    exit_code = 0
    try:
        # Reserve all logs exclusively up front, so another launch cannot race
        # into the same grid between preflight and a later tolerance's start.
        for run in runs:
            run.log_path.parent.mkdir(parents=True, exist_ok=True)
            with run.log_path.open("x", encoding="utf-8"):
                pass
        if install_signal_handlers:
            def on_signal(signum, _frame):
                stop["signal"] = signum
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.signal(signum, on_signal)
        while active or any(queues):
            changed = False
            if stop["signal"] is not None:
                exit_code = 128 + stop["signal"]
                state["error"] = f"Received signal {stop['signal']}"
                break
            for lane, (process, index, stream) in list(active.items()):
                result = process.poll()
                if result is None:
                    continue
                stream.close()
                del active[lane]
                job = state["jobs"][index]
                job.update(status="completed" if result == 0 else "failed",
                           exit_code=result, ended_at=_timestamp())
                changed = True
                print(f"[{job['status']}] {job['experiment']} exit={result}", flush=True)
                if result != 0:
                    exit_code = 1
            if exit_code:
                _write_status(status_file, state)
                break
            for lane, queue in enumerate(queues):
                if lane in active or not queue:
                    continue
                index = queue.popleft()
                run, job = runs[index], state["jobs"][index]
                stream = run.log_path.open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(run.argv, cwd=work_dir, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=(os.name == "posix"))
                except BaseException:
                    stream.close()
                    job.update(status="failed", ended_at=_timestamp())
                    raise
                active[lane] = (process, index, stream)
                job.update(status="running", pid=process.pid, started_at=_timestamp())
                changed = True
                print(f"[start lane={lane} gpu={gpu}] {run.experiment} pid={process.pid}", flush=True)
            if changed:
                _write_status(status_file, state)
            if active:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        exit_code = 130
        state["error"] = "KeyboardInterrupt"
    except Exception as error:
        exit_code = 1
        state["error"] = f"{type(error).__name__}: {error}"
        print(state["error"], file=sys.stderr, flush=True)
    finally:
        if active:
            _stop_children(active, grace_seconds)
            for process, index, stream in active.values():
                stream.close()
                state["jobs"][index].update(status="cancelled", exit_code=process.poll(),
                                             ended_at=_timestamp())
        for job in state["jobs"]:
            if job["status"] == "pending":
                job.update(status="cancelled", ended_at=_timestamp())
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        state.update(status="completed" if exit_code == 0 else "failed",
                     exit_code=exit_code, ended_at=_timestamp())
        _write_status(status_file, state)
    return exit_code


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=range(1, 7),
                        help="Run only this setting/tolerance's three seeds (1..6), suitable for a Slurm array")
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", help="Single GPU token for dry-run display only")
    parser.add_argument("--runs-per-gpu", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--threads-per-run", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.30)
    parser.add_argument("--feature-root")
    parser.add_argument("--meta-root")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--termination-grace-seconds", type=float, default=20.0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.threads_per_run < 1 or args.termination_grace_seconds < 0:
            raise ValueError("Threads must be positive and termination grace non-negative")
        if (not math.isfinite(args.cuda_memory_fraction) or args.cuda_memory_fraction <= 0
                or args.runs_per_gpu * args.cuda_memory_fraction >= 1):
            raise ValueError("Require 0 < runs_per_gpu * cuda_memory_fraction < 1")
        runs = parse_grid(args.commands, args.work_dir, feature_root=args.feature_root,
                          meta_root=args.meta_root)
        runs = select_group(runs, args.group)
        if args.dry_run:
            gpu = allocated_gpu(args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "allocated-GPU"))
            print("DRY RUN: no processes, files, Slurm, or CUDA queries. Lanes advance independently.")
            if args.group is not None:
                print(f"Selected group {args.group}/6 after validating all 18 commands.")
            for index, run in enumerate(runs):
                print(f"{index + 1:02d} lane={index % args.runs_per_gpu} GPU={gpu} "
                      f"{run.policy} {run.setting} tol={run.tolerance:g} seed={run.seed} "
                      f"experiment={run.experiment}")
            print(f"{len(runs)} runs; at most {args.runs_per_gpu} concurrent; "
                  f"{args.threads_per_run} threads/run; allocator fraction={args.cuda_memory_fraction:g}/run")
            return 0
        if args.gpus is not None:
            raise ValueError("--gpus is only allowed with --dry-run; real jobs inherit the Slurm allocation")
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Real execution requires a Slurm allocation (SLURM_JOB_ID)")
        gpu = allocated_gpu(os.environ.get("CUDA_VISIBLE_DEVICES"))
        if os.name != "posix":
            raise ValueError("Real execution requires Linux/POSIX Slurm")
        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus and args.runs_per_gpu * args.threads_per_run > int(cpus):
            raise ValueError("Concurrent thread budget exceeds SLURM_CPUS_PER_TASK")
        suffix = f"_group{args.group}" if args.group is not None else ""
        if os.environ.get("SLURM_ARRAY_TASK_ID"):
            suffix += f"_array{os.environ['SLURM_ARRAY_TASK_ID']}"
        status = args.status_file or args.work_dir / "logs_prototype_BC" / (
            f"launcher_{args.commands.stem}_{os.environ['SLURM_JOB_ID']}{suffix}.json")
        preflight_outputs(runs, args.work_dir, status)
        import torch  # Deliberately absent from the dry-run path.
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("The allocated environment must expose exactly one usable CUDA GPU")
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"Allocated GPU token={gpu}; {properties.name}; "
              f"device total={properties.total_memory / 2**30:.2f} GiB; "
              f"free={free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
        print("The per-run fraction caps PyTorch's allocator, not all CUDA allocations. "
              "CPU RAM and actual GPU capacity are separate resources.", flush=True)
        return run_queue(runs, work_dir=args.work_dir, gpu=gpu, status_file=status,
                         runs_per_gpu=args.runs_per_gpu, threads_per_run=args.threads_per_run,
                         cuda_memory_fraction=args.cuda_memory_fraction,
                         grace_seconds=args.termination_grace_seconds, group=args.group)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
