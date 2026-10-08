"""One command: audit saved tables, evaluate checkpoints, and build a report.

Inputs under save/ and logs/ are read-only. All new files go to --results-dir.
The Slurm wrapper uses two long-lived workers, one per allocated GPU.
"""

import argparse
import ast
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

PHASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PHASE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_phase_9.analysis.common import (
    DEFAULTS, METHOD_KEYS, PROTOCOL_KEYS, fingerprint, method_label,
    path_stat_fingerprint, read_json, sha256_file, write_csv, write_json,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value, base=PHASE_ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def log_config(path):
    """Read argparse's Namespace representation without executing log contents."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for _, line in zip(range(100), handle):
            if not line.strip().startswith("Namespace("):
                continue
            try:
                node = ast.parse(line.strip(), mode="eval").body
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "Namespace" or node.args or any(k.arg is None for k in node.keywords):
                    continue
                return {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            except (SyntaxError, ValueError, TypeError):
                return None
    return None


def input_inventory(save_root, logs_root, run_ids):
    paths = set()
    for name in run_ids:
        config = save_root / name / "config.json"
        if config.is_file():
            paths.add(config)
        metrics = save_root / "metrics" / name
        if metrics.is_dir():
            paths.update(p for p in metrics.rglob("*") if p.is_file() and p.suffix in (".csv", ".json")
                         and not any(part.startswith("test_only_") for part in p.relative_to(metrics).parts))
        log = logs_root / (name + ".log")
        if log.is_file():
            paths.add(log)
    return [path_stat_fingerprint(p) for p in sorted(paths)]


def discover_runs(args):
    save_root, logs_root = resolve_path(args.save_root), resolve_path(args.logs_root)
    names = {p.parent.name for p in save_root.glob("*/config.json")}
    names.update(p.stem for p in logs_root.glob("*.log"))
    runs, issues = [], []
    for name in sorted(names):
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in args.runs):
            continue
        # Names are directory components, never shell code or arbitrary paths.
        if name in (".", "..", "metrics") or "/" in name or "\\" in name:
            issues.append({"run_id": name, "status": "invalid_name"})
            continue
        run_dir, log_path = save_root / name, logs_root / (name + ".log")
        try:
            raw = read_json(run_dir / "config.json")
            source = "config.json"
            if raw is None:
                raw, source = log_config(log_path), "log_namespace"
            if not isinstance(raw, dict) or "seed" not in raw:
                raise ValueError("No readable config.json or safe Namespace with a seed")
            cfg = {**DEFAULTS, **raw}
            n, per_step = int(cfg["num_classes"]), int(cfg["class_num_per_step"])
            if n <= 0 or per_step <= 0 or n % per_step:
                raise ValueError("Invalid incremental class counts")
            meta = resolve_path(args.meta_root or cfg.get("meta_root", "../data2/balance"))
            metadata = {file: sha256_file(meta / file) if (meta / file).is_file() else "unavailable"
                        for file in ("category_encode_dict.npy", "all_id_category_dict.npy", "all_classId_vid_dict.npy")}
            # Keep source identity when files are unavailable instead of claiming equivalence.
            data_identity = {"metadata_hashes": metadata,
                             "meta_source": str(meta) if "unavailable" in metadata.values() else "sha256",
                             "feature_source": str(resolve_path(args.feature_root or cfg.get("feature_root", "../../../datasets/VGGSound")))}
            protocol = {key: cfg[key] for key in PROTOCOL_KEYS}
            protocol.update(data_identity)
            label = method_label(cfg)
            runs.append({"run_id": name, "run_dir": str(run_dir),
                         "metrics_dir": str(save_root / "metrics" / name), "log_path": str(log_path),
                         "config": cfg, "config_source": source,
                         "config_defaulted_keys": sorted(set(DEFAULTS) - set(raw)),
                         "method": label, "method_id": label + "_" + fingerprint({key: cfg[key] for key in METHOD_KEYS})[:10],
                         "protocol_id": fingerprint(protocol)[:16], "seed": int(cfg["seed"]),
                         "num_steps": n // per_step, "data_identity": data_identity})
        except Exception as exc:
            issues.append({"run_id": name, "status": "config_error", "error": str(exc)})
    return runs, issues


def make_manifest(args):
    runs, issues = discover_runs(args)
    if not runs:
        raise ValueError("No analyzable runs matched --runs; check --save-root and --logs-root")
    root = resolve_path(args.results_dir)
    inputs = [resolve_path(args.save_root), resolve_path(args.logs_root)]
    if any(root == path or root.is_relative_to(path) or path.is_relative_to(root) for path in inputs):
        raise ValueError("--results-dir must be separate from input save/ and logs/ trees")
    jobs, missing = [], []
    for run in runs:
        selected_steps = range(run["num_steps"]) if args.steps is None else args.steps
        for step in selected_steps:
            if not 0 <= step < run["num_steps"]:
                raise ValueError(f"Step {step} is outside run {run['run_id']}")
            for kind in args.checkpoint_kinds:
                checkpoint = Path(run["run_dir"]) / f"step_{step}_{kind}_model.pt"
                job = {"job_id": f"{run['run_id']}__step_{step}__{kind}", "run_id": run["run_id"],
                       "checkpoint": str(checkpoint), "step": step, "kind": kind}
                (jobs if checkpoint.is_file() else missing).append(job)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "experiments_phase_9/analysis",
                                             "experiments_phase_9/run_analysis.py"], cwd=REPO_ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        git_sha, dirty = None, None
    code_hashes = {str(p.relative_to(PHASE_ROOT)): sha256_file(p)
                   for p in sorted((PHASE_ROOT / "analysis").glob("*.py"))}
    code_hashes["run_analysis.py"] = sha256_file(Path(__file__))
    options = {"mode": args.mode, "device": args.device, "gpus": args.gpus,
               "batch_size": args.batch_size, "num_workers": args.num_workers,
               "feature_root": str(resolve_path(args.feature_root)) if args.feature_root else None,
               "meta_root": str(resolve_path(args.meta_root)) if args.meta_root else None,
               "splits": args.splits, "force": args.force, "torch_threads": args.cpu_threads,
               "checkpoint_kinds": args.checkpoint_kinds, "steps": args.steps,
               "allow_missing_checkpoints": args.allow_missing_checkpoints}
    return {"schema_version": 1, "invocation_id": str(uuid.uuid4()), "created_at_utc": utc_now(), "phase_root": str(PHASE_ROOT),
            "save_root": str(inputs[0]), "logs_root": str(inputs[1]), "output_root": str(root),
            "options": options, "runs": runs, "jobs": jobs, "missing_checkpoints": missing,
            "discovery_issues": issues, "analysis_git_sha": git_sha, "analysis_dirty": dirty,
            "analysis_code_sha256": code_hashes,
            "training_git_sha": "not recorded by the training script; do not infer from analysis checkout",
            "input_inventory": input_inventory(inputs[0], inputs[1], [r["run_id"] for r in runs])}


def run_workers(manifest, manifest_path):
    if not manifest["jobs"]:
        return []
    options = manifest["options"]
    if options["device"] == "cuda":
        import torch
        visible = torch.cuda.device_count()
        if visible < options["gpus"]:
            raise RuntimeError(f"Requested {options['gpus']} CUDA workers but only {visible} GPUs are visible")
        count = min(options["gpus"], len(manifest["jobs"]))
    else:
        count = 1
    log_root = Path(manifest["output_root"]) / "worker_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    children, handles = [], []
    try:
        for rank in range(count):
            handle = (log_root / f"worker_{rank}.log").open("w", encoding="utf-8")
            handles.append(handle)
            env = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = str(options["torch_threads"])
            env["PYTHONUNBUFFERED"] = "1"
            command = [sys.executable, "-u", "-m", "experiments_phase_9.analysis.inference",
                       "--manifest", str(manifest_path), "--rank", str(rank), "--world-size", str(count),
                       "--device", options["device"]]
            children.append(subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env))
        last_progress = 0.0
        while any(child.poll() is None for child in children):
            if time.monotonic() - last_progress > 60:
                completed = sum((Path(manifest["output_root"]) / "cache" / job["job_id"] / "status.json").exists()
                                for job in manifest["jobs"])
                print(f"Checkpoint workers running: {completed}/{len(manifest['jobs'])} jobs have status files. Logs: {log_root}", flush=True)
                last_progress = time.monotonic()
            time.sleep(1)
        return [child.returncode for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-root", default=str(PHASE_ROOT / "save"))
    parser.add_argument("--logs-root", default=str(PHASE_ROOT / "logs"))
    parser.add_argument("--results-dir", default=str(PHASE_ROOT / "results" / "analysis"))
    parser.add_argument("--runs", nargs="+", default=["*"], help="Run-name globs, quoted in a shell")
    parser.add_argument("--mode", choices=["all", "offline"], default="all")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--checkpoint-kinds", nargs="+", choices=["best", "last"], default=["best", "last"])
    parser.add_argument("--steps", nargs="+", type=int, default=None, help="Zero-based steps; omitted means all")
    parser.add_argument("--splits", nargs="+", choices=["test", "val"], default=["test", "val"])
    parser.add_argument("--feature-root", default=None, help="Override feature paths in saved configs")
    parser.add_argument("--meta-root", default=None, help="Override metadata paths in saved configs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Recompute inference even if cache fingerprint matches")
    parser.add_argument("--allow-missing-checkpoints", action="store_true",
                        help="Allow an incomplete inference plan; missing models are still reported")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.gpus, args.batch_size, args.cpu_threads) < 1 or args.num_workers < 0:
        parser.error("gpus/batch-size/cpu-threads must be positive; num-workers must be nonnegative")
    for name in ("checkpoint_kinds", "splits", "steps"):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            parser.error(f"--{name.replace('_', '-')} contains duplicates")
    manifest = make_manifest(args)
    root = Path(manifest["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".analysis.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(f"Analysis lock exists: {lock_path}. Use another results directory or remove a stale lock after checking no analysis is running.")
    lock.write(f"pid={os.getpid()} start={utc_now()}\n")
    lock.close()
    errors, worker_codes, offline, report = [], [], {}, {}
    try:
        manifest_path = root / "manifest.json"
        write_json(manifest_path, manifest)
        write_csv(root / "tables" / "missing_checkpoints.csv", manifest["missing_checkpoints"],
                  ["job_id", "run_id", "checkpoint", "step", "kind"])
        write_json(root / "pipeline_status.json", {"status": "running", "started_at_utc": utc_now()})
        print(f"Analysis: {len(manifest['runs'])} runs, {len(manifest['jobs'])} available checkpoints, "
              f"{len(manifest['missing_checkpoints'])} missing. Output: {root}", flush=True)
        try:
            from experiments_phase_9.analysis.offline import analyze
            offline = analyze(manifest)
        except Exception:
            errors.append({"stage": "offline", "traceback": traceback.format_exc()})
        if args.mode == "all":
            if manifest["missing_checkpoints"] and not args.allow_missing_checkpoints:
                errors.append({"stage": "missing_checkpoints", "message": "Selected checkpoint files are missing; other available jobs still run."})
            try:
                worker_codes = run_workers(manifest, manifest_path)
                if any(code != 0 for code in worker_codes):
                    errors.append({"stage": "inference", "message": f"Worker exit codes: {worker_codes}; see worker logs and cache/*/status.json"})
            except Exception:
                errors.append({"stage": "inference", "traceback": traceback.format_exc()})
        if manifest["discovery_issues"]:
            errors.append({"stage": "discovery", "message": "Some selected runs could not be read", "issues": manifest["discovery_issues"]})
        after = input_inventory(Path(manifest["save_root"]), Path(manifest["logs_root"]), [r["run_id"] for r in manifest["runs"]])
        if after != manifest["input_inventory"]:
            errors.append({"stage": "inputs_changed", "message": "Training files changed during analysis. Rerun when the transfer/training is complete."})
        # Write provisional status so even a partial report can explain failures.
        status = {"status": "partial" if errors else "complete", "finished_at_utc": utc_now(),
                  "mode": args.mode, "worker_exit_codes": worker_codes, "errors": errors,
                  "missing_checkpoints": len(manifest["missing_checkpoints"]), "offline": offline}
        write_json(root / "pipeline_status.json", status)
        try:
            from experiments_phase_9.analysis.reporting import build_report
            report = build_report(manifest)
            if args.mode == "all" and report.get("cache_splits", 0) != len(manifest["jobs"]) * len(args.splits):
                errors.append({"stage": "prediction_coverage", "message": "Some requested prediction caches were absent or rejected; see report warnings."})
            if report.get("reproduction_failures", 0):
                errors.append({"stage": "checkpoint_reproduction", "message": "Recomputed predictions did not reproduce some saved metrics/gates; see tables/inference_reproduction_audit.csv."})
        except Exception:
            errors.append({"stage": "report", "traceback": traceback.format_exc()})
            (root / "report.md").write_text("# Analysis incomplete\n\nSee pipeline_status.json and tables/ for completed stages.\n", encoding="utf-8")
        status.update(status="partial" if errors else "complete", errors=errors, report=report, finished_at_utc=utc_now())
        write_json(root / "pipeline_status.json", status)
        print(f"Analysis {status['status']}: {root / 'report.md'}", flush=True)
        if errors:
            for error in errors:
                print(json.dumps(error, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1 if errors else 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
