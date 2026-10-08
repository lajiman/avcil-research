"""Integration contracts for unattended, read-only analysis orchestration."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from experiments_phase_9 import run_analysis as pipeline
from experiments_phase_9.analysis.common import DEFAULTS, read_json, write_json


def fixture_args(tmp_path):
    save, logs = tmp_path / "save", tmp_path / "logs"
    logs.mkdir()
    for name, mode in (("control", "uniform"), ("method", "periodic")):
        config = {**DEFAULTS, "seed": 42, "fusion_mode": mode, "num_classes": 4,
                  "class_num_per_step": 2, "max_epoches": 2, "experiment_name": name}
        write_json(save / name / "config.json", config)
    return pipeline.build_parser().parse_args([
        "--save-root", str(save), "--logs-root", str(logs),
        "--results-dir", str(tmp_path / "results"), "--mode", "offline"])


def test_manifest_matches_protocol_not_name_and_never_invents_checkpoints(tmp_path):
    args = fixture_args(tmp_path)
    manifest = pipeline.make_manifest(args)
    assert len(manifest["runs"]) == 2
    assert manifest["jobs"] == [] and len(manifest["missing_checkpoints"]) == 8
    a, b = manifest["runs"]
    assert a["protocol_id"] == b["protocol_id"] and a["method_id"] != b["method_id"]
    config_path = Path(b["run_dir"]) / "config.json"
    config = read_json(config_path)
    config["memory_size"] += 10
    write_json(config_path, config)
    changed = pipeline.make_manifest(args)
    assert changed["runs"][0]["protocol_id"] != changed["runs"][1]["protocol_id"]
    args.results_dir = str(Path(args.save_root) / "results")
    with pytest.raises(ValueError, match="separate"):
        pipeline.make_manifest(args)


def test_log_namespace_never_executes_code(tmp_path):
    path, marker = tmp_path / "run.log", tmp_path / "injected"
    path.write_text(f"Namespace(seed=42, x=__import__('pathlib').Path({str(marker)!r}).touch())\n")
    assert pipeline.log_config(path) is None
    assert not marker.exists()
    path.write_text("Namespace(seed=42, milestones=[100], lr_decay=False, experiment_name='run')\n")
    assert pipeline.log_config(path)["milestones"] == [100]


def test_missing_checkpoints_report_after_other_stages_and_return_nonzero(tmp_path, monkeypatch):
    from experiments_phase_9.analysis import offline, reporting
    args = fixture_args(tmp_path)
    called = []
    monkeypatch.setattr(offline, "analyze", lambda manifest: called.append("offline") or {"ok": True})
    def report(manifest):
        called.append("report")
        assert read_json(Path(manifest["output_root"]) / "pipeline_status.json")["status"] == "partial"
        return {"ok": True}
    monkeypatch.setattr(reporting, "build_report", report)
    argv = ["--save-root", args.save_root, "--logs-root", args.logs_root,
            "--results-dir", args.results_dir, "--mode", "all", "--device", "cpu"]
    assert pipeline.main(argv) == 1
    assert called == ["offline", "report"]
    assert not (Path(args.results_dir) / ".analysis.lock").exists()
    assert read_json(Path(args.results_dir) / "pipeline_status.json")["errors"][0]["stage"] == "missing_checkpoints"


def test_two_gpu_workers_keep_slurm_visibility_and_propagate_failure(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    manifest = pipeline.make_manifest(args)
    manifest["options"].update(device="cuda", gpus=2)
    manifest["jobs"] = manifest["missing_checkpoints"][:3]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-a,GPU-allocated-b")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    commands = []
    class Child:
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
            self.returncode = 0 if len(commands) == 1 else 7
        def poll(self):
            return self.returncode
    monkeypatch.setattr(pipeline.subprocess, "Popen", Child)
    assert pipeline.run_workers(manifest, tmp_path / "manifest.json") == [0, 7]
    assert len(commands) == 2
    for rank, (command, kwargs) in enumerate(commands):
        assert command[command.index("--rank") + 1] == str(rank)
        assert command[command.index("--world-size") + 1] == "2"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-a,GPU-allocated-b"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "6"


@pytest.mark.parametrize("status", [0, 7])
def test_two_gpu_slurm_entry_forwards_options_and_failure(tmp_path, status):
    bash = shutil.which("bash")
    if bash is None and Path("C:/Program Files/Git/bin/bash.exe").exists():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("Bash is required for the launcher test")
    script = Path(__file__).resolve().parents[1] / "run_analysis.slurm"
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:nvidia_h200_nvl:2" in text
    assert "#SBATCH --mem=300G" in text
    assert "#SBATCH --reservation=xie" in text
    (tmp_path / "run_analysis.py").write_text("# Mock analysis entry, not executed")
    hook = tmp_path / "conda/etc/profile.d/conda.sh"
    hook.parent.mkdir(parents=True)
    hook.write_text('conda() { export CONDA_PREFIX="$2"; }\n', newline="\n")
    driver = tmp_path / "driver.sh"
    driver.write_text(
        'srun() { printf "%s\\n" "$@" > "$AVCIL_PROJECT_ROOT/args.txt"; '
        'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" > "$AVCIL_PROJECT_ROOT/gpus.txt"; '
        f'return {status}; }}\nexport -f srun\n'
        'source "$LAUNCHER" --batch-size 16 --runs "phase9_*"\n', newline="\n")
    env = {**os.environ, "AVCIL_PROJECT_ROOT": tmp_path.as_posix(),
           "AVCIL_CONDA_ROOT": (tmp_path / "conda").as_posix(),
           "SLURM_CPUS_PER_TASK": "12", "SLURM_JOB_ID": "12345",
           "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b", "LAUNCHER": script.as_posix()}
    result = subprocess.run([bash, driver.as_posix()], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == status, result.stdout + result.stderr
    words = (tmp_path / "args.txt").read_text().splitlines()
    assert words[:3] == ["--nodes=1", "--ntasks=1", "--cpus-per-task=12"]
    assert words[words.index("--gpus") + 1] == "2"
    assert words[words.index("--cpu-threads") + 1] == "6"
    assert words[-4:] == ["--batch-size", "16", "--runs", "phase9_*"]
    assert (tmp_path / "gpus.txt").read_text().strip() == "GPU-a,GPU-b"


def test_full_cpu_pipeline_reads_real_checkpoints_and_resumes(tmp_path):
    from experiments_phase_9.tests.test_training import make_fixture
    from experiments_phase_9.train_incremental_fusion_modular import main as train_main
    from experiments_phase_9.analysis.common import read_csv, sha256_file

    features, meta = make_fixture(tmp_path)
    save, logs, results = tmp_path / "trained", tmp_path / "logs", tmp_path / "results"
    logs.mkdir()
    base = ["--feature_root", str(features), "--meta_root", str(meta), "--output_root", str(save),
            "--num_classes", "6", "--class_num_per_step", "2", "--memory_size", "8",
            "--max_epoches", "2", "--train_batch_size", "4", "--infer_batch_size", "4",
            "--exemplar_batch_size", "4", "--fusion_batch_size", "4", "--fusion_warmup_epochs", "1",
            "--fusion_update_interval", "1", "--device", "cpu", "--seed", "7"]
    for run, mode in (("baseline", "uniform"), ("dynamic", "periodic")):
        train_main(base + ["--experiment_name", run, "--fusion_mode", mode])
    checkpoints = {str(path): sha256_file(path) for path in save.glob("*/*.pt")}
    argv = ["--save-root", str(save), "--logs-root", str(logs), "--results-dir", str(results),
            "--mode", "all", "--device", "cpu", "--cpu-threads", "2", "--batch-size", "4",
            "--checkpoint-kinds", "best", "--steps", "1", "2"]
    assert pipeline.main(argv) == 0
    assert (results / "report.md").is_file() and (results / "report.html").is_file()
    assert len(read_csv(results / "tables/run_summary.csv")) == 2
    assert read_csv(results / "tables/mechanism_decomposition.csv")
    assert read_csv(results / "tables/validation_group_metrics.csv")
    status = read_json(results / "pipeline_status.json")
    assert status["status"] == "complete", status
    first_invocation = read_json(results / "manifest.json")["invocation_id"]
    assert pipeline.main(argv) == 0
    worker = read_json(results / "worker_0.json")
    assert len(worker["jobs"]) == 4 and all(row["cached"] for row in worker["jobs"])
    second_invocation = read_json(results / "manifest.json")["invocation_id"]
    assert second_invocation != first_invocation
    assert all(read_json(path)["invocation_id"] == second_invocation for path in (results / "cache").glob("*/status.json"))
    assert all(sha256_file(path) == digest for path, digest in checkpoints.items())
