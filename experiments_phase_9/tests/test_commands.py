"""Check the submitted experiment matrix against the actual training CLI."""

from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

from experiments_phase_9.generate_commands import main as generate_commands
from experiments_phase_9.train_incremental_fusion_modular import build_parser, validate_args


def test_two_command_files_have_the_correct_controls_and_three_seeds(tmp_path):
    feature_root = "/data/features with spaces/VGGSound"
    generate_commands(["--output_dir", str(tmp_path), "--feature_root", feature_root])
    names, logs = set(), set()
    for rule in ("fixed", "sample_aware"):
        commands = (tmp_path / f"commands_periodic_{rule}.txt").read_text(encoding="utf-8").splitlines()
        assert len(commands) == 3
        seeds = []
        for command in commands:
            words = shlex.split(command)
            redirect = words.index(">")
            parser = build_parser()
            args = parser.parse_args(words[3:redirect])
            validate_args(parser, args)
            assert words[:3] == ["python", "-u", "train_incremental_fusion_modular.py"]
            assert args.feature_root == feature_root
            assert args.fusion_mode == "periodic" and args.fusion_update_rule == rule
            assert args.device == "cuda" and args.max_epoches == 200
            assert args.instance_contrastive and args.class_contrastive and args.attn_score_distil
            assert args.experiment_name not in names and words[redirect + 1] not in logs
            names.add(args.experiment_name)
            logs.add(words[redirect + 1])
            seeds.append(args.seed)
        assert seeds == [42, 43, 44]
    assert len(names) == len(logs) == 6
    assert len(list(tmp_path.glob("commands*.txt"))) == 2

    # The checked-in files must match the generator's defaults, too.
    generate_commands(["--output_dir", str(tmp_path)])
    committed_dir = Path(__file__).resolve().parents[1] / "grid_commands"
    assert {p.name for p in committed_dir.glob("commands*.txt")} == {
        "commands_periodic_fixed.txt", "commands_periodic_sample_aware.txt"}
    for generated in tmp_path.glob("commands*.txt"):
        assert generated.read_text(encoding="utf-8") == (committed_dir / generated.name).read_text(encoding="utf-8")


@pytest.mark.parametrize("failing_seed", [None, 0])
def test_slurm_launcher_shares_one_gpu_and_reports_each_seed_failure(tmp_path, failing_seed):
    """One srun, identical GPU visibility, and a three-way concurrency barrier."""
    bash = shutil.which("bash")
    if bash is None and Path("C:/Program Files/Git/bin/bash.exe").exists():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("Bash is needed to exercise the Slurm launcher")
    commands = []
    for seed in range(3):
        exit_code = 7 if seed == failing_seed else 0
        commands.append(
            f'printf "%s %s %s %s %s\\n" "$CUDA_VISIBLE_DEVICES" "$OMP_NUM_THREADS" '
            f'"$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$NUMEXPR_NUM_THREADS" > env_{seed}; '
            f"touch started_{seed}; ready=0; "
            "for attempt in {1..80}; do "
            "if [[ -e started_0 && -e started_1 && -e started_2 ]]; then ready=1; break; fi; "
            "sleep 0.05; done; [[ $ready == 1 ]] || exit 9; "
            f"touch finished_{seed}; exit {exit_code}"
        )
    command_file = tmp_path / "commands.txt"
    command_file.write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    launcher = Path(__file__).resolve().parents[1] / "run.slurm"
    shutil.copyfile(launcher.with_name("run_shared_gpu.sh"), tmp_path / "run_shared_gpu.sh")
    allocation = launcher.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in allocation
    assert "#SBATCH --mem=300G" in allocation
    assert "#SBATCH --cpus-per-task=12" in allocation
    driver = tmp_path / "mock_slurm.sh"
    driver.write_text(
        'srun() {\n'
        '  printf "%s\\n" "$*" >> step_args.txt\n'
        '  while [[ "$1" != bash ]]; do shift; done\n'
        '  "$@"\n'
        '}\n'
        'export -f srun\n'
        'source "$PHASE9_LAUNCHER" "$PHASE9_COMMANDS"\n',
        encoding="utf-8", newline="\n",
    )
    env = {**os.environ, "SLURM_SUBMIT_DIR": tmp_path.as_posix(), "SLURM_CPUS_PER_TASK": "12",
           "CUDA_VISIBLE_DEVICES": "GPU-mock-allocated-card",
           "PHASE9_LAUNCHER": launcher.as_posix(), "PHASE9_COMMANDS": command_file.as_posix()}
    result = subprocess.run([bash, driver.as_posix()], env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == (0 if failing_seed is None else 1), result.stdout + result.stderr
    assert len(list(tmp_path.glob("finished_*"))) == 3, result.stdout + result.stderr
    steps = (tmp_path / "step_args.txt").read_text().splitlines()
    assert len(steps) == 1
    assert "--nodes=1 --ntasks=1 --cpus-per-task=12 bash run_shared_gpu.sh" in steps[0]
    for seed in range(3):
        assert (tmp_path / f"env_{seed}").read_text().strip() == "GPU-mock-allocated-card 4 4 4 4"
    if failing_seed is not None:
        assert "failed with exit code 7" in result.stderr
