"""Check the submitted experiment matrix against the actual training CLI."""

from pathlib import Path
import os
import shlex
import shutil
import subprocess

import pytest

from experiments_phase_9.generate_commands import main as generate_commands
from experiments_phase_9.class_fusion.fusion_method import should_update_gate
from experiments_phase_9.class_fusion.cl_history import should_record_cl_history
from experiments_phase_9.train_incremental_fusion_modular import build_parser, validate_args


def test_three_command_files_have_the_correct_controls_and_three_seeds(tmp_path):
    feature_root = "/data/features with spaces/VGGSound"
    generate_commands(["--output_dir", str(tmp_path), "--feature_root", feature_root])
    names, logs = set(), set()
    for label, mode, rule in (("periodic_fixed", "periodic", "fixed"),
                              ("periodic_sample_aware", "periodic", "sample_aware"),
                              ("uniform_cl_history", "uniform", None)):
        commands = (tmp_path / f"commands_{label}.txt").read_text(encoding="utf-8").splitlines()
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
            assert args.fusion_mode == mode
            if rule is not None:
                assert args.fusion_update_rule == rule
            else:
                assert "--record_cl_history" in words
                assert not any(should_update_gate(args, step, epoch)
                               for step in range(10) for epoch in range(201))
                assert should_record_cl_history(args, 1, 40)
                assert should_record_cl_history(args, 1, 200)
                assert should_record_cl_history(args, 1, 3, is_best=True)
                assert not should_record_cl_history(args, 0, 40)
            assert args.record_cl_history and args.cl_history_interval == 40
            assert args.fusion_temperature == 0.1 and args.fusion_min_samples == 2
            assert args.device == "cuda" and args.max_epoches == 200
            assert args.instance_contrastive and args.class_contrastive and args.attn_score_distil
            assert args.experiment_name not in names and words[redirect + 1] not in logs
            names.add(args.experiment_name)
            logs.add(words[redirect + 1])
            seeds.append(args.seed)
        assert seeds == [42, 43, 44]
    assert len(names) == len(logs) == 9
    assert len(list(tmp_path.glob("commands*.txt"))) == 3

    # The checked-in files must match the generator's defaults, too.
    generate_commands(["--output_dir", str(tmp_path)])
    committed_dir = Path(__file__).resolve().parents[1] / "grid_commands"
    assert {p.name for p in committed_dir.glob("commands*.txt")} == {
        "commands_periodic_fixed.txt", "commands_periodic_sample_aware.txt", "commands_uniform_cl_history.txt"}
    for generated in tmp_path.glob("commands*.txt"):
        assert generated.read_text(encoding="utf-8") == (committed_dir / generated.name).read_text(encoding="utf-8")


def test_generate_only_uniform_preserves_existing_periodic_files(tmp_path):
    existing = tmp_path / "commands_periodic_fixed.txt"
    existing.write_text("existing server-specific experiment commands\n", encoding="utf-8")
    before = existing.read_bytes()
    generate_commands(["--output_dir", str(tmp_path), "--settings", "uniform_cl_history"])
    assert existing.read_bytes() == before
    assert len((tmp_path / "commands_uniform_cl_history.txt").read_text().splitlines()) == 3
    assert not (tmp_path / "commands_periodic_sample_aware.txt").exists()


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
    assert "#SBATCH --gres=gpu:nvidia_h200_nvl:1" in allocation
    assert "#SBATCH --mem=180G" in allocation
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
    # The current Juno launcher initializes Conda inside the job; mock that
    # external dependency so this remains a local launcher/concurrency test.
    conda_root = tmp_path / "mock_conda"
    conda_hook = conda_root / "etc/profile.d/conda.sh"
    conda_hook.parent.mkdir(parents=True)
    conda_hook.write_text('conda() { export CONDA_PREFIX="$2"; }\n', encoding="utf-8", newline="\n")
    env.update(AVCIL_PROJECT_ROOT=tmp_path.as_posix(), AVCIL_CONDA_ROOT=conda_root.as_posix(),
               AVCIL_CONDA_ENV="phase9-test-environment")
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
