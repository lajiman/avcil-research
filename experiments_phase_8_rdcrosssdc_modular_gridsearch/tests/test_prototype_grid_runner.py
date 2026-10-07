"""CPU-only checks for grid validation, GPU isolation, queues, and failure cleanup."""

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR))
import run_prototype_grid as runner


class GridValidationTests(unittest.TestCase):
    def test_committed_grids_cover_36_distinct_runs_and_correct_settings(self):
        runs = []
        for policy, tag in (("pre_shrink", "B"), ("historical", "C")):
            group = runner.parse_grid(PHASE_DIR / "grid_commands" / f"commands_prototype_{tag}_s3_s4.txt", PHASE_DIR)
            self.assertEqual(len(group), 18)
            self.assertEqual({r.policy for r in group}, {policy})
            self.assertEqual([r.seed for r in group[:3]], [42, 43, 44])
            self.assertEqual([r.setting for r in group[:6]], ["s3"] * 3 + ["s4"] * 3)
            self.assertTrue(all(r.argv[0] == sys.executable for r in group))
            runs.extend(group)
        self.assertEqual(len({r.experiment for r in runs}), 36)
        self.assertEqual(len({r.log_path for r in runs}), 36)

    def test_override_paths_are_single_literal_arguments(self):
        runs = runner.parse_grid(PHASE_DIR / "grid_commands/commands_prototype_B_s3_s4.txt",
                                 PHASE_DIR, feature_root="/a path/features", meta_root="/another path/meta")
        self.assertEqual(runner._option(runs[0].argv, "--feature_root"), "/a path/features")
        self.assertEqual(runner._option(runs[0].argv, "--meta_root"), "/another path/meta")

    def test_rejects_shell_operators_duplicates_missing_and_mixed_policies(self):
        text = (PHASE_DIR / "grid_commands/commands_prototype_B_s3_s4.txt").read_text()
        bad_grids = [
            text.replace("python -u", "python -u ; touch bad", 1),
            text.replace(" --seed 42", " --seed 42 --seed 43", 1),
            text.replace("pre_shrink", "historical", 1),
            text.replace("2>&1", "2>&1 &", 1),
            "\n".join(text.splitlines()[:-1]),
            text.replace("--seed 42", "--seed 43", 1),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "commands.txt"
            for bad in bad_grids:
                with self.subTest(text=bad[:60]):
                    path.write_text(bad)
                    with self.assertRaises(ValueError):
                        runner.parse_grid(path, temp)

    def test_gpu_token_retained_and_multiple_or_empty_devices_rejected(self):
        for token in ("7", "GPU-1234-abcd", "MIG-GPU-1234/1/0"):
            self.assertEqual(runner.allocated_gpu(token), token)
        for token in (None, "", " ", "0,1", "GPU-abcd,GPU-efgh", "-1", " 0", "0 1"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                runner.allocated_gpu(token)

    def test_dry_run_needs_neither_slurm_nor_torch_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {}, clear=True):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), mock.patch.dict(sys.modules, {"torch": None}):
                result = runner.main(["--commands", str(PHASE_DIR / "grid_commands/commands_prototype_C_s3_s4.txt"),
                                      "--work-dir", temp, "--dry-run", "--gpus", "GPU-dry"])
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue().count("GPU=GPU-dry"), 18)
            self.assertIn("at most 3 concurrent", output.getvalue())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_real_run_cannot_override_gpu_or_run_outside_slurm(self):
        base = ["--commands", str(PHASE_DIR / "grid_commands/commands_prototype_B_s3_s4.txt")]
        for extra in ([], ["--gpus", "0"]):
            with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main(base + extra)
                self.assertEqual(error.exception.code, 2)

    def test_six_groups_select_exact_seed_triplets_without_overlap(self):
        runs = runner.parse_grid(PHASE_DIR / "grid_commands/commands_prototype_B_s3_s4.txt", PHASE_DIR)
        groups = [runner.select_group(runs, group) for group in range(1, 7)]
        self.assertEqual([run for group in groups for run in group], runs)
        self.assertEqual([[(run.tolerance, run.setting) for run in group] for group in groups],
                         [[(tol, setting)] * 3 for tol in (0.05, 0.1, 0.2) for setting in ("s3", "s4")])
        self.assertTrue(all([run.seed for run in group] == [42, 43, 44] for group in groups))
        self.assertIs(runner.select_group(runs, None), runs)
        for invalid in (0, 7):
            with self.assertRaises(ValueError):
                runner.select_group(runs, invalid)

    def test_group_preflight_ignores_existing_output_from_other_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = runner.parse_grid(PHASE_DIR / "grid_commands/commands_prototype_B_s3_s4.txt", temp)
            runs[0].log_path.parent.mkdir(parents=True, exist_ok=True)
            runs[0].log_path.write_text("Completed group 1")
            output = Path(temp) / "save" / runs[0].experiment
            output.mkdir(parents=True)
            (output / "best.pkl").write_text("Keep existing completed result")
            runner.preflight_outputs(runner.select_group(runs, 2), temp, Path(temp) / "group2.json")
            with self.assertRaises(FileExistsError):
                runner.preflight_outputs(runner.select_group(runs, 1), temp, Path(temp) / "group1.json")

    def test_group_dry_run_prints_only_selected_three_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {}, clear=True):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), mock.patch.dict(sys.modules, {"torch": None}):
                result = runner.main(["--commands", str(PHASE_DIR / "grid_commands/commands_prototype_B_s3_s4.txt"),
                                      "--work-dir", temp, "--dry-run", "--gpus", "GPU-dry", "--group", "4"])
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue().count("GPU=GPU-dry"), 3)
            self.assertEqual(output.getvalue().count("s4 tol=0.1 seed="), 3)
            self.assertIn("Selected group 4/6", output.getvalue())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_group_selection_still_validates_unselected_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad_grid.txt"
            text = (PHASE_DIR / "grid_commands/commands_prototype_B_s3_s4.txt").read_text()
            path.write_text(text.replace("--seed 42", "--seed 43", 1))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main(["--commands", str(path), "--dry-run", "--group", "6"])
            self.assertEqual(error.exception.code, 2)


class QueueTests(unittest.TestCase):
    def make_runs(self, directory, count=6, fail_index=None):
        directory = Path(directory)
        child = directory / "child.py"
        child.write_text(
            "import json, os, pathlib, sys, time\n"
            "root=pathlib.Path(sys.argv[1]); index=int(sys.argv[2]); delay=float(sys.argv[3])\n"
            "info={'start': time.time(), 'env': {k:os.environ.get(k) for k in "
            "['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',"
            "'NUMEXPR_NUM_THREADS','AVCIL_TORCH_THREADS','AVCIL_CUDA_MEMORY_FRACTION','AVCIL_RECORD_RESOURCES']}}\n"
            "(root / ('started_%s.json'%index)).write_text(json.dumps(info))\n"
            "if sys.argv[4]=='fail': sys.exit(7)\n"
            "time.sleep(delay)\n"
            "info['end']=time.time(); (root / ('ended_%s.json'%index)).write_text(json.dumps(info))\n",
            encoding="utf-8")
        runs = []
        for index in range(count):
            delay = 0.15 if index % 3 == 0 else 0.60
            if fail_index is not None and index != fail_index:
                delay = 20
            argv = (sys.executable, "-u", str(child), str(directory), str(index), str(delay),
                    "fail" if index == fail_index else "ok")
            runs.append(runner.GridRun(argv, directory / f"run{index}.log", f"run{index}",
                                       "pre_shrink", "s3", 0.05, 42 + index % 3))
        return runs

    def execute(self, runs, directory, **kwargs):
        gpu = "MIG-GPU-abcd/1/0"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, SLURM_JOB_ID="fixture")
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run_queue(runs, work_dir=directory, gpu=gpu,
                                    status_file=Path(directory) / "status.json", base_env=env,
                                    poll_seconds=0.01, grace_seconds=0.5, **kwargs)

    def test_three_concurrent_children_independent_lanes_and_inherited_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = self.make_runs(temp)
            self.assertEqual(self.execute(runs, temp), 0)
            state = json.loads((Path(temp) / "status.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertTrue(all(j["status"] == "completed" for j in state["jobs"]))
            self.assertEqual([j["lane"] for j in state["jobs"]], [0, 1, 2, 0, 1, 2])
            records = [json.loads((Path(temp) / f"ended_{i}.json").read_text()) for i in range(6)]
            events = sorted([(r["start"], 1) for r in records] + [(r["end"], -1) for r in records])
            active = maximum = 0
            for _, delta in events:
                active += delta
                maximum = max(maximum, active)
            self.assertEqual(maximum, 3)
            self.assertLess(records[3]["start"], records[1]["end"])
            for record in records:
                self.assertEqual(record["env"]["CUDA_VISIBLE_DEVICES"], "MIG-GPU-abcd/1/0")
                self.assertEqual(record["env"]["AVCIL_CUDA_MEMORY_FRACTION"], "0.3")
                self.assertEqual(record["env"]["AVCIL_RECORD_RESOURCES"], "1")
                for variable in runner.THREAD_VARIABLES + ("AVCIL_TORCH_THREADS",):
                    self.assertEqual(record["env"][variable], "4")

    def test_failure_is_nonzero_stops_queue_and_terminates_running_children(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = self.make_runs(temp, fail_index=0)
            start = time.monotonic()
            self.assertEqual(self.execute(runs, temp), 1)
            self.assertLess(time.monotonic() - start, 8)
            state = json.loads((Path(temp) / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["jobs"][0]["exit_code"], 7)
            self.assertEqual(state["jobs"][0]["status"], "failed")
            self.assertTrue(all(j["status"] == "cancelled" for j in state["jobs"][1:]))
            self.assertTrue(all(j["pid"] is None for j in state["jobs"][3:]))
            self.assertTrue(all(not (Path(temp) / f"started_{i}.json").exists() for i in range(3, 6)))
            self.assertTrue(all(not (Path(temp) / f"ended_{i}.json").exists() for i in range(6)))

    def test_entire_grid_preflight_prevents_even_first_process_start(self):
        for kind in ("log", "checkpoint", "metrics", "figure", "status"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                runs = self.make_runs(temp)
                path = {"log": runs[-1].log_path,
                        "checkpoint": Path(temp) / "save/run5/checkpoint.pt",
                        "metrics": Path(temp) / "save/metrics/run5/file.csv",
                        "figure": Path(temp) / "save/fig/run5/loss.png",
                        "status": Path(temp) / "status.json"}[kind]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep me")
                with mock.patch.object(runner.subprocess, "Popen") as popen, self.assertRaises(FileExistsError):
                    self.execute(runs, temp)
                popen.assert_not_called()
                self.assertEqual(path.read_text(), "keep me")
                self.assertFalse(runs[0].log_path.exists())

    def test_lower_concurrency_and_invalid_memory_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = self.make_runs(temp, count=2)
            with self.assertRaises(ValueError):
                self.execute(runs, temp, cuda_memory_fraction=0.34)
            self.assertEqual(self.execute(runs, temp, runs_per_gpu=1), 0)
            state = json.loads((Path(temp) / "status.json").read_text())
            self.assertEqual([j["lane"] for j in state["jobs"]], [0, 0])

    def test_signal_stops_children_and_marks_pending_cancelled(self):
        handlers = {}
        processes = []

        class FakeProcess:
            def __init__(self, *args, **kwargs):
                self.pid = 100 + len(processes)
                self.returncode = None
                processes.append(self)
                handlers[signal.SIGTERM](signal.SIGTERM, None)

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -signal.SIGTERM
                return self.returncode

        def install(sig, handler):
            old = handlers.get(sig, signal.SIG_DFL)
            handlers[sig] = handler
            return old

        with tempfile.TemporaryDirectory() as temp:
            runs = self.make_runs(temp)
            with mock.patch.object(runner.signal, "signal", side_effect=install), \
                    mock.patch.object(runner.subprocess, "Popen", FakeProcess), \
                    mock.patch.object(runner, "_signal_child") as send:
                self.assertEqual(self.execute(runs, temp), 128 + signal.SIGTERM)
            state = json.loads((Path(temp) / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertTrue(all(j["status"] == "cancelled" for j in state["jobs"]))
            self.assertEqual(send.call_count, len(processes))
            self.assertTrue(all(p.poll() is not None for p in processes))

    def test_posix_cleanup_signals_own_process_group(self):
        process = mock.Mock(pid=98765)
        process.poll.return_value = None
        with mock.patch.object(runner.os, "name", "posix"), \
                mock.patch.object(runner.os, "killpg", create=True) as killpg:
            runner._signal_child(process, signal.SIGTERM)
        killpg.assert_called_once_with(98765, signal.SIGTERM)
        process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
