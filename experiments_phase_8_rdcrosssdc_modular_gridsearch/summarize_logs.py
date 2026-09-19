#!/usr/bin/env python3
"""Summarize per-step Testing res across seeds, using only the standard library.

Default: read both grid command files and their redirected logs, then write
results/summary.md and results/summary.csv beside this script.
Exit codes: 0 = report written (missing/partial runs are allowed);
1 = report contains invalid runs; 2 = invalid command manifest or invocation.
"""

import argparse
import ast
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import shlex
import statistics
import sys


ROOT = Path(__file__).resolve().parent
SEED_SUFFIX = re.compile(r"_seed(-?\d+)$")
RESULT = re.compile(
    r"Incremental step\s*:?\s*(\d+)\s+Testing res"
    r"(?:\s*\(overall acc\))?\s*:\s*(\S+)", re.IGNORECASE
)
# Runtime locations and seed identity are not experimental hyperparameters.
IGNORE_PARAMS = {
    "seed", "experiment_name", "feature_root", "meta_root", "require_cuda",
    "test_only", "dump_tsne", "tsne_out_root", "tsne_feature",
    "tsne_max_points_per_class",
}
PARAM_ORDER = (
    "rd_cmr_penalty", "rd_mode", "lam_cmr", "rd_class_weight_alpha",
    "rd_margin_tolerance", "rd_cmr_scale",
)


@dataclass
class Run:
    experiment: str
    seed: int
    params: dict
    log: Path
    steps: int
    status: str = "missing"
    scores: dict = field(default_factory=dict)
    notes: str = ""
    logged_params: dict = field(default_factory=dict)


def value(text):
    """Parse literal CLI values without executing log or command contents."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def read_commands(paths, seeds):
    groups = {}
    log_owners = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
                    lexer.whitespace_split = True
                    tokens = list(lexer)
                    start = next(i for i, token in enumerate(tokens)
                                 if Path(token).name == "train_incremental_rd_crosssdc_modular.py")
                    redirect = tokens.index(">", start)
                    log = Path(tokens[redirect + 1])
                    if not log.is_absolute():
                        log = ROOT / log
                    params = {}
                    i = start + 1
                    while i < redirect:
                        token = tokens[i]
                        if not token.startswith("--"):
                            raise ValueError(f"unexpected argument {token!r}")
                        key, equals, inline = token[2:].partition("=")
                        if key in params:
                            raise ValueError(f"duplicate option --{key}")
                        values = [inline] if equals else []
                        i += 1
                        if not equals:
                            while i < redirect and not tokens[i].startswith("--"):
                                values.append(tokens[i])
                                i += 1
                        parsed = [value(item) for item in values]
                        params[key] = (parsed if key == "milestones" or len(parsed) > 1
                                       else parsed[0] if parsed else True)
                    name = params["experiment_name"]
                    seed = params["seed"]
                    suffix = SEED_SUFFIX.search(name)
                    if not suffix or type(seed) is not int or int(suffix[1]) != seed:
                        raise ValueError("experiment_name must end with the matching _seed<number>")
                    if seed not in seeds:
                        raise ValueError(f"seed {seed} is not listed in --seeds")
                    total = params["num_classes"]
                    increment = params["class_num_per_step"]
                    if type(total) is not int or type(increment) is not int or increment <= 0 or total < increment:
                        raise ValueError("invalid num_classes/class_num_per_step")
                    experiment = name[:suffix.start()]
                    run = Run(experiment, seed, params, log, total // increment)
                    group = groups.setdefault(experiment, {})
                    if seed in group:
                        raise ValueError(f"duplicate experiment/seed: {name}")
                    config = {k: v for k, v in params.items() if k not in IGNORE_PARAMS}
                    if group:
                        reference = next(iter(group.values())).params
                        if config != {k: v for k, v in reference.items() if k not in IGNORE_PARAMS}:
                            raise ValueError(f"inconsistent parameters across seeds of {experiment}")
                    if log.resolve() in log_owners:
                        raise ValueError(f"more than one run redirects to {log.name}")
                    log_owners.add(log.resolve())
                    group[seed] = run
                except (ValueError, KeyError, IndexError, StopIteration, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    if not groups:
        raise ValueError("no experiment commands found")
    return groups


def namespace(text):
    node = ast.parse(text, mode="eval").body
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or node.func.id != "Namespace" or node.args
            or any(kw.arg is None for kw in node.keywords)):
        raise ValueError("invalid Namespace header")
    keys = [kw.arg for kw in node.keywords]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate Namespace argument")
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def read_log(run, log_dirs):
    if log_dirs:
        matches = {candidate.resolve() for directory in log_dirs
                   if (candidate := directory / run.log.name).is_file()}
        if len(matches) > 1:
            run.status, run.notes = "error", "multiple log files with the same name"
            return
        if not matches:
            run.notes = "log not found"
            return
        run.log = matches.pop()
    if not run.log.is_file():
        run.notes = "log not found"
        return

    header_seen = False
    unfinished_line = False
    try:
        # Stream large training logs rather than loading them into memory.
        with run.log.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                text = line.strip()
                if text.startswith("Namespace("):
                    if header_seen or run.scores:
                        raise ValueError("multiple runs appended or Namespace after results")
                    try:
                        logged = namespace(text)
                    except SyntaxError:
                        if not line.endswith("\n"):
                            unfinished_line = True
                            break
                        raise
                    run.logged_params = logged
                    header_seen = True
                    for key, expected in run.params.items():
                        if key in IGNORE_PARAMS - {"seed", "experiment_name"}:
                            continue
                        if key not in logged or logged[key] != expected:
                            raise ValueError(f"Namespace disagrees with commands: {key}")
                if "Testing res" not in line:
                    continue
                match = RESULT.search(line)
                if not match:
                    if not line.endswith("\n"):
                        unfinished_line = True
                        break
                    raise ValueError("unrecognized Testing res line")
                try:
                    step, score = int(match[1]), float(match[2])
                except ValueError:
                    if not line.endswith("\n"):
                        unfinished_line = True
                        break
                    raise
                if step >= run.steps:
                    raise ValueError(f"step {step} is outside expected range 0-{run.steps - 1}")
                if step in run.scores:
                    raise ValueError(f"duplicate Testing res for step {step}; possible appended rerun")
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"step {step} accuracy must be finite and in [0, 1]")
                run.scores[step] = score
    except (ValueError, SyntaxError, OSError) as exc:
        run.status = "error"
        run.notes = f"line {number if 'number' in locals() else '?'}: {exc}"
        run.scores.clear()  # Never combine questionable results with valid seeds.
        return
    missing = sorted(set(range(run.steps)) - run.scores.keys())
    run.status = "partial" if missing else "complete"
    notes = ["Namespace absent; parameters from commands"] if not header_seen else []
    if unfinished_line:
        notes.append("unfinished trailing line ignored; rerun summary after more output")
    if missing:
        notes.append("missing steps: " + ", ".join(map(str, missing)))
    run.notes = "; ".join(notes)


def validate_logs(groups):
    """Check the actual files and configurations after optional log relocation."""
    owners = {}
    for group in groups.values():
        for run in group.values():
            if run.status in {"complete", "partial"}:
                owners.setdefault(run.log.resolve(), []).append(run)
    for runs in owners.values():
        if len(runs) > 1:
            for run in runs:
                run.status = "error"
                run.notes = "same log file resolves to multiple experiment/seed runs"
                run.scores.clear()
    for group in groups.values():
        runs = [run for run in group.values()
                if run.logged_params and run.status in {"complete", "partial"}]
        if len(runs) < 2:
            continue
        # Include settings omitted from the command line but printed as defaults.
        keys = set().union(*(run.logged_params.keys() for run in runs)) - IGNORE_PARAMS
        missing = object()
        differing = [key for key in sorted(keys) if any(
            run.logged_params.get(key, missing) != runs[0].logged_params.get(key, missing)
            for run in runs[1:]
        )]
        if differing:
            for run in runs:
                run.status = "error"
                run.notes = "Namespace differs across seeds: " + ", ".join(differing)
                run.scores.clear()


def parameter_columns(groups):
    configs = [next(iter(group.values())).params for group in groups.values()]
    keys = set().union(*(config.keys() for config in configs)) - IGNORE_PARAMS
    varying = {key for key in keys if any(config.get(key) != configs[0].get(key) for config in configs[1:])}
    # Keep the objective label even when summarizing only one grid file.
    if "rd_cmr_penalty" in keys:
        varying.add("rd_cmr_penalty")
    return [key for key in PARAM_ORDER if key in varying] + sorted(varying - set(PARAM_ORDER))


def display(value):
    if isinstance(value, float):
        return format(value, ".8g")
    return str(value)


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def write_reports(groups, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    params = parameter_columns(groups)
    max_steps = max(run.steps for group in groups.values() for run in group.values())
    fields = ["experiment", "row_type", "seed", "status", *params,
              *(f"step_{step}" for step in range(max_steps)),
              *(f"n_step_{step}" for step in range(max_steps)), "source_log", "notes"]
    lines = [
        "# Testing res 汇总", "",
        "指标为每个增量阶段的 overall accuracy（0–1），step 从 0 开始。",
        f"预期 seeds：{', '.join(map(str, seeds))}。complete 仅表示 Testing res 步骤齐全。",
        "参数取自命令清单，有 Namespace 时核对；仅展示变化参数及损失类型。",
        "每步按已有有效结果计算均值，n 表示该步实际参与的 seed 数。",
        "n≥2 时计算样本标准差（ddof=1）；n=1 时保留单次结果，标准差为 —；无结果为 —。", "",
    ]
    pending, errors = 0, 0
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment, group in groups.items():
            reference = next(iter(group.values()))
            settings = {key: display(reference.params.get(key, "")) for key in params}
            lines += [f"## {md_cell(experiment)}", "",
                      "参数：" + ", ".join(f"`{key}={md_cell(val)}`" for key, val in settings.items()), "",
                      "| seed | 状态 | " + " | ".join(f"step_{i}" for i in range(reference.steps)) + " |",
                      "| --- | --- | " + " | ".join("---" for _ in range(reference.steps)) + " |"]
            for seed in seeds:
                run = group.get(seed)
                status = run.status if run else "missing"
                note = run.notes if run else "seed absent from command manifest"
                scores = run.scores if run else {}
                pending += status in {"missing", "partial"}
                errors += status == "error"
                formatted = {f"step_{step}": f"{score:.6f}" for step, score in scores.items()}
                counts = {f"n_step_{step}": int(step in scores) for step in range(reference.steps)}
                writer.writerow(dict(experiment=experiment, row_type="seed", seed=seed,
                                     status=status, **settings, **formatted, **counts,
                                     source_log=run.log.name if run else "", notes=note))
                cells = [formatted.get(f"step_{i}", "—") for i in range(reference.steps)]
                lines.append(f"| {seed} | {status} | " + " | ".join(cells) + " |")

            means, stds, counts = {}, {}, {}
            for step in range(reference.steps):
                values = [group[seed].scores[step] for seed in seeds
                          if seed in group and step in group[seed].scores]
                counts[f"n_step_{step}"] = len(values)
                if values:
                    means[step] = statistics.mean(values)
                    if len(values) > 1:
                        stds[step] = statistics.stdev(values)
            status = "complete" if all(n == len(seeds) for n in counts.values()) else "partial"
            cells = [f"{means[i]:.6f} ± {stds[i]:.6f}" if i in stds
                     else f"{means[i]:.6f} ± —" if i in means else "—" for i in range(reference.steps)]
            lines.append(f"| mean ± std | {status} | " + " | ".join(cells) + " |")
            lines.append("| n | 有效 seed 数 | " + " | ".join(map(str, counts.values())) + " |")
            for kind, values in (("mean", means), ("std", stds)):
                writer.writerow(dict(experiment=experiment, row_type=kind, seed="",
                                     status=status, **settings, **counts,
                                     **{f"step_{step}": f"{score:.6f}" for step, score in values.items()},
                                     source_log="", notes="available seeds per step; std requires n >= 2"))
            notes = [f"seed {seed}: {group[seed].notes}" if seed in group else f"seed {seed}: missing command"
                     for seed in seeds if seed not in group or group[seed].notes]
            if notes:
                lines += ["", "说明：" + "; ".join(md_cell(note) for note in notes)]
            lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return pending, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commands", nargs="+", type=Path,
                        default=[ROOT / "grid_commands/commands_hinge.txt", ROOT / "grid_commands/commands_direct.txt"],
                        help="Grid command files (default: both hinge and direct)")
    parser.add_argument("--log-dir", nargs="+", type=Path,
                        help="Optional replacement log directories; match redirected filenames")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results",
                        help="Output directory (default: results beside this script)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates")
    try:
        groups = read_commands(args.commands, args.seeds)
        for group in groups.values():
            for run in group.values():
                read_log(run, args.log_dir)
        validate_logs(groups)
        pending, errors = write_reports(groups, args.seeds, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output_dir / 'summary.md'} and {args.output_dir / 'summary.csv'}")
    print(f"Experiments: {len(groups)}; expected seed runs: {len(groups) * len(args.seeds)}; "
          f"missing/partial: {pending}; invalid: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
