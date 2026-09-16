#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create compact paper-oriented plots from modality-analysis CSV outputs.

The updated plots use the main 30/40/30 Easy/Medium/Hard narrative and the
selected difficulty definition stored in the CSV files.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


TERNARY_STATES = [
    "A_easy_V_easy",
    "A_easy_V_medium",
    "A_easy_V_hard",
    "A_medium_V_easy",
    "A_medium_V_medium",
    "A_medium_V_hard",
    "A_hard_V_easy",
    "A_hard_V_medium",
    "A_hard_V_hard",
]


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def state_label(state: str) -> str:
    return (
        state.replace("A_", "A:")
        .replace("_V_", " / V:")
        .replace("easy", "E")
        .replace("medium", "M")
        .replace("hard", "H")
    )


def plot_z0_scatter(root: Path, out: Path) -> None:
    rows = read_csv(root / "core/01_z0_intrinsic_difficulty_final_step.csv")
    x = [f(r, "z0_audio_difficulty_percentile") for r in rows]
    y = [f(r, "z0_visual_difficulty_percentile") for r in rows]
    easy_q = f(rows[0], "z0_audio_difficulty_easy_quantile")
    hard_q = f(rows[0], "z0_audio_difficulty_hard_quantile")
    definition = rows[0]["z0_audio_difficulty_definition"]

    plt.figure(figsize=(5.4, 5.1))
    plt.scatter(x, y, alpha=0.75)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Equal difficulty")
    for threshold in [easy_q, hard_q]:
        plt.axvline(threshold, linestyle=":", linewidth=1.1)
        plt.axhline(threshold, linestyle=":", linewidth=1.1)
    plt.xlabel("Audio z0 difficulty percentile")
    plt.ylabel("Visual z0 difficulty percentile")
    plt.title(f"Intrinsic modality difficulty\n({definition}, E/M/H=30/40/30)")
    plt.legend(loc="best")
    savefig(out / "01_z0_audio_visual_difficulty_ternary.png")


def plot_mismatch_trajectory(root: Path, out: Path) -> None:
    rows = read_csv(root / "core/02_full_z1_modality_mismatch_by_step.csv")
    by_step = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    steps = sorted(by_step)

    mean_gap = [
        np.mean([f(r, "z1_difficulty_gap_abs") for r in by_step[step]])
        for step in steps
    ]
    ternary_mismatch_rate = [
        np.mean([int(float(r["z1_ternary_mismatch"])) for r in by_step[step]])
        for step in steps
    ]
    extreme_eh_rate = [
        np.mean(
            [int(float(r["z1_extreme_easy_hard_mismatch"])) for r in by_step[step]]
        )
        for step in steps
    ]

    plt.figure(figsize=(6.5, 4.3))
    plt.plot(steps, mean_gap, marker="o", label="Mean |difficulty gap|")
    plt.plot(steps, ternary_mismatch_rate, marker="s", label="A/V E-M-H mismatch ratio")
    plt.plot(steps, extreme_eh_rate, marker="^", label="Extreme E-H / H-E ratio")
    plt.xlabel("Incremental step")
    plt.ylabel("Value")
    plt.title("Dynamic modality mismatch in Full AVCIL")
    plt.legend()
    savefig(out / "02_full_z1_mismatch_trajectory_ternary.png")


def plot_contribution_relation(root: Path, out: Path, intervention: str) -> None:
    modality_rows = read_csv(root / "core/02_full_z1_modality_mismatch_by_step.csv")
    cf_rows = read_csv(root / "core/03_counterfactual_modality_contribution_by_class.csv")
    final_step = max(int(r["step"]) for r in modality_rows)
    difficulty = {
        (int(r["step"]), int(r["class_id"])): f(
            r, "z1_difficulty_gap_signed_audio_minus_visual"
        )
        for r in modality_rows
        if r["model"] == "full"
    }
    selected = [
        r
        for r in cf_rows
        if r["model"] == "full"
        and int(r["step"]) == final_step
        and r["intervention"] == intervention
    ]
    x = [difficulty[(int(r["step"]), int(r["class_id"]))] for r in selected]
    y = [f(r, "audio_minus_visual_logit_drop") for r in selected]

    plt.figure(figsize=(5.5, 4.7))
    plt.scatter(x, y, alpha=0.75)
    if len(x) >= 2 and not np.isclose(np.std(x), 0.0):
        slope, intercept = np.polyfit(np.asarray(x), np.asarray(y), 1)
        grid = np.linspace(min(x), max(x), 100)
        plt.plot(grid, slope * grid + intercept, linestyle="--")
    plt.axvline(0.0, linestyle=":")
    plt.axhline(0.0, linestyle=":")
    plt.xlabel("Audio − visual difficulty percentile")
    plt.ylabel("Audio − visual true-logit contribution")
    plt.title(f"Difficulty vs contribution ({intervention}, step {final_step})")
    savefig(out / f"03_difficulty_contribution_{intervention}.png")


def plot_contrastive_effect(root: Path, out: Path) -> None:
    rows = read_csv(root / "core/05_z1_contrastive_effect_by_status.csv")
    rows = [r for r in rows if r["intervention"] == "geometry"]
    if not rows:
        return
    final_step = max(int(r["step"]) for r in rows)
    indexed = {
        r["no_z1_ternary_state"]: r
        for r in rows
        if int(r["step"]) == final_step
    }
    states = [state for state in TERNARY_STATES if state in indexed]
    selected = [indexed[state] for state in states]
    recall = [
        float(r.get("mean_delta_full_minus_no_z1_recall", 0.0) or 0.0)
        for r in selected
    ]
    mismatch = [
        float(r.get("mean_delta_full_minus_no_z1_z1_difficulty_gap_abs", 0.0) or 0.0)
        for r in selected
    ]

    x = np.arange(len(states))
    width = 0.38
    plt.figure(figsize=(10.0, 4.8))
    plt.bar(x - width / 2, recall, width=width, label="Δ recall")
    plt.bar(x + width / 2, mismatch, width=width, label="Δ |difficulty gap|")
    plt.axhline(0.0, linestyle=":")
    plt.xticks(x, [state_label(s) for s in states], rotation=35, ha="right")
    plt.ylabel("Full − No-z1")
    plt.title(f"Effect of symmetric z1 contrastive by E/M/H state (step {final_step})")
    plt.legend()
    savefig(out / "04_z1_contrastive_effect_by_ternary_status.png")


def plot_z1_z2_composition(root: Path, out: Path) -> None:
    rows = read_csv(root / "core/06_z1_z2_ternary_composition.csv")
    if not rows:
        return
    final_step = max(int(r["step"]) for r in rows)
    indexed = {
        r["z1_ternary_state"]: r
        for r in rows
        if int(r["step"]) == final_step
    }
    states = [state for state in TERNARY_STATES if state in indexed]
    selected = [indexed[state] for state in states]
    easy = np.asarray([f(r, "z2_easy_ratio") for r in selected])
    medium = np.asarray([f(r, "z2_medium_ratio") for r in selected])
    hard = np.asarray([f(r, "z2_hard_ratio") for r in selected])

    x = np.arange(len(states))
    plt.figure(figsize=(10.0, 4.9))
    plt.bar(x, easy, label="z2 joint easy")
    plt.bar(x, medium, bottom=easy, label="z2 joint medium")
    plt.bar(x, hard, bottom=easy + medium, label="z2 joint hard")
    plt.xticks(x, [state_label(s) for s in states], rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Within-state class ratio")
    plt.title(f"z1 modality state → z2 joint difficulty (step {final_step})")
    plt.legend(ncol=3)
    savefig(out / "05_z1_z2_ternary_composition.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--intervention", default="z1_permutation")
    args = parser.parse_args()

    root = Path(args.result_root)
    out = Path(args.out_dir) if args.out_dir else root / "figures"
    plot_z0_scatter(root, out)
    plot_mismatch_trajectory(root, out)
    plot_contribution_relation(root, out, args.intervention)
    plot_contrastive_effect(root, out)
    plot_z1_z2_composition(root, out)
    print(f"Figures written to: {out}")


if __name__ == "__main__":
    main()
