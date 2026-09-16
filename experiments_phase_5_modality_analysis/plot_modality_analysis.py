#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create compact paper-oriented plots from analyze_modality_difficulty.py outputs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


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


def plot_z0_scatter(root: Path, out: Path) -> None:
    rows = read_csv(root / "core/01_z0_intrinsic_difficulty_final_step.csv")
    x = [f(r, "z0_audio_difficulty_percentile") for r in rows]
    y = [f(r, "z0_visual_difficulty_percentile") for r in rows]
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(x, y, alpha=0.75)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Audio z0 difficulty percentile")
    plt.ylabel("Visual z0 difficulty percentile")
    plt.title("Intrinsic modality difficulty")
    savefig(out / "01_z0_audio_visual_difficulty.png")


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
    asymmetry_rate = []
    for step in steps:
        states = [r["z1_binary_state"] for r in by_step[step]]
        asymmetric = sum(s in {"A_hard_V_easy", "A_easy_V_hard"} for s in states)
        asymmetry_rate.append(asymmetric / max(len(states), 1))

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(steps, mean_gap, marker="o", label="Mean |difficulty gap|")
    plt.plot(steps, asymmetry_rate, marker="s", label="EH/HE class ratio")
    plt.xlabel("Incremental step")
    plt.ylabel("Value")
    plt.title("Dynamic modality mismatch in Full AVCIL")
    plt.legend()
    savefig(out / "02_full_z1_mismatch_trajectory.png")


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
    plt.figure(figsize=(5.4, 4.6))
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
    selected = [r for r in rows if int(r["step"]) == final_step]
    states = [r["no_z1_binary_state"] for r in selected]
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
    plt.figure(figsize=(7.2, 4.4))
    plt.bar(x - width / 2, recall, width=width, label="Δ recall")
    plt.bar(x + width / 2, mismatch, width=width, label="Δ |difficulty gap|")
    plt.axhline(0.0, linestyle=":")
    plt.xticks(x, states, rotation=25, ha="right")
    plt.ylabel("Full − No-z1")
    plt.title(f"Effect of symmetric z1 contrastive (step {final_step})")
    plt.legend()
    savefig(out / "04_z1_contrastive_effect_by_status.png")


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
    print(f"Figures written to: {out}")


if __name__ == "__main__":
    main()