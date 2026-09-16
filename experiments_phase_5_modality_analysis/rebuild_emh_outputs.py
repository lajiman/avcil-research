#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild 30/40/30 E/M/H outputs from already generated analysis CSVs.

This script does not load checkpoints and does not run model inference. It uses
saved difficulty percentiles and counterfactual statistics, so existing random,
easy2hard, and hard2easy result folders can be upgraded quickly.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List

import analyze_modality_difficulty as analysis


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))




def coerce_numeric_rows(rows: List[Dict[str, Any]]) -> None:
    """Convert numeric CSV cells back to floats for analysis helpers."""
    skip_exact = {
        "model", "class_name", "category_name", "intervention",
        "nearest_class_name", "class_names", "class_ids",
    }
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", None):
                continue
            if (
                key in skip_exact
                or key.endswith("_state")
                or key.endswith("_binary")
                or key.endswith("_ternary")
                or "nearest_class_name" in key
            ):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass


def classify(pct: float, easy_ratio: float, hard_ratio: float) -> str:
    if pct < easy_ratio:
        return "easy"
    if pct < 1.0 - hard_ratio:
        return "medium"
    return "hard"


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_emh")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def relabel_z0(rows: List[Dict[str, Any]], easy_ratio: float, hard_ratio: float) -> None:
    for row in rows:
        a = classify(float(row["z0_audio_difficulty_percentile"]), easy_ratio, hard_ratio)
        v = classify(float(row["z0_visual_difficulty_percentile"]), easy_ratio, hard_ratio)
        row["z0_audio_difficulty_ternary"] = a
        row["z0_visual_difficulty_ternary"] = v
        row["z0_ternary_state"] = f"A_{a}_V_{v}"


def relabel_modality(rows: List[Dict[str, Any]], easy_ratio: float, hard_ratio: float) -> None:
    for row in rows:
        a = classify(float(row["z1_audio_difficulty_percentile"]), easy_ratio, hard_ratio)
        v = classify(float(row["z1_visual_difficulty_percentile"]), easy_ratio, hard_ratio)
        z2 = classify(float(row["z2_fusion_difficulty_percentile"]), easy_ratio, hard_ratio)
        row["z1_audio_difficulty_ternary"] = a
        row["z1_visual_difficulty_ternary"] = v
        row["z2_fusion_difficulty_ternary"] = z2
        row["z1_ternary_state"] = f"A_{a}_V_{v}"
        row["z2_difficulty_ternary"] = z2
        row["z1_z2_ternary_state"] = f"A_{a}_V_{v}_Z2_{z2}"


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--ternary_easy_ratio", type=float, default=0.30)
    parser.add_argument("--ternary_hard_ratio", type=float, default=0.30)
    args = parser.parse_args()

    if args.ternary_easy_ratio + args.ternary_hard_ratio >= 1.0:
        raise ValueError("easy + hard ratios must be below 1")

    root = Path(args.result_root)
    core = root / "core"
    detail = root / "detail"

    z0_path = core / "01_z0_intrinsic_difficulty_final_step.csv"
    modality_path = detail / "modality_difficulty_by_class_step.csv"
    cf_path = detail / "counterfactual_contribution_by_class.csv"
    if not cf_path.exists():
        cf_path = core / "03_counterfactual_modality_contribution_by_class.csv"

    for path in [z0_path, modality_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    z0_rows = read_csv(z0_path)
    modality_rows = read_csv(modality_path)
    cf_rows = read_csv(cf_path)
    coerce_numeric_rows(z0_rows)
    coerce_numeric_rows(modality_rows)
    coerce_numeric_rows(cf_rows)

    relabel_z0(z0_rows, args.ternary_easy_ratio, args.ternary_hard_ratio)
    relabel_modality(modality_rows, args.ternary_easy_ratio, args.ternary_hard_ratio)

    backup_once(z0_path)
    backup_once(modality_path)
    analysis.write_csv(z0_path, z0_rows)
    analysis.write_csv(modality_path, modality_rows)

    full_rows = [row for row in modality_rows if row["model"] == "full"]
    no_z1_rows = [row for row in modality_rows if row["model"] == "no_z1"]
    full_cf = [row for row in cf_rows if row["model"] == "full"]
    no_z1_cf = [row for row in cf_rows if row["model"] == "no_z1"]

    analysis.write_csv(core / "02_full_z1_modality_mismatch_by_step.csv", full_rows)
    detail_rows, binary_rows, ternary_rows, ternary_joint_rows = (
        analysis.build_contrastive_effect_rows(
            full_rows, no_z1_rows, full_cf, no_z1_cf
        )
    )
    composition = analysis.build_full_ternary_composition_rows(
        modality_rows, cf_rows, preferred_intervention="z1_permutation"
    )

    analysis.write_csv(core / "05_z1_contrastive_effect_by_status.csv", binary_rows)
    analysis.write_csv(
        core / "06_full_z1_z2_ternary_composition_by_step.csv", composition
    )
    analysis.write_csv(
        core / "07_z1_contrastive_effect_by_ternary_status.csv", ternary_rows
    )
    analysis.write_csv(
        core / "08_z1_contrastive_effect_by_ternary_joint_status.csv",
        ternary_joint_rows,
    )
    analysis.write_csv(detail / "z1_contrastive_effect_by_class.csv", detail_rows)

    print(f"Rebuilt 30/40/30 E/M/H outputs under: {root}")


if __name__ == "__main__":
    main()