#!/usr/bin/env python3
"""Read existing RD-CrossSDC CSVs; export diagnostic tables and PNG charts.

No model loading, GPU, dataset access, or training-code changes are needed.
Python 3.8+; dependencies: numpy, pandas, matplotlib.

Important: class Need is a bias-corrected EMA of mean positive margin drops.
It is NOT the fraction of samples drifting in a class. Four-quadrant charts
are explicitly CLASS-LEVEL EMA proxies, not sample-level joint counts.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DIRECTIONS = ("a_from_v", "v_from_a")
DISPLAY = {"a_from_v": "A<-V", "v_from_a": "V<-A"}
QUADRANTS = (
    "reliable_high_need", "reliable_low_need",
    "unreliable_high_need", "unreliable_low_need",
)
QUAD_LABELS = (
    "Reliable / high EMA Need", "Reliable / low EMA Need",
    "Unreliable / high EMA Need", "Unreliable / low EMA Need",
)
CMR_MODES = {"crosssdc_cmr", "adaptive_crosssdc_cmr"}


def required(df: pd.DataFrame, names: List[str], source: str) -> None:
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError("{}: missing columns {}".format(source, missing))


def numeric(df: pd.DataFrame, names: List[str], source: str) -> None:
    for col in names:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError("{}: non-finite {}".format(source, col))


def unique(df: pd.DataFrame, keys: List[str], source: str) -> None:
    if df.duplicated(keys).any():
        raise ValueError("{}: duplicate {}; do not mix reruns in one directory".format(source, keys))


def read_csv(path: Path, source_files: List[str]) -> pd.DataFrame:
    source_files.append(str(path.resolve()))
    # Preserve category names such as 'NA'; parse numeric columns separately.
    df = pd.read_csv(path, keep_default_na=False)
    if df.empty:
        raise ValueError("{} is empty/incomplete; retry after it finishes writing".format(path))
    return df


def parse_steps(text: str) -> Optional[set]:
    if text == "all":
        return None
    values = {int(x.strip()) for x in text.split(",")}
    if not values or min(values) < 1:
        raise ValueError("--steps must be 'all' or positive zero-based steps, e.g. 1,4,9")
    return values


def keep_step(step: int, args) -> bool:
    return int(step) > 0 and (args.step_set is None or int(step) in args.step_set)


def resolve_roots(path: Path) -> Tuple[Path, Path]:
    path = path.expanduser().resolve()
    if path.is_file() and path.name == "epoch_summary.csv":
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError("Metrics directory does not exist: {}".format(path))
    if (path / "rd_crosssdc").is_dir():
        return path, path / "rd_crosssdc"
    if (path / "epoch_summary.csv").exists() or list(path.glob("step_*_static_trust.csv")):
        return (path.parent if path.name == "rd_crosssdc" else path), path
    raise FileNotFoundError("No RD-CrossSDC CSV files found under {}".format(path))


def distribution(x, gap: float) -> dict:
    a = np.asarray(x, dtype=float)
    if len(a) == 0:
        return {}
    mean = float(a.mean())
    return {
        "n": len(a), "mean": mean, "std": float(a.std(ddof=0)),
        "min": float(a.min()), "p10": float(np.quantile(a, .1)),
        "p25": float(np.quantile(a, .25)), "median": float(np.median(a)),
        "p75": float(np.quantile(a, .75)), "p90": float(np.quantile(a, .9)),
        "max": float(a.max()), "iqr": float(np.quantile(a, .75) - np.quantile(a, .25)),
        "cv": float(a.std(ddof=0) / abs(mean)) if abs(mean) > 1e-12 else np.nan,
        "mean_deviation_threshold": gap,
        "fraction_abs_deviation_from_mean_gt_threshold": float((abs(a - mean) > gap).mean()),
    }


def rank_corr(a, b) -> float:
    a, b = pd.Series(np.asarray(a)), pd.Series(np.asarray(b))
    if a.nunique() < 2 or b.nunique() < 2 or len(a) < 2:
        return np.nan
    return float(a.rank(method="average").corr(b.rank(method="average")))


def paired_stats(a, b, threshold: float) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gap = a - b
    return {
        "n_classes": len(a), "mean_A_minus_V": float(gap.mean()),
        "mean_abs_gap": float(abs(gap).mean()),
        "p90_abs_gap": float(np.quantile(abs(gap), .9)),
        "gap_threshold": threshold,
        "fraction_abs_gap_gt_threshold": float((abs(gap) > threshold).mean()),
        "fraction_A_gt_V": float((gap > 0).mean()),
        "spearman": rank_corr(a, b),
    }


def quadrant_summary(scores, needs, r_threshold: float, n_threshold: float) -> dict:
    reliable = np.asarray(scores) > r_threshold
    high_need = np.asarray(needs) > n_threshold
    masks = (reliable & high_need, reliable & ~high_need,
             ~reliable & high_need, ~reliable & ~high_need)
    result = {name: float(mask.mean()) for name, mask in zip(QUADRANTS, masks)}
    result.update({name + "_count": int(mask.sum()) for name, mask in zip(QUADRANTS, masks)})
    result.update({"n_classes": len(reliable), "reliable_threshold": r_threshold,
                   "need_threshold": n_threshold})
    return result


def load_run(name: str, path: Path, args) -> dict:
    root, rd = resolve_roots(path)
    sources, notices, static, dynamic = [], [], [], []
    epoch_path = rd / "epoch_summary.csv"
    epochs = pd.DataFrame()
    if epoch_path.exists():
        epochs = read_csv(epoch_path, sources)
        required(epochs, ["step", "epoch"], str(epoch_path))
        numeric(epochs, ["step", "epoch"], str(epoch_path))
        unique(epochs, ["step", "epoch"], str(epoch_path))
        epochs = epochs[epochs["step"].apply(lambda t: keep_step(t, args))].copy()
        for c in ["cmr_tolerance", "rd_margin_tolerance"]:
            if c in epochs:
                epochs[c] = pd.to_numeric(epochs[c].replace("", np.nan), errors="raise")
    else:
        notices.append("No epoch_summary.csv: sample-exposure drift rates are unavailable.")

    for file in sorted(rd.glob("step_*_static_trust.csv")):
        match = re.fullmatch(r"step_(\d+)_static_trust\.csv", file.name)
        if not match or not keep_step(int(match[1]), args):
            continue
        df = read_csv(file, sources)
        cols = ["step", "class_id", "prototype_count"] + [
            p + "_" + d for p in ("reliability", "trust") for d in DIRECTIONS]
        required(df, cols + ["category_name"], str(file)); numeric(df, cols, str(file))
        unique(df, ["step", "class_id"], str(file))
        if not (df["step"] == int(match[1])).all():
            raise ValueError("Step mismatch in {}".format(file))
        ids = sorted(df["class_id"].tolist())
        if len(ids) < 2 or ids != list(range(len(ids))):
            raise ValueError("{}: expected a complete contiguous old-class bank".format(file))
        if (df["prototype_count"] <= 0).any():
            raise ValueError("{}: prototype count must be positive".format(file))
        for d in DIRECTIONS:
            if not df["reliability_" + d].between(0, 1).all():
                raise ValueError("{}: reliability outside [0,1]".format(file))
        df["num_old_classes"] = len(df)
        df["chance"] = 1.0 / len(df)
        for d in DIRECTIONS:
            df["chance_adjusted_" + d] = (
                (df["reliability_" + d] - df["chance"]) / (1.0 - df["chance"]))
            source_col = {"chance_adjusted": "chance_adjusted_", "raw": "reliability_",
                          "trust": "trust_"}[args.reliability_score] + d
            df["score_" + d] = df[source_col]
        static.append(df)

    for file in sorted(rd.glob("step_*_epoch_*_weights.csv")):
        match = re.fullmatch(r"step_(\d+)_epoch_(\d+)_weights\.csv", file.name)
        if not match or not keep_step(int(match[1]), args):
            continue
        df = read_csv(file, sources)
        cols = ["step", "epoch", "class_id"] + ["need_" + d for d in DIRECTIONS]
        required(df, cols, str(file)); numeric(df, cols, str(file))
        unique(df, ["step", "epoch", "class_id"], str(file))
        if not (df["step"] == int(match[1])).all() or not (df["epoch"] == int(match[2])).all():
            raise ValueError("Step/epoch mismatch in {}".format(file))
        if (df[["need_" + d for d in DIRECTIONS]] < 0).any().any():
            raise ValueError("{}: Need must be nonnegative".format(file))
        dynamic.append(df)

    st = pd.concat(static, ignore_index=True) if static else pd.DataFrame()
    dy = pd.concat(dynamic, ignore_index=True) if dynamic else pd.DataFrame()
    if st.empty:
        notices.append("No static Trust CSV: reliability and reliability/Need joint analyses unavailable.")
    if dy.empty:
        notices.append("No weight snapshots: class Need and quadrant proxies unavailable.")
    else:
        unique(dy, ["step", "epoch", "class_id"], "weight snapshots")
        for (step, epoch), group in dy.groupby(["step", "epoch"]):
            ids = sorted(group["class_id"].tolist())
            if ids != list(range(len(ids))):
                raise ValueError("Incomplete snapshot step={} epoch={}".format(step, epoch))
            if not st.empty:
                bank = st[st["step"] == step].sort_values("class_id")
                if not bank.empty:
                    if len(bank) != len(group):
                        raise ValueError("Bank/snapshot class count mismatch at step {}".format(step))
                    group = group.sort_values("class_id")
                    for d in DIRECTIONS:
                        col = "trust_" + d
                        if col in group and not np.allclose(
                                group[col].to_numpy(dtype=float), bank[col].to_numpy(dtype=float),
                                rtol=1e-5, atol=1e-7):
                            raise ValueError("Static/snapshot Trust mismatch; possibly mixed reruns")
        notices.append("Need=0 can also denote an unobserved class; per-class observation counts were not saved.")
    perf = pd.DataFrame()
    if (root / "per_class_metrics.csv").exists():
        perf = read_csv(root / "per_class_metrics.csv", sources)
        required(perf, ["step", "class_id", "f1", "forget_f1", "support"], "per_class_metrics.csv")
        numeric(perf, ["step", "class_id", "f1", "forget_f1", "support"], "per_class_metrics.csv")
        unique(perf, ["step", "class_id"], "per_class_metrics.csv")
    return dict(name=name, root=root, rd=rd, epochs=epochs, static=st, dynamic=dy,
                performance=perf, sources=sources, notices=notices)


def epoch_statistics(epochs: pd.DataFrame, args) -> pd.DataFrame:
    """Never interpret zeros from disabled CMR as observed no-drift."""
    rows = []
    for _, row in epochs.iterrows():
        mode = str(row.get("rd_mode", ""))
        enabled = mode in CMR_MODES
        tolerance = row.get("cmr_tolerance", row.get("rd_margin_tolerance", np.nan))
        if pd.isna(tolerance) and args.legacy_tolerance is not None:
            tolerance = args.legacy_tolerance
        known = pd.notna(tolerance) and np.isfinite(float(tolerance))
        if known and float(tolerance) < 0:
            raise ValueError("Negative saved tolerance")
        for d in DIRECTIONS:
            active = pd.to_numeric(row.get("cmr_active_" + d, np.nan), errors="coerce")
            mean_d = pd.to_numeric(row.get("mean_deficit_" + d, np.nan), errors="coerce")
            measured = enabled and pd.notna(active) and np.isfinite(float(active))
            if measured and not 0 <= active <= 1:
                raise ValueError("Invalid saved active fraction")
            exact = measured and known and float(tolerance) == 0.0
            status = ("not_measured" if not measured else "tolerance_unknown" if not known
                      else "exact_gt0" if exact else "only_gt_tolerance")
            rate = float(active) if measured else np.nan
            rows.append({
                "step": int(row["step"]), "epoch": int(row["epoch"]), "direction": d,
                "rd_mode": mode, "cmr_penalty": row.get("cmr_penalty", "unknown"),
                "cmr_scale": row.get("cmr_scale", np.nan),
                "tolerance": float(tolerance) if known else np.nan,
                "status": status, "observed_violation_fraction": rate,
                "drop_gt0_fraction": rate if exact else np.nan,
                "drop_gt0_lower_bound": rate if measured and known else np.nan,
                "mean_positive_deficit": float(mean_d) if measured and pd.notna(mean_d) else np.nan,
                "mean_drop_given_positive": (
                    float(mean_d) / rate if exact and rate > 0 and pd.notna(mean_d) else np.nan),
            })
    return pd.DataFrame(rows)


def build_tables(run: dict, args) -> Dict[str, pd.DataFrame]:
    st, dy = run["static"], run["dynamic"]
    tables = {"sample_drift_by_epoch": epoch_statistics(run["epochs"], args)}
    reliability, asym, need, quadrants, sensitivities = [], [], [], [], []
    if not st.empty:
        tables["static_class_scores"] = st.copy()
        for step, group in st.groupby("step"):
            for d in DIRECTIONS:
                info = {"step": int(step), "direction": d,
                        "reliability_score": args.reliability_score,
                        "reliable_threshold": args.reliable_threshold,
                        "above_chance_fraction": float((group["reliability_" + d] > group["chance"]).mean()),
                        "reliable_class_fraction": float((group["score_" + d] > args.reliable_threshold).mean()),
                        "raw_R_mean": group["reliability_" + d].mean(),
                        "raw_R_std": group["reliability_" + d].std(ddof=0),
                        "saved_Trust_std": group["trust_" + d].std(ddof=0)}
                info.update(distribution(group["score_" + d], args.reliability_gap))
                reliability.append(info)
            info = {"step": int(step), "epoch": -1, "metric": args.reliability_score}
            info.update(paired_stats(group["score_a_from_v"], group["score_v_from_a"], args.reliability_gap))
            info["binary_reliability_disagreement_fraction"] = float((
                (group["score_a_from_v"] > args.reliable_threshold)
                != (group["score_v_from_a"] > args.reliable_threshold)).mean())
            asym.append(info)
    if not dy.empty:
        for (step, epoch), group in dy.groupby(["step", "epoch"]):
            for d in DIRECTIONS:
                info = {"step": int(step), "epoch": int(epoch), "direction": d,
                        "high_need_class_fraction": float((group["need_" + d] > args.need_threshold).mean()),
                        "need_threshold": args.need_threshold,
                        "zero_need_fraction": float((group["need_" + d] == 0).mean())}
                info.update(distribution(group["need_" + d], args.need_gap)); need.append(info)
            info = {"step": int(step), "epoch": int(epoch), "metric": "need_ema"}
            info.update(paired_stats(group["need_a_from_v"], group["need_v_from_a"], args.need_gap))
            info["binary_need_disagreement_fraction"] = float((
                (group["need_a_from_v"] > args.need_threshold)
                != (group["need_v_from_a"] > args.need_threshold)).mean())
            asym.append(info)
    if not st.empty and not dy.empty:
        # Use explicit join keys. Never join on category names or row positions.
        cols = ["step", "class_id", "epoch", "need_a_from_v", "need_v_from_a"]
        joined = dy[cols].merge(st, on=["step", "class_id"], how="inner", validate="many_to_one")
        if len(joined) < len(dy):
            run["notices"].append("Some weight snapshots have no matching static bank; their joint analyses were skipped.")
        for d in DIRECTIONS:
            joined["reliable_" + d] = joined["score_" + d] > args.reliable_threshold
            joined["high_need_proxy_" + d] = joined["need_" + d] > args.need_threshold
        tables["class_epoch_joined"] = joined
        for (step, epoch), group in joined.groupby(["step", "epoch"]):
            for d in DIRECTIONS:
                base = {"step": int(step), "epoch": int(epoch), "direction": d,
                        "level": "class_ema_proxy", "reliability_score": args.reliability_score}
                info = dict(base)
                info.update(quadrant_summary(group["score_" + d], group["need_" + d],
                                             args.reliable_threshold, args.need_threshold))
                quadrants.append(info)
                # Threshold sensitivity is available for every saved snapshot.
                for r in sorted(set(args.reliability_thresholds + [args.reliable_threshold])):
                    for n in sorted(set(args.need_thresholds + [args.need_threshold])):
                        info = dict(base)
                        info.update(quadrant_summary(group["score_" + d], group["need_" + d], r, n))
                        sensitivities.append(info)
    tables["reliability_heterogeneity"] = pd.DataFrame(reliability)
    tables["need_heterogeneity"] = pd.DataFrame(need)
    tables["direction_asymmetry"] = pd.DataFrame(asym)
    tables["quadrants_class_ema"] = pd.DataFrame(quadrants)
    tables["quadrants_threshold_sensitivity"] = pd.DataFrame(sensitivities)
    if not st.empty and not run["performance"].empty:
        perf = run["performance"]
        # Static banks contain ONLY old classes. F1 is from the best checkpoint,
        # NOT necessarily the epoch represented by the last Need snapshot.
        joined = st.merge(perf[["step", "class_id", "f1", "forget_f1", "support"]],
                          on=["step", "class_id"], how="inner", validate="one_to_one")
        joined = joined[joined["support"] > 0].copy()
        tables["class_reliability_vs_test_forgetting"] = joined
        corr = []
        for step, group in joined.groupby("step"):
            for d in DIRECTIONS:
                corr.append({"step": int(step), "direction": d, "n_classes": len(group),
                             "spearman_score_vs_forget_f1": rank_corr(group["score_" + d], group["forget_f1"]),
                             "test_f1_drop_gt0_class_fraction": float((group["forget_f1"] > 0).mean()),
                             "note": "best-prior-F1 minus current best-checkpoint F1; NOT margin drift"})
        tables["test_forgetting_correlations"] = pd.DataFrame(corr)
    for frame in tables.values():
        if not frame.empty:
            frame.insert(0, "run", run["name"])
    return tables


def new_plot(title: str, xlabel: str, ylabel: str):
    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return fig, ax


def save_plot(fig, folder: Path, filename: str, gallery: List[Path]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = folder / (filename + ".png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    gallery.append(path)


def scatter_pair(a, b, title: str, xlabel: str, ylabel: str,
                 folder: Path, filename: str, gallery: List[Path]) -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    fig, ax = new_plot(title, xlabel, ylabel)
    ax.scatter(a, b, s=26, alpha=.8)
    low, high = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    pad = max((high - low) * .06, .005)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", label="Equal directions")
    ax.set_xlim(low - pad, high + pad); ax.set_ylim(low - pad, high + pad)
    ax.legend(fontsize=9)
    save_plot(fig, folder, filename, gallery)


def plot_run(run: dict, tables: Dict[str, pd.DataFrame], args, out: Path) -> List[Path]:
    gallery = []
    title_name = run["name"] if len(run["name"]) <= 52 else run["name"][:49] + "..."
    rates = tables["sample_drift_by_epoch"]
    if not rates.empty:
        for step, group in rates.groupby("step"):
            measured = group[group["status"] != "not_measured"]
            if measured.empty:
                continue
            prefix = "{} | step {}".format(title_name, step)
            folder = out / "figures" / "step_{}".format(step)
            exact = (measured["status"] == "exact_gt0").all()
            known = (measured["status"] != "tolerance_unknown").all()
            metric_title = ("P(Bref - Bcur > 0)" if exact else
                            "P(Bref - Bcur > saved tolerance)" if known else
                            "Saved active ratio (tolerance unknown)")
            fig, ax = new_plot(prefix + "\n" + metric_title + " | training exposures", "Epoch", "Fraction of replay exposures")
            for d in DIRECTIONS:
                part = measured[measured["direction"] == d].sort_values("epoch")
                ax.plot(part["epoch"], part["observed_violation_fraction"], label=DISPLAY[d])
            ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.legend()
            save_plot(fig, folder, "01_sample_drift_fraction", gallery)
            fig, ax = new_plot(prefix + "\nMean positive margin drop | training exposures", "Epoch", "Mean max(Bref - Bcur, 0)")
            for d in DIRECTIONS:
                part = measured[measured["direction"] == d].sort_values("epoch")
                ax.plot(part["epoch"], part["mean_positive_deficit"], label=DISPLAY[d])
            ax.set_ylim(bottom=0); ax.legend()
            save_plot(fig, folder, "02_mean_positive_deficit", gallery)
    st, dy = run["static"], run["dynamic"]
    if not st.empty:
        for step, group in st.groupby("step"):
            step = int(step); prefix = "{} | step {}".format(title_name, step)
            folder = out / "figures" / "step_{}".format(step)
            fig, ax = new_plot(prefix + "\nTeacher reliability heterogeneity (class ECDF)",
                               args.reliability_score + " score", "Fraction of old classes <= score")
            for d in DIRECTIONS:
                vals = np.sort(group["score_" + d].to_numpy())
                ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post", label=DISPLAY[d])
            ax.axvline(args.reliable_threshold, linestyle="--", label="Reliability threshold = {:g}".format(args.reliable_threshold))
            ax.set_ylim(0, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.legend(fontsize=9)
            save_plot(fig, folder, "03_reliability_distribution", gallery)
            scatter_pair(group["score_a_from_v"], group["score_v_from_a"],
                         prefix + "\nPaired teacher reliability: state, not causal effect",
                         DISPLAY["a_from_v"] + " " + args.reliability_score,
                         DISPLAY["v_from_a"] + " " + args.reliability_score,
                         folder, "04_direction_reliability", gallery)
    joined = tables.get("class_epoch_joined", pd.DataFrame())
    if not dy.empty:
        for step, group in dy.groupby("step"):
            step = int(step); prefix = "{} | step {}".format(title_name, step)
            folder = out / "figures" / "step_{}".format(step)
            chosen_epoch = int(group["epoch"].max()) if args.snapshot_epoch == "last" else int(args.snapshot_epoch)
            snap = group[group["epoch"] == chosen_epoch].sort_values("class_id")
            need_vmax = max(float(group[["need_" + d for d in DIRECTIONS]].max().max()), 1e-12)
            for d in DIRECTIONS:
                matrix = group.pivot(index="class_id", columns="epoch", values="need_" + d).sort_index().sort_index(axis=1)
                fig, ax = new_plot(prefix + "\n" + DISPLAY[d] + " class Need EMA (not sample drift rate)", "Saved epoch", "Old class ID")
                im = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=need_vmax)
                xticks = np.unique(np.linspace(0, matrix.shape[1] - 1, min(8, matrix.shape[1])).astype(int))
                yticks = np.unique(np.linspace(0, matrix.shape[0] - 1, min(15, matrix.shape[0])).astype(int))
                ax.set_xticks(xticks); ax.set_xticklabels([int(matrix.columns[i]) for i in xticks])
                ax.set_yticks(yticks); ax.set_yticklabels([int(matrix.index[i]) for i in yticks])
                fig.colorbar(im, ax=ax, label="Bias-corrected EMA of mean positive margin drop")
                save_plot(fig, folder, "09_need_heatmap_" + d, gallery)
            if snap.empty:
                run["notices"].append("Requested snapshot epoch {} absent at step {}; snapshot plots skipped.".format(chosen_epoch, step))
                continue
            label = prefix + " | epoch {}".format(chosen_epoch)
            scatter_pair(snap["need_a_from_v"], snap["need_v_from_a"],
                         label + "\nPaired class Need EMA: state, not causal effect",
                         "A<-V Need EMA", "V<-A Need EMA", folder, "08_direction_need_epoch_{}".format(chosen_epoch), gallery)
            sel = joined[(joined["step"] == step) & (joined["epoch"] == chosen_epoch)] if not joined.empty else pd.DataFrame()
            if sel.empty:
                continue
            fig, ax = new_plot(label + "\nFour CLASS-LEVEL groups (EMA proxy; not sample quadrants)\n" + "Reliability ({}) > {:g}; EMA Need > {:g}".format(args.reliability_score, args.reliable_threshold, args.need_threshold), "", "Fraction of old classes")
            xs = np.arange(4)
            for idx, d in enumerate(DIRECTIONS):
                q = quadrant_summary(sel["score_" + d], sel["need_" + d], args.reliable_threshold, args.need_threshold)
                values = [q[k] for k in QUADRANTS]
                bars = ax.bar(xs + (idx - .5) * .36, values, width=.36, label=DISPLAY[d])
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + .014,
                            "{:.1%}".format(val), ha="center", va="bottom", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(["Reliable\nHigh EMA Need", "Reliable\nLow EMA Need",
                                "Unreliable\nHigh EMA Need", "Unreliable\nLow EMA Need"], fontsize=9)
            ax.set_ylim(0, 1.15); ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.legend(fontsize=9)
            save_plot(fig, folder, "05_quadrants_class_ema_epoch_{}".format(chosen_epoch), gallery)
            for d in DIRECTIONS:
                fig, ax = new_plot(label + "\n" + DISPLAY[d] + " reliability vs class Need EMA", args.reliability_score + " reliability score", "Class Need EMA (not drift probability)")
                ax.scatter(sel["score_" + d], sel["need_" + d], s=27, alpha=.8)
                ax.axvline(args.reliable_threshold, linestyle="--", label="Reliability threshold = {:g}".format(args.reliable_threshold))
                ax.axhline(args.need_threshold, linestyle=":", label="Need threshold = {:g}".format(args.need_threshold))
                ax.legend(fontsize=9)
                save_plot(fig, folder, "06_reliability_need_{}_epoch_{}".format(d, chosen_epoch), gallery)
    need = tables["need_heterogeneity"]
    if not need.empty:
        for step, group in need.groupby("step"):
            folder = out / "figures" / "step_{}".format(step)
            fig, ax = new_plot("{} | step {}\nBetween-class heterogeneity of Need EMA".format(title_name, step), "Epoch", "Standard deviation across old classes")
            for d in DIRECTIONS:
                part = group[group["direction"] == d].sort_values("epoch")
                ax.plot(part["epoch"], part["std"], label=DISPLAY[d])
            ax.set_ylim(bottom=0); ax.legend()
            save_plot(fig, folder, "10_need_heterogeneity", gallery)
    perf = tables.get("class_reliability_vs_test_forgetting", pd.DataFrame())
    if not perf.empty:
        for step, group in perf.groupby("step"):
            fig, ax = new_plot("{} | step {}\nTeacher reliability vs TEST F1 forgetting (not margin drift)".format(title_name, step), args.reliability_score + " reliability score", "Best prior F1 - current best-checkpoint F1")
            for d, marker in zip(DIRECTIONS, ("o", "x")):
                ax.scatter(group["score_" + d], group["forget_f1"], s=27, marker=marker, alpha=.8, label=DISPLAY[d])
            ax.axhline(0, linestyle="--"); ax.legend()
            save_plot(fig, out / "figures" / "step_{}".format(step), "11_test_forgetting_not_margin", gallery)
    return gallery


def latest_per_step(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if df.empty or "epoch" not in df:
        return df
    return df.sort_values("epoch").groupby(keys, as_index=False).tail(1)


def write_report(run: dict, tables: dict, gallery: List[Path], args, out: Path) -> None:
    warnings = list(run["notices"])
    if args.reliability_score == "trust":
        warnings.append("Saved Trust is shrunk; Trust>0 does NOT necessarily mean raw R>chance.")
    if args.need_threshold == 0:
        warnings.append("Need>0 may persist after earlier drift; it does NOT identify current-epoch drifting classes.")
    notes = [
        "所有比例在 CSV 中为 0--1，图中按百分比显示。",
        "Q1 是训练期间 replay exposure 的 batch 等权平均；固定 batch size 下等于 exposure 加权平均，不是去重样本比例或 epoch 末评估。",
        "仅 tolerance=0 且 CMR 已测量时，cmr_active 才等于 P(Bref-Bcur>0)。非零 tolerance 不冒充零阈值比例。",
        "四象限的单位是类别：reliability 阈值 × 类别 Need EMA 阈值；不是逐样本可靠性×逐样本退化。",
        "Need=0 也可能由未观测类别产生，原 CSV 没有保存 epoch_count 或 need_steps，无法进一步区分。",
        "可靠性默认 score=(R-1/C)/(1-1/C)，不裁剪、不 shrink；score>0 仅表示超过均匀概率基线，不是统计显著性或高准确率。",
        "方向不对称是 prototype-space 状态差异，不是 CrossSDC loss/gradient 或因果收益差异。",
        "不同 step 的旧类竞争集合与 teacher 不同；不要把跨 step 原始 margin 尺度变化直接归因为遗忘。",
        "四组新实验是在不同 CMR objective 下观测到的轨迹，不是未经修改的 CrossSDC baseline 的反事实。",
        "这里的两个 CMR query 方向，不等同于原 CrossSDC 第二项 old_audio->cur_visual 的 anchor/reduction 方向。",
        "均值、标准差和阈值比例是描述性统计，不是拒绝零假设的显著性检验。",
    ]
    parts = ["<!doctype html><meta charset='utf-8'><title>CrossSDC diagnostics</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 18px;line-height:1.6}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid}section{overflow-x:auto;margin-bottom:32px}</style>",
             "<h1>{}</h1>".format(html.escape(run["name"])), "<h2>Interpretation / 口径</h2><ul>"]
    parts += ["<li>{}</li>".format(html.escape(s)) for s in notes + warnings]
    parts += ["</ul><h2>Analysis settings</h2><pre>{}</pre>".format(html.escape(json.dumps({
        "reliability_score": args.reliability_score, "reliable_threshold": args.reliable_threshold,
        "need_threshold": args.need_threshold, "reliability_gap": args.reliability_gap,
        "need_gap": args.need_gap, "snapshot_epoch": args.snapshot_epoch,
    }, ensure_ascii=False, indent=2)))]
    report_tables = {
        "Q1 — latest saved epoch per step and direction": latest_per_step(tables["sample_drift_by_epoch"], ["step", "direction"]),
        "Q2 — latest class-level EMA quadrants": latest_per_step(tables["quadrants_class_ema"], ["step", "direction"]),
        "H1 — between-class teacher reliability": tables["reliability_heterogeneity"],
        "H2 — latest between-class Need": latest_per_step(tables["need_heterogeneity"], ["step", "direction"]),
        "H3 — direction differences (state only)": latest_per_step(tables["direction_asymmetry"], ["step", "metric"]),
    }
    for title, frame in report_tables.items():
        parts.append("<section><h2>{}</h2>".format(html.escape(title)))
        parts.append("<p>Unavailable from these inputs.</p>" if frame.empty else frame.to_html(index=False, float_format=lambda x: "{:.5g}".format(x)))
        parts.append("</section>")
    parts.append("<h2>CSV tables</h2><ul>")
    for filename, frame in tables.items():
        if not frame.empty:
            parts.append("<li><a href='tables/{0}.csv'>{0}.csv</a></li>".format(filename))
    parts.append("</ul><h2>Figures</h2>")
    for path in gallery:
        rel = path.relative_to(out).as_posix()
        parts.append("<section><h3>{}</h3><img loading='lazy' src='{}'></section>".format(html.escape(path.stem), html.escape(rel)))
    (out / "report.html").write_text("\n".join(parts), encoding="utf-8")
    metadata = {"run": run["name"], "input_root": str(run["root"]), "source_files": run["sources"],
                "definitions": notes, "warnings": warnings,
                "settings": {k: v for k, v in vars(args).items() if k not in ("step_set",)},
                "tables": {k: len(v) for k, v in tables.items()}, "png_count": len(gallery)}
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Run: " + run["name"], "Report: " + str(out / "report.html")]
    q1 = report_tables["Q1 — latest saved epoch per step and direction"]
    if not q1.empty:
        lines.append(q1[["step", "epoch", "direction", "status", "drop_gt0_fraction", "observed_violation_fraction"]].to_string(index=False))
    lines.extend("NOTE: " + s for s in warnings)
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def compare_runs(results: List[Tuple[dict, dict]], args, root: Path) -> None:
    for table_name in ("sample_drift_by_epoch", "reliability_heterogeneity", "need_heterogeneity",
                       "direction_asymmetry", "quadrants_class_ema"):
        frames = [tables[table_name] for _, tables in results if not tables[table_name].empty]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(root / ("all_runs_" + table_name + ".csv"), index=False)
        if table_name != "sample_drift_by_epoch" or args.no_plots or len(results) < 2:
            continue
        for (step, d), group in combined.groupby(["step", "direction"]):
            exact = group[group["status"] == "exact_gt0"]
            if exact.empty:
                continue
            fig, ax = new_plot("Across runs | step {} | {}\nP(Bref-Bcur>0), training exposures; not a causal test".format(step, DISPLAY[d]), "Epoch", "Fraction of replay exposures")
            for name, part in exact.groupby("run", sort=False):
                part = part.sort_values("epoch")
                ax.plot(part["epoch"], part["drop_gt0_fraction"], label=name if len(name) <= 45 else name[:42] + "...")
            ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.legend(fontsize=8)
            save_plot(fig, root / "comparison", "drift_step_{}_{}".format(step, d), [])


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", type=Path, help="One save/metrics/EXPERIMENT or rd_crosssdc directory")
    p.add_argument("--label", help="Short name for --metrics-dir, e.g. hinge")
    p.add_argument("--run", action="append", default=[], metavar="NAME=PATH", help="Repeat for multi-run analysis")
    p.add_argument("--out", type=Path, required=True, help="Separate output directory (no source files changed)")
    p.add_argument("--steps", default="all", help="Zero-based incremental step IDs: all or 1,4,9")
    p.add_argument("--snapshot-epoch", default="last", help="Last saved epoch per step, or an explicit epoch; affects snapshot figures only")
    p.add_argument("--reliability-score", choices=["chance_adjusted", "raw", "trust"], default="chance_adjusted")
    p.add_argument("--reliable-threshold", type=float, default=None, help="Strict > threshold; default 0 for chance_adjusted/trust, 0.5 for raw")
    p.add_argument("--need-threshold", type=float, default=0.0, help="Strict > threshold for CLASS EMA proxy, not sample drift")
    p.add_argument("--reliability-gap", type=float, default=.05, help="Descriptive heterogeneity/asymmetry threshold, in chosen score units")
    p.add_argument("--need-gap", type=float, default=.1, help="Descriptive heterogeneity/asymmetry threshold in margin units")
    p.add_argument("--reliability-thresholds", type=float, nargs="+", default=[0.0, .1, .25, .5], help="Quadrant sensitivity grid, in chosen score units")
    p.add_argument("--need-thresholds", type=float, nargs="+", default=[0.0, .01, .1, .5], help="Quadrant sensitivity grid in margin units")
    p.add_argument("--legacy-tolerance", type=float, help="Explicit fallback for old CSVs with no saved tolerance; never overrides a known tolerance")
    p.add_argument("--no-plots", action="store_true", help="CSV/HTML/txt only")
    return p


def main(argv=None) -> int:
    p = make_parser(); args = p.parse_args(argv)
    try:
        args.step_set = parse_steps(args.steps)
        if args.snapshot_epoch != "last" and int(args.snapshot_epoch) < 0:
            raise ValueError("Snapshot epoch must be nonnegative")
        if args.reliable_threshold is None:
            args.reliable_threshold = .5 if args.reliability_score == "raw" else 0.0
        for key in ("need_threshold", "reliability_gap", "need_gap"):
            val = getattr(args, key)
            if not np.isfinite(val) or val < 0:
                raise ValueError("{} must be finite and nonnegative".format(key))
        for val in [args.reliable_threshold] + args.reliability_thresholds + args.need_thresholds:
            if not np.isfinite(val):
                raise ValueError("Thresholds must be finite")
        if min(args.need_thresholds) < 0:
            raise ValueError("Need thresholds must be nonnegative")
        if args.legacy_tolerance is not None and (not np.isfinite(args.legacy_tolerance) or args.legacy_tolerance < 0):
            raise ValueError("Legacy tolerance must be finite and nonnegative")
        specs = []
        if args.metrics_dir is not None:
            default_name = args.metrics_dir.parent.name if args.metrics_dir.name == "rd_crosssdc" else args.metrics_dir.name
            specs.append((args.label or default_name, args.metrics_dir))
        for spec in args.run:
            if "=" not in spec:
                raise ValueError("--run requires NAME=PATH")
            name, path = spec.split("=", 1)
            specs.append((name, Path(path)))
        if not specs:
            p.error("Provide --metrics-dir or at least one --run NAME=PATH")
        slugs = [re.sub(r"[^A-Za-z0-9_.-]", "_", name) for name, _ in specs]
        if len(set(slugs)) != len(slugs) or any(s in ("", ".", "..") for s in slugs):
            raise ValueError("Run labels must produce distinct nonempty safe folder names")
        root = args.out.expanduser().resolve()
        # Refuse to mix analysis products with original metrics directories.
        for _, path in specs:
            metrics_root, rd_root = resolve_roots(path)
            if root == metrics_root or root == rd_root:
                raise ValueError("--out must be a separate analysis directory")
        root.mkdir(parents=True, exist_ok=True)
        results, links = [], []
        for (name, path), slug in zip(specs, slugs):
            run = load_run(name, path, args)
            tables = build_tables(run, args)
            out = root / slug; (out / "tables").mkdir(parents=True, exist_ok=True)
            for filename, frame in tables.items():
                if not frame.empty:
                    frame.to_csv(out / "tables" / (filename + ".csv"), index=False)
            gallery = [] if args.no_plots else plot_run(run, tables, args, out)
            write_report(run, tables, gallery, args, out)
            results.append((run, tables)); links.append((slug, name))
        compare_runs(results, args, root)
        body = ["<!doctype html><meta charset='utf-8'><h1>CrossSDC diagnostic reports</h1>",
                "<p>Each report distinguishes exact exposure rates, class EMA proxies, and unavailable causal claims.</p>"]
        for slug, name in links:
            body.append("<p><a href='{}/report.html'>{}</a></p>".format(html.escape(slug), html.escape(name)))
        if (root / "comparison").exists() and not args.no_plots and len(results) > 1:
            body.append("<p>Comparison PNGs are in comparison/. CSV summaries are all_runs_*.csv.</p>")
        (root / "index.html").write_text("\n".join(body), encoding="utf-8")
        print("\nOpen: {}".format(root / "index.html"))
        return 0
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        p.exit(2, "ERROR: {}\n".format(exc))


if __name__ == "__main__":
    sys.exit(main())
