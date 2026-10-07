"""Generate separate Bash command files for periodic fusion and AVCIL CL history."""

import argparse
from pathlib import Path
import shlex

# [P9 新增] 生成融合实验的命令矩阵；输出沿用 P8 grid_commands 的“一行一个实验”形式，
# 并非复制 P8 训练函数，也不复用 RD/CMR 参数网格。

SETTINGS = {
    "periodic_fixed": ("periodic", "fixed"),
    "periodic_sample_aware": ("periodic", "sample_aware"),
    "uniform_cl_history": ("uniform", None),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_root", default="../../../datasets/VGGSound")
    parser.add_argument("--meta_root", default="../data2/balance")
    parser.add_argument("--dataset", default="VGGSound_random_balance_crosssdc_z1_seed42")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max_epoches", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--name_prefix", default="phase9")
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS),
                        help="Generate selected command files only (default: all three settings)")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "grid_commands"))
    args = parser.parse_args(argv)
    if args.max_epoches <= 0 or args.num_workers < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs must be positive, workers nonnegative, and seeds unique")
    if len(set(args.settings)) != len(args.settings):
        parser.error("settings must be unique")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in args.settings:
        mode, rule = SETTINGS[label]
        lines = []
        for seed in args.seeds:
            name = f"{args.name_prefix}_{label}_h{args.max_epoches}_seed{seed}"
            command = ["python", "-u", "train_incremental_fusion_modular.py", "--dataset", args.dataset,
                       "--feature_root", args.feature_root, "--meta_root", args.meta_root,
                       "--num_classes", "100", "--class_num_per_step", "10", "--memory_size", "500",
                       "--max_epoches", str(args.max_epoches), "--num_workers", str(args.num_workers),
                       "--train_batch_size", "128", "--infer_batch_size", "32", "--exemplar_batch_size", "128",
                       "--lr", "1e-3", "--weight_decay", "1e-4", "--lr_decay", "False", "--milestones", "100",
                       "--instance_contrastive", "--class_contrastive", "--attn_score_distil",
                       "--instance_contrastive_temperature", "0.05", "--class_contrastive_temperature", "0.05",
                       "--lam", "0.5", "--lam_I", "0.1", "--lam_C", "1.0"]
            if mode == "periodic":
                command += ["--fusion_mode", mode, "--fusion_update_rule", rule, "--fusion_classifier", "linear",
                            "--fusion_warmup_epochs", "40", "--fusion_update_interval", "40",
                            "--fusion_eta_max", "0.5", "--fusion_n_ref", "10", "--fusion_temperature", "0.1"]
            else:
                # [P9 新增] 等权 AVCIL 只记录历史；不生成任何 gate 更新参数。
                # 温度/最少样本数/batch size 用于 CL 原型观测，与动态融合实验保持一致。
                command += ["--fusion_mode", "uniform", "--fusion_classifier", "linear",
                            "--record_cl_history", "--cl_history_interval", "40",
                            "--fusion_temperature", "0.1", "--fusion_min_samples", "2", "--fusion_batch_size", "128"]
            command += ["--seed", str(seed), "--experiment_name", name, "--device", "cuda"]
            lines.append(shlex.join(command) + " > " + shlex.quote(f"logs/{name}.log") + " 2>&1")
        path = output_dir / f"commands_{label}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {len(lines)} commands to {path}")


if __name__ == "__main__":
    main()
