"""Read-only analysis of the supplied summary; write derived tables and figures here.

Accuracy columns use percent; differences use percentage points. AIA is the
unweighted mean of steps 0--9, computed only for seeds with all ten results.
Paired contrasts never impute missing observations or mix seed identities.
"""
from pathlib import Path
import collections
import csv
import hashlib
import html
import json
import re
import statistics as stats

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / "results" / "summary.csv"
source_bytes = SOURCE.read_bytes()
with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
    source_rows = list(csv.DictReader(handle))
seed_rows = [r for r in source_rows if r["row_type"] == "seed"]
groups = collections.defaultdict(dict)
params = {}
seed_table = []
for row in seed_rows:
    ident = re.search(r"_grid_(.+?)_lc", row["experiment"])[1]
    seed = int(row["seed"])
    assert seed not in groups[ident], (ident, seed)
    values = [float(row[f"step_{i}"]) * 100 if row[f"step_{i}"] else None for i in range(10)]
    assert all(v is None or 0 <= v <= 100 for v in values)
    groups[ident][seed] = values
    params[ident] = {k: row[k] for k in ("experiment", "rd_cmr_penalty", "lam_cmr", "rd_class_weight_alpha", "rd_cmr_scale")}
    present = [i for i, v in enumerate(values) if v is not None]
    assert present == list(range(len(present))), (ident, seed, "non-prefix results")
    seed_table.append(dict(config=ident, seed=seed, status=row["status"], observed_steps=len(present),
                           final_accuracy_pct=values[9],
                           aia_0_9_pct=stats.mean(values) if len(present) == 10 else None))

def mean_sd(values):
    return (stats.mean(values) if values else None,
            stats.stdev(values) if len(values) > 1 else None)

def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

config_table = []
for ident, runs in groups.items():
    finals = [v[9] for v in runs.values() if v[9] is not None]
    full = [stats.mean(v) for v in runs.values() if all(x is not None for x in v)]
    fa, fs = mean_sd(finals)
    aa, ass = mean_sd(full)
    config_table.append(dict(config=ident, **params[ident], n_final=len(finals),
                             final_accuracy_pct=fa, final_sd_pp=fs,
                             n_full_trajectory=len(full), aia_0_9_pct=aa, aia_sd_pp=ass))

# Verify the supplied aggregate rows against the seed-level input (six-decimal rounding).
aggregate_error = 0.0
for row in source_rows:
    if row["row_type"] not in ("mean", "std"):
        continue
    ident = re.search(r"_grid_(.+?)_lc", row["experiment"])[1]
    for step in range(10):
        values = [v[step] for v in groups[ident].values() if v[step] is not None]
        assert int(row[f"n_step_{step}"]) == len(values)
        expected = mean_sd(values)[row["row_type"] == "std"]
        actual = row[f"step_{step}"]
        assert bool(actual) == (expected is not None)
        if actual:
            aggregate_error = max(aggregate_error, abs(float(actual) * 100 - expected))
assert aggregate_error < 0.00011, aggregate_error

pairs = [("g010", "g001"), ("g011", "g002"), ("g010", "g011"),
         ("g008", "g005"), ("g008", "g002"), ("g009", "g006"),
         ("g007", "g004"), ("g008", "g007"), ("g008", "g009"),
         ("g005", "g004"), ("g002", "g001")]
paired_table = []
for a, b in pairs:
    for label, steps in [("final_step_9", [9]), ("aia_0_9", list(range(10))),
                         ("mean_steps_1_8", list(range(1, 9))), ("mean_steps_1_3", [1, 2, 3])]:
        differences = {}
        for seed in groups["exp_" + a]:
            av, bv = groups["exp_" + a][seed], groups["exp_" + b][seed]
            if all(av[i] is not None and bv[i] is not None for i in steps):
                differences[seed] = stats.mean(av[i] - bv[i] for i in steps)
        mean, sd = mean_sd(list(differences.values()))
        paired_table.append(dict(candidate=a, comparator=b, metric=label,
                                 n_paired=len(differences), mean_delta_pp=mean,
                                 sd_delta_pp=sd, per_seed_delta_pp=json.dumps(differences)))

# Archived hinge runs are a separate cohort: do not pool them with current exp runs.
archive = ROOT / "save_commands_cmr_hinge_tolerance_focus_3seeds" / "metrics"
if __import__("os").name == "nt":
    archive = Path("\\\\?\\" + str(archive))
hinge_groups = collections.defaultdict(list)
hinge_step0 = collections.defaultdict(set)
for run in sorted(archive.iterdir()):
    with (run / "per_class_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ab, tol = re.search(r"_hinge_([AB])_.+?_tol([^_]+)_", run.name).groups()
    seed = int(re.search(r"_seed(\d+)$", run.name)[1])
    values = [100 * next(float(r["overall_acc"]) for r in rows if int(r["step"]) == i) for i in range(10)]
    hinge_step0[seed].add(round(values[0], 5))
    final = [r for r in rows if r["step"] == "9"]
    def accuracy(subset):
        return 100 * sum(int(r["tp"]) for r in subset) / sum(int(r["support"]) for r in subset)
    hinge_groups[(ab, float(tol.replace("p", ".")))].append(dict(seed=seed, faa=values[9], aia=stats.mean(values),
        old=accuracy([r for r in final if int(r["class_id"]) < 90]),
        new=accuracy([r for r in final if int(r["class_id"]) >= 90])))
hinge_table = []
for (ab, tol), runs in sorted(hinge_groups.items()):
    mean, sd = mean_sd([r["faa"] for r in runs])
    hinge_table.append(dict(cohort="archived_hinge", group=ab, lam_cmr=.03 if ab == "A" else .1,
        alpha=.5 if ab == "A" else 1, tolerance=tol, n=3, final_accuracy_pct=mean, final_sd_pp=sd,
        aia_0_9_pct=stats.mean(r["aia"] for r in runs), old_accuracy_step9_pct=stats.mean(r["old"] for r in runs),
        new_accuracy_step9_pct=stats.mean(r["new"] for r in runs)))

write_csv("exp_config_summary.csv", config_table)
write_csv("exp_seed_metrics.csv", seed_table)
write_csv("paired_comparisons.csv", paired_table)
write_csv("archived_hinge_reference.csv", hinge_table)
audit = dict(source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    planned_configs=len(groups), planned_runs=len(seed_rows),
    statuses=dict(collections.Counter(r["status"] for r in seed_rows)),
    configs_with_observations=sum(any(v[0] is not None for v in rs.values()) for rs in groups.values()),
    configs_complete_three_seeds=sum(all(all(x is not None for x in v) for v in rs.values()) for rs in groups.values()),
    aggregate_reconstruction_max_error_pp=aggregate_error,
    exp_step0_pct={s: sorted({v[0] for runs in groups.values() for seed, v in runs.items() if seed == s and v[0] is not None}) for s in [42,43,44]},
    archived_hinge_step0_pct={s: sorted(v) for s,v in hinge_step0.items()},
    interpretation="Descriptive, exploratory results; SD across seeds is not a confidence interval. Missing runs are not failed runs. No imputation. No current-exp checkpoints, raw logs or feature stores found locally.")
(OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.05, 1.4]}, layout="constrained")
for ident, color in [("exp_g010", "#007C83"), ("exp_g008", "#CB7029"), ("exp_g003", "#4169A1"), ("exp_g007", "#89728E")]:
    arr = np.array(list(groups[ident].values()), dtype=float)
    mean, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
    p = params[ident]
    label = f"{ident[4:]}: lambda={p['lam_cmr']}, alpha={p['rd_class_weight_alpha']}, s={p['rd_cmr_scale']}"
    axes[0].plot(range(10), mean, marker="o", ms=4, color=color, label=label)
    axes[0].fill_between(range(10), mean-sd, mean+sd, color=color, alpha=.10)
axes[0].set(title="All-seen test accuracy: complete 3-seed configurations", xlabel="Incremental step", ylabel="Accuracy (%)", xticks=range(10))
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(alpha=.16)
available = sorted([r for r in config_table if r["n_final"]], key=lambda r:r["final_accuracy_pct"], reverse=True)
for y, row in enumerate(available):
    runs = groups[row["config"]]
    for seed, color in [(42,"#4269AD"),(43,"#D88930"),(44,"#44966C")]:
        if runs[seed][9] is not None:
            axes[1].scatter(runs[seed][9], y, color=color, s=35, label=f"Seed {seed}" if y == 0 else None)
    axes[1].scatter(row["final_accuracy_pct"], y, marker="D", s=50,
        facecolors="black" if row["n_final"] == 3 else "none", edgecolors="black", zorder=5)
labels = [f"{r['config'][4:]}  lambda={r['lam_cmr']}, alpha={r['rd_class_weight_alpha']}, s={r['rd_cmr_scale']}  (n={r['n_final']})" for r in available]
axes[1].set(yticks=range(len(available)), yticklabels=labels, xlabel="Final-step accuracy (%)", title="Observed final scores; unequal n is explicit")
axes[1].invert_yaxis()
axes[1].legend(loc="lower right", fontsize=8)
axes[1].grid(axis="x", alpha=.16)
fig.suptitle("Exp-CMR results | bands = sample SD; diamonds = observed means, hollow when n < 3", fontsize=13)
fig.savefig(OUT / "exp_overview.png")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
for a,b,seeds,stop,label in [("g010","g001",[42,43,44],9,"s=5: 3 paired seeds, steps 0-8"),
                            ("g011","g002",[42,43],10,"s=1: 2 paired seeds, steps 0-9")]:
    arr=np.array([[groups['exp_'+a][s][i]-groups['exp_'+b][s][i] for i in range(stop)] for s in seeds])
    axes[0].plot(range(stop),arr.mean(0),marker="o",ms=4,label=label)
axes[0].set(title="Lower lambda: 0.03 minus 0.1 (alpha=0.5)",xlabel="Step",ylabel="Paired accuracy difference (pp)",xticks=range(10))
axes[0].legend(fontsize=7)
for ax, comparisons, title in [
    (axes[1],[("g008","g005","s=1"),("g009","g006","s=2"),("g007","g004","s=5")],"Alpha=1 minus alpha=0 (lambda=0.1)"),
    (axes[2],[("g005","g004","alpha=0"),("g008","g007","alpha=1")],"Scale=1 minus scale=5 (lambda=0.1)")]:
    labels=[]
    for i,(a,b,label) in enumerate(comparisons):
        row=next(r for r in paired_table if r['candidate']==a and r['comparator']==b and r['metric']=='final_step_9')
        differences=list(json.loads(row['per_seed_delta_pp']).values())
        ax.scatter([i]*len(differences),differences,s=35,color="#4269AD")
        ax.scatter(i,stats.mean(differences),marker="D",color="black",s=45)
        labels.append(label+f"\npaired n={len(differences)}")
    ax.set(title=title,xticks=range(len(labels)),xticklabels=labels,ylabel="Final accuracy difference (pp)")
for ax in axes:
    ax.axhline(0,color="gray",lw=.8,linestyle="--")
    ax.grid(axis="y",alpha=.15)
fig.suptitle("Matched seed comparisons: dots are seeds; no missing result is imputed",fontsize=13)
fig.savefig(OUT / "exp_paired_comparisons.png")
plt.close(fig)

def html_table(rows, fields):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape('—' if r[k] is None else f'{r[k]:.3f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table>'
report = '''<!doctype html><meta charset="utf-8"><title>CMR experiment analysis</title>
<style>body{font:16px system-ui;max-width:1250px;margin:36px auto;padding:0 24px;color:#183040}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dde3e8;text-align:right}th{background:#edf5f5}p{line-height:1.65}</style>
<aside style="padding:20px;background:#fff0ce;border:2px solid #b7791f"><strong>已被更正报告替代。</strong>本页仅分析 results 顶层不完整的 exp 汇总，不能代表 results 全部实验。本文候选排序及补实验建议不再作为当前建议。请阅读 <a href="../cmr_selected_analysis/report.html">四个子文件夹的完整重新分析</a>。</aside>
<h1>历史分析：顶层 exp 汇总（范围不完整）</h1><p>此处的顶层 summary 仅包含 exp：27 个配置、81 次计划运行，27 次完整、5 次部分完成、49 次缺失。缺失不等于失败。FAA 为 step 9 准确率；AIA 为同一 seed 的 step 0–9 简单平均，仅纳入完整轨迹。标准差为 seed 间样本标准差，不是置信区间。</p>
<p>λ=0.03 是目前最一致的候选方向；g010 当前平均最好，g011 的 scale 对照仍需补齐。α 的效果不单调。当前命令采用原有 offset、shrinkage 和权重裁剪；不能据此评价新加入的未裁剪公式。</p>
<img src="exp_overview.png"><img src="exp_paired_comparisons.png"><h2>当前 exp 配置（百分比）</h2>'''
report += html_table(available,["config","lam_cmr","rd_class_weight_alpha","rd_cmr_scale","n_final","final_accuracy_pct","final_sd_pp","aia_0_9_pct","aia_sd_pp"])
report += '<h2>旧 hinge 存档：单独参照</h2><p>同 seed 的 step 0 与当前 exp 不完全一致，不能作为严格的 penalty 单变量对照。A 组 λ=.03/α=.5，B 组 λ=.1/α=1；组间同时改变两个参数。</p>'
report += html_table(hinge_table,["group","tolerance","final_accuracy_pct","final_sd_pp","aia_0_9_pct","old_accuracy_step9_pct","new_accuracy_step9_pct"])
report += '''<h2>参数解释与后续分析</h2>
<p>固定 alpha=.5，scale=5 时 lambda=.03 相对 .1 在三个匹配 seed 的 step 1–8 平均提高 1.562 个百分点；scale=1 时两个完整匹配 seed 的末阶段分别提高 1.58、1.74 个百分点。scale=1 在 lambda=.1 下优于 scale=5 的证据较一致，但在 lambda=.03 下不能据现有两个 seed 排除 scale=1。alpha=1 相对 alpha=0 的改善并不一致。</p>
<p>Exp 单样本目标为 s*exp(g/s)，g=B_ref-B_cur-tolerance，关于当前 margin 的导数为 -exp(g/s)。g=0 时梯度不随 scale 改变，损失值却等于 s；因此不能用原始 loss 或 weighted_cmr/train_loss 大小直接比较约束强度。较低 lambda 可能缓解对旧表示的持续推动与新类学习的冲突，但这是待用梯度与 old/new accuracy 验证的机制假设。</p>
<ol><li>优先补齐 g001 的 seed42/43、g002/g005 的 seed44，以及 g011 的 seed44。先检查远端日志、完整检查点和评估是否已有结果，不把缺失当训练失败；当前程序没有完整训练状态恢复机制。</li>
<li>分析对照：g010 vs g001 隔离 lambda；g010 vs g011 隔离 scale；g005/g002/g008 隔离 alpha。保留同 seed 对照，并分解 task×step 准确率矩阵、旧类/新类准确率、旧→新和新→旧误分。</li>
<li>检查两个模态方向的 signed margin gap 分位数、违反比例、低/高 Trust 类别差异。Need 是类级 EMA，并非样本退化率。用固定 probe 样本集比较最佳模型，并区分每个实验自己的教师参照与共享教师参照。</li>
<li>在固定 batch 上测量 CMR 与其余训练损失的梯度范数比例、余弦相似度，以及 exp 梯度是否集中于少数严重退化样本。同步查看实际权重的离散程度，而不只看 alpha。</li>
<li>补足同协议基线：关闭 CMR 且不加 CrossSDC-C；保留原 CrossSDC-C；匹配超参数的 hinge。无这些对照，当前结果只能说明 exp 内部的参数差异。</li></ol>
<p>检查点加特征数据可以重新计算预测、逐类指标、特征与固定 probe 上的梯度。原始 CMR 原型/LOO 还依赖当阶段的 replay IDs；相同 seed 只有在代码、样本顺序、特征可用性和随机数消耗一致时才可能复原。最佳检查点不能恢复所有 epoch 的优化器状态、Need 轨迹或历史 batch 统计。当前本地没有 exp 对应 metrics/检查点、原始日志或特征文件，已有数据仅为旧 hinge 存档。</p>
<p>后续保存 git commit、完整参数、类别映射和数据版本、每阶段 replay IDs、prototype bank、best_epoch、固定 probe IDs 和逐样本预测。新加入的关闭权重裁剪公式应单独命名对照；它要求 alpha&lt;1，不能直接沿用旧路径的 alpha=1。</p>'''
report += '<p>可下载同目录中的 exp_config_summary.csv、exp_seed_metrics.csv、paired_comparisons.csv、archived_hinge_reference.csv；audit.json 记录输入哈希和核对信息。analyze_results.py 可重新生成所有结果。原始 summary 文件未修改。</p>'
(OUT / "report.html").write_text(report,encoding="utf-8")
assert SOURCE.read_bytes() == source_bytes
print(json.dumps(audit,indent=2))
print('Artifacts:', OUT)
