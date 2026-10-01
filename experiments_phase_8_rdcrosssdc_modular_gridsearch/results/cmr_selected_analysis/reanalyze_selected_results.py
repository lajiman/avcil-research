"""Analyze exactly the four user-specified result folders; never use root summary.

Source CSV/Markdown files are immutable inputs. All accuracy values below use
percent; paired differences use percentage points. AIA is mean(step_0..step_9)
within one complete seed, followed by averaging across seeds. No imputation.
"""
from pathlib import Path
import collections
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics as stats

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
ROOT = RESULTS.parent
FOLDERS = ["logs_direct", "logs_exp_selected", "logs_hinge_selected", "logs_log1p_selected"]
original_bytes = {}
runs, configs, sources, errors = [], {}, [], []
max_aggregate_error = 0.0

def digest(data):
    return hashlib.sha256(data).hexdigest()

def mean_sd(values):
    return (stats.mean(values) if values else None,
            stats.stdev(values) if len(values) > 1 else None)

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def write_csv(name, rows):
    with (OUT/name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

for folder in FOLDERS:
    files = sorted(p for p in (RESULTS/folder).rglob("*") if p.is_file())
    csv_path, md_path = RESULTS/folder/"summary.csv", RESULTS/folder/"summary.md"
    for path in files:
        original_bytes[path] = path.read_bytes()
    rows = read_csv(csv_path)
    seed_rows = [r for r in rows if r["row_type"] == "seed"]
    by_experiment = collections.defaultdict(list)
    md_seeds = {}
    experiment = None
    for line in md_path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("## "):
            experiment = line[3:].strip()
        if re.match(r"^\| (42|43|44) \|", line):
            cells = [v.strip() for v in line.split("|")[1:-1]]
            md_seeds[(experiment, int(cells[0]))] = cells[1:]
    assert len(md_seeds) == len(seed_rows)
    for row in seed_rows:
        name, seed, penalty = row["experiment"], int(row["seed"]), row["rd_cmr_penalty"]
        assert f"_{penalty}_" in name
        assert (name, seed) in md_seeds
        cells = md_seeds[(name, seed)]
        assert cells[0] == row["status"]
        for i in range(10):
            assert cells[i+1] == (row[f"step_{i}"] or "—")
        lam, alpha = float(row["lam_cmr"]), float(row["rd_class_weight_alpha"])
        name_tol = float(re.search(r"_tol([^_]+)_", name)[1].replace("p", "."))
        name_scale = float(re.search(r"_s([^_]+)_h200", name)[1].replace("p", "."))
        tol, scale = float(row.get("rd_margin_tolerance", name_tol)), float(row.get("rd_cmr_scale", name_scale))
        assert tol == name_tol and scale == name_scale
        ab = re.search(r"_tol_focus_v1_\w+_([AB])_", name)
        group = ab[1] if ab else "grid"
        ident = f"{penalty}_{group}_t{tol:g}" if ab else "direct_"+re.search(r"_(g\d+)_", name)[1]
        values = [float(row[f"step_{i}"])*100 if row[f"step_{i}"] else None for i in range(10)]
        if row["status"] == "complete":
            assert all(v is not None and math.isfinite(v) and 0 <= v <= 100 for v in values)
        else:
            assert row["status"] == "error" and all(v is None for v in values)
            errors.append(dict(folder=folder, config=ident, seed=seed, source_log=row["source_log"], reason=row["notes"]))
        config = configs.setdefault(ident, dict(config=ident, folder=folder, experiment=name,
            penalty=penalty, group=group, lam_cmr=lam, alpha=alpha, tolerance=tol, scale=scale,
            lambda_effective=lam*math.exp(-tol/scale) if penalty == "exp" else None,
            values={}))
        assert seed not in config["values"]
        if row["status"] == "complete":
            config["values"][seed] = values
        runs.append(dict(folder=folder, config=ident, seed=seed, status=row["status"],
            final_accuracy_pct=values[-1], aia_0_9_pct=stats.mean(values) if row["status"]=="complete" else None,
            source_log=row["source_log"], notes=row["notes"], **{f"step_{i}_pct":v for i,v in enumerate(values)}))
        by_experiment[name].append(row)
    for row in rows:
        if row["row_type"] not in ("mean", "std"):
            continue
        for step in range(10):
            values = [float(r[f"step_{step}"])*100 for r in by_experiment[row["experiment"]] if r[f"step_{step}"]]
            assert int(row[f"n_step_{step}"]) == len(values)
            expected = mean_sd(values)[row["row_type"]=="std"]
            actual = row[f"step_{step}"]
            assert bool(actual) == (expected is not None)
            if actual:
                max_aggregate_error = max(max_aggregate_error, abs(float(actual)*100-expected))
    sources.append(dict(folder=folder, files=[str(p.relative_to(RESULTS)) for p in files],
        hashes={p.name:digest(original_bytes[p]) for p in files}, configurations=len(by_experiment),
        run_slots=len(seed_rows), statuses=dict(collections.Counter(r["status"] for r in seed_rows)),
        raw_logs_present=any(p.suffix==".log" for p in files)))
assert max_aggregate_error <= 0.00011

table=[]
for config in configs.values():
    complete=list(config["values"].values())
    fm,fs=mean_sd([v[-1] for v in complete]); am,ass=mean_sd([stats.mean(v) for v in complete])
    table.append({k:v for k,v in config.items() if k!="values"} | dict(n=len(complete),
        final_accuracy_pct=fm, final_sd_pp=fs, aia_0_9_pct=am, aia_sd_pp=ass))
table.sort(key=lambda r:(r["penalty"],r["group"],r["lam_cmr"],r["alpha"],r["tolerance"]))

pairs=[("hinge_A_t0.01","hinge_A_t0"),("hinge_A_t0.1","hinge_A_t0.01"),
       ("log1p_A_t0.05","log1p_A_t0"),("log1p_A_t0.1","log1p_A_t0.05"),
       ("log1p_B_t0.01","log1p_B_t0"),("exp_A_t5","exp_A_t0.5"),
       ("exp_B_t1","exp_B_t0.1"),("direct_g006","direct_g009"),
       ("direct_g004","direct_g006"),("direct_g008","direct_g002"),
       ("hinge_A_t0.01","log1p_A_t0.05"),("hinge_A_t0.01","exp_A_t5"),
       ("hinge_B_t0","exp_B_t1"),("hinge_B_t0.1","exp_B_t0.1"),
       ("log1p_B_t0.1","exp_B_t0.1")]
pairs += [(f"hinge_{ab}_t{tol}",f"log1p_{ab}_t{tol}") for ab in ["A","B"] for tol in ["0","0.01","0.05","0.1"]]
paired=[]
for a,b in pairs:
    av,bv=configs[a]["values"],configs[b]["values"]
    seeds=sorted(av.keys()&bv.keys())
    for metric,indices in [("final_step_9",[9]),("aia_0_9",list(range(10)))]:
        ds={s:stats.mean(av[s][i]-bv[s][i] for i in indices) for s in seeds}
        mean,sd=mean_sd(list(ds.values()))
        paired.append(dict(candidate=a, comparator=b, metric=metric,n=len(ds),mean_delta_pp=mean,
            sd_delta_pp=sd,seed_wins=sum(d>0 for d in ds.values()),per_seed_delta_pp=json.dumps(ds),
            step0_matches=all(av[s][0]==bv[s][0] for s in seeds)))

# The available detailed hinge metrics must match the selected source before reuse.
archive=ROOT/"save_commands_cmr_hinge_tolerance_focus_3seeds"/"metrics"
if os.name=="nt": archive=Path("\\\\?\\"+str(archive))
hinge_metrics={}; archive_error=0.0
for ident,c in configs.items():
    if c["penalty"]!="hinge": continue
    for seed,values in c["values"].items():
        rows=read_csv(archive/(c["experiment"]+f"_seed{seed}")/"per_class_metrics.csv")
        hinge_metrics[(ident,seed)]=rows
        for i,v in enumerate(values):
            actual=100*next(float(r["overall_acc"]) for r in rows if int(r["step"])==i)
            archive_error=max(archive_error,abs(actual-v))
assert archive_error < .00006
detail=[]
for a,b in [("hinge_A_t0.01","hinge_A_t0"),("hinge_A_t0.1","hinge_A_t0.01")]:
    for seed in [42,43,44]:
        ar={int(r["class_id"]):r for r in hinge_metrics[(a,seed)] if r["step"]=="9"}
        br={int(r["class_id"]):r for r in hinge_metrics[(b,seed)] if r["step"]=="9"}
        def acc(rs,ids): return 100*sum(int(rs[c]["tp"]) for c in ids)/sum(int(rs[c]["support"]) for c in ids)
        detail.append(dict(candidate=a,comparator=b,seed=seed,
            old_accuracy_delta_pp=acc(ar,range(90))-acc(br,range(90)),
            new_accuracy_delta_pp=acc(ar,range(90,100))-acc(br,range(90,100)),
            overall_delta_pp=acc(ar,range(100))-acc(br,range(100)),
            task_forgetting_delta_pp=100*(float(ar[0]["forgetting"])-float(br[0]["forgetting"]))))

write_csv("config_summary.csv",table)
write_csv("seed_results.csv",runs)
write_csv("paired_comparisons.csv",paired)
write_csv("excluded_runs.csv",errors)
write_csv("hinge_old_new_decomposition.csv",detail)
step0={p:{s:sorted({v[0] for c in configs.values() if c['penalty']==p for seed,v in c['values'].items() if seed==s}) for s in [42,43,44]} for p in ['direct','exp','hinge','log1p']}
audit=dict(sources=sources,total_configs=len(configs),total_run_slots=len(runs),
    complete_runs=sum(r['status']=='complete' for r in runs),excluded_runs=len(errors),
    configs_with_three_complete_seeds=sum(r['n']==3 for r in table),
    max_aggregate_reconstruction_error_pp=max_aggregate_error,md_csv_agreement=True,
    hinge_archive_max_error_pp=archive_error,step0_accuracy_pct=step0,
    root_summary_used=False,notes=["Input contains per-folder summaries, not raw logs.",
    "Direct errors are num_workers manifest disagreements, not evidence of training failure.",
    "Selected exp/hinge/log1p share observed step0, which is a consistency check rather than proof of identical complete protocols.",
    "The earlier root-summary-only analysis is superseded."])
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
colors={'hinge':'#007F85','log1p':'#A35186','exp':'#CA782B','direct':'#4F70A5'}
for ab,ax in zip(['A','B'],axes[0]):
    for penalty in ['hinge','log1p']:
        rows=sorted([r for r in table if r['penalty']==penalty and r['group']==ab],key=lambda r:r['tolerance'])
        ax.errorbar(range(4),[r['final_accuracy_pct'] for r in rows],yerr=[r['final_sd_pp'] for r in rows],marker='o',capsize=4,label=penalty,color=colors[penalty])
    ax.set(title=f"{ab}: lambda={.03 if ab=='A' else .1}, alpha={.5 if ab=='A' else 1}",
           xticks=range(4),xticklabels=['0','.01','.05','.1'],xlabel='Tolerance (categorical spacing)',ylabel='Final accuracy (%)')
    ax.legend()
for ab,color in [('A','#CA782B'),('B','#587393')]:
    rows=sorted([r for r in table if r['penalty']=='exp' and r['group']==ab],key=lambda r:r['tolerance'])
    axes[1,0].errorbar([r['tolerance']/r['scale'] for r in rows],[r['final_accuracy_pct'] for r in rows],
        yerr=[r['final_sd_pp'] for r in rows],marker='o',capsize=4,color=color,label=f"{ab}: lambda={rows[0]['lam_cmr']}, alpha={rows[0]['alpha']}, s={rows[0]['scale']}")
axes[1,0].set(title='Exp: tolerance rescales lambda exactly',xlabel='Tolerance / scale',ylabel='Final accuracy (%)',xticks=[.1,.5,1])
axes[1,0].legend(fontsize=8)
for lam,color in [(.03,'#007F85'),(.1,'#CA782B'),(.3,'#A35186')]:
    rows=sorted([r for r in table if r['penalty']=='direct' and r['lam_cmr']==lam],key=lambda r:r['alpha'])
    axes[1,1].plot([r['alpha'] for r in rows],[r['final_accuracy_pct'] for r in rows],color=color,alpha=.55,label=f'lambda={lam}')
    for r in rows:
        axes[1,1].errorbar(r['alpha'],r['final_accuracy_pct'],yerr=r['final_sd_pp'] if r['n']>1 else None,
            fmt='o',color=color,mfc=color if r['n']==3 else 'white',capsize=4)
axes[1,1].set(title='Direct: hollow point = only one valid seed',xlabel='Alpha',ylabel='Final accuracy (%)',xticks=[0,.5,1])
axes[1,1].legend(fontsize=8)
for ax in axes.flat: ax.grid(alpha=.15)
fig.suptitle('Four result folders | mean +/- seed SD (not a confidence interval)',fontsize=14)
fig.savefig(OUT/'parameter_sweeps.png'); plt.close(fig)

chosen=['hinge_A_t0.01','hinge_A_t0.1','log1p_A_t0.05','hinge_B_t0','exp_A_t5','direct_g004']
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for y,ident in enumerate(chosen):
    row=next(r for r in table if r['config']==ident)
    for ax,metric,sd in [(axes[0],'final_accuracy_pct','final_sd_pp'),(axes[1],'aia_0_9_pct','aia_sd_pp')]:
        ax.errorbar(row[metric],y,xerr=row[sd],fmt='D',capsize=4,color=colors[row['penalty']],markersize=6)
for ax,title in zip(axes,['Final-step accuracy (%)','Average incremental accuracy (%)']):
    ax.set(yticks=range(len(chosen)),yticklabels=chosen,xlabel=title)
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.15)
fig.suptitle('Candidate trade-offs | all n=3; direct has a different observed step-0 baseline',fontsize=13)
fig.savefig(OUT/'candidate_tradeoffs.png');plt.close(fig)

def html_table(rows,fields):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape('—' if r[k] is None else f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table>'
report='''<!doctype html><meta charset="utf-8"><title>四组 CMR 结果：更正分析</title>
<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#193442}p,li{line-height:1.7}img{width:100%}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dce4e8;text-align:right}th{background:#eaf3f3}.note{padding:16px;background:#fff4d9}</style>
<h1>四组 CMR 结果：更正分析</h1><p class="note">本报告替代此前仅基于 results 顶层 summary 的分析。输入明确限定为 logs_direct、logs_exp_selected、logs_hinge_selected、logs_log1p_selected 四个文件夹各自的 summary.csv/md。未使用顶层 summary，未改动输入文件。</p>
<p>31 个配置、93 个 seed 条目：85 个完整有效结果，8 个 direct 条目因 num_workers 与命令清单不一致被汇总器排除。27 个配置有完整的三个 seed；4 个 direct 配置只有一个有效 seed。汇总排除不意味着训练失败。四个目录仅含汇总文件，本地没有这些目录对应的原始日志。</p>
<p>末阶段指标为 step 9 overall accuracy；AIA 为每个完整 seed 的 step 0–9 简单平均后再跨 seed 平均。所有误差棒是 seed 间样本标准差，不能解释为置信区间或显著性。CSV 均值、标准差、有效 seed 数及 Markdown 数据均已逐项核对。</p>
<img src="parameter_sweeps.png"><img src="candidate_tradeoffs.png">
<h2>候选与参数规律</h2><p>hinge A/tolerance=.01 是优先的末阶段表现与稳定性候选；log1p A/tolerance=.05 的 AIA 最好，应作为并行候选。hinge A/.1 的末阶段均值只比 A/.01 高 .04 个百分点，方差更大，不能据此认定更好。hinge B/tolerance=0 是较稳定的另一个配置。A/B 同时改变 lambda 和 alpha，exp 还改变 scale，不能将组间差异归因于单一参数。</p>
<p>Exp 的 tolerance 不是硬容忍区间：lambda*s*exp((D-epsilon)/s) = [lambda*exp(-epsilon/s)]*s*exp(D/s)。当前固定 Trust 权重下，epsilon 与 lambda 在目标上可折叠成 lambda_effective。A 组的有效系数依次为 .027145、.018196、.011036；B 组为 .090484、.060653、.036788。B 组减弱约束后三个 seed 的末阶段均改善；A 组平均改善但 seed43 下降。这支持继续分析有效约束强度，不构成 tolerance 越大越好的证明。</p>
<p>Hinge 与 log1p 都仅在 D&gt;epsilon 时有梯度。前者对 margin 的梯度幅度为 1；后者为 1/(1+(D-epsilon)/s)，会降低大 gap 样本的权重。Exp 的幅度为 exp((D-epsilon)/s)，Direct 为 1 且始终作用。两种单侧目标都值得保留：hinge 与 log1p 的胜负随配置和评价指标变化。</p>
<p>在相同的 lambda=.1、alpha=1、scale=1、tolerance=.1 下，hinge 相对 exp 的 FAA/AIA 平均提高 2.3733/1.2223 个百分点，log1p 相对 exp 提高 1.7067/1.0580，均为三个 seed 同向改善。这是优先分析单侧目标的直接依据，但只有三个 seed，不能据此宣布统计显著或普遍优势。log1p A/.05 相对 A/0 的 FAA/AIA 也在三个 seed 上全部提高，均值分别为 .34/.3737 个百分点。</p>
<p>Direct 在 alpha=1 下，将 lambda 从 .3 降至 .03，FAA/AIA 分别提高 1.48/1.9337 个百分点，三个 seed 全部改善。但 alpha=0 的 lambda=.3 相对 .1 在 FAA 上略高、AIA 更低，因此不能宣称所有切片都单调。Direct 中 lambda=.03、alpha=.5 是三个有效 seed 配置中的最佳均值；alpha=0 的同 lambda 配置只有 seed44，不能按其单次 39.56% 排为最佳。固定 lambda=.03，alpha=.5 相对 1 的 FAA 提升主要由 seed43 贡献，alpha 的最佳值仍不明确。</p>
<p>诊断中的 cmr_active 是 D&gt;epsilon 的违反比例，exp/direct 即使 D≤epsilon 仍有梯度；跨 tolerance 比较时，应同时用共同阈值统计 D 的分布，避免把阈值改变误读为表示退化减少。不同 penalty 的原始 loss 数值也不能直接作为约束强度比较，应查看乘上 lambda 和类别权重后的梯度。</p>
<h2>全部配置（百分比；有效 n 明示）</h2>'''
report+=html_table(table,['config','lam_cmr','alpha','tolerance','scale','lambda_effective','n','final_accuracy_pct','final_sd_pp','aia_0_9_pct','aia_sd_pp'])
report+='''<h2>由已有 hinge 明细验证的性能来源</h2><p>已验证本地 24 个 hinge 运行的逐类测试数据与 selected 汇总一致。A/tolerance=.01 相对 A/0 的末阶段平均改善 .8267 个百分点：旧类准确率提高 .8741，新类提高 .4；由于末阶段有 90 个旧类、10 个新类且每类测试数相同，加权贡献分别为 .7867、.04。旧类准确率三个 seed 均改善，新类并非每个 seed 改善。</p>
<p>A/.1 相对 A/.01 的总体均值仅提高 .04：旧类平均提高 .2370，新类平均下降 1.7333，且 seed 间差异大。不能从 overall accuracy 的极小提升推出学习与保留均改善。</p>
<h2>下一步：优先实验对照</h2><ol>
<li>立即利用已有 hinge 明细分析 A/0 vs A/.01 vs A/.1：task×step 矩阵、旧/新类别、两个方向的 violation 与 Need，并对齐验证最佳 epoch；最后一个 epoch 的诊断不自动对应测试模型。</li>
<li>取得 log1p A/.05 的 metrics/检查点，和 hinge A/.05 做 penalty 单变量对照；另比较 hinge B/.1 vs log1p B/.1，这一组 hinge 在所有三个 seed 的 FAA 和 AIA 都更好。</li>
<li>取得 exp B/.1 与 B/1，分析减弱 lambda_effective 后的旧类保留、新类学习与梯度冲突；无需再独立穷举 exp 的 lambda 和 tolerance。</li>
<li>补足不含 CMR 的匹配基线及原 CrossSDC-C 基线。当前结果只能排序已有 CMR 配置，尚不能证明 CMR 本身提升。</li>
<li>从服务器取回 direct 中被排除的 8 个原始日志，按实际 Namespace 核对 worker/代码/随机数条件，再决定分组。不能恢复 CSV 中已经清空的数值，也不能直接把它们当训练失败。</li></ol>
<p>Selected exp/hinge/log1p 的相同 seed 在 step0 完全相同；direct 的 step0 不同。前者提供可比性支持，但完整运行参数/代码版本仍需核对。direct 跨方法比较仅作描述，不作为严格的 penalty 单变量因果结论。新未裁剪权重公式也不能由缺少完整参数的这些汇总直接评估。</p>
<h2>检查点补数据的边界</h2><p>本地可确认与本次输入对应的完整 metrics/检查点只有 hinge 存档。检查点配合输入特征可以生成预测、混淆矩阵、特征、固定 probe 上的 margin 与梯度。原训练口径还依赖 replay IDs 和当时的教师原型；同 seed 不足以无条件恢复原 memory。只有最佳检查点无法重建所有 epoch 的 Need、历史回放曝光或优化器轨迹。新的 probe 诊断必须标明样本和参照，不冒充原训练统计。</p>
<p>后续保存运行清单、代码版本、类别映射/数据版本、replay IDs、prototype bank、best_epoch、probe IDs 和逐样本预测。所有派生表及输入哈希见同目录 CSV 和 audit.json；reanalyze_selected_results.py 可重新生成本报告。</p>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
for path,data in original_bytes.items(): assert path.read_bytes()==data
print(json.dumps(audit,indent=2,ensure_ascii=False))
print('OUTPUT:',OUT)
