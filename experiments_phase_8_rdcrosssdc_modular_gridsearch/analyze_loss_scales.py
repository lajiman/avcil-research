"""Read archived epoch summaries; never substitute checkpoint probes for history."""
import argparse
import collections
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def long_path(path):
    path = str(Path(path).resolve())
    return Path('\\\\?\\' + path) if os.name == 'nt' and not path.startswith('\\\\?\\') else Path(path)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=here/'save_commands_cmr_hinge_tolerance_focus_3seeds')
    parser.add_argument('--output', type=Path, default=here/'results'/'loss_scale_analysis')
    args = parser.parse_args()
    archive, out = long_path(args.archive), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(archive.rglob('epoch_summary.csv'))
    if not paths:
        raise ValueError('No epoch_summary.csv files found')
    records, sources, best_epochs = [], [], []
    for path in paths:
        run = path.parents[1].name
        match = re.search(r'_hinge_([AB])_lc([^_]+)_a([^_]+)_tol([^_]+)_s([^_]+)_h200_seed(\d+)$', run)
        if not match:
            raise ValueError('This archive analyzer expects the hinge tolerance runs: ' + run)
        group, lam, alpha, tol, scale, seed = match.groups()
        lam, alpha, tol, scale = (float(v.replace('p', '.')) for v in [lam, alpha, tol, scale])
        payload = path.read_bytes()
        with path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2000 and len({(r['step'], r['epoch']) for r in rows}) == 2000
        assert {(int(r['step']), int(r['epoch'])) for r in rows} == {(s, e) for s in range(10) for e in range(200)}
        sources.append(dict(run=run, rows=len(rows), sha256=hashlib.sha256(payload).hexdigest()))
        for row in rows:
            r = {k:float(v) for k,v in row.items() if k not in ['rd_mode', 'cmr_penalty']}
            assert row['cmr_penalty'] == 'hinge'
            assert abs(r['weighted_cmr'] - lam*r['cmr']) < 1e-8
            assert abs(r['weighted_cross_sdc'] - .1*r['cross_sdc_i']) < 1e-8
            r.update(config=f'hinge_{group}_t{tol:g}', run=run, seed=int(seed), lam_cmr=lam, alpha=alpha,
                     tolerance=tol, step=int(r['step']), epoch=int(r['epoch']))
            r['other_combined'] = r['train_loss'] - r['weighted_cross_sdc'] - r['weighted_cmr']
            r['cmr_fraction_of_total'] = r['weighted_cmr']/r['train_loss'] if r['train_loss'] else None
            records.append(r)
        for step in range(10):
            candidates = [r for r in rows if int(r['step']) == step]
            best = max(candidates, key=lambda r:float(r['val_acc']))  # first tie, like training
            best_epochs.append(dict(run=run, step=step, best_epoch=int(best['epoch']), val_acc=float(best['val_acc'])))
        assert path.read_bytes() == payload
    assert len(paths) == 24 and len(records) == 48000
    increments = [r for r in records if r['step'] > 0]
    metrics = ['train_loss','cross_sdc_i','cross_sdc_c','weighted_cross_sdc','cmr','weighted_cmr','other_combined']
    summaries = []
    for config in sorted({r['config'] for r in increments}):
        for phase, pred in [('all',lambda e:True),('first10',lambda e:e<10),('last20',lambda e:e>=180)]:
            subset = [r for r in increments if r['config']==config and pred(r['epoch'])]
            item = dict(config=config, lam_cmr=subset[0]['lam_cmr'], alpha=subset[0]['alpha'],
                        tolerance=subset[0]['tolerance'], phase=phase, n_seeds=3, n_epoch_records=len(subset))
            for metric in metrics:
                values = [r[metric] for r in subset]
                item[metric+'_mean'] = statistics.mean(values)
                item[metric+'_p10'], item[metric+'_p90'] = np.quantile(values, [.1,.9])
            item['cmr_share_pct_ratio_of_sums'] = 100*sum(r['weighted_cmr'] for r in subset)/sum(r['train_loss'] for r in subset)
            item['cmr_share_pct_median_epoch'] = statistics.median(100*r['cmr_fraction_of_total'] for r in subset)
            summaries.append(item)
    write_csv(out/'epoch_loss_scales.csv', records)
    write_csv(out/'config_phase_summary.csv', summaries)
    write_csv(out/'best_epochs.csv', best_epochs)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':170,
                         'axes.spines.right':False,'axes.spines.top':False})
    chosen = ['hinge_A_t0.01','hinge_B_t0']
    fig, axes = plt.subplots(2,2, figsize=(13,8), layout='constrained')
    colors = {'train_loss':'#243E54','weighted_cross_sdc':'#D88526','weighted_cmr':'#007F85','other_combined':'#9B649B'}
    for col, config in enumerate(chosen):
        rs = [r for r in increments if r['config']==config]
        for metric in colors:
            # Average nine steps within each seed, then show mean and range of three seeds.
            curves = np.array([[statistics.mean(r[metric] for r in rs if r['seed']==seed and r['epoch']==epoch)
                                for epoch in range(200)] for seed in [42,43,44]])
            axes[0,col].plot(curves.mean(axis=0),color=colors[metric],label=metric)
            axes[0,col].fill_between(range(200),curves.min(axis=0),curves.max(axis=0),alpha=.13,color=colors[metric])
        axes[0,col].set(title=config+' | steps 1-9 averaged',yscale='log',ylabel='Weighted loss (log scale)',xlabel='Epoch within step')
        axes[0,col].legend(fontsize=8)
        for step in [1,5,9]:
            means = [statistics.mean(r['weighted_cmr'] for r in rs if r['step']==step and r['epoch']==epoch) for epoch in range(200)]
            axes[1,col].plot(means,label=f'step {step}')
        axes[1,col].set(yscale='log',xlabel='Epoch within step',ylabel='Weighted CMR (seed mean)',title=config+' | individual steps')
        axes[1,col].legend()
    for ax in axes.flat: ax.grid(alpha=.15)
    fig.suptitle('Recorded training losses | upper shading: seed range, not confidence interval',fontsize=13)
    fig.savefig(out/'recorded_loss_curves.png'); plt.close(fig)

    fig, axes = plt.subplots(1,2, figsize=(12,4.5), layout='constrained')
    for group,ax in zip(['A','B'],axes):
        rs = [r for r in increments if f'hinge_{group}_' in r['config']]
        for tol in [0,.01,.05,.1]:
            subset=[r for r in rs if r['tolerance']==tol]
            shares=[100*sum(r['weighted_cmr'] for r in subset if r['epoch']==epoch)/sum(r['train_loss'] for r in subset if r['epoch']==epoch) for epoch in range(200)]
            ax.plot(shares,label=f'tolerance={tol:g}')
        ax.set(yscale='log',title=f'{group}: lambda={.03 if group=="A" else .1}, alpha={.5 if group=="A" else 1}',
               xlabel='Epoch within step',ylabel='CMR / total scalar loss (%)')
        ax.legend(); ax.grid(alpha=.15)
    fig.suptitle('Scalar proportions are NOT gradient or optimizer-update proportions',fontsize=13)
    fig.savefig(out/'cmr_scalar_share.png'); plt.close(fig)
    selected=[r for r in summaries if r['config'] in chosen]
    fields=['config','phase','train_loss_mean','weighted_cross_sdc_mean','cmr_mean','weighted_cmr_mean','other_combined_mean','cmr_share_pct_ratio_of_sums']
    table='<table><tr>'+''.join('<th>'+k+'</th>' for k in fields)+'</tr>'
    for r in selected:
        table+='<tr>'+''.join('<td>'+html.escape(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>'
    table+='</table>'
    report='''<!doctype html><meta charset="utf-8"><title>训练 Loss 尺度</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#193442}p,li{line-height:1.7}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd}th{background:#edf5f5}</style>
<h1>现有 hinge 实验的真实训练 loss 尺度</h1>
<p>来源是 24 个运行的 epoch_summary.csv，共 48,000 条 epoch 记录。主图和统计排除只训练 CE 的 step0，包含 43,200 条增量训练记录。每个 epoch 数值是训练过程中各 batch loss 的平均，并非在 epoch 末或最佳 checkpoint 上重算。每个配置含 3 个 seed、9 个增量阶段、200 个 epoch。</p>
<p>已逐行校验 weighted_cmr = lambda × cmr，以及 weighted_cross_sdc = 0.1 × cross_sdc_i。CrossSDC-C 原始值虽然约为 2–3，此批实验的加权贡献为零。other_combined = total − weighted_cross_sdc − weighted_cmr，是 CE、KD、当前模态对比项与注意力蒸馏的合计；原数据不能将其继续拆开。CMR 原始值已包含类别权重和两个方向的平均，只是尚未乘全局 lambda。</p>
<img src="recorded_loss_curves.png"><img src="cmr_scalar_share.png">
<p>阶段定义：first10 是各增量阶段 epoch 0–9；last20 是 epoch 180–199；all 是 0–199。表中占比是 sum(weighted CMR)/sum(total)，不是先算每个 epoch 百分比再平均。CSV 另含 epoch 占比中位数和各 loss 的 10%/90% 分位数。</p>'''+table+'''
<p>A 与 B 同时改变 lambda 和 alpha，不是 lambda 单变量实验。模型训练后的 CMR 值本身也会随约束改变，因此 lambda 增加 3.33 倍不意味着训练完成后 weighted_cmr 必然增加 3.33 倍。即使标量占比低，也不能据此判定梯度或优化作用可忽略。</p>
<p>解释 lambda 应同时看：原始 loss、加权 loss、加权梯度范数、CMR 与其他目标梯度的夹角，以及旧/新类表现。梯度是反向传播信号，也不是 Adam 实际更新比例；Adam 状态与 weight decay 还会影响最终更新。Direct 的 loss 可以为负，不能把 signed loss 比值当组成百分比。</p>
<p>只有最佳 checkpoint 无法恢复历史全程。带输入特征、教师、回放样本与原型可做清楚标记的固定 probe，比较同一状态下不同 lambda 的瞬时效果；不能把这种反事实测量当作另一个 lambda 的训练轨迹。best_epochs.csv 是验证集首次达到最大值的 epoch，匹配训练的严格大于更新规则。</p>'''
    (out/'report.html').write_text(report,encoding='utf-8')
    audit=dict(source_files=sources,total_rows=len(records),incremental_rows=len(increments),
               missing_component_history=['ce','kd','instance_contrastive','class_contrastive','attn_spatial','attn_temporal'],
               source_kind='recorded_training_epoch_means',source_files_modified=False)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(paths),rows=len(records),incremental_rows=len(increments),output=str(out)),indent=2))


if __name__ == '__main__':
    main()
