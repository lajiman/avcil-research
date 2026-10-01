"""Audit and analyze returned offline probes using CSV/JSON only; no inference."""
from pathlib import Path
import collections
import csv
import hashlib
import html
import json
import math
import os
import statistics as st

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
EXPERIMENT = BASE.parent.parent
REPO = EXPERIMENT.parent
RUNS = [('A', 'lambda_0p03_alpha_0p5'), ('B', 'lambda_0p1_alpha_1p0')]
COMPONENTS = ['ce', 'kd', 'instance_contrastive', 'class_contrastive', 'cross_sdc_i',
              'cross_sdc_c', 'cmr', 'attn_spatial', 'attn_temporal']
LABELS = ['CE', 'KD', 'Instance contrastive', 'Class contrastive', 'CrossSDC-I',
          'CrossSDC-C', 'CMR', 'Spatial attention', 'Temporal attention']
SCOPE = 'feature_and_attention'
STEPS = [1, 5, 9]
original = {}


def native(path):
    path = str(Path(path).resolve())
    return Path('\\\\?\\' + path) if os.name == 'nt' and not path.startswith('\\\\?\\') else Path(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    path = native(path)
    original[path] = path.read_bytes()
    return original[path]


def read_csv(path):
    return list(csv.DictReader(read_bytes(path).decode('utf-8-sig').splitlines()))


def number(row, field):
    return float(row[field]) if row[field] != '' else None


def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def near(a, b, rtol=1e-7, atol=1e-10):
    assert math.isclose(a, b, rel_tol=rtol, abs_tol=atol), (a, b)


audit = dict(source='offline_checkpoint_fixed_train_probe', input_files={}, runs={},
             checkpoints_match_archive=True, code_matches_after_lf_normalization=True,
             summary_reconstruction=True, gram_matrix_reconstruction=True,
             new_model_evaluations=0, optimizer_updates=0)
data, manifests = {}, {}
cmr_rows, pair_rows, component_rows, history_rows, coverage = [], [], [], [], []
discrepancy_rows = []
for alias, folder in RUNS:
    run = BASE/folder
    meta = json.loads(read_bytes(run/'metadata.json'))
    manifest = json.loads(read_bytes(run/'probe_manifest.json'))
    assert meta['status'] == 'complete' and meta['optimizer_updates'] == 0
    assert meta['cli']['device'] == 'cpu' and meta['cli']['steps'] == STEPS
    assert all(s['parameters_and_buffers_unchanged'] for s in meta['states'])
    manifests[alias] = manifest
    raw = read_csv(run/'components.csv')
    summary = read_csv(run/'component_summary.csv')
    pairs = read_csv(run/'gradient_pairs.csv')
    sweep = read_csv(run/'lambda_sweep.csv')
    assert (len(raw), len(summary), len(pairs), len(sweep)) == (243, 81, 972, 27)
    assert len({(r['step'], r['batch'], r['parameter_scope'], r['component']) for r in raw}) == 243
    assert len({(r['step'], r['batch'], r['parameter_scope'], r['component_a'], r['component_b']) for r in pairs}) == 972
    assert len({(r['step'], r['batch'], r['parameter_scope']) for r in sweep}) == 27
    assert all((int(r['n_current']), int(r['n_replay'])) == (128, 128) for r in raw)
    for r in raw:
        assert int(r['enabled']) == 1
        near(float(r['weighted_loss']), float(r['raw_loss'])*float(r['coefficient']))
        near(float(r['weighted_grad_norm']), float(r['raw_grad_norm'])*abs(float(r['coefficient'])))
    for r in summary:
        rs = [x for x in raw if all(x[k] == r[k] for k in ['step', 'parameter_scope', 'component'])]
        assert len(rs) == 3 and r['batches'] == '3'
        for field in ['raw_loss', 'weighted_loss', 'raw_grad_norm', 'weighted_grad_norm',
                      'weighted_grad_over_non_cmr', 'cosine_with_non_cmr']:
            vals = [number(x, field) for x in rs if number(x, field) is not None]
            if vals:
                near(st.mean(vals), float(r[field+'_mean']))
                near(st.stdev(vals), float(r[field+'_batch_sd']))
            else:
                assert r[field+'_mean'] == r[field+'_batch_sd'] == ''
        component_rows.append(dict(run=alias, **r))
    # Independently reconstruct gradient sums from pairwise cosines and norms.
    for r in sweep:
        rs = {x['component']:x for x in raw if all(x[k] == r[k] for k in ['step', 'batch', 'parameter_scope'])}
        ps = [x for x in pairs if all(x[k] == r[k] for k in ['step', 'batch', 'parameter_scope'])]
        assert set(rs) == set(COMPONENTS) and len(ps) == 36
        near(sum(float(x['weighted_loss']) for x in rs.values()), float(rs['cmr']['total_loss']), rtol=2e-6)
        norms = np.array([float(rs[c]['weighted_grad_norm']) for c in COMPONENTS])
        gram = np.diag(norms**2)
        for p in ps:
            i, j = COMPONENTS.index(p['component_a']), COMPONENTS.index(p['component_b'])
            gram[i,j] = gram[j,i] = norms[i]*norms[j]*(number(p, 'weighted_cosine') or 0)
        assert np.linalg.eigvalsh(gram).min() > -1e-8
        mask = np.ones(len(COMPONENTS)); mask[COMPONENTS.index('cmr')] = 0
        b = math.sqrt(max(0, mask @ gram @ mask))
        near(b, float(r['non_cmr_grad_norm']))
        near(math.sqrt(max(0, gram.sum())), float(r['total_grad_norm']))
        if norms[6] > 0:
            near(float(mask @ gram[:,6])/(b*norms[6]), float(r['cmr_cosine_with_non_cmr']))
            near(norms[6]/b, float(r['cmr_over_non_cmr']))
        if r['parameter_scope'] == 'classifier':
            assert all(float(rs[c]['weighted_grad_norm']) == 0 for c in COMPONENTS if c not in ['ce', 'kd'])
    code_checks = []
    for remote, expected in meta['source_sha256'].items():
        rel = remote.split('AV-CIL_ICCV2023_share_h200/', 1)[1]
        actual = digest(read_bytes(REPO/rel).replace(b'\r\n', b'\n'))
        assert actual == expected
        code_checks.append(rel)
    for fname, expected in manifest['metadata_sha256'].items():
        assert digest(read_bytes(REPO/'data2/balance'/fname)) == expected
    for state in meta['states']:
        step, args = state['step'], state['args']
        assert (args['seed'], args['rd_cmr_penalty'], args['rd_margin_tolerance']) == (42, 'hinge', .01)
        assert meta['cli']['lambdas'] == [args['lam_cmr']]
        archive = EXPERIMENT/'save_commands_cmr_hinge_tolerance_focus_3seeds'
        for role, s in [('student',step), ('teacher',step-1)]:
            p = archive/args['experiment_name']/f'step_{s}_best_model.pkl'
            # Checkpoint hashes only: no pickle loading or model execution.
            h = hashlib.sha256()
            with native(p).open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):
                    h.update(block)
            assert h.hexdigest() == state['checkpoint_sha256'][role]
        ep = read_csv(archive/'metrics'/args['experiment_name']/'rd_crosssdc/epoch_summary.csv')
        ep = [x for x in ep if int(x['step']) == step]
        best = max(ep, key=lambda x:float(x['val_acc']))
        test = next(x for x in read_csv(archive/'metrics'/args['experiment_name']/'per_class_metrics.csv') if int(x['step']) == step)
        rs = [x for x in raw if int(x['step']) == step and x['parameter_scope'] == SCOPE and x['component'] == 'cmr']
        ss = [x for x in sweep if int(x['step']) == step and x['parameter_scope'] == SCOPE]
        cmr = dict(run=alias, step=step, lambda_cmr=args['lam_cmr'], alpha=args['rd_class_weight_alpha'], batches=3,
            raw_cmr_loss=st.mean(float(x['raw_loss']) for x in rs),
            weighted_cmr_loss=st.mean(float(x['weighted_loss']) for x in rs),
            total_loss=st.mean(float(x['total_loss']) for x in rs),
            scalar_share_pct=st.mean(100*float(x['weighted_loss'])/float(x['total_loss']) for x in rs),
            raw_cmr_grad_norm=st.mean(float(x['raw_grad_norm']) for x in rs),
            weighted_cmr_grad_norm=st.mean(float(x['weighted_grad_norm']) for x in rs),
            non_cmr_grad_norm=st.mean(float(x['non_cmr_grad_norm']) for x in rs))
        for label, values in [
            ('rho_pct',[100*float(x['cmr_over_non_cmr']) for x in ss]),
            ('cosine',[float(x['cmr_cosine_with_non_cmr']) for x in ss]),
            ('turn_degrees',[math.degrees(math.acos(min(1,float(x['total_cosine_with_non_cmr'])))) for x in ss]),
            ('lambda_equal',[float(x['lambda_for_equal_grad_norm']) for x in ss]),
            ('total_norm_increase_pct',[100*(float(x['total_grad_norm'])/float(x['non_cmr_grad_norm'])-1) for x in ss])]:
            cmr[label+'_mean'], cmr[label+'_min'], cmr[label+'_max'] = st.mean(values), min(values), max(values)
        cmr_rows.append(cmr)
        neighboring = [float(x['weighted_cmr']) for x in ep if abs(int(x['epoch'])-int(best['epoch'])) <= 5]
        discrepancy_rows.append(dict(run=alias, step=step, best_epoch=int(best['epoch']),
            train_best_epoch_weighted_cmr=float(best['weighted_cmr']),
            train_neighbor_epochs=len(neighboring), train_neighbor_min=min(neighboring),
            train_neighbor_max=max(neighboring), train_neighbor_mean=st.mean(neighboring),
            probe_batch_1_weighted_cmr=float(rs[0]['weighted_loss']),
            probe_batch_2_weighted_cmr=float(rs[1]['weighted_loss']),
            probe_batch_3_weighted_cmr=float(rs[2]['weighted_loss']),
            probe_mean_weighted_cmr=cmr['weighted_cmr_loss']))
        history_rows.append(dict(run=alias,step=step,best_epoch=int(best['epoch']), test_accuracy_pct=100*float(test['overall_acc']),
            train_epoch_mean_loss=float(best['train_loss']), probe_total_loss=cmr['total_loss'],
            train_epoch_mean_weighted_cmr=float(best['weighted_cmr']),probe_weighted_cmr=cmr['weighted_cmr_loss'],
            probe_over_train_cmr=cmr['weighted_cmr_loss']/float(best['weighted_cmr']),
            train_epoch_mean_weighted_cross_sdc=float(best['weighted_cross_sdc']),
            probe_weighted_cross_sdc=st.mean(float(x['weighted_loss']) for x in raw if int(x['step'])==step and x['parameter_scope']==SCOPE and x['component']=='cross_sdc_i')))
        for component in COMPONENTS:
            if component in ['cmr','cross_sdc_c']:
                continue
            ps = [x for x in pairs if int(x['step']) == step and x['parameter_scope'] == SCOPE
                  and {x['component_a'],x['component_b']} == {'cmr',component}]
            vals = [float(x['weighted_cosine']) for x in ps]
            pair_rows.append(dict(run=alias,step=step,other_component=component,
                                 cosine_mean=st.mean(vals),cosine_min=min(vals),cosine_max=max(vals),negative_batches=sum(v<0 for v in vals)))
        plan = manifest['steps'][str(step)]
        coverage.append(dict(run=alias,step=step,memory_size=len(plan['memory_ids']),
                             unique_current=len({v for b in plan['batches'] for v in b['current_ids']}),
                             unique_replay=len({v for b in plan['batches'] for v in b['replay_ids']})))
    relative_residuals = [float(x['gradient_reconstruction_residual'])/float(next(y['total_grad_norm'] for y in sweep
                           if y['step']==x['step'] and y['batch']==x['batch'] and y['parameter_scope']=='all'))
                          for x in raw if x['parameter_scope']=='all' and x['component']=='cmr']
    audit['runs'][alias] = dict(folder=folder, completed=True, batches=9, steps=STEPS, scope_count=3,
        max_gradient_reconstruction_residual=max(float(x['gradient_reconstruction_residual']) for x in raw),
        max_relative_gradient_reconstruction_residual=max(relative_residuals),code_files_verified=code_checks)
    data[alias] = dict(raw=raw,summary=summary,pairs=pairs,sweep=sweep,metadata=meta)

assert manifests['A'] == manifests['B']
audit['identical_manifests'] = True
audit['coverage'] = coverage
args_a, args_b = [data[a]['metadata']['states'][0]['args'] for a in ['A','B']]
audit['configuration_differences'] = {k:[args_a.get(k),args_b.get(k)] for k in args_a.keys()|args_b.keys() if args_a.get(k)!=args_b.get(k)}
audit['limitations'] = ['One training seed, three probe batches per checkpoint; no population significance claims.',
                       'Lambda and alpha both differ; each run uses its own previous-step teacher.',
                       'Fresh probe memory and reconstructed Trust, not the original training replay.',
                       'Historical scalars are within-epoch batch averages, not checkpoint probe quantities.',
                       'Gradient norms are not Adam update fractions or additive shares.']
audit['interpretation_correction'] = {
    'date': '2026-09-30',
    'historical_training_gradient_estimates_validated': False,
    'historical_replay_ids_recovered': False,
    'different_sampling_and_reference_confirmed': True,
    'individual_causes_of_scalar_gap_quantified': False,
    'withdrawn_inference': 'The 19x/69x gap alone does not establish a retention-generalization gap or training-time gradient balance.',
    'required_next_check': 'Reconstruct and validate historical memory and reference/weights, then compare scalars at saved checkpoints before computing gradients.'}
write_csv('cmr_summary.csv', cmr_rows)
write_csv('loss_gradient_summary.csv', component_rows)
write_csv('cmr_pairwise_summary.csv', pair_rows)
write_csv('training_probe_comparison.csv', history_rows)
write_csv('probe_coverage.csv', coverage)
write_csv('discrepancy_audit.csv', discrepancy_rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':175,
                     'axes.spines.top':False,'axes.spines.right':False})
columns = [(a,s) for a in ['A','B'] for s in STEPS]
column_labels = [f'{a} / step {s}' for a,s in columns]
def summary_value(a,step,component,field):
    return float(next(r[field] for r in data[a]['summary'] if int(r['step'])==step and r['component']==component and r['parameter_scope']==SCOPE))

fig, axes = plt.subplots(1,2,figsize=(15,6),layout='constrained')
for ax,field,title in [(axes[0],'weighted_loss_mean','Weighted scalar loss'),(axes[1],'weighted_grad_norm_mean','Weighted parameter-gradient L2 norm')]:
    arr=np.array([[summary_value(a,s,c,field) for a,s in columns] for c in COMPONENTS])
    positive=arr[arr>0]
    norm=LogNorm(positive.min(),positive.max())
    im=ax.imshow(np.ma.masked_where(arr<=0,arr),aspect='auto',cmap='viridis',norm=norm)
    ax.set(xticks=range(6),xticklabels=column_labels,yticks=range(9),yticklabels=LABELS,title=title)
    ax.tick_params(axis='x',rotation=35)
    for i in range(9):
        for j in range(6):
            v=arr[i,j]
            ax.text(j,i,f'{v:.3g}',ha='center',va='center',fontsize=9,color='black' if v==0 or norm(v)>.6 else 'white')
    ax.axvline(2.5,color='#ff6b6b',lw=1.2)
    fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle('Offline probes | A: lambda=.03, alpha=.5; B: lambda=.1, alpha=1\nFeature + attention parameters; mean of 3 batches at each best checkpoint',fontsize=12)
fig.savefig(OUT/'loss_and_gradient_scales.png');plt.close(fig)

fig, axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
for alias,color in [('A','#007f85'),('B','#c56b2d')]:
    rs=[r for r in cmr_rows if r['run']==alias]
    for ax,field,title in [(axes[0],'rho_pct','CMR / non-CMR gradient norm (%)'),
                           (axes[1],'cosine','Cosine: CMR vs non-CMR'),
                           (axes[2],'turn_degrees','Change in total gradient direction (degrees)')]:
        y=np.array([r[field+'_mean'] for r in rs])
        lower=y-np.array([r[field+'_min'] for r in rs]);upper=np.array([r[field+'_max'] for r in rs])-y
        ax.errorbar(STEPS,y,yerr=[lower,upper],marker='o',capsize=4,color=color,label=alias)
        ax.set(xticks=STEPS,xlabel='Incremental step',title=title)
        ax.grid(alpha=.2)
        ax.legend()
axes[1].axhline(0,color='gray',lw=.8,linestyle='--')
fig.suptitle('New-memory checkpoint probes, own lambda | bars span 3 batches (not confidence intervals)',fontsize=12)
fig.savefig(OUT/'cmr_influence.png');plt.close(fig)

others=[c for c in COMPONENTS if c not in ['cmr','cross_sdc_c']]
arr=np.array([[next(r['cosine_mean'] for r in pair_rows if r['run']==a and r['step']==s and r['other_component']==c) for a,s in columns] for c in others])
fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained')
im=ax.imshow(arr,aspect='auto',cmap='RdBu',vmin=-.4,vmax=.4)
ax.set(xticks=range(6),xticklabels=column_labels,yticks=range(len(others)),yticklabels=[LABELS[COMPONENTS.index(c)] for c in others],
       title='New-memory probe: CMR vs each loss gradient | feature + attention parameters')
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        ax.text(j,i,f'{arr[i,j]:+.3f}',ha='center',va='center',color='white' if abs(arr[i,j])>.26 else 'black')
ax.axvline(2.5,color='black',lw=1)
fig.colorbar(im,ax=ax,label='Mean of 3 probe-batch cosines')
fig.savefig(OUT/'cmr_pairwise_alignment.png');plt.close(fig)

def table(rows, fields):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.5g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table>'

report='''<!doctype html><meta charset="utf-8"><title>CPU checkpoint 梯度测量分析</title>
<style>body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:0 24px;color:#183340}p,li{line-height:1.75}img{width:100%}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf4f5}.note{background:#fff4d9;padding:16px}</style>
<h1>CPU checkpoint 梯度测量分析</h1>
<p class="note"><strong>2026-09-30 解释更正：</strong>此次 probe 每阶段重新抽取旧类 memory，并重建原型和 Trust；它没有还原原训练的回放样本与 CMR 参照。下列梯度数字只描述这个新构造的离线目标，不能作为原训练梯度尺度的估计。此前据此解释“训练中 CMR 较弱、KD 主导、没有整体冲突”的表述应撤回。19/69 倍差异不能单独支持“原回放的保持效果未泛化”的解释；样本、原型、权重和测量时点的影响尚未分离。原始 CSV 数值保留，历史梯度结论未获验证。</p>
<p class="note">两次运行均完整。A=hinge λ=.03/α=.5，B=hinge λ=.1/α=1；共同 tolerance=.01、训练 seed42。step 1/5/9，每个 checkpoint 3 个 current128+replay128 batch。下述梯度均为 feature_and_attention 参数范围；全模型结果见 CSV。所有结论对应保存的最佳状态和新的固定 train probe，不代表整个训练历史。</p>
<p>样本清单字节内容相同；3 个 metadata 文件、6 份源代码（统一换行符）及两组学生/教师检查点哈希均与本地存档一致。CSV 汇总、加权关系、梯度内积矩阵与合成梯度范数均独立复核。两组最大梯度重构绝对误差约 1.05e-5，相对误差见 audit.json；不是影响结论的量级。参数及 buffer 未改变，optimizer 更新次数为零。本分析只读 CSV/JSON/存档，没有再次前向、反向或训练。</p>
<h2>新 memory 上测得的 CMR 尺度</h2>
<p>仅在这个新构造的 probe 目标上，CMR 是较小的修正项。A 在 step 1/5/9 的加权梯度相对非 CMR 合计为 .317%/2.327%/1.589%；B 为 .832%/6.655%/6.709%。B 后期已与类别对比项、空间注意力项处于接近量级，但未主导此 probe 的 CE/KD 及其合计。这些比例尚不能外推到原训练。</p>
<p>从 A 到 B，加权 CMR 梯度均值分别放大约 3.06/3.47/3.90 倍，接近但不严格等于 λ 的 3.33 倍。这是两个不同训练状态且 α 不同的比较，不可将差异全部归因于 λ。</p>
<img src="cmr_influence.png">'''
report+=table(cmr_rows,['run','step','raw_cmr_loss','weighted_cmr_loss','raw_cmr_grad_norm','weighted_cmr_grad_norm','non_cmr_grad_norm','rho_pct_mean','rho_pct_min','rho_pct_max','cosine_mean','turn_degrees_mean'])
report+='''<h2>其他 loss 与 CMR 的位置</h2>
<p>在六个 checkpoint probe 中，KD 的加权梯度均为最大项，约 1.75–3.27，且与非 CMR 合计的余弦约 .80–.97。CE 后期约 1.36–1.73；实例对比约 .35–.67；CrossSDC-I 约 .18–.51；空间注意力约 .25–.41；类别对比约 .16–.24；时间注意力约 .014–.026。CrossSDC-C 系数为零，梯度贡献为零，即使其原始 loss 和原始梯度较大。</p>
<p>标量排名不能代替梯度排名。例如 A/step9 的加权 loss：CE=.552、KD=.104、CMR=.00408；对应梯度范数：CE=1.709、KD=3.265、CMR=.0626。KD 标量小于 CE，参数梯度却更大。空间注意力 loss 也很小，但梯度大于 CMR。CMR 对 classifier 的直接梯度为零；全模型与 feature_and_attention 的 CMR 相对尺度接近，当前“小梯度”不是单纯被 classifier 参数范围稀释。</p>
<img src="loss_and_gradient_scales.png">
<h2>新 probe 中的梯度方向</h2>
<p>CMR 与非 CMR 合计在这 18 个新 probe batch 中余弦均为正；checkpoint 均值约 .125–.213，说明在这些测量条件下弱同向，不能判断原训练是否存在冲突。它与实例对比及 CrossSDC-I 的后期余弦约 .24–.35，与 CE 为 .07–.19；与 KD 的平均相关较弱。与类别对比在 step5/9 的全部 probe batch 中呈小幅负相关；与空间/时间注意力在 step1 较明显负相关，后期大多接近正交或轻微负相关。负余弦是局部一阶冲突，不等同于性能损害。</p>
<p>把 CMR 加到其余目标后，A 的总梯度方向平均转动 .180°/1.295°/.900°，B 为 .472°/3.692°/3.753°。这些是欧氏参数梯度的瞬时变化，不是 Adam 更新角度，也不代表长期累积效果可以忽略。</p>
<img src="cmr_pairwise_alignment.png">'''
report+=table(pair_rows,['run','step','other_component','cosine_mean','cosine_min','cosine_max','negative_batches'])
report+='''<h2>必须区分原训练与新的 probe</h2>
<p>新的 memory 每阶段覆盖全部旧类，但不保证等于原 replay memory。A/step9 的加权 CMR，训练最佳 epoch 的 batch 均值约 .000214，probe 为 .00408（约 19.1 倍）；B/step9 对应 .000219 与 .01517（约 69.4 倍）。CrossSDC-I 和总 loss 也存在差异。训练数字是 epoch 内、不断变化的参数及原 batch 下的平均；probe 是最佳 checkpoint 下的新 batch、新 memory 和重建 Trust。不能将这些比值直接命名为过拟合程度，也不能将 probe 梯度当作当年训练末期的梯度。</p>'''
report+='''<p>去掉全局 lambda 后，step9 的原训练最佳 epoch 中 raw CMR 为 A=.00713、B=.00219；新 probe 为 A=.13598、B=.15168。这里的 raw 仅去掉全局 lambda，仍含 Trust 加权。由于回放样本及 CMR 参照一并变化，这个次序变化不能解释为保持能力的泛化变差。首先需要验证离线过程能复现原训练目标，而非先作泛化归因。</p>
<h2>实际采样与差异核查</h2>
<p>原训练在每阶段只为刚变成旧类的 10 个类别抽样，已有类别保留旧列表的前 k 个；k=500/(10×step) 的向下取整。随机流从训练 seed42 延续，候选按 metadata 原顺序排列。新 probe 则每个被测阶段重新创建 random.Random(42+step×100003)，将候选 ID 排序后，为全部旧类重新抽取 k 个。相同 seed 数字不等于相同 memory。</p>
<p>step1/5/9 分别使用 500/500/450 个新抽的 memory 样本，每类 50/10/5 个。各 batch 独立从当前新类 train 池抽 128 个、从新 memory 抽 128 个；batch 内不重复，跨 batch 可重复。step9 的 3 批共覆盖 291 个不同旧样本。两组实验的 manifest 相同，但它们都没有验证为原训练 memory。CMR 仅计算旧样本，原型和 Trust 用全部 memory 构建，不是只用 128 个 batch 样本构建。</p>
<p>step9 每类仅 5 个原型样本，正类原型还会排除当前样本，故只剩 4 个。换 memory 同时改变被测样本、正负类原型、教师参考 margin 和 Trust。hinge 的零阈值还使原均值非常接近零时出现很大的倍率。它们是能够造成系统差异的机制，现有输出不能定量分离各自贡献。</p>
<p>下表核对全部 3 个 probe batch 以及最佳 epoch 前后各 5 个 epoch 的原训练均值。step9 的每个 probe batch 都远高于原记录，且附近 11 个 epoch 的原记录持续很低，不能只用 batch 数少或挑中了一个异常低 epoch 来解释。</p>'''
report+=table(discrepancy_rows,['run','step','train_best_epoch_weighted_cmr','train_neighbor_min','train_neighbor_max','probe_batch_1_weighted_cmr','probe_batch_2_weighted_cmr','probe_batch_3_weighted_cmr'])
report+=table(history_rows,['run','step','best_epoch','test_accuracy_pct','train_epoch_mean_loss','probe_total_loss','train_epoch_mean_weighted_cmr','probe_weighted_cmr','probe_over_train_cmr','train_epoch_mean_weighted_cross_sdc','probe_weighted_cross_sdc'])
report+='''<h2>对 λ 与下一步的判断</h2>
<p>原训练 CSV 中的标量、原测试准确率仍有效；新 probe 的梯度、夹角与 λ 的瞬时缩放关系只在新 probe 条件下成立。当前数据没有回答“实际训练中 λ=.03/.1 分别有多强”的核心问题，也不能据此推荐调整 λ。A/B 的 λ 与 α 同时变化，教师也各自不同，不能作单参数因果比较。</p>
<p>不需要重复训练。下一步应先按原代码的逐阶段抽样、metadata 顺序、feature 可用性及随机流尝试重建原 memory，并用存档的 static_trust 和 epoch weights 核验；只凭 seed 不能宣称已恢复成功。再在保存的学生/教师 checkpoint 上，以原 memory、原型、权重和 batch size 先做便宜的标量对齐检查，对齐后才计算梯度。即使完成，也只解释保存状态附近的原目标梯度，不能恢复没有保存的每个训练时刻。当前分析没有执行新的模型推理或反向计算。</p>
<p>只有一个训练 seed，且 3 个 probe batch 存在样本重叠；图中范围表示 batch 波动，不是置信区间。实验自己的教师与重建权重也随实验而变。运行记录中的原始梯度和夹角已经足以离线推导同一 checkpoint 下其他 λ 的瞬时总梯度，但不能推导另一 λ 训练完成后的结果。</p>
<p>文件：cmr_summary.csv、loss_gradient_summary.csv、cmr_pairwise_summary.csv、training_probe_comparison.csv、probe_coverage.csv、discrepancy_audit.csv、audit.json。analyze_returned_probes.py 可复现本报告；原输入未修改。</p>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
for path, content in original.items():
    assert path.read_bytes() == content
    audit['input_files'][str(path)] = digest(content)
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(runs=audit['runs'],identical_manifests=True,configuration_differences=audit['configuration_differences'],output=str(OUT)),indent=2,ensure_ascii=False))
