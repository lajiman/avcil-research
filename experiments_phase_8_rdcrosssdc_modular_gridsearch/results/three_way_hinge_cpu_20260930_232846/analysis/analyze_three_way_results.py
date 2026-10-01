"""Audit returned three-way probes and derive endpoint diagnostics from saved CSVs.

No inference, training, or checkpoint deserialization. Run normally for tables;
run with --plots-only in a matplotlib environment for scientific figures.
"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
EXPERIMENT = BASE.parents[1]
REPO = EXPERIMENT.parent
SCOPE = 'feature_and_attention'
RUNS = {'A':'lambda_0p03_alpha_0p5', 'B':'lambda_0p1_alpha_1p0'}
GROUPS = ['original_memory', 'heldout_fixed_reference', 'fresh_memory']
GROUP_NAMES = dict(zip(GROUPS, ['原 memory', 'memory 外／原参考', '相同样本／新参考']))
COMPONENTS = ['ce','kd','instance_contrastive','class_contrastive','cross_sdc_i','cross_sdc_c','cmr','attn_spatial','attn_temporal']


def native(path):
    path = str(Path(path).resolve())
    return Path('\\\\?\\'+path) if os.name == 'nt' and not path.startswith('\\\\?\\') else Path(path)


def sha(path):
    h = hashlib.sha256()
    with native(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return pd.read_csv(native(path), encoding='utf-8-sig')


def close(a, b, *, atol=2e-6, rtol=2e-5):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


def local_source(server_path):
    marker = EXPERIMENT.name+'/'
    if marker in server_path:
        return EXPERIMENT/server_path.split(marker,1)[1]
    return REPO/'model'/Path(server_path).name


def scalar_summary(rows, fields):
    result = {}
    for field in fields:
        values = pd.to_numeric(rows[field], errors='raise')
        result[field+'_mean'] = values.mean()
        result[field+'_min'] = values.min()
        result[field+'_max'] = values.max()
        result[field+'_batch_sd'] = values.std(ddof=1)
    return result


def analyze():
    OUT.mkdir(exist_ok=True)
    batch_metadata = json.loads((BASE/'batch_metadata.json').read_text(encoding='utf-8-sig'))
    assert batch_metadata['status'] == 'complete' and len(batch_metadata['runs']) == 2
    mapping = np.load(native(REPO/'data2/balance/all_id_category_dict.npy'), allow_pickle=True).item()['train']
    encoding = np.load(native(REPO/'data2/balance/category_encode_dict.npy'), allow_pickle=True).item()
    label = lambda vid: int(encoding[mapping[vid]])
    overview, components_summary, effects, distribution, alignments, reference_shifts, accuracy = [], [], [], [], [], [], []
    manifests, sample_frames, all_argsets = {}, {}, {}
    audit = {'measurement':'checkpoint endpoint, not historical training gradients', 'scope':SCOPE,
             'runs':{}, 'input_sha256':{}, 'source_checks':[], 'checkpoint_checks':[], 'historical_checks':[]}
    for alias, directory in RUNS.items():
        folder = BASE/directory
        for path in sorted(folder.glob('*')):
            if path.is_file():
                audit['input_sha256'][str(path.relative_to(REPO))] = sha(path)
        metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8-sig'))
        assert metadata['status'] == 'complete' and not metadata['cli']['loss_only']
        assert metadata['optimizer_updates'] == 0 and not metadata['historical_gradient_reconstruction']
        assert metadata['model_mode'] == 'eval' and metadata['cli']['steps'] == [1,5,9] and metadata['cli']['batches'] == 3
        for server, expected in metadata['source_sha256'].items():
            path = local_source(server)
            content = native(path).read_bytes()
            matched = hashlib.sha256(content).hexdigest() == expected
            normalized = hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest() == expected
            assert matched or normalized, path
            audit['source_checks'].append(dict(run=alias, path=str(path.relative_to(REPO)), matches=True,
                                               newline_normalization_required=not matched))
        manifest = json.loads((folder/'probe_manifest.json').read_text(encoding='utf-8-sig'))
        manifests[alias] = manifest
        assert manifest['schema_version'] == 2
        for name, expected in manifest['metadata_sha256'].items():
            assert sha(REPO/'data2/balance'/name) == expected
        original_ids = json.loads((folder/'original_replay_ids.json').read_text(encoding='utf-8-sig'))
        c, s, b, g, p, checks, references, history, coverage = [read(folder/(name+'.csv')) for name in
            ['components','cmr_samples','cmr_batches','lambda_sweep','gradient_pairs','reference_checks',
             'class_reference','historical_comparison','coverage']]
        assert len(c)==729 and len(s)==3456 and len(b)==27 and len(g)==81 and len(p)==2916
        assert not c.duplicated(['experiment_group','step','batch','parameter_scope','component']).any()
        assert not s.duplicated(['experiment_group','step','batch','sample_index']).any()
        assert len(checks)==1050 and (checks.passed==1).all()
        close(checks.absolute_error, abs(checks.reconstructed-checks.historical), atol=1e-12, rtol=1e-8)
        assert (checks.absolute_error <= checks.atol+checks.rtol*np.maximum(abs(checks.reconstructed),abs(checks.historical))).all()
        assert np.isfinite(c[['raw_loss','weighted_loss','total_loss','raw_grad_norm','weighted_grad_norm','non_cmr_grad_norm']].to_numpy()).all()
        close(c.weighted_loss, c.coefficient*c.raw_loss)
        close(c.weighted_grad_norm, abs(c.coefficient)*c.raw_grad_norm)
        close(c.weighted_grad_over_non_cmr, c.weighted_grad_norm/c.non_cmr_grad_norm)
        assert (c[(c.component=='cmr')&(c.parameter_scope=='classifier')].raw_grad_norm==0).all()
        # Norms across disjoint parameter scopes reconcile by sum of squares.
        norms = c.pivot(index=['experiment_group','step','batch','component'],columns='parameter_scope',values='raw_grad_norm')
        close(norms['all']**2, norms.feature_and_attention**2+norms.classifier**2)
        for keys, rows in c.groupby(['experiment_group','step','batch','parameter_scope']):
            assert set(rows.component)==set(COMPONENTS)
            close(rows.weighted_loss.sum(), rows.total_loss.iloc[0])
            sw = g[(g.experiment_group==keys[0])&(g.step==keys[1])&(g.batch==keys[2])&(g.parameter_scope==keys[3])]
            assert len(sw)==1
            assert rows.gradient_reconstruction_residual.max() <= 2e-5+2e-4*float(g[(g.experiment_group==keys[0])&(g.step==keys[1])&(g.batch==keys[2])&(g.parameter_scope=='all')].total_grad_norm.iloc[0])
        all_argsets[alias] = metadata['states'][0]['args']
        for state in metadata['states']:
            step, args = state['step'], state['args']
            assert state['status']=='complete' and state['original_reference_validated'] and state['parameters_and_buffers_unchanged']
            archive = EXPERIMENT/'save_commands_cmr_hinge_tolerance_focus_3seeds'
            for role, stage in [('student',step),('teacher',step-1)]:
                path=archive/args['experiment_name']/f'step_{stage}_best_model.pkl'
                assert sha(path)==state['checkpoint_sha256'][role]
                audit['checkpoint_checks'].append(dict(run=alias,step=step,role=role,matches=True))
            for source in state['historical_sources'].values():
                path=local_source(source['path'])
                assert sha(path)==source['sha256']
                audit['historical_checks'].append(dict(run=alias,step=step,path=str(path.relative_to(REPO)),matches=True))
            plan=manifest['steps'][str(step)]
            original, fresh = plan['original_memory_ids'],plan['fresh_memory_ids']
            assert original==original_ids[str(step)] and not set(original)&set(fresh)
            assert len(original)==len(fresh)==(args['memory_size']//(step*10))*step*10
            assert sorted(map(label,original))==sorted(map(label,fresh))
            assert set(map(label,original))==set(range(step*10))
            assert len(set(original))==len(original) and len(set(fresh))==len(fresh)
            for index, batch in enumerate(plan['batches']):
                assert len(batch['current_ids'])==len(set(batch['current_ids']))==128
                assert all(step*10<=label(v)<step*10+10 for v in batch['current_ids'])
                assert list(map(label,batch['original_replay_ids']))==list(map(label,batch['fresh_replay_ids']))
                assert set(batch['original_replay_ids'])<=set(original) and set(batch['fresh_replay_ids'])<=set(fresh)
                for group in GROUPS:
                    ids=batch['original_replay_ids'] if group=='original_memory' else batch['fresh_replay_ids']
                    assert len(set(ids))==len(ids)==128
                    sr=s[(s.experiment_group==group)&(s.step==step)&(s.batch==index)].sort_values('sample_index')
                    assert sr.sample_id.tolist()==ids and sr.class_id.tolist()==list(map(label,ids))
                    assert (sr.in_reference_memory==(group!='heldout_fixed_reference')).all()
                    br=b[(b.experiment_group==group)&(b.step==step)&(b.batch==index)].iloc[0]
                    cr=c[(c.experiment_group==group)&(c.step==step)&(c.batch==index)&(c.parameter_scope==SCOPE)&(c.component=='cmr')].iloc[0]
                    expected_contribution=np.zeros(128)
                    reference=references[(references.step==step)&(references.reference==('fresh' if group=='fresh_memory' else 'original'))].set_index('class_id')
                    for d in ['a_from_v','v_from_a']:
                        drop=sr['ref_'+d]-sr['cur_'+d]
                        close(sr['margin_drop_'+d],drop)
                        close(sr['deficit_'+d],np.maximum(drop,0))
                        close(sr['violation_'+d],np.maximum(drop-args['rd_margin_tolerance'],0))
                        close(sr['penalty_'+d],sr['violation_'+d])
                        w=sr['weight_'+d].to_numpy()
                        close(w,reference.loc[sr.class_id,'cmr_weight_'+d].to_numpy(),atol=1e-8)
                        expected_contribution+=.5*w*sr['penalty_'+d].to_numpy()/w.sum()
                        close(br['active_'+d],(sr['violation_'+d]>0).mean(),atol=1e-12)
                        close(br['mean_violation_'+d],sr['violation_'+d].mean())
                    close(sr.raw_cmr_contribution,expected_contribution)
                    close(sr.raw_cmr_contribution.sum(),br.raw_cmr)
                    close(sr.weighted_cmr_contribution,args['lam_cmr']*sr.raw_cmr_contribution)
                    close(br.weighted_cmr,cr.weighted_loss)
                    close(br.raw_cmr,cr.raw_loss)
            # Verify the CSV's historical comparator directly, independently of its reported ratio.
            ep=read(local_source(state['historical_sources']['epochs']['path']))
            ep=ep[ep.step==step].sort_values('epoch'); best=ep.loc[ep.val_acc.idxmax()]
            hr=history[history.step==step]
            assert (hr.best_epoch==best.epoch).all()
            close(hr.train_epoch_mean_raw_cmr,best.cmr)
            close(hr.train_epoch_mean_weighted_cmr,best.weighted_cmr)
            rr=references[(references.reference=='original')&(references.step==step)].sort_values('class_id')
            wr=read(local_source(state['historical_sources']['weights']['path'])).sort_values('class_id')
            for direction in ['a_from_v','v_from_a']:
                close(rr['cmr_weight_'+direction],wr['cmr_weight_'+direction],atol=1e-8)
        # Groups 2/3 have identical non-CMR weighted objectives and norms (CrossSDC-C is disabled).
        fixed=c[c.experiment_group==GROUPS[1]].set_index(['step','batch','parameter_scope','component'])
        fresh=c[c.experiment_group==GROUPS[2]].set_index(['step','batch','parameter_scope','component'])
        mask=fixed.index.get_level_values('component')!='cmr'
        close(fixed.loc[mask,'weighted_loss'],fresh.loc[mask,'weighted_loss'],atol=1e-8)
        close(fixed.loc[mask,'weighted_grad_norm'],fresh.loc[mask,'weighted_grad_norm'],atol=1e-8)
        close(fixed.non_cmr_grad_norm,fresh.non_cmr_grad_norm,atol=1e-8)
        # Query margins should not depend on which companion samples occur in a batch.
        margin_spread=s.groupby(['experiment_group','step','sample_id'])[['ref_a_from_v','cur_a_from_v','ref_v_from_a','cur_v_from_a']].agg(lambda x:x.max()-x.min()).to_numpy().max()
        assert margin_spread<2e-5
        audit['runs'][alias]=dict(status='complete', component_rows=len(c), sample_occurrences=len(s),
            reference_checks=len(checks), reference_max_absolute_error=float(checks.absolute_error.max()),
            gradient_reconstruction_max_residual=float(c.gradient_reconstruction_residual.max()),
            repeated_sample_max_margin_difference=float(margin_spread))
        sample_frames[alias]=s
        cc=c[c.parameter_scope==SCOPE].copy()
        nonzero_cmr=cc[(cc.component=='cmr')&(cc.raw_grad_norm>0)]
        assert (nonzero_cmr.cosine_with_non_cmr>0).all()
        audit['runs'][alias]['nonzero_cmr_batches_with_positive_base_cosine']=len(nonzero_cmr)
        cc['rho_pct']=100*cc.weighted_grad_over_non_cmr
        cc['scalar_pct']=100*cc.weighted_loss/cc.total_loss
        gg=g[g.parameter_scope==SCOPE].copy()
        gg['turn_degrees']=np.degrees(np.arccos(np.clip(gg.total_cosine_with_non_cmr,-1,1)))
        for (step,group), rows in cc[cc.component=='cmr'].groupby(['step','experiment_group']):
            br=b[(b.step==step)&(b.experiment_group==group)]
            gr=gg[(gg.step==step)&(gg.experiment_group==group)]
            hr=history[(history.step==step)&(history.experiment_group==group)].iloc[0]
            item=dict(run=alias,step=int(step),experiment_group=group,lambda_cmr=float(rows.coefficient.iloc[0]),
                alpha=all_argsets[alias]['rd_class_weight_alpha'],batches=len(rows),
                **scalar_summary(rows,['raw_loss','weighted_loss','raw_grad_norm','weighted_grad_norm','non_cmr_grad_norm','rho_pct','scalar_pct','cosine_with_non_cmr']),
                active_pct_mean=50*(br.active_a_from_v.mean()+br.active_v_from_a.mean()),
                active_a_pct=100*br.active_a_from_v.mean(),active_v_pct=100*br.active_v_from_a.mean(),
                turn_degrees_mean=gr.turn_degrees.mean(),train_epoch_weighted_cmr=hr.train_epoch_mean_weighted_cmr,
                probe_over_train_epoch_cmr=hr.probe_over_train_epoch_cmr)
            overview.append(item)
        for (step,group,component), rows in cc.groupby(['step','experiment_group','component']):
            components_summary.append(dict(run=alias,step=int(step),experiment_group=group,component=component,
                **scalar_summary(rows,['raw_loss','weighted_loss','raw_grad_norm','weighted_grad_norm','rho_pct','cosine_with_non_cmr'])))
        for (step,group), rows in s.groupby(['step','experiment_group']):
            # De-duplicate query IDs for distribution diagnostics; do not treat repeats as independent data.
            unique=rows.drop_duplicates('sample_id')
            for direction in ['a_from_v','v_from_a','both']:
                directions=['a_from_v','v_from_a'] if direction=='both' else [direction]
                violations=np.concatenate([unique['violation_'+d].to_numpy() for d in directions])
                drops=np.concatenate([unique['margin_drop_'+d].to_numpy() for d in directions])
                active=violations[violations>0]
                distribution.append(dict(run=alias,step=int(step),experiment_group=group,direction=direction,
                    unique_samples=len(unique),old_classes_covered=unique.class_id.nunique(),directions_per_sample=len(directions),
                    active_pct=100*np.mean(violations>0),mean_violation=violations.mean(),
                    active_mean_violation=active.mean() if len(active) else 0.,
                    median_signed_drop=np.median(drops),p90_violation=np.quantile(violations,.9),
                    max_violation=violations.max()))
        pp=p[(p.parameter_scope==SCOPE)&((p.component_a=='cmr')|(p.component_b=='cmr'))].copy()
        pp['other']=np.where(pp.component_a=='cmr',pp.component_b,pp.component_a)
        for (step,group,other), rows in pp.groupby(['step','experiment_group','other']):
            alignments.append(dict(run=alias,step=int(step),experiment_group=group,other_component=other,
                raw_cosine_mean=rows.raw_cosine.mean(),raw_cosine_min=rows.raw_cosine.min(),raw_cosine_max=rows.raw_cosine.max(),
                weighted_cosine_mean=rows.weighted_cosine.mean(),defined_batches=rows.raw_cosine.notna().sum()))
        for step in [1,5,9]:
            for direction in ['a_from_v','v_from_a']:
                rr=references[(references.step==step)&(references.reference=='original')].sort_values('class_id')
                fr=references[(references.step==step)&(references.reference=='fresh')].sort_values('class_id')
                delta=fr['cmr_weight_'+direction].to_numpy()-rr['cmr_weight_'+direction].to_numpy()
                reference_shifts.append(dict(run=alias,step=step,direction=direction,
                    class_weight_mean_abs_change=np.abs(delta).mean(),class_weight_max_abs_change=np.abs(delta).max()))
        args=all_argsets[alias]
        for seed in [42,43,44]:
            name=args['experiment_name'].replace('_seed42',f'_seed{seed}')
            test=read(EXPERIMENT/'save_commands_cmr_hinge_tolerance_focus_3seeds'/'metrics'/name/'per_class_metrics.csv')
            final=test[test.step==9].overall_acc.unique()
            assert len(final)==1
            accuracy.append(dict(run=alias,seed=seed,final_accuracy_pct=100*final[0]))
    assert manifests['A']==manifests['B'], 'Settings must share exact query/reference IDs'
    audit['identical_manifests_across_settings']=True
    differences=[key for key in all_argsets['A'] if all_argsets['A'][key]!=all_argsets['B'].get(key)]
    assert set(differences)=={'experiment_name','lam_cmr','rd_class_weight_alpha'}
    audit['checkpoint_argument_differences']=differences
    table=pd.DataFrame(overview)
    for alias in RUNS:
        for step in [1,5,9]:
            rows=table[(table.run==alias)&(table.step==step)].set_index('experiment_group')
            original, fixed, fresh=(rows.loc[group] for group in GROUPS)
            effects.append(dict(run=alias,step=step,
                fixed_over_original_raw_loss=fixed.raw_loss_mean/original.raw_loss_mean,
                fresh_vs_fixed_loss_change_pct=100*(fresh.raw_loss_mean/fixed.raw_loss_mean-1),
                fixed_over_original_cmr_grad=fixed.weighted_grad_norm_mean/original.weighted_grad_norm_mean,
                fixed_over_original_base_grad=fixed.non_cmr_grad_norm_mean/original.non_cmr_grad_norm_mean,
                fixed_over_original_rho=fixed.rho_pct_mean/original.rho_pct_mean,
                fresh_vs_fixed_cmr_grad_change_pct=100*(fresh.weighted_grad_norm_mean/fixed.weighted_grad_norm_mean-1)))
    # Paired query diagnostics for changing only the reference system, de-duplicated.
    paired=[]
    for alias,s in sample_frames.items():
        for step in [1,5,9]:
            fixed=s[(s.step==step)&(s.experiment_group==GROUPS[1])].drop_duplicates('sample_id').set_index('sample_id').sort_index()
            fresh=s[(s.step==step)&(s.experiment_group==GROUPS[2])].drop_duplicates('sample_id').set_index('sample_id').sort_index()
            assert fixed.index.equals(fresh.index)
            for d in ['a_from_v','v_from_a']:
                x,y=fixed['violation_'+d],fresh['violation_'+d]
                paired.append(dict(run=alias,step=step,direction=d,unique_samples=len(x),
                    active_in_both=int(((x>0)&(y>0)).sum()),active_only_fixed=int(((x>0)&(y==0)).sum()),
                    active_only_fresh=int(((x==0)&(y>0)).sum()),inactive_in_both=int(((x==0)&(y==0)).sum()),
                    mean_violation_change=float((y-x).mean()),mean_absolute_violation_change=float(abs(y-x).mean())))
    frames={'cmr_summary':table,'component_scales':pd.DataFrame(components_summary),
        'controlled_effects':pd.DataFrame(effects),'unique_sample_distributions':pd.DataFrame(distribution),
        'cmr_pairwise_alignment':pd.DataFrame(alignments),'reference_weight_changes':pd.DataFrame(reference_shifts),
        'paired_reference_effects':pd.DataFrame(paired),'final_accuracy':pd.DataFrame(accuracy)}
    for name,frame in frames.items():
        frame.to_csv(OUT/(name+'.csv'),index=False)
    audit['all_checks_passed']=True
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    report()
    print(table[['run','step','experiment_group','raw_loss_mean','active_pct_mean','weighted_grad_norm_mean','rho_pct_mean']].to_string(index=False))
    print(frames['controlled_effects'].to_string(index=False))
    print(frames['final_accuracy'].to_string(index=False))
    print('All returned-data audits passed.')


def report():
    data=read(OUT/'cmr_summary.csv')
    effects=read(OUT/'controlled_effects.csv')
    scales=read(OUT/'component_scales.csv')
    alignment=read(OUT/'cmr_pairwise_alignment.csv')
    samples=read(OUT/'unique_sample_distributions.csv')
    accuracy=read(OUT/'final_accuracy.csv')
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    def table(frame, fields, labels, formats=None):
        formats=formats or {}
        body=[]
        for _,row in frame.iterrows():
            cells=[]
            for field in fields:
                value=row[field]
                if pd.isna(value): text='—'
                elif field=='experiment_group': text=GROUP_NAMES[value]
                elif field in formats: text=format(value,formats[field])
                else: text=str(value)
                cells.append('<td>'+html.escape(text)+'</td>')
            body.append('<tr>'+''.join(cells)+'</tr>')
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in labels)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    focus=data[data.step==9].copy()
    order={g:i for i,g in enumerate(GROUPS)}
    focus['order']=focus.experiment_group.map(order)
    focus=focus.sort_values(['run','order'])
    accuracy_summary=accuracy.groupby('run').final_accuracy_pct.agg(['mean','std']).reset_index()
    original=data[data.experiment_group=='original_memory']
    summary_table=table(focus,
        ['run','experiment_group','raw_loss_mean','active_pct_mean','weighted_grad_norm_mean','non_cmr_grad_norm_mean','rho_pct_mean','turn_degrees_mean'],
        ['设置','评估组','raw CMR','违反比例 %','CMR 梯度范数','非 CMR 合成梯度范数','梯度比 R %','加入 CMR 后转角 °'],
        {'raw_loss_mean':'.6f','active_pct_mean':'.2f','weighted_grad_norm_mean':'.5f','non_cmr_grad_norm_mean':'.4f','rho_pct_mean':'.2f','turn_degrees_mean':'.2f'})
    effect_table=table(effects,['run','step','fixed_over_original_raw_loss','fresh_vs_fixed_loss_change_pct','fixed_over_original_cmr_grad','fixed_over_original_rho'],
        ['设置','step','memory 外／原 memory loss 倍数','新参考相对原参考 loss 变化 %','memory 外／原 memory CMR 梯度倍数','梯度比 R 的倍数'],
        {'fixed_over_original_raw_loss':'.2f','fresh_vs_fixed_loss_change_pct':'.2f','fixed_over_original_cmr_grad':'.2f','fixed_over_original_rho':'.2f'})
    history_table=table(original,['run','step','train_epoch_weighted_cmr','weighted_loss_mean','probe_over_train_epoch_cmr'],
        ['设置','step','原 best epoch 的加权 CMR 均值','原 memory checkpoint 重算','重算／历史均值'],
        {'train_epoch_weighted_cmr':'.7f','weighted_loss_mean':'.7f','probe_over_train_epoch_cmr':'.3f'})
    dist=samples[(samples.step==9)&(samples.direction=='both')].copy()
    dist['order']=dist.experiment_group.map(order);dist=dist.sort_values(['run','order'])
    distribution_table=table(dist,['run','experiment_group','unique_samples','old_classes_covered','active_pct','active_mean_violation','median_signed_drop','p90_violation'],
        ['设置','评估组','不同查询数','旧类别覆盖数','去重后违反比例 %','活跃样本平均违反幅度','有符号 margin 差的中位数','violation P90'],
        {'active_pct':'.2f','active_mean_violation':'.4f','median_signed_drop':'.4f','p90_violation':'.4f'})
    component_rows=scales[(scales.step==9)&(scales.experiment_group=='original_memory')]
    component_table=table(component_rows,['run','component','weighted_loss_mean','weighted_grad_norm_mean'],
        ['设置','loss 项','加权 loss 均值','加权梯度范数均值'],{'weighted_loss_mean':'.6f','weighted_grad_norm_mean':'.5f'})
    align=alignment[(alignment.step==9)&(alignment.other_component.isin(['ce','kd','class_contrastive','cross_sdc_i']))]
    alignment_table=table(align,['run','experiment_group','other_component','raw_cosine_mean','raw_cosine_min','raw_cosine_max'],
        ['设置','评估组','与 CMR 比较的项','余弦均值','三个 batch 最小值','三个 batch 最大值'],
        {'raw_cosine_mean':'.3f','raw_cosine_min':'.3f','raw_cosine_max':'.3f'})
    accuracy_table=table(accuracy_summary,['run','mean','std'],['设置','三个训练 seed 的最终准确率均值 %','seed 标准差 pp'],{'mean':'.2f','std':'.2f'})
    checked=sum(r['reference_checks'] for r in audit['runs'].values())
    maximum=max(r['reference_max_absolute_error'] for r in audit['runs'].values())
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三组离线评估：回放约束与样本外保持</title>
<style>body{{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#192333;background:#fff;margin:32px auto;max-width:1220px;padding:0 24px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:34px}}p{{max-width:1050px}}table{{border-collapse:collapse;font-size:14px;width:100%;font-variant-numeric:tabular-nums}}th{{background:#e9eef5;text-align:left}}th,td{{padding:9px 12px;border-bottom:1px solid #dbe2ea;white-space:nowrap}}td{{text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.scroll{{overflow-x:auto;margin:18px 0}}img{{max-width:100%;height:auto}}.note{{font-size:14px;color:#4e5969}}a{{color:#1c57a3}}li{{margin:6px 0}}</style>
<h1>三组离线评估：回放约束与样本外保持</h1>
<p>A：λ=0.03、α=0.5；B：λ=0.1、α=1。两组均为 hinge、tol=0.01、seed42，评估 step1、5、9 的最佳 checkpoint。每个阶段、每个协议测 3 个 batch，每批 128 个当前类样本和 128 个旧类样本。</p>
<p><strong>主要发现：</strong>原 memory 上的重算接近历史记录；固定原参考体系后，memory 外样本的 CMR 已明显升高。重建原型与权重引起的均值变化相对小。B 对原 memory 的约束满足更充分，但在所测 memory 外样本上没有呈现同等幅度的保持收益。loss 的几十倍差异不对应梯度的几十倍差异。</p>
<h2>数据来源与有效性</h2>
<p>两次运行和六个 checkpoint 状态均完成；{checked:,} 个历史统计字段全部通过校验，最大绝对误差为 {maximum:.3g}。本分析独立核对了 checkpoint、历史 CSV、元数据及测量源码哈希，逐样本 margin→hinge→batch CMR 的重建、各 loss 的总量、梯度分解，以及采样清单。未加载或更新模型参数。</p>
<p>两种设置使用完全相同的样本 ID。新 memory 与原 memory 不重合，三组每批的旧类别标签序列与当前类样本相同。第二、三组查询 ID 完全相同。第二组使用完整原参考原型，第三组对新 memory 内查询使用正确的 leave-one-out。这里“memory 外”是旧类 train 样本，不是测试集，也不等于历史上从未训练过的样本。</p>
<p class="note">“原 memory”身份由采样流程重建并通过保存的 count、Reliability、Trust、权重指纹校验；没有伪称找到了训练时保存的原始 ID 日志。审计明细：<a href="audit.json">audit.json</a>。</p>
<h2>step9：原约束满足与 memory 外保持</h2>
{summary_table}
<p class="note">raw CMR 去掉全局 λ，仍保留方向内的 Trust 类别权重。违反比例是两个方向的均值，按采样出现次数统计。梯度范围统一为特征层与注意力层，排除分类器；R = ‖λ∇L_CMR‖ / ‖∇Σ非CMR加权loss‖，不能视为可相加的更新百分比，也不等于 Adam 的实际更新比例。</p>
<p>固定原参考体系后，A 的 raw CMR 从 0.006528 升到 0.166550（25.51 倍），B 从 0.002336 升到 0.167305（71.63 倍）。违反比例分别从 17.32% / 9.24% 升到约 59%。这使“固定 replay 上满足的约束没有同等延伸到其他旧类样本”成为本批数据支持的现象，而不仅是旧探针混合改变样本、原型后的猜测。</p>
<p>对完全相同的 memory 外查询，改用新参考体系后，A 的 loss 下降 13.71%，B 下降 7.42%。这批采样中，更换参考体系不是几十倍差异的主要来源；不代表任何原型采样都不会产生较大影响。</p>
<figure><img src="cmr_three_way_comparison.png" alt="三个阶段三种评估下的 CMR loss、违反比例和梯度相对尺度"><figcaption class="note">线上的点为三个 batch 的均值；只有一份新 memory，不把这些点当作多次独立 memory 重采样。</figcaption></figure>
{effect_table}
<p>差距在 step5/9 更突出，伴随每类 memory 从 50 个减少到 10、5 个。但类别数、训练历史、teacher 和阶段难度也同时变化，不能仅凭这三个阶段把原因单独归为 memory 容量。</p>
<h2>不是少数重复样本造成的表象</h2>
{distribution_table}
<p>step9 的原 memory 与 memory 外评估分别覆盖 285、291 个不同样本，两边都覆盖 90 个旧类。去掉 batch 间重复查询后，memory 外仍有约 58%–59% 的方向约束违反；活跃 violation 的平均幅度约 0.28–0.29，而原 memory 为约 0.038 / 0.025。违反更普遍，也更严重。这些是一个固定采样集合上的描述，不是对整个旧类分布的置信区间。</p>
<h2>原 memory 检查有意义，但不能恢复训练轨迹</h2>
{history_table}
<p>原 memory 重算与原 best epoch 均值的比值约为 0.916–1.174。原记录是更新过程中的 batch 均值，checkpoint 是该 epoch 结束后的状态，二者无需相等。这里没有再出现 19 倍或 69 倍的系统差异，支持本次原参考恢复的有效性。</p>
<p>最佳 checkpoint 按验证准确率选择，并非每项 loss 的极值。step9 的原 memory 上，CMR 的加权梯度范数仍分别为 0.03719、0.10943。CMR loss 很小并没有让该项梯度归零；总目标驻点也不要求各分项梯度分别为零。</p>
<h2>λ 对应的梯度尺度</h2>
<p>step9，原 memory 上 R 为 A 1.31%、B 4.33%；memory 外且固定参考时为 A 1.75%、B 6.77%。B 在原 memory 上的加权 CMR loss 与 A 接近，但其 CMR 梯度约为 A 的 2.94 倍。两个模型自身的 raw 梯度不同，因此不能把这个比值简单当作全局 λ 比值。</p>
<p>原 memory 换成 memory 外样本时，CMR loss 分别增大 25.51 / 71.63 倍，CMR 梯度仅增大 1.86 / 2.17 倍。hinge 在活跃区域的导数不随 violation 幅度线性增大；梯度还受激活样本数、特征 Jacobian 和方向抵消影响。与此同时非 CMR 合成梯度也增大，所以 R 的变化更小。</p>
<figure><img src="step9_component_gradients.png" alt="step9 各 loss 的加权梯度范数"><figcaption class="note">每根柱表示单个分项梯度范数；不能将这些范数直接相加得到总梯度范数。</figcaption></figure>
{component_table}
<p>原 memory 上，类别对比的加权 loss 最大（约 0.117），但 KD 梯度范数最大（A 2.746、B 2.434）。因此不能按 loss 数值大小判断优化影响。CMR 在这些终点测量中是较小的附加梯度，却并非零作用。</p>
<h2>梯度方向：合成方向一致不等于每项都一致</h2>
<p>在所有 CMR 梯度非零的被测 batch 中，CMR 与非 CMR 合成梯度的余弦为正。step9 的原 memory 上，CMR 与 CE 的平均余弦为 A −0.121、B −0.111，与类别对比也略为负；与 KD 则约为 +0.20。CMR 与合成方向同向，部分来自较大的 KD 梯度，并不意味着它和分类目标不存在局部冲突。</p>
{alignment_table}
<h2>如何比较两种设置</h2>
<p>B 的原 memory raw CMR 比 A 低约 64%，但固定原参考的 memory 外 raw CMR 几乎相同（0.167305 对 0.166550）。本批结果更支持“较强设置在原 replay 上约束得更充分，收益没有同等延伸”，不足以支持“λ 更大所以整体保持更好”。λ 与 α 同时变化，后续阶段各自的 teacher 也不同，不能拆成 λ 的单独因果效应。</p>
{accuracy_table}
<p>历史三 seed 的最终分类准确率，A 为 40.31%±0.36pp，B 为 39.61%±0.75pp；当前 probe 的 seed42 分别为 39.90%、38.98%。因此若只在这两个既有设置中选后续重点，A 更适合作为性能参考，B 适合作为“原 memory 约束更强”的对照。三 seed 的标准差不是差异显著性的证明。</p>
<h2>接下来优先做什么</h2>
<p>先利用已经保存的逐样本 margin、类别 Trust 和权重，按旧类别所属阶段、Trust 高低以及违反幅度拆分，检查强约束的收益是否集中在特定类别，以及哪类旧样本的损失最大。这一步不需要重新训练，也不需要再次加载 checkpoint。若后续需要更稳健的总体判断，应增加独立的 memory 外采样或参考库，而不是只在同一份 memory 内增加重叠 batch。</p>
<p class="note">边界：这些结论对应最佳 checkpoint 的局部性质；不还原未保存 epoch 的梯度，不证明优化过程的累计因果效应，也不把 raw CMR 直接等同于分类准确率。原 memory 内查询使用 LOO、外部查询使用完整参考原型，是当前协议的规定；需要更严格隔离此项时，应另做统一目标原型策略的敏感性检查。</p>
<h2>数据文件</h2><ul>
<li><a href="cmr_summary.csv">CMR 三组对比与 batch 范围</a>；<a href="component_scales.csv">各 loss 的尺度与梯度</a></li>
<li><a href="controlled_effects.csv">更换查询和参考体系的变化</a>；<a href="unique_sample_distributions.csv">去重后的违反分布</a></li>
<li><a href="cmr_pairwise_alignment.csv">CMR 与各项梯度的夹角</a>；<a href="paired_reference_effects.csv">相同查询更换参考后的激活变化</a></li>
<li><a href="reference_weight_changes.csv">参考权重变化</a>；<a href="final_accuracy.csv">历史三 seed 的准确率</a></li>
<li>原始返回记录：<a href="../lambda_0p03_alpha_0p5/metadata.json">A metadata</a>、<a href="../lambda_0p1_alpha_1p0/metadata.json">B metadata</a>；<a href="../lambda_0p03_alpha_0p5/cmr_samples.csv">A 逐样本记录</a>、<a href="../lambda_0p1_alpha_1p0/cmr_samples.csv">B 逐样本记录</a></li>
</ul></html>'''
    (OUT/'report.html').write_text(content,encoding='utf-8')


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=read(OUT/'cmr_summary.csv')
    colors=['#2563eb','#d97706','#16a34a']
    names=['Original memory / original reference','Outside memory / original reference','Same outside queries / new reference']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14.5,8),layout='constrained')
    for row,alias in enumerate(RUNS):
        for col,(field,title) in enumerate([('raw_loss_mean','Raw CMR loss (log scale)'),
                                            ('active_pct_mean','Constraint violations (%)'),
                                            ('rho_pct_mean','CMR / non-CMR gradient norm (%)')]):
            ax=axes[row,col]
            for group,color,name in zip(GROUPS,colors,names):
                values=data[(data.run==alias)&(data.experiment_group==group)].sort_values('step')
                ax.plot(values.step,values[field],marker='o',color=color,label=name,linewidth=1.8)
            ax.set_title(title if row==0 else '')
            ax.set_xticks([1,5,9]);ax.set_xlabel('Incremental step');ax.grid(alpha=.2)
            if col==0:
                ax.set_yscale('log');ax.set_ylabel('A: lambda=0.03, alpha=0.5' if alias=='A' else 'B: lambda=0.1, alpha=1')
            else: ax.set_ylim(bottom=0)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Best-checkpoint diagnostics: three matched protocols\nMeans of 3 batches; one memory draw per step; gradients over features + attention',fontsize=13)
    fig.savefig(OUT/'cmr_three_way_comparison.png',dpi=170)
    plt.close(fig)
    scales=read(OUT/'component_scales.csv')
    selected=['ce','kd','instance_contrastive','class_contrastive','cross_sdc_i','cmr','attn_spatial','attn_temporal']
    labels=['CE','KD','Inst. contrast','Class contrast','CrossSDC-I','CMR','Spatial KL','Temporal KL']
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
    x=np.arange(len(selected))
    for ax,alias in zip(axes,RUNS):
        for i,(group,color,name) in enumerate(zip(GROUPS,colors,names)):
            v=scales[(scales.run==alias)&(scales.step==9)&(scales.experiment_group==group)].set_index('component').loc[selected]
            ax.bar(x+(i-1)*.24,v.weighted_grad_norm_mean,width=.23,color=color,label=name)
        ax.set_xticks(x,labels,rotation=32,ha='right');ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
        ax.set_title('A: lambda=0.03, alpha=0.5' if alias=='A' else 'B: lambda=0.1, alpha=1')
        ax.set_ylabel('Weighted component gradient norm (log scale)')
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Step 9: individual loss gradients at the saved checkpoint\nFeature + attention parameters; CrossSDC-C omitted because its coefficient is zero',fontsize=13)
    fig.savefig(OUT/'step9_component_gradients.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',action='store_true')
    cli=parser.parse_args()
    plots() if cli.plots_only else analyze()
