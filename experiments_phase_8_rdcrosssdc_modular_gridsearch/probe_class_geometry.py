"""Step 2: read-only checkpoint geometry and confusion analysis (CPU default).

Fixed TRAIN reference IDs, disjoint TEST queries, shared by methods/steps/seeds.
This is not a reconstruction of original replay memory or training gradients.
"""
import argparse
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import random
import sys
import time

import class_analysis_common as common

sys.path.insert(0, str(common.HERE.parent))


def make_plan(args, order):
    import numpy as np
    root = common.native(args.meta_root)
    names = np.load(root/'category_encode_dict.npy', allow_pickle=True).item()
    by_class = np.load(root/'all_classId_vid_dict.npy', allow_pickle=True).item()
    by_id = np.load(root/'all_id_category_dict.npy', allow_pickle=True).item()
    if names != {name:c for c,name in enumerate(order)}:
        raise ValueError('Feature metadata class mapping differs from saved metrics')
    plan = dict(sample_seed=args.sample_seed, reference_split='train', query_split='test',
                reference_ids=[], reference_labels=[], query_ids=[], query_labels=[])
    for c in range(100):
        for split, prefix, count in [('train','reference',args.reference_per_class),
                                     ('test','query',args.query_per_class)]:
            pool = sorted(by_class[split].get(str(c), by_class[split].get(c, [])))
            if len(pool) != len(set(pool)) or not pool:
                raise ValueError(f'Empty/duplicate {split} class {c}')
            if count > len(pool):
                raise ValueError(f'{split} class {c} has {len(pool)}, requested {count}')
            for vid in pool:
                if by_id[split][vid] != order[c]:
                    raise ValueError('Sample/category mismatch: '+vid)
            selected = pool if count == 0 else sorted(random.Random(args.sample_seed+c*1009+(split=='test')).sample(pool,count))
            plan[prefix+'_ids'].extend(selected)
            plan[prefix+'_labels'].extend([c]*len(selected))
    if set(plan['reference_ids']) & set(plan['query_ids']):
        raise ValueError('TRAIN reference and TEST query IDs overlap')
    for split in ['reference','query']:
        if len(set(plan[split+'_ids'])) != len(plan[split+'_ids']):
            raise ValueError('Duplicate IDs across classes')
    return plan


def load_runtime(args):
    global np, torch, geom
    # Set before importing NumPy/PyTorch; geometry matrix products use BLAS too.
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(args.threads)
    import numpy as np
    import torch
    import class_geometry_math as geom
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')


def feature_store(args):
    # Reuse the existing read-only, retrying HDF5 reader, without importing trainer.
    from dataloader_ours import _LazyVisualH5Mixin
    class Store(_LazyVisualH5Mixin):
        def __init__(self):
            self.args = argparse.Namespace(dataset='VGGSound', h5_read_retries=5, h5_retry_delay=2.)
            self.modality = 'audio-visual'
            self.visual_pretrained_feature_path = str(common.native(args.feature_root)/'visual_features.h5')
            self.all_visual_pretrained_features = None
            self._visual_h5_owner_pid = None
            self.audio = np.load(common.native(args.feature_root)/'audio_pretrained_feature/audio_pretrained_feature_dict.npy', allow_pickle=True).item()
        def batch(self, ids):
            visual = np.stack([np.asarray(self._visual_feature(i), dtype=np.float32) for i in ids])
            audio = np.stack([np.asarray(self.audio[i], dtype=np.float32) for i in ids])
            if not np.isfinite(visual).all() or not np.isfinite(audio).all():
                raise ValueError('Nonfinite input features')
            return visual, audio
    return Store()


def subset_plan(plan, nclasses):
    result = {}
    for split in ['reference','query']:
        selected = [(i,c) for i,c in zip(plan[split+'_ids'],plan[split+'_labels']) if c<nclasses]
        result[split+'_ids'] = np.asarray([i for i,c in selected])
        result[split+'_labels'] = np.asarray([c for i,c in selected], dtype=np.int64)
    return result


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key:data[key] for key in data.files}


def checkpoint_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_model(model, run, step):
    args = model.args
    expected = {'avcil':(0.,0.,.5), 'hinge_A':(.1,.03,.5), 'hinge_B':(.1,.1,1.)}[run['method']]
    actual = (args.lam_cross_sdc_i,args.lam_cmr,args.rd_class_weight_alpha)
    # alpha has no effect in AVCIL; still audit the established saved configuration.
    if actual != expected or args.lam_cross_sdc_c != 0 or args.class_num_per_step != 10 or args.num_classes != 100:
        raise ValueError('Unexpected saved objective/configuration: '+run['name'])
    if args.seed != run['seed'] or args.modality != 'audio-visual':
        raise ValueError('Checkpoint identity mismatch')
    if run['method'] != 'avcil' and (args.rd_cmr_penalty != 'hinge' or args.rd_margin_tolerance != .01):
        raise ValueError('Expected hinge tolerance .01')
    if getattr(args,'z1_cm_projection_head',False):
        raise ValueError('Projection-head variants need an explicit geometry protocol')
    for key, expected_value in dict(lam_I=.1,lam_C=1.,lam=.5,memory_size=500,
        instance_contrastive=True,class_contrastive=True,attn_score_distil=True).items():
        if getattr(args,key,None) != expected_value:
            raise ValueError('Unmatched baseline configuration: '+key)
    return copy.deepcopy(vars(args))


def extract_input(store, plan, args, out):
    path = out/'cache/input_features.npz'
    if path.exists():
        return read_npz(path)
    arrays = subset_plan(plan, 100)
    last = time.monotonic()
    for split in ['reference','query']:
        collected = {'audio':[], 'visual':[]}
        ids = arrays[split+'_ids']
        for start in range(0,len(ids),args.batch_size):
            visual, audio = store.batch(ids[start:start+args.batch_size])
            visual = visual.reshape(len(visual),-1,visual.shape[-1]).mean(1)
            for name,value in [('audio',audio),('visual',visual)]:
                collected[name].append(geom.normalize(value).astype(np.float32))
            if time.monotonic()-last > 30:
                print(f'Input {split}: {min(start+args.batch_size,len(ids))}/{len(ids)}', flush=True)
                last = time.monotonic()
        arrays.update({split+'_'+k:np.concatenate(v) for k,v in collected.items()})
    save_npz(path,arrays)
    print('Fixed input features cached',flush=True)
    return arrays


def cache_path(out, run, step):
    return out/'cache'/f"{run['method']}_seed{run['seed']}_step{step}.npz"


def extract_states(store, plan, runs, steps, args, out):
    """Share each HDF5 batch across pending models at the same seed/step."""
    needed = sorted(set(steps) | {t-1 for t in steps})
    for seed in args.seeds:
        for step in needed:
            pending = [r for r in runs if r['seed']==seed and not cache_path(out,r,step).exists()]
            if not pending:
                print(f'Cached seed {seed}, step {step}',flush=True)
                continue
            models, buffers, state_hashes, saved_args = {}, {}, {}, {}
            for r in pending:
                name = r['method']
                path = common.native(r['checkpoints'])/f'step_{step}_best_model.pkl'
                model = torch.load(path, map_location='cpu', weights_only=False)
                saved_args[name] = validate_model(model,r,step)
                model.eval().to(args.device)
                state_hashes[name] = checkpoint_hash(model)
                models[name] = model
                buffers[name] = {}
            selected = subset_plan(plan,(step+1)*10)
            last = time.monotonic()
            with torch.inference_mode():
                for split in ['reference','query']:
                    ids = selected[split+'_ids']
                    collected = {r['method']:{k:[] for k in ['audio','visual','fusion','logits']} for r in pending}
                    for start in range(0,len(ids),args.batch_size):
                        visual,audio = store.batch(ids[start:start+args.batch_size])
                        visual = torch.from_numpy(visual).to(args.device)
                        audio = torch.from_numpy(audio).to(args.device)
                        for name,model in models.items():
                            result = model(visual=visual,audio=audio,return_dict=True,out_analysis_features=True)
                            values = dict(audio=result['z1_audio_norm'],visual=result['z1_visual_norm'],
                                          fusion=result['z2_fusion_norm'],logits=result['logits'])
                            if values['logits'].shape[1] != (step+1)*10:
                                raise ValueError('Classifier size does not match step')
                            for key,value in values.items():
                                value = value.cpu().numpy().copy()
                                if not np.isfinite(value).all():
                                    raise ValueError('Nonfinite checkpoint output')
                                if key != 'logits' or split == 'query':
                                    collected[name][key].append(value)
                        if time.monotonic()-last > 30:
                            print(f'Seed {seed} step {step} {split}: {min(start+args.batch_size,len(ids))}/{len(ids)}; {len(models)} models',flush=True)
                            last = time.monotonic()
                    for name in models:
                        buffers[name].update({split+'_'+k:np.concatenate(v) for k,v in collected[name].items() if v})
            for r in pending:
                name = r['method']
                if checkpoint_hash(models[name]) != state_hashes[name]:
                    raise RuntimeError('Parameters or buffers changed during inference')
                buffers[name].update(selected)
                buffers[name]['saved_args_json'] = np.asarray(json.dumps(saved_args[name]))
                buffers[name]['state_unchanged'] = np.asarray(True)
                save_npz(cache_path(out,r,step),buffers[name])
            del models, buffers
            print(f'Completed extraction: seed {seed}, step {step}',flush=True)


def analyze_inputs(data, order, ranks, args, out):
    rows, protos = [], {}
    for feature in ['audio','visual']:
        values, _, proto = geom.geometry(data['reference_'+feature],data['reference_labels'],
            data['query_'+feature],data['query_labels'],100,args.knn_k)
        protos[feature] = proto
        rows.extend(dict(feature=feature,candidate_classes=100,
                         **{**common.class_info(r['class_id'],9,order,ranks),**r}) for r in values)
    similarities = {k:v@v.T for k,v in protos.items()}
    agreement, pairs, proximity = [], [], []
    for c in range(100):
        mask = np.arange(100)!=c
        agreement.append(dict(class_id=c,category_name=order[c],
            audio_visual_neighbor_profile_spearman=geom.profile_correlation(similarities['audio'][c,mask],similarities['visual'][c,mask])))
        for other in range(c+1,100):
            pairs.append(dict(class_id=c,other_class_id=other,category_name=order[c],other_category=order[other],
                audio_similarity=float(similarities['audio'][c,other]),visual_similarity=float(similarities['visual'][c,other])))
    for step in range(1,10):
        for c in range(step*10):
            row = dict(step=step,**common.class_info(c,step,order,ranks))
            for feature,sim in similarities.items():
                new = np.arange(step*10,(step+1)*10)
                nearest = int(new[sim[c,new].argmax()])
                row.update({feature+'_nearest_new_class':nearest,feature+'_nearest_new_name':order[nearest],
                            feature+'_max_new_similarity':float(sim[c,nearest])})
            proximity.append(row)
    for filename, values in [('input_geometry.csv',rows),('input_modal_agreement.csv',agreement),
                              ('input_class_pairs.csv',pairs),('new_class_proximity.csv',proximity)]:
        common.write_csv(out/filename,values)
    return {(r['feature'],r['class_id']):r for r in rows}


def analyze_state(data, teacher, run, step, order, ranks, tables, args):
    nclasses = (step+1)*10; old = step*10
    labels = data['query_labels']
    context = dict(method=run['method'],seed=run['seed'],step=step)
    geometry_rows, samples, retention_rows, retention_samples, cross_rows = [], [], [], [], []
    for feature in ['audio','visual','fusion']:
        values, scores, _ = geom.geometry(data['reference_'+feature],data['reference_labels'],
            data['query_'+feature],labels,nclasses,args.knn_k)
        geometry_rows.extend(dict(**context,feature=feature,
            **{**common.class_info(r['class_id'],step,order,ranks),**r}) for r in values)
        for i,vid in enumerate(data['query_ids']):
            samples.append(dict(**context,feature=feature,sample_id=str(vid),class_id=int(labels[i]),
                centroid_margin=float(scores['hard_margin'][i]),knn_purity=float(scores['knn_purity'][i]),
                centroid_prediction=int(scores['predicted'][i]),hard_negative_class=int(scores['hard_negative'][i])))
    # Absolute cross-modal separability: prototypes rebuilt within EACH checkpoint.
    for query_feature,ref_feature,direction in [('audio','visual','a_from_v'),('visual','audio','v_from_a')]:
        proto,_ = geom.prototypes(data['reference_'+ref_feature],data['reference_labels'],nclasses)
        scores = geom.margin_scores(data['query_'+query_feature],proto,labels,args.margin_temperature)
        for c in range(nclasses):
            mask = labels==c
            cross_rows.append(dict(**context,direction=direction,**common.class_info(c,step,order,ranks),
                candidate_classes=nclasses,mean_logodds_margin=float(scores['logodds_margin'][mask].mean()),
                mean_hard_margin=float(scores['hard_margin'][mask].mean()),
                centroid_accuracy=float((scores['predicted'][mask]==c).mean())))
        old_mask = labels<old
        if not np.array_equal(data['query_ids'][old_mask],teacher['query_ids']):
            raise ValueError('Teacher and student query IDs differ')
        if not np.array_equal(data['reference_ids'][data['reference_labels']<old],teacher['reference_ids']):
            raise ValueError('Teacher and student reference IDs differ')
        measures = geom.retention(teacher['query_'+query_feature],data['query_'+query_feature][old_mask],
            teacher['reference_'+ref_feature],teacher['reference_labels'],teacher['query_labels'],old,
            args.margin_temperature,args.retention_tolerance)
        for c in range(old):
            mask = teacher['query_labels']==c
            retention_rows.append(dict(**context,direction=direction,**common.class_info(c,step,order,ranks),
                candidate_classes=old,reference_kind='fixed_train_reference_test_query',
                **{'mean_'+k:float(v[mask].mean()) for k,v in measures.items()},
                p90_violation=float(np.quantile(measures['violation'][mask],.9))))
        for i,vid in enumerate(teacher['query_ids']):
            retention_samples.append(dict(**context,direction=direction,sample_id=str(vid),
                class_id=int(teacher['query_labels'][i]),**{k:float(v[i]) for k,v in measures.items()}))
    prf, matrix = geom.confusion_rows(labels,data['query_logits'].argmax(1),nclasses)
    performance = []
    restricted = data['query_logits'][:,:old].argmax(1)
    for r in prf:
        c = r['class_id']; history = tables[run['method'],run['seed']][step,c]
        performance.append(dict(**context,**{**common.class_info(c,step,order,ranks),**r},
            historical_f1=history['f1'],historical_support=history['support'],
            full_test_query=args.query_per_class==0,
            historical_counts_match=all(r[k]==history[k] for k in ['tp','fp','fn']) if args.query_per_class==0 else None,
            old_head_only_recall=float((restricted[labels==c]==c).mean()) if c<old else None,
            old_to_new_error_rate=float(matrix[c,old:].sum()/r['support']) if c<old else None))
    confusion = [dict(**context,true_class=c,predicted_class=d,true_category=order[c],predicted_category=order[d],
        true_is_old=c<old,predicted_is_old=d<old,count=int(matrix[c,d]))
        for c,d in zip(*np.nonzero(matrix))]
    return dict(geometry=geometry_rows,query_geometry=samples,retention=retention_rows,
                query_retention=retention_samples,cross_modal_geometry=cross_rows,
                performance=performance,confusion=confusion),matrix


def run(args):
    load_runtime(args)
    ranks = common.difficulty(args.difficulty)
    runs = common.discover_runs(args.archive_root,args.seeds)
    tables,order = common.load_metrics(runs,ranks)
    plan = make_plan(args,order)
    needed = sorted(set(args.steps)|{s-1 for s in args.steps})
    checkpoint_sources = {r['name']:{str(s):common.sha256(common.native(r['checkpoints'])/f'step_{s}_best_model.pkl') for s in needed} for r in runs}
    source_paths = [common.HERE/p for p in ['probe_class_geometry.py','class_geometry_math.py','class_analysis_common.py','dataloader_ours.py']]
    source_paths += [common.HERE.parent/'model/audio_visual_model_incremental.py',common.HERE.parent/'model/layers.py']
    feature_files = [common.native(args.feature_root)/p for p in ['visual_features.h5','audio_pretrained_feature/audio_pretrained_feature_dict.npy']]
    config = dict(options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['resume','output']},
        difficulty=ranks,plan=plan,checkpoints=checkpoint_sources,
        metric_hashes={r['name']:common.sha256(Path(r['metrics'])/'per_class_metrics.csv') for r in runs},
        source_hashes={str(p.relative_to(common.HERE.parent)):common.sha256(p) for p in source_paths},
        metadata_hashes={p.name:common.sha256(p) for p in common.native(args.meta_root).glob('*.npy')},
        feature_identity={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in feature_files},
        feature_identity_note='Feature stores checked by path/size/mtime, not full content SHA256',
        numpy_version=np.__version__,torch_version=torch.__version__)
    out = common.prepare_output(args.output,config,args.resume)
    common.write_json(out/'sample_manifest.json',plan)
    store = feature_store(args)
    try:
        # Fail before expensive inference on a missing feature; never silently skip.
        for vid in plan['reference_ids']+plan['query_ids']:
            if vid not in store.audio or not store._has_visual_feature(vid):
                raise ValueError('Missing feature: '+vid)
        input_data = extract_input(store,plan,args,out)
        input_geometry = analyze_inputs(input_data,order,ranks,args,out)
        del input_data
        extract_states(store,plan,runs,args.steps,args,out)
    finally:
        store.close_visual_features_h5()
    all_rows = {k:[] for k in ['geometry','query_geometry','retention','query_retention','cross_modal_geometry','performance','confusion']}
    matrices = {}
    for run_info in runs:
        for step in args.steps:
            data = read_npz(cache_path(out,run_info,step))
            teacher = read_npz(cache_path(out,run_info,step-1))
            rows,matrix = analyze_state(data,teacher,run_info,step,order,ranks,tables,args)
            for key,values in rows.items():
                all_rows[key].extend(values)
            matrices[run_info['method'],run_info['seed'],step] = matrix
            print(f"Analyzed {run_info['method']} seed {run_info['seed']} step {step}",flush=True)
    for key,values in all_rows.items():
        common.write_csv(out/(key+'.csv'),values)
    deltas = {(r['method'],r['seed'],r['step'],r['class_id']):r for r in common.paired_deltas(tables,order,ranks)}
    base_geometry = {(r['seed'],r['step'],r['class_id'],r['feature']):r for r in all_rows['geometry'] if r['method']=='avcil'}
    comparisons = []
    for r in all_rows['geometry']:
        if r['method']=='avcil':
            continue
        baseline = base_geometry[r['seed'],r['step'],r['class_id'],r['feature']]
        row = dict(deltas[r['method'],r['seed'],r['step'],r['class_id']],feature=r['feature'])
        for field in ['mean_margin','p10_margin','query_dispersion','reference_dispersion','centroid_accuracy','knn_purity']:
            row.update({'avcil_'+field:baseline[field],'method_'+field:r[field],'delta_'+field:r[field]-baseline[field]})
        if r['feature'] in ['audio','visual']:
            fixed = input_geometry[r['feature'],r['class_id']]
            row.update({'fixed_input_'+k:fixed[k] for k in ['mean_margin','knn_purity','reference_dispersion']})
        comparisons.append(row)
    common.write_csv(out/'geometry_deltas.csv',comparisons)
    changes = []
    for (method,seed,step),matrix in matrices.items():
        if method=='avcil':
            continue
        baseline = matrices['avcil',seed,step]
        for c,d in zip(*np.nonzero(matrix-baseline)):
            changes.append(dict(method=method,seed=seed,step=step,true_class=int(c),predicted_class=int(d),
                true_category=order[c],predicted_category=order[d],true_is_old=bool(c<step*10),
                predicted_is_old=bool(d<step*10),avcil_count=int(baseline[c,d]),method_count=int(matrix[c,d]),
                count_change=int(matrix[c,d]-baseline[c,d])))
    if changes:
        common.write_csv(out/'confusion_changes.csv',changes)
    mismatches = [r for r in all_rows['performance'] if r['historical_counts_match'] is False]
    common.write_json(out/'audit.json',dict(status='complete',parameters_and_buffers_unchanged=True,
        historical_count_mismatch_rows=mismatches,confusion_changes=len(changes),
        optimizer_updates=0,backward_calls=0,query_split='test',reference_split='train',
        geometry_candidates='all seen classes; comparisons only within the same evaluation step',
        retention_candidates='old classes only, own previous-step teacher; fixed TRAIN references, not original memory',
        fixed_input_candidates=100,reference_ids_shared=True,query_ids_shared=True,
        cross_model_coordinate_comparison=False,uncertainty='Default geometry uses one seed and one reference draw; descriptive only'))
    content = '<!doctype html><meta charset="utf-8"><title>Checkpoint 几何分析</title><style>body{font:17px sans-serif;max-width:1000px;margin:40px auto;line-height:1.7}</style><h1>Checkpoint 几何与混淆分析</h1>'
    content += '<p>'+html.escape(common.ATTRIBUTION)+'</p><p>固定训练参考样本与独立测试 query；原型在各自模型空间内重算。retention 使用该运行前一阶段教师的冻结参考，不能解释为原 replay memory 的训练损失。</p>'
    content += '<p>输入几何固定使用全部 100 类；checkpoint 几何使用当时已见类；retention 仅使用旧类。不同候选类数下的 margin 不直接比较。默认单 seed、单参考抽样只用于机制定位。</p>'
    content += f'<p>与历史测试计数不一致的 class/step 行：{len(mismatches)}。完整测试时应查看 audit.json；若非零，先核对 CPU 数值差异、模型实现与输入版本，勿将推理差异当成方法收益。</p><ul>'
    for p in sorted(out.glob('*.csv')):
        content += f'<li><a href="{p.name}">{p.name}</a></li>'
    content += '</ul><p>geometry_deltas.csv 已关联历史逐类 F1 收益、难度、年龄和几何差值。query_geometry.csv / query_retention.csv 支持尾部样本检查。cache 中保存已抽取特征和 logits，可恢复计算，无需重新 forward。</p>'
    (out/'report.html').write_text(content,encoding='utf-8')
    common.write_json(out/'complete.json',dict(status='complete',historical_count_mismatches=len(mismatches)))
    print('Step 2 complete:',out,flush=True)
    if mismatches:
        print('ATTENTION: historical test counts differ; inspect audit.json before interpreting results.',flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,default=common.HERE)
    p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--meta-root',type=Path,default=common.HERE.parent/'data2/balance')
    p.add_argument('--difficulty',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--steps',type=int,nargs='+',default=[1,5,9])
    p.add_argument('--sample-seed',type=int,default=20261001)
    p.add_argument('--reference-per-class',type=int,default=20)
    p.add_argument('--query-per-class',type=int,default=0,help='0 = full test split; positive values subsample and disable historical-count validation')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--knn-k',type=int,default=5)
    p.add_argument('--margin-temperature',type=float,default=.1)
    p.add_argument('--retention-tolerance',type=float,default=.01)
    p.add_argument('--resume',action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if (not args.steps or any(s<1 or s>9 for s in args.steps) or len(set(args.steps))!=len(args.steps)
        or len(set(args.seeds))!=len(args.seeds) or min(args.reference_per_class,args.batch_size,args.threads,args.knn_k)<1
        or args.query_per_class<0 or args.margin_temperature<=0 or args.retention_tolerance<0):
        raise ValueError('Invalid steps, seeds, counts, temperature or tolerance')
    run(args)
