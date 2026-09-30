"""Offline loss/gradient measurements; never import or execute the trainer.

Uses trusted full-model checkpoints and fixed TRAIN-split probe batches.
No optimizer, backward(), checkpoint save, or historical-trajectory claim.
Run --help without PyTorch; actual probes use the original training environment.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def native(path):
    path = str(Path(path).resolve())
    return Path('\\\\?\\' + path) if os.name == 'nt' and not path.startswith('\\\\?\\') else Path(path)


def sha256(path):
    h = hashlib.sha256()
    with native(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lambda_geometry(base_sq, cmr_sq, base_dot_cmr, coefficient):
    """Counterfactual geometry at one fixed model, batch, teacher and bank."""
    b, c = math.sqrt(max(base_sq, 0)), math.sqrt(max(cmr_sq, 0))
    total_sq = base_sq + 2 * coefficient * base_dot_cmr + coefficient**2 * cmr_sq
    total = math.sqrt(max(total_sq, 0))
    cos = base_dot_cmr / (b*c) if b*c > 0 else None
    total_cos = (base_sq + coefficient*base_dot_cmr)/(b*total) if b*total > 0 else None
    return dict(lambda_cmr=coefficient, non_cmr_grad_norm=b, raw_cmr_grad_norm=c,
                weighted_cmr_grad_norm=abs(coefficient)*c,
                cmr_over_non_cmr=abs(coefficient)*c/b if b > 0 else None,
                cmr_cosine_with_non_cmr=max(-1., min(1., cos)) if cos is not None else None,
                cmr_projection_over_non_cmr=coefficient*base_dot_cmr/base_sq if base_sq > 0 else None,
                total_grad_norm=total,
                total_cosine_with_non_cmr=max(-1., min(1., total_cos)) if total_cos is not None else None,
                lambda_for_equal_grad_norm=b/c if c > 0 else None)


def load_runtime():
    global torch, F, exact, rd, exemplarLoader, default_collate
    import torch
    from torch.nn import functional as F
    from torch.utils.data import default_collate
    from dataloader_ours import exemplarLoader
    from rd_crosssdc import exact_losses as exact, rd_method as rd


def probe_components(model, teacher, current, replay, args, step, bank, weights):
    """Same loss definitions/reductions as the modular trainer, for steps > 0."""
    (visual, audio), labels = current
    (old_visual, old_audio), old_labels = replay
    n, m = labels.numel(), old_labels.numel()
    k = args.class_num_per_step
    old_classes = step*k
    visual, audio = torch.cat((visual, old_visual)), torch.cat((audio, old_audio))
    outputs = model(visual=visual, audio=audio, out_feature_before_fusion=True, out_attn_score=True)
    out, af, vf, spatial, temporal = outputs
    with torch.no_grad():
        old_out, taf, tvf, ts, tt = teacher(
            visual=visual, audio=audio, out_feature_before_fusion=True, out_attn_score=True)
    ce_new = exact.ce_loss(k, out[:n, old_classes:], labels % k)
    ce_old = exact.ce_loss(old_classes, out[n:, :old_classes], old_labels)
    ce = (n*ce_new + m*ce_old)/(n+m)
    if args.dataset == 'AVE' and k == 4 and step == 1:
        ce = exact.ce_loss(old_classes+k, out, torch.cat((labels, old_labels)))
    kd = out.new_zeros(())
    for task in range(step):
        sl = slice(task*k, (task+1)*k)
        kd = kd + F.kl_div(F.log_softmax(out[:, sl]/2, dim=1),
                          F.softmax(old_out[:, sl]/2, dim=1), reduction='batchmean')*4
    both_labels = torch.cat((labels, old_labels))
    inst = exact.cal_contrastive_loss(af, vf, args.instance_contrastive_temperature) if args.instance_contrastive else None
    cls = exact.class_contrastive_loss(af, vf, both_labels, args.class_contrastive_temperature) if args.class_contrastive else None
    cross_args = dict(cur_audio=af[n:], cur_visual=vf[n:], old_audio=taf[n:], old_visual=tvf[n:],
                      temperature=args.cross_sdc_temperature)
    if args.rd_mode == 'adaptive_crosssdc_cmr':
        cross_i = exact.cross_sdc_instance_loss(**cross_args)
        cross_c, _ = exact.weighted_cross_sdc_class_loss(
            **cross_args, labels=old_labels, class_weight_a_from_v=weights[0], class_weight_v_from_a=weights[1])
    else:
        cross_i, cross_c = exact.cross_sdc_z1_loss(**cross_args, labels=old_labels)
    # Measure CMR even if the saved lambda is zero, for the local lambda sweep.
    terms = rd.compute_margin_terms(af[n:], vf[n:], taf[n:], tvf[n:], old_labels, bank,
                                    args.rd_margin_temperature, args.rd_margin_tolerance)
    cmr, stats = rd.cmr_loss(terms, old_labels, weights[0], weights[1],
                            penalty=args.rd_cmr_penalty, penalty_scale=args.rd_cmr_scale,
                            tolerance=args.rd_margin_tolerance)
    sa = ta = None
    if args.attn_score_distil:
        student_s, target_s = spatial[n:].transpose(2, 3), ts[n:].transpose(2, 3)
        student_t, target_t = temporal[n:].transpose(1, 2), tt[n:].transpose(1, 2)
        sa = F.kl_div(student_s.reshape(-1, student_s.shape[-1]).log(),
                      target_s.reshape(-1, target_s.shape[-1]), reduction='sum')/m
        ta = F.kl_div(student_t.reshape(-1, student_t.shape[-1]).log(),
                      target_t.reshape(-1, target_t.shape[-1]), reduction='sum')/m
    components = dict(ce=(ce, 1.), kd=(kd, 1.), instance_contrastive=(inst, args.lam_I),
                      class_contrastive=(cls, args.lam_C), cross_sdc_i=(cross_i, args.lam_cross_sdc_i),
                      cross_sdc_c=(cross_c, args.lam_cross_sdc_c), cmr=(cmr, args.lam_cmr),
                      attn_spatial=(sa, args.lam), attn_temporal=(ta, 1-args.lam))
    total = sum(value*coefficient for value, coefficient in components.values() if value is not None)
    return components, total, stats


def measure_gradients(model, components, total, lambdas):
    """Per-batch norms and Gram matrices; zero/unused parameters stay in scope."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = tuple(p for _, p in named)
    items = list(components.items())
    names = [n for n, _ in items]
    coefficients = torch.tensor([v[1] for _, v in items], dtype=torch.float64)
    raw_vectors = []
    for name, (value, coefficient) in items:
        if value is None or not value.requires_grad:
            grads = (None,)*len(params)
        else:
            grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        raw_vectors.append(torch.cat([g.detach().reshape(-1).float().cpu() if g is not None
                                      else torch.zeros(p.numel()) for g, p in zip(grads, params)]))
    raw = torch.stack(raw_vectors)
    actual = torch.autograd.grad(total, params, retain_graph=False, allow_unused=True)
    actual = torch.cat([g.detach().reshape(-1).float().cpu() if g is not None
                        else torch.zeros(p.numel()) for g, p in zip(actual, params)])
    reconstructed = (raw*coefficients.float()[:, None]).sum(dim=0)
    err = float((actual-reconstructed).norm())
    norm = float(actual.norm())
    if err > 2e-5 + 2e-4*norm:
        raise RuntimeError(f'Gradient reconstruction mismatch: residual={err}, total_norm={norm}')
    # No parameter .grad is populated by autograd.grad.
    assert all(p.grad is None for p in params)
    classifier = torch.cat([torch.full((p.numel(),), n.startswith('classifier.'), dtype=torch.bool) for n, p in named])
    scopes = {'all':torch.ones_like(classifier), 'feature_and_attention':~classifier, 'classifier':classifier}
    cmr_index = names.index('cmr')
    base_weights = coefficients.clone()
    base_weights[cmr_index] = 0
    rows, pairs, sweep = [], [], []
    for scope, mask in scopes.items():
        if not bool(mask.any()):
            continue
        vectors = raw[:, mask].double()
        gram = vectors @ vectors.T
        base_sq = float(base_weights @ gram @ base_weights)
        base_norm = math.sqrt(max(base_sq, 0))
        cmr_sq = float(gram[cmr_index, cmr_index])
        dot_bc = float(base_weights @ gram[:, cmr_index])
        for i, (name, (value, coefficient)) in enumerate(items):
            raw_norm = math.sqrt(max(float(gram[i, i]), 0))
            weighted_norm = abs(coefficient)*raw_norm
            dot_with_base = coefficient*float(gram[i] @ base_weights)
            cosine = dot_with_base/(weighted_norm*base_norm) if weighted_norm*base_norm > 0 else None
            rows.append(dict(parameter_scope=scope, component=name, enabled=int(value is not None), coefficient=coefficient,
                raw_loss=float(value.detach()) if value is not None else None,
                weighted_loss=coefficient*float(value.detach()) if value is not None else 0.,
                total_loss=float(total.detach()), raw_grad_norm=raw_norm, weighted_grad_norm=weighted_norm,
                non_cmr_grad_norm=base_norm, weighted_grad_over_non_cmr=weighted_norm/base_norm if base_norm > 0 else None,
                cosine_with_non_cmr=max(-1., min(1., cosine)) if cosine is not None else None,
                gradient_reconstruction_residual=err))
            for j in range(i+1, len(items)):
                den = math.sqrt(max(float(gram[i, i]*gram[j, j]), 0))
                cos = float(gram[i, j])/den if den > 0 else None
                weighted_cos = cos if coefficient*float(coefficients[j]) > 0 else (-cos if coefficient*float(coefficients[j]) < 0 and cos is not None else None)
                pairs.append(dict(parameter_scope=scope, component_a=name, component_b=names[j],
                                  raw_cosine=cos, weighted_cosine=weighted_cos))
        for lam in sorted(set(lambdas + [float(components['cmr'][1])])):
            sweep.append(dict(parameter_scope=scope, **lambda_geometry(base_sq, cmr_sq, dot_bc, lam)))
    return rows, pairs, sweep


def make_probe_plan(dataset, args, step, seed, batches, current_batch_size, replay_batch_size, replay_ids=None):
    rng = random.Random(seed + step*100003)
    old_classes = step*args.class_num_per_step
    if replay_ids is None:
        count = args.memory_size//old_classes
        if count < 1:
            raise ValueError('Memory size too small to cover all old classes')
        memory = []
        for c in range(old_classes):
            valid = sorted(v for v in dataset._get_class_vids(c) if dataset._has_feature(v))
            if not valid:
                raise ValueError(f'No available train features for old class {c}')
            memory.extend(rng.sample(valid, min(count, len(valid))))
    else:
        memory = list(replay_ids)
    current_pool = sorted(v for c in range(old_classes, old_classes+args.class_num_per_step)
                          for v in dataset._get_class_vids(c) if dataset._has_feature(v))
    n, m = min(current_batch_size, len(current_pool)), min(replay_batch_size, len(memory))
    if min(n, m) < 1:
        raise ValueError('Empty current/replay probe pool')
    return dict(memory_source='supplied_replay_ids' if replay_ids is not None else 'new_fixed_class_balanced_probe',
                memory_ids=memory, batches=[dict(current_ids=rng.sample(current_pool, n),
                                                replay_ids=rng.sample(memory, m)) for _ in range(batches)])


def validate_plan(dataset, args, step, plan):
    old_classes = step*args.class_num_per_step
    def label(vid):
        if not dataset._has_feature(vid):
            raise ValueError(f'Missing probe feature: {vid}')
        return int(dataset.category_encode_dict[dataset.all_id_category_dict[vid]])
    memory = plan['memory_ids']
    if len(set(memory)) != len(memory):
        raise ValueError('Memory IDs must be unique for leave-one-out CMR')
    if {label(v) for v in memory} != set(range(old_classes)):
        raise ValueError('Memory must cover exactly the old classes')
    if not plan['batches']:
        raise ValueError('No probe batches')
    for batch in plan['batches']:
        if not batch['current_ids'] or not batch['replay_ids']:
            raise ValueError('Empty batch')
        if not set(batch['replay_ids']).issubset(memory):
            raise ValueError('Replay batch must be part of prototype memory for correct leave-one-out')
        if any(not old_classes <= label(v) < old_classes+args.class_num_per_step for v in batch['current_ids']):
            raise ValueError('Current probe contains a sample outside the current task')


def load_batch(dataset, ids, device):
    dataset.exemplar_vids_set = ids
    (visual, audio), labels = default_collate([dataset[i] for i in range(len(ids))])
    return (visual.to(device), audio.to(device)), labels.to(device).long()


def state_fingerprint(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def summarize(rows):
    result = []
    for step, scope, component in sorted({(r['step'], r['parameter_scope'], r['component']) for r in rows}):
        rs = [r for r in rows if (r['step'], r['parameter_scope'], r['component']) == (step, scope, component)]
        item = dict(step=step, parameter_scope=scope, component=component, batches=len(rs))
        for field in ['raw_loss', 'weighted_loss', 'raw_grad_norm', 'weighted_grad_norm',
                      'weighted_grad_over_non_cmr', 'cosine_with_non_cmr']:
            vals = [r[field] for r in rs if r[field] is not None]
            item[field+'_mean'] = statistics.mean(vals) if vals else None
            item[field+'_batch_sd'] = statistics.stdev(vals) if len(vals) > 1 else None
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Directory containing step_N_best_model.pkl')
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--meta-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New or empty analysis directory')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 5, 9])
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--probe-seed', type=int, default=20260930)
    parser.add_argument('--current-batch-size', type=int, help='Default: original training batch size')
    parser.add_argument('--replay-batch-size', type=int, help='Default: original exemplar batch size')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0, .01, .03, .1, .3, 1])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--probe-manifest', type=Path, help='Reuse probe_manifest.json from an earlier run')
    parser.add_argument('--replay-ids', type=Path, help='Optional JSON: step string -> original memory ID list')
    cli = parser.parse_args()
    if cli.batches < 1 or cli.threads < 1 or any(s < 1 for s in cli.steps) or len(set(cli.steps)) != len(cli.steps):
        parser.error('Positive batches/threads and unique incremental steps >=1 required')
    if any(not math.isfinite(v) or v < 0 for v in cli.lambdas):
        parser.error('Lambda values must be finite and non-negative')
    if any(v is not None and v < 1 for v in [cli.current_batch_size, cli.replay_batch_size]):
        parser.error('Batch sizes must be positive')
    if cli.probe_manifest and cli.replay_ids:
        parser.error('Use either a full probe manifest or replay IDs')
    out = cli.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be new or empty; existing results are never overwritten')
    try:
        out.relative_to(cli.run_dir.resolve())
    except ValueError:
        pass
    else:
        parser.error('Write analysis outside the checkpoint directory')
    try:
        load_runtime()
    except ImportError as exc:
        raise SystemExit(f'Use the original training environment (PyTorch/h5py/numpy/tqdm required): {exc}')
    if cli.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; explicitly use --device cpu for CPU probes')
    torch.set_num_threads(cli.threads)
    torch.manual_seed(cli.probe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.probe_seed)
    device = torch.device(cli.device)
    out.mkdir(parents=True, exist_ok=True)
    supplied = json.loads(cli.probe_manifest.read_text(encoding='utf-8-sig')) if cli.probe_manifest else None
    replay_ids = json.loads(cli.replay_ids.read_text(encoding='utf-8-sig')) if cli.replay_ids else {}
    metadata_hashes = {name:sha256(cli.meta_root/name) for name in
                       ['all_id_category_dict.npy', 'category_encode_dict.npy', 'all_classId_vid_dict.npy']}
    if supplied and supplied['metadata_sha256'] != metadata_hashes:
        raise ValueError('Probe manifest metadata/class mapping differs from supplied metadata')
    manifest = dict(kind='offline_fixed_train_probe_not_historical_batches', probe_seed=cli.probe_seed,
                    metadata_sha256=metadata_hashes, steps={})
    all_rows, all_pairs, all_sweeps, states = [], [], [], []
    source_hashes = {str(p):sha256(p) for p in [Path(__file__), HERE/'rd_crosssdc'/'exact_losses.py',
                       HERE/'rd_crosssdc'/'rd_method.py', HERE/'rd_crosssdc'/'cmr_penalties.py',
                       HERE/'dataloader_ours.py', HERE.parent/'model'/'audio_visual_model_incremental.py']}
    for step in cli.steps:
        student_path = native(cli.run_dir/f'step_{step}_best_model.pkl')
        teacher_path = native(cli.run_dir/f'step_{step-1}_best_model.pkl')
        hashes = {'student':sha256(student_path), 'teacher':sha256(teacher_path)}
        model = torch.load(student_path, map_location='cpu', weights_only=False)
        teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(teacher, 'module'):
            teacher = teacher.module
        args = copy.deepcopy(model.args)
        if args.rd_mode not in ['crosssdc_cmr', 'adaptive_crosssdc_cmr']:
            raise ValueError('This probe expects CMR checkpoints; baseline support must specify matching CMR parameters explicitly')
        if args.modality != 'audio-visual' or model.num_classes != (step+1)*args.class_num_per_step or teacher.num_classes != step*args.class_num_per_step:
            raise ValueError('Checkpoint architecture/class count does not match step')
        args.feature_root, args.meta_root = str(cli.feature_root.resolve()), str(cli.meta_root.resolve())
        args.num_workers = 0
        model.to(device).eval().requires_grad_(True)
        teacher.to(device).eval().requires_grad_(False)
        model.zero_grad(set_to_none=True)
        before = (state_fingerprint(model), state_fingerprint(teacher))
        dataset = exemplarLoader(args, modality='audio-visual', incremental_step=step)
        try:
            plan = supplied['steps'][str(step)] if supplied else make_probe_plan(
                dataset, args, step, cli.probe_seed, cli.batches,
                cli.current_batch_size or args.train_batch_size,
                cli.replay_batch_size or args.exemplar_batch_size,
                replay_ids.get(str(step)))
            validate_plan(dataset, args, step, plan)
            manifest['steps'][str(step)] = plan
            (out/'probe_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
            dataset.exemplar_vids_set = plan['memory_ids']
            adaptive = args.rd_mode == 'adaptive_crosssdc_cmr'
            bank = rd.build_old_teacher_prototype_bank(teacher, dataset, step*args.class_num_per_step,
                min(32, args.exemplar_batch_size), 0, device, args.rd_margin_temperature,
                adaptive, args.rd_trust_shrinkage_beta)
            if adaptive:
                controller = rd.AdaptiveWeightController(bank.trust_a_from_v, bank.trust_v_from_a,
                    args.rd_class_weight_alpha, args.rd_trust_offset, args.rd_trust_gamma,
                    args.rd_need_delta, args.rd_need_eta, args.rd_need_ema_momentum,
                    args.rd_weight_min, args.rd_weight_max,
                    clip_weights=not getattr(args, 'rd_disable_weight_clipping', False))
                weights = (controller.cmr_weight_a, controller.cmr_weight_v)
            else:
                weights = (torch.ones(step*args.class_num_per_step, device=device),)*2
            for batch_index, batch in enumerate(plan['batches']):
                print(f'Probe step={step}, batch={batch_index+1}/{len(plan["batches"])} (no parameter update)', flush=True)
                current = load_batch(dataset, batch['current_ids'], device)
                replay = load_batch(dataset, batch['replay_ids'], device)
                components, total, cmr_stats = probe_components(model, teacher, current, replay, args, step, bank, weights)
                if not torch.isfinite(total):
                    raise FloatingPointError('Nonfinite probe loss')
                rows, pairs, sweep = measure_gradients(model, components, total, cli.lambdas)
                context = dict(step=step, batch=batch_index, n_current=len(batch['current_ids']), n_replay=len(batch['replay_ids']))
                all_rows += [dict(**context, **r) for r in rows]
                all_pairs += [dict(**context, **r) for r in pairs]
                all_sweeps += [dict(**context, **r) for r in sweep]
                del components, total, current, replay, cmr_stats
            if before != (state_fingerprint(model), state_fingerprint(teacher)):
                raise RuntimeError('Probe mutated model parameters or buffers')
            if hashes != {'student':sha256(student_path), 'teacher':sha256(teacher_path)}:
                raise RuntimeError('Checkpoint file changed during analysis')
            states.append(dict(step=step, checkpoint_sha256=hashes, memory_source=plan['memory_source'],
                               parameters_and_buffers_unchanged=True, args=vars(args)))
        finally:
            dataset.close_visual_features_h5()
        write_csv(out/'components.csv', all_rows)
        write_csv(out/'gradient_pairs.csv', all_pairs)
        write_csv(out/'lambda_sweep.csv', all_sweeps)
        write_csv(out/'component_summary.csv', summarize(all_rows))
        del dataset, model, teacher, bank, weights
        if adaptive:
            del controller
    metadata = dict(status='complete', measurement='offline_checkpoint_fixed_train_probe',
        model_mode='eval', teacher_reference='own_previous_step_best_checkpoint',
        optimizer_updates=0, historical_gradient_reconstruction=False,
        gradient_statistics='per_batch_norms_not_norm_of_dataset_mean_gradient',
        batch_standard_deviation='probe_batch_variability_not_seed_uncertainty',
        source_sha256=source_hashes, cli=vars(cli), torch_version=torch.__version__, states=states)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    print('Saved offline measurements to', out)


if __name__ == '__main__':
    main()
