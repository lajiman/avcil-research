"""Three controlled offline loss/gradient experiments, without training.

Original memory / held-out queries with original references / new memory.
Uses trusted project checkpoints; defaults to CPU and writes under results/.
"""
import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import statistics

import offline_memory_protocols as protocol
import probe_checkpoint_losses as probe

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    probe.native(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def find_metrics_dir(run_dir, explicit=None):
    base = explicit if explicit is not None else run_dir.parent/'metrics'/run_dir.name
    directory = base if probe.native(base/'epoch_summary.csv').is_file() else base/'rd_crosssdc'
    if not probe.native(directory/'epoch_summary.csv').is_file():
        raise ValueError(f'Historical epoch_summary.csv missing in {directory}; use --metrics-dir')
    return directory


def load_historical_reference(directory, args, step, dataset):
    paths = {'epochs': directory/'epoch_summary.csv', 'static': directory/f'step_{step}_static_trust.csv'}
    best = protocol.best_epoch_row(protocol.read_csv(probe.native(paths['epochs'])), step)
    epoch = int(best['epoch'])
    for column, expected in [('cmr_tolerance', args.rd_margin_tolerance), ('cmr_scale', args.rd_cmr_scale)]:
        if not math.isclose(float(best[column]), expected, abs_tol=1e-10, rel_tol=1e-8):
            raise ValueError(f'Historical {column} differs from checkpoint')
    if best['cmr_penalty'] != args.rd_cmr_penalty or best['rd_mode'] != args.rd_mode:
        raise ValueError('Historical CMR mode/penalty differs from checkpoint')
    if not math.isclose(float(best['weighted_cmr']), args.lam_cmr*float(best['cmr']), abs_tol=1e-9, rel_tol=1e-5):
        raise ValueError('Historical CMR coefficient differs from checkpoint')
    paths['weights'] = directory/f'step_{step}_epoch_{epoch}_weights.csv'
    n = step*args.class_num_per_step
    static = protocol.indexed_class_rows(protocol.read_csv(probe.native(paths['static'])), n, step)
    weights = protocol.indexed_class_rows(protocol.read_csv(probe.native(paths['weights'])), n, step, epoch)
    categories = {int(v): str(k) for k, v in dataset.category_encode_dict.items()}
    combined = []
    for c in range(n):
        if static[c]['category_name'] != categories[c] or weights[c]['category_name'] != categories[c]:
            raise ValueError(f'Historical category mapping differs for class {c}')
        row = dict(static[c])
        for direction in ['a_from_v', 'v_from_a']:
            trust = 'trust_'+direction
            if not math.isclose(float(weights[c][trust]), float(static[c][trust]), abs_tol=1e-8, rel_tol=1e-6):
                raise ValueError('Static/epoch Trust records disagree')
            key = 'cmr_weight_'+direction
            value = float(weights[c][key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Historical CMR weights must be finite and positive')
            if not math.isclose(value, float(weights[c]['class_weight_'+direction]), abs_tol=1e-8, rel_tol=1e-6):
                raise ValueError('Different CMR/CrossSDC class weights are unsupported')
            row[key] = value
        combined.append(row)
    return best, combined, {key: dict(path=str(path), sha256=probe.sha256(path)) for key, path in paths.items()}


def build_reference(teacher, dataset, args, step, ids, device):
    probe.require_memory_prototype_policy(args)
    dataset.exemplar_vids_set = list(ids)
    bank = probe.rd.build_old_teacher_prototype_bank(teacher, dataset, step*args.class_num_per_step,
        min(32, args.exemplar_batch_size), 0, device, args.rd_margin_temperature, True, args.rd_trust_shrinkage_beta)
    controller = probe.rd.AdaptiveWeightController(bank.trust_a_from_v, bank.trust_v_from_a,
        args.rd_class_weight_alpha, args.rd_trust_offset, args.rd_trust_gamma, args.rd_need_delta,
        args.rd_need_eta, args.rd_need_ema_momentum, args.rd_weight_min, args.rd_weight_max,
        clip_weights=not getattr(args, 'rd_disable_weight_clipping', False))
    return bank, (controller.cmr_weight_a, controller.cmr_weight_v)


def reference_rows(bank, weights, dataset, step):
    categories = {int(v): str(k) for k, v in dataset.category_encode_dict.items()}
    rows = []
    fields = ['reliability_a_from_v', 'reliability_v_from_a', 'trust_a_from_v', 'trust_v_from_a']
    for c in range(len(bank.counts)):
        rows.append(dict(step=step, class_id=c, category_name=categories[c], prototype_count=float(bank.counts[c]),
            **{key: float(getattr(bank, key)[c]) for key in fields},
            cmr_weight_a_from_v=float(weights[0][c]), cmr_weight_v_from_a=float(weights[1][c])))
    return rows


def cmr_sample_rows(terms, labels, ids, membership, weights, args, context):
    """Detached per-query margins and exact contributions to the batch CMR."""
    if args.rd_cmr_penalty == 'hinge':
        penalties = (terms.violation_a_from_v, terms.violation_v_from_a)
    else:
        penalties = tuple(probe.rd.cmr_penalty_per_sample(
            current_margin=getattr(terms, 'cur_'+d), reference_margin=getattr(terms, 'ref_'+d),
            penalty=args.rd_cmr_penalty, tolerance=args.rd_margin_tolerance, scale=args.rd_cmr_scale)
            for d in ['a_from_v', 'v_from_a'])
    sample_weights = [w.index_select(0, labels).detach() for w in weights]
    contributions = .5 * sum(p.detach()*w/w.sum().clamp_min(1e-12) for p, w in zip(penalties, sample_weights))
    vectors = {'class_id': labels.tolist(), 'in_reference_memory': membership.tolist(),
               'raw_cmr_contribution': contributions.tolist()}
    for j, d in enumerate(['a_from_v', 'v_from_a']):
        for prefix in ['ref', 'cur', 'deficit', 'violation']:
            vectors[prefix+'_'+d] = getattr(terms, prefix+'_'+d).detach().tolist()
        vectors['margin_drop_'+d] = (getattr(terms, 'ref_'+d)-getattr(terms, 'cur_'+d)).detach().tolist()
        vectors['penalty_'+d] = penalties[j].detach().tolist()
        vectors['weight_'+d] = sample_weights[j].tolist()
    return [dict(**context, sample_index=i, sample_id=vid, **{k:v[i] for k,v in vectors.items()},
                 weighted_cmr_contribution=args.lam_cmr*vectors['raw_cmr_contribution'][i])
            for i, vid in enumerate(ids)]


def scalar_rows(components, total):
    # An explicit loss-only row never contains fabricated gradient zeros.
    return [dict(parameter_scope='not_measured', component=name, enabled=int(value is not None), coefficient=coefficient,
        raw_loss=float(value) if value is not None else None,
        weighted_loss=coefficient*float(value) if value is not None else 0., total_loss=float(total),
        raw_grad_norm=None, weighted_grad_norm=None, non_cmr_grad_norm=None,
        weighted_grad_over_non_cmr=None, cosine_with_non_cmr=None, gradient_reconstruction_residual=None)
        for name, (value, coefficient) in components.items()]


def flush_outputs(out, tables):
    for name, rows in tables.items():
        probe.write_csv(probe.native(out/(name+'.csv')), rows)
    summary = []
    for group in protocol.GROUPS:
        rows = [r for r in tables['components'] if r['experiment_group'] == group]
        summary.extend(dict(experiment_group=group, **r) for r in probe.summarize(rows))
    probe.write_csv(probe.native(out/'component_summary.csv'), summary)


def run(cli):
    out = cli.output.resolve()
    probe.load_runtime()
    torch = probe.torch
    if cli.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(cli.threads)
    torch.manual_seed(cli.probe_seed)
    device = torch.device(cli.device)
    metrics_dir = find_metrics_dir(cli.run_dir, cli.metrics_dir)
    supplied_ids = json.loads(probe.native(cli.replay_ids).read_text(encoding='utf-8-sig')) if cli.replay_ids else None
    supplied_plan = json.loads(probe.native(cli.probe_manifest).read_text(encoding='utf-8-sig')) if cli.probe_manifest else None
    metadata_hashes = {name: probe.sha256(cli.meta_root/name) for name in
        ['all_id_category_dict.npy', 'category_encode_dict.npy', 'all_classId_vid_dict.npy']}
    if supplied_plan and (supplied_plan.get('schema_version') != 2 or supplied_plan['metadata_sha256'] != metadata_hashes):
        raise ValueError('Use a three-way v2 manifest with the same metadata; legacy manifests are incompatible')
    manifest = dict(schema_version=2, kind='three_way_disjoint_train_probe_not_historical_batches',
                    probe_seed=supplied_plan['probe_seed'] if supplied_plan else cli.probe_seed,
                    metadata_sha256=metadata_hashes, steps={})
    candidate_ids = {}
    tables = {name: [] for name in ['components', 'gradient_pairs', 'lambda_sweep', 'cmr_batches', 'cmr_samples',
                                  'class_reference', 'reference_checks', 'historical_comparison', 'coverage']}
    source_paths = [Path(__file__), Path(probe.__file__), Path(protocol.__file__),
        HERE/'rd_crosssdc'/'rd_method.py', HERE/'rd_crosssdc'/'exact_losses.py', HERE/'rd_crosssdc'/'cmr_penalties.py',
        HERE/'dataloader_ours.py', HERE.parent/'model'/'audio_visual_model_incremental.py']
    metadata = dict(status='running', measurement='offline_checkpoint_three_way', model_mode='eval',
        optimizer_updates=0, historical_gradient_reconstruction=False,
        original_memory_identity='candidate_IDs_checked_against_saved_counts_reliability_trust_and_weights',
        gradient_statistics='per_batch_norms_not_norm_of_dataset_mean_gradient',
        batch_standard_deviation='conditional_on_one_bank_not_bank_or_seed_uncertainty',
        teacher_reference='own_previous_step_best_checkpoint', cli=vars(cli),
        python_version=platform.python_version(), torch_version=torch.__version__,
        source_sha256={str(p):probe.sha256(p) for p in source_paths}, states=[])
    write_json(out/'metadata.json', metadata)
    try:
        for step in cli.steps:
            print(f'Step {step}: loading checkpoints and checking original reference', flush=True)
            student_path, teacher_path = (cli.run_dir/f'step_{s}_best_model.pkl' for s in [step, step-1])
            hashes = {'student': probe.sha256(student_path), 'teacher': probe.sha256(teacher_path)}
            model = torch.load(probe.native(student_path), map_location='cpu', weights_only=False)
            teacher = torch.load(probe.native(teacher_path), map_location='cpu', weights_only=False)
            model, teacher = getattr(model, 'module', model), getattr(teacher, 'module', teacher)
            args = copy.deepcopy(model.args)
            if args.rd_mode != 'adaptive_crosssdc_cmr' or getattr(args, 'rd_need_eta', 0) != 0:
                raise ValueError('Three-way historical validation currently requires adaptive_crosssdc_cmr with Trust-only weights')
            if (args.modality != 'audio-visual' or model.num_classes != (step+1)*args.class_num_per_step
                    or teacher.num_classes != step*args.class_num_per_step):
                raise ValueError('Checkpoint architecture/class count does not match requested step')
            args.feature_root, args.meta_root = str(cli.feature_root.resolve()), str(cli.meta_root.resolve())
            args.num_workers = 0
            model.to(device).eval().requires_grad_(not cli.loss_only)
            teacher.to(device).eval().requires_grad_(False)
            model.zero_grad(set_to_none=True)
            before = (probe.state_fingerprint(model), probe.state_fingerprint(teacher))
            dataset = probe.exemplarLoader(args, modality='audio-visual', incremental_step=step)
            state = dict(step=step, status='checking_reference', checkpoint_sha256=hashes,
                         original_reference_validated=False, parameters_and_buffers_unchanged=None, args=vars(args))
            metadata['states'].append(state)
            try:
                original = (supplied_ids[str(step)] if supplied_ids is not None else
                            protocol.reconstruct_replay_ids(dataset, args, [step])[str(step)])
                source = 'supplied_replay_ids' if supplied_ids is not None else 'reconstructed_python_sample_prefix_shrink'
                state['original_memory_source'] = source
                protocol.validate_memory(dataset, args, step, original)
                candidate_ids[str(step)] = list(original)
                write_json(out/'original_replay_ids.json', candidate_ids)
                best, saved_rows, historical_sources = load_historical_reference(metrics_dir, args, step, dataset)
                state['historical_sources'] = historical_sources
                write_json(out/'metadata.json', metadata)
                original_bank, computed_weights = build_reference(teacher, dataset, args, step, original, device)
                actual_rows = reference_rows(original_bank, computed_weights, dataset, step)
                checks = protocol.compare_reference_rows(actual_rows, saved_rows, step)
                tables['reference_checks'].extend(checks)
                probe.write_csv(probe.native(out/'reference_checks.csv'), tables['reference_checks'])
                failed = [r for r in checks if not r['passed']]
                if failed:
                    raise ValueError(f'Step {step}: original reference validation failed ({len(failed)} fields). '
                        'See reference_checks.csv. Check original metadata order, feature files, sampling code/seed '
                        'and teacher; supply verified --replay-ids if available. No fallback to a new original memory.')
                state['original_reference_validated'] = True
                state['status'] = 'measuring'
                # Freeze the actual saved training weights for groups 1 and 2.
                original_weights = tuple(torch.tensor([float(r['cmr_weight_'+d]) for r in saved_rows],
                    device=device, dtype=torch.float32) for d in ['a_from_v', 'v_from_a'])
                n, m = cli.current_batch_size or args.train_batch_size, cli.replay_batch_size or args.exemplar_batch_size
                plan = (supplied_plan['steps'][str(step)] if supplied_plan else protocol.make_three_way_plan(
                    dataset, args, step, original, cli.probe_seed, cli.batches, n, m))
                protocol.validate_three_way_plan(dataset, args, step, plan, original, cli.batches, n, m)
                manifest['steps'][str(step)] = plan
                write_json(out/'probe_manifest.json', manifest)
                fresh_bank, fresh_weights = build_reference(teacher, dataset, args, step, plan['fresh_memory_ids'], device)
                references = {'original': (original_bank, original_weights, set(original)),
                              'fresh': (fresh_bank, fresh_weights, set(plan['fresh_memory_ids']))}
                for reference, (bank, weights, _) in references.items():
                    tables['class_reference'].extend(dict(reference=reference, **r)
                        for r in reference_rows(bank, weights, dataset, step))
                for group, (reference, query_key) in protocol.GROUPS.items():
                    bank, weights, memory_set = references[reference]
                    print(f'Step {step}: {group}; original reference validation passed', flush=True)
                    for batch_index, batch in enumerate(plan['batches']):
                        print(f'  batch {batch_index+1}/{cli.batches}; no parameter update', flush=True)
                        ids = batch[query_key]
                        membership = torch.tensor([v in memory_set for v in ids], dtype=torch.bool, device=device)
                        current, replay = probe.load_batch(dataset, batch['current_ids'], device), probe.load_batch(dataset, ids, device)
                        with torch.set_grad_enabled(not cli.loss_only):
                            components, total, stats, terms = probe.probe_components(model, teacher, current, replay,
                                args, step, bank, weights, replay_membership=membership, return_terms=True)
                            if not torch.isfinite(total):
                                raise FloatingPointError('Nonfinite probe loss')
                            context = dict(experiment_group=group, step=step, batch=batch_index, n_current=n, n_replay=m)
                            samples = cmr_sample_rows(terms, replay[1], ids, membership, weights, args, context)
                            batch_stats = {key: float(value) for key, value in vars(stats).items()}
                            cmr_value = float(components['cmr'][0].detach())
                            batch_stats.update(raw_cmr=cmr_value, weighted_cmr=cmr_value*args.lam_cmr)
                            for d in ['a_from_v', 'v_from_a']:
                                violations = getattr(terms, 'violation_'+d).detach()
                                active = violations[violations > 0]
                                batch_stats['mean_violation_'+d] = float(violations.mean())
                                batch_stats['active_mean_violation_'+d] = float(active.mean()) if active.numel() else 0.
                            if cli.loss_only:
                                rows, pairs, sweeps = scalar_rows(components, total), [], []
                            else:
                                rows, pairs, sweeps = probe.measure_gradients(model, components, total,
                                    cli.lambdas if cli.lambdas is not None else [args.lam_cmr])
                        for name, values in [('components', rows), ('gradient_pairs', pairs), ('lambda_sweep', sweeps)]:
                            tables[name].extend(dict(**context, **r) for r in values)
                        tables['cmr_samples'].extend(samples)
                        tables['cmr_batches'].append(dict(**context, **batch_stats))
                        flush_outputs(out, tables)
                        del components, total, stats, terms, current, replay
                    group_batches = [r for r in tables['cmr_batches'] if r['step'] == step and r['experiment_group'] == group]
                    mean_cmr = statistics.mean(r['weighted_cmr'] for r in group_batches)
                    epoch_cmr = float(best['weighted_cmr'])
                    tables['historical_comparison'].append(dict(experiment_group=group, step=step,
                        best_epoch=int(best['epoch']), lambda_cmr=args.lam_cmr, alpha=args.rd_class_weight_alpha,
                        train_epoch_mean_raw_cmr=float(best['cmr']), train_epoch_mean_weighted_cmr=epoch_cmr,
                        checkpoint_probe_mean_weighted_cmr=mean_cmr,
                        probe_over_train_epoch_cmr=mean_cmr/epoch_cmr if epoch_cmr != 0 else None,
                        train_epoch_active_a_from_v=float(best['cmr_active_a_from_v']),
                        train_epoch_active_v_from_a=float(best['cmr_active_v_from_a']),
                        probe_active_a_from_v=statistics.mean(r['active_a_from_v'] for r in group_batches),
                        probe_active_v_from_a=statistics.mean(r['active_v_from_a'] for r in group_batches)))
                    tables['coverage'].append(dict(experiment_group=group, step=step, reference_memory_size=len(memory_set),
                        query_slots=m*cli.batches, unique_queries=len({v for b in plan['batches'] for v in b[query_key]}),
                        queries_in_original_memory=len({v for b in plan['batches'] for v in b[query_key]} & set(original)),
                        unique_current=len({v for b in plan['batches'] for v in b['current_ids']})))
                if before != (probe.state_fingerprint(model), probe.state_fingerprint(teacher)):
                    raise RuntimeError('Probe mutated model parameters or buffers')
                if hashes != {'student':probe.sha256(student_path), 'teacher':probe.sha256(teacher_path)}:
                    raise RuntimeError('Checkpoint file changed during analysis')
                state['parameters_and_buffers_unchanged'] = True
                state['status'] = 'complete'
                flush_outputs(out, tables)
                write_json(out/'metadata.json', metadata)
            finally:
                dataset.close_visual_features_h5()
            del model, teacher, dataset, original_bank, fresh_bank, references, bank, weights
        metadata['status'] = 'complete'
    except Exception as exc:
        metadata['status'], metadata['error'] = 'failed', str(exc)
        raise
    finally:
        write_json(out/'metadata.json', metadata)
    print('Saved all three offline experiments to', out, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--meta-root', type=Path, required=True)
    parser.add_argument('--metrics-dir', type=Path, help='Run-specific historical metrics dir (or its rd_crosssdc subdir)')
    parser.add_argument('--output', type=Path, help='New/empty dir; default: results/three_way_TIMESTAMP_RUN')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 5, 9])
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--probe-seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--current-batch-size', type=int)
    parser.add_argument('--replay-batch-size', type=int)
    parser.add_argument('--lambdas', type=float, nargs='+', help='Optional local sweep; default uses saved training lambda only')
    parser.add_argument('--replay-ids', type=Path, help='Verified historical IDs as step -> list; default reconstructs and validates')
    parser.add_argument('--probe-manifest', type=Path, help='Reuse a v2 three-way manifest; original IDs are still independently validated')
    parser.add_argument('--loss-only', action='store_true', help='Forward-only diagnostics; gradient fields are left empty')
    return parser


def main(argv=None):
    parser = build_parser()
    cli = parser.parse_args(argv)
    if cli.batches < 1 or cli.threads < 1 or any(s < 1 for s in cli.steps) or len(set(cli.steps)) != len(cli.steps):
        parser.error('Positive batches/threads and unique steps >= 1 required')
    if any(v is not None and v < 1 for v in [cli.current_batch_size, cli.replay_batch_size]):
        parser.error('Batch sizes must be positive')
    if cli.lambdas is not None and any(not math.isfinite(v) or v < 0 for v in cli.lambdas):
        parser.error('Lambdas must be finite and non-negative')
    if cli.output is None:
        cli.output = HERE/'results'/f'three_way_{datetime.now():%Y%m%d_%H%M%S}_{cli.run_dir.name}'
    out = cli.output.resolve()
    if out.is_relative_to(cli.run_dir.resolve()):
        parser.error('Analysis output must be outside the checkpoint directory')
    native_out = probe.native(out)
    if native_out.exists() and (not native_out.is_dir() or any(native_out.iterdir())):
        parser.error('Output must be new or empty; previous results are never overwritten')
    probe.native(out).mkdir(parents=True, exist_ok=True)
    try:
        run(cli)
    except ImportError as exc:
        raise SystemExit(f'Use a training-compatible environment with PyTorch/numpy/h5py/tqdm: {exc}') from exc


if __name__ == '__main__':
    main()
