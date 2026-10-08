"""Analysis regression tests with hand-computable, synthetic run outputs."""

from pathlib import Path

import pytest

from experiments_phase_9.analysis.common import DEFAULTS, read_csv, read_json, write_csv, write_json
from experiments_phase_9.analysis.offline import analyze, expected_loss, parse_namespace, _tertiles


def fixture_manifest(tmp_path):
    runs = []
    for method in ("uniform", "periodic"):
        cfg = {**DEFAULTS, "fusion_mode": method, "fusion_update_rule": "sample_aware", "num_classes": 4,
               "class_num_per_step": 2, "max_epoches": 3, "fusion_warmup_epochs": 1, "fusion_update_interval": 1}
        run_dir = tmp_path / "save" / method
        metrics = tmp_path / "save/metrics" / method
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", cfg)
        run = {"run_id": method, "run_dir": str(run_dir), "metrics_dir": str(metrics), "log_path": None,
               "config": cfg, "method": method, "method_id": method, "protocol_id": "shared", "seed": 42, "num_steps": 2}
        runs.append(run)
        epochs, classes, results = [], [], []
        for step in range(2):
            nclasses = (step + 1) * 2
            snapshots = {0: [.5] * nclasses}
            if step and method == "periodic":
                # n=10, n_ref=10 => eta=.25, raw=.75.
                snapshots[1] = [.5625] * nclasses
                snapshots[2] = [.609375] * nclasses
            for after, gates in snapshots.items():
                rows = []
                for c, gate in enumerate(gates):
                    row = dict(step=step, after_epoch=after, active_from_epoch=after + 1,
                               gate_version=after, class_id=c, audio_gate=gate, visual_gate=1 - gate)
                    if after:
                        row.update(counts=10, scored_counts=10, reliability_a=.6, reliability_v=.2,
                                   valid=True, eta=.25, previous_gate=snapshots[after - 1][c], raw_gate=.75)
                    rows.append(row)
                filename = f"step_{step}_initial_gates.csv" if after == 0 else f"step_{step}_after_epoch_{after}_gates.csv"
                write_csv(metrics / "class_fusion" / filename, rows)
            for epoch in (1, 2, 3):
                active_after = max(after for after in snapshots if after < epoch)
                g = snapshots[active_after][0]
                row = dict(step=step, epoch=epoch, train_loss=1., ce=1., kd=0., instance=0., **{"class": 0.}, spatial=0., temporal=0.,
                           val_acc=.9 if epoch >= 2 else .8, gate_version=active_after, gate_min=g, gate_max=g, gate_mean=g)
                epochs.append(row)
            test_gate = snapshots[max(after for after in snapshots if after < 2)]
            tp = 8 if method == "periodic" and step else 7
            for c in range(nclasses):
                # Equal support; fp==fn gives precision==recall==f1.
                classes.append(dict(step=step, class_id=c, category_name=f"class{c}", tp=tp, fp=10-tp, fn=10-tp,
                                    support=10, precision=tp/10, recall=tp/10, f1=tp/10, first_seen_step=c//2))
            result = dict(step=step, overall_acc=tp/10, macro_f1=tp/10, forgetting=None if step == 0 else .7-tp/10,
                          gate_version=max(after for after in snapshots if after < 2), audio_gates=test_gate)
            results.append(result)
            write_json(metrics / f"step_{step}_test.json", result)
        write_csv(metrics / "class_fusion/epoch_summary.csv", epochs)
        write_csv(metrics / "per_class_metrics.csv", classes)
        write_json(metrics / "summary.json", dict(steps=results, average_incremental_accuracy=sum(r['overall_acc'] for r in results)/2, average_forgetting=results[-1]['forgetting']))
        history = [dict(step=1, epoch=0, events="task_start", class_id=c, comparable=True, teacher_R_a=.2 + c*.3, teacher_R_v=.3,
                        teacher_classification_valid=True, teacher_old_only_accuracy=.6+c*.1,
                        teacher_audio_only_correct_rate=.2, teacher_visual_only_correct_rate=.1+c*.1) for c in range(2)]
        write_csv(metrics / "cl_history/class_history.csv", history)
    return {"schema_version": 1, "phase_root": str(tmp_path), "save_root": str(tmp_path / "save"),
            "logs_root": str(tmp_path / "logs"), "output_root": str(tmp_path / "results"), "options": {}, "runs": runs}


def test_namespace_parser_never_executes(tmp_path):
    assert parse_namespace("Namespace(seed=42, flags=[1, 2], enabled=True, absent=None)") == {
        "seed": 42, "flags": [1, 2], "enabled": True, "absent": None}
    marker = tmp_path / "executed"
    with pytest.raises(ValueError):
        parse_namespace(f"Namespace(seed=__import__('pathlib').Path({str(marker)!r}).touch())")
    assert not marker.exists()
    for text in ("dict(seed=2)", "Namespace(**{'seed':2})", "Namespace(1)", "Namespace(seed=1, seed=2)"):
        with pytest.raises(ValueError):
            parse_namespace(text)


def test_offline_known_metrics_gate_timing_and_tied_best(tmp_path):
    manifest = fixture_manifest(tmp_path)
    before = {p: p.read_bytes() for p in (tmp_path / 'save').rglob('*') if p.is_file()}
    result = analyze(manifest)
    assert result["counts"]["run_summary"] == 2
    table_dir = tmp_path / "results/tables"
    summaries = {r["run_id"]: r for r in read_csv(table_dir / "run_summary.csv")}
    assert summaries["periodic"]["average_incremental_accuracy"] == pytest.approx(.75)
    assert summaries["periodic"]["final_forgetting"] == pytest.approx(-.1)
    paired = next(r for r in read_csv(table_dir / 'paired_metrics.csv') if r['metric'] == 'final_accuracy')
    assert paired['delta_pp'] == pytest.approx(10)
    step = next(r for r in read_csv(table_dir / 'steps.csv') if r['run_id'] == 'periodic' and r['step'] == 1)
    assert step['best_epoch'] == 2  # strict best selects first tied max
    assert step['gate_version'] == 1  # update after epoch 2 must not leak into best
    assert step['gate_mean_abs_deviation'] == pytest.approx(.0625)
    assert not [r for r in read_csv(table_dir/'audit.csv') if r['status'] == 'fail']
    assert all(p.read_bytes() == content for p, content in before.items())
    assert any(r['variable'] == 'reference_difficulty' for r in read_csv(table_dir/'subgroup_metrics.csv'))
    assert not any('baseline_test_difficulty' in str(r) for r in read_csv(table_dir/'subgroup_membership.csv'))


def test_partial_or_duplicate_classes_never_get_primary_aggregate(tmp_path):
    manifest = fixture_manifest(tmp_path)
    dynamic_root = Path(manifest['runs'][1]['metrics_dir'])
    rows = read_csv(dynamic_root / 'per_class_metrics.csv')
    write_csv(dynamic_root / 'per_class_metrics.csv', [*rows, rows[-1]])
    analyze(manifest)
    row = next(r for r in read_csv(tmp_path/'results/tables/run_summary.csv') if r['run_id'] == 'periodic')
    assert row['complete'] is False
    assert row['average_incremental_accuracy'] == ''
    assert not read_csv(tmp_path/'results/tables/paired_metrics.csv')
    assert any(r['check'] == 'duplicate_class_metrics' and r['status'] == 'fail' for r in read_csv(tmp_path/'results/tables/audit.csv'))


def test_loss_recombination_and_quantile_ties():
    row = dict(step=1, ce=1., kd=2., instance=3., **{'class':4.}, spatial=5., temporal=6.)
    assert expected_loss(row, DEFAULTS) == pytest.approx(12.8)
    assert expected_loss({**row, 'step':0}, DEFAULTS) == 1.
    assert expected_loss(row, {**DEFAULTS, 'attn_score_distil':False, 'class_contrastive':False}) == pytest.approx(3.3)
    bins = _tertiles([{'class_id':c, 'x':.1} for c in range(6)], 'x')
    assert len(set(bins.values())) == 1  # never split equal-score classes to manufacture contrast


def test_gate_corruption_is_audited_and_nonfinite_epochs_not_imputed(tmp_path):
    manifest = fixture_manifest(tmp_path)
    dynamic = Path(manifest['runs'][1]['metrics_dir'])
    gate_path = dynamic / 'class_fusion/step_1_after_epoch_1_gates.csv'
    gates = read_csv(gate_path)
    gates[0]['eta'] = .9
    write_csv(gate_path, gates)
    epoch_path = dynamic / 'class_fusion/epoch_summary.csv'
    epochs = read_csv(epoch_path)
    epochs[0]['train_loss'] = float('nan')
    write_csv(epoch_path, epochs)
    analyze(manifest)
    failures = {r['check'] for r in read_csv(tmp_path/'results/tables/audit.csv') if r['status'] == 'fail'}
    assert {'gate_eta', 'gate_update_formula', 'finite_epoch_metrics'} <= failures
    assert not any(r['run_id'] == 'periodic' and r['step'] == 0 and r['epoch'] == 1 for r in read_csv(tmp_path/'results/tables/epoch_losses.csv'))


@pytest.mark.parametrize('bad_version', ['', float('nan'), -1, .5])
def test_bad_epoch_version_and_prototype_identifiers_do_not_abort_other_runs(tmp_path, bad_version):
    manifest = fixture_manifest(tmp_path)
    # Process the damaged run first, proving that the next run is still analyzed.
    manifest['runs'].reverse()
    damaged = Path(manifest['runs'][0]['metrics_dir'])
    epoch_path = damaged / 'class_fusion/epoch_summary.csv'
    epochs = read_csv(epoch_path)
    epochs[0]['gate_version'] = bad_version
    write_csv(epoch_path, epochs)
    good = dict(step=1, after_epoch=1, class_id=0, history_used=True, reason='ok')
    write_csv(damaged / 'prototype_bank/step_1_after_epoch_1.csv', [
        {**good, 'step': ''}, {**good, 'after_epoch': 'not_an_epoch'},
        {**good, 'class_id': float('nan')}, {**good, 'class_id': 4},
        {**good, 'step': 2}, {**good, 'after_epoch': -1},
        {**good, 'class_id': .5}, good,
    ])
    result = analyze(manifest)
    assert result['counts']['run_summary'] == 2
    tables = tmp_path / 'results/tables'
    assert next(r for r in read_csv(tables/'run_summary.csv') if r['run_id'] == 'uniform')['complete'] is True
    assert not any(r['run_id'] == 'periodic' and r['step'] == 0 and r['epoch'] == 1 for r in read_csv(tables/'epoch_losses.csv'))
    prototypes = read_csv(tables/'prototype_bank.csv')
    assert len(prototypes) == 1 and prototypes[0]['class_id'] == 0
    failures = [r for r in read_csv(tables/'audit.csv') if r['run_id'] == 'periodic' and r['status'] == 'fail']
    assert any(r['check'] in ('finite_epoch_metrics', 'integer_epoch_identifiers') for r in failures)
    assert next(r for r in failures if r['check'] == 'prototype_row_identifiers')['count'] == 7
