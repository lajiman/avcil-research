"""Shared, torch-free inputs for the two independent class analyses."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ATTRIBUTION = ('A/B versus AVCIL compares CrossSDC-I + CMR against AVCIL, not CMR alone. '
               'A versus B changes both lambda and alpha. Seeds share one class order.')


def native(path):
    path = str(Path(path).resolve())
    return Path('\\\\?\\' + path) if os.name == 'nt' and not path.startswith('\\\\?\\') else Path(path)


def read_csv(path):
    with native(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = native(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Refusing to silently omit an empty table: ' + str(path))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path, value):
    path = native(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with native(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_output(output, config, resume=False):
    out = native(output)
    record = out / 'manifest.json'
    if out.exists() and any(out.iterdir()):
        if not resume or not record.is_file():
            raise ValueError('Output is not empty; choose a new directory or use --resume: ' + str(out))
        previous = json.loads(record.read_text(encoding='utf-8'))
        if previous['signature'] != signature(config):
            raise ValueError('Resume inputs/options/source changed; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(record, {'signature': signature(config), 'config': config, 'attribution': ATTRIBUTION})
    return out


def difficulty(path=None):
    values = json.loads(native(path or HERE/'class_difficulty_easy2hard.json').read_text(encoding='utf-8'))
    if len(values) != 100 or sorted(values.values()) != list(range(100)):
        raise ValueError('Difficulty mapping must contain the 100 unique ranks 0..99')
    return values


def discover_runs(root, seeds):
    """Explicit default controls: classic AVCIL, hinge A/B at tolerance .01."""
    root = native(root)
    result = []
    for seed in seeds:
        for method in ['avcil', 'hinge_A', 'hinge_B']:
            if method == 'avcil':
                archive = root/'save_supp_ckpt_v1'
                pattern = f'VGGSound_random_balance_avcil_only_supp_ckpt_v1_h200_seed{seed}'
            else:
                archive = root/'save_commands_cmr_hinge_tolerance_focus_3seeds'
                tag, setting = ('A', 'lc0p03_a0p5') if method == 'hinge_A' else ('B', 'lc0p1_a1p0')
                pattern = f'VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_tol_focus_v1_hinge_{tag}_{setting}_tol0p01_s1p0_h200_seed{seed}'
            metrics = archive/'metrics'/pattern
            if not (metrics/'per_class_metrics.csv').is_file():
                raise FileNotFoundError(str(metrics/'per_class_metrics.csv'))
            result.append(dict(method=method, seed=seed, name=pattern,
                               metrics=str(metrics), checkpoints=str(archive/pattern)))
    return result


def load_metrics(runs, ranks):
    tables, order = {}, None
    for run in runs:
        rows = read_csv(Path(run['metrics'])/'per_class_metrics.csv')
        index = {}
        for raw in rows:
            row = dict(raw)
            for k in ['step', 'class_id', 'first_seen_step', 'support', 'tp', 'fp', 'fn']:
                row[k] = int(row[k])
            for k in ['f1', 'precision', 'recall']:
                row[k] = float(row[k])
            key = row['step'], row['class_id']
            if key in index:
                raise ValueError('Duplicate class/step in ' + run['name'])
            if row['category_name'] not in ranks or row['first_seen_step'] != row['class_id']//10:
                raise ValueError('Category mapping/first-seen mismatch in ' + run['name'])
            tp, fp, fn = (row[k] for k in ['tp', 'fp', 'fn'])
            expected = dict(precision=tp/(tp+fp) if tp+fp else 0.,
                            recall=tp/(tp+fn) if tp+fn else 0.,
                            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            if row['support'] != tp+fn or min(tp, fp, fn) < 0:
                raise ValueError('Invalid counts in ' + run['name'])
            if any(not math.isclose(row[k], v, abs_tol=1e-6) for k, v in expected.items()):
                raise ValueError('Counts do not reconstruct PRF in ' + run['name'])
            index[key] = row
        expected_keys = {(t, c) for t in range(10) for c in range((t+1)*10)}
        if set(index) != expected_keys:
            raise ValueError('Expected all 550 class/step rows in ' + run['name'])
        current_order = tuple(index[9, c]['category_name'] for c in range(100))
        if order is None:
            order = current_order
        if order != current_order or any(r['category_name'] != order[r['class_id']] for r in index.values()):
            raise ValueError('Runs have different class orders')
        tables[run['method'], run['seed']] = index
    for (method, seed), table in tables.items():
        base = tables['avcil', seed]
        if any(table[k]['support'] != base[k]['support'] for k in table):
            raise ValueError('Paired runs have different test support')
    return tables, order


def class_info(c, step, order, ranks):
    rank = ranks[order[c]]
    age = step-c//10
    return dict(class_id=c, category_name=order[c], first_seen_step=c//10, age=age,
                difficulty_rank=rank, difficulty_quartile=rank//25+1,
                age_group='new' if age == 0 else ('1-2' if age <= 2 else ('3-5' if age <= 5 else '6+')))


def paired_deltas(tables, order, ranks):
    result = []
    for (method, seed), table in tables.items():
        if method == 'avcil':
            continue
        base = tables['avcil', seed]
        for (step, c), row in sorted(table.items()):
            ref, first = base[step, c], c//10
            delta = 100*(row['f1']-ref['f1'])
            initial = 100*(table[first, c]['f1']-base[first, c]['f1'])
            previous = 100*(table[step-1, c]['f1']-base[step-1, c]['f1']) if step > first else None
            out = dict(method=method, seed=seed, step=step, **class_info(c, step, order, ranks),
                       support=row['support'], initial_delta_f1_pp=initial,
                       relative_retention_delta_f1_pp=delta-initial,
                       step_delta_change_pp=delta-previous if previous is not None else None)
            for metric in ['f1', 'precision', 'recall']:
                out.update({f'avcil_{metric}': ref[metric], f'method_{metric}': row[metric],
                            f'delta_{metric}_pp': 100*(row[metric]-ref[metric])})
            for metric in ['tp', 'fp', 'fn']:
                out.update({f'avcil_{metric}': ref[metric], f'method_{metric}': row[metric]})
            result.append(out)
    return result


def summarize(values):
    values = [float(x) for x in values if x is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None,
                n_positive=sum(x>1e-6 for x in values), n_negative=sum(x < -1e-6 for x in values))
