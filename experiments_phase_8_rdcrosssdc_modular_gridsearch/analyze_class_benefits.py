"""Step 1: paired class/time/difficulty analysis using saved CSVs only."""
import argparse
from collections import defaultdict
import html
from pathlib import Path
import statistics

import class_analysis_common as common


def aggregate(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        out = dict(zip(keys, key))
        for field in fields:
            out.update({field+'_'+k: v for k, v in common.summarize(r[field] for r in group).items()})
        result.append(out)
    return result


def attach_trust(rows, runs):
    references = {}
    for run in runs:
        if run['method'] == 'avcil':
            continue
        for step in range(1, 10):
            directory = Path(run['metrics'])/'rd_crosssdc'
            static = common.read_csv(directory/f'step_{step}_static_trust.csv')
            # These selected experiments use fixed trust-only weights. Save the
            # source epoch explicitly instead of presenting it as best-epoch Need.
            weights = {int(r['class_id']): r for r in common.read_csv(directory/f'step_{step}_epoch_0_weights.csv')}
            for r in static:
                c = int(r['class_id'])
                if len(static) != step*10 or c not in weights:
                    raise ValueError('Incomplete historical Trust table')
                references[run['method'], run['seed'], step, c] = dict(
                    historical_prototype_count=float(r['prototype_count']), weight_source_epoch=0,
                    **{k: float(r[k]) for k in ['trust_a_from_v', 'trust_v_from_a',
                                               'reliability_a_from_v', 'reliability_v_from_a']},
                    **{k: float(weights[c][k]) for k in ['cmr_weight_a_from_v', 'cmr_weight_v_from_a']})
    for row in rows:
        row.update(references.get((row['method'], row['seed'], row['step'], row['class_id']), {}))


def grouped_summaries(rows):
    buckets = defaultdict(list)
    for r in rows:
        groups = [('all', 'all'), ('difficulty', str(r['difficulty_quartile'])),
                  ('age', r['age_group']), ('cohort', str(r['first_seen_step'])),
                  ('difficulty_age', f"Q{r['difficulty_quartile']}|{r['age_group']}")]
        for dimension, value in groups:
            buckets[r['method'], r['seed'], r['step'], dimension, value].append(r)
    fields = ['delta_f1_pp', 'delta_precision_pp', 'delta_recall_pp',
              'initial_delta_f1_pp', 'relative_retention_delta_f1_pp']
    per_seed = []
    for key, group in sorted(buckets.items()):
        out = dict(zip(['method', 'seed', 'step', 'group_by', 'group'], key))
        out.update(n_classes=len(group), mean_difficulty_rank=statistics.mean(r['difficulty_rank'] for r in group))
        out.update({f: statistics.mean(r[f] for r in group) for f in fields})
        out['fraction_classes_improved'] = sum(r['delta_f1_pp']>1e-6 for r in group)/len(group)
        per_seed.append(out)
    across_seed = aggregate(per_seed, ['method', 'step', 'group_by', 'group'], fields+['fraction_classes_improved'])
    return per_seed, across_seed


def table(rows, columns):
    def cell(value):
        return f'{value:.3f}' if isinstance(value, float) else str(value if value is not None else '')
    return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(cell(r.get(c)))+'</td>' for c in columns)+'</tr>' for r in rows)+'</table>'


def run(args):
    ranks = common.difficulty(args.difficulty)
    runs = common.discover_runs(args.archive_root, args.seeds)
    tables, order = common.load_metrics(runs, ranks)
    config = dict(seeds=args.seeds, runs=runs, difficulty=ranks,
                  input_hashes={r['name']: common.sha256(Path(r['metrics'])/'per_class_metrics.csv') for r in runs},
                  source_hashes={p: common.sha256(common.HERE/p) for p in ['analyze_class_benefits.py', 'class_analysis_common.py']})
    # Include every historical diagnostic read, so a resume cannot silently mix revisions.
    config['trust_hashes'] = {r['name']: {p.name: common.sha256(p)
        for p in sorted(common.native(r['metrics']).glob('rd_crosssdc/step_*'))
        if p.name.endswith('_static_trust.csv') or p.name.endswith('_epoch_0_weights.csv')}
        for r in runs if r['method'] != 'avcil'}
    out = common.prepare_output(args.output, config, args.resume)
    rows = common.paired_deltas(tables, order, ranks)
    attach_trust(rows, runs)
    common.write_csv(out/'class_deltas.csv', rows)
    fields = ['delta_f1_pp', 'delta_precision_pp', 'delta_recall_pp',
              'initial_delta_f1_pp', 'relative_retention_delta_f1_pp', 'step_delta_change_pp']
    summaries = aggregate(rows, ['method', 'step', 'class_id', 'category_name',
                                'first_seen_step', 'age', 'difficulty_rank', 'difficulty_quartile'], fields)
    common.write_csv(out/'class_summary.csv', summaries)
    common.write_csv(out/'final_class_summary.csv', [r for r in summaries if r['step'] == 9])
    per_seed, grouped = grouped_summaries(rows)
    common.write_csv(out/'group_by_seed.csv', per_seed)
    common.write_csv(out/'group_summary.csv', grouped)
    common.write_csv(out/'class_mapping.csv', [common.class_info(c, 9, order, ranks) for c in range(100)])
    checks = []
    for (method, seed), data in tables.items():
        if method == 'avcil':
            continue
        base = tables['avcil', seed]
        checks.append(dict(method=method, seed=seed, n_class_step_rows=len(data),
            step0_counts_identical=all(all(data[0,c][k] == base[0,c][k] for k in ['tp','fp','fn']) for c in range(10))))
    common.write_json(out/'audit.json', dict(runs=checks, class_order_shared=True,
        difficulty_join='category_name, not class_id', paired_unit='seed + class + evaluation_step',
        uncertainty='SD across seeds; no independence assumed across steps, classes or repeated test samples',
        retention='delta_f1(t) - delta_f1(first_seen); not the max-history forget_f1 column',
        support_values=sorted({r['support'] for t in tables.values() for r in t.values()})))
    overview = [r for r in grouped if r['step']==9 and r['group_by']=='all']
    content = '<!doctype html><meta charset="utf-8"><title>逐类收益分析</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}th{background:#eef3f8}</style>'
    content += '<h1>AVCIL 与 hinge A/B：逐类、时间与难度</h1><p>'+html.escape(common.ATTRIBUTION)+'</p>'
    content += '<p>所有差值均为方法减 AVCIL，单位为百分点。难度四分位 Q1 最易、Q4 最难。先在每个 seed 内按类别取均值，再汇总 seed；不将全部 class×step 行当成独立重复。</p>'
    content += table(overview, ['method','delta_f1_pp_mean','delta_f1_pp_sd','relative_retention_delta_f1_pp_mean'])
    for method in ['hinge_A','hinge_B']:
        final = sorted([r for r in summaries if r['method']==method and r['step']==9], key=lambda r:r['delta_f1_pp_mean'])
        content += '<h2>'+method+'：最终差值最低／最高各 10 类</h2>'
        content += table(final[:10]+final[-10:], ['category_name','difficulty_rank','first_seen_step',
            'delta_f1_pp_mean','delta_f1_pp_sd','delta_f1_pp_n_positive','delta_f1_pp_n_negative',
            'initial_delta_f1_pp_mean','relative_retention_delta_f1_pp_mean'])
    content += '<p>完整数据：'+ ' · '.join(f'<a href="{p}">{p}</a>' for p in ['class_deltas.csv','class_summary.csv','final_class_summary.csv','group_summary.csv','group_by_seed.csv','class_mapping.csv','audit.json'])+'</p>'
    (out/'report.html').write_text(content, encoding='utf-8')
    common.write_json(out/'complete.json', dict(status='complete', comparisons=len(rows)))
    print('Step 1 complete:', out, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root', type=Path, default=common.HERE)
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--difficulty', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seeds')
    run(args)
