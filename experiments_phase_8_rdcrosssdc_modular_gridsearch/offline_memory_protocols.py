"""Sampling and reference checks used only by the offline three-way probe.

No trainer imports, optimizer operations, or global random-state changes.
"""
from collections import Counter
import csv
import math
from pathlib import Path
import random


GROUPS = {
    'original_memory': ('original', 'original_replay_ids'),
    'heldout_fixed_reference': ('original', 'fresh_replay_ids'),
    'fresh_memory': ('fresh', 'fresh_replay_ids'),
}
REFERENCE_ATOL = 2e-5
REFERENCE_RTOL = 2e-4


def sample_label(dataset, vid):
    return int(dataset.category_encode_dict[dataset.all_id_category_dict[vid]])


def available_ids(dataset, class_id, *, ordered=False):
    ids = [str(v) for v in dataset._get_class_vids(class_id) if dataset._has_feature(v)]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate train IDs in class {class_id}')
    if any(sample_label(dataset, v) != class_id for v in ids):
        raise ValueError(f'Train class mapping mismatch for class {class_id}')
    return ids if ordered else sorted(ids)


def reconstruct_replay_ids(dataset, args, steps):
    """Replay Python's original sample/shrink sequence, preserving metadata order.

    These are CANDIDATE historical IDs until teacher statistics match the saved
    static CSV. The seed alone is not evidence of historical identity.
    """
    rng = random.Random(int(args.seed))
    class_memory, result = [], {}
    requested = set(steps)
    for step in range(1, max(steps) + 1):
        count = args.memory_size // (step * args.class_num_per_step)
        if count < 1:
            raise ValueError('Original memory cannot cover all old classes')
        new_classes = []
        for c in range((step-1)*args.class_num_per_step, step*args.class_num_per_step):
            pool = available_ids(dataset, c, ordered=True)
            selected = rng.sample(pool, min(len(pool), count))
            new_classes.append(selected + [None] * (count-len(selected)))
        class_memory = [ids[:count] for ids in class_memory] + new_classes
        if step in requested:
            result[str(step)] = [v for ids in class_memory for v in ids if v is not None]
    return result


def validate_memory(dataset, args, step, ids):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Memory must contain unique, nonempty IDs')
    labels = []
    for vid in ids:
        if vid not in dataset.all_id_category_dict or not dataset._has_feature(vid):
            raise ValueError(f'Unknown train sample or missing feature: {vid}')
        labels.append(sample_label(dataset, vid))
    if set(labels) != set(range(step*args.class_num_per_step)):
        raise ValueError('Memory must cover exactly the old classes')
    return Counter(labels)


def make_three_way_plan(dataset, args, step, original_ids, seed, batches,
                        current_batch_size, replay_batch_size):
    """Match current IDs and old-label sequences; groups 2/3 share query IDs.

    The fresh bank is disjoint from the original bank, with equal per-class
    counts. No silent shrinking/replacement: insufficient pools are an error.
    """
    counts = validate_memory(dataset, args, step, original_ids)
    rng = random.Random(seed + step*100003)
    old_classes = step*args.class_num_per_step
    original_set = set(original_ids)
    fresh, fresh_by_class = [], {}
    for c in range(old_classes):
        pool = [v for v in available_ids(dataset, c) if v not in original_set]
        if len(pool) < counts[c]:
            raise ValueError(f'Class {c}: need {counts[c]} non-memory train samples, found {len(pool)}')
        fresh_by_class[c] = rng.sample(pool, counts[c])
        fresh.extend(fresh_by_class[c])
    current_pool = [v for c in range(old_classes, old_classes+args.class_num_per_step)
                    for v in available_ids(dataset, c)]
    if len(current_pool) < current_batch_size or len(original_ids) < replay_batch_size:
        raise ValueError('Insufficient samples for requested batch size; sizes are never silently reduced')
    batch_plans = []
    for _ in range(batches):
        current = rng.sample(current_pool, current_batch_size)
        original = rng.sample(list(original_ids), replay_batch_size)
        labels = [sample_label(dataset, v) for v in original]
        sampled = {c: iter(rng.sample(fresh_by_class[c], n)) for c, n in sorted(Counter(labels).items())}
        queries = [next(sampled[c]) for c in labels]
        batch_plans.append(dict(current_ids=current, original_replay_ids=original, fresh_replay_ids=queries))
    return dict(original_memory_ids=list(original_ids), fresh_memory_ids=fresh, batches=batch_plans)


def validate_three_way_plan(dataset, args, step, plan, original_ids,
                            batches, current_batch_size, replay_batch_size):
    if plan['original_memory_ids'] != list(original_ids):
        raise ValueError('Manifest original memory differs from reconstructed/supplied IDs')
    old_counts = validate_memory(dataset, args, step, plan['original_memory_ids'])
    if validate_memory(dataset, args, step, plan['fresh_memory_ids']) != old_counts:
        raise ValueError('Fresh bank must have the original per-class counts')
    old, fresh = set(plan['original_memory_ids']), set(plan['fresh_memory_ids'])
    if old & fresh:
        raise ValueError('Fresh bank must be disjoint from original memory')
    if len(plan['batches']) != batches:
        raise ValueError('Manifest batch count differs from --batches')
    old_classes = step*args.class_num_per_step
    for batch in plan['batches']:
        for key, expected in [('current_ids', current_batch_size),
                              ('original_replay_ids', replay_batch_size), ('fresh_replay_ids', replay_batch_size)]:
            ids = batch[key]
            if len(ids) != expected or len(set(ids)) != expected:
                raise ValueError(f'{key}: wrong batch size or repeated IDs within a batch')
            if any(v not in dataset.all_id_category_dict or not dataset._has_feature(v) for v in ids):
                raise ValueError(f'{key}: unknown train ID or missing feature')
        if not set(batch['original_replay_ids']) <= old or not set(batch['fresh_replay_ids']) <= fresh:
            raise ValueError('Replay queries do not belong to their specified memory')
        if [sample_label(dataset, v) for v in batch['original_replay_ids']] != [sample_label(dataset, v) for v in batch['fresh_replay_ids']]:
            raise ValueError('Original/fresh batches must match old-class labels in order')
        if any(not old_classes <= sample_label(dataset, v) < old_classes+args.class_num_per_step for v in batch['current_ids']):
            raise ValueError('Current batch contains a sample outside the current classes')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def best_epoch_row(rows, step):
    rows = sorted((r for r in rows if int(r['step']) == step), key=lambda r: int(r['epoch']))
    if not rows or len({int(r['epoch']) for r in rows}) != len(rows):
        raise ValueError(f'Missing or duplicated epoch records for step {step}')
    if any(not math.isfinite(float(r['val_acc'])) for r in rows):
        raise ValueError('Nonfinite historical validation accuracy')
    # Trainer uses strict >, so the first epoch wins ties.
    return max(rows, key=lambda r: float(r['val_acc']))


def indexed_class_rows(rows, num_classes, step, epoch=None):
    if len(rows) != num_classes or {int(r['class_id']) for r in rows} != set(range(num_classes)):
        raise ValueError('Historical CSV must contain each old class exactly once')
    if any(int(r['step']) != step or (epoch is not None and int(r['epoch']) != epoch) for r in rows):
        raise ValueError('Historical CSV step/epoch mismatch')
    return {int(r['class_id']): r for r in rows}


def compare_reference_rows(actual_rows, saved_rows, step):
    """Return an auditable numerical fingerprint check, never just a seed check."""
    saved = indexed_class_rows(saved_rows, len(actual_rows), step)
    indexed_class_rows(actual_rows, len(actual_rows), step)
    checks = []
    for row in actual_rows:
        c = int(row['class_id'])
        if row['category_name'] != saved[c]['category_name']:
            raise ValueError(f'Historical category mapping differs at class {c}')
        for field in ['prototype_count', 'reliability_a_from_v', 'reliability_v_from_a',
                      'trust_a_from_v', 'trust_v_from_a', 'cmr_weight_a_from_v', 'cmr_weight_v_from_a']:
            a, b = float(row[field]), float(saved[c][field])
            atol, rtol = (0., 0.) if field == 'prototype_count' else (REFERENCE_ATOL, REFERENCE_RTOL)
            passed = math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
            checks.append(dict(step=step, class_id=c, field=field, reconstructed=a, historical=b,
                               absolute_error=abs(a-b), atol=atol, rtol=rtol, passed=int(passed)))
    return checks


def membership_margin_terms(current_audio, current_visual, old_audio, old_visual,
                            labels, bank, temperature, tolerance, membership):
    """Use teacher LOO only for queries actually included in the prototype sum.

    Shared training functions remain unchanged. Zero subtraction for external
    queries leaves their target prototype equal to the full reference prototype.
    """
    import torch
    from rd_crosssdc import rd_method as rd
    if membership.shape != labels.shape or membership.dtype != torch.bool:
        raise ValueError('Membership must be a boolean vector matching replay labels')
    if bool(membership.all()):
        return rd.compute_margin_terms(current_audio, current_visual, old_audio, old_visual,
                                       labels, bank, temperature, tolerance)
    mask = membership.to(old_audio.dtype).unsqueeze(1)
    def margin(query, prototypes, sums, positives):
        return rd._cross_modal_margin(query, labels, prototypes, sums, bank.counts,
                                      positives.detach()*mask, temperature)
    ref_a = margin(old_audio, bank.visual_prototypes, bank.visual_sums, old_visual).detach()
    cur_a = margin(current_audio, bank.visual_prototypes, bank.visual_sums, old_visual)
    ref_v = margin(old_visual, bank.audio_prototypes, bank.audio_sums, old_audio).detach()
    cur_v = margin(current_visual, bank.audio_prototypes, bank.audio_sums, old_audio)
    return rd.MarginTerms(ref_a, cur_a, ref_v, cur_v,
                          torch.relu(ref_a-cur_a), torch.relu(ref_v-cur_v),
                          torch.relu(ref_a-cur_a-tolerance), torch.relu(ref_v-cur_v-tolerance))
