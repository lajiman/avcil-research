import os
import math
import random
from collections import defaultdict, Counter
import numpy as np
import csv
import h5py

RNG_SEED = 42


def parse_vggsound_csv(csv_path: str):
    """
    Each row: youtube_id, t, label, split
    vid = f"{youtube_id}_{t:06d}"
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) < 4:
                continue
            youtube_id = parts[0].strip()
            t = int(float(parts[1]))
            label = parts[2].strip()
            split = parts[3].strip().lower()
            vid = f"{youtube_id}_{t:06d}"
            rows.append((vid, label, split))
    return rows


def load_feature_vid_sets(feature_root: str):
    """
    Return:
      visual_vids: set of vids in visual_features.h5 root keys
      audio_vids:  set of vids in audio_pretrained_feature_dict.npy keys
    """
    visual_h5_path = os.path.join(feature_root, "visual_features.h5")
    audio_npy_path = os.path.join(feature_root, "audio_pretrained_feature", "audio_pretrained_feature_dict.npy")

    if not os.path.exists(visual_h5_path):
        raise FileNotFoundError(visual_h5_path)
    if not os.path.exists(audio_npy_path):
        raise FileNotFoundError(audio_npy_path)

    with h5py.File(visual_h5_path, "r") as f:
        visual_vids = set(f.keys())

    audio_dict = np.load(audio_npy_path, allow_pickle=True).item()
    audio_vids = set(audio_dict.keys())

    return visual_vids, audio_vids


def filter_rows_by_features(rows, visual_vids, audio_vids, require="both"):
    """
    require:
      "visual" : require vid in visual_vids
      "audio"  : require vid in audio_vids
      "both"   : require vid in both
      "either" : require vid in (visual union audio)

    Return:
      filtered_rows, stats
    """
    stats = {
        "before": len(rows),
        "after": 0,
        "dropped": 0,
        "dropped_by_split": defaultdict(int),
        "kept_by_split": defaultdict(int),
        "dropped_by_split_label": {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int),
        },
        "kept_by_split_label": {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int),
        },
    }

    out = []
    for vid, label, split in rows:
        has_v = (vid in visual_vids)
        has_a = (vid in audio_vids)

        if require == "visual":
            ok = has_v
        elif require == "audio":
            ok = has_a
        elif require == "both":
            ok = (has_v and has_a)
        elif require == "either":
            ok = (has_v or has_a)
        else:
            raise ValueError("require must be one of: visual/audio/both/either")

        if ok:
            out.append((vid, label, split))
            stats["kept_by_split"][split] += 1
            if split in stats["kept_by_split_label"]:
                stats["kept_by_split_label"][split][label] += 1
        else:
            stats["dropped_by_split"][split] += 1
            if split in stats["dropped_by_split_label"]:
                stats["dropped_by_split_label"][split][label] += 1

    stats["after"] = len(out)
    stats["dropped"] = stats["before"] - stats["after"]
    return out, stats


def save_feature_filter_stats(stats, out_path):
    """
    Save per-label per-split kept/dropped counts to CSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    kept = stats["kept_by_split_label"]
    drop = stats["dropped_by_split_label"]

    labels = set()
    for sp in ["train", "val", "test"]:
        labels |= set(kept.get(sp, {}).keys())
        labels |= set(drop.get(sp, {}).keys())
    labels = sorted(labels)

    header = [
        "label",
        "train_kept", "train_dropped", "train_before",
        "val_kept",   "val_dropped",   "val_before",
        "test_kept",  "test_dropped",  "test_before",
        "total_kept", "total_dropped", "total_before",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for lbl in labels:
            tr_k = kept["train"].get(lbl, 0)
            tr_d = drop["train"].get(lbl, 0)
            va_k = kept["val"].get(lbl, 0)
            va_d = drop["val"].get(lbl, 0)
            te_k = kept["test"].get(lbl, 0)
            te_d = drop["test"].get(lbl, 0)

            row = {
                "label": lbl,
                "train_kept": tr_k, "train_dropped": tr_d, "train_before": tr_k + tr_d,
                "val_kept": va_k,   "val_dropped": va_d,   "val_before": va_k + va_d,
                "test_kept": te_k,  "test_dropped": te_d,  "test_before": te_k + te_d,
                "total_kept": tr_k + va_k + te_k,
                "total_dropped": tr_d + va_d + te_d,
                "total_before": (tr_k + tr_d) + (va_k + va_d) + (te_k + te_d),
            }
            w.writerow(row)


def lt_targets(C, Nmax, gamma=10.0):
    """
    N_i = floor(Nmax * gamma^{-i/(C-1)})
    i=0 head, i=C-1 tail
    """
    out = []
    for i in range(C):
        val = Nmax * (gamma ** (-i / (C - 1)))
        out.append(max(1, int(math.floor(val))))
    return out


def adjust_to_total_with_caps(targets, caps, total, rng):
    """
    Adjust integer targets so sum == total, with 1<=targets[i]<=caps[i].
    If impossible (caps too small), warn and return best-effort.
    """
    targets = [max(1, min(int(t), int(c))) for t, c in zip(targets, caps)]
    cur = sum(targets)
    if cur == total:
        return targets

    def rebuild_inc_dec():
        inc = [i for i in range(len(targets)) if targets[i] < caps[i]]
        dec = [i for i in range(len(targets)) if targets[i] > 1]
        rng.shuffle(inc)
        rng.shuffle(dec)
        return inc, dec

    inc, dec = rebuild_inc_dec()

    if cur < total:
        need = total - cur
        j = 0
        while need > 0:
            if not inc:
                print(f"[WARN] Cannot reach total={total}. Short by {need} due to caps.")
                break
            i = inc[j % len(inc)]
            if targets[i] < caps[i]:
                targets[i] += 1
                need -= 1
            else:
                inc, dec = rebuild_inc_dec()
                j = 0
                continue
            j += 1
            if j % 10000 == 0:
                inc, dec = rebuild_inc_dec()
    else:
        need = cur - total
        j = 0
        while need > 0:
            if not dec:
                print(f"[WARN] Cannot reduce to total={total}. Over by {need} due to min=1.")
                break
            i = dec[j % len(dec)]
            if targets[i] > 1:
                targets[i] -= 1
                need -= 1
            else:
                inc, dec = rebuild_inc_dec()
                j = 0
                continue
            j += 1
            if j % 10000 == 0:
                inc, dec = rebuild_inc_dec()
    return targets


def make_category_encode_dict(class_order):
    return {label: int(cid) for cid, label in enumerate(class_order)}


def make_all_id_category_dict_from_classId_vid_dict(classId_vid_dict, class_order):
    """
    classId_vid_dict: {"train": {"0":[vid,...], ...}, "val": {...}, "test": {...}}
    return:
      {"train": {vid: label, ...}, "val": {...}, "test": {...}}
    """
    out = {"train": {}, "val": {}, "test": {}}
    for split in ["train", "val", "test"]:
        if split not in classId_vid_dict:
            continue
        split_map = out[split]
        for cid_str, vids in classId_vid_dict[split].items():
            cid = int(cid_str)
            label = class_order[cid]
            for vid in vids:
                if vid in split_map and split_map[vid] != label:
                    raise RuntimeError(
                        f"[Inconsistent] split={split} vid={vid} mapped to both "
                        f"'{split_map[vid]}' and '{label}'"
                    )
                split_map[vid] = label
    return out


def save_triplet(out_dir, all_classId_vid_dict, class_order):
    os.makedirs(out_dir, exist_ok=True)
    category_encode_dict = make_category_encode_dict(class_order)
    all_id_category_dict = make_all_id_category_dict_from_classId_vid_dict(all_classId_vid_dict, class_order)

    np.save(os.path.join(out_dir, "all_classId_vid_dict.npy"), all_classId_vid_dict, allow_pickle=True)
    np.save(os.path.join(out_dir, "category_encode_dict.npy"), category_encode_dict, allow_pickle=True)
    np.save(os.path.join(out_dir, "all_id_category_dict.npy"), all_id_category_dict, allow_pickle=True)


def sample_train_dict_from_targets(train_pool_per_label, class_order_fixed, targets_by_cid):
    C = len(class_order_fixed)
    out = {}
    for cid in range(C):
        lbl = class_order_fixed[cid]
        vids = train_pool_per_label[lbl]
        k = min(int(targets_by_cid[cid]), len(vids))
        out[str(cid)] = vids[:k]
    return out


def build_base_pools(csv_path: str,
                     top_k=100,
                     ranking="all",
                     val_per_class=50,
                     test_per_class=50,
                     pool_shuffle_seed=42,
                     feature_root=None,
                     require_features="both"):
    """
    If feature_root is provided:
      - Filter out rows whose vid is missing required features BEFORE any counting/sampling.

    Note:
      This function no longer shuffles class order.
      It only builds top_labels + val/test/train pools.
    """
    rows = parse_vggsound_csv(csv_path)

    if feature_root is not None:
        visual_vids, audio_vids = load_feature_vid_sets(feature_root)
        rows, stats = filter_rows_by_features(rows, visual_vids, audio_vids, require=require_features)
        print("[Feature Filter]")
        stats_csv = os.path.join(os.path.dirname(csv_path), "feature_filter_stats.csv")
        save_feature_filter_stats(stats, out_path=stats_csv)
        print("  saved per-class stats to:", stats_csv)
        print(f"  require = {require_features}")
        print(f"  before  = {stats['before']}")
        print(f"  after   = {stats['after']}")
        print(f"  dropped = {stats['dropped']}")
        print(f"  kept_by_split    = {dict(stats['kept_by_split'])}")
        print(f"  dropped_by_split = {dict(stats['dropped_by_split'])}")

    split_label_vids = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for vid, label, split in rows:
        if split in split_label_vids:
            split_label_vids[split][label].append(vid)

    splits_present = {sp for _, _, sp in rows}
    has_val_split = ("val" in splits_present)

    eligible_labels = []
    for lbl in split_label_vids["test"].keys():
        if len(split_label_vids["test"][lbl]) >= test_per_class:
            eligible_labels.append(lbl)

    if len(eligible_labels) < top_k:
        raise RuntimeError(
            f"Only {len(eligible_labels)} labels satisfy test_per_class >= {test_per_class} "
            f"(need {top_k})."
        )

    if ranking == "train":
        counts = Counter({
            lbl: len(split_label_vids["train"][lbl])
            for lbl in eligible_labels
        })
    else:
        counts = Counter()
        for sp in ["train", "val", "test"]:
            for lbl in eligible_labels:
                counts[lbl] += len(split_label_vids[sp][lbl])

    top_labels = [lbl for lbl, _ in counts.most_common(top_k)]

    if len(top_labels) < top_k:
        raise RuntimeError(f"Only found {len(top_labels)} labels after filtering, cannot build top_k={top_k}.")
    top_set = set(top_labels)

    for sp in ["train", "val", "test"]:
        split_label_vids[sp] = defaultdict(
            list,
            {lbl: vids for lbl, vids in split_label_vids[sp].items() if lbl in top_set}
        )

    rng_pool = random.Random(pool_shuffle_seed)

    per_label_pools = {}
    for lbl in top_labels:
        test_pool = split_label_vids["test"][lbl][:]
        if len(test_pool) < test_per_class:
            raise RuntimeError(f"Class '{lbl}' has only {len(test_pool)} test samples (<{test_per_class}).")
        rng_pool.shuffle(test_pool)
        test_pool = test_pool[:test_per_class]

        if has_val_split:
            val_pool = split_label_vids["val"][lbl][:]
            if len(val_pool) < val_per_class:
                raise RuntimeError(f"Class '{lbl}' has only {len(val_pool)} val samples (<{val_per_class}).")
            rng_pool.shuffle(val_pool)
            val_pool = val_pool[:val_per_class]

            train_pool = split_label_vids["train"][lbl][:]
            if len(train_pool) < 1:
                raise RuntimeError(f"Class '{lbl}' has 0 train samples.")
            rng_pool.shuffle(train_pool)
        else:
            train_full = split_label_vids["train"][lbl][:]
            if len(train_full) < val_per_class + 1:
                raise RuntimeError(
                    f"Class '{lbl}' has only {len(train_full)} train samples; "
                    f"cannot carve val={val_per_class} and still keep >=1 for train."
                )
            rng_pool.shuffle(train_full)
            val_pool = train_full[:val_per_class]
            train_pool = train_full[val_per_class:]
            if len(train_pool) < 1:
                raise RuntimeError(f"Class '{lbl}' has 0 remaining train samples after carving val.")

        per_label_pools[lbl] = {
            "val": val_pool,
            "test": test_pool,
            "train": train_pool,
        }

    return top_labels, per_label_pools


def build_ordered_dicts_from_label_order(class_order, per_label_pools):
    val_dict = {}
    test_dict = {}
    train_pool_per_label = {}

    for cid, lbl in enumerate(class_order):
        if lbl not in per_label_pools:
            raise KeyError(f"Label '{lbl}' not found in per_label_pools.")
        val_dict[str(cid)] = per_label_pools[lbl]["val"][:]
        test_dict[str(cid)] = per_label_pools[lbl]["test"][:]
        train_pool_per_label[lbl] = per_label_pools[lbl]["train"][:]

    avail_by_cid = [len(train_pool_per_label[class_order[cid]]) for cid in range(len(class_order))]
    if min(avail_by_cid) == 0:
        raise RuntimeError("Some classes have 0 remaining train samples after pooling (unexpected).")

    return val_dict, test_dict, train_pool_per_label, avail_by_cid


def make_targets_lt_and_bal(avail_by_cid, gamma=10.0, mean_cap=512, seed=42, nmax_policy="cid_last"):
    """
    nmax_policy:
      - "cid_last": Nmax = avail[cid=C-1]
      - "min":      Nmax = min(avail)
      - "max":      Nmax = max(avail)
    """
    rng = random.Random(seed)
    C = len(avail_by_cid)

    if nmax_policy == "cid_last":
        Nmax = avail_by_cid[C - 1]
    elif nmax_policy == "min":
        Nmax = min(avail_by_cid)
    elif nmax_policy == "max":
        Nmax = max(avail_by_cid)
    else:
        raise ValueError("nmax_policy must be one of: cid_last/min/max")

    lt_t = lt_targets(C, Nmax, gamma=gamma)
    lt_t = [min(lt_t[i], avail_by_cid[i]) for i in range(C)]
    lt_mean = sum(lt_t) / C

    bal_per_class = int(round(lt_mean))
    bal_t = [min(bal_per_class, avail_by_cid[i]) for i in range(C)]
    bal_mean = sum(bal_t) / C

    if bal_mean > mean_cap:
        scale = mean_cap / bal_mean
        lt_scaled = [max(1, int(round(x * scale))) for x in lt_t]
        bal_scaled = [max(1, int(round(x * scale))) for x in bal_t]
        total_target = mean_cap * C
        lt_t = adjust_to_total_with_caps(lt_scaled, caps=avail_by_cid, total=total_target, rng=rng)
        bal_t = adjust_to_total_with_caps(bal_scaled, caps=avail_by_cid, total=total_target, rng=rng)
    else:
        lt_t = [max(1, min(lt_t[i], avail_by_cid[i])) for i in range(C)]
        bal_t = [max(1, min(bal_t[i], avail_by_cid[i])) for i in range(C)]

    return lt_t, bal_t


def permute_targets(targets, seed):
    rng = random.Random(seed)
    idx = list(range(len(targets)))
    rng.shuffle(idx)
    return [targets[i] for i in idx]


def print_split_stats(name, d, C):
    def counts(split):
        return [len(d[split][str(i)]) for i in range(C)]

    tr = counts("train")
    va = counts("val")
    te = counts("test")
    print(
        f"[{name}] train min/mean/max = {min(tr)}/{np.mean(tr):.2f}/{max(tr)} | "
        f"val = {min(va)}/{np.mean(va):.2f}/{max(va)} | "
        f"test = {min(te)}/{np.mean(te):.2f}/{max(te)}"
    )


def load_difficulty_csv(difficulty_csv):
    """
    CSV must contain:
      - category_name
      - difficulty

    Returns:
      diff_map[label] = float(difficulty)
    """
    diff_map = {}
    with open(difficulty_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"category_name", "difficulty"}
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"difficulty_csv must contain columns {required_cols}, "
                f"but got {reader.fieldnames}"
            )

        for row in reader:
            label = row["category_name"].strip()
            difficulty = float(row["difficulty"])
            diff_map[label] = difficulty

    return diff_map


def validate_difficulty_map(top_labels, diff_map):
    missing = [lbl for lbl in top_labels if lbl not in diff_map]
    if missing:
        preview = missing[:10]
        raise ValueError(
            f"The following labels are missing in difficulty_csv: {preview} "
            f"(total missing={len(missing)})"
        )


def save_dataset_variant(base_out_dir,
                         dataset_name,
                         class_order,
                         val_dict,
                         test_dict,
                         train_pool_per_label,
                         targets):
    train_dict = sample_train_dict_from_targets(train_pool_per_label, class_order, targets)
    all_dict = {"train": train_dict, "val": val_dict, "test": test_dict}
    save_triplet(os.path.join(base_out_dir, dataset_name), all_dict, class_order)
    print_split_stats(dataset_name, all_dict, len(class_order))


def generate_all_datasets_for_order(base_out_dir,
                                    class_order,
                                    val_dict,
                                    test_dict,
                                    train_pool_per_label,
                                    avail_by_cid,
                                    gamma,
                                    mean_cap,
                                    random_seeds,
                                    nmax_policy,
                                    balance_max_per_class=578):
    C = len(class_order)

    lt_t, bal_t = make_targets_lt_and_bal(
        avail_by_cid=avail_by_cid,
        gamma=gamma,
        mean_cap=mean_cap,
        seed=RNG_SEED,
        nmax_policy=nmax_policy,
    )

    save_dataset_variant(
        base_out_dir=base_out_dir,
        dataset_name="balance",
        class_order=class_order,
        val_dict=val_dict,
        test_dict=test_dict,
        train_pool_per_label=train_pool_per_label,
        targets=bal_t,
    )

    save_dataset_variant(
        base_out_dir=base_out_dir,
        dataset_name="ordered",
        class_order=class_order,
        val_dict=val_dict,
        test_dict=test_dict,
        train_pool_per_label=train_pool_per_label,
        targets=lt_t,
    )

    reversed_t = list(reversed(lt_t))
    save_dataset_variant(
        base_out_dir=base_out_dir,
        dataset_name="reversed",
        class_order=class_order,
        val_dict=val_dict,
        test_dict=test_dict,
        train_pool_per_label=train_pool_per_label,
        targets=reversed_t,
    )

    for idx, seed in enumerate(random_seeds, start=1):
        rt = permute_targets(lt_t, seed)
        save_dataset_variant(
            base_out_dir=base_out_dir,
            dataset_name=f"random{idx}",
            class_order=class_order,
            val_dict=val_dict,
            test_dict=test_dict,
            train_pool_per_label=train_pool_per_label,
            targets=rt,
        )

    balance_max_t = [min(balance_max_per_class, avail_by_cid[i]) for i in range(C)]
    save_dataset_variant(
        base_out_dir=base_out_dir,
        dataset_name="balance_max",
        class_order=class_order,
        val_dict=val_dict,
        test_dict=test_dict,
        train_pool_per_label=train_pool_per_label,
        targets=balance_max_t,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="./data/vggsound.csv")
    p.add_argument("--difficulty_csv", type=str, required=True,
                   help="CSV containing at least columns: category_name, difficulty")
    p.add_argument("--out_root", type=str, default="./drafts/vggsound_hard_easy_orders")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--gamma", type=float, default=100.0)
    p.add_argument("--val_per_class", type=int, default=50)
    p.add_argument("--test_per_class", type=int, default=50)
    p.add_argument("--mean_cap", type=int, default=512)
    p.add_argument("--ranking", type=str, default="all", choices=["all", "train"])

    p.add_argument("--pool_shuffle_seed", type=int, default=86)
    p.add_argument("--random_seeds", type=int, nargs=3, default=[101, 202, 303])

    p.add_argument("--feature_root", type=str, default="/mnt/data2/wpian/dataset/VGGSound",
                   help="If set, filter out csv rows whose vid is missing required features.")
    p.add_argument("--require_features", type=str, default="both",
                   choices=["visual", "audio", "both", "either"])

    p.add_argument("--nmax_policy", type=str, default="min",
                   choices=["cid_last", "min", "max"])
    p.add_argument("--balance_max_per_class", type=int, default=578)

    args = p.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # 1) Build label pools without deciding final class order yet
    top_labels, per_label_pools = build_base_pools(
        csv_path=args.csv,
        top_k=args.top_k,
        ranking=args.ranking,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        pool_shuffle_seed=args.pool_shuffle_seed,
        feature_root=args.feature_root,
        require_features=args.require_features,
    )

    # 2) Load difficulty and derive two deterministic class orders
    diff_map = load_difficulty_csv(args.difficulty_csv)
    validate_difficulty_map(top_labels, diff_map)

    # smaller difficulty = easier
    class_order_hard2easy = sorted(top_labels, key=lambda x: diff_map[x], reverse=True)
    class_order_easy2hard = sorted(top_labels, key=lambda x: diff_map[x])

    print("\n[Class Order Summary]")
    print("data_hard2easy: hardest -> easiest")
    print("  first 10:", class_order_hard2easy[:10])
    print("  last  10:", class_order_hard2easy[-10:])
    print("data_easy2hard: easiest -> hardest")
    print("  first 10:", class_order_easy2hard[:10])
    print("  last  10:", class_order_easy2hard[-10:])

    # 3) Build dicts for hard->easy order
    val_h2e, test_h2e, train_pool_h2e, avail_h2e = build_ordered_dicts_from_label_order(
        class_order_hard2easy,
        per_label_pools,
    )

    # 4) Build dicts for easy->hard order
    val_e2h, test_e2h, train_pool_e2h, avail_e2h = build_ordered_dicts_from_label_order(
        class_order_easy2hard,
        per_label_pools,
    )

    # 5) Under each order, generate the full family of datasets
    out_h2e = os.path.join(args.out_root, "data_hard2easy")
    out_e2h = os.path.join(args.out_root, "data_easy2hard")

    print("\n[Generating data_hard2easy]")
    generate_all_datasets_for_order(
        base_out_dir=out_h2e,
        class_order=class_order_hard2easy,
        val_dict=val_h2e,
        test_dict=test_h2e,
        train_pool_per_label=train_pool_h2e,
        avail_by_cid=avail_h2e,
        gamma=args.gamma,
        mean_cap=args.mean_cap,
        random_seeds=args.random_seeds,
        nmax_policy=args.nmax_policy,
        balance_max_per_class=args.balance_max_per_class,
    )

    print("\n[Generating data_easy2hard]")
    generate_all_datasets_for_order(
        base_out_dir=out_e2h,
        class_order=class_order_easy2hard,
        val_dict=val_e2h,
        test_dict=test_e2h,
        train_pool_per_label=train_pool_e2h,
        avail_by_cid=avail_e2h,
        gamma=args.gamma,
        mean_cap=args.mean_cap,
        random_seeds=args.random_seeds,
        nmax_policy=args.nmax_policy,
        balance_max_per_class=args.balance_max_per_class,
    )

    print("\nAll datasets saved under:", args.out_root)
    print("Subfolders:")
    print("  - data_hard2easy/{balance, ordered, reversed, random1, random2, random3, balance_max}")
    print("  - data_easy2hard/{balance, ordered, reversed, random1, random2, random3, balance_max}")
