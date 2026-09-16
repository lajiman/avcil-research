# make_vggsound_lt_bal_dicts_with_fixed_val_test.py
import os
import math
import random
from collections import defaultdict, Counter
import numpy as np
import csv

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

    # build mutable lists of indices
    def rebuild_inc_dec():
        inc = [i for i in range(len(targets)) if targets[i] < caps[i]]
        dec = [i for i in range(len(targets)) if targets[i] > 1]
        rng.shuffle(inc); rng.shuffle(dec)
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
    # label -> class_id
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
                # sanity: one vid should map to exactly one label in this split
                if vid in split_map and split_map[vid] != label:
                    raise RuntimeError(
                        f"[Inconsistent] split={split} vid={vid} mapped to both "
                        f"'{split_map[vid]}' and '{label}'"
                    )
                split_map[vid] = label
    return out


def remap_class_ids(all_classId_vid_dict, old_class_order, new_class_order):
    """
    Reindex class IDs according to new_class_order.
    - all_classId_vid_dict uses class_id keys as strings ("0","1",...)
    - old_class_order[cid] gives label for that old cid
    - new_class_order defines the desired cid->label mapping

    Return a new dict with same samples per label, but class ids reassigned.
    """
    label_to_oldcid = {label: i for i, label in enumerate(old_class_order)}
    label_to_newcid = {label: i for i, label in enumerate(new_class_order)}

    out = {"train": {}, "val": {}, "test": {}}
    for split in ["train", "val", "test"]:
        out[split] = {str(i): [] for i in range(len(new_class_order))}
        for label in new_class_order:
            oldcid = label_to_oldcid[label]
            newcid = label_to_newcid[label]
            vids = all_classId_vid_dict[split][str(oldcid)]
            out[split][str(newcid)] = vids
    return out


def save_triplet(out_dir, all_classId_vid_dict, class_order):
    """
    Save:
      all_classId_vid_dict.npy
      category_encode_dict.npy
      all_id_category_dict.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    category_encode_dict = make_category_encode_dict(class_order)
    all_id_category_dict = make_all_id_category_dict_from_classId_vid_dict(all_classId_vid_dict, class_order)

    np.save(os.path.join(out_dir, "all_classId_vid_dict.npy"), all_classId_vid_dict, allow_pickle=True)
    np.save(os.path.join(out_dir, "category_encode_dict.npy"), category_encode_dict, allow_pickle=True)
    np.save(os.path.join(out_dir, "all_id_category_dict.npy"), all_id_category_dict, allow_pickle=True)


def build_lt_bal(csv_path: str,
                 out_dir: str,
                 top_k=100,
                 gamma=10.0,
                 val_per_class=50,
                 test_per_class=50,
                 mean_cap=512,
                 ranking="all"):  # "all" or "train"
    """
    Build two datasets (LT and BAL) from vggsound.csv.

    Output npy dict structure (JSON after conversion):
      {
        "train": {"0":[vid,...], "1":[...], ...},
        "val":   {"0":[...], ...},
        "test":  {"0":[...], ...}
      }

    Protocol:
      - Select Top-K classes by count (ranking="all" uses train+test(+val if exists), ranking="train" uses train only).
      - Per class: reserve test_per_class samples from original TEST split.
      - Per class: reserve val_per_class samples from original VAL split if it exists;
                   otherwise carve val_per_class from original TRAIN split.
      - Train pool = remaining TRAIN samples (after carving val if needed).
      - Construct LT train distribution with gamma (head->tail ordered by TRAIN counts).
      - Construct BAL train distribution with mean = LT mean.
      - If BAL mean > mean_cap, scale BOTH LT and BAL to mean_cap (approximately, with total adjusted).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(RNG_SEED)

    rows = parse_vggsound_csv(csv_path)

    # split -> label -> vids
    split_label_vids = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }
    for vid, label, split in rows:
        if split in split_label_vids:
            split_label_vids[split][label].append(vid)

    # Determine whether CSV has a val split at all
    splits_present = {sp for _, _, sp in rows}
    has_val_split = ("val" in splits_present)

    # Determine top_k labels by counts
    if ranking == "train":
        counts = Counter({lbl: len(v) for lbl, v in split_label_vids["train"].items()})
    else:
        # ranking by all splits combined (train+test +val if exists)
        all_counts = Counter()
        for sp in ["train", "val", "test"]:
            all_counts.update({lbl: len(v) for lbl, v in split_label_vids[sp].items()})
        counts = all_counts

    top_labels = [lbl for lbl, _ in counts.most_common(top_k)]
    if len(top_labels) < top_k:
        raise RuntimeError(f"Only found {len(top_labels)} labels, cannot build top_k={top_k}.")
    top_set = set(top_labels)

    # filter to top_set
    for sp in ["train", "val", "test"]:
        split_label_vids[sp] = defaultdict(
            list, {lbl: vids for lbl, vids in split_label_vids[sp].items() if lbl in top_set}
        )

    # class order: head->tail by TRAIN count (ordered-LT)
    train_counts = {lbl: len(split_label_vids["train"][lbl]) for lbl in top_labels}
    class_order = sorted(top_labels, key=lambda x: train_counts.get(x, 0), reverse=True)
    C = len(class_order)

    # --- 1) reserve fixed val/test per class, and build remaining train pool per class ---
    val_dict = {}
    test_dict = {}
    train_pool_per_class = {}

    for cid, lbl in enumerate(class_order):
        # TEST must come from original test split
        test_pool = split_label_vids["test"][lbl][:]
        if len(test_pool) < test_per_class:
            raise RuntimeError(
                f"Class '{lbl}' has only {len(test_pool)} test samples (<{test_per_class})."
            )
        rng.shuffle(test_pool)
        test_pick = test_pool[:test_per_class]
        test_dict[str(cid)] = test_pick

        if has_val_split:
            # VAL comes from original val split
            val_pool = split_label_vids["val"][lbl][:]
            if len(val_pool) < val_per_class:
                raise RuntimeError(
                    f"Class '{lbl}' has only {len(val_pool)} val samples (<{val_per_class})."
                )
            rng.shuffle(val_pool)
            val_pick = val_pool[:val_per_class]
            val_dict[str(cid)] = val_pick

            # TRAIN pool uses original train split (no carving needed)
            train_pool = split_label_vids["train"][lbl][:]
            if len(train_pool) < 1:
                raise RuntimeError(f"Class '{lbl}' has 0 train samples.")
            rng.shuffle(train_pool)
            train_pool_per_class[lbl] = train_pool
        else:
            # No official val split => carve val from TRAIN split
            train_pool_full = split_label_vids["train"][lbl][:]
            if len(train_pool_full) < val_per_class + 1:
                raise RuntimeError(
                    f"Class '{lbl}' has only {len(train_pool_full)} train samples; "
                    f"cannot carve val={val_per_class} and still keep >=1 for train."
                )
            rng.shuffle(train_pool_full)
            val_pick = train_pool_full[:val_per_class]
            val_dict[str(cid)] = val_pick

            # Remaining for TRAIN (LT/BAL)
            train_pool_rem = train_pool_full[val_per_class:]
            if len(train_pool_rem) < 1:
                raise RuntimeError(
                    f"Class '{lbl}' has 0 remaining train samples after carving val."
                )
            train_pool_per_class[lbl] = train_pool_rem

    # --- 2) availability for LT/BAL train ---
    avail = [len(train_pool_per_class[lbl]) for lbl in class_order]
    if min(avail) == 0:
        zeros = [class_order[i] for i, a in enumerate(avail) if a == 0]
        raise RuntimeError(f"Some classes have 0 remaining train samples: {zeros[:5]}...")

    # --- 3) LT targets on TRAIN remaining ---
    Nmax = max(avail)
    lt_t = lt_targets(C, Nmax, gamma=gamma)
    lt_t = [min(lt_t[i], avail[i]) for i in range(C)]
    lt_mean = sum(lt_t) / C

    # --- 4) BAL targets: mean = LT mean ---
    bal_per_class = int(round(lt_mean))
    bal_t = [min(bal_per_class, avail[i]) for i in range(C)]
    bal_mean = sum(bal_t) / C

    # --- 5) scale if balanced mean > mean_cap ---
    if bal_mean > mean_cap:
        scale = mean_cap / bal_mean
        lt_scaled = [max(1, int(round(x * scale))) for x in lt_t]
        bal_scaled = [max(1, int(round(x * scale))) for x in bal_t]

        total_target = mean_cap * C
        lt_t = adjust_to_total_with_caps(lt_scaled, caps=avail, total=total_target, rng=rng)
        bal_t = adjust_to_total_with_caps(bal_scaled, caps=avail, total=total_target, rng=rng)
    else:
        lt_t = [max(1, min(lt_t[i], avail[i])) for i in range(C)]
        bal_t = [max(1, min(bal_t[i], avail[i])) for i in range(C)]

    # --- 6) sample TRAIN vids (without replacement) ---
    lt_train_dict = {}
    bal_train_dict = {}
    for cid, lbl in enumerate(class_order):
        vids = train_pool_per_class[lbl]
        # vids already shuffled
        lt_k = min(lt_t[cid], len(vids))
        bal_k = min(bal_t[cid], len(vids))
        lt_train_dict[str(cid)] = vids[:lt_k]
        bal_train_dict[str(cid)] = vids[:bal_k]

    all_classId_vid_dict_lt = {"train": lt_train_dict, "val": val_dict, "test": test_dict}
    all_classId_vid_dict_bal = {"train": bal_train_dict, "val": val_dict, "test": test_dict}

    # np.save(os.path.join(out_dir, "all_classId_vid_dict_lt.npy"), all_classId_vid_dict_lt, allow_pickle=True)
    # np.save(os.path.join(out_dir, "all_classId_vid_dict_bal.npy"), all_classId_vid_dict_bal, allow_pickle=True)

    # --- 7) build and save category_encode_dict + all_id_category_dict for LT/BAL ---
    category_encode_dict = make_category_encode_dict(class_order)

    all_id_category_dict_lt = make_all_id_category_dict_from_classId_vid_dict(
        all_classId_vid_dict_lt, class_order
    )
    all_id_category_dict_bal = make_all_id_category_dict_from_classId_vid_dict(
        all_classId_vid_dict_bal, class_order
    )

    # np.save(os.path.join(out_dir, "category_encode_dict_lt.npy"), category_encode_dict, allow_pickle=True)
    # np.save(os.path.join(out_dir, "category_encode_dict_bal.npy"), category_encode_dict, allow_pickle=True)

    # np.save(os.path.join(out_dir, "all_id_category_dict_lt.npy"), all_id_category_dict_lt, allow_pickle=True)
    # np.save(os.path.join(out_dir, "all_id_category_dict_bal.npy"), all_id_category_dict_bal, allow_pickle=True)

    return all_classId_vid_dict_bal, all_classId_vid_dict_lt, class_order

    # --- stats print ---
    def split_stats(d, split):
        counts = [len(d[split][str(i)]) for i in range(C)]
        return min(counts), max(counts), float(np.mean(counts))

    vmin, vmax, vmean = split_stats(all_classId_vid_dict_lt, "val")
    tmin, tmax, tmean = split_stats(all_classId_vid_dict_lt, "test")
    lt_min, lt_max, lt_mean2 = split_stats(all_classId_vid_dict_lt, "train")
    bal_min, bal_max, bal_mean2 = split_stats(all_classId_vid_dict_bal, "train")

    print("Done.")
    print(f"Top-{top_k} classes (ordered by TRAIN count head->tail). gamma={gamma}, ranking={ranking}")
    print(f"VAL fixed:  per-class={val_per_class} (min={vmin}, max={vmax}, mean={vmean:.2f})")
    print(f"TEST fixed: per-class={test_per_class} (min={tmin}, max={tmax}, mean={tmean:.2f})")
    print(f"LT TRAIN:   min={lt_min}, max={lt_max}, mean={lt_mean2:.2f}")
    print(f"BAL TRAIN:  min={bal_min}, max={bal_max}, mean={bal_mean2:.2f}")
    print(f"Saved: {out_dir}/all_classId_vid_dict_lt.npy and ..._bal.npy")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="./data/vggsound.csv")
    p.add_argument("--out_root", type=str, default="./drafts/vggsound_6sets")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--val_per_class", type=int, default=50)
    p.add_argument("--test_per_class", type=int, default=50)
    p.add_argument("--mean_cap", type=int, default=512)
    p.add_argument("--ranking", type=str, default="all", choices=["all", "train"])
    p.add_argument("--random_seeds", type=int, nargs=3, default=[101, 202, 303])
    args = p.parse_args()

    # 1) build base datasets (ordered LT + balance), and ordered class order
    bal_dict, ordered_lt_dict, ordered_class_order = build_lt_bal(
        csv_path=args.csv,
        out_dir=args.out_root,  # out_dir here is only used for os.makedirs / prints; we won't save inside build_lt_bal
        top_k=args.top_k,
        gamma=args.gamma,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        mean_cap=args.mean_cap,
        ranking=args.ranking,
    )

    C = len(ordered_class_order)

    # 2) Save BALANCE (class order = ordered_class_order, but BAL train distribution)
    save_triplet(os.path.join(args.out_root, "balance"), bal_dict, ordered_class_order)

    # 3) Save ORDERED LT (head->tail first)
    save_triplet(os.path.join(args.out_root, "ordered"), ordered_lt_dict, ordered_class_order)

    # 4) Save REVERSED LT (tail->head first) by reindexing class ids
    reversed_class_order = list(reversed(ordered_class_order))
    reversed_lt_dict = remap_class_ids(ordered_lt_dict, ordered_class_order, reversed_class_order)
    save_triplet(os.path.join(args.out_root, "reversed"), reversed_lt_dict, reversed_class_order)

    # 5) Save RANDOM1/2/3 (LT + shuffled class order)
    for idx, seed in enumerate(args.random_seeds, start=1):
        rng = random.Random(seed)
        random_class_order = ordered_class_order[:]
        rng.shuffle(random_class_order)

        random_lt_dict = remap_class_ids(ordered_lt_dict, ordered_class_order, random_class_order)
        save_triplet(os.path.join(args.out_root, f"random{idx}"), random_lt_dict, random_class_order)

    print("\nAll 6 datasets saved under:", args.out_root)
    print("Subfolders: balance, ordered, reversed, random1, random2, random3")
