import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _auto_perplexity(n: int, default: int = 30) -> int:
    if n <= 2:
        return 1
    p = max(5, (n - 1) // 3)
    return int(min(default, p, n - 1))


def _sample_cache_path(out_root: str, step: int):
    return os.path.join(out_root, f"step_{step:02d}", "sample_indices.json")


def _build_fixed_sample_indices(labels: np.ndarray, max_points_per_class: int, seed: int):
    rng = np.random.RandomState(seed)
    chosen = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > max_points_per_class:
            idx = rng.choice(idx, size=max_points_per_class, replace=False)
        chosen.append(np.sort(idx))
    if len(chosen) == 0:
        return np.array([], dtype=np.int64)
    chosen = np.concatenate(chosen, axis=0)
    chosen.sort()
    return chosen.astype(np.int64)


def _load_or_create_fixed_sample_indices(
    labels: np.ndarray,
    out_root: str,
    step: int,
    max_points_per_class: int,
    seed: int
):
    step_dir = os.path.join(out_root, f"step_{step:02d}")
    os.makedirs(step_dir, exist_ok=True)
    cache_path = _sample_cache_path(out_root, step)

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            arr = json.load(f)
        return np.array(arr, dtype=np.int64)

    chosen = _build_fixed_sample_indices(labels, max_points_per_class, seed)
    with open(cache_path, "w") as f:
        json.dump(chosen.tolist(), f)
    return chosen


@torch.no_grad()
def extract_features_labels_preds(
    model,
    dataset,
    args,
    feature_type: str = "joint_concat",
):
    """
    提取 all-seen dataset 上的:
      - features
      - y_true
      - y_pred

    feature_type:
      - audio
      - visual
      - joint_mean
      - joint_concat
      - logits
    """
    assert feature_type in ["audio", "visual", "joint_mean", "joint_concat", "logits"]

    loader = DataLoader(
        dataset,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    feats_all = []
    labels_all = []
    preds_all = []

    device = next(model.parameters()).device
    model.eval()

    for data, labels in loader:
        labels = labels.long()
        visual = data[0].to(device, non_blocking=True)
        audio = data[1].to(device, non_blocking=True)

        if feature_type == "logits":
            logits = model(visual=visual, audio=audio)
            feat = logits.detach().float().cpu()
            pred = logits.argmax(dim=1).detach().cpu().long()
        else:
            out = model(visual=visual, audio=audio, out_feature_before_fusion=True)
            if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                raise RuntimeError(
                    "Model forward did not return (logits, audio_feature, visual_feature). "
                    "Please ensure model(..., out_feature_before_fusion=True) returns them."
                )

            logits, audio_f, visual_f = out[0], out[1], out[2]
            pred = logits.argmax(dim=1).detach().cpu().long()

            audio_f = audio_f.detach().float().cpu()
            visual_f = visual_f.detach().float().cpu()

            if feature_type == "audio":
                feat = audio_f
            elif feature_type == "visual":
                feat = visual_f
            elif feature_type == "joint_mean":
                a = torch.nn.functional.normalize(audio_f, dim=1)
                v = torch.nn.functional.normalize(visual_f, dim=1)
                feat = (a + v) * 0.5
            elif feature_type == "joint_concat":
                a = torch.nn.functional.normalize(audio_f, dim=1)
                v = torch.nn.functional.normalize(visual_f, dim=1)
                feat = torch.cat([a, v], dim=1)
            else:
                raise ValueError(feature_type)

        feats_all.append(feat)
        labels_all.append(labels.cpu())
        preds_all.append(pred.cpu())

    feats_all = torch.cat(feats_all, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    preds_all = torch.cat(preds_all, dim=0).numpy()

    return feats_all, labels_all, preds_all


def fit_tsne(features: np.ndarray, seed: int):
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for TSNE. Please install scikit-learn.")

    if features.shape[0] < 5:
        raise ValueError(f"Too few samples for TSNE: n={features.shape[0]}")

    perplexity = _auto_perplexity(features.shape[0], default=30)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    z = tsne.fit_transform(features)
    return z, perplexity


def plot_global_highlight_task(
    z: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step: int,
    task_id: int,
    class_num_per_step: int,
    dataset_name: str,
    out_path: str,
):
    """
    global 图：
      - all seen classes 都在同一张图上
      - 当前 task 高亮
      - 其他 task 灰色
      - 当前 task 内错分样本为 x
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lo = task_id * class_num_per_step
    hi = (task_id + 1) * class_num_per_step
    seen_hi = (step + 1) * class_num_per_step

    seen_mask = (y_true >= 0) & (y_true < seen_hi)
    focus_mask = (y_true >= lo) & (y_true < hi)
    bg_mask = seen_mask & (~focus_mask)

    err_mask = (y_true != y_pred)
    cor_mask = (y_true == y_pred)

    plt.figure(figsize=(7, 6))

    if bg_mask.sum() > 0:
        plt.scatter(
            z[bg_mask, 0], z[bg_mask, 1],
            s=6, alpha=0.20, c="lightgray", marker="o", linewidths=0
        )

    cmap = plt.cm.get_cmap("tab10", class_num_per_step)

    for c in range(lo, hi):
        cls_mask = (y_true == c)
        cls_cor = cls_mask & cor_mask
        cls_err = cls_mask & err_mask
        local_c = c - lo

        if cls_cor.sum() > 0:
            plt.scatter(
                z[cls_cor, 0], z[cls_cor, 1],
                s=16, alpha=0.90, color=cmap(local_c), marker="o",
                label=f"class {c}"
            )

        if cls_err.sum() > 0:
            plt.scatter(
                z[cls_err, 0], z[cls_err, 1],
                s=28, alpha=0.95, color=cmap(local_c), marker="x"
            )

    n_focus = int(focus_mask.sum())
    n_err = int((focus_mask & err_mask).sum())

    plt.title(
        f"[{dataset_name}] step={step:02d}, task={task_id:02d} GLOBAL\n"
        f"highlight task classes {lo}-{hi-1}, n={n_focus}, errors={n_err}"
    )
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=8, ncol=2, frameon=False, markerscale=1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_local_task_only(
    features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step: int,
    task_id: int,
    class_num_per_step: int,
    dataset_name: str,
    out_path: str,
    seed: int,
):
    """
    local 图：
      - 只保留当前 task 的类
      - 对该 task 自己的数据单独 fit 一个 t-SNE
      - 错分样本为 x
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lo = task_id * class_num_per_step
    hi = (task_id + 1) * class_num_per_step

    mask = (y_true >= lo) & (y_true < hi)
    X = features[mask]
    y = y_true[mask]
    p = y_pred[mask]

    if X.shape[0] < 5:
        with open(out_path + ".txt", "w") as f:
            f.write(f"Too few samples for local TSNE: task={task_id}, n={X.shape[0]}\n")
        return

    z_local, perplexity = fit_tsne(X, seed=seed)

    err_mask = (y != p)
    cor_mask = (y == p)

    plt.figure(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab10", class_num_per_step)

    for c in range(lo, hi):
        cls_mask = (y == c)
        cls_cor = cls_mask & cor_mask
        cls_err = cls_mask & err_mask
        local_c = c - lo

        if cls_cor.sum() > 0:
            plt.scatter(
                z_local[cls_cor, 0], z_local[cls_cor, 1],
                s=16, alpha=0.90, color=cmap(local_c), marker="o",
                label=f"class {c}"
            )

        if cls_err.sum() > 0:
            plt.scatter(
                z_local[cls_err, 0], z_local[cls_err, 1],
                s=28, alpha=0.95, color=cmap(local_c), marker="x"
            )

    n_task = int(mask.sum())
    n_err = int(err_mask.sum())

    plt.title(
        f"[{dataset_name}] step={step:02d}, task={task_id:02d} LOCAL\n"
        f"task-only classes {lo}-{hi-1}, n={n_task}, errors={n_err}, perp={perplexity}"
    )
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=8, ncol=2, frameon=False, markerscale=1.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_tsne_plots_for_step(
    args,
    step: int,
    test_set,
    ckpt_path: str,
    out_root: str,
    feature_type: str = "joint_concat",
    max_points_per_class: int = 200,
):
    """
    保持原有调用方式不变。

    每个 step:
      1. 提取 all-seen feature / true / pred
      2. 固定 sample ids
      3. 在 all-seen samples 上 fit 一次 global TSNE
      4. 对每个 task 输出两张图:
         - task_xx_global.png
         - task_xx_local.png
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(ckpt_path, map_location="cpu")
    model = model.to(device)
    model.eval()

    feats_all, labels_all, preds_all = extract_features_labels_preds(
        model=model,
        dataset=test_set,
        args=args,
        feature_type=feature_type,
    )

    chosen = _load_or_create_fixed_sample_indices(
        labels=labels_all,
        out_root=out_root,
        step=step,
        max_points_per_class=max_points_per_class,
        seed=args.seed + step,
    )

    feats = feats_all[chosen]
    labels = labels_all[chosen]
    preds = preds_all[chosen]

    # all-seen 的 global t-SNE：只 fit 一次
    z_global, global_perplexity = fit_tsne(feats, seed=args.seed + step)

    step_dir = os.path.join(out_root, f"step_{step:02d}")
    os.makedirs(step_dir, exist_ok=True)

    meta = {
        "step": step,
        "feature_type": feature_type,
        "num_points": int(len(labels)),
        "global_perplexity": int(global_perplexity),
        "max_points_per_class": int(max_points_per_class),
    }
    with open(os.path.join(step_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for task_id in range(step + 1):
        global_path = os.path.join(step_dir, f"task_{task_id:02d}_global_{feature_type}.png")
        local_path = os.path.join(step_dir, f"task_{task_id:02d}_local_{feature_type}.png")

        plot_global_highlight_task(
            z=z_global,
            y_true=labels,
            y_pred=preds,
            step=step,
            task_id=task_id,
            class_num_per_step=args.class_num_per_step,
            dataset_name=args.dataset,
            out_path=global_path,
        )

        plot_local_task_only(
            features=feats,
            y_true=labels,
            y_pred=preds,
            step=step,
            task_id=task_id,
            class_num_per_step=args.class_num_per_step,
            dataset_name=args.dataset,
            out_path=local_path,
            seed=args.seed + step + task_id,
        )