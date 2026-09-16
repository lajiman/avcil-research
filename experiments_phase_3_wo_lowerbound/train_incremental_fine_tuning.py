import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_incremental_fine_tuning import IcreLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from model.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def CE_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    return loss

def _normalize_category_name(name):
    return str(name).strip().lower()


def load_category_encode_dict(args):
    """
    Load current dataset's category_encode_dict.npy.

    This dict defines the actual class-id encoding used by the current dataset.
    For example, in easy2hard:
        "scuba diving" -> 0
    while in the original difficulty CSV, "scuba diving" may have class_id 37.

    Therefore difficulty must be aligned by category_name, not by the old CSV class_id.
    """
    if args.meta_root is None:
        raise ValueError(
            "--meta_root must be provided when using difficulty weights, "
            "because category_encode_dict.npy is needed to align difficulty to the current class order."
        )

    category_encode_path = os.path.join(args.meta_root, "category_encode_dict.npy")

    if not os.path.exists(category_encode_path):
        raise FileNotFoundError(
            f"category_encode_dict.npy not found at: {category_encode_path}"
        )

    category_encode_dict = np.load(
        category_encode_path,
        allow_pickle=True
    ).item()

    return category_encode_dict


def load_difficulty_weights(args, step_out_class_num):
    """
    Load class-level difficulty from CSV and convert it into class weights.

    Alignment:
        difficulty_csv["category_name"]
            -> current category_encode_dict.npy
            -> current class_id
            -> weights[current_class_id]

    Supported weight modes:

        legacy:
            Use the old behavior controlled by args.difficulty_smooth.
            If difficulty_smooth=True:
                w = 1 + lambda * D
            else:
                w = max(D, eps)

        raw:
            w = max(D, eps)

        smooth:
            w = 1 + lambda * D

        power:
            w = (D + eps) ** power

        exp:
            w = exp(alpha * D)

    After weight construction, optional mean normalization is applied:
        w = w / mean(w)
    """
    if not args.use_difficulty_weight:
        return None

    if args.difficulty_csv is None or len(args.difficulty_csv) == 0:
        raise ValueError("--difficulty_csv must be provided when --use_difficulty_weight=True")

    if args.difficulty_raw_col not in {"raw_difficulty_recall", "raw_difficulty_f1"}:
        raise ValueError(
            "--difficulty_raw_col must be either 'raw_difficulty_recall' or 'raw_difficulty_f1'"
        )

    category_encode_dict = load_category_encode_dict(args)

    name_to_current_id = {
        str(name): int(cid)
        for name, cid in category_encode_dict.items()
    }

    normalized_name_to_current_id = {}
    for name, cid in category_encode_dict.items():
        norm_name = _normalize_category_name(name)
        if norm_name in normalized_name_to_current_id:
            raise ValueError(
                f"Duplicate normalized category name found in category_encode_dict: {norm_name}"
            )
        normalized_name_to_current_id[norm_name] = int(cid)

    # Store raw difficulty first.
    # Weight is computed after all difficulty values are loaded.
    difficulty_values = np.zeros(step_out_class_num, dtype=np.float32)

    found_visible_ids = set()
    matched_by_exact = 0
    matched_by_normalized = 0

    with open(args.difficulty_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"category_name", args.difficulty_raw_col}
        missing_cols = required_cols - set(reader.fieldnames or [])

        if missing_cols:
            raise ValueError(
                f"Missing columns in difficulty_csv: {sorted(missing_cols)}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            category_name = str(row["category_name"])

            if category_name in name_to_current_id:
                current_class_id = name_to_current_id[category_name]
                matched_by_exact += 1
            else:
                norm_name = _normalize_category_name(category_name)
                if norm_name not in normalized_name_to_current_id:
                    continue
                current_class_id = normalized_name_to_current_id[norm_name]
                matched_by_normalized += 1

            if current_class_id < 0 or current_class_id >= step_out_class_num:
                continue

            difficulty = float(row[args.difficulty_raw_col])
            difficulty_values[current_class_id] = difficulty
            found_visible_ids.add(current_class_id)

    missing_ids = [
        c for c in range(step_out_class_num)
        if c not in found_visible_ids
    ]

    if len(missing_ids) > 0:
        id_to_name = {
            int(cid): str(name)
            for name, cid in category_encode_dict.items()
        }
        missing_names = [
            id_to_name.get(c, f"<unknown:{c}>")
            for c in missing_ids[:20]
        ]

        raise ValueError(
            "difficulty_csv does not provide difficulty for some visible classes.\n"
            f"Missing class ids: {missing_ids[:20]}"
            + (" ..." if len(missing_ids) > 20 else "")
            + "\n"
            f"Missing class names: {missing_names}"
            + (" ..." if len(missing_ids) > 20 else "")
        )

    D = difficulty_values.astype(np.float32)

    # Convert raw difficulty D into class weights.
    if args.difficulty_weight_mode == "legacy":
        if args.difficulty_smooth:
            weights = 1.0 + args.difficulty_lambda * D
        else:
            weights = np.maximum(D, args.difficulty_eps)

    elif args.difficulty_weight_mode == "raw":
        weights = np.maximum(D, args.difficulty_eps)

    elif args.difficulty_weight_mode == "smooth":
        weights = 1.0 + args.difficulty_lambda * D

    elif args.difficulty_weight_mode == "power":
        weights = np.power(D + args.difficulty_eps, args.difficulty_power)

    elif args.difficulty_weight_mode == "exp":
        weights = np.exp(args.difficulty_exp_alpha * D)

    else:
        raise ValueError(f"Unknown difficulty_weight_mode: {args.difficulty_weight_mode}")

    weights = weights.astype(np.float32)

    if args.difficulty_mean_norm:
        mean_weight = weights.mean()
        if mean_weight <= 0:
            raise ValueError("Mean difficulty weight must be positive.")
        weights = weights / mean_weight

    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("[Difficulty Weight]")
    print(f"  difficulty csv: {args.difficulty_csv}")
    print(f"  current meta_root: {args.meta_root}")
    print(f"  align key: category_name -> current category_encode_dict.npy")
    print(f"  source column: {args.difficulty_raw_col}")
    print(f"  weight mode: {args.difficulty_weight_mode}")
    print(f"  legacy smooth: {args.difficulty_smooth}")
    print(f"  lambda: {args.difficulty_lambda}")
    print(f"  power: {args.difficulty_power}")
    print(f"  exp alpha: {args.difficulty_exp_alpha}")
    print(f"  mean_norm: {args.difficulty_mean_norm}")
    print(f"  matched by exact name: {matched_by_exact}")
    print(f"  matched by normalized name: {matched_by_normalized}")
    print(
        "  raw difficulty values: "
        f"min={D.min():.6f}, "
        f"max={D.max():.6f}, "
        f"mean={D.mean():.6f}"
    )
    print(
        "  visible class weights: "
        f"min={weights.min().item():.6f}, "
        f"max={weights.max().item():.6f}, "
        f"mean={weights.mean().item():.6f}"
    )

    id_to_name = {
        int(cid): str(name)
        for name, cid in category_encode_dict.items()
    }

    print("  first visible class weights:")
    for cid in range(min(10, step_out_class_num)):
        print(
            f"    class_id={cid:03d}, "
            f"name={id_to_name.get(cid, '<unknown>')}, "
            f"D={D[cid]:.6f}, "
            f"weight={weights[cid].item():.6f}"
        )

    return weights


def difficulty_weighted_CE_loss(num_classes, logits, label, class_weights=None):
    """
    CE loss with optional class-level difficulty weights.

    label must be global class id in [0, num_classes).
    class_weights must have shape [num_classes] and be indexed by global class id.
    """
    if class_weights is None:
        return CE_loss(num_classes, logits, label)

    if class_weights.shape[0] != num_classes:
        raise ValueError(
            f"class_weights length ({class_weights.shape[0]}) must match num_classes ({num_classes})."
        )

    ce_per_sample = F.cross_entropy(logits, label, reduction="none")
    sample_weights = class_weights[label]
    loss = (ce_per_sample * sample_weights).mean()

    return loss

def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups: 
            param_group['lr'] = new_lr

def save_model(model, save_path):
    if torch.cuda.device_count() > 1:
        torch.save(model.module, save_path)
    else:
        torch.save(model, save_path)

def train(args, step, train_data_set, val_data_set):
    train_loader = DataLoader(train_data_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False, shuffle=True)
    val_loader = DataLoader(val_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)
    
    step_out_class_num = (step + 1) * args.class_num_per_step

    if step == 0 or args.upper_bound:
        model = IncreAudioVisualNet(args, step_out_class_num)
    else:
        model = torch.load('./save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step-1, args.modality))
        model.incremental_classifier(step_out_class_num)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_weights = load_difficulty_weights(args, step_out_class_num)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0
    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()
        for data, labels in tqdm(train_loader):
            labels = labels.to(device)
            if args.modality == 'visual':
                visual = data
                visual = visual.to(device)
                out = model(visual=visual)
            elif args.modality == 'audio':
                audio = data
                audio = audio.to(device)
                out = model(audio=audio)
            else:
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                out = model(visual=visual, audio=audio)

            loss = difficulty_weighted_CE_loss(
                step_out_class_num,
                out,
                labels,
                class_weights
            )     
             
            if epoch == 0 and num_steps == 0 and class_weights is not None:
                with torch.no_grad():
                    batch_weights = class_weights[labels]
                    print("[Batch Weight Debug]")
                    print(f"  labels[:20]: {labels[:20].detach().cpu().numpy().tolist()}")
                    print(f"  weights[:20]: {batch_weights[:20].detach().cpu().numpy().tolist()}")
                    print(
                        f"  batch weight min={batch_weights.min().item():.6f}, "
                        f"max={batch_weights.max().item():.6f}, "
                        f"mean={batch_weights.mean().item():.6f}"
                    )
      
            model.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                # val_labels = val_labels.to(device)
                if args.modality == 'visual':
                    val_visual = val_data
                    val_visual = val_visual.to(device)
                    val_out_logits = model(visual=val_visual)
                elif args.modality == 'audio':
                    val_audio = val_data
                    val_audio = val_audio.to(device)
                    val_out_logits = model(audio=val_audio)
                else:
                    val_visual = val_data[0]
                    val_audio = val_data[1]
                    val_visual = val_visual.to(device)
                    val_audio = val_audio.to(device)
                    val_out_logits = model(visual=val_visual, audio=val_audio)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        print('Epoch:{} val_res:{:.6f} '.format(epoch, val_top1), flush=True)

        if val_top1 > best_val_res:
            best_val_res = val_top1
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            model_save_path = './save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step, args.modality)

            save_model(model, model_save_path)
        if (epoch + 1) % 10 == 0:
            latest_model_save_path = './save/{}/{}/step_{}_latest_{}_model.pkl'.format(args.dataset, args.modality, step, args.modality)
            save_model(model, latest_model_save_path)
            print('Saving latest model at Epoch {}'.format(epoch), flush=True)
        
        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/{}_train_loss_step_{}.png'.format(args.dataset, args.modality, args.modality, step))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig('./save/fig/{}/{}/{}_val_acc_step_{}.png'.format(args.dataset, args.modality, args.modality, step))
        plt.close()

        if args.lr_decay:
            adjust_learning_rate(args, opt, epoch)

def detailed_test(args, step, test_data_set, task_best_acc_list):
    print("=====================================")
    print("Start testing...")
    print("=====================================")

    model_path = './save/{}/{}/step_{}_best_{}_model.pkl'.format(args.dataset, args.modality, step, args.modality)
    model = torch.load(model_path)
    
    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    all_test_out_logits = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            # test_labels = test_labels.to(device)
            if args.modality == 'visual':
                test_visual = test_data
                test_visual = test_visual.to(device)
                test_out_logits = model(visual=test_visual)
            elif args.modality == 'audio':
                test_audio = test_data
                test_audio = test_audio.to(device)
                test_out_logits = model(audio=test_audio)
            else:
                test_visual = test_data[0]
                test_audio = test_data[1]
                test_visual = test_visual.to(device)
                test_audio = test_audio.to(device)
                test_out_logits = model(visual=test_visual, audio=test_audio)
            test_out_logits = F.softmax(test_out_logits, dim=-1).detach().cpu()
            all_test_out_logits = torch.cat((all_test_out_logits, test_out_logits), dim=0)
            all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
    test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
    print("Incremental step {} Testing res: {:.6f}".format(step, test_top1))

    if args.upper_bound:
        return None
    
    old_task_acc_list = []
    for i in range(step+1):
        step_class_list = range(i*args.class_num_per_step, (i+1)*args.class_num_per_step)
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        i_labels = torch.Tensor(all_test_labels.numpy()[step_class_idxs])
        i_logits = torch.Tensor(all_test_out_logits.numpy()[step_class_idxs])
        i_acc = top_1_acc(i_logits, i_labels)
        if i == step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)
    if step > 0:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        print('forgetting: {:.6f}'.format(forgetting))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None
    task_best_acc_list.append(curren_step_acc)

    return forgetting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AVE')
    parser.add_argument('--modality', type=str, default='audio-visual', choices=['visual', 'audio', 'audio-visual'])
    parser.add_argument('--feature_root', type=str, default="/mnt/data2/wpian/dataset/VGGSound", help='Root dir for feature files: visual_features.h5, audio_pretrained_feature_dict.npy, etc.')
    parser.add_argument('--meta_root', type=str, default=None, help='Root dir for metadata dicts: all_id_category_dict.npy, category_encode_dict.npy, all_classId_vid_dict.npy, etc.')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--infer_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=28)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--class_num_per_step', type=int, default=7)
    parser.add_argument('--upper_bound', type=boolean_string, default=False)

    # Difficulty-weighted loss options.
    parser.add_argument('--use_difficulty_weight', type=boolean_string, default=False,
                        help='Whether to use class-level difficulty weights in CE loss.')

    parser.add_argument('--difficulty_csv', type=str, default=None,
                        help='CSV containing class_id, raw_difficulty_recall, raw_difficulty_f1.')

    parser.add_argument('--difficulty_raw_col', type=str, default='raw_difficulty_f1',
                        choices=['raw_difficulty_recall', 'raw_difficulty_f1'],
                        help='Which raw difficulty column to use as the original difficulty.')

    parser.add_argument('--difficulty_smooth', type=boolean_string, default=True,
                        help='True: weight=1+lambda*difficulty; False: weight=max(difficulty, eps).')

    parser.add_argument('--difficulty_lambda', type=float, default=1.0,
                        help='Strength of smoothed difficulty weighting.')

    parser.add_argument('--difficulty_eps', type=float, default=1e-6,
                        help='Minimum weight for non-smoothed raw difficulty weighting.')

    parser.add_argument('--difficulty_mean_norm', type=boolean_string, default=True,
                        help='Normalize visible class weights so their mean is 1.')

    parser.add_argument('--difficulty_weight_mode', type=str, default='legacy',
                        choices=['legacy', 'raw', 'smooth', 'power', 'exp'],
                        help='How to convert difficulty into class weights.')

    parser.add_argument('--difficulty_power', type=float, default=0.5,
                        help='Power parameter for difficulty_weight_mode=power. '
                            'Example: 0.5 means weight=(D+eps)^0.5.')

    parser.add_argument('--difficulty_exp_alpha', type=float, default=2.0,
                        help='Alpha parameter for difficulty_weight_mode=exp. '
                            'Example: 2.0 means weight=exp(2D).')
                        

    args = parser.parse_args()
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)
    
    print('Training start time: {}'.format(datetime.now()))

    train_set = IcreLoader(args=args, mode='train', modality=args.modality)
    val_set = IcreLoader(args=args, mode='val', modality=args.modality)
    test_set = IcreLoader(args=args, mode='test', modality=args.modality)

    task_best_acc_list = []

    step_forgetting_list = []

    ckpts_root = './save/{}/{}/'.format(args.dataset, args.modality)
    figs_root = './save/fig/{}/{}/'.format(args.dataset, args.modality)

    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)
    if not os.path.exists(figs_root):
        os.makedirs(figs_root)

    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)

        print('Incremental step: {}'.format(step))
        train(args, step, train_set, val_set)

        step_forgetting = detailed_test(args, step, test_set, task_best_acc_list)
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)

    if not args.upper_bound:       
        Mean_forgetting = np.mean(step_forgetting_list)
        print('Average Forgetting: {:.6f}'.format(Mean_forgetting))

    if args.dataset != 'AVE' and args.modality != 'audio':
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()


