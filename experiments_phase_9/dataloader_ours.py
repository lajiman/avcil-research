"""Phase-8 dataset/replay interfaces, with unique IDs and worker-safe HDF5.

Class order, feature formats and random per-class memory selection are retained.
Missing feature pairs are excluded from both training and gate sample counts.
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# 本文件的 P8 来源：experiments_phase_8_rdcrosssdc_modular/dataloader_ours.py。
# 数据/回放接口及类别划分逻辑沿用；公共基类和特征校验经过重构，不是整文件复制。

# [P8 逻辑沿用] 合并 IcaAVELoader/exemplarLoader 的重复初始化及双模态读取代码。
class _FeatureDataset(Dataset):
    def __init__(self, args, mode, modality, shared_audio_features=None):
        if modality != "audio-visual":
            raise ValueError("Phase 9 requires audio-visual features")
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val or test")
        # [P8 逻辑沿用] 沿用元数据文件名、类别编码及 AVE/npy、其余数据集/HDF5 的格式。
        # 视觉特征改为延迟打开，而不是在初始化时持有 HDF5 句柄。
        self.args, self.mode, self.modality = args, mode, modality
        self.feature_root, self.meta_root = args.feature_root, args.meta_root
        self.category_encode_dict = np.load(os.path.join(self.meta_root, "category_encode_dict.npy"), allow_pickle=True).item()
        self.all_id_category_dict = np.load(os.path.join(self.meta_root, "all_id_category_dict.npy"), allow_pickle=True).item()[mode]
        self.all_classId_vid_dict = np.load(os.path.join(self.meta_root, "all_classId_vid_dict.npy"), allow_pickle=True).item()[mode]
        # [P9 新增] 一个实验内的四个数据集共用只读用途的音频字典，避免完整加载四份。
        # 不使用跨运行的全局缓存；样本选择仍受各自 split/当前类/memory ID 限制。
        self.all_audio_pretrained_features = shared_audio_features
        if self.all_audio_pretrained_features is None:
            self.all_audio_pretrained_features = np.load(
                os.path.join(self.feature_root, "audio_pretrained_feature", "audio_pretrained_feature_dict.npy"), allow_pickle=True
            ).item()
        self.visual_pretrained_feature_path = os.path.join(
            self.feature_root, "visual_pretrained_feature_dict.npy" if args.dataset == "AVE" else "visual_features.h5"
        )
        self.all_visual_pretrained_features = None
        self._visual_pid = None

    # [P9 新增] 按进程延迟打开视觉特征，避免 DataLoader worker 共享 HDF5 句柄。
    def _visual_store(self):
        if self.all_visual_pretrained_features is None or self._visual_pid != os.getpid():
            self.close_visual_features_h5()
            if self.args.dataset == "AVE":
                self.all_visual_pretrained_features = np.load(self.visual_pretrained_feature_path, allow_pickle=True).item()
            else:
                self.all_visual_pretrained_features = h5py.File(self.visual_pretrained_feature_path, "r")
            self._visual_pid = os.getpid()
        return self.all_visual_pretrained_features

    # [P9 新增] spawn 序列化时不传递已打开的 HDF5 句柄。
    def __getstate__(self):
        state = self.__dict__.copy()
        state["all_visual_pretrained_features"] = None
        state["_visual_pid"] = None
        return state

    # [P8 逻辑沿用] 同名关闭接口；新增类型判断和缓存重置，也兼容 AVE 的字典格式。
    def close_visual_features_h5(self):
        if isinstance(self.all_visual_pretrained_features, h5py.File):
            self.all_visual_pretrained_features.close()
        self.all_visual_pretrained_features = None
        self._visual_pid = None

    # [P8 逻辑沿用] 沿用 exemplarLoader._get_class_vids 的整数/字符串类别键兼容逻辑。
    def _get_class_vids(self, class_idx):
        if class_idx in self.all_classId_vid_dict:
            return self.all_classId_vid_dict[class_idx]
        return self.all_classId_vid_dict[str(class_idx)]

    # [P8 逻辑沿用] exemplarLoader._has_feature 的配对特征检查；此处仅支持双模态。
    def _has_feature(self, vid):
        return vid in self.all_audio_pretrained_features and vid in self._visual_store()

    # [P9 新增] 统一对所有 split 去重、过滤缺失配对并检查标签；P8 的回放已有特征过滤，
    # 这里将其扩展到当前训练/验证/测试集，以保证 gate 的样本数对应唯一有效视频。
    def _valid_class_vids(self, class_idx):
        vids = list(dict.fromkeys(self._get_class_vids(class_idx)))
        valid = [vid for vid in vids if self._has_feature(vid)]
        if len(valid) != len(vids):
            print(f"[{self.mode}] Class {class_idx}: skipped {len(vids) - len(valid)} missing feature pairs")
        for vid in valid:
            if self.category_encode_dict[self.all_id_category_dict[vid]] != class_idx:
                raise ValueError(f"Metadata class order mismatch for {vid}")
        return valid

    # [P8 逻辑沿用] 合并两个 loader 的 __getitem__ 双模态分支，保持
    # ((visual, audio), 全局类别 ID) 接口；张量转换与 HDF5 读取方式有调整。
    def _read(self, vid):
        visual = self._visual_store()[vid]
        if isinstance(visual, h5py.Dataset):
            visual = visual[()]
        audio = self.all_audio_pretrained_features[vid]
        label = int(self.category_encode_dict[self.all_id_category_dict[vid]])
        return (torch.as_tensor(np.asarray(visual), dtype=torch.float32),
                torch.as_tensor(np.asarray(audio), dtype=torch.float32)), label


class IcaAVELoader(_FeatureDataset):
    # [P8 逻辑沿用] 同名构造器的职责；公共初始化委托基类，类别切换合并到下方方法。
    def __init__(self, args, mode="train", modality="audio-visual", incremental_step=0, shared_audio_features=None):
        super().__init__(args, mode, modality, shared_audio_features)
        self.set_incremental_step(incremental_step)

    # [P8 逻辑沿用] 合并同名方法、set_current_step_classes 和 current_step_data：
    # train 只含当前新类，val/test 含所有已见类；额外应用上方有效样本过滤。
    def set_incremental_step(self, step):
        self.incremental_step = step
        lower = self.args.class_num_per_step * step if self.mode == "train" else 0
        upper = self.args.class_num_per_step * (step + 1)
        self.current_step_class = list(range(lower, upper))
        self.all_current_data_vids = [vid for c in self.current_step_class for vid in self._valid_class_vids(c)]
        self.close_visual_features_h5()

    # [P8 逻辑沿用] 按当前数据 ID 取样；双模态读取抽到 _read。
    def __getitem__(self, index):
        return self._read(self.all_current_data_vids[index])

    # [P8 原样沿用] IcaAVELoader.__len__。
    def __len__(self):
        return len(self.all_current_data_vids)


class exemplarLoader(_FeatureDataset):
    # [P8 逻辑沿用] 同名构造器仍建立训练 split 的回放集合；共享初始化委托基类。
    def __init__(self, args, modality="audio-visual", incremental_step=0, shared_audio_features=None):
        super().__init__(args, "train", modality, shared_audio_features)
        self.incremental_step = incremental_step
        self.exemplar_class_vids_set = []
        self.exemplar_vids_set = []

    # [P8 逻辑沿用] 合并同名方法、_update_exemplars_ 和 _init_new_memory_class_exemplars_。
    # [P9 新增] 检查 step 顺序，用有效 ID 列表代替可能含 None 的填充数组。
    def _set_incremental_step_(self, step):
        if step == 0:
            return
        expected_old = (step - 1) * self.args.class_num_per_step
        if len(self.exemplar_class_vids_set) != expected_old:
            raise ValueError("Replay steps must be initialized sequentially")
        self.incremental_step = step
        # [P8 逻辑沿用] memory_size // 旧类数；已有类别按原顺序截短，
        # 上一任务刚学过的类别用 random.sample 选回放，不引入新的 exemplar 选择算法。
        per_class = self.args.memory_size // (step * self.args.class_num_per_step)
        self.exemplar_class_vids_set = [vids[:per_class] for vids in self.exemplar_class_vids_set]
        for c in range(expected_old, step * self.args.class_num_per_step):
            vids = self._valid_class_vids(c)
            self.exemplar_class_vids_set.append(random.sample(vids, min(len(vids), per_class)))
        self.exemplar_vids_set = [vid for group in self.exemplar_class_vids_set for vid in group]
        self.close_visual_features_h5()

    # [P8 逻辑沿用] 按回放 ID 取样；双模态读取抽到 _read。
    def __getitem__(self, index):
        return self._read(self.exemplar_vids_set[index])

    # [P8 原样沿用] exemplarLoader.__len__。
    def __len__(self):
        return len(self.exemplar_vids_set)
