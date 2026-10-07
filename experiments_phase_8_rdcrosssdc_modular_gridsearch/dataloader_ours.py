import numpy as np
import errno
import logging
import math
import os
import re
import time
import torch
from torch.utils.data import Dataset
import random
import h5py
from tqdm import tqdm


DEFAULT_H5_READ_RETRIES = 5
DEFAULT_H5_RETRY_DELAY = 2.0
_MAX_H5_RETRY_DELAY = 30.0
_TRANSIENT_H5_ERRNOS = {
    getattr(errno, name)
    for name in (
        'EIO', 'ENXIO', 'ENODEV', 'ESTALE', 'ETIMEDOUT', 'EINTR',
        'EAGAIN', 'ECONNRESET', 'ENOTCONN',
    )
    if hasattr(errno, name)
}
_LOGGER = logging.getLogger(__name__)


def _h5_error_number(error):
    error_number = getattr(error, 'errno', None)
    if error_number is not None:
        return error_number
    # Some h5py versions wrap a low-level read error in KeyError/RuntimeError,
    # or an OSError without .errno, but retain the OS errno in the HDF5 message.
    match = re.search(r'(?:\berrno\s*=\s*|\[Errno\s+)(\d+)', str(error), re.IGNORECASE)
    return int(match.group(1)) if match else None


def h5_data_loader_kwargs(num_workers, persistent_workers=False):
    """Return DataLoader options that are valid for both 0 and N workers."""
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        # A visual batch is about 617 MB, so one prefetched batch per worker is
        # enough to overlap I/O without building a very large pinned queue.
        kwargs['prefetch_factor'] = 1
        kwargs['persistent_workers'] = persistent_workers
    return kwargs


class _LazyVisualH5Mixin:
    """Give each process its own lazily opened read-only HDF5 handle."""

    def _current_visual_feature_vids(self):
        raise NotImplementedError

    def _visual_cache_label(self):
        mode = getattr(self, 'mode', 'exemplar')
        step = getattr(self, 'incremental_step', 0)
        return '{} step {}'.format(mode, step)

    def preload_visual_features(self):
        """Load this dataset view's visual features once in the main process."""
        if self.args.dataset == 'AVE' or 'visual' not in self.modality:
            return 0

        vids = tuple(self._current_visual_feature_vids())
        if (
            getattr(self, '_visual_feature_cache', None) is not None
            and self._visual_feature_cache_vids == vids
        ):
            return self._visual_feature_cache_nbytes

        self.clear_visual_features_cache()
        if not vids:
            self._visual_feature_cache = {}
            self._visual_feature_cache_vids = vids
            self._visual_feature_cache_nbytes = 0
            return 0

        label = self._visual_cache_label()
        print(
            '[Visual cache] Loading {} samples for {}...'.format(len(vids), label),
            flush=True,
        )
        visual_cache = {}
        try:
            for vid in tqdm(vids, desc='Preload {}'.format(label), unit='sample'):
                if vid not in visual_cache:
                    visual_cache[vid] = np.asarray(
                        self._visual_feature(vid), dtype=np.float32
                    )
        finally:
            # Workers must never inherit the temporary parent-process H5 handle.
            self.close_visual_features_h5()

        cache_nbytes = sum(value.nbytes for value in visual_cache.values())
        self._visual_feature_cache = visual_cache
        self._visual_feature_cache_vids = vids
        self._visual_feature_cache_nbytes = cache_nbytes
        print(
            '[Visual cache] Ready {}: {:.2f} GiB'.format(
                label, cache_nbytes / float(1024 ** 3)
            ),
            flush=True,
        )
        return cache_nbytes

    def clear_visual_features_cache(self):
        """Release the current step's in-memory visual feature cache."""
        visual_cache = getattr(self, '_visual_feature_cache', None)
        if visual_cache is None:
            return

        cache_nbytes = getattr(self, '_visual_feature_cache_nbytes', 0)
        label = self._visual_cache_label()
        self._visual_feature_cache = None
        self._visual_feature_cache_vids = None
        self._visual_feature_cache_nbytes = 0
        print(
            '[Visual cache] Released {}: {:.2f} GiB'.format(
                label, cache_nbytes / float(1024 ** 3)
            ),
            flush=True,
        )

    def _visual_feature(self, vid):
        visual_cache = getattr(self, '_visual_feature_cache', None)
        if visual_cache is not None:
            return visual_cache[vid]

        if self.args.dataset == 'AVE':
            return self.all_visual_pretrained_features[vid]
        return self._with_visual_h5_retry(vid, lambda store: store[vid][()], 'read')

    def _has_visual_feature(self, vid):
        visual_cache = getattr(self, '_visual_feature_cache', None)
        if visual_cache is not None:
            return vid in visual_cache
        if self.args.dataset == 'AVE':
            return vid in self.all_visual_pretrained_features
        return self._with_visual_h5_retry(vid, lambda store: vid in store, 'lookup')

    def _with_visual_h5_retry(self, vid, operation, action):
        """Retry the same sample after reopening a transiently failing HDF5 file."""
        retries = getattr(self.args, 'h5_read_retries', DEFAULT_H5_READ_RETRIES)
        delay = getattr(self.args, 'h5_retry_delay', DEFAULT_H5_RETRY_DELAY)
        if retries < 0 or not math.isfinite(delay) or delay < 0:
            raise ValueError('HDF5 retries and retry delay must be finite and non-negative')
        delay = min(delay, _MAX_H5_RETRY_DELAY)

        for attempt in range(retries + 1):
            try:
                # Opening, resolving the dataset and reading its data must all
                # be retried together. Never reuse a Dataset from a closed file.
                result = operation(self._visual_feature_store())
            except (OSError, KeyError, RuntimeError) as error:
                error_number = _h5_error_number(error)
                try:
                    self.close_visual_features_h5()
                except (OSError, RuntimeError, ValueError) as close_error:
                    # The handle was already detached. Preserve the read error
                    # and allow the next attempt to open a fresh file.
                    _LOGGER.warning('[HDF5 retry] Failed to close %s: %s',
                                    self.visual_pretrained_feature_path, close_error)

                if error_number not in _TRANSIENT_H5_ERRNOS:
                    raise

                context = '{} sample={!r}, file={!r}, {}, pid={}'.format(
                    action, vid, os.path.abspath(self.visual_pretrained_feature_path),
                    self._visual_cache_label(), os.getpid(),
                )
                if attempt == retries:
                    message = (
                        'HDF5 {} failed after {} attempts. The sample was not skipped. '
                        'Check the file and storage mount before restarting. Last error: {}'
                    ).format(context, attempt + 1, error)
                    raise OSError(error_number, message,
                                  self.visual_pretrained_feature_path) from error

                _LOGGER.warning(
                    '[HDF5 retry] %s failed: %s; reopening in %.1fs (retry %d/%d).',
                    context, error, delay, attempt + 1, retries,
                )
            else:
                if attempt:
                    _LOGGER.warning('[HDF5 recovered] %s sample=%r, file=%r after %d retries.',
                                    action, vid, self.visual_pretrained_feature_path, attempt)
                return result

            # Fixed backoff deliberately avoids advancing the experiment RNG.
            time.sleep(delay)
            delay = min(delay * 2, _MAX_H5_RETRY_DELAY)

    def _visual_feature_store(self):
        if self.args.dataset == 'AVE':
            return self.all_visual_pretrained_features

        process_id = os.getpid()
        if (
            self.all_visual_pretrained_features is None
            or self._visual_h5_owner_pid != process_id
        ):
            # A forked DataLoader worker must not reuse the parent's handle.
            self.close_visual_features_h5()
            self.all_visual_pretrained_features = h5py.File(
                self.visual_pretrained_feature_path,
                'r',
            )
            self._visual_h5_owner_pid = process_id

        return self.all_visual_pretrained_features

    def close_visual_features_h5(self):
        if self.args.dataset == 'AVE':
            return

        visual_h5 = getattr(self, 'all_visual_pretrained_features', None)
        # Even if closing a damaged handle raises, it must never be reused.
        self.all_visual_pretrained_features = None
        self._visual_h5_owner_pid = None
        if visual_h5 is not None:
            visual_h5.close()

    def __getstate__(self):
        """Never pickle an open h5py handle when using spawn."""
        state = self.__dict__.copy()
        if self.args.dataset != 'AVE':
            state['all_visual_pretrained_features'] = None
            state['_visual_h5_owner_pid'] = None
        return state


class IcaAVELoader(_LazyVisualH5Mixin, Dataset):
    def __init__(self, args, mode='train', modality='visual', incremental_step=0, audio_features=None):
        self.mode = mode
        self.args = args
        self.modality = modality
        self._visual_h5_owner_pid = None
        self._visual_feature_cache = None
        self._visual_feature_cache_vids = None
        self._visual_feature_cache_nbytes = 0
        
        if args.dataset == 'AVE':
            self.feature_root = args.feature_root
            self.meta_root = args.meta_root
            self.visual_pretrained_feature_path = os.path.join(self.feature_root, 'visual_pretrained_feature_dict.npy')
            self.all_visual_pretrained_features = np.load(self.visual_pretrained_feature_path, allow_pickle=True).item()
        else:
            if args.dataset == 'ksounds':
                self.feature_root = args.feature_root
                self.meta_root = args.meta_root
            elif 'VGGSound' in args.dataset:
                # self.data_root = '../data/VGGSound_100'
                self.feature_root = args.feature_root
                self.meta_root = args.meta_root
            self.visual_pretrained_feature_path = os.path.join(self.feature_root, 'visual_features.h5')
            self.all_visual_pretrained_features = None
        
        self.audio_pretrained_feature_path = os.path.join(self.feature_root, 'audio_pretrained_feature', 'audio_pretrained_feature_dict.npy')
        # The trainer can share one read-only feature dictionary among its four
        # dataset views. Split-specific IDs and labels are still loaded below;
        # sharing feature storage does not change which samples a view exposes.
        self.all_audio_pretrained_features = (
            np.load(self.audio_pretrained_feature_path, allow_pickle=True).item()
            if audio_features is None else audio_features
        )

        self.all_id_category_dict = np.load(
            os.path.join(self.meta_root, 'all_id_category_dict.npy'), allow_pickle=True
        ).item()
        self.category_encode_dict = np.load(
            os.path.join(self.meta_root, 'category_encode_dict.npy'), allow_pickle=True
        ).item()
        self.all_classId_vid_dict = np.load(
            os.path.join(self.meta_root, 'all_classId_vid_dict.npy'), allow_pickle=True
        ).item()

        if self.mode == 'train':
            self.all_id_category_dict = self.all_id_category_dict['train']
            self.all_classId_vid_dict = self.all_classId_vid_dict['train']
        elif self.mode == 'val':
            self.all_id_category_dict = self.all_id_category_dict['val']
            self.all_classId_vid_dict = self.all_classId_vid_dict['val']
        elif self.mode == 'test':
            self.all_id_category_dict = self.all_id_category_dict['test']
            self.all_classId_vid_dict = self.all_classId_vid_dict['test']
        else:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')
        
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')

        self.incremental_step = incremental_step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

    def set_current_step_classes(self):
        if self.mode == 'train':
            current_step_class = np.array(range(self.args.class_num_per_step * self.incremental_step, self.args.class_num_per_step * (self.incremental_step + 1)))
        else:
            current_step_class = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
        return current_step_class

    def _get_class_vids(self, class_idx):
        """
        Compatible with dict keys being int or str.
        """
        try:
            return self.all_classId_vid_dict[class_idx]
        except KeyError:
            try:
                return self.all_classId_vid_dict[str(class_idx)]
            except KeyError:
                # Optional: show a helpful error message
                sample_keys = list(self.all_classId_vid_dict.keys())[:10]
                raise KeyError(
                    f"class_idx {class_idx} not found in all_classId_vid_dict. "
                    f"key types example: {type(sample_keys[0]) if sample_keys else None}, "
                    f"first keys: {sample_keys}"
                )

    def _current_visual_feature_vids(self):
        return self.all_current_data_vids

    def _has_feature(self, vid):
        has_visual = True
        has_audio = True

        if 'visual' in self.modality:
            has_visual = self._has_visual_feature(vid)

        if 'audio' in self.modality:
            has_audio = (vid in self.all_audio_pretrained_features)

        return has_visual and has_audio

    # def current_step_data(self):
    #     all_current_data_vids = []
    #     missing_count = 0
    #     total_count = 0

    #     for class_idx in self.current_step_class:
    #         vids = self._get_class_vids(int(class_idx))

    #         for vid in vids:
    #             total_count += 1
    #             if self._has_feature(vid):
    #                 all_current_data_vids.append(vid)
    #             else:
    #                 missing_count += 1

    #     if missing_count > 0:
    #         print(f"[Dataset] Step {self.incremental_step} | "
    #             f"Missing {missing_count}/{total_count} samples "
    #             f"({missing_count/total_count:.4f}) skipped.")

    #     return all_current_data_vids

    def current_step_data(self):
        all_current_data_vids = []
        for class_idx in self.current_step_class:
            all_current_data_vids += self.all_classId_vid_dict[str(class_idx)]
        return all_current_data_vids

    def set_incremental_step(self, step):
        self.clear_visual_features_cache()
        self.incremental_step = step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

    def __getitem__(self, index):
        vid = self.all_current_data_vids[index]
        category = self.all_id_category_dict[vid]
        category_id = self.category_encode_dict[category]

        if 'visual' in self.modality:
            visual_feature = self._visual_feature(vid)
            visual_feature = torch.as_tensor(visual_feature, dtype=torch.float32)
        
        if 'audio' in self.modality:
            audio_feature = self.all_audio_pretrained_features[vid]
            audio_feature = torch.Tensor(audio_feature)
        
        if self.modality == 'visual':
            return visual_feature, category_id
        elif self.modality == 'audio':
            return audio_feature, category_id
        else:
            return (visual_feature, audio_feature), category_id

    def __len__(self):
        return len(self.all_current_data_vids)


class exemplarLoader(_LazyVisualH5Mixin, Dataset):
    def __init__(self, args, modality='visual', incremental_step=0, audio_features=None):
        self.args = args
        self.modality = modality
        self._visual_h5_owner_pid = None
        self._visual_feature_cache = None
        self._visual_feature_cache_vids = None
        self._visual_feature_cache_nbytes = 0
        
        if args.dataset == 'AVE':
            self.feature_root = args.feature_root
            self.meta_root = args.meta_root
            self.visual_pretrained_feature_path = os.path.join(self.feature_root, 'visual_pretrained_feature_dict.npy')
            self.all_visual_pretrained_features = np.load(self.visual_pretrained_feature_path, allow_pickle=True).item()
        else:
            if args.dataset == 'ksounds':
                self.feature_root = args.feature_root
                self.meta_root = args.meta_root
            # elif args.dataset == 'VGGSound':
            #     print('dataset: VGGSound')
            #     self.feature_root = args.feature_root
            #     self.meta_root = args.meta_root
            elif 'VGGSound' in args.dataset:
                # self.data_root = '../data/VGGSound_100'
                self.feature_root = args.feature_root
                self.meta_root = args.meta_root
            self.visual_pretrained_feature_path = os.path.join(self.feature_root, 'visual_features.h5')
            self.all_visual_pretrained_features = None

        self.audio_pretrained_feature_path = os.path.join(self.feature_root, 'audio_pretrained_feature', 'audio_pretrained_feature_dict.npy')
        # Reuse the caller's store without copying it or modifying its arrays.
        # Replay membership and labels continue to come only from train metadata.
        self.all_audio_pretrained_features = (
            np.load(self.audio_pretrained_feature_path, allow_pickle=True).item()
            if audio_features is None else audio_features
        )

        self.all_id_category_dict = np.load(
            os.path.join(self.meta_root, 'all_id_category_dict.npy'), allow_pickle=True
        ).item()['train']
        self.all_classId_vid_dict = np.load(
            os.path.join(self.meta_root, 'all_classId_vid_dict.npy'), allow_pickle=True
        ).item()['train']
        self.category_encode_dict = np.load(
            os.path.join(self.meta_root, 'category_encode_dict.npy'), allow_pickle=True
        ).item()
        
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')

        self.incremental_step = incremental_step

        self.exemplar_class_vids_set = []
        self.exemplar_vids_set = []
    
    def _set_incremental_step_(self, step):
        self.clear_visual_features_cache()
        self.incremental_step = step
        self._update_exemplars_()

    def _update_exemplars_(self):
        if self.incremental_step == 0:
            return
        try:
            new_memory_classes = range((self.incremental_step - 1) * self.args.class_num_per_step, self.incremental_step * self.args.class_num_per_step)
            exemplar_num_per_class = self.args.memory_size // (self.incremental_step * self.args.class_num_per_step)
            new_memory_class_exemplars = self._init_new_memory_class_exemplars_(new_memory_classes, exemplar_num_per_class)

            if self.incremental_step == 1:
                self.exemplar_class_vids_set += new_memory_class_exemplars
            else:
                for i in range(len(self.exemplar_class_vids_set)):
                    self.exemplar_class_vids_set[i] = self.exemplar_class_vids_set[i][:exemplar_num_per_class]

                self.exemplar_class_vids_set += new_memory_class_exemplars

            self.exemplar_vids_set = np.array(self.exemplar_class_vids_set).reshape(-1).tolist()
            self.exemplar_vids_set = [vid for vid in self.exemplar_vids_set if vid is not None]
        finally:
            # Do not leave a parent-process handle open before workers fork.
            self.close_visual_features_h5()

    def _get_class_vids(self, class_idx):
        """
        Compatible with dict keys being int or str.
        """
        try:
            return self.all_classId_vid_dict[class_idx]
        except KeyError:
            try:
                return self.all_classId_vid_dict[str(class_idx)]
            except KeyError:
                # Optional: show a helpful error message
                sample_keys = list(self.all_classId_vid_dict.keys())[:10]
                raise KeyError(
                    f"class_idx {class_idx} not found in all_classId_vid_dict. "
                    f"key types example: {type(sample_keys[0]) if sample_keys else None}, "
                    f"first keys: {sample_keys}"
                )

    def _current_visual_feature_vids(self):
        return self.exemplar_vids_set

    def _has_feature(self, vid):
        has_visual = True
        has_audio = True

        if 'visual' in self.modality:
            has_visual = self._has_visual_feature(vid)

        if 'audio' in self.modality:
            has_audio = (vid in self.all_audio_pretrained_features)

        return has_visual and has_audio

    def _init_new_memory_class_exemplars_(self, new_memory_classes, exemplar_num_per_class):
        new_memory_class_exemplars = []

        for i in new_memory_classes:
            class_vids = self._get_class_vids(int(i))

            # 先过滤
            class_vids = [v for v in class_vids if self._has_feature(v)]

            if len(class_vids) == 0:
                print(f"[Exemplar] Warning: class {i} has no valid samples after filtering.")

            class_exemplar = random.sample(class_vids, min(len(class_vids), exemplar_num_per_class))

            if len(class_vids) < exemplar_num_per_class:
                class_exemplar += [None for _ in range(exemplar_num_per_class - len(class_vids))]

            new_memory_class_exemplars.append(class_exemplar)

        return new_memory_class_exemplars
    
    def __getitem__(self, index):
        vid = self.exemplar_vids_set[index]

        category = self.all_id_category_dict[vid]
        category_id = self.category_encode_dict[category]

        if 'visual' in self.modality:
            visual_feature = self._visual_feature(vid)
            visual_feature = torch.as_tensor(visual_feature, dtype=torch.float32)
        
        if 'audio' in self.modality:
            audio_feature = self.all_audio_pretrained_features[vid]
            audio_feature = torch.Tensor(audio_feature)
        
        if self.modality == 'visual':
            return visual_feature, category_id
        elif self.modality == 'audio':
            return audio_feature, category_id
        else:
            return (visual_feature, audio_feature), category_id

    def __len__(self):
        return len(self.exemplar_vids_set)
