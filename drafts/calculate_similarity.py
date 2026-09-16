# -*- coding: utf-8 -*-
"""
Build class-level similarity matrices for VGGSound-style datasets.

Supported similarity types:
1. CLIP text similarity
2. Audio feature class-mean similarity
3. Video feature class-mean similarity
4. Audio-video joint class-mean similarity

Outputs:
- pandas DataFrame with row/column headers = class names
- CSV file
- NPY file

Author: ChatGPT
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class SimilarityConfig:
    # Paths
    audio_feature_path: Optional[str] = None
    video_feature_path: Optional[str] = None
    classid_vids_path: Optional[str] = None   # .npy containing dict, usually all_classId_vid_dict.npy
    output_dir: str = "./similarity_outputs"

    # Text encoder
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_prompt_template: str = "a video of {}"
    device: str = "cuda"

    # Reduction
    sample_reduce_audio: str = "mean"   # mean / flatten
    sample_reduce_video: str = "mean"   # mean / flatten

    # Joint fusion
    joint_fusion_method: str = "concat"  # concat / mean
    joint_fusion_alpha: float = 0.5

    # Normalization
    normalize_each_sample: bool = False
    normalize_class_vector: bool = True

    # Split name if classid_vids_path stores {"train": ..., "val": ..., "test": ...}
    split: str = "train"


# ============================================================
# User category dict
# ============================================================

CATEGORY_ENCODE_DICT = {
    "airplane flyby": 53,
    "arc welding": 56,
    "baby laughter": 7,
    "beat boxing": 43,
    "bowling impact": 25,
    "canary calling": 71,
    "cap gun shooting": 4,
    "cat purring": 34,
    "child singing": 1,
    "child speech, kid speaking": 6,
    "civil defense siren": 15,
    "dog growling": 99,
    "dog howling": 32,
    "driving motorcycle": 39,
    "electric shaver, electric razor shaving": 80,
    "female speech, woman speaking": 65,
    "fire truck siren": 12,
    "fireworks banging": 81,
    "goose honking": 28,
    "hail": 96,
    "heart sounds, heartbeat": 2,
    "helicopter": 30,
    "lathe spinning": 18,
    "lawn mowing": 79,
    "lighting firecrackers": 9,
    "male speech, man speaking": 91,
    "missile launch": 51,
    "mynah bird singing": 57,
    "ocean burbling": 67,
    "orchestra": 48,
    "people booing": 98,
    "people cheering": 23,
    "people crowd": 62,
    "people marching": 26,
    "people whispering": 17,
    "people whistling": 89,
    "pheasant crowing": 72,
    "pigeon, dove cooing": 60,
    "planing timber": 78,
    "playing accordion": 92,
    "playing acoustic guitar": 70,
    "playing bagpipes": 13,
    "playing banjo": 11,
    "playing bass drum": 10,
    "playing bass guitar": 27,
    "playing bassoon": 85,
    "playing bongo": 59,
    "playing cello": 44,
    "playing clarinet": 94,
    "playing cornet": 45,
    "playing cymbal": 97,
    "playing double bass": 90,
    "playing electric guitar": 88,
    "playing electronic organ": 24,
    "playing erhu": 31,
    "playing flute": 69,
    "playing french horn": 14,
    "playing glockenspiel": 20,
    "playing hammond organ": 21,
    "playing harp": 49,
    "playing harpsichord": 95,
    "playing mandolin": 64,
    "playing marimba, xylophone": 52,
    "playing piano": 58,
    "playing saxophone": 54,
    "playing sitar": 87,
    "playing squash": 41,
    "playing steel guitar, slide guitar": 47,
    "playing synthesizer": 33,
    "playing tabla": 93,
    "playing table tennis": 8,
    "playing tambourine": 38,
    "playing theremin": 19,
    "playing trombone": 29,
    "playing vibraphone": 63,
    "playing violin, fiddle": 76,
    "playing volleyball": 0,
    "police car (siren)": 22,
    "police radio chatter": 86,
    "printer printing": 77,
    "race car, auto racing": 61,
    "roller coaster running": 50,
    "rope skipping": 5,
    "scuba diving": 37,
    "sharpen knife": 75,
    "singing bowl": 3,
    "singing choir": 68,
    "skiing": 35,
    "slot machine": 42,
    "stream burbling": 84,
    "subway, metro, underground": 55,
    "tap dancing": 82,
    "tapping guitar": 83,
    "tractor digging": 16,
    "vacuum cleaner cleaning floors": 40,
    "volcano explosion": 66,
    "wind noise": 46,
    "wood thrush calling": 36,
    "woodpecker pecking tree": 73,
    "yodelling": 74
}


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_npy_dict(path: str) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    elif hasattr(obj, "item"):
        try:
            obj = obj.item()
        except Exception:
            pass
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a dict-like object.")
    return obj


def invert_category_dict(category_encode_dict: Dict[str, int]) -> Dict[int, str]:
    return {cid: name for name, cid in category_encode_dict.items()}


def get_ordered_class_names(category_encode_dict: Dict[str, int]) -> List[str]:
    id_to_name = invert_category_dict(category_encode_dict)
    class_ids = sorted(id_to_name.keys())
    return [id_to_name[cid] for cid in class_ids]


def get_ordered_class_ids(category_encode_dict: Dict[str, int]) -> List[int]:
    return sorted(category_encode_dict.values())


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


# ============================================================
# Sample feature -> unified 1D vector
# ============================================================

def flatten_to_vector(x: Any) -> np.ndarray:
    x = to_numpy(x).astype(np.float32)
    if x.ndim == 0:
        return x.reshape(1)
    return x.reshape(-1)


def reduce_sample_feature(x: Any, reduce: str = "mean") -> np.ndarray:
    """
    Convert one sample's raw feature into one 1D vector.

    Rules:
    - ndim == 0: scalar -> (1,)
    - ndim == 1: keep
    - ndim >= 2:
        * mean: average over all axes except the last
        * flatten: flatten all dims
    """
    x = to_numpy(x).astype(np.float32)

    if x.ndim == 0:
        return x.reshape(1)

    if x.ndim == 1:
        return x

    if reduce == "flatten":
        return x.reshape(-1)

    if reduce == "mean":
        axes = tuple(range(x.ndim - 1))
        x = x.mean(axis=axes)
        if x.ndim == 0:
            x = x.reshape(1)
        return x.reshape(-1)

    raise ValueError(f"Unsupported reduce mode: {reduce}")


# ============================================================
# Similarity
# ============================================================

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    vectors: (C, D)
    returns: (C, C)
    """
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    vectors = vectors / norms
    return vectors @ vectors.T


def vectors_to_similarity_dataframe(
    class_names: List[str],
    vectors: np.ndarray
) -> pd.DataFrame:
    sim = cosine_similarity_matrix(vectors)
    df = pd.DataFrame(sim, index=class_names, columns=class_names)
    return df


# ============================================================
# Text encoder (CLIP)
# ============================================================

class ClipTextEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        from transformers import CLIPTokenizer, CLIPTextModel

        if device == "cpu":
            self.device = "cpu"
        elif device.startswith("cuda") and torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        if getattr(outputs, "pooler_output", None) is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0, :]

        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)


def build_text_vectors(
    category_encode_dict: Dict[str, int],
    encoder: ClipTextEncoder,
    prompt_template: str = "a video of {}",
) -> Tuple[List[str], np.ndarray]:
    class_ids = get_ordered_class_ids(category_encode_dict)
    id_to_name = invert_category_dict(category_encode_dict)
    class_names = [id_to_name[cid] for cid in class_ids]

    texts = [prompt_template.format(name) for name in class_names]
    vectors = encoder.encode(texts)
    return class_names, vectors


# ============================================================
# Feature readers
# ============================================================

class AudioFeatureReader:
    def __init__(self, audio_feature_path: str):
        self.audio_store = load_npy_dict(audio_feature_path)

    def has(self, vid: str) -> bool:
        return vid in self.audio_store

    def read(self, vid: str, reduce: str = "mean") -> np.ndarray:
        if vid not in self.audio_store:
            raise KeyError(f"Audio feature missing: {vid}")
        feat = self.audio_store[vid]
        return reduce_sample_feature(feat, reduce=reduce)


class VideoFeatureReader:
    def __init__(self, video_feature_path: str):
        self.video_feature_path = video_feature_path
        self.h5 = None

    def __enter__(self):
        self.h5 = h5py.File(self.video_feature_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5 is not None:
            self.h5.close()

    def has(self, vid: str) -> bool:
        return vid in self.h5

    def read(self, vid: str, reduce: str = "mean") -> np.ndarray:
        if vid not in self.h5:
            raise KeyError(f"Video feature missing: {vid}")
        feat = self.h5[vid][()]
        return reduce_sample_feature(feat, reduce=reduce)


# ============================================================
# Joint fusion
# ============================================================

def fuse_audio_video(
    audio_vec: np.ndarray,
    video_vec: np.ndarray,
    method: str = "concat",
    alpha: float = 0.5,
) -> np.ndarray:
    audio_vec = flatten_to_vector(audio_vec)
    video_vec = flatten_to_vector(video_vec)

    if method == "concat":
        return np.concatenate([audio_vec, video_vec], axis=0).astype(np.float32)

    if method == "mean":
        if audio_vec.shape[0] != video_vec.shape[0]:
            raise ValueError(
                f"audio/video dims do not match for mean fusion: "
                f"{audio_vec.shape[0]} vs {video_vec.shape[0]}"
            )
        return (alpha * audio_vec + (1.0 - alpha) * video_vec).astype(np.float32)

    raise ValueError(f"Unsupported fusion method: {method}")


# ============================================================
# Class-level aggregation
# ============================================================

def aggregate_vectors_to_class_prototypes(
    per_class_sample_vectors: Dict[str, List[np.ndarray]],
    normalize_each_sample: bool = False,
    normalize_class_vector: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Input:
        per_class_sample_vectors[class_name] = [vec1, vec2, ...]
    Output:
        class_names: ordered list
        class_matrix: (C, D)
    """
    class_names = sorted(per_class_sample_vectors.keys())
    class_vectors = []

    for class_name in class_names:
        sample_vecs = per_class_sample_vectors[class_name]
        if len(sample_vecs) == 0:
            raise ValueError(f"No valid samples for class: {class_name}")

        processed = []
        for v in sample_vecs:
            v = flatten_to_vector(v)
            if normalize_each_sample:
                v = l2_normalize(v)
            processed.append(v)

        dims = [v.shape[0] for v in processed]
        if len(set(dims)) != 1:
            raise ValueError(
                f"Inconsistent vector dims for class '{class_name}': {set(dims)}"
            )

        class_vec = np.mean(np.stack(processed, axis=0), axis=0)
        if normalize_class_vector:
            class_vec = l2_normalize(class_vec)

        class_vectors.append(class_vec.astype(np.float32))

    class_matrix = np.stack(class_vectors, axis=0).astype(np.float32)
    return class_names, class_matrix


# ============================================================
# Load class -> vids mapping
# ============================================================

def load_classid_vids(
    classid_vids_path: str,
    split: str = "train"
) -> Dict[int, List[str]]:
    """
    Supports either:
    1. {class_id: [vids]}
    2. {"train": {class_id: [vids]}, "val": ..., "test": ...}
    """
    data = load_npy_dict(classid_vids_path)

    if all(isinstance(k, (int, np.integer)) for k in data.keys()):
        return {int(k): list(v) for k, v in data.items()}

    if split in data:
        sub = data[split]
        return {int(k): list(v) for k, v in sub.items()}

    raise ValueError(
        f"Cannot find split='{split}' in {classid_vids_path}. "
        f"Available keys: {list(data.keys())[:10]}"
    )


# ============================================================
# Build per-class vectors for audio / video / joint
# ============================================================

def build_audio_class_matrix(
    category_encode_dict: Dict[str, int],
    classid_vids_dict: Dict[int, List[str]],
    audio_feature_path: str,
    sample_reduce: str = "mean",
    normalize_each_sample: bool = False,
    normalize_class_vector: bool = True,
) -> Tuple[List[str], np.ndarray]:
    id_to_name = invert_category_dict(category_encode_dict)
    per_class_sample_vectors: Dict[str, List[np.ndarray]] = {}

    reader = AudioFeatureReader(audio_feature_path)

    for cid in sorted(category_encode_dict.values()):
        class_name = id_to_name[cid]
        vids = classid_vids_dict.get(cid, [])
        sample_vecs = []

        for vid in vids:
            if not reader.has(vid):
                continue
            vec = reader.read(vid, reduce=sample_reduce)
            sample_vecs.append(vec)

        per_class_sample_vectors[class_name] = sample_vecs

    return aggregate_vectors_to_class_prototypes(
        per_class_sample_vectors,
        normalize_each_sample=normalize_each_sample,
        normalize_class_vector=normalize_class_vector,
    )


def build_video_class_matrix(
    category_encode_dict: Dict[str, int],
    classid_vids_dict: Dict[int, List[str]],
    video_feature_path: str,
    sample_reduce: str = "mean",
    normalize_each_sample: bool = False,
    normalize_class_vector: bool = True,
) -> Tuple[List[str], np.ndarray]:
    id_to_name = invert_category_dict(category_encode_dict)
    per_class_sample_vectors: Dict[str, List[np.ndarray]] = {}

    with VideoFeatureReader(video_feature_path) as reader:
        for cid in sorted(category_encode_dict.values()):
            class_name = id_to_name[cid]
            vids = classid_vids_dict.get(cid, [])
            sample_vecs = []

            for vid in vids:
                if not reader.has(vid):
                    continue
                vec = reader.read(vid, reduce=sample_reduce)
                sample_vecs.append(vec)

            per_class_sample_vectors[class_name] = sample_vecs

    return aggregate_vectors_to_class_prototypes(
        per_class_sample_vectors,
        normalize_each_sample=normalize_each_sample,
        normalize_class_vector=normalize_class_vector,
    )


def build_joint_class_matrix(
    category_encode_dict: Dict[str, int],
    classid_vids_dict: Dict[int, List[str]],
    audio_feature_path: str,
    video_feature_path: str,
    sample_reduce_audio: str = "mean",
    sample_reduce_video: str = "mean",
    fusion_method: str = "concat",
    fusion_alpha: float = 0.5,
    normalize_each_sample: bool = False,
    normalize_class_vector: bool = True,
) -> Tuple[List[str], np.ndarray]:
    id_to_name = invert_category_dict(category_encode_dict)
    per_class_sample_vectors: Dict[str, List[np.ndarray]] = {}

    audio_reader = AudioFeatureReader(audio_feature_path)
    with VideoFeatureReader(video_feature_path) as video_reader:
        for cid in sorted(category_encode_dict.values()):
            class_name = id_to_name[cid]
            vids = classid_vids_dict.get(cid, [])
            sample_vecs = []

            for vid in vids:
                if not audio_reader.has(vid):
                    continue
                if not video_reader.has(vid):
                    continue

                audio_vec = audio_reader.read(vid, reduce=sample_reduce_audio)
                video_vec = video_reader.read(vid, reduce=sample_reduce_video)
                joint_vec = fuse_audio_video(
                    audio_vec,
                    video_vec,
                    method=fusion_method,
                    alpha=fusion_alpha
                )
                sample_vecs.append(joint_vec)

            per_class_sample_vectors[class_name] = sample_vecs

    return aggregate_vectors_to_class_prototypes(
        per_class_sample_vectors,
        normalize_each_sample=normalize_each_sample,
        normalize_class_vector=normalize_class_vector,
    )


# ============================================================
# Save outputs
# ============================================================

def save_similarity_outputs(
    df: pd.DataFrame,
    output_dir: str,
    stem: str,
) -> None:
    ensure_dir(output_dir)

    csv_path = os.path.join(output_dir, f"{stem}.csv")
    npy_path = os.path.join(output_dir, f"{stem}.npy")

    df.to_csv(csv_path, encoding="utf-8")

    payload = {
        "class_names": df.index.tolist(),
        "similarity_matrix": df.values.astype(np.float32),
    }
    np.save(npy_path, payload, allow_pickle=True)

    print(f"[Saved] CSV: {csv_path}")
    print(f"[Saved] NPY: {npy_path}")


def min_max_normalize_matrix(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a matrix to [0, 1] using global min-max normalization.

    Args:
        matrix: input similarity matrix, shape (C, C)
        eps: small value to avoid division by zero

    Returns:
        normalized matrix in [0, 1]
    """
    matrix = matrix.astype(np.float32)
    min_val = matrix.min()
    max_val = matrix.max()

    if max_val - min_val < eps:
        # all values are the same
        return np.zeros_like(matrix, dtype=np.float32)

    normalized = (matrix - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def load_similarity_npy(npy_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load a saved similarity npy file.

    Expected format:
    {
        "class_names": [...],
        "similarity_matrix": np.ndarray
    }
    """
    payload = np.load(npy_path, allow_pickle=True).item()

    if "class_names" not in payload:
        raise KeyError(f"'class_names' not found in {npy_path}")
    if "similarity_matrix" not in payload:
        raise KeyError(f"'similarity_matrix' not found in {npy_path}")

    class_names = list(payload["class_names"])
    sim_matrix = np.asarray(payload["similarity_matrix"], dtype=np.float32)
    return class_names, sim_matrix


def save_similarity_dataframe_and_npy(
    class_names: List[str],
    sim_matrix: np.ndarray,
    output_dir: str,
    stem: str,
) -> None:
    """
    Save similarity matrix as both csv and npy, with class names as row/col headers.
    """
    ensure_dir(output_dir)

    df = pd.DataFrame(sim_matrix, index=class_names, columns=class_names)

    csv_path = os.path.join(output_dir, f"{stem}.csv")
    npy_path = os.path.join(output_dir, f"{stem}.npy")

    df.to_csv(csv_path, encoding="utf-8")

    payload = {
        "class_names": class_names,
        "similarity_matrix": sim_matrix.astype(np.float32),
    }
    np.save(npy_path, payload, allow_pickle=True)

    print(f"[Saved normalized CSV] {csv_path}")
    print(f"[Saved normalized NPY] {npy_path}")


def normalize_existing_similarity_file(
    input_npy_path: str,
    output_dir: Optional[str] = None,
    suffix: str = "_norm01",
) -> None:
    """
    Normalize an existing similarity npy file to [0, 1], then save csv + npy.

    Args:
        input_npy_path: existing .npy file path
        output_dir: output directory; if None, use the same directory as input
        suffix: suffix added to output file stem
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_npy_path)

    class_names, sim_matrix = load_similarity_npy(input_npy_path)
    norm_matrix = min_max_normalize_matrix(sim_matrix)

    stem = os.path.splitext(os.path.basename(input_npy_path))[0] + suffix
    save_similarity_dataframe_and_npy(
        class_names=class_names,
        sim_matrix=norm_matrix,
        output_dir=output_dir,
        stem=stem,
    )


# ============================================================
# Public API
# ============================================================

def build_text_clip_similarity(
    category_encode_dict: Dict[str, int],
    config: SimilarityConfig,
) -> pd.DataFrame:
    encoder = ClipTextEncoder(
        model_name=config.clip_model_name,
        device=config.device
    )
    class_names, vectors = build_text_vectors(
        category_encode_dict=category_encode_dict,
        encoder=encoder,
        prompt_template=config.clip_prompt_template,
    )
    return vectors_to_similarity_dataframe(class_names, vectors)


def build_audio_similarity(
    category_encode_dict: Dict[str, int],
    config: SimilarityConfig,
) -> pd.DataFrame:
    if config.audio_feature_path is None:
        raise ValueError("config.audio_feature_path is required for audio similarity.")
    if config.classid_vids_path is None:
        raise ValueError("config.classid_vids_path is required for audio similarity.")

    classid_vids_dict = load_classid_vids(config.classid_vids_path, split=config.split)
    class_names, vectors = build_audio_class_matrix(
        category_encode_dict=category_encode_dict,
        classid_vids_dict=classid_vids_dict,
        audio_feature_path=config.audio_feature_path,
        sample_reduce=config.sample_reduce_audio,
        normalize_each_sample=config.normalize_each_sample,
        normalize_class_vector=config.normalize_class_vector,
    )
    return vectors_to_similarity_dataframe(class_names, vectors)


def build_video_similarity(
    category_encode_dict: Dict[str, int],
    config: SimilarityConfig,
) -> pd.DataFrame:
    if config.video_feature_path is None:
        raise ValueError("config.video_feature_path is required for video similarity.")
    if config.classid_vids_path is None:
        raise ValueError("config.classid_vids_path is required for video similarity.")

    classid_vids_dict = load_classid_vids(config.classid_vids_path, split=config.split)
    class_names, vectors = build_video_class_matrix(
        category_encode_dict=category_encode_dict,
        classid_vids_dict=classid_vids_dict,
        video_feature_path=config.video_feature_path,
        sample_reduce=config.sample_reduce_video,
        normalize_each_sample=config.normalize_each_sample,
        normalize_class_vector=config.normalize_class_vector,
    )
    return vectors_to_similarity_dataframe(class_names, vectors)


def build_joint_similarity(
    category_encode_dict: Dict[str, int],
    config: SimilarityConfig,
) -> pd.DataFrame:
    if config.audio_feature_path is None:
        raise ValueError("config.audio_feature_path is required for joint similarity.")
    if config.video_feature_path is None:
        raise ValueError("config.video_feature_path is required for joint similarity.")
    if config.classid_vids_path is None:
        raise ValueError("config.classid_vids_path is required for joint similarity.")

    classid_vids_dict = load_classid_vids(config.classid_vids_path, split=config.split)
    class_names, vectors = build_joint_class_matrix(
        category_encode_dict=category_encode_dict,
        classid_vids_dict=classid_vids_dict,
        audio_feature_path=config.audio_feature_path,
        video_feature_path=config.video_feature_path,
        sample_reduce_audio=config.sample_reduce_audio,
        sample_reduce_video=config.sample_reduce_video,
        fusion_method=config.joint_fusion_method,
        fusion_alpha=config.joint_fusion_alpha,
        normalize_each_sample=config.normalize_each_sample,
        normalize_class_vector=config.normalize_class_vector,
    )
    return vectors_to_similarity_dataframe(class_names, vectors)


# ============================================================
# Example main
# ============================================================

def main():
    config = SimilarityConfig(
        audio_feature_path="/mnt/data2/wpian/dataset/VGGSound/audio_pretrained_feature/audio_pretrained_feature_dict.npy",
        video_feature_path="/mnt/data2/wpian/dataset/VGGSound/visual_features.h5",
        classid_vids_path="/home/dal400642/Projects/AV-CIL_ICCV2023/data2/balance/all_classId_vid_dict.npy",
        output_dir="./similarity_outputs",
        clip_model_name="openai/clip-vit-base-patch32",
        clip_prompt_template="a video of {}",
        device="cpu",
        sample_reduce_audio="mean",
        sample_reduce_video="mean",
        joint_fusion_method="concat",
        joint_fusion_alpha=0.5,
        normalize_each_sample=False,
        normalize_class_vector=True,
        split="train",
    )

    ensure_dir(config.output_dir)

    # # 1. text clip
    # df_text = build_text_clip_similarity(CATEGORY_ENCODE_DICT, config)
    # save_similarity_outputs(df_text, config.output_dir, "text_clip_similarity")

    # # 2. audio mean
    # df_audio = build_audio_similarity(CATEGORY_ENCODE_DICT, config)
    # save_similarity_outputs(df_audio, config.output_dir, "audio_mean_similarity")

    # # 3. video mean
    # df_video = build_video_similarity(CATEGORY_ENCODE_DICT, config)
    # save_similarity_outputs(df_video, config.output_dir, "video_mean_similarity")

    # # 4. joint mean
    # df_joint = build_joint_similarity(CATEGORY_ENCODE_DICT, config)
    # save_similarity_outputs(df_joint, config.output_dir, "joint_mean_similarity")

    # print("All similarity matrices have been built successfully.")

    existing_npy_files = [
        os.path.join(config.output_dir, "text_clip_similarity.npy"),
        os.path.join(config.output_dir, "audio_mean_similarity.npy"),
        os.path.join(config.output_dir, "video_mean_similarity.npy"),
        os.path.join(config.output_dir, "joint_mean_similarity.npy"),
    ]

    for npy_path in existing_npy_files:
        if os.path.exists(npy_path):
            normalize_existing_similarity_file(
                input_npy_path=npy_path,
                output_dir=config.output_dir,
                suffix="_norm01",
            )
        else:
            print(f"[Skip] File not found: {npy_path}")


if __name__ == "__main__":
    main()