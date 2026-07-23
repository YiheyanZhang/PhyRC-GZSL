from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def resolve_path(root: Path, primary: str, fallback: str | None = None) -> Path:
    path = root / primary
    if path.exists():
        return path
    if fallback:
        fallback_path = root / fallback
        if fallback_path.exists():
            return fallback_path
    raise FileNotFoundError(f"Missing file: {path}")


def load_mat_array(path: Path, key: str, fallback_key: str | None = None) -> np.ndarray:
    payload = sio.loadmat(path)
    if key in payload:
        return np.asarray(payload[key])
    if fallback_key and fallback_key in payload:
        return np.asarray(payload[fallback_key])
    visible = [name for name in payload if not name.startswith("__")]
    raise KeyError(f"Key {key!r} not found in {path}. Available keys: {visible}")


def normalize_hsi(x: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.all():
        x = np.nan_to_num(x)
    stats = x if reference is None else np.nan_to_num(np.asarray(reference, dtype=np.float32))
    min_value = float(stats.min())
    max_value = float(stats.max())
    if max_value <= min_value:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - min_value) / (max_value - min_value)).astype(np.float32)


def filter_background(x: np.ndarray, y: np.ndarray):
    y = np.asarray(y).reshape(-1)
    x = np.asarray(x)
    mask = y != 0
    return x[mask], y[mask]


def flatten_hsi(x: np.ndarray, gt: np.ndarray, ignore_background: bool = True):
    if x.ndim != 3:
        raise ValueError("HSI data must have shape [H, W, Bands]")
    if gt.shape[:2] != x.shape[:2]:
        raise ValueError("Ground truth shape must match HSI spatial dimensions")
    spectra = x.reshape(-1, x.shape[-1])
    labels = gt.reshape(-1).astype(np.int64)
    if ignore_background:
        spectra, labels = filter_background(spectra, labels)
    return spectra.astype(np.float32), labels.astype(np.int64)


def class_mask(labels: np.ndarray, classes: Iterable[int]) -> np.ndarray:
    return np.isin(labels, np.asarray(list(classes), dtype=np.int64))


def stratified_train_mask(
    labels: np.ndarray,
    classes: Iterable[int],
    train_ratio: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    train_mask = np.zeros(labels.shape[0], dtype=bool)
    ratio = float(train_ratio)
    if ratio >= 1.0:
        return class_mask(labels, classes)
    if ratio <= 0.0:
        raise ValueError("train_ratio must be positive")
    rng = np.random.default_rng(int(seed))
    for class_id in classes:
        indices = np.where(labels == int(class_id))[0]
        if indices.size == 0:
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        count = max(1, int(round(indices.size * ratio)))
        train_mask[shuffled[:count]] = True
    return train_mask


class PixelHSIDataset(Dataset):
    def __init__(self, spectra: np.ndarray, labels: np.ndarray, label_to_index: dict[int, int] | None = None):
        if len(spectra) != len(labels):
            raise ValueError("spectra and labels must have the same length")
        mapped = np.asarray(labels, dtype=np.int64)
        if label_to_index is not None:
            mapped = np.asarray([label_to_index[int(label)] for label in mapped], dtype=np.int64)
        self.spectra = torch.from_numpy(np.asarray(spectra, dtype=np.float32))
        self.labels = torch.from_numpy(mapped).long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.spectra[index], self.labels[index]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_feature_cache(path: str | Path) -> dict:
    payload = _torch_load(Path(path))
    required = {"features", "labels", "rows", "cols"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"Feature cache missing keys: {missing}")
    payload["features"] = payload["features"].float()
    payload["labels"] = payload["labels"].long()
    payload["rows"] = payload["rows"].long()
    payload["cols"] = payload["cols"].long()
    return payload


class FeatureCacheDataset(Dataset):
    def __init__(self, payload: dict):
        features = payload["features"]
        labels = payload["labels"]
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same length")
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def load_hsi_from_config(config: dict, project_root: Path):
    ds = config["dataset"]
    raw_root = project_root / ds.get("raw_dir", "data/raw")
    data_path = resolve_path(raw_root, ds["data_file"], ds.get("fallback_data_file"))
    gt_path = resolve_path(raw_root, ds["gt_file"], ds.get("fallback_gt_file"))
    x = load_mat_array(data_path, ds["data_key"], ds.get("fallback_data_key"))
    gt = load_mat_array(gt_path, ds["gt_key"], ds.get("fallback_gt_key"))
    normalization = str(config.get("data_split", {}).get("normalization", "global")).lower()
    if normalization == "seen_train":
        split = config["data_split"]
        mask = stratified_train_mask(
            gt.reshape(-1), config["classes"]["seen_classes"],
            train_ratio=float(split.get("train_ratio", 1.0)),
            seed=int(split.get("seed", config.get("runtime", {}).get("seed", 42))),
        )
        reference = x.reshape(-1, x.shape[-1])[mask]
    elif normalization == "global":
        reference = None
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization}")
    return normalize_hsi(x, reference=reference), gt.astype(np.int64)


def load_paviac_from_config(config: dict, project_root: Path):
    return load_hsi_from_config(config, project_root)


def processed_prefix(config: dict) -> str:
    return str(config.get("dataset", {}).get("name", "PaviaC"))
