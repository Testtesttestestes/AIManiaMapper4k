"""
Training data for the hybrid network: random fixed-length crops of (features, parameter targets).

Layout produced by prepare_hybrid_dataset.py:

    <root>/index.json                       songs, maps, feature config, versions
    <root>/features/<song_id>.npy           float16 [F, D] stacked normalized representations
    <root>/features/<song_id>.<variant>.npy same for degraded audio quality variants (optional)
    <root>/features/<song_id>.coarse.npy    float16 [F/100, D] 1 s summaries (global context)
    <root>/maps/<map_id>.npz                exact ParamTrack of one difficulty

Splits are made by SONG (all difficulties of one audio go to the same side), otherwise
validation would contain the training audio and look much better than reality.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_features import BEAT_DIM, FeatureConfig, beat_features
from .params import PARAM_VERSION, ParamTrack

INDEX_VERSION = "hybrid-dataset-v1"
LOSS_MARGIN_FRAMES = 200  # frames before the first / after the last note that still count as "no note here"


class HybridIndex:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        path = self.root / "index.json"
        if not path.exists():
            raise FileNotFoundError(f"'{path}' not found. Run: python prepare_hybrid_dataset.py --output_dir {self.root}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != INDEX_VERSION:
            raise ValueError(f"dataset index version {data.get('version')} != {INDEX_VERSION}; re-run prepare_hybrid_dataset.py")
        if data.get("param_version") != PARAM_VERSION:
            raise ValueError(f"dataset param version {data.get('param_version')} != {PARAM_VERSION}; re-run prepare_hybrid_dataset.py")
        self.data = data
        self.feature_config = FeatureConfig.from_dict(data["feature_config"])
        self.songs: Dict[str, Dict[str, Any]] = data["songs"]
        self.maps: List[Dict[str, Any]] = [m for m in data["maps"] if m.get("roundtrip_exact", False)]
        self._feat_cache: Dict[str, np.ndarray] = {}
        self._coarse_cache: Dict[str, np.ndarray] = {}

    def __getstate__(self) -> Dict[str, Any]:
        # DataLoader workers (spawned on Windows) must not receive memory-mapped arrays: they re-open lazily
        state = dict(self.__dict__)
        state["_feat_cache"] = {}
        state["_coarse_cache"] = {}
        return state

    def split_by_song(self, val_fraction: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
        songs = sorted({m["song_id"] for m in self.maps})
        rng = random.Random(seed)
        rng.shuffle(songs)
        n_val = max(1, int(round(len(songs) * val_fraction))) if len(songs) > 1 else 0
        val_songs = set(songs[:n_val])
        train = [i for i, m in enumerate(self.maps) if m["song_id"] not in val_songs]
        val = [i for i, m in enumerate(self.maps) if m["song_id"] in val_songs]
        return train, val

    def features(self, song_id: str, variant: int = 0) -> np.ndarray:
        files = self.songs[song_id]["features"]
        key = f"{song_id}:{variant}"
        if key not in self._feat_cache:
            self._feat_cache[key] = np.load(self.root / files[variant % len(files)], mmap_mode="r")
        return self._feat_cache[key]

    def num_variants(self, song_id: str) -> int:
        return len(self.songs[song_id]["features"])

    def coarse(self, song_id: str) -> np.ndarray:
        if song_id not in self._coarse_cache:
            self._coarse_cache[song_id] = np.load(self.root / self.songs[song_id]["coarse"]).astype(np.float32)
        return self._coarse_cache[song_id]

    def track(self, map_idx: int) -> ParamTrack:
        return ParamTrack.load_npz(self.root / self.maps[map_idx]["track"])


class SparseTrack:
    """Event-only copy of a ParamTrack (~20x smaller in RAM than the dense arrays)."""

    def __init__(self, track: ParamTrack):
        ev = track.event_frames()
        self.frames = ev.astype(np.int64)
        self.num_frames = track.num_frames
        self.frame_ms = track.frame_ms
        self.timing_points = track.timing_points
        self.time_class = track.time_class[ev].astype(np.int8)
        self.lane_code = track.lane_code[ev].astype(np.int8)
        self.event_mask = track.event_mask[ev].astype(np.uint8)
        self.stagger = track.stagger[ev].astype(np.int8)
        self.ln_flag = track.ln_flag[ev].astype(bool)
        self.ln_len = track.ln_len[ev].astype(np.int32)

    def event_frames(self) -> np.ndarray:
        return self.frames

    def crop_arrays(self, start: int, length: int) -> Dict[str, np.ndarray]:
        out = {
            "time_class": np.zeros(length, dtype=np.int64),
            "lane_code": np.full(length, -1, dtype=np.int64),
            "event_mask": np.zeros(length, dtype=np.int64),
            "stagger": np.zeros((length, 4), dtype=np.int64),
            "ln_flag": np.zeros((length, 4), dtype=np.float32),
            "ln_len": np.zeros((length, 4), dtype=np.int64),
        }
        i0, i1 = np.searchsorted(self.frames, [start, start + length])
        if i1 > i0:
            rows = self.frames[i0:i1] - start
            out["time_class"][rows] = self.time_class[i0:i1]
            out["lane_code"][rows] = self.lane_code[i0:i1]
            out["event_mask"][rows] = self.event_mask[i0:i1]
            out["stagger"][rows] = self.stagger[i0:i1]
            out["ln_flag"][rows] = self.ln_flag[i0:i1]
            out["ln_len"][rows] = self.ln_len[i0:i1]
        return out


def _crop_rows(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    out = np.zeros((length,) + arr.shape[1:], dtype=np.float32)
    s0, s1 = max(0, start), min(arr.shape[0], start + length)
    if s1 > s0:
        out[s0 - start: s1 - start] = arr[s0:s1]
    return out


class HybridCropDataset(Dataset):
    def __init__(
        self,
        index: HybridIndex,
        map_indices: Sequence[int],
        crop_frames: int = 2048,
        samples_per_epoch: int = 8000,
        train: bool = True,
        seed: int = 0,
        num_fixed_crops: int = 0,
        augment: bool = True,
    ):
        self.index = index
        self.map_indices = list(map_indices)
        if not self.map_indices:
            raise ValueError("empty map list")
        self.crop = int(crop_frames)
        self.train = train
        self.seed = seed
        self.augment = augment and train
        self.epoch = 0
        self.tracks = [SparseTrack(index.track(i)) for i in self.map_indices]
        self.entries = [index.maps[i] for i in self.map_indices]
        self.spans: List[Tuple[int, int]] = []
        for tr in self.tracks:
            ev = tr.event_frames()
            self.spans.append((int(ev[0]), int(ev[-1])) if ev.size else (0, tr.num_frames - 1))
        span_len = np.array([max(1, b - a + self.crop) for a, b in self.spans], dtype=np.float64)
        self.weights = span_len / span_len.sum()
        self.samples_per_epoch = int(samples_per_epoch)
        self.fixed: Optional[List[Tuple[int, int]]] = None
        if not train:
            n = num_fixed_crops or samples_per_epoch
            rng = np.random.default_rng(seed + 12345)
            self.fixed = [self._draw(rng) for _ in range(n)]
        self.slices = index.feature_config.slices()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.fixed) if self.fixed is not None else self.samples_per_epoch

    def _draw(self, rng: np.random.Generator) -> Tuple[int, int]:
        k = int(rng.choice(len(self.tracks), p=self.weights))
        a, b = self.spans[k]
        lo, hi = a - self.crop // 2, max(a - self.crop // 2 + 1, b - self.crop // 2)
        return k, int(rng.integers(lo, hi))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if self.fixed is not None:
            k, start = self.fixed[i]
            rng = np.random.default_rng(self.seed + i)
        else:
            rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 7_919 + i) % (2 ** 63))
            k, start = self._draw(rng)
        entry, track = self.entries[k], self.tracks[k]
        song_id = entry["song_id"]
        variant = 0
        if self.augment and self.index.num_variants(song_id) > 1 and rng.random() < 0.5:
            variant = int(rng.integers(1, self.index.num_variants(song_id)))
        feats_all = self.index.features(song_id, variant)
        n_audio = feats_all.shape[0]
        feats = _crop_rows(feats_all, start, self.crop)

        tps = track.timing_points
        beat_shift = int(rng.integers(-3, 4)) if self.augment else 0  # small timing-estimate error
        beat = beat_features(tps, start + beat_shift, self.crop, track.frame_ms) if tps else np.zeros((self.crop, BEAT_DIM), np.float32)

        frames = np.arange(start, start + self.crop)
        a, b = self.spans[k]
        valid = (frames >= 0) & (frames < max(n_audio, track.num_frames))
        valid &= (frames >= a - LOSS_MARGIN_FRAMES) & (frames <= b + LOSS_MARGIN_FRAMES)

        if self.augment:
            feats = self._augment(feats, rng)

        tgt = track.crop_arrays(start, self.crop)
        coarse = self.index.coarse(song_id)
        item = {
            "feats": torch.from_numpy(feats),
            "beat": torch.from_numpy(beat),
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "sr": torch.tensor([float(entry.get("sr", 0.0))], dtype=torch.float32),
            "style": torch.tensor(entry["style"], dtype=torch.float32),
            "style_mask": torch.tensor(entry["style_mask"], dtype=torch.float32),
            "coarse": torch.from_numpy(coarse),
        }
        for key, arr in tgt.items():
            item[key] = torch.from_numpy(arr)
        return item

    def _augment(self, feats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for name, (s0, s1) in self.slices.items():
            if rng.random() < 0.5:
                feats[:, s0:s1] = feats[:, s0:s1] * rng.uniform(0.85, 1.15) + rng.uniform(-0.25, 0.25)
            if name in ("mel_fine", "mel_mid", "pitch") and rng.random() < 0.3:
                w = int(rng.integers(1, max(2, (s1 - s0) // 8)))
                f0 = int(rng.integers(s0, s1 - w))
                feats[:, f0:f0 + w] = 0.0
        if rng.random() < 0.5:
            feats += rng.normal(0.0, 0.05, size=feats.shape).astype(np.float32)
        return feats


def collate_hybrid(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for key in items[0]:
        if key == "coarse":
            continue
        batch[key] = torch.stack([it[key] for it in items])
    max_c = max(it["coarse"].shape[0] for it in items)
    D = items[0]["coarse"].shape[1]
    coarse = torch.zeros(len(items), max_c, D)
    mask = torch.zeros(len(items), max_c, dtype=torch.bool)
    for i, it in enumerate(items):
        n = it["coarse"].shape[0]
        coarse[i, :n] = it["coarse"]
        mask[i, :n] = True
    batch["coarse"] = coarse
    batch["coarse_mask"] = mask
    return batch
