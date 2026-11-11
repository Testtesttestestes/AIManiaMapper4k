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
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_features import BEAT_DIM, FeatureConfig, beat_features
from .params import PARAM_VERSION, ParamTrack

INDEX_VERSION = "hybrid-dataset-v1"
# rate augmentation: the map and the music sped up together (how players make harder versions); star ratings measured
# by rosu-pp at the clock rate (Peter on Crack 5.38* -> x1.2 6.31* -> x1.5 7.70*)
RATES: Tuple[float, ...] = (1.1, 1.2, 1.3, 1.4, 1.5)
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
        """Train/val split by song. Rate versions of a song (x1.1, x1.25 ... with their own sped-up audio file)
        and other audio files of the same piece stay on the same side (see :func:`split_key`)."""
        keys = {m["song_id"]: split_key(m) for m in self.maps}
        groups = sorted(set(keys.values()))
        rng = random.Random(seed)
        rng.shuffle(groups)
        n_val = max(1, int(round(len(groups) * val_fraction))) if len(groups) > 1 else 0
        val_groups = set(groups[:n_val])
        train = [i for i, m in enumerate(self.maps) if keys[m["song_id"]] not in val_groups]
        val = [i for i, m in enumerate(self.maps) if keys[m["song_id"]] in val_groups]
        return train, val

    def split(self, val_fraction: float = 0.1, seed: int = 42, mode: str = "piece") -> Tuple[List[int], List[int]]:
        """mode "piece" (default for new trainings) or "audio" (one group per audio file: checkpoints before it)."""
        return self.split_by_audio_file(val_fraction, seed) if mode == "audio" else self.split_by_song(val_fraction, seed)

    def split_by_audio_file(self, val_fraction: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
        """The original split (one group per audio file): used for checkpoints trained before the piece split."""
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


# rate / speed tags in titles (the content check in rate_groups.py finds the versions these miss)
_RATE_RE = re.compile(
    r"[\(\[\{]?\s*(?:"
    r"(?<![a-z0-9])[x\u00d7]\s*\d+(?:[.,]\d+)?|(?<![a-z])\d+(?:[.,]\d+)?\s*[x\u00d7](?![a-z0-9])"  # x1.1  1,25x  ×1.2 (not Mx3)
    r"|rate\s*[:=]?\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*rate"                     # rate 1.2  1.2 rate
    r"|\d{2,3}(?:[.,]\d+)?\s*%"                                                   # 130%
    r"|sped[\s-]*up|speed[\s-]*up|speedup|slowed(?:[\s-]*down)?|slow[\s-]*down"     # sped up, slowed
    r"|double[\s-]*time|half[\s-]*time|\bdt\b|\bht\b|\bnc\b|nightcore|daycore"
    r")(?:\s*ver(?:sion|\.)?)?\s*[\)\]\}]?", re.I)


def piece_name(text: str) -> str:
    """'Artist - Title (x1.25)' -> 'artist - title': rate / nightcore tags removed, case and spacing normalized."""
    t = _RATE_RE.sub(" ", text or "")
    t = re.sub(r"[^\w]+", " ", t.lower(), flags=re.UNICODE)
    return " ".join(t.split())


def split_key(entry: Dict[str, Any]) -> str:
    """Group used for the train/val split: the piece of music, not the audio file.

    Prefers the 'piece' stored by prepare_hybrid_dataset.py (artist + title without rate tags); for older
    indices it is derived from the .osu file name 'Artist - Title (Creator) [Version].osu'.
    """
    if entry.get("piece"):
        return str(entry["piece"])
    name = Path(str(entry.get("osu_file", ""))).name
    m = re.match(r"^(.*?)\s*\([^()]*\)\s*\[[^\]]*\]\.osu$", name)
    base = piece_name(m.group(1) if m else "")
    return base if base else str(entry.get("song_id"))


def rate_star_ratings(index: "HybridIndex", map_indices: Sequence[int], rates: Sequence[float] = RATES,
                      log=print) -> Dict[str, Dict[str, float]]:
    """{track file: {rate: star rating}} of every map at every clock rate (rosu-pp), cached in <root>/rate_srs.json."""
    import time as _time

    from .beatmap import ManiaMap, map_to_osu_text
    from .renderer import render_track

    cache_path = index.root / "rate_srs.json"
    cache: Dict[str, Dict[str, float]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    todo = [i for i in map_indices if index.maps[i]["track"] not in cache
            or any(f"{r:g}" not in cache[index.maps[i]["track"]] for r in rates)]
    if todo:
        try:
            import rosu_pp_py
        except ImportError:
            log("[rate] rosu-pp-py is not installed: rate augmentation is off (pip install rosu-pp-py)")
            return {}
        t0 = _time.time()
        for n, i in enumerate(todo, 1):
            e = index.maps[i]
            tr = index.track(i)
            notes = render_track(tr)
            text = map_to_osu_text(ManiaMap(notes=notes, timing_points=tr.timing_points, overall_difficulty=float(e.get("od", 8.0))))
            bm = rosu_pp_py.Beatmap(content=text)
            cache[e["track"]] = {f"{r:g}": float(rosu_pp_py.Difficulty(clock_rate=float(r)).calculate(bm).stars) for r in rates}
            if n % 200 == 0:
                log(f"  [rate] star ratings at {len(rates)} rates: {n}/{len(todo)} maps ({_time.time() - t0:.0f}s)")
        tmp = cache_path.with_name(f"rate_srs.{os.getpid()}.tmp")
        try:                            # atomic: every DDP rank computes and writes this file at the same time
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            os.replace(tmp, cache_path)
        except OSError:
            tmp.unlink(missing_ok=True)
    return cache


def scaled_timing_points(tps, start_ms: float, rate: float):
    """Red / green lines of a window that starts at ``start_ms`` and is played ``rate`` times faster."""
    from dataclasses import replace

    out = []
    for tp in tps:
        if tp.uninherited:
            out.append(replace(tp, time=start_ms + (tp.time - start_ms) / rate, beat_len=tp.beat_len / rate))
        else:
            out.append(replace(tp, time=start_ms + (tp.time - start_ms) / rate))
    return out


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
        self.pat: Optional[Dict[str, np.ndarray]] = None   # per-event pattern targets (motif_targets.py)
        self.weight: Optional[np.ndarray] = None           # per-event loss weight (self-generated maps, see set_quality)

    def set_quality(self, times_ms: np.ndarray, penalty: np.ndarray, scale: float = 3.0, floor: float = 0.05) -> None:
        """Loss weights from the per-event reward of a self-generated map (reward.note_rewards): an event with penalty
        p gets exp((p + 1) / scale), at most 1 (usual spots: p > -1) and at least ``floor`` - the network does not
        learn its own mistakes (a long one-hand run costs ~-10 -> 0.05)."""
        ev_ms = self.frames * self.frame_ms + self.time_class.astype(np.int64) - 1
        w = np.ones(self.frames.size, dtype=np.float32)
        t = np.asarray(times_ms, dtype=np.float64)
        if t.size and ev_ms.size:
            j = np.clip(np.searchsorted(t, ev_ms), 1, t.size - 1) if t.size > 1 else np.zeros(ev_ms.size, np.int64)
            if t.size > 1:
                j = np.where(np.abs(t[j - 1] - ev_ms) <= np.abs(t[j] - ev_ms), j - 1, j)
            near = np.abs(t[j] - ev_ms) <= 10
            pw = np.clip(np.exp((np.asarray(penalty, np.float64)[j] + 1.0) / scale), floor, 1.0)
            w[near] = pw[near].astype(np.float32)
        self.weight = w if (w < 1.0).any() else None

    def _frame_weight(self, rows: np.ndarray, idx: np.ndarray, length: int, radius: int = 15) -> np.ndarray:
        """Per-frame loss weight of a window: the lowest event weight within ``radius`` frames (150 ms)."""
        fw = np.ones(length, dtype=np.float32)
        if self.weight is None or not len(idx):
            return fw
        ew = self.weight[idx]
        for r, w in zip(np.asarray(rows)[ew < 1.0].tolist(), ew[ew < 1.0].tolist()):
            a, b = max(0, r - radius), min(length, r + radius + 1)
            fw[a:b] = np.minimum(fw[a:b], w)
        return fw

    def event_frames(self) -> np.ndarray:
        return self.frames

    def _plan(self, i0: int, i1: int, rows: np.ndarray, length: int, rate: float) -> Dict[str, np.ndarray]:
        """Plan of the window, per frame: technique family and copy kind of the bar each frame lies in (forward-filled
        from the events, the event before the window included; frames more than 2 s after a note are 'sparse')."""
        fam = np.zeros(length, dtype=np.int64)
        cpy = np.zeros(length, dtype=np.int64)
        if "family" not in self.pat:
            return {"plan_family": fam, "plan_copy": cpy}
        ev_rows = list(rows)
        ev_f = list(self.pat["family"][i0:i1]) if i1 > i0 else []
        ev_c = list(self.pat["copy"][i0:i1]) if i1 > i0 else []
        if i0 > 0:                                   # the bar the window starts in
            ev_rows = [-1] + ev_rows
            ev_f = [self.pat["family"][i0 - 1]] + ev_f
            ev_c = [self.pat["copy"][i0 - 1]] + ev_c
        if not ev_rows:
            return {"plan_family": fam, "plan_copy": cpy}
        ev_rows = np.asarray(ev_rows, dtype=np.int64)
        pos = np.searchsorted(ev_rows, np.arange(length), side="right") - 1
        has = pos >= 0
        far = np.zeros(length, bool)
        far[has] = (np.arange(length)[has] - ev_rows[pos[has]]) * rate > 200
        fam[has] = np.maximum(0, np.asarray(ev_f, dtype=np.int64)[pos[has]])
        cpy[has] = np.maximum(0, np.asarray(ev_c, dtype=np.int64)[pos[has]])
        fam[far] = 0
        cpy[far] = 0
        return {"plan_family": fam, "plan_copy": cpy}

    def _crop_rate(self, start: int, length: int, rate: float) -> Optional[Dict[str, np.ndarray]]:
        H = self.frame_ms
        span = int(math.ceil(length * rate)) + 1
        i0, i1 = np.searchsorted(self.frames, [start, start + span])
        ms = (self.frames[i0:i1] - start) * H + self.time_class[i0:i1].astype(np.int64) - 1
        new_ms = np.floor(ms / rate).astype(np.int64)
        keep = new_ms < length * H
        idx = np.arange(i0, i1)[keep]
        new_ms = new_ms[keep]
        rows = new_ms // H
        if rows.size and np.any(np.diff(rows) == 0):
            return None
        out = {
            "time_class": np.zeros(length, dtype=np.int64),
            "lane_code": np.full(length, -1, dtype=np.int64),
            "event_mask": np.zeros(length, dtype=np.int64),
            "stagger": np.zeros((length, 4), dtype=np.int64),
            "ln_flag": np.zeros((length, 4), dtype=np.float32),
            "ln_len": np.zeros((length, 4), dtype=np.int64),
        }
        if idx.size:
            out["time_class"][rows] = new_ms - rows * H + 1
            out["lane_code"][rows] = self.lane_code[idx]
            out["event_mask"][rows] = self.event_mask[idx]
            out["stagger"][rows] = np.clip(np.round(self.stagger[idx] / rate), 0, H - 1)
            out["ln_flag"][rows] = self.ln_flag[idx]
            out["ln_len"][rows] = np.where(self.ln_flag[idx], np.maximum(1, np.round(self.ln_len[idx] / rate)), 0)
        out["frame_weight"] = self._frame_weight(rows, idx, length)
        if self.pat is not None:
            from .motif_targets import PATTERN_HEADS
            for k in PATTERN_HEADS:
                a = np.full(length, -1, dtype=np.int64)
                a[rows] = self.pat[k][idx]
                out["pat_" + k] = a
            out.update(self._plan(int(idx[0]) if idx.size else i0, int(idx[-1]) + 1 if idx.size else i0, rows, length, rate))
            prior = np.zeros((length, 16), dtype=np.float32)
            prev = np.zeros((length, 2), dtype=np.int64)
            prior[rows] = self.pat["prior"][idx]
            prev[rows] = self.pat["prev"][idx]
            out["pat_prior"] = prior
            out["pat_prev"] = prev
        return out

    def crop_arrays(self, start: int, length: int, rate: float = 1.0) -> Optional[Dict[str, np.ndarray]]:
        """Targets of the window [start, start + length) frames; with ``rate`` > 1 the window covers
        ``length * rate`` frames of the map, played ``rate`` times faster (event times, LN lengths and staggers divided
        by the rate). None when two events of the map would fall into one frame at that speed."""
        if rate != 1.0:
            return self._crop_rate(start, length, rate)
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
        out["frame_weight"] = self._frame_weight(rows if i1 > i0 else np.zeros(0, np.int64), np.arange(i0, i1), length)
        if self.pat is not None:
            from .motif_targets import PATTERN_HEADS
            for k in PATTERN_HEADS:
                a = np.full(length, -1, dtype=np.int64)
                if i1 > i0:
                    a[rows] = self.pat[k][i0:i1]
                out["pat_" + k] = a
            out.update(self._plan(i0, i1, rows if i1 > i0 else np.zeros(0, np.int64), length, 1.0))
            prior = np.zeros((length, 16), dtype=np.float32)
            prev = np.zeros((length, 2), dtype=np.int64)
            if i1 > i0:
                prior[rows] = self.pat["prior"][i0:i1]
                prev[rows] = self.pat["prev"][i0:i1]
            out["pat_prior"] = prior
            out["pat_prev"] = prev
        return out


def sample_known_spans(T: int, rng: np.random.Generator, frame_ms: int = 10) -> np.ndarray:
    """True where a plan / canvas is given: a random share hidden in spans of 1-6 s."""
    known = np.ones(T, bool)
    need = int(rng.uniform(0.1, 0.7) * T)
    tries = 0
    while (~known).sum() < need and tries < 32:
        L = int(rng.integers(int(1000 / frame_ms), int(6000 / frame_ms) + 1))
        a = int(rng.integers(0, max(1, T - L + 1)))
        known[a:a + L] = False
        tries += 1
    return known


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
        sr_balance: float = 0.0,
        derived_weight: float = 0.3,
        pair_prob: float = 0.0,
        pitch_shift_prob: float = 0.0,
        ladder_k: int = 0,
        canvas_prob: float = 0.0,
        pattern_targets: bool = False,
        motifs=None,
        log=print,
        rate_aug: float = 0.0,
        sr_rare: float = 0.0,
        corrupt_prob: float = 0.0,
        plan_prob: float = 0.0,
        rate_max: float = 1.3,
        hard_tilt: float = 0.0,
    ):
        self.index = index
        self.map_indices = list(map_indices)
        if not self.map_indices:
            raise ValueError("empty map list")
        self.crop = int(crop_frames)
        self.train = train
        self.seed = seed
        self.augment = augment and train
        self.pitch_shift_prob = float(pitch_shift_prob) if self.augment else 0.0
        self.ladder_k = int(ladder_k)
        self.canvas_prob = float(canvas_prob)
        self.corrupt_prob = float(corrupt_prob)
        self.plan_prob = float(plan_prob)
        self.epoch = 0
        self.tracks = [SparseTrack(index.track(i)) for i in self.map_indices]
        self.entries = [index.maps[i] for i in self.map_indices]
        if pattern_targets:
            # pattern knowledge: structure targets of every map (whole map, cached) + the motif library's prior
            import time as _time
            from .motif_targets import targets_for_track
            t0 = _time.time()
            for n_done, (tr, e) in enumerate(zip(self.tracks, self.entries), 1):
                cache = index.root / "motif_targets" / (Path(str(e["track"])).stem + ".npz")
                tr.pat = targets_for_track(tr, float(e.get("sr", 0.0)), motifs, cache)
                if n_done % 200 == 0:
                    log(f"  [patterns] targets of {n_done}/{len(self.tracks)} maps ({_time.time() - t0:.0f}s)")
        n_q = 0
        for tr, e in zip(self.tracks, self.entries):
            if e.get("quality"):
                try:
                    z = np.load(index.root / e["quality"])
                    tr.set_quality(z["times"], z["penalty"])
                    n_q += tr.weight is not None
                except (OSError, KeyError, ValueError):
                    pass
        if n_q:
            log(f"[data] {n_q} self-generated maps weigh their bad spots less in the loss (per-note reward)")
        self.spans: List[Tuple[int, int]] = []
        for tr in self.tracks:
            ev = tr.event_frames()
            self.spans.append((int(ev[0]), int(ev[-1])) if ev.size else (0, tr.num_frames - 1))
        span_len = np.array([max(1, b - a + self.crop) for a, b in self.spans], dtype=np.float64)
        derived = np.array([bool(e.get("derived")) for e in self.entries])
        if sr_balance > 0:
            # rare star ratings (1-star bins) are drawn more often: weight x (maps in the bin)^-sr_balance,
            # counted separately for real and derived maps (derived maps must not dilute the real low bins)
            bins = np.array([int(np.clip(float(e.get("sr", 0.0)), 0, 9.99)) for e in self.entries]) + 10 * derived
            counts = np.bincount(bins, minlength=20).astype(np.float64)
            span_len = span_len * counts[bins] ** (-float(sr_balance))
        if hard_tilt > 0:
            # harder maps are drawn more often on top of the bin balance: weight x (1 + SR)^hard_tilt
            # (tilt 1: an 8* map ~3x as often as a 2* map of the same length, tilt 2: ~9x)
            srs = np.array([max(0.0, float(e.get("sr", 0.0))) for e in self.entries])
            span_len = span_len * (1.0 + srs) ** float(hard_tilt)
        if derived.any() and (~derived).any() and 0.0 <= derived_weight < 1.0:
            # derived (thinned) maps get a fixed share of the draws, whatever their number
            span_len = span_len.copy()
            span_len[derived] *= derived_weight / span_len[derived].sum()
            span_len[~derived] *= (1.0 - derived_weight) / span_len[~derived].sum()
        self.weights = span_len / span_len.sum()
        # rate augmentation: star ratings at every clock rate, and how rare each 1-star bin is among the real maps
        self.rate_aug = float(rate_aug) if train else 0.0
        self.rate_max = float(rate_max)
        self.rate_srs: List[Dict[str, float]] = [{} for _ in self.entries]
        if self.rate_aug > 0:
            table = rate_star_ratings(index, self.map_indices, log=log)
            self.rate_srs = [table.get(e["track"], {}) for e in self.entries]
            if not any(self.rate_srs):
                self.rate_aug = 0.0
        real_bins = [int(np.clip(float(e.get("sr", 0.0)), 0, 11.99)) for e in self.entries if not e.get("derived")]
        self.bin_counts = np.bincount(real_bins, minlength=12).astype(np.float64) if real_bins else np.ones(12)
        self.sr_rare = float(sr_rare)
        # pairs: the same window of the same song at another difficulty (teaches what changes with SR)
        self.pair_prob = float(pair_prob) if train else 0.0
        by_song: Dict[str, List[int]] = {}
        for k, e in enumerate(self.entries):
            by_song.setdefault(e["song_id"], []).append(k)
        # difficulty ladder: every map of the same song (real and derived, any star rating)
        self.song_maps: List[List[int]] = [by_song[e["song_id"]] for e in self.entries]
        self.siblings: List[List[int]] = [
            [j for j in by_song[e["song_id"]] if j != k and abs(float(self.entries[j].get("sr", 0)) - float(e.get("sr", 0))) >= 0.3]
            for k, e in enumerate(self.entries)
        ]
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

    def _item_seed(self, i: int) -> int:
        return (self.seed * 1_000_003 + self.epoch * 7_919 + i) % (2 ** 63)

    def _draw(self, rng: np.random.Generator) -> Tuple[int, int]:
        k = int(rng.choice(len(self.tracks), p=self.weights))
        a, b = self.spans[k]
        lo, hi = a - self.crop // 2, max(a - self.crop // 2 + 1, b - self.crop // 2)
        return k, int(rng.integers(lo, hi))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        pair_prev = 0.0
        pair_src = -1
        if self.fixed is not None:
            k, start = self.fixed[i]
            rng = np.random.default_rng(self.seed + i)
        else:
            rng = np.random.default_rng(self._item_seed(i))
            k, start = self._draw(rng)
            if self.pair_prob > 0 and i % 2 == 1:
                # odd items may become the partner of the previous item (same window, sibling difficulty);
                # with an even batch size both land in the same batch
                k_prev, start_prev = self._draw(np.random.default_rng(self._item_seed(i - 1)))
                if self.siblings[k_prev] and rng.random() < self.pair_prob:
                    k, start, pair_prev = int(rng.choice(self.siblings[k_prev])), start_prev, 1.0
                    pair_src = k_prev
        entry, track = self.entries[k], self.tracks[k]
        song_id = entry["song_id"]
        variant = 0
        if self.augment and self.index.num_variants(song_id) > 1 and rng.random() < 0.5:
            variant = int(rng.integers(1, self.index.num_variants(song_id)))
        feats_all = self.index.features(song_id, variant)
        n_audio = feats_all.shape[0]
        rate = self._choose_rate(pair_src if pair_src >= 0 else k, i - 1 if pair_src >= 0 else i)
        if rate != 1.0 and f"{rate:g}" not in self.rate_srs[k]:
            rate = 1.0
        tgt = track.crop_arrays(start, self.crop, rate)
        if tgt is None:                      # two events would share a frame at this speed: no augmentation
            rate = 1.0
            tgt = track.crop_arrays(start, self.crop)
        sr_label = float(entry.get("sr", 0.0)) if rate == 1.0 else float(self.rate_srs[k][f"{rate:g}"])
        H = track.frame_ms
        if rate == 1.0:
            feats = _crop_rows(feats_all, start, self.crop)
            src = np.arange(start, start + self.crop, dtype=np.float64)
            tps = track.timing_points
        else:
            # the same music played `rate` times faster: features resampled to the 10 ms grid, red lines scaled
            n_src = int(math.ceil(self.crop * rate)) + 2
            block = _crop_rows(feats_all, start, n_src)
            pos = (np.arange(self.crop) + 0.5) * rate - 0.5
            lo = np.clip(np.floor(pos).astype(np.int64), 0, n_src - 2)
            w = (pos - lo).clip(0.0, 1.0)[:, None].astype(np.float32)
            feats = block[lo] * (1.0 - w) + block[lo + 1] * w
            src = start + pos
            tps = scaled_timing_points(track.timing_points, start * H, rate)

        beat_shift = int(rng.integers(-3, 4)) if self.augment else 0  # small timing-estimate error
        beat = beat_features(tps, start + beat_shift, self.crop, H) if tps else np.zeros((self.crop, BEAT_DIM), np.float32)

        a, b = self.spans[k]
        valid = (src >= 0) & (src < max(n_audio, track.num_frames))
        valid &= (src >= a - LOSS_MARGIN_FRAMES) & (src <= b + LOSS_MARGIN_FRAMES)

        if self.augment:
            feats = self._augment(feats, rng)

        coarse = self.index.coarse(song_id)
        ladder = self._ladder(k, start, rng, rate) if self.ladder_k > 0 else None
        style = np.array(entry["style"], dtype=np.float32)
        if rate != 1.0:
            from .solver import STYLE_KEYS
            style[STYLE_KEYS.index("sr")] = sr_label / 10.0
            j = STYLE_KEYS.index("nps")
            nps = math.expm1(float(style[j]) * math.log1p(30.0))
            style[j] = math.log1p(nps * rate) / math.log1p(30.0)
        item = {
            "feats": torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
            "beat": torch.from_numpy(beat),
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "sr": torch.tensor([sr_label], dtype=torch.float32),
            "rate": torch.tensor([rate], dtype=torch.float32),
            "augmented": torch.tensor([1.0 if rate != 1.0 else 0.0], dtype=torch.float32),
            # rare star ratings weigh more in the loss: (real maps in the 1-star bin of this crop's rating)^-sr_rare
            "loss_weight": torch.tensor([float(max(1.0, self.bin_counts[int(np.clip(sr_label, 0, 11.99))]) ** -self.sr_rare)],
                                        dtype=torch.float32),
            "style": torch.from_numpy(style),
            "style_mask": torch.tensor(entry["style_mask"], dtype=torch.float32),
            "coarse": torch.from_numpy(coarse),
            "pair_prev": torch.tensor([pair_prev], dtype=torch.float32),
        }
        for key, arr in tgt.items():
            item[key] = torch.from_numpy(arr)
        if "plan_family" in tgt:
            # the plan as input: known in a share of the crops (whole, or with unknown spans), otherwise all unknown -
            # the network learns to map from a plan AND without one (then its plan head proposes one)
            from .motif_targets import N_PLAN, N_PLAN_COPY
            pf = np.full(self.crop, N_PLAN, dtype=np.int64)
            pc = np.full(self.crop, N_PLAN_COPY, dtype=np.int64)
            if self.plan_prob > 0 and rng.random() < self.plan_prob:
                pf, pc = tgt["plan_family"].copy(), tgt["plan_copy"].copy()
                if rng.random() < 0.5:
                    hide = ~sample_known_spans(self.crop, rng)
                    pf[hide], pc[hide] = N_PLAN, N_PLAN_COPY
            item["plan_in_family"], item["plan_in_copy"] = torch.from_numpy(pf), torch.from_numpy(pc)
        if self.canvas_prob > 0:
            from .canvas import MASK, canvas_from_targets, corrupt_canvas, sample_known
            wrong = np.zeros(self.crop, bool)
            if self.corrupt_prob > 0 and rng.random() < self.corrupt_prob:
                # "break and fix": almost the whole map is shown, part of it broken; every frame is predicted
                known = rng.random(self.crop) >= rng.uniform(0.0, 0.3)
                cs, cm, cl = canvas_from_targets(tgt, known)
                cs, cm, cl, wrong = corrupt_canvas(cs, cm, cl, rng, track.frame_ms)
                train_rows = np.ones(self.crop, np.float32)
            else:
                known = sample_known(self.crop, rng, track.frame_ms) if rng.random() < self.canvas_prob else np.zeros(self.crop, bool)
                cs, cm, cl = canvas_from_targets(tgt, known)
                train_rows = (cs == MASK).astype(np.float32)
            item["canvas_state"], item["canvas_ms"], item["canvas_ln"] = torch.from_numpy(cs), torch.from_numpy(cm), torch.from_numpy(cl)
            item["canvas_wrong"] = torch.from_numpy(wrong.astype(np.float32))
            item["canvas_train"] = torch.from_numpy(train_rows)
        if ladder is not None:
            item["ladder_presence"] = torch.from_numpy(ladder[0])
            item["ladder_sr"] = torch.from_numpy(ladder[1])
        return item

    def _choose_rate(self, k: int, i: int) -> float:
        """A clock rate for item i (1.0 = none). Items 2j and 2j+1 draw the same rate (pairs stay comparable). Rates
        whose star rating lands in a rare bin are preferred: the augmentation adds hard maps where they are missing."""
        if self.rate_aug <= 0 or not self.rate_srs[k] or self.fixed is not None:
            return 1.0
        rng = np.random.default_rng(self._item_seed(i - i % 2) + 7_777)
        if rng.random() >= self.rate_aug:
            return 1.0
        rates = [r for r in RATES if r <= self.rate_max + 1e-9 and f"{r:g}" in self.rate_srs[k]]
        if not rates:
            return 1.0
        srs = np.array([self.rate_srs[k][f"{r:g}"] for r in rates])
        rarity = 1.0 / np.maximum(1.0, self.bin_counts[np.clip(srs.astype(int), 0, 11)])
        rarity = np.where(srs <= 11.0, rarity, 0.0)
        if rarity.sum() <= 0:
            return 1.0
        return float(rates[int(rng.choice(len(rates), p=rarity / rarity.sum()))])

    def _ladder(self, k: int, start: int, rng: np.random.Generator, rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """[K, T] whether each of up to K maps of the same song (this one first) has a note starting in each frame
        (+-1 frame, different mappers place one hit a few ms apart; -1 outside that map's notes) and [K] their star
        ratings (-1 = no map)."""
        K = self.ladder_k
        pres = np.zeros((K, self.crop), dtype=np.float32)
        srs = np.full(K, -1.0, dtype=np.float32)
        others = [j for j in self.song_maps[k] if j != k]
        if len(others) > K - 1:
            others = list(rng.choice(others, size=K - 1, replace=False))
        for row, j in enumerate([k] + others):
            arr = self.tracks[j].crop_arrays(start, self.crop, rate)
            if arr is None or (rate != 1.0 and f"{rate:g}" not in self.rate_srs[j]):
                continue                                   # that map cannot be played at this rate: unknown row
            tc = arr["time_class"] > 0
            p = tc.copy()
            p[1:] |= tc[:-1]
            p[:-1] |= tc[1:]
            # outside that map's own first..last note it says nothing about the moment (-1 = unknown)
            fa, fb = self.spans[j]
            frames = start + (np.arange(self.crop) + 0.5) * rate - 0.5
            p = np.where((frames >= fa) & (frames <= fb), p.astype(np.float32), -1.0)
            pres[row] = p
            srs[row] = (float(self.entries[j].get("sr", 0.0)) if rate == 1.0 else float(self.rate_srs[j][f"{rate:g}"])) or -1.0
        return pres, srs

    def _augment(self, feats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.pitch_shift_prob > 0 and rng.random() < self.pitch_shift_prob:
            feats = pitch_shift_features(feats, self.slices, int(rng.choice([-2, -1, 1, 2])))
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


def pitch_shift_features(feats: np.ndarray, slices: Dict[str, Tuple[int, int]], semitones: int) -> np.ndarray:
    """Transposes the pitch-aware representations by a few semitones (the map stays the same).

    A transposed song needs the same rhythm and the same patterns, but it no longer matches the memorized
    spectrum of a training song: this is the cheapest way to stop the network from recognizing songs (the
    reason validation peaks after one or two epochs and then overfits). ``pitch`` (semitone bins) is shifted
    exactly, ``chroma`` is rotated, the mel bands are shifted by the nearest number of bands.
    """
    if semitones == 0:
        return feats
    out = feats.copy()

    def shift(a: int, b: int, k: int) -> None:
        if k == 0 or b - a <= abs(k):
            return
        block = feats[:, a:b]
        res = np.empty_like(block)
        if k > 0:
            res[:, k:] = block[:, :-k]
            res[:, :k] = block[:, :1]
        else:
            res[:, :k] = block[:, -k:]
            res[:, k:] = block[:, -1:]
        out[:, a:b] = res

    if "pitch" in slices:
        a, b = slices["pitch"]
        shift(a, b, semitones)
    if "chroma" in slices:
        a, b = slices["chroma"]
        out[:, a:b] = np.roll(feats[:, a:b], semitones, axis=1)
    for name, per_semitone in (("mel_fine", 0.35), ("mel_mid", 0.5)):  # mel bands are ~2-3 semitones wide
        if name in slices:
            a, b = slices[name]
            shift(a, b, int(round(semitones * per_semitone)))
    return out


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
