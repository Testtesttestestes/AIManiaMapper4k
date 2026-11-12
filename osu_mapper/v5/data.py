"""
Training windows for the v5 network, from the same dataset as the hybrid one (prepare_hybrid_dataset.py; best with
--external demucs_stems mert95m: the source selector and the cause labels need the stems).

Every item: a window of ``window_frames`` x 10 ms of the song (features + the stems' attack curves + the beat grid),
the event tokens of a human map in it (osu_mapper/v5/tokens.py), the beat phase of every TIME token, the frame of
every token's event (the decoder adds the encoder state there), the cause of every event frame, the map's star rating
and style. Maps are drawn by length, rarer star ratings more often (``sr_balance``), harder ones more often
(``hard_tilt``), and - "learn where it errs" (hard example mining) - maps on which the network's loss is high more
often (``set_hard_weights``, updated by train_v5.py after every epoch).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..hybrid.audio_features import BEAT_DIM, beat_features
from ..hybrid.causes import STEMS, note_causes, stem_attacks
from ..hybrid.dataset import HybridIndex, _crop_rows
from ..hybrid.metrics import SUBDIV_NAMES, subdivision_classes
from ..hybrid.renderer import render_track
from .sections import SECTION_DIM, find_sections, section_features
from .tokens import Vocab, encode

PHASE_NONE = len(SUBDIV_NAMES)


def group_columns(slices: Dict[str, Tuple[int, int]], sections: bool = True) -> Dict[str, int]:
    """Input groups of the source selector, in column order of ``feats + attacks + sections``: every representation,
    the four demucs stems separately, the stems' attack curves ("attacks", 4 columns after the features) and the part
    of the song ("section", osu_mapper/v5/sections.py, last)."""
    out: Dict[str, int] = {}
    for name, (a, b) in sorted(slices.items(), key=lambda kv: kv[1][0]):
        if name == "demucs_stems" and b - a == 128:
            for s in STEMS:
                out[f"stem_{s}"] = 32
        else:
            out[name] = b - a
    if "demucs_stems" in slices:
        out["attacks"] = 4
    if sections:
        out["section"] = SECTION_DIM
    return out


def song_inputs(feats: np.ndarray, slices: Dict[str, Tuple[int, int]], timing_points, sections: bool = True,
                frame_ms: int = 10, highlight=None) -> np.ndarray:
    """[F, D] network input of a whole song: features + stem attacks (+ the part of the song of every frame)."""
    parts = [np.asarray(feats, dtype=np.float32)]
    att = stem_attacks(feats, slices)
    if att is not None:
        parts.append(att)
    if sections:
        parts.append(section_features(find_sections(feats, slices, timing_points, frame_ms, highlight=highlight),
                                      feats.shape[0]))
    return np.concatenate(parts, axis=1)


class SongCache:
    """Per-song inputs that need the whole song: stem attack curves (robust statistics) and its parts (sections)."""

    def __init__(self, index: HybridIndex, sections: bool = True, highlight=None):
        self.index = index
        self.highlight = list(highlight or [])
        self.slices = {k: tuple(v) for k, v in (index.data.get("feature_slices") or index.feature_config.slices()).items()}
        self.use_sections = bool(sections)
        self._att: Dict[str, Optional[np.ndarray]] = {}
        self._sec: Dict[str, np.ndarray] = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_att"] = {}
        d["_sec"] = {}
        return d

    def sections(self, song_id: str, timing_points) -> np.ndarray:
        """[F, SECTION_DIM] of the song (with the red lines of the first map that asks for it)."""
        if song_id not in self._sec:
            feats = np.asarray(self.index.features(song_id, 0))
            secs = find_sections(feats, self.slices, timing_points, highlight=self.highlight)
            self._sec[song_id] = section_features(secs, feats.shape[0])
        return self._sec[song_id]

    def attacks(self, song_id: str) -> Optional[np.ndarray]:
        if song_id not in self._att:
            self._att[song_id] = stem_attacks(np.asarray(self.index.features(song_id, 0)), self.slices)
        return self._att[song_id]

    def window(self, song_id: str, start: int, length: int, timing_points=None, sections: Optional[bool] = None) -> np.ndarray:
        parts = [_crop_rows(self.index.features(song_id, 0), start, length)]
        att = self.attacks(song_id)
        if att is not None:
            parts.append(_crop_rows(att, start, length))
        if self.use_sections if sections is None else sections:
            parts.append(_crop_rows(self.sections(song_id, timing_points), start, length))
        return np.concatenate(parts, axis=1).astype(np.float32)


def window_item(cache: SongCache, entry: Dict, notes, timing_points, start_frame: int, vocab: Vocab, W: int,
                frame_ms: int = 10, style_known: bool = True) -> Dict[str, torch.Tensor]:
    start_ms = start_frame * frame_ms
    toks, ev_pos, _ = encode(notes, start_ms, vocab)
    toks = np.asarray(toks, dtype=np.int64)
    is_time = vocab.is_time(toks)
    t_in = np.where(is_time, toks - vocab.time0, -1)
    cur = np.maximum.accumulate(np.where(is_time, np.arange(len(toks)), -1))
    tok_ms = np.where(cur >= 0, t_in[np.maximum(cur, 0)], -1)
    tok_frame = np.where(tok_ms >= 0, tok_ms // frame_ms, -1)
    phase = np.full(len(toks), PHASE_NONE, dtype=np.int64)
    if is_time.any() and timing_points:
        phase[is_time] = subdivision_classes(start_ms + t_in[is_time], timing_points)
    cause = np.full(W, -1, dtype=np.int64)
    ev_ms = t_in[is_time]
    if ev_ms.size:
        c = note_causes(start_ms + ev_ms, cache.attacks(entry["song_id"]), frame_ms)
        cause[np.clip(ev_ms // frame_ms, 0, W - 1)] = c
    feats = cache.window(entry["song_id"], start_frame, W, timing_points)
    beat = beat_features(timing_points, start_frame, W, frame_ms) if timing_points else np.zeros((W, BEAT_DIM), np.float32)
    style = np.asarray(entry["style"], dtype=np.float32)
    smask = np.asarray(entry["style_mask"], dtype=np.float32) if style_known else np.zeros_like(style)
    return {
        "feats": torch.from_numpy(feats), "beat": torch.from_numpy(beat.astype(np.float32)),
        "tokens": torch.from_numpy(toks), "phase": torch.from_numpy(phase), "tok_frame": torch.from_numpy(tok_frame),
        "is_time": torch.from_numpy(is_time.astype(np.int64)), "cause": torch.from_numpy(cause),
        "sr": torch.tensor([float(entry.get("sr", 0.0))], dtype=torch.float32),
        "style": torch.from_numpy(style), "style_mask": torch.from_numpy(smask),
    }


def collate(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    L = max(int(it["tokens"].shape[0]) for it in items)
    for k in items[0]:
        if k in ("tokens", "phase", "tok_frame", "is_time"):
            pad = {"tokens": 0, "phase": PHASE_NONE, "tok_frame": -1, "is_time": 0}[k]
            out[k] = torch.stack([torch.nn.functional.pad(it[k], (0, L - it[k].shape[0]), value=pad) for it in items])
        else:
            out[k] = torch.stack([it[k] for it in items])
    return out


class V5Dataset(Dataset):
    def __init__(self, index: HybridIndex, map_indices: Sequence[int], vocab: Vocab, window_frames: int,
                 samples: int, seed: int = 0, train: bool = True, sr_balance: float = 0.5, hard_tilt: float = 0.0,
                 hard_share: float = 0.0, style_known_prob: float = 0.5, fixed: Optional[List[Tuple[int, int]]] = None,
                 max_tokens: int = 1536, sections: bool = True, highlight=None):
        self.index = index
        self.cache = SongCache(index, sections, highlight)
        self.map_indices = list(map_indices)
        self.vocab = vocab
        self.W = int(window_frames)
        self.samples = int(samples)
        self.seed = int(seed)
        self.train = train
        self.style_known_prob = float(style_known_prob)
        self.max_tokens = int(max_tokens)
        self.epoch = 0
        self.entries = [index.maps[i] for i in self.map_indices]
        self.notes = []
        self.tps = []
        self.spans: List[Tuple[int, int]] = []
        for i in self.map_indices:
            tr = index.track(i)
            n = sorted(render_track(tr))
            self.notes.append(n)
            self.tps.append(list(tr.timing_points))
            self.spans.append((n[0].start // 10, n[-1].start // 10) if n else (0, 0))
        span_len = np.array([max(1, b - a + self.W) for a, b in self.spans], dtype=np.float64)
        srs = np.array([max(0.0, float(e.get("sr", 0.0))) for e in self.entries])
        if sr_balance > 0:
            bins = np.clip(srs, 0, 9.99).astype(np.int64)
            counts = np.bincount(bins, minlength=10).astype(np.float64)
            span_len = span_len * counts[bins] ** (-float(sr_balance))
        if hard_tilt > 0:
            span_len = span_len * (1.0 + srs) ** float(hard_tilt)
        self.base = span_len / span_len.sum()
        self.hard = np.ones(len(self.entries))
        self.hard_share = float(hard_share)
        self.weights = self.base.copy()
        self.fixed = fixed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_hard_weights(self, per_map_loss: np.ndarray, gamma: float = 1.0) -> None:
        """Learn where it errs: a share ``hard_share`` of the draws goes to maps by (their loss / mean loss)^gamma."""
        loss = np.asarray(per_map_loss, dtype=np.float64)
        ok = np.isfinite(loss) & (loss > 0)
        if not ok.any() or self.hard_share <= 0:
            return
        rel = np.ones_like(loss)
        rel[ok] = (loss[ok] / loss[ok].mean()) ** float(gamma)
        self.hard = np.clip(rel, 0.25, 4.0)
        hard = self.base * self.hard
        self.weights = (1 - self.hard_share) * self.base + self.hard_share * hard / hard.sum()

    def __len__(self) -> int:
        return len(self.fixed) if self.fixed is not None else self.samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 7919 + i)
        if self.fixed is not None:
            k, start = self.fixed[i]
        else:
            k = int(np.searchsorted(np.cumsum(self.weights), rng.random() * self.weights.sum()))
            k = min(k, len(self.entries) - 1)
            a, b = self.spans[k]
            start = rng.randint(a - self.W // 4, max(a - self.W // 4 + 1, b - self.W // 2))
        start = max(0, start)
        item = window_item(self.cache, self.entries[k], self.notes[k], self.tps[k], start, self.vocab, self.W,
                           style_known=self.train and rng.random() < self.style_known_prob)
        if item["tokens"].shape[0] > self.max_tokens:            # extremely dense window: keep the first part
            for key in ("tokens", "phase", "tok_frame", "is_time"):
                item[key] = item[key][: self.max_tokens]
        item["map_k"] = torch.tensor([k], dtype=torch.int64)
        item["item_weight"] = torch.tensor([1.0], dtype=torch.float32)
        return item

    @staticmethod
    def fixed_windows(index: HybridIndex, map_indices: Sequence[int], W: int, per_map: int, seed: int = 0):
        rng = random.Random(seed)
        out = []
        for k, i in enumerate(map_indices):
            n = render_track(index.track(i))
            if not n:
                continue
            a, b = min(x.start for x in n) // 10, max(x.start for x in n) // 10
            for _ in range(per_map):
                out.append((k, rng.randint(max(0, a - W // 4), max(a, b - W // 2))))
        return out
