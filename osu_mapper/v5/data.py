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

import hashlib
import os
import random
from pathlib import Path
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
    """Per-song inputs that need the whole song: stem attack curves (robust statistics) and its parts (sections).

    They are computed once and kept as .npy files in ``<dataset>/v5_song_cache`` (memory-mapped): the DataLoader
    workers of every GPU then share the same pages instead of each holding its own copy of every song it has drawn
    (4 workers x 3 GPUs x ~1000 songs x ~2 MB ran the machine out of RAM). ``disk=False`` keeps them in memory only."""

    def __init__(self, index: HybridIndex, sections: bool = True, highlight=None, disk: bool = True):
        self.index = index
        self.highlight = list(highlight or [])
        self.slices = {k: tuple(v) for k, v in (index.data.get("feature_slices") or index.feature_config.slices()).items()}
        self.use_sections = bool(sections)
        self.disk = bool(disk)
        self._att: Dict[str, Optional[np.ndarray]] = {}
        self._sec: Dict[str, np.ndarray] = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_att"] = {}
        d["_sec"] = {}
        return d

    def _file(self, song_id: str, kind: str, extra: str = "") -> Optional[Path]:
        if not self.disk:
            return None
        root = getattr(self.index, "root", None)
        if root is None:
            return None
        files = self.index.songs.get(song_id, {}).get("features") or [""]
        h = hashlib.sha1(f"{song_id}|{files[0]}|{kind}|{extra}".encode("utf-8")).hexdigest()[:20]
        return Path(root) / "v5_song_cache" / f"{kind}_{h}.npy"

    @staticmethod
    def _load_or_make(path: Optional[Path], make):
        if path is not None and path.exists():
            try:
                return np.load(path, mmap_mode="r")
            except (OSError, ValueError):
                pass                                    # half-written or damaged: computed again
        arr = make()
        if path is None or arr is None:
            return arr
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
            np.save(tmp, np.ascontiguousarray(arr, dtype=np.float32))
            os.replace(tmp, path)                       # atomic: other workers see the whole file or none
            return np.load(path, mmap_mode="r")
        except OSError:
            return arr                                  # read-only dataset folder: kept in memory

    def sections(self, song_id: str, timing_points) -> np.ndarray:
        """[F, SECTION_DIM] of the song (with the red lines of the map that asks for it)."""
        tps = [t.as_tuple() if hasattr(t, "as_tuple") else tuple(t) for t in (timing_points or [])]
        key = song_id + "|" + hashlib.sha1(repr((tps, self.highlight)).encode("utf-8")).hexdigest()[:16]
        if key not in self._sec:
            def make():
                feats = np.asarray(self.index.features(song_id, 0))
                secs = find_sections(feats, self.slices, timing_points, highlight=self.highlight)
                return section_features(secs, feats.shape[0])
            self._sec[key] = self._load_or_make(self._file(song_id, "sec", key), make)
        return self._sec[key]

    def attacks(self, song_id: str) -> Optional[np.ndarray]:
        if song_id not in self._att:
            if "demucs_stems" not in self.slices:
                self._att[song_id] = None
            else:
                self._att[song_id] = self._load_or_make(
                    self._file(song_id, "att"), lambda: stem_attacks(np.asarray(self.index.features(song_id, 0)), self.slices))
        return self._att[song_id]

    def window(self, song_id: str, start: int, length: int, timing_points=None, sections: Optional[bool] = None) -> np.ndarray:
        parts = [_crop_rows(self.index.features(song_id, 0), start, length)]
        att = self.attacks(song_id)
        if att is not None:
            parts.append(_crop_rows(att, start, length))
        if self.use_sections if sections is None else sections:
            parts.append(_crop_rows(self.sections(song_id, timing_points), start, length))
        return np.concatenate(parts, axis=1).astype(np.float32)


def make_draft(notes, timing_points, rng: random.Random) -> list:
    """A draft of the human map for training the re-mapping passes: one to three kinds of damage (the bursts collapsed
    to 1/2, the critic's damage - lanes, timing, missing / extra notes, long notes made taps / taps made long notes /
    lengths scrambled / releases moved - a span of 1-4 s cut out, every other event dropped); the network learns to
    map the song again from it, keeping what is right and fixing the rest."""
    from .critic import blandify, corrupt

    d = list(notes)
    for _ in range(rng.randint(1, 3)):
        k = rng.random()
        if k < 0.25:
            d = blandify(d, timing_points)
        elif k < 0.75:
            d = corrupt(d, rng)
        elif k < 0.9 and d:
            a = rng.choice(d).start
            L = rng.randint(1000, 4000)
            d = [n for n in d if not (a <= n.start < a + L)]
        else:
            ts = sorted({n.start for n in d})
            keep = set(ts[::2])
            d = [n for n in d if n.start in keep]
    return d


def draft_frames(draft_notes, start_ms: int, W: int, frame_ms: int = 10) -> np.ndarray:
    """[W, MAP_DIM + 1]: picture of the draft (critic.map_frames) + a column of ones ("a draft is given")."""
    from .critic import map_frames

    x = map_frames(draft_notes, start_ms, W, frame_ms)
    return np.concatenate([x, np.ones((W, 1), np.float32)], axis=1)


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


def augment_notes(notes, mirror: bool = False, rotate: int = 0) -> list:
    """Lanes mirrored (1234 -> 4321) and / or rotated (rotate = 1: 1234 -> 4123). Star ratings stay the same (the
    mania star rating treats the columns alike; checked in tests/test_v5.py)."""
    from ..hybrid.beatmap import Note as _N

    if not mirror and not rotate:
        return list(notes)
    out = []
    for n in notes:
        c = int(n.col)
        if mirror:
            c = 3 - c
        c = (c + rotate) % 4
        out.append(_N(n.start, c, n.end, n.ln))
    return out


def _shifted(timing_points, ms: int):
    from ..hybrid.beatmap import RawTimingPoint as _R

    return [_R(tp.time - ms, tp.beat_len, tp.meter, tp.sample_set, tp.sample_index, tp.volume, tp.uninherited, tp.effects)
            for tp in timing_points]


def hide_frames(W: int, ratio: float, rng: random.Random) -> np.ndarray:
    """[W] bool: frames under the mask - at random or (half the time) as spans of 0.2 - 3 s (inpainting a phrase)."""
    k = int(round(min(1.0, max(0.0, ratio)) * W))
    hidden = np.zeros(W, dtype=bool)
    if k <= 0:
        return hidden
    if rng.random() < 0.5:
        hidden[np.array(rng.sample(range(W), k))] = True
        return hidden
    while hidden.sum() < k:
        L = rng.randint(20, 300)
        a = rng.randint(0, max(0, W - 1))
        hidden[a:a + L] = True
    return hidden


def canvas_item(cache: "SongCache", entry: Dict, notes, timing_points, start_frame: int, W: int, rng: random.Random,
                style_known: bool, frame_ms: int = 10, mirror: bool = False, rotate: int = 0, shift_ms: int = 0,
                p_empty: float = 0.1, p_fix: float = 0.25, seam: bool = False) -> Dict[str, torch.Tensor]:
    """One training window of the canvas (as the v4 canvas): the human map as frames x lanes, part of it hidden (random
    frames or whole phrases, any share from 5 to 100 %, "restore what is missing"), or - ``p_fix`` - a damaged copy of
    it visible ("find and fix": the loss counts every frame, the error head learns which visible frames are wrong), or
    - ``p_empty`` - nothing visible (a map from scratch). Augmentations: mirrored lanes, rotated lanes (tagged: the
    condition knows; generation never rotates), a shift of 0-9 ms inside the frame (features interpolated between
    frames, notes moved by the same milliseconds: every millisecond of a song can start a window).
    ``seam``: also the earlier of two windows that overlap by half, as in generation (``seam_feats`` / ``seam_beat``:
    the half window before this one; ``seam_state`` / ``seam_start`` / ``seam_end``: the human map of the earlier
    window), for learning the overlap zones (train_v5.seam_batch)."""
    from . import frames as fr

    start_ms = start_frame * frame_ms + shift_ms
    end_ms = start_ms + W * frame_ms
    win = [n for n in notes if (n.end if n.ln else n.start) >= start_ms - frame_ms and n.start < end_ms]
    win = augment_notes(win, mirror, rotate)
    st, so, eo, _ = fr.encode(win, start_ms, W, frame_ms)
    tps = _shifted(timing_points, shift_ms) if shift_ms else timing_points
    if shift_ms:
        rows = cache.window(entry["song_id"], start_frame, W + 1, timing_points)
        a = shift_ms / float(frame_ms)
        feats = (rows[:-1] * (1.0 - a) + rows[1:] * a).astype(np.float32)
    else:
        feats = cache.window(entry["song_id"], start_frame, W, timing_points)
    beat = beat_features(tps, start_frame, W, frame_ms) if tps else np.zeros((W, BEAT_DIM), np.float32)
    # what is visible: masked frames / a damaged copy / nothing
    u = rng.random()
    err = np.zeros(W, dtype=np.float32)
    err_mask = np.zeros(W, dtype=np.float32)
    if u < p_empty:
        hidden = np.ones(W, dtype=bool)
        vis = (st, so, eo)
        loss = np.ones(W, dtype=np.float32)
    elif u < p_empty + p_fix:
        dmg = make_draft(win, timing_points, rng)
        dst, dso, deo, _ = fr.encode(dmg, start_ms, W, frame_ms)
        vis = (dst, dso, deo)
        hidden = hide_frames(W, rng.uniform(0.0, 0.3), rng)
        loss = np.ones(W, dtype=np.float32)
        err = fr.frame_differs(vis, (st, so, eo)).astype(np.float32)
        err_mask = (~hidden).astype(np.float32)
    else:
        r = max(0.05, float(np.cos(np.pi / 2 * rng.random())))
        hidden = hide_frames(W, r, rng)
        vis = (st, so, eo)
        loss = hidden.astype(np.float32)
        err_mask = (~hidden).astype(np.float32)                 # visible frames are the truth here: "no error"
    s_in, a_in, e_in = fr.inputs(*vis, hidden)
    cause = np.full(W, -1, dtype=np.int64)
    ev = np.flatnonzero(((st == fr.TAP) | (st == fr.HEAD)).any(axis=1))
    if ev.size:
        t_ev = start_ms + ev * frame_ms + np.where(so[ev] >= 0, so[ev], 99).min(axis=1)
        cause[ev] = note_causes(t_ev, cache.attacks(entry["song_id"]), frame_ms)
    style = np.asarray(entry["style"], dtype=np.float32)
    smask = np.asarray(entry["style_mask"], dtype=np.float32) if style_known else np.zeros_like(style)
    item = {
        "feats": torch.from_numpy(np.ascontiguousarray(feats)), "beat": torch.from_numpy(beat.astype(np.float32)),
        "cv_state": torch.from_numpy(s_in), "cv_start": torch.from_numpy(a_in), "cv_end": torch.from_numpy(e_in),
        "cv_hidden": torch.from_numpy(hidden.astype(np.int64)),
        "cv_tgt_state": torch.from_numpy(st.astype(np.int64)), "cv_tgt_start": torch.from_numpy(so.astype(np.int64)),
        "cv_tgt_end": torch.from_numpy(eo.astype(np.int64)),
        "cv_loss": torch.from_numpy(loss), "cv_err": torch.from_numpy(err), "cv_err_mask": torch.from_numpy(err_mask),
        "cause": torch.from_numpy(cause),
        "sr": torch.tensor([float(entry.get("sr", 0.0))], dtype=torch.float32),
        "style": torch.from_numpy(style), "style_mask": torch.from_numpy(smask),
        "aug": torch.tensor([1 if rotate else 0], dtype=torch.int64),
    }
    if seam:
        # the earlier window [start - W/2, start + W/2): same augmentation, its own human map
        H2 = W // 2
        s0 = start_ms - H2 * frame_ms
        win_e = [n for n in notes if (n.end if n.ln else n.start) >= s0 - frame_ms and n.start < s0 + W * frame_ms]
        ste, soe, eoe, _ = fr.encode(augment_notes(win_e, mirror, rotate), s0, W, frame_ms)
        if shift_ms:
            rows = cache.window(entry["song_id"], start_frame - H2, H2 + 1, timing_points)
            a = shift_ms / float(frame_ms)
            pre = (rows[:-1] * (1.0 - a) + rows[1:] * a).astype(np.float32)
        else:
            pre = cache.window(entry["song_id"], start_frame - H2, H2, timing_points)
        pre_beat = beat_features(tps, start_frame - H2, H2, frame_ms) if tps else np.zeros((H2, BEAT_DIM), np.float32)
        item.update(seam_feats=torch.from_numpy(np.ascontiguousarray(pre)),
                    seam_beat=torch.from_numpy(pre_beat.astype(np.float32)),
                    seam_state=torch.from_numpy(ste.astype(np.int64)), seam_start=torch.from_numpy(soe.astype(np.int64)),
                    seam_end=torch.from_numpy(eoe.astype(np.int64)))
    return item


def collate(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if "tokens" not in items[0]:                       # canvas windows: every tensor has the same shape
        return {k: torch.stack([it[k] for it in items]) for k in items[0]}
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
                 max_tokens: int = 1536, sections: bool = True, highlight=None, draft_prob: float = 0.0,
                 decoder: str = "ar", mirror: float = 0.5, rotate: float = 0.25, shift: float = 0.5,
                 seam: float = 0.0, sync_seed: int = 0):
        self.index = index
        # seam batches (canvas): a share of the batches also carries the earlier of two overlapping windows; decided
        # per batch from a seed every GPU shares, so all of them take the slower step together
        self.p_seam, self.sync_seed, self.batch_size = float(seam), int(sync_seed), 1
        self.draft_prob = float(draft_prob)
        self.decoder = decoder
        self.p_mirror, self.p_rotate, self.p_shift = float(mirror), float(rotate), float(shift)
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

    def is_seam_batch(self, i: int) -> bool:
        if not (self.train and self.fixed is None and self.decoder == "canvas" and self.p_seam > 0):
            return False
        b = i // max(1, self.batch_size)
        return random.Random(self.sync_seed * 1_000_003 + self.epoch * 7919 + b * 31 + 17).random() < self.p_seam

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
        if self.decoder == "canvas":
            aug = self.train and self.fixed is None
            item = canvas_item(self.cache, self.entries[k], self.notes[k], self.tps[k], start, self.W, rng,
                               style_known=self.train and rng.random() < self.style_known_prob,
                               mirror=aug and rng.random() < self.p_mirror,
                               rotate=rng.randint(1, 3) if aug and rng.random() < self.p_rotate else 0,
                               shift_ms=rng.randint(1, 9) if aug and rng.random() < self.p_shift else 0,
                               seam=self.is_seam_batch(i))
            item["map_k"] = torch.tensor([k], dtype=torch.int64)
            item["item_weight"] = torch.tensor([1.0], dtype=torch.float32)
            return item
        item = window_item(self.cache, self.entries[k], self.notes[k], self.tps[k], start, self.vocab, self.W,
                           style_known=self.train and rng.random() < self.style_known_prob)
        if self.draft_prob > 0:
            # re-mapping: in a share of the windows the encoder also sees a damaged draft of the map (the target is
            # still the human map); in the others the draft input is empty (the first pass)
            from .critic import MAP_DIM
            if rng.random() < self.draft_prob:
                s_ms, e_ms = start * 10, (start + self.W) * 10
                win = [n for n in self.notes[k] if s_ms - 2000 <= n.start < e_ms]
                item["draft"] = torch.from_numpy(draft_frames(make_draft(win, self.tps[k], rng), s_ms, self.W))
            else:
                item["draft"] = torch.zeros(self.W, MAP_DIM + 1)
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
