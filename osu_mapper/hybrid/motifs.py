"""
Motif library: every recurring pattern of real maps, short and long, mined like an archiver mines repeats.

``patterns.py`` keeps an n-gram over the last two chords: enough to say "a [12] after a [34] is common at 5*",
not enough to know a stair that splits into two hands, an LN stair whose holds grow while the song slows down,
a trill answered by its mirror, or a 16-bar chorus drawing that comes back with variations. This module finds
all of them, on several levels of abstraction, without any audio:

Tokens (one per chord, chords = notes that start on the same millisecond)
------------------------------------------------------------------------
``exact``   rhythm (gap to the previous chord in beats: 1/16 .. 4, rest, off-grid) x lanes x long-note lanes
``lanes``   the lanes alone (shapes: stairs, trills, splits, brackets, jacks) whatever the rhythm
``rhythm``  gap in beats x chord size (rhythm figures)
``shape``   how a chord moves from the previous one: size, shared lanes, direction, near / far - the same
            stair or split anywhere on the 4 keys
``hands``   gap x which hand(s) x size (arrangements such as "left hand keeps the kick, right plays the melody")
``flow``    how the gap in *milliseconds* changes (accelerating / steady / slowing) x how the hold length changes x size:
            expanding LN stairs on a slowing song, accelerating bursts, ritardando endings
``ln``      lanes x long-note lanes x LN length in beats

Mining
------
1. **Motifs across maps** (``mine_ngrams``): every token sequence of length 2..``max_len`` that occurs in at least
   ``min_sets`` different beatmap sets (difficulties of one set copy each other, so they count once). Mirrored
   sequences are one motif. Apriori pruning (a sequence can be frequent only if both its prefix and its suffix
   are) keeps it linear in the size of the data. Every motif stores its count, the sets and maps it occurs in,
   a star-rating histogram (maps per 1* bin) and where in the bar it starts.
2. **Form inside a map** (``self_repeats``): an LZ77 parse of each map - the longest earlier passage that the
   current one copies exactly, mirrored, or with the same rhythm and changed lanes (a variation). This is the
   long-range drawing: verse/chorus copies, the same 8-bar phrase with a different ending, call and answer.
3. **Coverage**: how much of every map the frequent motifs explain (the "compression ratio" of mapping).

The result is a ``MotifLibrary`` (``.npz``), a few MB for hundreds of sets; ``mine_patterns.py`` builds it and
writes a readable report with every motif drawn on four columns. ``MotifLibrary.next_distribution`` gives a
PPM-style prediction (longest matching context first, backing off) for a generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint

MOTIF_VERSION = "motifs-v1"
SR_BINS = 12

# --- rhythm classes (gap to the previous chord, in beats) ---------------------------------------------------
RHYTHM_FRACS = np.array([1 / 16, 1 / 12, 1 / 8, 1 / 6, 3 / 16, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1.0, 1.5, 2.0, 3.0, 4.0])
RHYTHM_NAMES = ["1/16", "1/12", "1/8", "1/6", "3/16", "1/4", "1/3", "3/8", "1/2", "2/3", "3/4", "1", "3/2", "2", "3", "4"]
R_REST = RHYTHM_FRACS.size          # first chord / a pause longer than 4.5 beats
R_OFF = R_REST + 1                  # off the grid (1/5, 1/7, humanised, unsnapped)
N_R = R_OFF + 1
RHYTHM_NAMES = RHYTHM_NAMES + ["rest", "off"]

LN_LEN_BEATS = np.array([0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
N_LNC = LN_LEN_BEATS.size + 1       # 0 = no long note

POP = np.array([bin(m).count("1") for m in range(16)])
MIRROR_MASK = np.array([((m & 1) << 3) | ((m & 2) << 1) | ((m & 4) >> 1) | ((m & 8) >> 3) for m in range(16)])
CENTROID = np.array([np.mean([c for c in range(4) if (m >> c) & 1]) if m else 1.5 for m in range(16)])
HAND = np.array([0 if m and not (m & 0b1100) else (1 if m and not (m & 0b0011) else 2) for m in range(16)])  # L / R / both

LEVELS: Tuple[str, ...] = ("exact", "lanes", "rhythm", "shape", "hands", "flow", "ln")
VOCAB = {"exact": N_R * 256, "lanes": 16, "rhythm": N_R * 4, "shape": 96, "hands": N_R * 12, "flow": 80, "ln": N_LNC * 256}
FLOW_EDGES = np.array([-0.6, -0.15, 0.15, 0.6])  # log2(gap / previous gap): much faster, faster, steady, slower, much slower
FLOW_NAMES = ("accel+", "accel", "steady", "slow", "slow+")
LNR_NAMES = ("-", "shorter", "same", "longer")

_P = np.uint64(0x9E3779B97F4A7C15)
_P2 = np.uint64(0xC2B2AE3D27D4EB4F)


# ---------------------------------------------------------------------------------------------------------
# one map -> chords -> tokens
# ---------------------------------------------------------------------------------------------------------

@dataclass
class ChordSeq:
    """The chords of one map with everything the tokens need."""

    times: np.ndarray       # ms
    masks: np.ndarray       # lanes pressed
    lnmasks: np.ndarray     # lanes that are long notes
    lnlen: np.ndarray       # longest hold of the chord in beats (0 = no long note)
    beats: np.ndarray       # position in beats (continuous over red lines)
    beat_ms: np.ndarray     # beat length at the chord
    phase: np.ndarray       # position inside the bar in 1/4 beats (0..15 for 4/4), -1 if unknown

    @property
    def size(self) -> int:
        return int(self.times.size)


def beat_positions(times: np.ndarray, timing_points: Sequence[RawTimingPoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(beats since the first red line, beat length, phase inside the bar in 1/4 beats) of every time."""
    reds = sorted((tp for tp in timing_points if tp.uninherited and 1.0 < tp.beat_len < 10_000.0), key=lambda r: r.time)
    t = np.asarray(times, dtype=np.float64)
    if not reds:
        bl = np.full(t.size, 500.0)
        return t / 500.0, bl, np.full(t.size, -1, dtype=np.int64)
    rt = np.array([r.time for r in reds])
    rb = np.array([r.beat_len for r in reds])
    meter = np.array([max(1, int(r.meter or 4)) for r in reds])
    start_beats = np.concatenate([[0.0], np.cumsum(np.diff(rt) / rb[:-1])])
    k = np.clip(np.searchsorted(rt, t + 0.5, side="right") - 1, 0, rt.size - 1)
    local = (t - rt[k]) / rb[k]
    beats = start_beats[k] + local
    q = np.round(local * 4.0)
    on_grid = np.abs(local * 4.0 - q) * rb[k] / 4.0 <= 4.0
    phase = np.where(on_grid, np.mod(q, meter[k] * 4), -1).astype(np.int64)
    return beats, rb[k], phase


def chord_seq(notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], stagger_ms: int = 0) -> ChordSeq:
    groups: Dict[int, List[Note]] = {}
    for n in notes:
        groups.setdefault(int(n.start), []).append(n)
    times = np.array(sorted(groups), dtype=np.int64)
    masks = np.array([sum(1 << int(n.col) for n in groups[x]) & 15 for x in times], dtype=np.int64)
    lnm = np.array([sum(1 << int(n.col) for n in groups[x] if n.ln and n.end > n.start) & 15 for x in times], dtype=np.int64)
    beats, bl, phase = beat_positions(times, timing_points)
    hold = np.array([max([n.end - n.start for n in groups[x] if n.ln] or [0]) for x in times], dtype=np.float64)
    return ChordSeq(times, masks, lnm, hold / np.maximum(bl, 1.0), beats, bl, phase)


def rhythm_class(gap_beats: np.ndarray, beat_ms: np.ndarray) -> np.ndarray:
    g = np.asarray(gap_beats, dtype=np.float64)
    idx = np.argmin(np.abs(np.log(np.maximum(g, 1e-6))[:, None] - np.log(RHYTHM_FRACS)[None, :]), axis=1)
    err_ms = np.abs(g - RHYTHM_FRACS[idx]) * beat_ms
    tol = np.maximum(4.0, 0.06 * RHYTHM_FRACS[idx] * beat_ms)
    out = np.where(err_ms <= tol, idx, R_OFF)
    return np.where(g > 4.5, R_REST, out).astype(np.int64)


def ln_class(beats: np.ndarray) -> np.ndarray:
    b = np.asarray(beats, dtype=np.float64)
    c = np.argmin(np.abs(np.log(np.maximum(b, 1e-3))[:, None] - np.log(LN_LEN_BEATS)[None, :]), axis=1) + 1
    return np.where(b > 0, c, 0).astype(np.int64)


def tokenize(cs: ChordSeq) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Tokens of every level: {level: (tokens, mirrored tokens or None if the level is mirror-free)}."""
    n = cs.size
    if n == 0:
        return {lv: (np.zeros(0, np.int64), None) for lv in LEVELS}
    gb = np.diff(cs.beats, prepend=cs.beats[0] - 100.0)
    r = rhythm_class(gb, cs.beat_ms)
    r[0] = R_REST
    m, lm = cs.masks, cs.lnmasks
    mm, lmm = MIRROR_MASK[m], MIRROR_MASK[lm]
    size = np.clip(POP[m], 1, 4) - 1
    out: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    out["exact"] = (r * 256 + m * 16 + lm, r * 256 + mm * 16 + lmm)
    out["lanes"] = (m.copy(), mm.copy())
    out["rhythm"] = (r * 4 + size, None)
    prev = np.concatenate([[0], m[:-1]])
    overlap = np.clip(POP[prev & m], 0, 3)
    d = CENTROID[m] - CENTROID[prev]
    dirn = np.where(prev == 0, 1, np.sign(np.round(d, 3)).astype(np.int64) + 1)
    far = (np.abs(d) > 1.01).astype(np.int64)
    out["shape"] = (size + 4 * overlap + 16 * dirn + 48 * far, size + 4 * overlap + 16 * (2 - dirn) + 48 * far)
    hand = HAND[m]
    hand_m = np.where(hand == 2, 2, 1 - hand)
    out["hands"] = (r * 12 + hand * 4 + size, r * 12 + hand_m * 4 + size)
    gms = np.diff(cs.times, prepend=cs.times[0] - 10 ** 6).astype(np.float64)
    pg = np.concatenate([[gms[0]], gms[:-1]])
    flow = np.searchsorted(FLOW_EDGES, np.log2(np.maximum(gms, 1.0) / np.maximum(pg, 1.0)))
    flow[:2] = 2
    flow = np.where(gms > 4.5 * cs.beat_ms, 2, flow)
    hold_ms = cs.lnlen * cs.beat_ms
    lnr = np.zeros(n, dtype=np.int64)
    last = -1.0
    for i in range(n):
        if hold_ms[i] > 0:
            if last > 0:
                q = math.log2(hold_ms[i] / last)
                lnr[i] = 1 if q < -0.15 else (3 if q > 0.15 else 2)
            last = hold_ms[i]
    out["flow"] = (flow * 16 + lnr * 4 + size, None)
    lc = ln_class(cs.lnlen)
    out["ln"] = (lc * 256 + m * 16 + lm, lc * 256 + mm * 16 + lmm)
    return out


# ---------------------------------------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------------------------------------

def _seq_hash(tokens: Sequence[int]) -> np.uint64:
    h = np.uint64(0)
    with np.errstate(over="ignore"):
        for x in tokens:
            h = h * _P + np.uint64(int(x) + 1) * _P2
    return h


def canonical_hash(tokens: Sequence[int], level: str) -> np.uint64:
    h = _seq_hash(tokens)
    mt = mirror_tokens(np.asarray(tokens, dtype=np.int64), level)
    return h if mt is None else min(h, _seq_hash(mt))


def mirror_tokens(tokens: np.ndarray, level: str) -> Optional[np.ndarray]:
    t = np.asarray(tokens, dtype=np.int64)
    if level == "exact" or level == "ln":
        hi, m, lm = t // 256, (t // 16) % 16, t % 16
        return hi * 256 + MIRROR_MASK[m] * 16 + MIRROR_MASK[lm]
    if level == "lanes":
        return MIRROR_MASK[t]
    if level == "shape":
        rest, dirn = t % 16 + 48 * (t // 48), (t // 16) % 3
        return rest + 16 * (2 - dirn)
    if level == "hands":
        r, hand, size = t // 12, (t // 4) % 3, t % 4
        return r * 12 + np.where(hand == 2, 2, 1 - hand) * 4 + size
    return None


# ---------------------------------------------------------------------------------------------------------
# corpus and mining across maps
# ---------------------------------------------------------------------------------------------------------

@dataclass
class MapInfo:
    name: str
    set_id: int
    sr: float
    n_chords: int
    ln_share: float


@dataclass
class Corpus:
    """All maps, one token stream per level (maps separated by -1)."""

    maps: List[MapInfo] = field(default_factory=list)
    seqs: List[ChordSeq] = field(default_factory=list)
    tokens: Dict[str, List[np.ndarray]] = field(default_factory=lambda: {lv: [] for lv in LEVELS})
    mirrored: Dict[str, List[Optional[np.ndarray]]] = field(default_factory=lambda: {lv: [] for lv in LEVELS})

    def add(self, name: str, set_id: int, notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], sr: float) -> None:
        cs = chord_seq(notes, timing_points)
        if cs.size < 8:
            return
        tok = tokenize(cs)
        self.maps.append(MapInfo(name, int(set_id), float(sr), cs.size, float(np.mean([n.ln for n in notes]))))
        self.seqs.append(cs)
        for lv in LEVELS:
            self.tokens[lv].append(tok[lv][0])
            self.mirrored[lv].append(tok[lv][1])


def _concat(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates with -1 separators: (tokens, map index of every position, distance to the next separator)."""
    lens = np.array([a.size for a in arrs], dtype=np.int64)
    total = int(lens.sum() + lens.size)
    out = np.full(total, -1, dtype=np.int64)
    mid = np.full(total, -1, dtype=np.int64)
    room = np.zeros(total, dtype=np.int64)
    p = 0
    for k, a in enumerate(arrs):
        out[p:p + a.size] = a
        mid[p:p + a.size] = k
        room[p:p + a.size] = np.arange(a.size, 0, -1)
        p += a.size + 1
    return out, mid, room


@dataclass
class MotifTable:
    """Frequent token sequences of one level."""

    level: str
    lengths: np.ndarray     # [K] events
    offsets: np.ndarray     # [K] start of the tokens in ``flat``
    flat: np.ndarray        # all motif tokens, concatenated (canonical orientation)
    count: np.ndarray       # [K] occurrences (all maps, mirrored counted together)
    sets: np.ndarray        # [K] distinct beatmap sets
    nmaps: np.ndarray       # [K] distinct maps
    sr_hist: np.ndarray     # [K, SR_BINS] maps per star-rating bin that contain the motif
    phase_hist: np.ndarray  # [K, 4] start on a downbeat / beat / 1/2 / elsewhere
    beats: np.ndarray       # [K] mean length in beats (first to last chord)
    closed: np.ndarray      # [K] False: every occurrence is part of one longer motif
    hashes: np.ndarray      # [K] canonical hash, sorted

    def tokens(self, k: int) -> np.ndarray:
        return self.flat[self.offsets[k]: self.offsets[k] + self.lengths[k]]

    def __len__(self) -> int:
        return int(self.lengths.size)

    def find(self, tokens: Sequence[int]) -> int:
        h = canonical_hash(tokens, self.level)
        i = int(np.searchsorted(self.hashes, h))
        return i if i < self.hashes.size and self.hashes[i] == h else -1


def _phase_class(phase: np.ndarray, meter_quarters: int = 16) -> np.ndarray:
    return np.where(phase < 0, 3, np.where(phase == 0, 0, np.where(phase % 4 == 0, 1, np.where(phase % 2 == 0, 2, 3))))


def mine_ngrams(corpus: Corpus, level: str, min_sets: int = 3, max_len: int = 32, keep_all_upto: int = 6,
                log=print) -> MotifTable:
    """All token sequences of length 2..max_len that occur in at least ``min_sets`` beatmap sets."""
    T, MID, ROOM = _concat(corpus.tokens[level])
    mir = corpus.mirrored[level]
    has_mirror = mir and mir[0] is not None
    TM = _concat(mir)[0] if has_mirror else None
    set_of_map = np.array([m.set_id for m in corpus.maps], dtype=np.int64)
    sr_of_map = np.clip(np.floor([m.sr for m in corpus.maps]), 0, SR_BINS - 1).astype(np.int64)
    phase_all = np.concatenate([np.append(_phase_class(s.phase), 3) for s in corpus.seqs]) if corpus.seqs else np.zeros(0, np.int64)
    beats_all = np.concatenate([np.append(s.beats, 0.0) for s in corpus.seqs]) if corpus.seqs else np.zeros(0)
    N = T.size
    valid = T >= 0
    with np.errstate(over="ignore"):
        tok = np.where(valid, T, 0).astype(np.uint64) + np.uint64(1)
        tokm = (np.where(valid, TM, 0).astype(np.uint64) + np.uint64(1)) if has_mirror else None
        H = tok * _P2
        HM = tokm * _P2 if has_mirror else None
    freq_prev = valid.copy()  # length-1 "frequent" = every valid position
    tables = []
    for n in range(2, max_len + 1):
        idx = np.flatnonzero(freq_prev[:N - 1] & freq_prev[1:] & (ROOM[:N - 1] >= n))
        if idx.size == 0:
            break
        with np.errstate(over="ignore"):
            Hn = np.zeros(N, dtype=np.uint64)
            Hn[idx] = H[idx] * _P + tok[idx + n - 1] * _P2
            if has_mirror:
                HMn = np.zeros(N, dtype=np.uint64)
                HMn[idx] = HM[idx] * _P + tokm[idx + n - 1] * _P2
        canon = np.minimum(Hn[idx], HMn[idx]) if has_mirror else Hn[idx]
        maps_i = MID[idx]
        # support = distinct sets
        pair = np.unique(np.stack([canon, set_of_map[maps_i].astype(np.uint64)], axis=1), axis=0) if canon.size else np.zeros((0, 2), np.uint64)
        uh, sets_cnt = np.unique(pair[:, 0], return_counts=True)
        frequent = uh[sets_cnt >= min_sets]
        is_f = np.isin(canon, frequent)
        fidx = idx[is_f]
        freq_now = np.zeros(N, dtype=bool)
        freq_now[fidx] = True
        if fidx.size:
            fc = canon[is_f]
            hs, inv, cnt = np.unique(fc, return_inverse=True, return_counts=True)
            first = np.full(hs.size, -1, dtype=np.int64)
            np.minimum.at(first, inv, -1)  # placeholder to keep dtype
            first = np.zeros(hs.size, dtype=np.int64)
            first[inv[::-1]] = fidx[::-1]
            # orientation of the stored tokens: the canonical one
            use_m = (np.minimum(Hn[first], HMn[first]) != Hn[first]) if has_mirror else np.zeros(hs.size, bool)
            mp = MID[fidx]
            mp_pairs = np.unique(np.stack([inv, mp], axis=1), axis=0)
            nmaps = np.bincount(mp_pairs[:, 0], minlength=hs.size)
            sr_hist = np.zeros((hs.size, SR_BINS), dtype=np.int32)
            np.add.at(sr_hist, (mp_pairs[:, 0], sr_of_map[mp_pairs[:, 1]]), 1)
            st_pairs = np.unique(np.stack([inv, set_of_map[mp]], axis=1), axis=0)
            nsets = np.bincount(st_pairs[:, 0], minlength=hs.size)
            ph = np.zeros((hs.size, 4), dtype=np.int32)
            np.add.at(ph, (inv, phase_all[fidx]), 1)
            span = beats_all[fidx + n - 1] - beats_all[fidx]
            bsum = np.bincount(inv, weights=span, minlength=hs.size)
            toks = np.stack([T[first + j] for j in range(n)], axis=1)
            if has_mirror and use_m.any():
                toks[use_m] = np.stack([TM[first[use_m] + j] for j in range(n)], axis=1)
            tables.append(dict(n=n, hashes=hs, toks=toks, count=cnt, sets=nsets, nmaps=nmaps, sr_hist=sr_hist, phase=ph,
                               beats=bsum / np.maximum(cnt, 1), prefix=None, inv=inv, fidx=fidx))
            log(f"    [{level}] length {n:2d}: {hs.size:8d} motifs, {fidx.size:9d} occurrences")
        H, HM = Hn, (HMn if has_mirror else None)
        freq_prev = freq_now
        if not fidx.size:
            break
    # closedness: a motif is not closed if one longer motif (one more chord before or after) has the same count
    for a, b in zip(tables[:-1], tables[1:]):
        a["closed"] = np.ones(a["hashes"].size, dtype=bool)
        cnt_of = dict(zip(a["hashes"].tolist(), a["count"].tolist()))
        # canonical hashes of the prefix / suffix windows of b's occurrences, via the stored tokens
        for k in range(b["hashes"].size):
            tk = b["toks"][k]
            for part in (tk[:-1], tk[1:]):
                h = int(canonical_hash(part, level))
                c = cnt_of.get(h)
                if c is not None and c == int(b["count"][k]):
                    j = int(np.searchsorted(a["hashes"], np.uint64(h)))
                    a["closed"][j] = False
    if tables:
        tables[-1]["closed"] = np.ones(tables[-1]["hashes"].size, dtype=bool)
    # keep: everything up to keep_all_upto (context statistics for prediction), closed motifs beyond
    parts = []
    for tb in tables:
        keep = np.ones(tb["hashes"].size, bool) if tb["n"] <= keep_all_upto else tb["closed"]
        parts.append((tb, keep))
    return _build_table(level, parts)


def _build_table(level: str, parts) -> MotifTable:
    lengths, flat, hashes, count, sets, nmaps, srh, ph, beats, closed = [], [], [], [], [], [], [], [], [], []
    for tb, keep in parts:
        k = np.flatnonzero(keep)
        if not k.size:
            continue
        lengths.append(np.full(k.size, tb["n"], dtype=np.int16))
        flat.append(tb["toks"][k].astype(np.int16).ravel())
        hashes.append(tb["hashes"][k])
        count.append(tb["count"][k].astype(np.int32))
        sets.append(tb["sets"][k].astype(np.int32))
        nmaps.append(tb["nmaps"][k].astype(np.int32))
        srh.append(tb["sr_hist"][k].astype(np.int32))
        ph.append(tb["phase"][k].astype(np.int32))
        beats.append(tb["beats"][k].astype(np.float32))
        closed.append(tb["closed"][k])
    if not lengths:
        z = np.zeros(0, np.int32)
        return MotifTable(level, z.astype(np.int16), z, z.astype(np.int16), z, z, z, np.zeros((0, SR_BINS), np.int32),
                          np.zeros((0, 4), np.int32), z.astype(np.float32), z.astype(bool), z.astype(np.uint64))
    L = np.concatenate(lengths)
    H = np.concatenate(hashes)
    order = np.argsort(H, kind="stable")
    offs_unsorted = np.concatenate([[0], np.cumsum(L.astype(np.int64))[:-1]])
    F = np.concatenate(flat)
    # re-pack the token blocks in hash order
    newflat = np.concatenate([F[offs_unsorted[i]: offs_unsorted[i] + L[i]] for i in order]) if order.size else F
    L = L[order]
    offs = np.concatenate([[0], np.cumsum(L.astype(np.int64))[:-1]]).astype(np.int64)
    return MotifTable(level, L, offs, newflat, np.concatenate(count)[order], np.concatenate(sets)[order],
                      np.concatenate(nmaps)[order], np.concatenate(srh)[order], np.concatenate(ph)[order],
                      np.concatenate(beats)[order], np.concatenate(closed)[order], H[order])


# ---------------------------------------------------------------------------------------------------------
# form inside a map: LZ77 parse
# ---------------------------------------------------------------------------------------------------------

COPY_KINDS = ("exact", "mirror", "variation", "rhythm")   # variation: same rhythm, <= half of the lanes changed; rhythm: more
COPY_LEN_BEATS = np.array([2, 4, 8, 16, 32, 64, 128])     # bins of copied length (beats)
COPY_DIST_BEATS = np.array([4, 8, 16, 32, 64, 128, 256])  # bins of distance to the source (beats)


@dataclass
class Copy:
    start: int          # chord index of the copy
    src: int            # chord index of the source
    length: int         # chords
    kind: str           # exact / mirror / variation
    changed: float      # share of chords with other lanes (variation only)


def self_repeats(cs: ChordSeq, tok: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]], min_len: int = 8,
                 max_candidates: int = 24) -> List[Copy]:
    """Greedy LZ77 parse of one map: at every chord the longest earlier passage it repeats (exact or mirrored lanes
    with the same rhythm), else the longest passage with the same rhythm and chord sizes (a variation)."""
    ex, exm = tok["exact"]
    rh = tok["rhythm"][0]
    n = ex.size
    if n < 2 * min_len:
        return []
    k = min_len

    def windows(a: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            h = np.zeros(n - k + 1, dtype=np.uint64)
            for j in range(k):
                h = h * _P + (a[j:n - k + 1 + j].astype(np.uint64) + np.uint64(1)) * _P2
        return h

    hx, hxm, hr = windows(ex), windows(exm), windows(rh)
    idx_x: Dict[int, List[int]] = {}
    idx_r: Dict[int, List[int]] = {}
    out: List[Copy] = []
    i = 0
    added = 0

    def index_upto(limit: int) -> None:
        nonlocal added
        while added <= min(limit, n - k):
            idx_x.setdefault(int(hx[added]), []).append(added)
            idx_r.setdefault(int(hr[added]), []).append(added)
            added += 1

    while i <= n - k:
        index_upto(i - k)  # sources end before the copy starts (no self-overlap)
        best = None
        for h, arr, kind in ((int(hx[i]), ex, "exact"), (int(hxm[i]), exm, "mirror")):
            for j in reversed(idx_x.get(h, [])[-max_candidates:]):
                L = 0
                while i + L < n and j + L < i and arr[i + L] == ex[j + L]:
                    L += 1
                if L >= k and (best is None or L > best.length):
                    best = Copy(i, j, L, kind, 0.0)
        if best is None:
            for j in reversed(idx_r.get(int(hr[i]), [])[-max_candidates:]):
                L = 0
                while i + L < n and j + L < i and rh[i + L] == rh[j + L]:
                    L += 1
                if L >= k and (best is None or L > best.length) and np.unique(rh[i:i + L]).size >= 2:
                    ch = float(np.mean(cs.masks[i:i + L] != cs.masks[j:j + L]))
                    best = Copy(i, j, L, "variation" if ch <= 0.5 else "rhythm", ch)
        if best is not None:
            out.append(best)
            i += best.length
        else:
            i += 1
    return out


@dataclass
class FormStats:
    """How real maps repeat themselves, per star-rating bin."""

    chords: np.ndarray = field(default_factory=lambda: np.zeros(SR_BINS))                # all chords
    covered: np.ndarray = field(default_factory=lambda: np.zeros((SR_BINS, 4)))          # chords inside copies by kind
    length: np.ndarray = field(default_factory=lambda: np.zeros((SR_BINS, 4, COPY_LEN_BEATS.size + 1)))
    dist: np.ndarray = field(default_factory=lambda: np.zeros((SR_BINS, 4, COPY_DIST_BEATS.size + 1)))
    changed: np.ndarray = field(default_factory=lambda: np.zeros((SR_BINS, 11)))         # variations: share of lanes changed
    start_phase: np.ndarray = field(default_factory=lambda: np.zeros((SR_BINS, 4)))      # where copies start in the bar
    maps: np.ndarray = field(default_factory=lambda: np.zeros(SR_BINS))

    def add(self, cs: ChordSeq, copies: List[Copy], sr: float) -> None:
        s = int(np.clip(math.floor(sr), 0, SR_BINS - 1))
        self.maps[s] += 1
        self.chords[s] += cs.size
        ph = _phase_class(cs.phase)
        for c in copies:
            kd = COPY_KINDS.index(c.kind)
            self.covered[s, kd] += c.length
            span = float(cs.beats[c.start + c.length - 1] - cs.beats[c.start])
            dist = float(cs.beats[c.start] - cs.beats[c.src])
            self.length[s, kd, np.searchsorted(COPY_LEN_BEATS, span)] += 1
            self.dist[s, kd, np.searchsorted(COPY_DIST_BEATS, dist)] += 1
            if c.kind in ("variation", "rhythm"):
                self.changed[s, int(round(c.changed * 10))] += 1
            self.start_phase[s, ph[c.start]] += 1

    def arrays(self) -> Dict[str, np.ndarray]:
        return {f"form_{k}": getattr(self, k) for k in ("chords", "covered", "length", "dist", "changed", "start_phase", "maps")}

    @classmethod
    def from_arrays(cls, d) -> "FormStats":
        f = cls()
        for k in ("chords", "covered", "length", "dist", "changed", "start_phase", "maps"):
            if f"form_{k}" in d:
                setattr(f, k, np.asarray(d[f"form_{k}"], dtype=np.float64))
        return f


# ---------------------------------------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------------------------------------

@dataclass
class MotifLibrary:
    tables: Dict[str, MotifTable]
    form: FormStats
    coverage: Dict[str, np.ndarray]       # level -> [SR_BINS, 3]: chords, chords inside a motif >= 4, >= 8
    maps_per_bin: np.ndarray
    n_maps: int
    n_sets: int
    version: str = MOTIF_VERSION

    # --- persistence ------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        d = {"version": np.array(self.version), "maps_per_bin": self.maps_per_bin, "n_maps": np.array(self.n_maps),
             "n_sets": np.array(self.n_sets)}
        for lv, t in self.tables.items():
            for k in ("lengths", "offsets", "flat", "count", "sets", "nmaps", "sr_hist", "phase_hist", "beats", "closed", "hashes"):
                d[f"{lv}__{k}"] = getattr(t, k)
        for lv, c in self.coverage.items():
            d[f"cov__{lv}"] = c
        d.update(self.form.arrays())
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **d)

    @classmethod
    def load(cls, path: str | Path, levels: Optional[Sequence[str]] = None) -> "MotifLibrary":
        z = np.load(path, allow_pickle=False)
        tables = {}
        for lv in LEVELS:
            if levels is not None and lv not in levels:
                continue
            if f"{lv}__lengths" not in z:
                continue
            tables[lv] = MotifTable(lv, *(z[f"{lv}__{k}"] for k in (
                "lengths", "offsets", "flat", "count", "sets", "nmaps", "sr_hist", "phase_hist", "beats", "closed", "hashes")))
        cov = {lv: z[f"cov__{lv}"] for lv in LEVELS if f"cov__{lv}" in z and (levels is None or lv in levels)}
        return cls(tables, FormStats.from_arrays(z), cov, z["maps_per_bin"], int(z["n_maps"]), int(z["n_sets"]), str(z["version"]))

    # --- queries -----------------------------------------------------------------------------------------
    def sr_weight(self, level: str, sr: Optional[float]) -> np.ndarray:
        """Per motif: share of the maps of this star rating (+-1 bin) that use it."""
        t = self.tables[level]
        if sr is None:
            return t.nmaps / max(1, self.n_maps)
        b = int(np.clip(math.floor(sr), 0, SR_BINS - 1))
        bins = [x for x in (b - 1, b, b + 1) if 0 <= x < SR_BINS]
        w = np.array([1.0 if x == b else 0.35 for x in bins])
        num = (t.sr_hist[:, bins] * w).sum(axis=1)
        den = float((self.maps_per_bin[bins] * w).sum())
        return num / max(1.0, den)

    def top(self, level: str, sr: Optional[float] = None, min_len: int = 2, max_len: int = 999, k: int = 20,
            closed_only: bool = True, informative: bool = True) -> List[int]:
        t = self.tables[level]
        use = (t.lengths >= min_len) & (t.lengths <= max_len)
        if closed_only:
            use &= t.closed
        score = self.sr_weight(level, sr) * np.sqrt(t.lengths.astype(np.float64))
        if informative:
            score = score * self.distinct(level)
        score = np.where(use, score, -1.0)
        order = np.argsort(-score)
        return [int(i) for i in order[:k] if score[i] > 0]

    def distinct(self, level: str) -> np.ndarray:
        """Per motif: how many different tokens it has (1 = a plain periodic run, which says little)."""
        cache = self.__dict__.setdefault("_distinct", {})
        if level not in cache:
            t = self.tables[level]
            cache[level] = np.array([np.unique(t.tokens(k)).size for k in range(len(t))], dtype=np.float64)
        return cache[level]

    def next_distribution(self, level: str, context: Sequence[int], sr: Optional[float] = None, max_order: int = 6,
                          candidates: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """PPM-style prediction of the next token: the longest context that the library knows decides first, shorter
        contexts fill in (escape probability by the Witten-Bell rule). Returns (candidate tokens, probabilities)."""
        t = self.tables[level]
        cand = np.arange(VOCAB[level]) if candidates is None else np.asarray(candidates, dtype=np.int64)
        probs = np.zeros(cand.size)
        mass = 1.0
        ctx = list(context)[-max_order:]
        wt = self.sr_weight(level, sr) if sr is not None else None
        for order in range(len(ctx), 0, -1):
            c = ctx[len(ctx) - order:]
            cnt = np.zeros(cand.size)
            for j, x in enumerate(cand):
                k = t.find(list(c) + [int(x)])
                if k >= 0:
                    cnt[j] = t.count[k] * (0.2 + wt[k] if wt is not None else 1.0)
            tot = cnt.sum()
            if tot <= 0:
                continue
            distinct = float((cnt > 0).sum())
            esc = distinct / (tot + distinct)
            probs += mass * (1 - esc) * cnt / tot
            mass *= esc
        probs += mass / cand.size
        return cand, probs / probs.sum()

    def summary(self) -> str:
        rows = [f"motif library {self.version}: {self.n_maps} maps of {self.n_sets} sets"]
        for lv, t in self.tables.items():
            nl = np.bincount(t.lengths.astype(np.int64)) if len(t) else np.zeros(1)
            longest = int(t.lengths.max()) if len(t) else 0
            rows.append(f"  {lv:7s} {len(t):8d} motifs (closed {int(t.closed.sum())}), longest {longest} chords,"
                        f" >= 8 chords: {int(nl[8:].sum()) if nl.size > 8 else 0}")
        f = self.form
        tot = f.chords.sum()
        if tot > 0:
            cov = f.covered.sum(axis=0) / tot
            rows.append(f"  form: {cov[0]:.0%} of chords copy an earlier passage exactly, {cov[1]:.0%} mirrored, "
                        f"{cov[2]:.0%} as a variation (same rhythm, <= half of the lanes changed), {cov[3]:.0%} keep only the rhythm")
        return "\n".join(rows)


def coverage_of(corpus: Corpus, table: MotifTable, level: str, lengths=(4, 8)) -> np.ndarray:
    """[SR_BINS, 1 + len(lengths)]: chords, and chords that lie inside an occurrence of a motif of >= L chords."""
    out = np.zeros((SR_BINS, 1 + len(lengths)))
    hs = table.hashes
    for info, tk, tm in zip(corpus.maps, corpus.tokens[level], corpus.mirrored[level]):
        s = int(np.clip(math.floor(info.sr), 0, SR_BINS - 1))
        out[s, 0] += tk.size
        for col, L in enumerate(lengths, 1):
            if tk.size < L:
                continue
            with np.errstate(over="ignore"):
                h = np.zeros(tk.size - L + 1, dtype=np.uint64)
                hm = np.zeros_like(h)
                for j in range(L):
                    h = h * _P + (tk[j:tk.size - L + 1 + j].astype(np.uint64) + np.uint64(1)) * _P2
                    if tm is not None:
                        hm = hm * _P + (tm[j:tk.size - L + 1 + j].astype(np.uint64) + np.uint64(1)) * _P2
            c = np.minimum(h, hm) if tm is not None else h
            pos = np.clip(np.searchsorted(hs, c), 0, max(0, hs.size - 1))
            hit = (hs.size > 0) & (hs[pos] == c) if hs.size else np.zeros(c.size, bool)
            cov = np.zeros(tk.size + 1, dtype=np.int64)
            st = np.flatnonzero(hit)
            np.add.at(cov, st, 1)
            np.add.at(cov, st + L, -1)
            out[s, col] += float((np.cumsum(cov)[:-1] > 0).sum())
    return out


def build_library(corpus: Corpus, min_sets: int = 3, max_len: int = 32, levels: Sequence[str] = LEVELS, log=print) -> MotifLibrary:
    tables = {}
    for lv in levels:
        log(f"[motifs] level {lv}")
        tables[lv] = mine_ngrams(corpus, lv, min_sets=min_sets, max_len=max_len, log=log)
    log("[motifs] form of every map (LZ77 parse)")
    form = FormStats()
    for cs, info, k in zip(corpus.seqs, corpus.maps, range(len(corpus.maps))):
        tok = {lv: (corpus.tokens[lv][k], corpus.mirrored[lv][k]) for lv in ("exact", "rhythm")}
        form.add(cs, self_repeats(cs, tok), info.sr)
    log("[motifs] coverage")
    cov = {lv: coverage_of(corpus, tables[lv], lv) for lv in levels}
    mpb = np.bincount(np.clip(np.floor([m.sr for m in corpus.maps]), 0, SR_BINS - 1).astype(np.int64), minlength=SR_BINS)
    return MotifLibrary(tables, form, cov, mpb.astype(np.int64), len(corpus.maps), len({m.set_id for m in corpus.maps}))


# ---------------------------------------------------------------------------------------------------------
# reading motifs: decoding, drawing, naming
# ---------------------------------------------------------------------------------------------------------

def decode(level: str, tokens: Sequence[int]) -> List[dict]:
    out = []
    for x in tokens:
        x = int(x)
        if level == "exact":
            out.append(dict(r=x // 256, mask=(x // 16) % 16, ln=x % 16))
        elif level == "ln":
            out.append(dict(lnc=x // 256, mask=(x // 16) % 16, ln=x % 16))
        elif level == "lanes":
            out.append(dict(mask=x))
        elif level == "rhythm":
            out.append(dict(r=x // 4, size=x % 4 + 1))
        elif level == "shape":
            out.append(dict(size=x % 4 + 1, overlap=(x // 4) % 4, dir=(x // 16) % 3 - 1, far=x // 48))
        elif level == "hands":
            out.append(dict(r=x // 12, hand="LRB"[(x // 4) % 3], size=x % 4 + 1))
        elif level == "flow":
            out.append(dict(flow=FLOW_NAMES[x // 16], lnr=LNR_NAMES[(x // 4) % 4], size=x % 4 + 1))
    return out


def draw(level: str, tokens: Sequence[int]) -> str:
    """Four columns, time going up like in the editor (first chord at the bottom); '#' tap, 'O' long-note head."""
    ev = decode(level, tokens)
    lines = []
    for e in ev:
        if "mask" not in e:
            break
        m, ln = e["mask"], e.get("ln", 0)
        row = "".join(("O" if (ln >> c) & 1 else "#") if (m >> c) & 1 else "." for c in range(4))
        tag = RHYTHM_NAMES[e["r"]] if "r" in e else (f"LN {LN_LEN_BEATS[e['lnc'] - 1]:g}b" if e.get("lnc") else "")
        lines.append(f"|{row}| {tag}")
    if lines:
        return "\n".join(reversed(lines))
    return " ".join(describe_token(level, x) for x in tokens)


def describe_token(level: str, x: int) -> str:
    e = decode(level, [x])[0]
    if level == "rhythm":
        return f"{RHYTHM_NAMES[e['r']]}x{e['size']}"
    if level == "shape":
        mv = {-1: "<", 0: "=", 1: ">"}[e["dir"]]
        return f"{e['size']}{mv}{'far' if e['far'] else ''}{'o' * e['overlap']}"
    if level == "hands":
        return f"{RHYTHM_NAMES[e['r']]}:{e['hand']}{e['size']}"
    if level == "flow":
        return f"{e['flow']}{'/' + e['lnr'] if e['lnr'] != '-' else ''}x{e['size']}"
    return str(x)


def name_motif(level: str, tokens: Sequence[int]) -> str:
    """A mapper's name for a lane motif (heuristic): stairs, trill, jack, chordjack, split, jumptrill, ..."""
    ev = decode(level, tokens)
    if not ev or "mask" not in ev[0]:
        if level == "flow":
            fl = [e["flow"] for e in ev]
            lr = [e["lnr"] for e in ev]
            if lr.count("longer") >= max(2, len(ev) // 2):
                return "expanding holds"
            if lr.count("shorter") >= max(2, len(ev) // 2):
                return "shrinking holds"
            if sum(f.startswith("slow") for f in fl) >= max(2, len(ev) // 3):
                return "slowing down"
            if sum(f.startswith("accel") for f in fl) >= max(2, len(ev) // 3):
                return "speeding up"
            return "steady"
        if level == "shape":
            if all(e["overlap"] == e["size"] and e["overlap"] > 0 for e in ev[1:]):
                return "jack / chordjack"
            dirs = [e["dir"] for e in ev[1:]]
            if len(set(dirs)) == 1 and dirs[0] != 0 and all(e["overlap"] == 0 for e in ev[1:]):
                return "stairs" if all(e["size"] == 1 for e in ev) else "chord stairs"
            if all(d != 0 for d in dirs) and all(a == -b for a, b in zip(dirs[:-1], dirs[1:])):
                return "trill / zigzag"
        return ""
    masks = [e["mask"] for e in ev]
    lns = [e.get("ln", 0) for e in ev]
    sizes = [int(POP[m]) for m in masks]
    tags = []
    if any(lns):
        tags.append("LN")
    n = len(masks)
    if n >= 2 and all(masks[i] & masks[i - 1] for i in range(1, n)) and all(m == masks[0] for m in masks):
        tags.append("jack" if sizes[0] == 1 else "chordjack")
    elif n >= 3 and all(m == masks[i % 2] for i, m in enumerate(masks)) and masks[0] != masks[1]:
        tags.append(("jumptrill" if sizes[0] >= 2 and sizes[1] >= 2 else "trill") if not masks[0] & masks[1] else "anchor trill")
    elif max(sizes) == 1:
        cols = [int(CENTROID[m]) for m in masks]
        d = np.diff(cols)
        if np.all(d == 1) or np.all(d == -1):
            tags.append("stairs")
        elif np.all(np.abs(d) == 1):
            tags.append("roll")
        elif any(x == 0 for x in d):
            tags.append("minijack stream")
        else:
            tags.append("stream")
    else:
        hands = [HAND[m] for m in masks]
        if all(h == 2 for h in hands) and min(sizes) == 2 and all((m in (0b0101, 0b1010)) for m in masks):
            tags.append("split / bracket")
        elif all(s >= 3 for s in sizes[::2]):
            tags.append("handstream" if not any(masks[i] & masks[i - 1] for i in range(1, n)) else "chordjack mix")
        elif all(s >= 2 for s in sizes):
            tags.append("jumpstream" if not any(masks[i] & masks[i - 1] for i in range(1, n)) else "chord jacks")
        elif sum(s >= 2 for s in sizes) >= n // 3:
            c = [CENTROID[m] for m in masks]
            mono = np.all(np.diff(c) >= -0.01) or np.all(np.diff(c) <= 0.01)
            tags.append("chord stairs" if mono else "jumpstream")
        else:
            c = [CENTROID[m] for m in masks]
            mono = np.all(np.diff(c) >= -0.01) or np.all(np.diff(c) <= 0.01)
            tags.append("split stairs" if mono else "light jumpstream")
    return " ".join(tags)


# ---------------------------------------------------------------------------------------------------------
# prediction for a generator
# ---------------------------------------------------------------------------------------------------------

class LanePredictor:
    """Next chord (lane mask) from the chords before it, PPM-style over the library's ``lanes`` motifs.

    For a target star rating every motif of 2..``max_order``+1 chords is counted in both orientations, weighted by how
    many maps of that star rating (+-1 bin) use it. ``logprob(history)`` blends the longest known context with the
    shorter ones (Witten-Bell escapes), so a stair, a split, a trill answered by its mirror or a 6-chord figure of real
    maps continues the way real maps continue it."""

    def __init__(self, lib: MotifLibrary, sr: float, max_order: int = 6, level: str = "lanes"):
        t = lib.tables[level]
        self.max_order = int(max_order)
        b = int(np.clip(math.floor(sr), 0, SR_BINS - 1))
        w = np.zeros(SR_BINS)
        for x, ww in ((b, 1.0), (b - 1, 0.35), (b + 1, 0.35)):
            if 0 <= x < SR_BINS:
                w[x] = ww
        frac = (t.sr_hist.astype(np.float64) @ w) / np.maximum(1.0, t.nmaps.astype(np.float64))
        weight = t.count.astype(np.float64) * (frac + 0.03)
        self.table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.raw: Dict[Tuple[int, ...], float] = {}
        keep = np.flatnonzero((t.lengths >= 2) & (t.lengths <= self.max_order + 1))
        for k in keep:
            seq = t.tokens(int(k)).astype(np.int64)
            wk = float(weight[k])
            if wk <= 0:
                continue
            for s_ in ((seq, MIRROR_MASK[seq]) if not np.array_equal(seq, MIRROR_MASK[seq]) else (seq,)):
                ctx = tuple(int(x) for x in s_[:-1])
                arr = self.table.get(ctx)
                if arr is None:
                    arr = np.zeros(16)
                    self.table[ctx] = arr
                arr[int(s_[-1])] += wk
                self.raw[ctx] = self.raw.get(ctx, 0.0) + float(t.count[k])

    def logprob(self, history: Sequence[int]) -> np.ndarray:
        """log P(next mask) for all 16 masks (index 0 unused)."""
        probs = np.zeros(16)
        mass = 1.0
        h = [int(x) for x in history[-self.max_order:]]
        for order in range(len(h), 0, -1):
            cnt = self.table.get(tuple(h[len(h) - order:]))
            if cnt is None:
                continue
            tot = float(cnt.sum())
            if tot <= 0:
                continue
            distinct = float((cnt > 0).sum())
            esc = distinct / (self.raw.get(tuple(h[len(h) - order:]), tot) + distinct)   # Witten-Bell on occurrences
            probs += mass * (1.0 - esc) * cnt / tot
            mass *= esc
        probs[1:] += mass / 15.0
        probs[0] = 1e-12
        return np.log(np.maximum(probs, 1e-12))


def covered_share(tokens: np.ndarray, mirrored: Optional[np.ndarray], table: MotifTable, length: int = 8) -> float:
    """Share of the chords of one map that lie inside an occurrence of a library motif of ``length`` chords."""
    tk = np.asarray(tokens, dtype=np.int64)
    L = int(length)
    if tk.size < L or len(table) == 0:
        return 0.0
    with np.errstate(over="ignore"):
        h = np.zeros(tk.size - L + 1, dtype=np.uint64)
        hm = np.zeros_like(h)
        for j in range(L):
            h = h * _P + (tk[j:tk.size - L + 1 + j].astype(np.uint64) + np.uint64(1)) * _P2
            if mirrored is not None:
                hm = hm * _P + (np.asarray(mirrored)[j:tk.size - L + 1 + j].astype(np.uint64) + np.uint64(1)) * _P2
    c = np.minimum(h, hm) if mirrored is not None else h
    hs = table.hashes
    pos = np.clip(np.searchsorted(hs, c), 0, hs.size - 1)
    hit = hs[pos] == c
    cov = np.zeros(tk.size + 1, dtype=np.int64)
    st = np.flatnonzero(hit)
    np.add.at(cov, st, 1)
    np.add.at(cov, st + L, -1)
    return float((np.cumsum(cov)[:-1] > 0).mean())


def load_default_motifs(path: Optional[str] = None, log=print, levels: Sequence[str] = ("lanes",)) -> Optional[MotifLibrary]:
    """The motif library for the composer: ``path``, else checkpoints/hybrid/motifs.npz, else test_NN/motifs.npz.
    None (with a note) when there is none or the file is a Git LFS pointer that was not downloaded."""
    root = Path(__file__).resolve().parents[2]
    cands = [Path(path)] if path else [Path("checkpoints/hybrid/motifs.npz"), root / "checkpoints/hybrid/motifs.npz",
                                       root / "test_NN/motifs.npz"]
    for p in cands:
        if not p.exists():
            continue
        if p.stat().st_size < 1024:
            log(f"[motifs] {p} is a Git LFS pointer (git lfs pull), motif library not used")
            continue
        lib = MotifLibrary.load(p, levels=levels)
        log(f"[motifs] {p}: {lib.n_maps} maps of {lib.n_sets} sets")
        return lib
    if path:
        raise FileNotFoundError(f"motif library '{path}' not found (python mine_patterns.py)")
    log("[motifs] no motif library (python mine_patterns.py): lanes from the pattern model only")
    return None
