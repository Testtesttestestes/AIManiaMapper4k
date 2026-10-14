"""
Parameter space of the universal deterministic map algorithm.

A map is described by a dense track with one row per FRAME_MS (10 ms) frame:

    time_class[f]  0 = no event starts in frame f; k>0 = an event starts at f*FRAME_MS + (k-1) ms
    lane_code[f]   how the event's lanes are chosen (relative to the renderer's lane memory or explicit)
    stagger[f, m]  extra ms offset of chord member m (members = event lanes sorted ascending)
    ln_flag[f, m]  member m is a long note
    ln_len[f, m]   long note length in ms (end = start + ln_len)

plus a sparse ``overflow`` list for degenerate objects that the dense language cannot express
(negative times, zero-length LNs, two notes on one lane inside one 10 ms frame). Ranked maps are
expected to have an empty overflow list; ``hybrid_solvability_audit.py`` measures it.

Lane codes (25 classes): 10 relational operators applied to the previous chord masks
(SAME = jack, ALT = trill / two events ago, SHIFT/ROT = rolls, MIRROR, COMPLEMENT = jumptrills)
followed by 15 explicit masks. The solver always emits the first operator in LANE_OPS order that
reproduces the real chord, otherwise the explicit mask, which makes the encoding canonical and
universal. Because the renderer keeps the lane memory, the network never has to know which
lane the previous note was on: it predicts *relations* in one non-autoregressive pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint

PARAM_VERSION = "hybrid-params-v1"
FRAME_MS = 10
NUM_LANES = 4
MAX_MEMBERS = 4

LANE_OPS: Tuple[str, ...] = (
    "SAME",        # N = P            (jack / chordjack)
    "ALT",         # N = P2           (trill, jumptrill, any 2-periodic pattern)
    "SHIFT_R",     # N = P << 1       (roll / stair to the right)
    "SHIFT_L",     # N = P >> 1
    "ROT_R",       # N = rotl(P)      (roll that wraps 3 -> 0)
    "ROT_L",       # N = rotr(P)
    "SHIFT_R2",    # N = P << 2       (split patterns)
    "SHIFT_L2",    # N = P >> 2
    "MIRROR",      # N = reverse(P)   (symmetric patterns)
    "COMPLEMENT",  # N = ~P           (jumptrill halves, hand alternation)
)
N_OPS = len(LANE_OPS)
N_LANE_CODES = N_OPS + 15
# Explicit mask names list lanes left->right, e.g. "M1100" = lanes 0 and 1.
LANE_CODE_NAMES: Tuple[str, ...] = LANE_OPS + tuple(
    "M" + "".join("1" if (m >> c) & 1 else "0" for c in range(NUM_LANES)) for m in range(1, 16)
)

N_LN_BUCKETS = 48
LN_BUCKETS_PER_OCTAVE = 3
LN_MAX_MS = 2 ** (N_LN_BUCKETS // LN_BUCKETS_PER_OCTAVE)  # 65536 ms


def popcount4(mask: int) -> int:
    return bin(mask & 15).count("1")


def mask_lanes(mask: int) -> List[int]:
    return [c for c in range(NUM_LANES) if (mask >> c) & 1]


def lanes_to_mask(lanes: Sequence[int]) -> int:
    m = 0
    for c in lanes:
        m |= 1 << int(c)
    return m


def _reverse4(mask: int) -> int:
    out = 0
    for c in range(NUM_LANES):
        if (mask >> c) & 1:
            out |= 1 << (NUM_LANES - 1 - c)
    return out


def code_for_mask(mask: int) -> int:
    if not (1 <= mask <= 15):
        raise ValueError(f"explicit mask must be in 1..15, got {mask}")
    return N_OPS + mask - 1


def apply_lane_code(code: int, prev: int, prev2: int) -> int:
    """Returns the chord mask produced by ``code`` given the two previous chord masks (0 = invalid)."""
    code = int(code)
    if code >= N_OPS:
        if code >= N_LANE_CODES:
            return 0
        return code - N_OPS + 1
    if code < 0:
        return 0
    op = LANE_OPS[code]
    if op == "ALT":
        return prev2 & 15
    if prev == 0:
        return 0
    if op == "SAME":
        return prev
    if op == "SHIFT_R":
        return 0 if prev & 0b1000 else (prev << 1) & 15
    if op == "SHIFT_L":
        return 0 if prev & 0b0001 else prev >> 1
    if op == "ROT_R":
        return ((prev << 1) | (prev >> 3)) & 15
    if op == "ROT_L":
        return ((prev >> 1) | ((prev & 1) << 3)) & 15
    if op == "SHIFT_R2":
        return 0 if prev & 0b1100 else (prev << 2) & 15
    if op == "SHIFT_L2":
        return 0 if prev & 0b0011 else prev >> 2
    if op == "MIRROR":
        return _reverse4(prev)
    if op == "COMPLEMENT":
        return 0 if prev == 15 else (~prev) & 15
    return 0


# LANE_TABLE[prev, prev2, code] -> resulting mask (0 = invalid). Used by the fast decoder.
LANE_TABLE = np.zeros((16, 16, N_LANE_CODES), dtype=np.int16)
for _p in range(16):
    for _p2 in range(16):
        for _c in range(N_LANE_CODES):
            LANE_TABLE[_p, _p2, _c] = apply_lane_code(_c, _p, _p2)
CODE_POPCOUNT = np.array([[[popcount4(int(m)) for m in row] for row in plane] for plane in LANE_TABLE], dtype=np.int16)


def canonical_lane_code(mask: int, prev: int, prev2: int) -> int:
    """Canonical (first matching operator, else explicit) lane code for ``mask``."""
    if not (1 <= mask <= 15):
        raise ValueError(f"chord mask must be in 1..15, got {mask}")
    for code in range(N_OPS):
        if LANE_TABLE[prev & 15, prev2 & 15, code] == mask:
            return code
    return code_for_mask(mask)


def ln_length_to_bucket(length_ms: np.ndarray | int) -> Tuple[np.ndarray, np.ndarray]:
    """Log2 bucket (3 per octave) and in-bucket residual in [0, 1) for LN lengths >= 1 ms."""
    L = np.clip(np.asarray(length_ms, dtype=np.float64), 1.0, float(LN_MAX_MS) - 1e-6)
    pos = np.log2(L) * LN_BUCKETS_PER_OCTAVE
    bucket = np.clip(np.floor(pos), 0, N_LN_BUCKETS - 1).astype(np.int64)
    resid = np.clip(pos - bucket, 0.0, 1.0 - 1e-6)
    return bucket, resid


def ln_bucket_to_length(bucket: np.ndarray | int, resid: np.ndarray | float) -> np.ndarray:
    pos = np.asarray(bucket, dtype=np.float64) + np.clip(np.asarray(resid, dtype=np.float64), 0.0, 1.0)
    return np.maximum(1, np.round(np.power(2.0, pos / LN_BUCKETS_PER_OCTAVE))).astype(np.int64)


@dataclass
class ParamTrack:
    """Dense parameter track of one map (see module docstring)."""

    time_class: np.ndarray            # int16 [F]
    lane_code: np.ndarray             # int16 [F]   (-1 where no event)
    event_mask: np.ndarray            # uint8 [F]   (real chord mask; 0 where no event) - redundant check/target
    stagger: np.ndarray               # int16 [F, 4]
    ln_flag: np.ndarray               # bool  [F, 4]
    ln_len: np.ndarray                # int32 [F, 4]
    overflow: List[Note] = field(default_factory=list)
    timing_points: List[RawTimingPoint] = field(default_factory=list)
    frame_ms: int = FRAME_MS
    version: str = PARAM_VERSION

    @classmethod
    def empty(cls, num_frames: int, frame_ms: int = FRAME_MS) -> "ParamTrack":
        F = int(num_frames)
        return cls(
            time_class=np.zeros(F, dtype=np.int16),
            lane_code=np.full(F, -1, dtype=np.int16),
            event_mask=np.zeros(F, dtype=np.uint8),
            stagger=np.zeros((F, MAX_MEMBERS), dtype=np.int16),
            ln_flag=np.zeros((F, MAX_MEMBERS), dtype=bool),
            ln_len=np.zeros((F, MAX_MEMBERS), dtype=np.int32),
            frame_ms=frame_ms,
        )

    @property
    def num_frames(self) -> int:
        return int(self.time_class.shape[0])

    def event_frames(self) -> np.ndarray:
        return np.flatnonzero(self.time_class > 0)

    def num_events(self) -> int:
        return int((self.time_class > 0).sum())

    def chord_sizes(self) -> np.ndarray:
        """Chord size per frame (0 where no event)."""
        bits = (self.event_mask[:, None].astype(np.int64) >> np.arange(NUM_LANES)) & 1
        return bits.sum(axis=1)

    def save_npz(self, path: str | Path) -> None:
        ov = np.array([n.as_tuple() for n in self.overflow], dtype=np.int64).reshape(-1, 4)
        tp = np.array([t.as_tuple() for t in self.timing_points], dtype=np.float64).reshape(-1, 8)
        np.savez_compressed(
            path,
            time_class=self.time_class.astype(np.int16),
            lane_code=self.lane_code.astype(np.int16),
            event_mask=self.event_mask.astype(np.uint8),
            stagger=self.stagger.astype(np.int16),
            ln_flag=self.ln_flag.astype(bool),
            ln_len=self.ln_len.astype(np.int32),
            overflow=ov,
            timing_points=tp,
            frame_ms=np.array([self.frame_ms], dtype=np.int32),
            version=np.array([self.version]),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "ParamTrack":
        d = np.load(path, allow_pickle=False)
        overflow = [Note(int(r[0]), int(r[1]), int(r[2]), bool(r[3])) for r in d["overflow"]]
        tps = [
            RawTimingPoint(
                time=float(r[0]), beat_len=float(r[1]), meter=int(r[2]), sample_set=int(r[3]),
                sample_index=int(r[4]), volume=int(r[5]), uninherited=bool(r[6]), effects=int(r[7]),
            )
            for r in d["timing_points"]
        ]
        version = str(d["version"][0])
        if version != PARAM_VERSION:
            raise ValueError(f"ParamTrack version mismatch: file has '{version}', code expects '{PARAM_VERSION}'")
        return cls(
            time_class=d["time_class"].astype(np.int16),
            lane_code=d["lane_code"].astype(np.int16),
            event_mask=d["event_mask"].astype(np.uint8),
            stagger=d["stagger"].astype(np.int16),
            ln_flag=d["ln_flag"].astype(bool),
            ln_len=d["ln_len"].astype(np.int32),
            overflow=overflow,
            timing_points=tps,
            frame_ms=int(d["frame_ms"][0]),
            version=version,
        )

    def crop_arrays(self, start: int, length: int) -> Dict[str, np.ndarray]:
        """Training crop [start, start+length) with 'no event' padding outside the track."""
        F = self.num_frames
        out = {
            "time_class": np.zeros(length, dtype=np.int64),
            "lane_code": np.full(length, -1, dtype=np.int64),
            "event_mask": np.zeros(length, dtype=np.int64),
            "stagger": np.zeros((length, MAX_MEMBERS), dtype=np.int64),
            "ln_flag": np.zeros((length, MAX_MEMBERS), dtype=np.float32),
            "ln_len": np.zeros((length, MAX_MEMBERS), dtype=np.int64),
        }
        s0, s1 = max(0, start), min(F, start + length)
        if s1 > s0:
            d0, d1 = s0 - start, s1 - start
            out["time_class"][d0:d1] = self.time_class[s0:s1]
            out["lane_code"][d0:d1] = self.lane_code[s0:s1]
            out["event_mask"][d0:d1] = self.event_mask[s0:s1]
            out["stagger"][d0:d1] = self.stagger[s0:s1]
            out["ln_flag"][d0:d1] = self.ln_flag[s0:s1]
            out["ln_len"][d0:d1] = self.ln_len[s0:s1]
        return out


def describe_lane_code(code: int) -> str:
    if 0 <= code < N_LANE_CODES:
        return LANE_CODE_NAMES[code]
    return "NONE"
