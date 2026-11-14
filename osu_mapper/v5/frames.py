"""
The canvas of the v5 network: a map as a picture of 10 ms frames x 4 lanes (as the v4 canvas: the network sees the
whole window at once, some frames hidden under a mask, and restores them).

Every frame of every lane has a state and up to two millisecond offsets inside the frame:

    EMPTY  nothing
    TAP    a tap starts here                (start offset 0..9 ms)
    HEAD   a long note starts here          (start offset)
    HOLD   a long note is held through this frame
    TAIL   a long note ends here            (end offset 0..9 ms)

A long note is HEAD, HOLD ... HOLD, TAIL: the network sees how far every hold reaches and what comes after it, in
both directions. Lanes are independent (chords are several lanes starting in the same frame, each with its own
offset, so staggered chords are exact too).

``encode`` / ``decode`` are exact for every map whose long notes are at least one frame long and whose lanes never
have a note starting in the frame where a long note of that lane ends (both are checked by tests/test_v5.py on the
dataset; anything else moves by less than 10 ms and is counted).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..hybrid.beatmap import Note

NUM_LANES = 4
EMPTY, TAP, HEAD, HOLD, TAIL = 0, 1, 2, 3, 4
N_STATES = 5
MASK_STATE = N_STATES               # input token of a hidden frame
NO_OFF = 10                          # offset input of a lane without a start / end in that frame
MASK_OFF = 11                        # offset input of a hidden frame


def encode(notes: Sequence[Note], start_ms: int, n_frames: int, frame_ms: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(state [F, 4] int8, start offset [F, 4] int8 (-1 = none), end offset [F, 4] int8 (-1 = none), lossy notes) of
    the window [start_ms, start_ms + n_frames x frame_ms). Long notes that began before the window are HOLD frames
    from its first frame; those that end after it have no TAIL inside it."""
    st = np.zeros((n_frames, NUM_LANES), dtype=np.int8)
    so = np.full((n_frames, NUM_LANES), -1, dtype=np.int8)
    eo = np.full((n_frames, NUM_LANES), -1, dtype=np.int8)
    end_ms = start_ms + n_frames * frame_ms
    lossy = 0
    for n in sorted(notes):
        c = int(n.col)
        last = n.end if n.ln else n.start
        if last < start_ms or n.start >= end_ms:
            continue
        f0 = (int(n.start) - start_ms) // frame_ms
        if not n.ln:
            if 0 <= f0 < n_frames:
                st[f0, c] = TAP
                so[f0, c] = (int(n.start) - start_ms) % frame_ms
            continue
        f1 = (int(n.end) - start_ms) // frame_ms
        if f1 <= f0:                                    # shorter than its frame: the tail goes to the next frame
            f1 = f0 + 1
            lossy += 1
        if 0 <= f0 < n_frames:
            st[f0, c] = HEAD
            so[f0, c] = (int(n.start) - start_ms) % frame_ms
        a, b = max(0, f0 + 1), min(n_frames, f1)
        if b > a:
            st[a:b, c] = HOLD
        if 0 <= f1 < n_frames:
            if st[f1, c] in (TAP, HEAD):                # another note starts in the tail's frame: tail one earlier
                lossy += 1
                if f1 - 1 > f0 and f1 - 1 >= 0:
                    st[f1 - 1, c] = TAIL
                    eo[f1 - 1, c] = frame_ms - 1
            else:
                st[f1, c] = TAIL
                eo[f1, c] = (int(n.end) - start_ms) % frame_ms if (int(n.end) - start_ms) // frame_ms == f1 else 0
    return st, so, eo, lossy


def decode(state: np.ndarray, start_off: np.ndarray, end_off: np.ndarray, start_ms: int = 0, frame_ms: int = 10,
           release_gap_ms: int = 30) -> List[Note]:
    """Notes of a canvas. Repairs what a sampled canvas may get wrong: a HOLD with no open long note opens one; a long
    note that is still open when a new note starts on its lane (or at the end) is closed ``release_gap_ms`` before it."""
    F = state.shape[0]
    out: List[Note] = []
    for c in range(state.shape[1]):
        open_at: Optional[int] = None
        for f in range(F):
            s = int(state[f, c])
            t0 = start_ms + f * frame_ms
            if s in (TAP, HEAD):
                t = t0 + max(0, int(start_off[f, c]))
                if open_at is not None:                 # close the open hold before this note
                    out.append(Note(open_at, c, max(open_at + 1, t - release_gap_ms), True))
                    open_at = None
                if s == TAP:
                    out.append(Note(t, c, t, False))
                else:
                    open_at = t
            elif s == HOLD:
                # a hold with no head: opens a long note only when it really runs on (the next frame holds or ends)
                if open_at is None and f + 1 < F and int(state[f + 1, c]) in (HOLD, TAIL) \
                        and (f == 0 or int(state[f - 1, c]) in (EMPTY, TAIL, TAP)):
                    open_at = t0
            elif s == TAIL:
                if open_at is not None:
                    e = t0 + max(0, int(end_off[f, c]))
                    out.append(Note(open_at, c, max(open_at + 1, e), True))
                    open_at = None
        if open_at is not None:
            out.append(Note(open_at, c, max(open_at + 1, start_ms + F * frame_ms - 1), True))
    return sorted(out)


def inputs(state: np.ndarray, start_off: np.ndarray, end_off: np.ndarray, hidden: np.ndarray):
    """Model input tokens of a canvas with the frames ``hidden`` [F] bool under the mask: (state_in, start_in, end_in)
    [F, 4] int64, offsets NO_OFF where a lane has none and MASK_OFF where hidden."""
    s = state.astype(np.int64).copy()
    a = np.where(start_off >= 0, start_off, NO_OFF).astype(np.int64)
    e = np.where(end_off >= 0, end_off, NO_OFF).astype(np.int64)
    s[hidden] = MASK_STATE
    a[hidden] = MASK_OFF
    e[hidden] = MASK_OFF
    return s, a, e


def frame_differs(a: Tuple[np.ndarray, np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """[F] bool: frames where two canvases differ (any lane, state or offset)."""
    return ((a[0] != b[0]) | (a[1] != b[1]) | (a[2] != b[2])).any(axis=1)
