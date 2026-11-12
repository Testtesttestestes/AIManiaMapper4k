"""
Event tokens of a map window for the autoregressive v5 decoder (as in Mapperatorinator / Yi et al. 2023: a chart is a
sequence, not one independent decision per frame).

One event = all notes starting at the same millisecond:

    TIME(ms inside the window)  MASK(lanes 1..15)  then per lane of the mask, lowest first:  TAP | LN LEN10(q) LENR(r)

* TIME is the exact millisecond inside the window (0 .. window_ms - 1);
* a long note's length is q x 10 + r ms (q < LEN_STEPS, r < 10): exact up to LEN_STEPS x 10 ms, longer ones are
  clipped (``encode`` reports them);
* a window starts with SOS and ends with EOS; ``prefix`` events (the end of the previous window) are part of the
  sequence at generation and are given, not predicted.

The sequence is exact: ``decode(encode(notes)) == notes`` for every map whose long notes are shorter than the cap
(tests/test_v5.py runs it over the dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..hybrid.beatmap import Note

NUM_LANES = 4
PAD, SOS, EOS, TAP, LN = 0, 1, 2, 3, 4
N_SPECIAL = 5


@dataclass(frozen=True)
class Vocab:
    window_ms: int = 10240
    len_steps: int = 1000            # LN lengths up to 10 s

    @property
    def time0(self) -> int:
        return N_SPECIAL

    @property
    def mask0(self) -> int:
        return self.time0 + self.window_ms

    @property
    def len0(self) -> int:
        return self.mask0 + 15

    @property
    def lenr0(self) -> int:
        return self.len0 + self.len_steps

    @property
    def size(self) -> int:
        return self.lenr0 + 10

    # --- token kinds (for constrained sampling) ---------------------------------------------------------------------
    def is_time(self, tok: np.ndarray) -> np.ndarray:
        return (tok >= self.time0) & (tok < self.mask0)

    def kind(self, tok: int) -> str:
        if tok < N_SPECIAL:
            return ("pad", "sos", "eos", "tap", "ln")[tok]
        if tok < self.mask0:
            return "time"
        if tok < self.len0:
            return "mask"
        if tok < self.lenr0:
            return "len"
        return "lenr"


def encode(notes: Sequence[Note], start_ms: int, vocab: Vocab, add_eos: bool = True) -> Tuple[List[int], List[int], int]:
    """(tokens, event start index of every event in ``tokens``, number of clipped long notes) of the notes starting in
    [start_ms, start_ms + window_ms). Tokens start with SOS."""
    toks: List[int] = [SOS]
    ev_pos: List[int] = []
    clipped = 0
    by_t = {}
    for n in notes:
        if start_ms <= n.start < start_ms + vocab.window_ms:
            by_t.setdefault(int(n.start), []).append(n)
    for t in sorted(by_t):
        ns = sorted(by_t[t], key=lambda n: n.col)
        mask = 0
        for n in ns:
            mask |= 1 << int(n.col)
        ev_pos.append(len(toks))
        toks.append(vocab.time0 + (t - start_ms))
        toks.append(vocab.mask0 + mask - 1)
        for n in ns:
            if n.ln:
                L = int(n.end - n.start)
                q, r = divmod(max(1, L), 10)
                if q >= vocab.len_steps:
                    q, r = vocab.len_steps - 1, 9
                    clipped += 1
                toks += [LN, vocab.len0 + q, vocab.lenr0 + r]
            else:
                toks.append(TAP)
    if add_eos:
        toks.append(EOS)
    return toks, ev_pos, clipped


def decode(tokens: Sequence[int], start_ms: int, vocab: Vocab) -> List[Note]:
    """Notes of a token sequence (invalid or incomplete events at the end are dropped)."""
    out: List[Note] = []
    i = 0
    toks = list(tokens)
    while i < len(toks):
        tok = toks[i]
        if tok == EOS:
            break
        if not (vocab.time0 <= tok < vocab.mask0):
            i += 1
            continue
        t = start_ms + tok - vocab.time0
        if i + 1 >= len(toks) or not (vocab.mask0 <= toks[i + 1] < vocab.len0):
            break
        mask = toks[i + 1] - vocab.mask0 + 1
        lanes = [c for c in range(NUM_LANES) if mask >> c & 1]
        j = i + 2
        ev: List[Note] = []
        ok = True
        for c in lanes:
            if j < len(toks) and toks[j] == TAP:
                ev.append(Note(t, c, t, False))
                j += 1
            elif j + 2 < len(toks) and toks[j] == LN and vocab.len0 <= toks[j + 1] < vocab.lenr0 \
                    and vocab.lenr0 <= toks[j + 2] < vocab.size:
                L = (toks[j + 1] - vocab.len0) * 10 + (toks[j + 2] - vocab.lenr0)
                ev.append(Note(t, c, t + max(1, L), True))
                j += 3
            else:
                ok = False
                break
        if not ok:
            break
        out.extend(ev)
        i = j
    return out


def allowed_next(prev: Sequence[int], vocab: Vocab, last_time: int, members_left: int, in_ln: int,
                 min_time: int = 0) -> Tuple[Tuple[int, int], ...]:
    """Token ranges [lo, hi) the grammar allows after ``prev`` (used to mask the logits while sampling).

    ``last_time``: TIME token of the last event (-1 = none); ``members_left``: lanes of the current mask still to be
    given; ``in_ln``: 0, or 1 / 2 after LN / LEN10 of the current member; ``min_time``: no event before this ms."""
    if in_ln == 1:
        return ((vocab.len0, vocab.lenr0),)
    if in_ln == 2:
        return ((vocab.lenr0, vocab.size),)
    last = prev[-1] if prev else SOS
    if vocab.time0 <= last < vocab.mask0:
        return ((vocab.mask0, vocab.len0),)
    if members_left > 0:
        return ((TAP, TAP + 1), (LN, LN + 1))
    lo = vocab.time0 + max(min_time, last_time + 1)
    return ((lo, vocab.mask0), (EOS, EOS + 1)) if lo < vocab.mask0 else ((EOS, EOS + 1),)


class GrammarState:
    """Tracks where a sequence is in the event grammar while sampling token by token."""

    def __init__(self, vocab: Vocab, min_time: int = 0):
        self.v = vocab
        self.last_time = -1
        self.members_left = 0
        self.in_ln = 0
        self.min_time = int(min_time)
        self.done = False

    def ranges(self, prev: Sequence[int]):
        return allowed_next(prev, self.v, self.last_time, self.members_left, self.in_ln, self.min_time)

    def push(self, tok: int) -> None:
        v = self.v
        if tok == EOS:
            self.done = True
        elif v.time0 <= tok < v.mask0:
            self.last_time = tok - v.time0
        elif v.mask0 <= tok < v.len0:
            self.members_left = bin(tok - v.mask0 + 1).count("1")
        elif tok == TAP:
            self.members_left -= 1
        elif tok == LN:
            self.in_ln = 1
        elif v.len0 <= tok < v.lenr0:
            self.in_ln = 2
        elif v.lenr0 <= tok < v.size:
            self.in_ln = 0
            self.members_left -= 1


def time_of_tokens(tokens: np.ndarray, vocab: Vocab) -> np.ndarray:
    """For every token the ms (inside the window) of the event it belongs to (-1 before the first event)."""
    t = np.full(len(tokens), -1, dtype=np.int64)
    cur = -1
    for i, tok in enumerate(np.asarray(tokens).tolist()):
        if vocab.time0 <= tok < vocab.mask0:
            cur = tok - vocab.time0
        t[i] = cur
    return t


def prefix_split(tokens: Sequence[int], ev_pos: Sequence[int], times_ms: Sequence[int], cut_ms: int) -> int:
    """Index of the first token of the first event at or after ``cut_ms`` (len(tokens) - 1 = the EOS when none)."""
    for p, t in zip(ev_pos, times_ms):
        if t >= cut_ms:
            return int(p)
    return max(1, len(tokens) - 1)


def event_times(notes: Sequence[Note], start_ms: int, window_ms: int) -> List[int]:
    return sorted({int(n.start) - start_ms for n in notes if start_ms <= n.start < start_ms + window_ms})


def clip_ln_report(notes: Sequence[Note], vocab: Optional[Vocab] = None) -> int:
    v = vocab or Vocab()
    return sum(1 for n in notes if n.ln and n.end - n.start >= v.len_steps * 10)
