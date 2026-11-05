"""
Learned mapping patterns: what real mappers place where, at every difficulty.

A count model (n-gram with back-off) over the whole dataset, conditioned on the star rating. It learns

* **lanes** - P(chord mask | previous two chord masks, time since the previous chord, star rating):
  which rolls, trills, jumps, jacks, splits, hand alternations real maps use after each pattern and how
  that changes with speed and difficulty (a 1/4 jack at 4* is rare, a jumptrill after a [12] chord is
  common...). Every map is also counted mirrored (left-right), so the model is symmetric.
* **chords** - P(chord size | star rating, metric position, gap): how many 2-, 3-, 4-note chords a
  difficulty has and where (downbeats vs. fills).
* **rhythm** - P(a tick of beat divisor d carries a note | star rating): at 2* mappers use 1/1 and 1/2,
  at 5* most 1/4 ticks, 1/3 only in swing parts...
* **long notes** - P(long note | star rating, gap to the next chord) and the LN share per difficulty.

Unlike the network it cannot overfit a few songs: it stores frequencies of local patterns over all maps
(millions of events) and smooths sparse contexts by backing off to coarser ones. The composer uses it as
its main "is this appropriate and nice at this difficulty" score, the network adds what is specific to
the audio.

    python learn_patterns.py --data dataset_hybrid          # from the prepared dataset (recommended)
    python learn_patterns.py --input_dir "sample beatmaps"  # straight from .osz archives (no audio needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint

PATTERN_VERSION = "patterns-v2"
SR_BINS = 12                                               # 0-1*, 1-2*, ..., 10-11*, 11+*
GAP_EDGES_MS = np.array([45, 70, 95, 125, 160, 210, 280, 380, 550, 800, 1500], dtype=np.float64)
N_GAP = GAP_EDGES_MS.size + 1
DIV_CLASSES: Tuple[str, ...] = ("1/1", "1/2", "1/4", "1/3-1/6", "1/8", "1/12-1/16", "off-grid")
N_DIV = len(DIV_CLASSES)
DIV_TICKS_PER_BEAT = np.array([1, 1, 2, 4, 4, 12, 1], dtype=np.float64)  # ticks of each class per beat
POP = np.array([bin(m).count("1") for m in range(16)])
REPEAT_BIN_MS = 5
REPEAT_BINS = 400                                          # 0..2 s
LN_SPAN_BINS = 33                                          # 0..31 chords during a hold, 32 = more


def gap_bin(gap_ms: float) -> int:
    return int(np.searchsorted(GAP_EDGES_MS, float(gap_ms), side="right"))


def sr_bin(sr: float) -> int:
    return int(np.clip(math.floor(float(sr)), 0, SR_BINS - 1))


def _mirror_mask(m: int) -> int:
    return ((m & 1) << 3) | ((m & 2) << 1) | ((m & 4) >> 1) | ((m & 8) >> 3)


MIRROR = np.array([_mirror_mask(m) for m in range(16)], dtype=np.int64)


def div_class(t: float, reds: Sequence[RawTimingPoint], tol_ms: float = 3.0) -> int:
    if not reds:
        return N_DIV - 1
    times = [r.time for r in reds]
    i = max(0, int(np.searchsorted(times, t, side="right")) - 1)
    bl = reds[i].beat_len
    beats = (t - reds[i].time) / bl
    for d, cls in ((1, 0), (2, 1), (4, 2), (3, 3), (6, 3), (8, 4), (12, 5), (16, 5)):
        if abs(round(beats * d) / d - beats) * bl <= tol_ms:
            return cls
    return N_DIV - 1


def metric_bin(cls: int) -> int:
    return 0 if cls == 0 else (1 if cls in (1, 2) else 2)


def chord_events(notes: Sequence[Note]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(start times, chord masks, LN-member masks) of the map's chords, sorted by time."""
    groups: Dict[int, List[Note]] = {}
    for n in notes:
        groups.setdefault(int(n.start), []).append(n)
    times = np.array(sorted(groups), dtype=np.int64)
    masks = np.array([sum(1 << n.col for n in groups[t]) & 15 for t in times], dtype=np.int64)
    lns = np.array([sum(1 << n.col for n in groups[t] if n.ln) & 15 for t in times], dtype=np.int64)
    return times, masks, lns


@dataclass
class PatternModel:
    lane: np.ndarray        # [SR, GAP, 16 prev, 16 prev2, 16 mask] counts
    chord: np.ndarray       # [SR, 3 metric, GAP, 4 sizes]
    rhythm_hits: np.ndarray # [SR, DIV] events on ticks of the class
    rhythm_beats: np.ndarray  # [SR] beats covered by maps of the bin (denominator)
    ln: np.ndarray          # [SR, GAP(next), 2] tap / LN member counts
    maps: np.ndarray        # [SR] maps per bin
    repeat_gap: Optional[np.ndarray] = None  # [SR, REPEAT_BINS] same-lane gaps (previous press or LN release -> next note), 5 ms bins
    ln_span: Optional[np.ndarray] = None     # [SR, LN_SPAN_BINS] chords that start while a long note is held
    density_cv: Optional[np.ndarray] = None  # [SR, 2] sum / count of the density variation (std/mean of notes per 4 s)
    event_gap: Optional[np.ndarray] = None   # [SR, REPEAT_BINS] chord-to-chord gaps (any lanes), 5 ms bins
    intensity_fit: Optional[np.ndarray] = None  # [SR, 3] least squares of log section density vs log audio intensity: sum xy, sum xx, n
    version: str = PATTERN_VERSION

    # ------------------------------------------------------------------------------------------
    @classmethod
    def empty(cls) -> "PatternModel":
        return cls(
            lane=np.zeros((SR_BINS, N_GAP, 16, 16, 16), np.float32),
            chord=np.zeros((SR_BINS, 3, N_GAP, 4), np.float32),
            rhythm_hits=np.zeros((SR_BINS, N_DIV), np.float64),
            rhythm_beats=np.zeros(SR_BINS, np.float64),
            ln=np.zeros((SR_BINS, N_GAP, 2), np.float64),
            maps=np.zeros(SR_BINS, np.int64),
            repeat_gap=np.zeros((SR_BINS, REPEAT_BINS), np.float64),
            ln_span=np.zeros((SR_BINS, LN_SPAN_BINS), np.float64),
            density_cv=np.zeros((SR_BINS, 2), np.float64),
            intensity_fit=np.zeros((SR_BINS, 3), np.float64),
            event_gap=np.zeros((SR_BINS, REPEAT_BINS), np.float64),
        )

    def add_map(self, notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], star_rating: float) -> None:
        times, masks, lns = chord_events(notes)
        if times.size < 8:
            return
        s = sr_bin(star_rating)
        self.maps[s] += 1
        reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
        gaps = np.diff(times, prepend=times[0] - 10 ** 6)
        gnext = np.append(np.diff(times), 10 ** 6)
        gb = np.searchsorted(GAP_EDGES_MS, gaps, side="right")
        gnb = np.searchsorted(GAP_EDGES_MS, gnext, side="right")
        cls = np.array([div_class(float(t), reds) for t in times], dtype=np.int64)
        for mirror in (False, True):
            mk = MIRROR[masks] if mirror else masks
            prev = np.concatenate([[0], mk[:-1]])
            prev2 = np.concatenate([[0, 0], mk[:-2]])
            np.add.at(self.lane, (s, gb, prev, prev2, mk), 0.5)
        sizes = np.clip(POP[masks] - 1, 0, 3)
        np.add.at(self.chord, (s, np.array([metric_bin(c) for c in cls]), gb, sizes), 1.0)
        np.add.at(self.rhythm_hits, (s, cls), 1.0)
        if reds:
            span = float(times[-1] - times[0])
            beat = float(np.median([r.beat_len for r in reds]))
            self.rhythm_beats[s] += span / max(1.0, beat)
        # how fast real maps of this difficulty go at all (bursts of 1/8, 1/12, 1/16)
        if self.event_gap is None:
            self.event_gap = np.zeros((SR_BINS, REPEAT_BINS), np.float64)
        np.add.at(self.event_gap[s], np.clip(np.diff(times) // REPEAT_BIN_MS, 0, REPEAT_BINS - 1), 1.0)
        # how fast real maps of this difficulty repeat a lane (jacks), measured from the press / LN release
        by_lane: Dict[int, List[Note]] = {}
        for n in notes:
            by_lane.setdefault(int(n.col), []).append(n)
        for lane_notes in by_lane.values():
            lane_notes.sort(key=lambda n: n.start)
            for a, b in zip(lane_notes[:-1], lane_notes[1:]):
                g = b.start - (a.end if a.ln else a.start)
                if g >= 0:
                    self.repeat_gap[s, min(REPEAT_BINS - 1, g // REPEAT_BIN_MS)] += 1
        # how unevenly real maps of this difficulty distribute their notes (quiet parts vs. drops)
        win = 4000
        n_win = int((times[-1] - times[0]) // win)
        if n_win >= 4:
            cnt = np.bincount(((times - times[0]) // win).clip(0, n_win - 1), minlength=n_win).astype(np.float64)
            self.density_cv[s] += (float(cnt.std() / max(1e-6, cnt.mean())), 1.0)
        # how many chords real mappers start while a long note is held (dense holds vs. isolated holds)
        for n in notes:
            if n.ln and n.end > n.start:
                inside = int(np.searchsorted(times, n.end, side="left") - np.searchsorted(times, n.start, side="right"))
                self.ln_span[s, min(LN_SPAN_BINS - 1, inside)] += 1
        n_ln = POP[lns]
        np.add.at(self.ln, (s, gnb, 1), n_ln)
        np.add.at(self.ln, (s, gnb, 0), POP[masks] - n_ln)

    # ------------------------------------------------------------------------------------------
    def _sr_weights(self, sr: float) -> List[Tuple[int, float]]:
        x = float(np.clip(sr - 0.5, 0.0, SR_BINS - 1.0))
        a = int(math.floor(x))
        w = x - a
        out = [(a, 1.0 - w)]
        if w > 0 and a + 1 < SR_BINS:
            out.append((a + 1, w))
        # empty bins borrow from the nearest filled one
        res = []
        filled = np.flatnonzero(self.maps > 0)
        for b, wt in out:
            if self.maps[b] == 0 and filled.size:
                b = int(filled[np.argmin(np.abs(filled - b))])
            res.append((b, wt))
        return res

    def lane_logprob(self, sr: float, gap_ms: float, prev: int, prev2: int, size: int, alpha: float = 4.0) -> np.ndarray:
        """log P(mask | context) over all 16 masks, restricted to masks with ``size`` lanes (-inf elsewhere)."""
        g = gap_bin(gap_ms)
        valid = POP == size
        out = np.zeros(16)
        if not hasattr(self, "_m_uni"):
            self._m_uni = self.lane.sum(axis=(1, 2, 3)) + self.lane.sum(axis=(0, 1, 2, 3))[None] * 0.05  # [SR, 16]
            self._m_gap = self.lane.sum(axis=(2, 3))                                                   # [SR, GAP, 16]
            self._m_prev = self.lane.sum(axis=3)                                                       # [SR, GAP, 16, 16]
        for b, wt in self._sr_weights(sr):
            c_uni = self._m_uni[b]
            c_gap = self._m_gap[b, g]
            c_prev = self._m_prev[b, g, prev]
            c_full = self.lane[b, g, prev, prev2]
            p = (c_uni + 1e-3) * valid
            p = p / max(1e-12, p.sum())
            for c in (c_gap, c_prev, c_full):
                c = c * valid
                n = c.sum()
                p = (c + alpha * p) / (n + alpha)
            out += wt * p
        out = out * valid
        with np.errstate(divide="ignore"):
            return np.where(valid, np.log(np.maximum(out / max(1e-12, out.sum()), 1e-9)), -np.inf)

    def chord_distribution(self, sr: float, metric: int = 1, gap_ms: Optional[float] = None) -> np.ndarray:
        """P(chord size 1..4 | star rating [, metric position, gap])."""
        p = np.zeros(4)
        for b, wt in self._sr_weights(sr):
            c = self.chord[b, metric] if metric is not None else self.chord[b].sum(axis=0)
            c = c[gap_bin(gap_ms)] * 1.0 + 0.2 * c.sum(axis=0) if gap_ms is not None else c.sum(axis=0)
            c = c + 0.5
            p += wt * c / c.sum()
        return p / p.sum()

    def chord_marginal(self, sr: float) -> np.ndarray:
        p = np.zeros(4)
        for b, wt in self._sr_weights(sr):
            c = self.chord[b].sum(axis=(0, 1)) + 0.5
            p += wt * c / c.sum()
        return p / p.sum()

    def tick_usage(self, sr: float) -> np.ndarray:
        """P(a tick of each divisor class carries a note) at this star rating."""
        u = np.zeros(N_DIV)
        for b, wt in self._sr_weights(sr):
            beats = max(1.0, self.rhythm_beats[b])
            u += wt * np.clip(self.rhythm_hits[b] / (beats * DIV_TICKS_PER_BEAT), 1e-4, 0.999)
        return u

    def repeat_floor(self, sr: float, quantile: float = 0.01, after_ln: bool = False) -> Optional[float]:
        """Lane-repeat gap (ms) that only ``quantile`` of the repeats in real maps of this difficulty are faster than."""
        if self.repeat_gap is None or self.repeat_gap.sum() == 0:
            return None
        h = np.zeros(REPEAT_BINS)
        for b, wt in self._sr_weights(sr):
            hb = self.repeat_gap[b]
            if hb.sum() > 0:
                h += wt * hb / hb.sum()
        if h.sum() == 0:
            return None
        c = np.cumsum(h) / h.sum()
        return float(np.searchsorted(c, quantile) * REPEAT_BIN_MS)

    def _hist(self, table: Optional[np.ndarray], sr: float) -> Optional[np.ndarray]:
        if table is None or table.sum() == 0:
            return None
        h = np.zeros(table.shape[1])
        for b, wt in self._sr_weights(sr):
            if table[b].sum() > 0:
                h += wt * table[b] / table[b].sum()
        return h if h.sum() > 0 else None

    def gap_share_below(self, sr: float, gap_ms: float) -> Optional[float]:
        """Share of chord-to-chord gaps in real maps of this difficulty that are shorter than ``gap_ms``."""
        h = self._hist(self.event_gap, sr)
        if h is None:
            return None
        return float(h[: max(0, int(gap_ms) // REPEAT_BIN_MS)].sum() / h.sum())

    def event_gap_floor(self, sr: float, quantile: float = 0.005) -> Optional[float]:
        """Chord-to-chord gap (ms) that only ``quantile`` of real gaps at this difficulty are shorter than."""
        h = self._hist(self.event_gap, sr)
        if h is None:
            return None
        return float(np.searchsorted(np.cumsum(h) / h.sum(), quantile) * REPEAT_BIN_MS)

    def repeat_cdf(self, sr: float) -> Optional[np.ndarray]:
        """[REPEAT_BINS] share of lane repeats in real maps of this difficulty that are faster than each 5 ms bin."""
        if self.repeat_gap is None or self.repeat_gap.sum() == 0:
            return None
        h = np.zeros(REPEAT_BINS)
        for b, wt in self._sr_weights(sr):
            hb = self.repeat_gap[b]
            if hb.sum() > 0:
                h += wt * hb / hb.sum()
        return np.cumsum(h) / max(1e-12, h.sum())

    def ln_span_limit(self, sr: float, quantile: float = 0.9) -> Optional[int]:
        """Number of chords that ``quantile`` of the long notes of real maps of this difficulty hold over at most."""
        if self.ln_span is None or self.ln_span.sum() == 0:
            return None
        h = np.zeros(LN_SPAN_BINS)
        for b, wt in self._sr_weights(sr):
            hb = self.ln_span[b]
            if hb.sum() > 0:
                h += wt * hb / hb.sum()
        if h.sum() == 0:
            return None
        return int(np.searchsorted(np.cumsum(h) / h.sum(), quantile))

    def density_variation(self, sr: float) -> Optional[float]:
        """Typical std/mean of the note count in 4 s windows for real maps of this difficulty."""
        if self.density_cv is None or self.density_cv[:, 1].sum() == 0:
            return None
        tot = wsum = 0.0
        for b, wt in self._sr_weights(sr):
            if self.density_cv[b, 1] > 0:
                tot += wt * self.density_cv[b, 0] / self.density_cv[b, 1]
                wsum += wt
        return tot / wsum if wsum > 0 else None

    def add_sections(self, notes: Sequence[Note], sections, star_rating: float) -> None:
        """Audio intensity of every section (composer analysis) vs the note density real mappers gave it."""
        if self.intensity_fit is None:
            self.intensity_fit = np.zeros((SR_BINS, 3), np.float64)
        secs = [x for x in sections if x.intensity > 0 and x.dur_s >= 2.0]
        if len(secs) < 3 or not notes:
            return
        starts = np.array(sorted(n.start for n in notes), dtype=np.int64)
        dens = np.array([(np.searchsorted(starts, x.end_ms) - np.searchsorted(starts, x.start_ms)) / x.dur_s for x in secs])
        base = np.array([0.3 + 0.7 * x.intensity for x in secs])
        w = np.array([x.dur_s for x in secs])
        ok = dens > 0
        if ok.sum() < 3:
            return
        y = np.log(dens[ok]) - np.average(np.log(dens[ok]), weights=w[ok])
        x = np.log(base[ok]) - np.average(np.log(base[ok]), weights=w[ok])
        b = sr_bin(star_rating)
        self.intensity_fit[b] += (float((w[ok] * x * y).sum()), float((w[ok] * x * x).sum()), 1.0)

    def intensity_exponent(self, sr: float, min_maps: int = 3) -> Optional[float]:
        """How strongly real maps of this difficulty follow the audio intensity: section density ~ (0.3 + 0.7 i)^e."""
        if self.intensity_fit is None:
            return None
        xy = xx = n = 0.0
        for b, wt in self._sr_weights(sr):
            xy += wt * self.intensity_fit[b, 0]
            xx += wt * self.intensity_fit[b, 1]
            n += wt * self.intensity_fit[b, 2]
        if n < min_maps * 0.5 or xx <= 1e-9:
            return None
        return float(np.clip(xy / xx, 0.0, 6.0))

    def max_chord(self, sr: float, min_share: float = 0.003) -> int:
        """Largest chord size that real maps of this difficulty actually use (share >= min_share)."""
        p = self.chord_marginal(sr)
        ok = [k + 1 for k in range(4) if p[k] >= min_share]
        return max(ok) if ok else 1

    def ln_share(self, sr: float) -> float:
        tot = 0.0
        for b, wt in self._sr_weights(sr):
            c = self.ln[b].sum(axis=0)
            tot += wt * (c[1] + 0.5) / (c.sum() + 1.0)
        return float(tot)

    def ln_prob(self, sr: float, gap_next_ms: float) -> float:
        tot = 0.0
        g = gap_bin(gap_next_ms)
        for b, wt in self._sr_weights(sr):
            c = self.ln[b, g]
            tot += wt * (c[1] + 0.5) / (c.sum() + 1.0)
        return float(tot)

    # ------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, lane=self.lane, chord=self.chord, rhythm_hits=self.rhythm_hits,
                            rhythm_beats=self.rhythm_beats, ln=self.ln, maps=self.maps,
                            repeat_gap=self.repeat_gap if self.repeat_gap is not None else np.zeros((SR_BINS, REPEAT_BINS)),
                            ln_span=self.ln_span if self.ln_span is not None else np.zeros((SR_BINS, LN_SPAN_BINS)),
                            density_cv=self.density_cv if self.density_cv is not None else np.zeros((SR_BINS, 2)),
                            intensity_fit=self.intensity_fit if self.intensity_fit is not None else np.zeros((SR_BINS, 3)),
                            event_gap=self.event_gap if self.event_gap is not None else np.zeros((SR_BINS, REPEAT_BINS)),
                            version=np.array([self.version]))

    @classmethod
    def load(cls, path: str | Path) -> "PatternModel":
        d = np.load(path, allow_pickle=False)
        v = str(d["version"][0])
        if v != PATTERN_VERSION:
            raise ValueError(f"pattern model '{path}' has version {v}, code expects {PATTERN_VERSION}: re-run learn_patterns.py")
        return cls(lane=d["lane"], chord=d["chord"], rhythm_hits=d["rhythm_hits"], rhythm_beats=d["rhythm_beats"],
                   ln=d["ln"], maps=d["maps"], repeat_gap=d["repeat_gap"] if "repeat_gap" in d else None,
                   ln_span=d["ln_span"] if "ln_span" in d else None,
                   density_cv=d["density_cv"] if "density_cv" in d else None,
                   intensity_fit=d["intensity_fit"] if "intensity_fit" in d else None,
                   event_gap=d["event_gap"] if "event_gap" in d else None, version=v)

    def summary(self) -> str:
        rows = [f"pattern model: {int(self.maps.sum())} maps, {int(self.lane.sum())} chord transitions"]
        for b in range(SR_BINS):
            if self.maps[b] == 0:
                continue
            ch = self.chord[b].sum(axis=(0, 1))
            ch = ch / max(1.0, ch.sum())
            u = self.tick_usage(b + 0.5)
            L = self.lane[b]
            tot = max(1.0, L.sum())
            jack = sum(L[:, p, :, p].sum() for p in range(1, 16)) / tot
            fast_jack = sum(L[:4, p, :, p].sum() for p in range(1, 16)) / tot
            rows.append(f"  {b}-{b + 1}*: {int(self.maps[b]):4d} maps | chords 1/2/3/4 = {ch[0]:.2f}/{ch[1]:.2f}/{ch[2]:.2f}/{ch[3]:.2f} "
                        f"| tick usage 1/1 {u[0]:.2f} 1/2 {u[1]:.2f} 1/4 {u[2]:.2f} 1/3 {u[3]:.2f} 1/8 {u[4]:.2f} "
                        f"| fastest 0.5% of gaps < {self.event_gap_floor(b + 0.5) or 0:.0f} ms "
                        f"| repeats {100 * jack:.1f}% (<125ms {100 * fast_jack:.1f}%) | lane repeat 1%/5% faster than "
                        f"{self.repeat_floor(b + 0.5, 0.01) or 0:.0f}/{self.repeat_floor(b + 0.5, 0.05) or 0:.0f} ms "
                        f"| LN {100 * self.ln_share(b + 0.5):.0f}%, 90% of holds span <= {self.ln_span_limit(b + 0.5) or 0} chords"
                        f" | density variation {self.density_variation(b + 0.5) or 0:.2f}"
                        + (f", follows audio intensity ^{e:.2f}" if (e := self.intensity_exponent(b + 0.5)) is not None else ""))
        return "\n".join(rows)


def default_pattern_path() -> Path:
    return Path("checkpoints") / "hybrid" / "patterns.npz"


def load_default(path: Optional[str] = None, log=print) -> Optional[PatternModel]:
    p = Path(path) if path else default_pattern_path()
    if not p.exists():
        if path:
            raise FileNotFoundError(f"pattern model '{p}' not found (python learn_patterns.py --data dataset_hybrid)")
        return None
    m = PatternModel.load(p)
    log(f"[patterns] {p}: {int(m.maps.sum())} maps")
    return m


# ----------------------------------------------------------------------------------------------
# density graph (the stepped "intensity" bars of the osu! editor timeline)
# ----------------------------------------------------------------------------------------------

def density_profile(notes: Sequence[Note], duration_ms: int, window_ms: int = 2000) -> np.ndarray:
    """Notes per second in consecutive windows over the whole track."""
    n_win = max(1, int(math.ceil(duration_ms / window_ms)))
    starts = np.array([n.start for n in notes], dtype=np.int64)
    idx = np.clip(starts // window_ms, 0, n_win - 1)
    return np.bincount(idx, minlength=n_win).astype(np.float64) / (window_ms / 1000.0)


def density_correlation(a: Sequence[Note], b: Sequence[Note], duration_ms: int, window_ms: int = 2000) -> float:
    """Pearson correlation of two density graphs (1 = notes rise and fall at the same places)."""
    pa, pb = density_profile(a, duration_ms, window_ms), density_profile(b, duration_ms, window_ms)
    if pa.std() < 1e-9 or pb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(pa, pb)[0, 1])


def density_graph(notes: Sequence[Note], duration_ms: int, width: int = 72, scale: Optional[float] = None) -> str:
    """One-line bar graph of the note density, like the editor timeline (scale = notes/s of a full bar)."""
    if duration_ms <= 0:
        return ""
    prof = density_profile(notes, duration_ms, max(250, int(math.ceil(duration_ms / width))))
    top = float(scale or max(1e-6, prof.max()))
    bars = " ▁▂▃▄▅▆▇█"
    return "".join(bars[int(round(np.clip(v / top, 0.0, 1.0) * (len(bars) - 1)))] for v in prof)
