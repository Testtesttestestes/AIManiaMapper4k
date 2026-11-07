"""
Composer: a planning map generator that never "averages" the difficulty.

Why the plain hybrid decoder averages
-------------------------------------
The network predicts, for every 10 ms frame, the probability that a note starts there. Training with
one reference map per song (most songs have one or two difficulties) makes that probability an
*expectation over mappers and difficulties*: optional notes (1/8 fills, chord members, ghost notes) end
up around 0.3-0.6 whatever star rating is requested, and one threshold turns them into a mid-density
map. The same happens to lanes (argmax = the most common pattern) and to sections (a quiet verse and a
drop get similar densities). On top of that the network sees only +-10 s, so a chorus that comes back
after two minutes is mapped differently every time.

What the composer does instead
------------------------------
The network (optional!) is used only for what it is good at - *where* musically meaningful notes are
and which lanes it likes - and every decision that defines a difficulty is made explicitly:

1. **Song analysis** (whole track, O(N)): beat grid from the red lines (or an estimated one), attack
   strength at every tick of 1/1..1/8 and 1/3, 1/6, binary-vs-triplet feel per section, loudness,
   percussive/harmonic balance, melody pitch, timbre of every sound.
2. **Form**: novelty-based segmentation into sections on bar lines, section intensity (rank of
   loudness + attack rate inside the song) and *repetitions*: a section that sounds like an earlier one
   remembers how that one was mapped (rhythm and lanes, optionally mirrored). This is the long-range
   context the 20 s network window cannot have.
3. **Difficulty plan**: one "level" L sets the note budget per second, the minimum gap between notes
   and the chord rate. Budgets are distributed over sections by intensity with an explicit contrast, so
   breaks stay empty and drops get dense instead of everything being medium. Inside a section notes are
   chosen greedily by salience (attack strength + metric weight + network probability).
4. **Pattern plan**: every section gets a pattern family (stream/rolls, jumpstream, handstream,
   chordjack, jacks, trills, melodic lanes following pitch, long notes) chosen from the audio, the level,
   a variety constraint and seeded randomness (``creativity``). Lanes are then chosen event by event by a
   scorer that combines the family, the network's lane prior and ergonomics (no jack faster than the
   level allows, hand balance, no anchors, held long notes block their lane).
5. **Closed loop on star rating**: L is found by bisection so that the rosu-pp star rating of the
   *final rendered map* equals the request (+-0.05 by default). Several variants (seeds) are composed and
   a critic picks the best one (SR error, playability, pattern variety, notes on attacks).

Works for any track length (everything is linear in the number of beats) and any star rating the music
can carry; with no checkpoint it maps from the audio alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import ManiaMap, Note, RawTimingPoint, map_to_osu_text
from .params import LANE_TABLE, N_LANE_CODES, NUM_LANES

MASKS_BY_SIZE: Dict[int, List[int]] = {k: [m for m in range(1, 16) if bin(m).count("1") == k] for k in range(1, 5)}
LANES_OF = [[c for c in range(NUM_LANES) if (m >> c) & 1] for m in range(16)]
POP = np.array([bin(m).count("1") for m in range(16)])
CENTROID = np.array([np.mean(LANES_OF[m]) if m else 1.5 for m in range(16)])

# divisor -> metric weight of a tick (same scale as thinning.DIVISOR_STRENGTH)
TICK_DIVISORS: Tuple[Tuple[int, float], ...] = ((1, 3.0), (2, 2.0), (4, 1.2), (3, 1.0), (6, 0.7), (8, 0.5), (12, 0.4), (16, 0.25))
TRIPLET = {3, 6, 12}
BINARY_FINE = [4, 8, 16]
OFF_GRID = 0             # "divisor" of a candidate that sits on an audio attack between grid ticks
# tick divisor -> PatternModel divisor class (1/1, 1/2, 1/4, 1/3-1/6, 1/8, 1/12-1/16, off-grid)
DIV_TO_CLASS = np.full(17, 6, dtype=np.int64)
DIV_TO_CLASS[[1, 2, 4, 3, 6, 8, 12, 16]] = [0, 1, 2, 3, 3, 4, 5, 5]   # index 0 (OFF_GRID) stays 6

FAMILIES: Tuple[str, ...] = ("stream", "stairs", "jumpstream", "handstream", "chordjack", "jack", "trill", "melodic", "ln")
# density multiplier, chord multiplier, LN share bonus, lowest level that suits the family
FAMILY_SHAPE: Dict[str, Tuple[float, float, float, float]] = {
    "stream":     (1.10, 0.45, 0.00, 0.00),
    "stairs":     (1.05, 1.00, 0.00, 0.25),   # rolls that keep their direction while chords grow
    "jumpstream": (1.00, 1.20, 0.00, 0.30),
    "handstream": (0.95, 1.55, 0.00, 0.45),
    "chordjack":  (0.80, 1.90, 0.00, 0.45),
    "jack":       (0.90, 0.80, 0.00, 0.40),
    "trill":      (1.05, 0.70, 0.00, 0.15),
    "melodic":    (1.00, 0.80, 0.05, 0.00),
    "ln":         (0.85, 0.80, 0.45, 0.00),
}


# ----------------------------------------------------------------------------------------------
# configuration / results
# ----------------------------------------------------------------------------------------------

@dataclass
class ComposerConfig:
    creativity: float = 0.5          # 0 = conservative and repeatable, 1 = adventurous pattern choices
    contrast: float = 1.0            # how strongly section intensity shapes density (0 = flat, 2 = extreme)
    ln_ratio: Optional[float] = None # exact share of long notes (overrides ln_style)
    ln_style: Optional[str] = None   # rice / light / hybrid / ln / full; None = the most common style at this SR
    max_chord: int = 4
    variants: int = 4                # composed variants, the critic keeps the best one
    sr_tolerance: float = 0.05
    max_iterations: int = 16
    families: Optional[Sequence[str]] = None   # allowed pattern families (None = all)
    net_weight: float = 1.0          # weight of the network's event probability / lane prior (if given)
    net_rhythm: float = 3.0          # how strongly the network's note probability ranks the candidates (x net_weight)
    net_shape: float = 1.0           # 0..1: section densities from the audio intensity (0) or the network (1)
    repeat_memory: bool = True       # repeated sections reuse the earlier mapping
    mirror_repeats: Optional[bool] = None      # None = decided per repeat by the seed
    seed: int = 0
    bar_follow: float = 0.7          # 0..1: note density follows the loudness of every bar (1) or only of the section (0)
    silence_db: float = 26.0         # bars this far below the loud parts of the song are silence: no notes at all
    release_gap_ms: int = 40
    release_gap_beats: float = 0.25
    od: float = 8.0


@dataclass
class Section:
    start_ms: int
    end_ms: int
    intensity: float = 0.5           # 0..1 rank inside the song
    loud: float = 0.5
    perc: float = 0.5                # percussive share 0..1
    sustain: float = 0.5             # harmonic / sustained share 0..1
    pitch_motion: float = 0.0        # how much the melody moves (semitones per event)
    ramp: float = 0.0                # -1..1: energy falls / rises through the section (build-up, fade-out)
    triplet: bool = False
    repeat_of: Optional[int] = None  # section that contains the original passage
    repeat_src_ms: Optional[int] = None  # where the original passage starts (aligned with this section's start)
    similarity: float = 0.0
    family: str = "stream"
    mirror: bool = False

    @property
    def dur_s(self) -> float:
        return max(0.001, (self.end_ms - self.start_ms) / 1000.0)


@dataclass
class ComposeResult:
    notes: List[Note]
    star_rating: float
    level: float
    sections: List[Section]
    report: Dict[str, float] = field(default_factory=dict)

    def plan_text(self) -> str:
        rows = []
        for i, s in enumerate(self.sections):
            src = f"#{s.repeat_of}, " if s.repeat_of is not None else ""
            rep = (f" = repeat of {s.repeat_src_ms / 1000:.1f}s ({src}similarity {s.similarity:.2f}){' mirrored' if s.mirror else ''}"
                   if s.repeat_src_ms is not None else "")
            n = sum(1 for x in self.notes if s.start_ms <= x.start < s.end_ms)
            ramp = " build-up" if s.ramp > 0.25 else (" fade" if s.ramp < -0.25 else "")
            rows.append(f"  #{i:<2d} {s.start_ms / 1000:7.1f}-{s.end_ms / 1000:7.1f}s  intensity {s.intensity:.2f}  "
                        f"{s.family:<10s}{' 1/3' if s.triplet else '    '}  {n / s.dur_s:5.1f} nps{ramp}{rep}")
        return "\n".join(rows)


# ----------------------------------------------------------------------------------------------
# song analysis
# ----------------------------------------------------------------------------------------------

def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(x.size)
    return r / max(1, x.size - 1)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


@dataclass
class SongAnalysis:
    frame_ms: int
    num_frames: int
    duration_ms: int
    attack_ms: np.ndarray            # per-ms attack envelope (>= 0)
    loud: np.ndarray                 # [F] 0..1 smoothed loudness rank
    perc: np.ndarray                 # [F] percussive share 0..1
    harm: np.ndarray                 # [F] harmonic energy (z)
    pitch: np.ndarray                # [F] melody pitch (semitone bin)
    timbre: np.ndarray               # [F, 16] centred log-mel groups
    ticks: np.ndarray                # [T] int ms
    tick_div: np.ndarray             # [T] smallest divisor that hits the tick
    tick_metric: np.ndarray          # [T] metric weight
    tick_beat: np.ndarray            # [T] beat length (ms) at the tick
    tick_attack: np.ndarray          # [T] attack strength rank 0..1
    beats: np.ndarray                # beat times (ms)
    bar_beats: int
    timing_points: List[RawTimingPoint]
    repeats: List[Tuple[int, int, int, float]] = field(default_factory=list)  # (start, end, source start, similarity)
    tick_audible: Optional[np.ndarray] = None  # [T] attack above median + 2 MAD of the whole track
    tick_attack_raw: Optional[np.ndarray] = None  # [T] attack strength at every tick (raw)
    onset_peaks: Optional[np.ndarray] = None   # ms of sharp attacks (local maxima), shifted by peak_lag
    peak_lag: int = 0                          # how much later than the mapper's grid an attack peak comes (ms)
    flux: Optional[np.ndarray] = None          # [F] superflux onset strength (z, per song)
    power: Optional[np.ndarray] = None         # [F] log power (natural log), absolute: silence / loudness per bar

    def power_top(self) -> float:
        """Log power of the loud parts of the song (95th percentile of the frames)."""
        if self.power is None or self.power.size == 0:
            return 0.0
        if getattr(self, "_power_top", None) is None:
            self._power_top = float(np.percentile(self.power, 95))
        return self._power_top

    def silent_frames(self, silence_db: float = 26.0) -> np.ndarray:
        """[F] True where the audio is ``silence_db`` below the loud parts of the song. Measured on 9 songs: bars that
        no human difficulty maps sit 26-36 dB below the song's loud parts, mapped bars at most ~28 dB below."""
        if self.power is None:
            return self.loud < 0.02
        return self.power < self.power_top() - silence_db / (10.0 / math.log(10.0))

    def source_time(self, t: np.ndarray) -> np.ndarray:
        """Time of the same moment in the original of a repeated phrase (-1 where t is not in a repeat)."""
        t = np.asarray(t, dtype=np.int64)
        out = np.full(t.shape, -1, dtype=np.int64)
        for a, b, src, _ in self.repeats:
            m = (t >= a) & (t < b)
            out[m] = t[m] - a + src
        return out

    def attack_at(self, t: np.ndarray, radius: int = 8) -> np.ndarray:
        env = self.attack_ms
        M = env.size
        t = np.clip(np.asarray(t, dtype=np.int64), 0, M - 1)
        out = np.zeros(t.shape, dtype=np.float64)
        for d in range(-radius, radius + 1, 2):
            out = np.maximum(out, env[np.clip(t + d, 0, M - 1)])
        return out


def fade_out_start(an: "SongAnalysis", silence_db: float = 26.0, drop: float = 0.5, max_fade_s: float = 25.0) -> Optional[int]:
    """Where the mapping of a song with a fade-out ends.

    A fade-out is a monotonic fall of the loudness into silence at the end of the audio; reverb and cymbal noise in
    it still look like attacks. Human maps stop when the fade has fallen ``drop`` (log power, ~2 dB) below its start
    (Blackmagik Blazing: humans end at 262.6 s, this gives 262.5 s). None when the song does not end with a fade."""
    if an.power is None or an.power.size < 300:
        return None
    H = an.frame_ms
    step = max(1, 100 // H)
    p = np.convolve(an.power, np.ones(100 // H) / (100 // H), mode="same")[::step]   # 1 s smoothing, 100 ms steps
    top = an.power_top()
    thr = top - silence_db / (10.0 / math.log(10.0))
    loud = np.flatnonzero(p >= thr)
    if loud.size == 0:
        return None
    e = int(loud[-1])
    full = np.flatnonzero(p[:e + 1] >= top - 0.25)          # the last moment at full loudness before the silence
    if full.size == 0:
        return None
    i = int(full[-1])
    seg = p[i:e + 1]
    if (e - i) * step * H > max_fade_s * 1000 or seg.size < 10 or p[i] - p[e] < 2.0:
        return None
    if float(np.mean(np.diff(seg) <= 0.02)) < 0.7:          # mostly falling: a fade, not a quiet mapped outro
        return None
    after = np.flatnonzero(p[i:e + 1] < p[i] - drop)
    return int((i + int(after[0])) * step * H) if after.size else None


def _reds(timing_points: Sequence[RawTimingPoint]) -> List[RawTimingPoint]:
    return sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)


def build_ticks(timing_points: Sequence[RawTimingPoint], duration_ms: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    reds = _reds(timing_points)
    if not reds:
        reds = [RawTimingPoint(0.0, 500.0)]
    times: List[float] = []
    divs: List[int] = []
    metric: List[float] = []
    blen: List[float] = []
    beats: List[float] = []
    for i, r in enumerate(reds):
        seg_end = reds[i + 1].time if i + 1 < len(reds) else float(duration_ms)
        start = r.time
        if i == 0:  # extend the first red line backwards to the start of the audio
            n_back = int(math.floor(start / r.beat_len))
            start = r.time - n_back * r.beat_len
        n_beats = int(math.ceil((seg_end - start) / r.beat_len))
        meter = max(1, int(r.meter or 4))
        first_beat_index = int(round((start - r.time) / r.beat_len))
        for b in range(n_beats):
            tb = start + b * r.beat_len
            if tb >= seg_end - 1e-6:
                break
            beats.append(tb)
            downbeat = (first_beat_index + b) % meter == 0
            seen = set()
            for d, w in TICK_DIVISORS:
                for k in range(d):
                    num, den = k, d
                    g = math.gcd(num, den)
                    key = (num // g, den // g)
                    if key in seen:
                        continue
                    seen.add(key)
                    t = tb + r.beat_len * k / d
                    if t >= seg_end - 1e-6 or t < 0:
                        continue
                    times.append(t)
                    divs.append(key[1])
                    metric.append(dict(TICK_DIVISORS)[key[1]] + (1.0 if (key[0] == 0 and downbeat) else 0.0))
                    blen.append(r.beat_len)
    order = np.argsort(times, kind="stable")
    t = np.round(np.array(times)[order]).astype(np.int64)
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) >= 3
    meter = max(1, int(reds[0].meter or 4))
    return (t[keep], np.array(divs)[order][keep], np.array(metric)[order][keep], np.array(blen)[order][keep],
            np.round(np.array(beats)).astype(np.int64), meter)


def analyze_song(
    feats: np.ndarray,
    slices: Dict[str, Tuple[int, int]],
    timing_points: Sequence[RawTimingPoint],
    duration_ms: int,
    frame_ms: int = 10,
) -> SongAnalysis:
    F = int(feats.shape[0])

    def block(name: str) -> Optional[np.ndarray]:
        if name not in slices:
            return None
        a, b = slices[name]
        return np.asarray(feats[:, a:b], dtype=np.float32)

    fine = block("fine_env")
    onset = block("onset")
    if fine is not None:
        half = fine.shape[1] // 2
        attack = np.maximum(0.0, fine[:, half:]).reshape(-1).astype(np.float64)
        attack = np.convolve(attack, np.ones(3) / 3.0, mode="same")
    elif onset is not None:
        o = onset[:, -2:].mean(axis=1)
        attack = np.repeat(np.maximum(0.0, o - np.median(o)), frame_ms).astype(np.float64)
    else:
        attack = np.zeros(F * frame_ms)
    if onset is not None:  # blend in the band-wise flux (sharper for soft attacks), upsampled to 1 ms
        o = onset[:, -2:].mean(axis=1).astype(np.float64)
        o = np.maximum(0.0, (o - np.median(o)) / (o.std() + 1e-6))
        attack = attack / (attack.std() + 1e-6) + 0.5 * np.repeat(o, frame_ms)[: attack.size]

    spec = block("spectral")
    mel = block("mel_mid")
    loud_raw = spec[:, 0] if spec is not None else (mel.mean(axis=1) if mel is not None else np.zeros(F))
    loud = _rank01(_smooth(loud_raw.astype(np.float64), 50))
    hp = block("hpss")
    if hp is not None:
        perc_tot, harm_tot = hp[:, -2].astype(np.float64), hp[:, -1].astype(np.float64)
        perc = 1.0 / (1.0 + np.exp(-(_smooth(perc_tot, 30) - _smooth(harm_tot, 30))))
        harm = harm_tot
    else:
        perc = np.full(F, 0.5)
        harm = np.zeros(F)
    pitch_b = block("pitch")
    if pitch_b is not None:
        lo, hi = 24, min(pitch_b.shape[1], 84)
        P = pitch_b[:, lo:hi] - np.median(pitch_b[:, lo:hi], axis=0, keepdims=True)
        pitch = (np.argmax(P, axis=1) + lo).astype(np.float64)
    else:
        pitch = np.zeros(F)
    if mel is not None:
        g = mel.shape[1] // 16
        timbre = mel[:, : g * 16].reshape(F, 16, g).mean(axis=2)
        timbre = timbre - timbre.mean(axis=1, keepdims=True)
    else:
        timbre = np.zeros((F, 16), np.float32)

    tps = list(timing_points)
    ticks, tdiv, tmetric, tbeat, beats, meter = build_ticks(tps, duration_ms)
    keep = ticks < duration_ms
    ticks, tdiv, tmetric, tbeat = ticks[keep], tdiv[keep], tmetric[keep], tbeat[keep]
    an = SongAnalysis(frame_ms, F, int(duration_ms), attack, loud, perc, harm, pitch, timbre,
                      ticks, tdiv, tmetric, tbeat, np.zeros(ticks.size), beats[beats < duration_ms], meter, tps)
    if onset is not None:
        an.flux = onset[:, -1].astype(np.float64)
    if spec is not None:
        an.power = spec[:, 0].astype(np.float64)
    raw = an.attack_at(ticks)
    an.tick_attack = _rank01(raw)
    med = float(np.median(attack))
    mad = float(np.median(np.abs(attack - med))) + 1e-9
    an.tick_audible = raw > med + 2.0 * mad   # an attack clearly above the track's own noise floor
    an.tick_attack_raw = raw
    find_onset_peaks(an)
    return an


def find_onset_peaks(an: SongAnalysis, window_ms: int = 10, mad_k: float = 2.0) -> None:
    """Sharp attacks of the track at 1 ms resolution, and the lag between the grid and the attack peaks.

    The peak of the energy derivative comes a few ms after the moment a mapper puts the note on, and how many ms
    depends on the song and on the mapper (measured: -2..+8 ms). The red lines of a template were placed by the
    mapper, so the lag is read off the strong beats: the median offset from beat/half-beat ticks with a clear attack
    to the nearest peak. Off-grid candidates use the peaks minus that lag, consistent with the mapper's grid.
    """
    from scipy.ndimage import maximum_filter1d

    env = an.attack_ms
    if env.size < 100:
        an.onset_peaks = np.zeros(0, dtype=np.int64)
        return
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-9
    mx = maximum_filter1d(env, 2 * window_ms + 1)
    peaks = np.flatnonzero((env >= mx) & (env > med + mad_k * mad)).astype(np.int64)
    lag = 0
    strong = an.ticks[np.isin(an.tick_div, [1, 2]) & (an.tick_audible if an.tick_audible is not None else True)]
    if peaks.size > 1 and strong.size >= 30:
        j = np.clip(np.searchsorted(peaks, strong), 1, peaks.size - 1)
        a, b = peaks[j] - strong, peaks[j - 1] - strong
        d = np.where(np.abs(a) < np.abs(b), a, b)
        d = d[np.abs(d) <= 15]
        if d.size >= 30:
            lag = int(round(float(np.median(d))))
    an.peak_lag = lag
    an.onset_peaks = peaks - lag


# ----------------------------------------------------------------------------------------------
# form: sections and repetitions
# ----------------------------------------------------------------------------------------------

def _beat_vectors(an: SongAnalysis) -> np.ndarray:
    F, H = an.num_frames, an.frame_ms
    b = an.beats
    if b.size < 2:
        return np.zeros((0, 1))
    edges = np.clip(np.append(b, b[-1] + np.median(np.diff(b))) // H, 0, F).astype(np.int64)
    cs_t = np.vstack([np.zeros((1, an.timbre.shape[1])), np.cumsum(an.timbre, axis=0)])
    cs_l = np.concatenate([[0.0], np.cumsum(an.loud)])
    cs_p = np.concatenate([[0.0], np.cumsum(an.perc)])
    n = np.maximum(1, edges[1:] - edges[:-1])[:, None]
    tim = (cs_t[edges[1:]] - cs_t[edges[:-1]]) / n
    lo = ((cs_l[edges[1:]] - cs_l[edges[:-1]]) / n[:, 0])[:, None]
    pe = ((cs_p[edges[1:]] - cs_p[edges[:-1]]) / n[:, 0])[:, None]
    v = np.concatenate([tim, 3.0 * lo, 2.0 * pe], axis=1)
    v = (v - v.mean(axis=0)) / (v.std(axis=0) + 1e-6)
    return v


def segment_sections(an: SongAnalysis, target_sec: float = 8.0, min_bars: int = 2, max_bars: int = 16) -> List[Section]:
    """Sections of the song at beat precision.

    Novelty (how different the sound after a beat is from the sound before it, over 1 and 2 bars) is measured at
    every beat, not only on bar lines: a drum entry or a drop one beat before the bar (pickup) starts its section
    on that beat. Boundaries on a bar line get a small bonus (most changes are there), sections are at least
    ``min_bars`` long; about one section per ``target_sec`` seconds plus every clearly stronger change."""
    beats, bar = an.beats, max(1, an.bar_beats)
    if beats.size < 2 * bar:
        return [Section(0, an.duration_ms)]
    V = _beat_vectors(an)
    nb = V.shape[0]
    cs = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0)])
    nov = np.zeros(nb)
    for w in (bar, 2 * bar):
        for b in range(nb):
            a0, a1, b1 = max(0, b - w), b, min(nb, b + w)
            if a1 - a0 < max(2, w // 2) or b1 - a1 < max(2, w // 2):
                continue
            left = (cs[a1] - cs[a0]) / (a1 - a0)
            right = (cs[b1] - cs[a1]) / (b1 - a1)
            nov[b] += 0.5 * float(np.linalg.norm(left - right))
    # where the change is: the beat of the biggest step inside +-1 beat (short windows of 2 beats)
    step = np.zeros(nb)
    for b in range(2, nb - 2):
        step[b] = float(np.linalg.norm(V[b:b + 2].mean(axis=0) - V[b - 2:b].mean(axis=0)))
    score = nov * (1.0 + 0.15 * (np.arange(nb) % bar == 0)) + 0.25 * step
    beat_ms = float(np.median(np.diff(beats)))
    n_target = max(1, int(round(an.duration_ms / 1000.0 / target_sec)))
    min_gap = min_bars * bar
    pos = score[score > 0]
    strong = float(np.median(pos) + 2.0 * np.median(np.abs(pos - np.median(pos)))) if pos.size else np.inf
    chosen = [0]
    for i in np.argsort(-score):
        if score[i] <= 0:
            break
        if len(chosen) >= n_target and score[i] < strong:
            break
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
    chosen = sorted(chosen)
    bounds: List[int] = []
    for k, c in enumerate(chosen):  # split sections longer than max_bars (on bar lines)
        nxt = chosen[k + 1] if k + 1 < len(chosen) else nb
        bounds.append(c)
        n_split = (nxt - c - 1) // (max_bars * bar)
        for q in range(1, n_split + 1):
            x = c + q * (nxt - c) // (n_split + 1)
            bounds.append(int(round(x / bar)) * bar)
    bounds = sorted(set(b for b in bounds if 0 <= b < nb))
    out: List[Section] = []
    for k, bi in enumerate(bounds):
        start = 0 if k == 0 else int(beats[bi])
        end = int(beats[bounds[k + 1]]) if k + 1 < len(bounds) else an.duration_ms
        if end - start >= beat_ms:
            out.append(Section(start, end))
    out[-1].end_ms = an.duration_ms
    _describe_sections(an, out, V)
    return out


def _describe_sections(an: SongAnalysis, sections: List[Section], V: np.ndarray) -> None:
    H = an.frame_ms
    loud_m, rate_m, flux_m = [], [], []
    onset_flux = an.flux if an.flux is not None else None
    for s in sections:
        f0, f1 = s.start_ms // H, max(s.start_ms // H + 1, s.end_ms // H)
        s.loud = float(an.loud[f0:f1].mean())
        s.perc = float(an.perc[f0:f1].mean())
        sel = (an.ticks >= s.start_ms) & (an.ticks < s.end_ms)
        strong = an.tick_attack[sel] > 0.75
        rate_m.append(float(strong.sum()) / s.dur_s)
        flux_m.append(float(onset_flux[f0:f1].mean()) if onset_flux is not None else 0.0)
        loud_m.append(s.loud)
        harm = an.harm[f0:f1]
        s.sustain = float(1.0 / (1.0 + np.exp(-(np.median(harm) - np.median(an.harm)))))
        p = an.pitch[f0:f1:5]
        s.pitch_motion = float(np.mean(np.abs(np.diff(p)) > 0.5)) if p.size > 2 else 0.0
        trip = sel & np.isin(an.tick_div, list(TRIPLET))
        binary = sel & np.isin(an.tick_div, BINARY_FINE)
        a_t = an.tick_attack[trip].mean() if trip.any() else 0.0
        a_b = an.tick_attack[binary].mean() if binary.any() else 0.0
        s.triplet = bool(a_t > a_b + 0.06)
        # build-up / break-down: loudness and attack density at the end vs the start of the section
        n3 = max(1, (f1 - f0) // 3)
        dl = float(an.loud[f1 - n3:f1].mean() - an.loud[f0:f0 + n3].mean())
        ts = an.ticks[sel]
        ta = an.tick_attack[sel]
        if ts.size >= 6:
            third = (ts - s.start_ms) / max(1.0, s.end_ms - s.start_ms)
            da = float(ta[third >= 2 / 3].mean() - ta[third < 1 / 3].mean()) if (third >= 2 / 3).any() and (third < 1 / 3).any() else 0.0
        else:
            da = 0.0
        s.ramp = float(np.clip(1.5 * dl + 1.2 * da, -1.0, 1.0))
    # intensity = how "full" a section is, relative to the fullest section of the song (values, not ranks: ranks
    # over a handful of sections would turn a slightly quieter verse into "nothing"). Measured on real maps
    # (section density vs features, 10 songs): harmonic energy 0.85, loudness 0.73, spectral flux 0.47,
    # attack rate ~0 (dense music has attacks everywhere); the mix below correlates 0.85 (0.87 on a Camellia
    # 200step remix), the old loudness + attack-rate mix 0.71.
    flux = np.array(flux_m)
    raw = 0.5 * np.array([s.sustain for s in sections]) + 0.3 * np.array(loud_m) + 0.2 / (1.0 + np.exp(-flux))
    rel = raw / max(1e-6, raw.max())
    silent = an.silent_frames()
    for s, v in zip(sections, rel):
        s.intensity = float(np.clip(v, 0.05, 1.0))
        f0, f1 = s.start_ms // H, max(s.start_ms // H + 1, s.end_ms // H)
        # silence is absolute (26 dB below the loud parts): the quietest part of a song is not a break by itself
        if float(silent[f0:f1].mean()) > 0.6:
            s.intensity = 0.0
    find_repeats(an, sections, V)


def find_repeats(an: SongAnalysis, sections: List[Section], V: np.ndarray, min_sim: float = 0.8,
                 max_lookback_s: float = 600.0, phrase_bars: int = 4) -> None:
    """Finds repeated phrases: every 4-bar phrase gets the earlier bar-aligned passage that sounds the most alike
    beat by beat (if similar enough). The composer maps a repeated phrase like the original (rhythm and lanes, or
    mirrored) - the song-wide context a 20 s network window cannot see. A section whose phrases mostly repeat an
    earlier section also takes over its pattern family."""
    an.repeats = []
    beats = an.beats
    if beats.size < 8 or V.shape[0] < 8:
        return
    nb = V.shape[0]
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    bar = max(1, an.bar_beats)
    L = bar * phrase_bars
    beat_ms = float(np.median(np.diff(beats)))
    look = int(max_lookback_s * 1000.0 / beat_ms)
    for b0 in range(L, nb - L + 1, L):
        lo = max(0, b0 - look)
        starts = np.arange(lo - (lo % bar), b0 - L + 1, bar)
        if starts.size == 0:
            continue
        A = U[b0:b0 + L]
        sims = np.array([float((A * U[p:p + L]).sum(axis=1).mean()) for p in starts])
        k = int(np.argmax(sims))
        if sims[k] >= min_sim:
            end = int(beats[b0 + L]) if b0 + L < beats.size else an.duration_ms
            an.repeats.append((int(beats[b0]), end, int(beats[int(starts[k])]), float(sims[k])))
    starts_ms = [x.start_ms for x in sections]
    for i, s in enumerate(sections):
        inside = [r for r in an.repeats if s.start_ms <= r[0] < s.end_ms]
        if not inside:
            continue
        covered = sum(r[1] - r[0] for r in inside) / max(1.0, s.end_ms - s.start_ms)
        src_sec = [int(np.searchsorted(starts_ms, r[2], side="right") - 1) for r in inside]
        j = max(set(src_sec), key=src_sec.count)
        s.similarity = float(np.mean([r[3] for r in inside]))
        s.repeat_src_ms = inside[0][2] - (inside[0][0] - s.start_ms)
        if covered >= 0.6 and 0 <= j < i:
            s.repeat_of = j


# ----------------------------------------------------------------------------------------------
# network hints
# ----------------------------------------------------------------------------------------------

@dataclass
class NetHints:
    p_event: np.ndarray              # [F]
    offset_ms: np.ndarray            # [F] most likely ms inside the frame
    lane_logits: Optional[np.ndarray]
    chord_logprobs: Optional[np.ndarray]
    ln_prob: Optional[np.ndarray]    # [F] max member LN probability
    ln_len_ms: Optional[np.ndarray] = None  # [F] LN length the network predicts (it hears how long a sound lasts)
    lag_ms: int = 0                          # how much later than the mapper's grid the network's milliseconds come

    @classmethod
    def from_outputs(cls, out) -> "NetHints":
        from .params import ln_bucket_to_length
        tp = out.time_probs
        ln = 1.0 / (1.0 + np.exp(-out.ln_logits[:, 0])) if out.ln_logits is not None else None
        ln_len = None
        if getattr(out, "ln_bucket", None) is not None and getattr(out, "ln_resid", None) is not None:
            ln_len = ln_bucket_to_length(out.ln_bucket[:, 0], out.ln_resid[:, 0]).astype(np.int64)
        return cls(1.0 - tp[:, 0], np.argmax(tp[:, 1:], axis=1), out.lane_logits, out.chord_logprobs, ln, ln_len)

    def calibrate_lag(self, ticks: np.ndarray, tick_div: np.ndarray, frame_ms: int) -> None:
        """Per-song offset of the network's predicted milliseconds against the template's grid: median distance from
        confident network events to the nearest beat / half-beat tick. Measured: 0 on most songs, but on some audio the
        network's milliseconds are shifted (Exit This Earth's Atomosphere: 61 % of the old decoder's notes off-grid), and
        a candidate weighting by the distance to the predicted ms then prefers the wrong neighbouring tick."""
        pe = self.p_event
        fr = np.flatnonzero((pe >= 0.5) & (pe >= np.roll(pe, 1)) & (pe >= np.roll(pe, -1)))
        strong = np.sort(ticks[np.isin(tick_div, [1, 2])])
        if fr.size < 30 or strong.size < 30:
            return
        ms = fr * frame_ms + self.offset_ms[fr]
        j = np.clip(np.searchsorted(strong, ms), 1, strong.size - 1)
        a, b = ms - strong[j], ms - strong[j - 1]
        d = np.where(np.abs(a) < np.abs(b), a, b)
        d = d[np.abs(d) <= 20]
        if d.size >= 30:
            self.lag_ms = int(round(float(np.median(d))))

    def p_at(self, t: np.ndarray, frame_ms: int, radius_ms: int = 15, sigma_ms: float = 4.0) -> np.ndarray:
        """Network note probability AT a candidate time: the probability of each nearby frame, weighted by how far the
        candidate is from the millisecond the network predicts inside that frame. Neighbouring 1/16 ticks are 30 ms
        apart at 125 BPM - a plain max over +-15 ms gave both the same probability and lost the network's timing."""
        F = self.p_event.size
        t = np.asarray(t, dtype=np.int64)
        f = t // frame_ms
        r = max(1, int(math.ceil(radius_ms / frame_ms)))
        out = np.zeros(t.shape)
        for d in range(-r, r + 1):
            ff = np.clip(f + d, 0, F - 1)
            ms = ff * frame_ms + self.offset_ms[ff] - self.lag_ms
            w = np.exp(-0.5 * ((t - ms) / sigma_ms) ** 2)
            out = np.maximum(out, self.p_event[ff] * w)
        return out


# ----------------------------------------------------------------------------------------------
# the composer
# ----------------------------------------------------------------------------------------------

LEVEL_MAX = 2.0


def _level_params(L: float) -> Tuple[float, float, float]:
    """Events per second (at intensity 1), minimum gap (ms) and chord rate for a level L in [0, 1.6]."""
    L = max(0.0, float(L))
    eps = 0.5 * (16.0 / 0.5) ** min(L, 1.0) * (1.0 + 0.8 * max(0.0, L - 1.0))
    gap = 420.0 * (38.0 / 420.0) ** min(1.0, L)
    chord = min(0.92, 0.02 + 0.55 * L ** 1.4)
    return eps, gap, chord


def _nested_sizes(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Chord sizes from ranks: a 2nd note where u[:, 0] < r[:, 0], a 3rd only on those that have a 2nd and pass the 3-note
    rank, a 4th only on 3-note chords. Independent checks gave twice the triples the data asks for (13.5 % vs 6.6 %)."""
    two = u[:, 0] < r[:, 0]
    # among the chords that got a 2nd note, the 3rd goes to the best of them by the 3-note rank, as many as r3 asks
    out = 1 + two.astype(int)
    if two.any():
        idx = np.flatnonzero(two)
        n3 = int(round(float(np.mean(r[:, 1])) * u.shape[0]))
        if n3 > 0:
            best3 = idx[np.argsort(u[idx, 1], kind="stable")[:n3]]
            out[best3] = 3
            n4 = int(round(float(np.mean(r[:, 2])) * u.shape[0]))
            if n4 > 0:
                best4 = best3[np.argsort(u[best3, 2], kind="stable")[:n4]]
                out[best4] = 4
    return out


def _greedy_pick(t: np.ndarray, idx: np.ndarray, sc: np.ndarray, budget: int, gaps: np.ndarray,
                 taken0: Optional[np.ndarray] = None) -> List[int]:
    """Lazy greedy selection of ``budget`` candidates by score. A candidate close to already chosen notes is not
    forbidden, its score just drops the closer it gets (``gaps`` = comfortable gap per candidate); the budget alone sets
    the density, so the star rating changes smoothly with the level instead of jumping when the gap crosses the song's
    1/4 spacing. Two notes closer than 20 ms are one note."""
    import heapq

    heap = [(-float(sc[q]), q) for q in range(idx.size)]
    heapq.heapify(heap)
    taken: List[int] = []
    taken_t = np.sort(np.asarray(taken0, dtype=np.int64)) if taken0 is not None else np.empty(0, dtype=np.int64)

    def penalty(tk: int, g: float) -> float:
        if not taken_t.size:
            return 0.0
        p = int(np.searchsorted(taken_t, tk))
        d = min(taken_t[p] - tk if p < taken_t.size else 10 ** 9, tk - taken_t[p - 1] if p > 0 else 10 ** 9)
        if d < 20:
            return 1e9
        return 12.0 * max(0.0, 1.0 - d / g) ** 1.5

    while heap and len(taken) < budget:
        _, q = heapq.heappop(heap)
        tk = int(t[idx[q]])
        cur = float(sc[q]) - penalty(tk, float(gaps[q]))
        if heap and cur < -heap[0][0] - 1e-9:
            if cur > -1e8:
                heapq.heappush(heap, (-cur, q))
            continue
        if cur <= -1e8:
            continue
        taken_t = np.insert(taken_t, int(np.searchsorted(taken_t, tk)), tk)
        taken.append(int(idx[q]))
    return taken


def _jack_min(L: float) -> float:
    """Fastest allowed repeat of a lane (ms): 300 ms on the easiest level, 120 ms on the hardest."""
    return 300.0 * (120.0 / 300.0) ** min(1.0, max(0.0, L))


def stair_next(prev: int, size: int, direction: int) -> int:
    """Continues a staircase: the next chord starts right after the previous one and wraps around 4 -> 1.

    With growing chord sizes this is the classic accelerating stair 1 2 3 4 1 23 4 12 34 123 4 12 341.
    """
    if not prev:
        return 0
    lanes = LANES_OF[prev]
    start = (max(lanes) + 1) % NUM_LANES if direction >= 0 else (min(lanes) - 1) % NUM_LANES
    out = 0
    for j in range(size):
        out |= 1 << ((start + j * (1 if direction >= 0 else -1)) % NUM_LANES)
    return out


def _gumbel(rng: np.random.Generator, n: int) -> np.ndarray:
    return -np.log(-np.log(np.clip(rng.random(n), 1e-9, 1 - 1e-9)))


class Composer:
    def __init__(self, an: SongAnalysis, sections: List[Section], cfg: ComposerConfig, hints: Optional[NetHints] = None,
                 log: Callable[[str], None] = print, patterns=None, target_sr: float = 4.0):
        self.an, self.sections, self.cfg, self.hints, self.log = an, sections, cfg, hints, log
        self.patterns = patterns          # learned PatternModel (patterns.py) or None
        self.chord_boost = 1.0            # > 1 when the density alone cannot reach the requested star rating
        self.repeat_scale = 1.0           # strength of the lane-repeat (jack / anchor) penalties, calibrated to the data
        self.target_sr = float(target_sr)
        self._candidates()
        self._bars()

    # --- bars: loudness of every bar, silence ------------------------------------------------------
    def _bars(self) -> None:
        """Bar edges, the loudness of every bar relative to the loud parts of the song and silent bars.

        Measured on 9 songs x all human difficulties: the human note density of a bar follows its log power with
        exponent ~0.84 (density ~ power^0.84, Spearman 0.56 on average against 0.49 for the section intensity) - a
        drum entry raises the density in the very bar it starts, not at the next section boundary."""
        an, H = self.an, self.an.frame_ms
        bb = max(1, an.bar_beats)
        b = an.beats[::bb] if an.beats.size else np.zeros(0, np.int64)
        edges = np.unique(np.concatenate([[0], b[b > 0], [an.duration_ms]])).astype(np.int64)
        self.bar_edges = edges
        nb = edges.size - 1
        silent_f = an.silent_frames(self.cfg.silence_db)
        lp = np.zeros(nb)
        sil = np.zeros(nb, dtype=bool)
        for i in range(nb):
            f0, f1 = int(edges[i] // H), max(int(edges[i] // H) + 1, int(edges[i + 1] // H))
            f1 = min(f1, an.num_frames)
            if f0 >= f1:
                sil[i] = True
                continue
            if an.power is not None:
                lp[i] = float(an.power[f0:f1].mean())
            sil[i] = float(silent_f[f0:f1].mean()) > 0.5
        self.bar_silent = sil
        top = an.power_top() if an.power is not None else 0.0
        self.bar_rel = lp - top if an.power is not None else np.zeros(nb)
        self.bar_of_t = np.clip(np.searchsorted(edges, self.t, side="right") - 1, 0, max(0, nb - 1))
        # silent moments at 10 ms resolution: no candidate there at all (intros, outros, the end of the audio)
        f = np.clip(self.t // H, 0, an.num_frames - 1)
        self.t_silent = silent_f[f] & ~self.audible
        self.fade_cut = fade_out_start(an, self.cfg.silence_db)
        if self.fade_cut is not None:
            self.t_silent |= self.t >= self.fade_cut
            self.bar_silent |= self.bar_edges[:-1] >= self.fade_cut

    def bar_shape(self) -> np.ndarray:
        """Relative density of every bar (duration-weighted mean 1): the section's share (form, contrast, network)
        refined by the bar's own loudness."""
        sec = self.section_shape()
        edges = self.bar_edges
        mid = 0.5 * (edges[:-1] + edges[1:])
        so = np.clip(np.searchsorted([x.start_ms for x in self.sections], mid, side="right") - 1, 0, len(self.sections) - 1)
        base = sec[so]
        w = float(np.clip(self.cfg.bar_follow, 0.0, 1.0)) if self.an.power is not None else 0.0
        local = np.exp(0.84 * self.cfg.contrast * np.clip(self.bar_rel, -8.0, 1.0))
        dur = np.maximum(1.0, np.diff(edges).astype(np.float64))
        act = ~self.bar_silent
        if act.any():
            local = local / max(1e-6, (local[act] * dur[act]).sum() / dur[act].sum())
        shape = np.where(base > 0, np.maximum(base, 1e-3) ** (1.0 - w) * np.maximum(local, 1e-3) ** w, 0.0)
        shape[self.bar_silent] = 0.0
        return shape / max(1e-6, (shape * dur).sum() / dur.sum())

    def jack_min(self, L: float) -> float:
        """Fastest normal lane repeat: the 1st percentile of real maps at the target star rating (dataset
        statistics), or a level-based default when no pattern model is loaded."""
        if self.patterns is not None:
            floor = self.patterns.repeat_floor(self.target_sr, 0.01)
            if floor is not None and floor > 0:
                return float(floor)
        return _jack_min(L)

    # --- candidate events -------------------------------------------------------------------------
    def _candidates(self) -> None:
        an, H = self.an, self.an.frame_ms
        t = an.ticks.copy()
        metric = an.tick_metric.astype(np.float64)
        attack = an.tick_attack.astype(np.float64)
        div = an.tick_div.copy()
        net = np.zeros(t.size)
        if self.hints is not None:
            self.hints.calibrate_lag(an.ticks, an.tick_div, H)
            net = self.hints.p_at(t, H)
            # confident network events far from every tick (off-grid notes, wrong tempo estimate)
            pe = self.hints.p_event
            peaks = np.flatnonzero((pe >= 0.5) & (pe >= np.roll(pe, 1)) & (pe >= np.roll(pe, -1)))
            et = peaks * H + self.hints.offset_ms[peaks] - self.hints.lag_ms
            if t.size:
                pos = np.clip(np.searchsorted(t, et), 1, t.size - 1)
                dist = np.minimum(np.abs(t[pos] - et), np.abs(t[pos - 1] - et))
                et = et[dist > 20]
            if et.size:
                t = np.concatenate([t, et])
                metric = np.concatenate([metric, np.full(et.size, 0.3)])
                attack = np.concatenate([attack, _rank01(an.attack_at(et))])
                div = np.concatenate([div, np.full(et.size, OFF_GRID)])
                net = np.concatenate([net, pe[et // H]])
                o = np.argsort(t, kind="stable")
                t, metric, attack, div, net = t[o], metric[o], attack[o], div[o], net[o]
        # sharp audio attacks between grid ticks (glitch, fills, humanized playing): candidates at the attack's own
        # millisecond, shifted by the song's grid-vs-attack lag (how the mapper's red line sits on the attacks)
        if an.onset_peaks is not None and an.onset_peaks.size and t.size:
            pk = an.onset_peaks
            pos = np.clip(np.searchsorted(t, pk), 1, t.size - 1)
            dist = np.minimum(np.abs(t[pos] - pk), np.abs(t[pos - 1] - pk))
            pk = pk[dist > 6]
            if pk.size:
                raw_ticks = np.sort(an.tick_attack_raw) if an.tick_attack_raw is not None else None
                raw = an.attack_at(pk + an.peak_lag)
                att = np.searchsorted(raw_ticks, raw) / max(1, raw_ticks.size) if raw_ticks is not None else _rank01(raw)
                t = np.concatenate([t, pk])
                metric = np.concatenate([metric, np.full(pk.size, 0.2)])
                attack = np.concatenate([attack, att])
                div = np.concatenate([div, np.full(pk.size, OFF_GRID)])
                net = np.concatenate([net, self.hints.p_at(pk, H) if self.hints is not None else np.zeros(pk.size)])
                o = np.argsort(t, kind="stable")
                t, metric, attack, div, net = t[o], metric[o], attack[o], div[o], net[o]
        self.t, self.metric, self.attack, self.div, self.net = t, metric, attack, div, net
        med = float(np.median(an.attack_ms))
        mad = float(np.median(np.abs(an.attack_ms - med))) + 1e-9
        self.audible = an.attack_at(t) > med + 2.0 * mad
        self.sec_of = np.clip(np.searchsorted([s.start_ms for s in self.sections], t, side="right") - 1, 0, len(self.sections) - 1)
        f = np.clip(t // H, 0, an.num_frames - 1)
        self.pitch = an.pitch[f]
        tim = an.timbre[np.clip(f + 1, 0, an.num_frames - 1)]
        self.timbre = tim / (np.linalg.norm(tim, axis=1, keepdims=True) + 1e-6)
        self.loud = an.loud[f]
        # position inside the section (-0.5..0.5) and loudness relative to the section: the accelerations and
        # decelerations inside a section (build-ups, drops, fade-outs) shape density and chord size
        starts = np.array([s.start_ms for s in self.sections], dtype=np.float64)
        ends = np.array([s.end_ms for s in self.sections], dtype=np.float64)
        so = self.sec_of
        self.pos = (t - starts[so]) / np.maximum(1.0, ends[so] - starts[so]) - 0.5
        sec_loud = np.array([s.loud for s in self.sections])
        self.energy = np.clip((self.loud - sec_loud[so]) * 3.0, -1.0, 1.0)

    # --- pattern families ---------------------------------------------------------------------------
    def plan_families(self, L: float, rng: np.random.Generator) -> None:
        allowed = list(self.cfg.families) if self.cfg.families else list(FAMILIES)
        prev = None
        for i, s in enumerate(self.sections):
            if s.repeat_of is not None and s.repeat_of < i and self.cfg.repeat_memory and self.sections[s.repeat_of].family in allowed:
                s.family = self.sections[s.repeat_of].family
                s.mirror = bool(rng.random() < 0.5) if self.cfg.mirror_repeats is None else bool(self.cfg.mirror_repeats)
                prev = s.family
                continue
            score = {}
            for fam in allowed:
                _, _, _, min_level = FAMILY_SHAPE[fam]
                v = 0.0
                if fam == "stream":
                    v = 0.6 + 0.4 * s.perc
                elif fam == "stairs":
                    v = 0.25 + 0.5 * s.intensity + 1.2 * max(0.0, s.ramp) + 0.2 * s.pitch_motion
                elif fam == "jumpstream":
                    v = 0.4 + 0.8 * s.intensity + 0.3 * s.perc
                elif fam == "handstream":
                    v = 0.1 + 1.1 * s.intensity * s.perc
                elif fam == "chordjack":
                    v = 0.0 + 1.3 * s.perc * s.intensity - 0.4 * s.pitch_motion
                elif fam == "jack":
                    v = 0.1 + 0.8 * s.perc - 0.6 * s.pitch_motion
                elif fam == "trill":
                    v = 0.3 + 0.3 * (1.0 - s.pitch_motion)
                elif fam == "melodic":
                    v = 0.2 + 1.0 * s.pitch_motion + 0.4 * (1.0 - s.perc)
                elif fam == "ln":
                    v = -0.2 + 1.4 * s.sustain * (1.0 - s.perc) + (0.8 if (self.cfg.ln_ratio or 0) > 0.3 else 0.0)
                    if self.cfg.ln_ratio is not None and self.cfg.ln_ratio <= 0.02:
                        v -= 5.0
                if L < min_level:
                    v -= 2.0 * (min_level - L) / max(0.05, min_level)
                if fam == prev:
                    v -= 0.6
                score[fam] = v
            noise = _gumbel(rng, len(allowed)) * (0.15 + 0.85 * self.cfg.creativity) * 0.5
            vals = np.array([score[f] for f in allowed]) + noise
            s.family = allowed[int(np.argmax(vals))]
            s.mirror = False
            prev = s.family

    def section_shape(self) -> np.ndarray:
        """Relative note density of every section (duration-weighted mean 1).

        Louder / busier sections get more notes, breaks none. How strong the contrast is comes from the data:
        the exponent fitted on real maps of the target star rating (section density vs audio intensity, needs a
        pattern model learned from a prepared dataset), else matched to how unevenly real maps spread their notes,
        else 1.5; always x ``contrast``.
        """
        if getattr(self, "_shape", None) is not None:
            return self._shape
        inten = np.array([s.intensity for s in self.sections])
        dur = np.array([s.dur_s for s in self.sections])

        def shape_for(e: float) -> np.ndarray:
            sh = np.where(inten > 0, (0.3 + 0.7 * inten) ** e, 0.0)
            return sh / max(1e-6, (sh * dur).sum() / dur.sum())

        e = 1.5
        fitted = self.patterns.intensity_exponent(self.target_sr) if self.patterns is not None else None
        target = self.patterns.density_variation(self.target_sr) if (self.patterns is not None and fitted is None) else None
        if fitted is not None:
            e = fitted  # measured on real maps: how their section density follows the audio intensity
        elif target is not None and len(self.sections) > 2:
            # section-to-section part of the variation (the rest comes from the rhythm inside sections)
            goal = 0.75 * target
            lo, hi = 0.0, 6.0
            for _ in range(30):
                e = 0.5 * (lo + hi)
                sh = shape_for(e)
                active = inten > 0
                w = dur[active] / dur[active].sum()
                cv = float(np.sqrt((w * (sh[active] - (w * sh[active]).sum()) ** 2).sum()) / max(1e-6, (w * sh[active]).sum()))
                if cv < goal:
                    lo = e
                else:
                    hi = e
        shape = shape_for(e * self.cfg.contrast)
        w = float(np.clip(self.cfg.net_shape * self.cfg.net_weight, 0.0, 1.0)) if self.hints is not None else 0.0
        if w > 0:
            # the network's expected number of notes per section: on real songs its density graph follows the mapper's
            # far better than the audio intensity does (Poisoned Bathwater: 0.94 vs 0.75)
            H = self.an.frame_ms
            pe = self.hints.p_event
            dens = np.array([float(pe[s.start_ms // H: max(s.start_ms // H + 1, s.end_ms // H)].mean()) for s in self.sections])
            net_shape = np.maximum(dens, 1e-4) ** self.cfg.contrast
            net_shape = net_shape / max(1e-6, (net_shape * dur).sum() / dur.sum())
            shape = np.where(shape > 0, shape, 1e-3) ** (1.0 - w) * net_shape ** w
            shape = shape / max(1e-6, (shape * dur).sum() / dur.sum())
        self._shape = shape
        return self._shape

    # --- rhythm (which candidates become events) ---------------------------------------------------
    def select_events(self, L: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        eps, gap, _ = _level_params(L)
        t = self.t
        w_net = cfg.net_weight if self.hints is not None else 0.0
        # simple rhythms first on easy levels, audio decides on hard ones
        w_metric = 0.55 * (1.2 - min(1.0, L))
        base = 1.6 * self.attack + w_metric * self.metric / 4.0
        if self.hints is not None:
            # the network knows which sounds mappers map: its note probability ranks the candidates first
            q = np.clip(self.net, 0.02, 0.98)
            base = base + cfg.net_rhythm * w_net * np.log(q / (1.0 - q))
        if self.patterns is not None:
            # how often real maps of this difficulty put a note on a tick of this divisor
            use = self.patterns.tick_usage(self.target_sr)
            cls = DIV_TO_CLASS[np.clip(self.div, 0, 16)]
            base = base + 0.45 * np.log(use[cls] / (1.0 - use[cls]))
        base = base + 0.05 * self.cfg.creativity * rng.standard_normal(t.size)
        shape = self.section_shape()
        if getattr(self, "bar_edges", None) is not None:
            self._bar_shape_cache = self.bar_shape()
            # a section's budget follows the bar-refined shape too (sections themselves are long and coarse)
            dur_b = np.diff(self.bar_edges).astype(np.float64)
            shape = shape.copy()
            for si, sec_ in enumerate(self.sections):
                ov = np.clip(np.minimum(self.bar_edges[1:], sec_.end_ms) - np.maximum(self.bar_edges[:-1], sec_.start_ms), 0, None)
                if ov.sum() > 0 and shape[si] > 0:
                    shape[si] = float((self._bar_shape_cache * ov).sum() / ov.sum())
        chosen = np.zeros(t.size, dtype=bool)
        sec_of = self.sec_of
        chosen_t: List[np.ndarray] = []  # times chosen so far (sections are processed in order)
        groups = []
        global_pick = self.hints is not None and cfg.net_shape * cfg.net_weight >= 0.5
        for si, s in enumerate(self.sections):
            idx = np.flatnonzero(sec_of == si)
            if idx.size == 0 or shape[si] <= 0:
                continue
            fam_d = FAMILY_SHAPE[s.family][0]
            budget = int(round(s.dur_s * eps * shape[si] * fam_d))
            # notes need something to hit: a section gets at most ~1.25x as many notes as it has audible attacks
            # (plus what the network is confident about); filler on silent ticks is what makes maps feel random
            audible = int((self.audible[idx] | (self.net[idx] >= 0.5)).sum())
            budget = min(budget, int(round(audible * (1.25 + 0.5 * cfg.creativity))) + 2)
            if budget <= 0:
                continue
            sc = base[idx].copy()
            sc = sc + 0.8 * s.ramp * self.pos[idx] + 0.5 * self.energy[idx]
            div = self.div[idx]
            # binary vs triplet feel of the section; where the network is sure about a note of the other kind (a 1/12
            # fill in a straight section), it wins
            other = np.isin(div, BINARY_FINE) if s.triplet else np.isin(div, list(TRIPLET))
            sure = np.clip(self.net[idx], 0.0, 1.0) * min(1.0, w_net) if self.hints is not None else 0.0
            sc = sc - 0.8 * other * (1.0 - sure)
            if cfg.repeat_memory and chosen_t and self.an.repeats:
                # remember the rhythm of the earlier occurrence of a repeated phrase
                src = np.sort(np.concatenate(chosen_t))
                rel = self.an.source_time(t[idx])
                if src.size:
                    pos = np.clip(np.searchsorted(src, rel), 1, max(1, src.size - 1))
                    d = np.minimum(np.abs(src[np.minimum(pos, src.size - 1)] - rel), np.abs(src[pos - 1] - rel))
                    sc = sc + 1.2 * ((d <= 6) & (rel >= 0))
            g = gap * (1.25 if s.family in ("chordjack", "jack") else 1.0)
            gv = np.full(idx.size, g)
            if self.hints is not None:
                # a note the network is sure about may sit closer to its neighbours (real 1/8 and 1/16 runs)
                gv = gv * (1.0 - 0.8 * np.clip(self.net[idx], 0.0, 1.0) * min(1.0, w_net))
            groups.append((idx, sc, budget, gv))
            if not global_pick:
                if getattr(self, "bar_edges", None) is not None and cfg.bar_follow > 0:
                    # the section's budget is shared out over its bars by their loudness: the density changes in the
                    # bar where the music changes (a drum entry), not at the next section boundary
                    bars = self.bar_of_t[idx]
                    bshape = self._bar_shape_cache
                    ub = np.unique(bars)
                    wb = np.array([bshape[b] * float(min(self.bar_edges[b + 1], s.end_ms) - max(self.bar_edges[b], s.start_ms))
                                   for b in ub])
                    want = budget * wb / max(1e-9, wb.sum())
                    carry = 0.0
                    prev_t = np.concatenate(chosen_t) if chosen_t else np.empty(0, np.int64)
                    taken_all: List[int] = []
                    for b, nb_want in zip(ub, want):
                        sel = bars == b
                        nb_i = int(math.floor(nb_want + carry + 0.5))
                        carry += nb_want - nb_i
                        aud = int((self.audible[idx[sel]] | (self.net[idx[sel]] >= 0.5)).sum())
                        nb_i = min(nb_i, int(round(aud * (1.25 + 0.5 * cfg.creativity))) + 1)
                        if nb_i <= 0:
                            continue
                        near = np.concatenate([prev_t[-8:], t[taken_all[-8:]]]) if taken_all or prev_t.size else None
                        tk = _greedy_pick(t, idx[sel], sc[sel], nb_i, gv[sel], near)
                        taken_all += tk
                    taken = taken_all
                else:
                    taken = _greedy_pick(t, idx, sc, budget, gv)
                chosen[taken] = True
                chosen_t.append(t[taken])
        if global_pick and groups:
            # with a network the whole song is one pool: the network decides WHERE the notes go (its density shape
            # is already in the section budgets), per-section quotas would move notes away from where it wants them
            idx = np.concatenate([g_[0] for g_ in groups])
            sc = np.concatenate([g_[1] for g_ in groups])
            gv = np.concatenate([g_[3] for g_ in groups])
            taken = _greedy_pick(t, idx, sc, int(sum(g_[2] for g_ in groups)), gv)
            chosen[taken] = True
        ev = np.flatnonzero(chosen)
        # two notes closer than 20 ms across a section border are one note
        keep = np.ones(ev.size, dtype=bool)
        last = -10 ** 9
        for i, k in enumerate(ev):
            if t[k] - last < 20:
                keep[i] = False
            else:
                last = t[k]
        ev = ev[keep]
        if cfg.repeat_memory and self.an.repeats:
            ev = self._copy_repeats(ev)
        return self._drop_silent(ev), base

    def _drop_silent(self, ev: np.ndarray) -> np.ndarray:
        """No note in silence: silent bars (intro / outro / after the music ends) and silent moments without an attack."""
        if getattr(self, "bar_silent", None) is None or ev.size == 0:
            return ev
        bad = self.bar_silent[self.bar_of_t[ev]] | self.t_silent[ev]
        return ev[~bad]

    def _copy_repeats(self, ev: np.ndarray) -> np.ndarray:
        """A repeated phrase gets the rhythm of its original (copy-paste, as mappers do); phrases are processed in
        time order, so a repeat of a repeat copies the already copied rhythm."""
        t = self.t
        chosen = set(int(k) for k in ev)
        for a, b, src, _ in sorted(self.an.repeats):
            ks_src = [k for k in chosen if src <= t[k] < src + (b - a)]
            if not ks_src:
                continue
            want = np.array(sorted(int(t[k]) + (a - src) for k in ks_src), dtype=np.int64)
            pos = np.clip(np.searchsorted(t, want), 1, t.size - 1)
            near = np.where(np.abs(t[pos - 1] - want) <= np.abs(t[pos] - want), pos - 1, pos)
            ok = np.abs(t[near] - want) <= 6
            chosen = {k for k in chosen if not (a <= t[k] < b)} | set(int(k) for k in near[ok])
        return np.array(sorted(chosen), dtype=np.int64)

    def _bursts(self, ev: np.ndarray, base: np.ndarray, L: float, rng: np.random.Generator) -> np.ndarray:
        """Speed-ups: a short run switches to twice the tempo (1/4 -> 1/8, 1/2 -> 1/4) where the audio has fast
        attacks between the chosen notes, e.g. a stair 1 2 3 4 played as 1/8 instead of 1/4.

        The data is a prior, not a ban: how fast and how often real maps of the target star rating burst (their
        fastest chord gaps, their share of gaps at the doubled speed) sets the typical amount, ``creativity`` lets the
        composer go up to 40 % faster and up to 4x more often than that. Physical floor: 25 ms.
        """
        if ev.size < 4:
            return ev
        t = self.t
        floor = share = None
        if self.patterns is not None:
            floor = self.patterns.event_gap_floor(self.target_sr)
        if floor is None:
            floor = max(38.0, _level_params(min(1.6, L + 0.35))[1])
        floor = max(25.0, floor * (1.0 - 0.4 * self.cfg.creativity))
        often = 1.0 + 3.0 * self.cfg.creativity
        te = t[ev]
        gaps = np.diff(te)
        added: List[int] = []
        i = 0
        budget_left = None
        while i < gaps.size - 2:
            g = int(gaps[i])
            half = g / 2.0
            # a regular run of at least 3 equal gaps
            j = i
            while j < gaps.size and abs(int(gaps[j]) - g) <= 3:
                j += 1
            run = j - i
            if run >= 3 and half >= floor:
                if self.patterns is not None:
                    share = self.patterns.gap_share_below(self.target_sr, half + 3)
                    if budget_left is None:
                        budget_left = int(max(share or 0.0, 0.002) * often * ev.size)
                if budget_left is not None and budget_left <= 0:
                    break
                s = self.sections[int(self.sec_of[ev[i]])]
                # the in-between ticks must carry real attacks; build-ups and loud parts burst more willingly
                mids = te[i:j] + int(round(half))
                pos = np.clip(np.searchsorted(t, mids), 0, t.size - 1)
                pos = np.where(np.abs(t[np.maximum(pos - 1, 0)] - mids) < np.abs(t[pos] - mids), np.maximum(pos - 1, 0), pos)
                ok = np.abs(t[pos] - mids) <= 4
                strong = (self.attack[pos] >= 0.72 - 0.15 * max(0.0, s.ramp) - 0.05 * self.cfg.creativity) & self.audible[pos]
                want = ok & strong
                if self.hints is not None:
                    want = want & (self.net[pos] >= 0.35)  # with a network: only where it hears a note too
                # take the longest stretch of consecutive wanted fills, at most 2 beats
                best_a = best_b = cur = 0
                for q in range(want.size):
                    cur = cur + 1 if want[q] else 0
                    if cur > best_b - best_a:
                        best_a, best_b = q - cur + 1, q + 1
                n_fill = best_b - best_a
                if n_fill >= 2 and rng.random() < 0.5 + 0.4 * self.cfg.creativity:
                    n_fill = min(n_fill, max(2, int(round(2 * float(self.an.tick_beat[min(self.an.tick_beat.size - 1, int(np.searchsorted(self.an.ticks, te[i])))]) / g))))
                    if budget_left is not None:
                        n_fill = min(n_fill, budget_left)
                        budget_left -= n_fill
                    added.extend(int(p) for p in pos[best_a:best_a + n_fill])
            i = max(j, i + 1)
        if not added:
            return ev
        return np.unique(np.concatenate([ev, np.array(added, dtype=np.int64)]))

    # --- chord sizes -------------------------------------------------------------------------------
    def chord_sizes(self, ev: np.ndarray, L: float, rng: np.random.Generator) -> np.ndarray:
        _, _, rate = _level_params(L)
        sizes = np.ones(ev.size, dtype=np.int64)
        # largest chord: what real maps of this star rating use (pattern model), else a level-based default
        cap = min(self.cfg.max_chord, 2 if L < 0.35 else (3 if L < 0.7 else 4))
        accent = self.attack[ev] + 0.25 * (self.metric[ev] >= 3.0) + 0.05 * self.cfg.creativity * rng.random(ev.size)
        ramp = np.array([s.ramp for s in self.sections])[self.sec_of[ev]]
        accent = accent + 1.0 * ramp * self.pos[ev] + 0.4 * self.energy[ev]   # chords grow along a build-up
        net_tail = None
        if self.hints is not None and self.hints.chord_logprobs is not None:
            # the network hears which hits are chords (exact chord size on 92 % of the chords of Poisoned Bathwater):
            # P(>= 2), P(>= 3), P(4) rank the moments for each chord size; the composer only sets how many
            f = np.clip(self.t[ev] // self.an.frame_ms, 0, self.an.num_frames - 1)
            p = np.exp(self.hints.chord_logprobs[f])
            net_tail = np.stack([p[:, 1:].sum(axis=1), p[:, 2:].sum(axis=1), p[:, 3]], axis=1)
            accent = accent + 0.6 * self.cfg.net_weight * net_tail[:, 0]
        sec = self.sec_of[ev]
        t = self.t[ev]
        gap_prev = np.diff(t, prepend=t[0] - 10 ** 6) if t.size else t
        mbin = np.where(self.metric[ev] >= 3.0, 0, np.where(self.metric[ev] >= 1.2, 1, 2))
        if self.patterns is not None:
            # chord shares of real maps at this star rating, metric position and speed
            pm = self.patterns
            dist = np.array([pm.chord_distribution(self.target_sr, int(mb), float(g)) for mb, g in zip(mbin, gap_prev)])
            tails = np.stack([dist[:, 1:].sum(axis=1), dist[:, 2:].sum(axis=1), dist[:, 3]], axis=1)
            # usual largest chord of this star rating; creative maps may go one bigger on the strongest accents
            cap = min(self.cfg.max_chord, pm.max_chord(self.target_sr) + (1 if (self.cfg.creativity >= 0.7 or self.chord_boost > 1.0) else 0))
        # families and intensity move chords between sections; with a pattern model the song-wide chord share
        # stays exactly what real maps of this star rating have (the multipliers are mean-preserving)
        mults = np.array([FAMILY_SHAPE[s.family][1] * (0.6 + 0.8 * s.intensity) for s in self.sections])
        counts = np.bincount(sec, minlength=len(self.sections)).astype(np.float64)
        if self.patterns is not None and counts.sum() > 0:
            mults = mults / max(1e-6, float((mults * counts).sum() / counts.sum()))
        for si, s in enumerate(self.sections):
            m = np.flatnonzero(sec == si)
            if m.size == 0:
                continue
            mult = mults[si]
            u = 1.0 - _rank01(accent[m])           # strongest accents first
            if net_tail is not None and self.cfg.net_weight > 0:
                w = min(1.0, self.cfg.net_weight)
                u = np.stack([1.0 - _rank01(w * net_tail[m, j] + (1.0 - w) * accent[m] / 3.0 + 0.1 * accent[m] / 3.0) for j in range(3)], axis=1)
            else:
                u = np.stack([u, u, u], axis=1)
            if self.patterns is not None:
                r = np.minimum(0.97, tails[m] * mult * self.chord_boost)
                sz = _nested_sizes(u, r)
            else:
                rr = min(0.97, rate * mult * self.chord_boost)
                sz = _nested_sizes(u, np.tile([rr, rr * rr * 0.55, rr ** 3 * 0.25], (m.size, 1)))
            sizes[m] = np.minimum(sz, cap)
        # two chords closer than a playable jack must fit into 4 lanes together
        jack_min = self.jack_min(L)
        for i in range(1, ev.size):
            if gap_prev[i] < jack_min and sizes[i] + sizes[i - 1] > NUM_LANES and self.sections[sec[i]].family != "chordjack":
                weak = i if accent[i] <= accent[i - 1] else i - 1
                sizes[weak] = max(1, NUM_LANES - sizes[i if weak == i - 1 else i - 1])
        return sizes

    # --- lanes, long notes -------------------------------------------------------------------------
    def assign(self, ev: np.ndarray, sizes: np.ndarray, L: float, rng: np.random.Generator) -> List[Note]:
        an, cfg = self.an, self.cfg
        t_all = self.t[ev]
        n = ev.size
        if n == 0:
            return []
        jack_min = self.jack_min(L)                                # fastest lane repeat real maps of this SR use
        rep_cdf = self.patterns.repeat_cdf(self.target_sr) if self.patterns is not None else None
        span_limit = self.patterns.ln_span_limit(self.target_sr, min(0.99, 0.85 + 0.14 * cfg.creativity)) if self.patterns is not None else None
        gaps_next = np.append(np.diff(t_all), 10 ** 6)
        ln_plan = self.plan_long_notes(ev, sizes, rng) if (self.patterns is not None and self.patterns.has_ln_model()) else None
        if ln_plan is not None:
            span_limit = ln_plan[2]
        ln_target = self._ln_targets()
        ln_score = self._ln_scores(ev, gaps_next)
        from .renderer import release_gap_fn
        gap_fn = release_gap_fn(an.timing_points, cfg.release_gap_ms, cfg.release_gap_beats)

        last_start = np.full(NUM_LANES, -10 ** 9, dtype=np.int64)
        busy_until = np.full(NUM_LANES, -10 ** 9, dtype=np.int64)
        hold_note: List[Optional[int]] = [None] * NUM_LANES
        load = np.zeros(2)
        last_t = -10 ** 9
        prev = prev2 = 0
        hist: List[int] = []
        direction = 1
        notes: List[Optional[Note]] = []
        masks_out = np.zeros(n, dtype=np.int64)
        sec = self.sec_of[ev]
        mask_at: Dict[int, int] = {}  # time -> chord mask of everything placed so far
        src_of = self.an.source_time(t_all)  # original moment of repeated phrases (-1 = not a repeat)
        net_ll = self.hints.lane_logits if (self.hints is not None and self.hints.lane_logits is not None) else None
        H = an.frame_ms
        pitch = self.pitch[ev]
        timbre = self.timbre[ev]
        # per-section pitch range for melodic lanes
        p_lo, p_hi = {}, {}
        for si in range(len(self.sections)):
            pp = pitch[sec == si]
            if pp.size:
                p_lo[si], p_hi[si] = float(np.percentile(pp, 10)), float(np.percentile(pp, 90))
        ln_quota: Dict[int, float] = {}
        ln_seen: Dict[int, int] = {}

        for i in range(n):
            t = int(t_all[i])
            si = int(sec[i])
            s = self.sections[si]
            fam = s.family
            k = int(sizes[i])
            dt = t - last_t
            # long notes still held: release them if they would block this chord or force a jack
            free = [c for c in range(NUM_LANES) if busy_until[c] < t - 1]
            fresh = [c for c in free if t - last_start[c] >= jack_min]
            if len(free) < k or (len(fresh) < k and len(free) < NUM_LANES):
                for c in sorted(range(NUM_LANES), key=lambda c: busy_until[c]):
                    if busy_until[c] >= t - 1 and hold_note[c] is not None:
                        j = hold_note[c]
                        old = notes[j]
                        end = t - gap_fn(t)
                        notes[j] = Note(old.start, old.col, end, True) if end - old.start >= 60 else Note(old.start, old.col, old.start, False)
                        busy_until[c] = notes[j].end
                        hold_note[c] = None
                        free.append(c)
                        fresh = [c for c in free if t - last_start[c] >= jack_min]
                        if len(free) >= k and len(fresh) >= k:
                            break
                k = min(k, len(free))
            if fam in ("stream", "trill", "ln") and rng.random() < 0.08 + 0.2 * cfg.creativity or (fam == "stairs" and rng.random() < 0.03 + 0.06 * cfg.creativity):
                direction = -direction  # rolls change direction now and then
            cands = MASKS_BY_SIZE[max(1, k)]
            best_m, best_v = cands[0], -1e18
            # memory of the repeated section: same relative position -> same (or mirrored) chord
            remembered = None
            rel = int(src_of[i])
            if rel >= 0 and cfg.repeat_memory:
                for d in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5):
                    if rel + d in mask_at:
                        remembered = mask_at[rel + d]
                        if s.mirror:
                            remembered = int(sum(1 << (3 - c) for c in LANES_OF[remembered]))
                        break
            same_sound = bool(i > 0 and float(timbre[i] @ timbre[i - 1]) > 0.93 and abs(pitch[i] - pitch[i - 1]) < 0.5)
            prior = None
            if net_ll is not None:
                f = min(an.num_frames - 1, t // H)
                lg = net_ll[f].astype(np.float64)
                masks = LANE_TABLE[prev, prev2]
                prior = np.full(16, -30.0)
                for code in range(N_LANE_CODES):
                    m = int(masks[code])
                    if m:
                        prior[m] = np.logaddexp(prior[m], lg[code])
                prior = prior - np.max(prior[1:])
            noise = _gumbel(rng, len(cands)) * (0.1 + 0.6 * cfg.creativity) * 0.6
            learned = self.patterns.lane_logprob(self.target_sr, dt, prev, prev2, max(1, k)) if self.patterns is not None else None
            fam_w = 0.5 if learned is not None else 1.0
            for ci, m in enumerate(cands):
                lanes = LANES_OF[m]
                v = 0.0
                # a lane under a held long note is not forbidden, but expensive: choosing it releases the LN early
                held = sum(1 for c in lanes if busy_until[c] >= t - 1)
                if held:
                    v -= 8.0 * held
                # ergonomics
                for c in lanes:
                    # from the last press - or from the RELEASE of a long note: a tap right after a released hold on the
                    # same lane is the same awkward repeat as a jack (it used to be measured from the hold's start)
                    since = t - max(int(last_start[c]), min(int(busy_until[c]), t))
                    if rep_cdf is not None and since < 2000:
                        # repeats rarer in real maps of this SR than the 5% fastest cost more the rarer they are
                        q = float(rep_cdf[min(rep_cdf.size - 1, int(since) // 5)])
                        if q < 0.05:
                            v -= self.repeat_scale * 3.0 * (0.05 - q) / 0.05
                    if since < jack_min:
                        allowed_jack = (fam == "jack" and same_sound) or (fam == "chordjack" and since >= 0.7 * jack_min)
                        v -= self.repeat_scale * (1.0 if allowed_jack else (6.0 * (1.0 - since / jack_min) + 2.0) * (1.0 - 0.5 * cfg.creativity))
                lh = sum(1 for c in lanes if c < 2)
                rh = len(lanes) - lh
                v -= 0.6 * max(0.0, (load[0] + lh) - (load[1] + rh)) ** 2 / 8.0 + 0.6 * max(0.0, (load[1] + rh) - (load[0] + lh)) ** 2 / 8.0
                if len(hist) >= 3 and sum(1 for h in hist[-3:] if h & m) >= 3:
                    v -= 1.5 * self.repeat_scale  # anchor
                if k == 2 and m in (0b0011, 0b1100) and fam not in ("chordjack", "handstream"):
                    v -= 0.3
                v_before_family = v
                # family
                if fam == "stairs" or (fam == "stream" and k > 1):
                    if prev:
                        if m == stair_next(prev, k, direction):
                            v += 2.6
                        elif m == stair_next(prev, k, -direction):
                            v += 1.0
                elif fam == "stream" or (fam == "ln" and k == 1):
                    if prev and k == 1 and POP[prev] == 1:
                        pc = LANES_OF[prev][0]
                        nxt = pc + direction
                        if nxt < 0 or nxt > 3:
                            nxt = pc - direction
                        v += 2.0 if lanes[0] == nxt else (0.6 if abs(lanes[0] - pc) == 2 else 0.0)
                elif fam == "trill":
                    if prev2 and m == prev2 and not (m & prev):
                        v += 2.2
                    elif prev and not (m & prev) and POP[m] == POP[prev]:
                        v += 0.8
                elif fam in ("jumpstream", "handstream"):
                    v += 1.0 if not (m & prev) else -0.4 * POP[m & prev]
                    v += 0.3 * abs(CENTROID[m] - CENTROID[prev]) if prev else 0.0
                elif fam == "chordjack":
                    ov = POP[m & prev]
                    v += 1.4 if ov == 1 else (0.5 if ov == 2 and k >= 3 else (-0.8 if m == prev else 0.0))
                elif fam == "jack":
                    if same_sound and prev and m == prev:
                        v += 2.5
                    elif not (m & prev):
                        v += 0.5
                if fam in ("melodic", "ln") and si in p_lo:
                    span = max(1.0, p_hi[si] - p_lo[si])
                    target = 3.0 * np.clip((pitch[i] - p_lo[si]) / span, 0.0, 1.0)
                    v -= 1.2 * abs(CENTROID[m] - target)
                    if i > 0 and prev:
                        dp = np.sign(pitch[i] - pitch[i - 1])
                        dc = np.sign(CENTROID[m] - CENTROID[prev])
                        v += 0.3 * (dp == dc and dp != 0)
                v = v_before_family + fam_w * (v - v_before_family)
                if learned is not None:
                    v += 1.5 * float(learned[m])  # what real maps of this difficulty do after these chords
                if remembered is not None and m == remembered:
                    v += 4.0
                if prior is not None:
                    v += 0.5 * cfg.net_weight * float(prior[m])
                v += noise[ci]
                if v > best_v:
                    best_v, best_m = v, m
            if best_v <= -1e17:
                continue
            m = best_m
            for c in LANES_OF[m]:  # cut a long note that is still held on a lane this chord takes
                if busy_until[c] >= t - 1 and hold_note[c] is not None:
                    j = hold_note[c]
                    old = notes[j]
                    end = t - gap_fn(t)
                    notes[j] = Note(old.start, old.col, end, True) if end - old.start >= 60 else Note(old.start, old.col, old.start, False)
                    busy_until[c] = notes[j].end
                    hold_note[c] = None
            masks_out[i] = m
            mask_at[t] = m
            # long notes
            quota = ln_target[si]
            ln_seen[si] = ln_seen.get(si, 0) + 1
            planned_lanes: set = set()
            if ln_plan is not None and ln_plan[0][i] > 0:
                lanes_m = list(LANES_OF[m])
                rng.shuffle(lanes_m)
                planned_lanes = set(lanes_m[: int(ln_plan[0][i])])
            for c in LANES_OF[m]:
                make_ln = False
                if ln_plan is not None:
                    if c in planned_lanes:
                        length = int(ln_plan[1][i])
                        if span_limit is not None and i + span_limit + 1 < n:
                            length = min(length, int(t_all[i + span_limit + 1] - t) - gap_fn(int(t_all[i + span_limit + 1])))
                        if length >= 40:
                            notes.append(Note(t, c, t + length, True))
                            hold_note[c] = len(notes) - 1
                            busy_until[c] = t + length
                            last_start[c] = t
                            continue
                elif quota > 0 and ln_score[i] > 0:
                    have = ln_quota.get(si, 0.0)
                    if have < quota * ln_seen[si] * POP[m]:
                        make_ln = ln_score[i] >= 0.35 or have < 0.5 * quota * ln_seen[si] * POP[m]
                if make_ln:
                    length = self._ln_length(t, i, t_all, gaps_next)
                    if span_limit is not None and i + span_limit + 1 < n:
                        # hold over at most as many chords as real maps of this SR usually do
                        length = min(length, int(t_all[i + span_limit + 1] - t) - gap_fn(int(t_all[i + span_limit + 1])))
                    if length >= 60:
                        notes.append(Note(t, c, t + length, True))
                        hold_note[c] = len(notes) - 1
                        busy_until[c] = t + length
                        last_start[c] = t
                        ln_quota[si] = ln_quota.get(si, 0.0) + 1.0
                        continue
                notes.append(Note(t, c, t, False))
                last_start[c] = t
                busy_until[c] = t
                hold_note[c] = None
            decay = math.exp(-max(0, dt) / 700.0)
            load = load * decay + np.array([sum(1 for c in LANES_OF[m] if c < 2), sum(1 for c in LANES_OF[m] if c >= 2)])
            hist.append(m)
            prev2, prev = prev, m
            last_t = t
        from .renderer import resolve_lane_conflicts
        return resolve_lane_conflicts([x for x in notes if x is not None], cfg.release_gap_ms, 30, gap_fn)

    def ln_style_and_share(self) -> Tuple[int, float]:
        """LN style of the map and its target LN share. The style is a mapper's choice (real maps of one star rating
        range from 0 % to 55 % long notes), so it is never averaged: an explicit ratio, a named style, or the most
        common style of real maps at this star rating."""
        from .patterns import LN_STYLE_NAMES, LN_STYLE_SHARE, ln_style_bin
        if self.cfg.ln_ratio is not None:
            r = float(self.cfg.ln_ratio)
            return ln_style_bin(r), r
        if self.cfg.ln_style:
            st = LN_STYLE_NAMES.index(self.cfg.ln_style)
            return st, LN_STYLE_SHARE[st]
        st = self.patterns.typical_ln_style(self.target_sr)
        return st, LN_STYLE_SHARE[st]

    def plan_long_notes(self, ev: np.ndarray, sizes: np.ndarray, rng: np.random.Generator):
        """(long members per event, LN length in ms per event, hold span limit) from the learned LN model.

        Which chords get long notes comes from the rhythm context in real maps of this LN style and star rating
        (time to the next / from the previous chord in beats, chord size, metric position): measured on real maps the
        rhythm predicts LN placement (AUC 0.65), the sound of the note barely does (0.53). The most likely chords are
        taken until the target LN share is reached. The length is drawn from the real length distribution in beats
        for that context, preferring releases on a following chord's press as often as real maps do it.
        """
        pm = self.patterns
        n = ev.size
        members = np.zeros(n, dtype=np.int64)
        lengths = np.zeros(n, dtype=np.int64)
        if n == 0:
            return members, lengths, None
        style, share = self.ln_style_and_share()
        if share <= 0.0:
            return members, lengths, 0
        from .patterns import LN_LEN_BEATS
        t = self.t[ev]
        tb = self.an.ticks
        beat = self.an.tick_beat[np.clip(np.searchsorted(tb, t), 0, max(0, tb.size - 1))] if tb.size else np.full(n, 500.0)
        gn = np.append(np.diff(t), 10 ** 6) / beat
        gp = np.diff(t, prepend=t[0] - 10 ** 6) / beat
        mbin = np.where(self.metric[ev] >= 3.0, 0, np.where(self.metric[ev] >= 1.2, 1, 2))
        p = np.array([pm.ln_chord_prob(self.target_sr, style, float(a), float(b), int(k), int(mb))
                      for a, b, k, mb in zip(gn, gp, sizes, mbin)])
        p = np.clip(p, 1e-4, 1 - 1e-4)
        score = np.log(p / (1 - p))
        fam_bonus = np.array([0.8 if self.sections[int(si)].family == "ln" else 0.0 for si in self.sec_of[ev]])
        score = score + fam_bonus + (_gumbel(rng, n) * 0.4 * self.cfg.creativity if self.hints is None else 0.0)
        if self.hints is not None and self.hints.ln_prob is not None:
            f = np.clip(t // self.an.frame_ms, 0, self.an.num_frames - 1)
            q = np.clip(self.hints.ln_prob[f], 1e-4, 1 - 1e-4)
            score = score + 2.0 * self.cfg.net_weight * np.log(q / (1 - q))  # the network hears which sounds are held
        target = share * float(sizes.sum())
        have = 0.0
        r_on_chord = pm.ln_release_on_chord(self.target_sr, style)
        for i in np.argsort(-score, kind="stable"):
            if have >= target:
                break
            dist = pm.ln_member_count(self.target_sr, style, int(sizes[i]))
            # how many chord notes are held: the most common count in real maps of this style (not a dice roll);
            # creative maps draw it from the distribution
            k = int(rng.choice(dist.size, p=dist)) + 1 if self.cfg.creativity > 0.6 else int(np.argmax(dist)) + 1
            # length: drawn from the real distribution (in beats) for this gap, not its mode - taking the most likely
            # length every time would average the map again. How often a release lands on a later press is matched
            # to real maps of this style. The network (if any) adds its own length: it hears how long a sound lasts.
            pl = pm.ln_length_dist(self.target_sr, style, float(gn[i]))
            ends = t[i] + np.round(LN_LEN_BEATS * beat[i]).astype(np.int64)
            if self.hints is not None and self.hints.ln_len_ms is not None:
                f = int(min(self.an.num_frames - 1, t[i] // self.an.frame_ms))
                ends = np.append(ends, t[i] + int(self.hints.ln_len_ms[f] * 1.0))
                w_len = min(0.8, 0.8 * self.cfg.net_weight)  # the network's length (it hears how long the sound lasts) first
                pl = np.append(pl * (1.0 - w_len), w_len)
            j = np.clip(np.searchsorted(t, ends), 0, n - 1)
            jm = np.clip(j - 1, 0, n - 1)
            near = np.minimum(np.abs(t[j] - ends), np.abs(t[jm] - ends))
            on_chord = near <= np.maximum(8, beat[i] / 16)
            m_on = float(pl[on_chord].sum())
            if 0.0 < m_on < 1.0:
                pl = np.where(on_chord, pl * r_on_chord / m_on, pl * (1.0 - r_on_chord) / (1.0 - m_on))
            temp = 1.0 + 0.6 * max(0.0, self.cfg.creativity - 0.5)   # creative maps spread the lengths a little more
            pl = pl ** (1.0 / temp)
            pl = pl / pl.sum()
            c = int(rng.choice(pl.size, p=pl))
            end = int(ends[c])
            if on_chord[c]:  # release exactly on that press
                end = int(t[j[c]] if abs(t[j[c]] - end) <= abs(t[jm[c]] - end) else t[jm[c]])
            elif self.an.ticks.size:  # a free release still sits on the grid (real maps: ~100 %)
                q = int(np.searchsorted(self.an.ticks, end))
                cand = self.an.ticks[max(0, q - 1): q + 1]
                end = int(cand[np.argmin(np.abs(cand - end))]) if cand.size else end
            if end - t[i] < 40:
                continue
            members[i] = min(k, int(sizes[i]))
            lengths[i] = end - t[i]
            have += members[i]
        span = pm.ln_span_limit_style(self.target_sr, style, min(0.99, 0.85 + 0.14 * self.cfg.creativity))
        return members, lengths, span

    def _ln_targets(self) -> Dict[int, float]:
        out = {}
        for si, s in enumerate(self.sections):
            if self.cfg.ln_ratio is None and self.patterns is not None:
                learned = self.patterns.ln_share(self.target_sr)
                audio = 0.5 + 1.5 * max(-0.3, s.sustain - 0.5) * (1.0 - s.perc)
                base = learned * audio + FAMILY_SHAPE[s.family][2]
            elif self.cfg.ln_ratio is not None:
                base = float(self.cfg.ln_ratio)
                base = min(0.95, base + (0.25 if s.family == "ln" and base > 0 else 0.0))
            else:
                base = 0.04 + 0.35 * max(0.0, s.sustain - 0.45) * (1.0 - s.perc) + FAMILY_SHAPE[s.family][2]
            out[si] = float(np.clip(base, 0.0, 0.95))
        return out

    def _ln_scores(self, ev: np.ndarray, gaps_next: np.ndarray) -> np.ndarray:
        an, H = self.an, self.an.frame_ms
        t = self.t[ev]
        f0 = np.clip(t // H, 0, an.num_frames - 1)
        f1 = np.clip((t + np.minimum(gaps_next, 1500)) // H, 0, an.num_frames - 1)
        cs = np.concatenate([[0.0], np.cumsum(an.harm)])
        mean_after = (cs[f1 + 1] - cs[f0 + 1]) / np.maximum(1, f1 - f0)
        z = (mean_after - np.median(an.harm)) / (an.harm.std() + 1e-6)
        score = 1.0 / (1.0 + np.exp(-z)) * (gaps_next >= 120)
        if self.hints is not None and self.hints.ln_prob is not None:
            score = 0.5 * score + 0.5 * self.cfg.net_weight * self.hints.ln_prob[f0]
        return score

    def _ln_length(self, t: int, i: int, t_all: np.ndarray, gaps_next: np.ndarray) -> int:
        an, H = self.an, self.an.frame_ms
        f0 = min(an.num_frames - 1, t // H)
        level = an.harm[f0: f0 + 3].mean()
        horizon = int(min(4000, max(gaps_next[i] * 3, 200)))
        seg = an.harm[f0: min(an.num_frames, f0 + horizon // H)]
        below = np.flatnonzero(seg < level - 1.0)
        length = int(below[0] * H) if below.size else horizon
        length = max(length, int(gaps_next[i]))
        # snap the release to the previous tick
        ticks = an.ticks
        p = int(np.searchsorted(ticks, t + length, side="right")) - 1
        if 0 <= p < ticks.size and ticks[p] > t:
            length = int(ticks[p] - t)
        return max(0, min(length, 8000))

    # --- one complete map for a level ----------------------------------------------------------------
    def render(self, L: float, seed: int, plan_level: Optional[float] = None) -> List[Note]:
        """One complete map. The pattern plan and the speed-up plan depend on ``plan_level`` (default: L) and stay
        fixed while the star-rating search moves L, so the search sees a (nearly) monotonic star rating."""
        P = L if plan_level is None else plan_level
        self.plan_families(P, np.random.default_rng(seed + 7))
        self._shape = None
        key = (seed, round(P, 4))
        if getattr(self, "_burst_key", None) != key:
            ev0, base0 = self.select_events(P, np.random.default_rng(seed))
            with_b = self._bursts(ev0, base0, P, np.random.default_rng(seed + 11))
            self._burst_ticks = np.setdiff1d(with_b, ev0)
            self._burst_key = key
            self._shape = None
        rng = np.random.default_rng(seed)
        ev, _ = self.select_events(L, rng)
        burst = self._burst_ticks
        if burst.size and ev.size:
            # a planned speed-up note is used when the run it fills is present at this level
            pos = np.clip(np.searchsorted(ev, burst), 1, ev.size - 1)
            left, right = self.t[ev[pos - 1]], self.t[ev[np.minimum(pos, ev.size - 1)]]
            tb = self.t[burst]
            span = right - left
            keep = (left < tb) & (tb < right) & (span <= 2.2 * np.minimum(tb - left, right - tb) + 8)
            burst = burst[keep]
            ev = np.unique(np.concatenate([ev, burst]))
            if self.cfg.repeat_memory and self.an.repeats:
                ev = self._copy_repeats(ev)  # speed-ups of the original are repeated too
            ev = self._drop_silent(ev)
        sizes = self.chord_sizes(ev, L, rng)
        if burst.size:
            sizes[np.isin(ev, burst)] = 1   # speed-ups are single-note runs (a stair at double tempo)
            nb = np.isin(ev, burst)
            nxt = np.flatnonzero(nb[:-1])
            sizes[nxt + 1] = np.minimum(sizes[nxt + 1], 2)  # and do not slam into a big chord
        return self.assign(ev, sizes, L, rng)


# ----------------------------------------------------------------------------------------------
# star rating, critic, public entry point
# ----------------------------------------------------------------------------------------------

def star_rating(notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], od: float = 8.0) -> float:
    from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating

    if not notes:
        return 0.0
    tps = list(timing_points) or [RawTimingPoint(0.0, 500.0)]
    text = map_to_osu_text(ManiaMap(notes=list(notes), timing_points=tps, overall_difficulty=od))
    hits = [(n.col, n.start, n.end, n.ln) for n in notes]
    return float(calculate_osu_mania_star_rating(osu_text=text, hit_objects=hits, overall_difficulty=od, keys=4))


def pattern_entropy(notes: Sequence[Note]) -> float:
    """Entropy (bits) of consecutive chord-to-chord transitions: 0 = one pattern repeated."""
    groups: Dict[int, int] = {}
    for n in notes:
        groups[n.start] = groups.get(n.start, 0) | (1 << n.col)
    seq = [groups[t] for t in sorted(groups)]
    if len(seq) < 3:
        return 0.0
    pairs: Dict[Tuple[int, int], int] = {}
    for a, b in zip(seq[:-1], seq[1:]):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    p = np.array(list(pairs.values()), dtype=np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def fast_jack_share(notes: Sequence[Note], jack_ms: float = 160.0) -> float:
    """Share of chords that repeat a lane of the previous chord less than ``jack_ms`` after it."""
    from .ergonomics import ergonomics_report
    return float(ergonomics_report(notes, jack_ms=jack_ms).get("fast_jack_pct", 0.0)) / 100.0


def critic(notes: Sequence[Note], sr: float, target: float, an: SongAnalysis,
           jack_target: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
    """Score of a composed variant. Jacks are judged against what real maps (or the chosen style) do at this star
    rating: a chordjack style wants jacks, a stream style does not; without data, fewer jacks are better."""
    from .ergonomics import ergonomics_report

    if not notes:
        return -1e9, {}
    er = ergonomics_report(notes)
    ent = pattern_entropy(notes)
    starts = np.array([n.start for n in notes])
    on_attack = float(np.mean(an.attack_at(starts) > np.percentile(an.attack_ms, 75)))
    jack_pct = er.get("fast_jack_pct", 0.0)
    if jack_target is not None:
        jack_term = 8.0 * abs(fast_jack_share(notes) - jack_target)  # same 160 ms cut as the model statistics
    else:
        jack_term = 0.04 * jack_pct
    score = (-20.0 * max(0.0, abs(sr - target) - 0.02) - jack_term - 0.25 * max(0.0, er.get("max_one_hand_run", 0.0) - 6)
             - 1.5 * er.get("hand_imbalance", 0.0) + 0.35 * min(ent, 7.0) + 1.0 * on_attack)
    return float(score), {"pattern_entropy": ent, "on_attack": on_attack, **er}


def compose(
    an: SongAnalysis,
    target_sr: float,
    cfg: Optional[ComposerConfig] = None,
    hints: Optional[NetHints] = None,
    sections: Optional[List[Section]] = None,
    log: Callable[[str], None] = print,
    patterns=None,
) -> ComposeResult:
    """Composes the best of ``cfg.variants`` maps whose star rating matches ``target_sr``."""
    cfg = cfg or ComposerConfig()
    secs = sections if sections is not None else segment_sections(an)
    comp = Composer(an, secs, cfg, hints, log, patterns=patterns, target_sr=target_sr)
    # how often real maps (or the chosen style) repeat a lane within 150 ms at this star rating
    jack_target = patterns.fast_repeat_share(target_sr) if (patterns is not None and hasattr(patterns, "fast_repeat_share")) else None
    best: Optional[ComposeResult] = None
    best_score = -float("inf")
    for v in range(max(1, cfg.variants)):
        seed = cfg.seed * 1000 + v
        lo, hi = 0.0, LEVEL_MAX
        comp.chord_boost = 1.0
        comp.repeat_scale = 1.0
        cache: Dict[float, Tuple[float, List[Note]]] = {}

        plan_level = float(np.clip(0.15 * target_sr + 0.17, 0.0, 1.6))  # typical level of this SR (measured on real songs)

        def run(L: float) -> Tuple[float, List[Note]]:
            if L not in cache:
                notes = comp.render(L, seed, plan_level)
                cache[L] = (star_rating(notes, an.timing_points, cfg.od), notes)
            return cache[L]

        L = plan_level
        sr, notes = run(L)
        for _ in range(cfg.max_iterations):
            if abs(sr - target_sr) <= cfg.sr_tolerance:
                break
            if sr < target_sr:
                lo = L
            else:
                hi = L
            if hi - lo < 1e-3:
                break
            L = 0.5 * (lo + hi)
            sr, notes = run(L)
        # the star rating is a step function of L (notes come and go in whole events): when bisection did not
        # land inside the tolerance, scan around the bracket and keep the closest level
        L, (sr, notes) = min(cache.items(), key=lambda kv: abs(kv[1][0] - target_sr))
        if abs(sr - target_sr) > cfg.sr_tolerance:
            for Lg in np.linspace(max(0.0, L - 0.08), min(LEVEL_MAX, L + 0.08), 13):
                run(float(round(Lg, 4)))
            L, (sr, notes) = min(cache.items(), key=lambda kv: abs(kv[1][0] - target_sr))
        if sr < target_sr - cfg.sr_tolerance:
            # the music has no more room for notes: the remaining difficulty comes from chords (chordjacks,
            # handstreams), the way the hardest real maps get there
            L = max(L, max(cache, key=lambda x: cache[x][0]))
            b_lo, b_hi = 1.0, 4.0
            best_b = (abs(sr - target_sr), 1.0, sr, notes)
            for _ in range(10):
                b = 0.5 * (b_lo + b_hi)
                comp.chord_boost = b
                n_b = comp.render(L, seed, plan_level)
                sr_b = star_rating(n_b, an.timing_points, cfg.od)
                if abs(sr_b - target_sr) < best_b[0]:
                    best_b = (abs(sr_b - target_sr), b, sr_b, n_b)
                if abs(sr_b - target_sr) <= cfg.sr_tolerance:
                    break
                if sr_b < target_sr:
                    b_lo = b
                else:
                    b_hi = b
            _, comp.chord_boost, sr, notes = best_b
        if jack_target is not None:
            L, sr, notes = _calibrate_repeats(comp, an, cfg, L, sr, notes, seed, plan_level, target_sr, jack_target)
        comp.plan_families(plan_level, np.random.default_rng(seed + 7))
        score, rep = critic(notes, sr, target_sr, an, jack_target)
        rep.update({"level": L, "variant": v, "critic": score, "chord_boost": comp.chord_boost,
                    "repeat_scale": comp.repeat_scale, "jack_target": jack_target})
        log(f"    variant {v + 1}/{cfg.variants}: level {L:.3f} -> {len(notes)} notes, SR {sr:.2f} | "
            f"patterns {rep.get('pattern_entropy', 0):.2f} bits | on attacks {100 * rep.get('on_attack', 0):.0f}% | "
            f"fast jacks {rep.get('fast_jack_pct', 0):.1f}%"
            + (f" (real maps {100 * jack_target:.1f}%)" if jack_target is not None else "") + f" | critic {score:.2f}")
        if score > best_score:
            best_score = score
            import copy as _copy
            best = ComposeResult(notes=notes, star_rating=sr, level=L, sections=_copy.deepcopy(comp.sections), report=rep)
    assert best is not None
    return best


def _calibrate_repeats(comp: "Composer", an: SongAnalysis, cfg: ComposerConfig, L: float, sr: float, notes: List[Note],
                       seed: int, plan_level: float, target_sr: float, jack_target: float):
    """Sets the strength of the jack / anchor penalties so that the share of fast lane repeats matches the data (the
    dataset at this star rating, or the chosen style: a chordjack mapper's 60 % as well as a stream mapper's 2 %).
    The level is re-searched after every change, since jacks change the star rating."""
    rate = fast_jack_share(notes)
    if abs(rate - jack_target) <= 0.02:
        return L, sr, notes

    def search(L0: float):
        lo, hi = max(0.0, L0 - 0.25), min(LEVEL_MAX, L0 + 0.25)
        best = None
        Lc = L0
        for _ in range(7):
            n_ = comp.render(Lc, seed, plan_level)
            s_ = star_rating(n_, an.timing_points, cfg.od)
            if best is None or abs(s_ - target_sr) < abs(best[1] - target_sr):
                best = (Lc, s_, n_)
            if abs(s_ - target_sr) <= cfg.sr_tolerance:
                break
            lo, hi = (Lc, hi) if s_ < target_sr else (lo, Lc)
            Lc = 0.5 * (lo + hi)
        return best

    best = (abs(rate - jack_target) + 2.0 * max(0.0, abs(sr - target_sr) - cfg.sr_tolerance), 1.0, L, sr, notes)
    lo, hi = math.log(0.02), math.log(6.0)   # more penalty = fewer jacks
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        comp.repeat_scale = math.exp(mid)
        Lm, srm, nm = search(L)
        r = fast_jack_share(nm)
        err = abs(r - jack_target) + 2.0 * max(0.0, abs(srm - target_sr) - cfg.sr_tolerance)
        if err < best[0]:
            best = (err, comp.repeat_scale, Lm, srm, nm)
        if abs(r - jack_target) <= 0.02:
            break
        if r > jack_target:
            lo = mid
        else:
            hi = mid
    _, comp.repeat_scale, L, sr, notes = best
    return L, sr, notes
