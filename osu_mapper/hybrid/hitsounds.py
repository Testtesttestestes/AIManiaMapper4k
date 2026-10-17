"""
Automatic hitsounding: volume and sound of every key press, from the audio.

Times and lanes are not touched; every hit object gets the osu! hitsound fields
``hitSound`` (whistle 2 / finish 4 / clap 8 on top of the always-played normal sound) and
``normalSet:additionSet:index:volume`` (sets: 1 normal, 2 soft, 3 drum).

Rules (what mappers usually do, measured on the per-song normalized audio features):

* volume  - follows the smoothed loudness of the song (quiet verse quiet, chorus loud), mapped to the
            percentile range ``volume_range``; strong attacks get a small accent;
* set     - per section (~4 s): percussive sections -> drum, quiet sections -> soft, otherwise normal;
* finish  - crash-like moments: loud broadband attack on a strong beat or a section start (rare);
* clap    - snare-like mid-band attacks on beats (backbeat);
* whistle - bright high-band accents, sparse.

In chords only one note carries the addition, so sounds do not stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint
from .thinning import metric_strength

WHISTLE, FINISH, CLAP = 2, 4, 8
SET_NORMAL, SET_SOFT, SET_DRUM = 1, 2, 3
SET_NAMES = {SET_NORMAL: "normal", SET_SOFT: "soft", SET_DRUM: "drum"}


@dataclass
class HitSound:
    hit_sound: int = 0
    normal_set: int = 0
    addition_set: int = 0
    volume: int = 0


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1 or x.size == 0:
        return x
    k = np.ones(frames, dtype=np.float64) / frames
    pad = frames // 2
    return np.convolve(np.pad(x.astype(np.float64), (pad, frames - 1 - pad), mode="edge"), k, mode="valid")


def _peak(x: np.ndarray, f: int) -> float:
    return float(x[max(0, f - 1): f + 3].max()) if 0 <= f < x.size else 0.0


def compute_hitsounds(
    notes: Sequence[Note],
    feats: np.ndarray,
    slices: Dict[str, Tuple[int, int]],
    timing_points: Optional[Sequence[RawTimingPoint]] = None,
    frame_ms: int = 10,
    volume_range: Tuple[int, int] = (40, 90),
    additions: bool = True,
    sample_set: str = "auto",
) -> Dict[Tuple[int, int], HitSound]:
    """(start, lane) -> HitSound for every note."""
    if not notes:
        return {}
    F = feats.shape[0]
    reds = sorted((tp for tp in (timing_points or []) if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)

    def col(name: str, a: int, b: Optional[int] = None) -> np.ndarray:
        if name not in slices:
            return np.zeros(F, np.float32)
        s0, s1 = slices[name]
        block = np.asarray(feats[:, s0 + a: s0 + (b if b is not None else a + 1)], dtype=np.float32)
        return block.mean(axis=1)

    loud = col("spectral", 0) if "spectral" in slices else col("mel_mid", 0, 96)
    low, mid, high = col("onset", 0, 2), col("onset", 2, 5), col("onset", 5, 8)
    total = col("onset", 8)
    hp = slices.get("hpss")
    perc = np.zeros(F, np.float32)
    if hp:
        n_g = (hp[1] - hp[0] - 2) // 2
        perc = col("hpss", 2 * n_g) - col("hpss", 2 * n_g + 1)  # log percussive - log harmonic energy

    loud_s = _smooth(loud, 50)            # 0.5 s
    sect_loud = _smooth(loud, 400)        # 4 s sections
    sect_perc = _smooth(perc, 400)
    long_prev = _smooth(loud, 200)

    groups: Dict[int, List[Note]] = {}
    for n in sorted(notes):
        groups.setdefault(n.start, []).append(n)
    times = sorted(groups)
    frames = np.clip(np.array(times) // frame_ms, 0, F - 1)
    ranks = np.argsort(np.argsort(loud_s[frames])) / max(1, len(frames) - 1)
    vmin, vmax = volume_range
    sp_hi, sp_lo = np.percentile(sect_perc, 75), np.percentile(sect_perc, 25)
    sl_lo = np.percentile(sect_loud, 25)

    out: Dict[Tuple[int, int], HitSound] = {}
    last_finish = last_whistle = -10 ** 9
    for t, f, r in zip(times, frames, ranks):
        strength = metric_strength(t, reds) if reds else 2.0
        a_low, a_mid, a_high, a_tot = _peak(low, f), _peak(mid, f), _peak(high, f), _peak(total, f)
        vol = vmin + (vmax - vmin) * float(r) + (5 if a_tot > 1.5 else 0)
        vol = int(np.clip(round(vol / 5) * 5, 5, 100))
        if sample_set == "auto":
            s = SET_DRUM if sect_perc[f] > sp_hi else SET_SOFT if (sect_loud[f] < sl_lo and sect_perc[f] < sp_lo) else SET_NORMAL
        else:
            s = {"normal": SET_NORMAL, "soft": SET_SOFT, "drum": SET_DRUM}[sample_set]
        add = 0
        if additions:
            section_start = loud_s[f] - long_prev[max(0, f - 150)] > 1.0
            if ((a_low > 1.2 and a_high > 1.2 and strength >= 3.0) or (section_start and a_tot > 1.0)) and t - last_finish > 2000:
                add, last_finish = FINISH, t
            elif a_mid > 1.0 and a_mid >= a_low and strength >= 2.0:
                add = CLAP
            elif a_high > 1.3 and a_high > a_mid and t - last_whistle > 250:
                add, last_whistle = WHISTLE, t
        members = sorted(groups[t], key=lambda n: n.col)
        for i, n in enumerate(members):
            out[(n.start, n.col)] = HitSound(add if i == 0 else 0, s, s if (add and i == 0) else 0, vol)
    return out


def apply_hitsounds_to_text(text: str, hs: Dict[Tuple[int, int], HitSound], keys: int = 4) -> str:
    """Rewrites the [HitObjects] section of an .osu text with the given hitsounds."""
    head, sep, body = text.partition("[HitObjects]")
    if not sep:
        return text
    lines_out = []
    rest_started = False
    tail: List[str] = []
    for line in body.splitlines():
        s = line.strip()
        if rest_started or (s.startswith("[") and s.endswith("]")):
            rest_started = True
            tail.append(line)
            continue
        parts = s.split(",")
        if len(parts) < 5:
            if s:
                lines_out.append(line)
            continue
        x, t, typ = int(float(parts[0])), int(float(parts[2])), int(parts[3])
        lane = min(keys - 1, max(0, int(x * keys / 512)))
        h = hs.get((t, lane))
        if h is None:
            lines_out.append(s)
            continue
        sample = f"{h.normal_set}:{h.addition_set}:0:{h.volume}:"
        if typ & 128:
            end = parts[5].split(":")[0] if len(parts) > 5 else str(t)
            lines_out.append(f"{parts[0]},{parts[1]},{t},{typ},{h.hit_sound},{end}:{sample}")
        else:
            lines_out.append(f"{parts[0]},{parts[1]},{t},{typ},{h.hit_sound},{sample}")
    return head + sep + "\n" + "\n".join(lines_out) + "\n" + ("\n".join(tail) + "\n" if tail else "")


def hitsound_summary(hs: Dict[Tuple[int, int], HitSound]) -> str:
    if not hs:
        return "no notes"
    vals = list(hs.values())
    n = len(vals)
    cnt = lambda bit: sum(1 for v in vals if v.hit_sound & bit)  # noqa: E731
    sets = {k: sum(1 for v in vals if v.normal_set == k) for k in SET_NAMES}
    vols = [v.volume for v in vals]
    return (f"clap {cnt(CLAP)}, whistle {cnt(WHISTLE)}, finish {cnt(FINISH)} | sets "
            + " ".join(f"{SET_NAMES[k]} {100 * c / n:.0f}%" for k, c in sets.items())
            + f" | volume {min(vols)}..{max(vols)} (median {int(np.median(vols))})")
