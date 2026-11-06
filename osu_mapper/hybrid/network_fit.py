"""
Network-first maps at a requested star rating.

The network's own map (its event probabilities decoded by the universal algorithm) plays best: it only puts
notes where it hears something. Its weaknesses are (1) the star rating: one threshold gives whatever density the
network finds typical, often 0.5-3 stars away from the request, and (2) on some audio its milliseconds are shifted
against the template's grid (a constant per song, up to ~10 ms).

Measured on maps the v2 network never saw: at the same number of notes its maps are 0.3-1.5 stars harder than the
human map, and the reason is its lanes - 16-27 % fast jacks against 3-13 % in the human maps of the same songs.
Re-assigning lanes against jacks brings it to the human rating (EXHAUST 4.97 -> 4.58 = target); long notes and
chord sizes change it by 0.1 at most.

``fit_network_map`` keeps the network's moments, chords and long notes and
(1) subtracts the per-song lag of its milliseconds against the red lines;
(2) re-assigns only as many lanes as needed for the share of fast jacks real maps of this star rating have (or the
    chosen style: a chordjack mapper's 60 % stay), with the ergonomics beam search;
(3) moves the event threshold until the rendered map has the requested star rating, never below ``min_threshold``
    (below it the network does not hear a note: such notes would be on silence).
If the request needs more notes than the network hears, the map stays at the closest reachable rating and the
result says so, instead of filling silence.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .beatmap import ManiaMap, Note, RawTimingPoint, map_to_osu_text
from .renderer import DecodeKnobs, HeadOutputs, decode_outputs


@dataclass
class NetworkFit:
    notes: List[Note]
    star_rating: float
    threshold: float
    lag_ms: int
    reached: bool
    lane_strength: float = 0.0   # weight of the pattern-model lane prior (or ergonomics strength) used for the jack share
    jack_share: float = 0.0
    jack_target: Optional[float] = None          # False: the requested rating needs notes the network does not hear (or fewer than it is sure of)


def star_rating(notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], od: float = 8.0) -> float:
    from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating

    if not notes:
        return 0.0
    tps = list(timing_points) or [RawTimingPoint(0.0, 500.0)]
    text = map_to_osu_text(ManiaMap(notes=list(notes), timing_points=tps, overall_difficulty=od))
    return float(calculate_osu_mania_star_rating(osu_text=text, hit_objects=[(n.col, n.start, n.end, n.ln) for n in notes],
                                                 overall_difficulty=od, keys=4))


def network_lag(out: HeadOutputs, timing_points: Sequence[RawTimingPoint], min_events: int = 30) -> int:
    """Median offset (ms) of the network's confident events against the beat / half-beat ticks of the red lines."""
    reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
    if not reds:
        return 0
    pe = 1.0 - out.time_probs[:, 0]
    fr = np.flatnonzero((pe >= 0.5) & (pe >= np.roll(pe, 1)) & (pe >= np.roll(pe, -1)))
    if fr.size < min_events:
        return 0
    ms = fr * out.frame_ms + np.argmax(out.time_probs[fr, 1:], axis=1)
    ticks = []
    end = float(ms.max()) + 1000.0
    for i, r in enumerate(reds):
        stop = reds[i + 1].time if i + 1 < len(reds) else end
        start = r.time - np.floor(r.time / (r.beat_len / 2)) * (r.beat_len / 2) if i == 0 else r.time
        ticks.append(np.arange(start, stop, r.beat_len / 2))
    t = np.sort(np.round(np.concatenate(ticks)).astype(np.int64))
    j = np.clip(np.searchsorted(t, ms), 1, t.size - 1)
    a, b = ms - t[j], ms - t[j - 1]
    d = np.where(np.abs(a) < np.abs(b), a, b)
    d = d[np.abs(d) <= 20]
    return int(round(float(np.median(d)))) if d.size >= min_events else 0


def apply_repeats(notes: Sequence[Note], repeats, min_similarity: float = 0.9, gap_fn=None) -> List[Note]:
    """Repeated phrases get the rhythm and chord sizes of their original, as mappers do (measured on human maps: the
    rhythm of a repeat matches its original at F1 0.89, the lanes only 40 %: copies alternate with mirrors). The network
    alone kept the rhythm at 0.27 on some maps (Circles of Death chy's Insane). ``repeats``: (start, end, source start,
    similarity) from the composer's analysis (find_repeats); phrases are processed in time order, every second one is
    mirrored. Long notes are cut at the phrase end."""
    from .renderer import resolve_lane_conflicts

    out = sorted(notes)
    k = 0
    for a, b, src, sim in sorted(repeats):
        if sim < min_similarity or b <= a:
            continue
        span = b - a
        orig = [n for n in out if src <= n.start < src + span]
        if len(orig) < 4:
            continue
        mirror = k % 2 == 1
        k += 1
        shift = a - src
        copied = []
        for n in orig:
            col = 3 - n.col if mirror else n.col
            end = min(n.end + shift, b - 1) if n.ln else n.start + shift
            copied.append(Note(n.start + shift, col, end, bool(n.ln and end - (n.start + shift) >= 30)))
        out = sorted([n for n in out if not (a <= n.start < b)] + copied)
    return resolve_lane_conflicts(out, 40, 30, gap_fn)


def with_ladder(out: HeadOutputs, target_sr: float, mix: float = 0.5) -> HeadOutputs:
    """Event probabilities weighted by the difficulty ladder (a model trained with the importance head): the moment's
    probability becomes p_event^(1-mix) * P(mapped at target_sr)^mix. The ladder learnt from songs with several
    difficulties which hits every map keeps (the strong ones) and which only hard maps add, so at a low rating the
    threshold drops the fills first instead of whatever the network is least sure of. Unchanged without the head."""
    pl = out.ladder_probs(target_sr)
    if pl is None or mix <= 0:
        return out
    tp = out.time_probs.astype(np.float64)
    pe = np.clip(1.0 - tp[:, 0], 1e-9, 1.0)
    pn = pe ** (1.0 - mix) * np.clip(pl, 1e-9, 1.0) ** mix
    new = np.empty_like(tp)
    new[:, 0] = 1.0 - pn
    new[:, 1:] = tp[:, 1:] * (pn / pe)[:, None]
    return dataclasses.replace(out, time_probs=new.astype(np.float32))


def fast_jack_share(notes: Sequence[Note], jack_ms: float = 160.0) -> float:
    from .ergonomics import ergonomics_report
    return float(ergonomics_report(notes, jack_ms=jack_ms).get("fast_jack_pct", 0.0)) / 100.0


def fit_network_map(
    out: HeadOutputs,
    knobs: DecodeKnobs,
    timing_points: Sequence[RawTimingPoint],
    target_sr: float,
    od: float = 8.0,
    min_threshold: float = 0.15,
    max_threshold: float = 0.95,
    tolerance: float = 0.05,
    iterations: int = 14,
    fix_lag: bool = True,
    jack_target: Optional[float] = None,
    log=None,
    lane_prior=None,
    hand_run_continue=None,
    repeats=None,
    ladder_mix: float = 0.5,
    hedge_fallback: Sequence[tuple] = ((60, 1.2), (60, 1.0), (40, 1.0)),
    two_sided=None,
    two_sided_temperature: float = 1.0,
    two_sided_net_weight: float = 0.0,
) -> NetworkFit:
    """The network's map at the requested star rating (see module docstring). ``jack_target``: share of fast jacks
    (< 160 ms) to aim for, normally ``PatternModel.fast_repeat_share(target_sr)``; None keeps the network's lanes.

    ``hedge_fallback``: (hedge_ms, hedge_sum) steps tried, in order, only when the rating is not reached at the lowest
    threshold. The hedge merges two close, weak events into one note; at the lowest threshold it also merges real
    neighbours (15 unseen maps: rating reached on 11 -> 15, mean error 0.24 -> 0.03; on the 4 maps it changed, notes
    match the human map -0.02 on average (all 15: 0.703 -> 0.698), notes <= 30 ms from another up to 2.2 %). The other maps are not touched.

    ``two_sided``: a PatternModel with two-sided lane statistics - after the threshold fit the lanes of every chord are
    re-chosen knowing the chords before and after (lanes.two_sided_lanes, sampled at ``two_sided_temperature``), then
    the threshold is corrected for the new lanes; the fast-jack calibration is not needed then (the sampled lanes
    follow the jack share of real maps by themselves)."""
    lag = network_lag(out, timing_points) if fix_lag else 0
    out = with_ladder(out, target_sr, ladder_mix)
    fit = _fit(out, knobs, timing_points, target_sr, od, min_threshold, max_threshold, tolerance, iterations, lag,
               jack_target, log, lane_prior, hand_run_continue, repeats, two_sided, two_sided_temperature, two_sided_net_weight)
    for hm, hs in hedge_fallback:
        if fit.reached or fit.star_rating > target_sr or not knobs.hedge_ms:
            break
        k2 = dataclasses.replace(knobs, hedge_ms=int(min(hm, knobs.hedge_ms)), hedge_sum=float(min(hs, knobs.hedge_sum)))
        if (k2.hedge_ms, k2.hedge_sum) == (knobs.hedge_ms, knobs.hedge_sum):
            continue
        f2 = _fit(out, k2, timing_points, target_sr, od, min_threshold, max_threshold, tolerance, iterations, lag,
                  jack_target, None, lane_prior, hand_run_continue, repeats, two_sided, two_sided_temperature,
                  two_sided_net_weight)
        if abs(f2.star_rating - target_sr) < abs(fit.star_rating - target_sr):
            if log:
                log(f"    rating {fit.star_rating:.2f} not reachable at the lowest threshold: close weak events kept apart "
                    f"(hedge {k2.hedge_ms} ms / {k2.hedge_sum:g}) -> {f2.star_rating:.2f}")
            fit = f2
    return fit


def _fit(out, knobs, timing_points, target_sr, od, min_threshold, max_threshold, tolerance, iterations, lag,
         jack_target, log, lane_prior, hand_run_continue, repeats, two_sided=None, two_temp=1.0, two_w=0.0) -> NetworkFit:
    from .ergonomics import improve_ergonomics
    base = dataclasses.replace(knobs, onset_shift_ms=int(knobs.onset_shift_ms) - lag)
    if hand_run_continue is not None:
        base = dataclasses.replace(base, hand_run_continue=hand_run_continue)
    strength = 0.0

    from .renderer import release_gap_fn
    gap_fn = release_gap_fn(timing_points, base.ln_release_gap_ms, base.ln_release_gap_beats)

    def decode(th: float, st: float, two: bool = False) -> List[Note]:
        notes = _decode(th, st)
        if two:
            from .lanes import two_sided_lanes
            notes = two_sided_lanes(notes, two_sided, target_sr, out, int(base.onset_shift_ms), net_weight=two_w,
                                    gap_fn=gap_fn, temperature=two_temp, seed=0)
        return apply_repeats(notes, repeats, gap_fn=gap_fn) if repeats else notes

    def _decode(th: float, st: float) -> List[Note]:
        if lane_prior is not None:
            # the network chooses lanes together with what real maps do after these chords (pattern model); st is the
            # weight of that prior here, not an after-the-fact re-assignment
            return decode_outputs(out, dataclasses.replace(base, onset_threshold=th, lane_prior=lane_prior if st > 0 else None,
                                                           lane_prior_weight=st), timing_points)
        notes = decode_outputs(out, dataclasses.replace(base, onset_threshold=th), timing_points)
        if st > 0:
            notes = improve_ergonomics(notes, timing_points, st, release_gap_ms=base.ln_release_gap_ms,
                                       release_gap_beats=base.ln_release_gap_beats)
        return notes

    def calibrate_lanes(th: float, current: float) -> float:
        """The least re-assignment that gives the fast-jack share of real maps at threshold ``th``."""
        share = fast_jack_share(decode(th, current))
        if jack_target is None or abs(share - jack_target) <= 0.02 or (share < jack_target and current == 0.0):
            return current
        lo_s, hi_s, best = 0.0, (0.6 if lane_prior is not None else 1.0), (abs(share - jack_target), current)
        for _ in range(8):
            mid = 0.5 * (lo_s + hi_s)
            sh = fast_jack_share(decode(th, mid))
            if abs(sh - jack_target) < best[0] - 1e-9:
                best = (abs(sh - jack_target), mid)
            if sh > jack_target:
                lo_s = mid
            else:
                hi_s = mid
        return best[1]

    def fit_threshold():
        cache = {}

        def run(th: float):
            th = round(float(th), 4)
            if th not in cache:
                notes = decode(th, strength)
                cache[th] = (star_rating(notes, timing_points, od), notes)
            return cache[th]

        lo, hi = float(min_threshold), float(max_threshold)
        sr_lo, _ = run(lo)          # the most notes the network hears
        sr_hi, _ = run(hi)          # only what it is sure about
        if target_sr >= sr_lo:
            return (lo,) + run(lo)
        if target_sr <= sr_hi:
            return (hi,) + run(hi)
        for _ in range(iterations):  # star rating falls as the threshold rises
            mid = 0.5 * (lo + hi)
            sr, _ = run(mid)
            if abs(sr - target_sr) <= tolerance:
                break
            if sr > target_sr:
                lo = mid
            else:
                hi = mid
        th, (sr, notes) = min(cache.items(), key=lambda kv: abs(kv[1][0] - target_sr))
        return th, sr, notes

    # (2) lanes, (3) threshold; the threshold changes the jack share a little, so lanes are checked once more at the
    # fitted threshold and, if they moved, the threshold is fitted again
    th0 = float(np.clip(knobs.onset_threshold, min_threshold, max_threshold))
    first = fast_jack_share(decode(th0, 0.0))
    use_two = two_sided is not None and getattr(two_sided, "has_two_sided", False)
    if use_two:
        strength = 1.0 if lane_prior is not None else 0.0   # left-to-right lanes are only the starting point
        th, sr, notes = fit_threshold()
        th, sr, notes = _two_sided_threshold(lambda x: decode(x, strength, True), th, target_sr, timing_points, od,
                                             min_threshold, max_threshold, tolerance)
        if log:
            log(f"    lanes chosen from both sides (chords before and after): fast jacks {100 * fast_jack_share(notes):.0f}%"
                + (f" (real maps {100 * jack_target:.0f}%)" if jack_target is not None else ""))
        return NetworkFit(notes, sr, th, lag, abs(sr - target_sr) <= tolerance, strength, fast_jack_share(notes), jack_target)
    strength = calibrate_lanes(th0, 0.0)
    th, sr, notes = fit_threshold()
    again = calibrate_lanes(th, strength)
    if again != strength:
        strength = again
        th, sr, notes = fit_threshold()
    if log and jack_target is not None:
        log(f"    lanes: fast jacks {100 * first:.0f}% -> {100 * fast_jack_share(notes):.0f}% (real maps {100 * jack_target:.0f}%), "
            f"re-assignment strength {strength:.2f}")
    return NetworkFit(notes, sr, th, lag, abs(sr - target_sr) <= tolerance, strength, fast_jack_share(notes), jack_target)


def _two_sided_threshold(decode_two, th, target_sr, timing_points, od, min_threshold, max_threshold, tolerance,
                         window: float = 0.15, iterations: int = 6):
    """Lanes re-chosen from both sides change the rating a little (jacks): the threshold found with the left-to-right
    lanes is corrected by a short bisection around it."""
    cache = {}

    def run(x):
        x = round(float(np.clip(x, min_threshold, max_threshold)), 4)
        if x not in cache:
            n = decode_two(x)
            cache[x] = (star_rating(n, timing_points, od), n)
        return cache[x]

    sr, _ = run(th)
    if abs(sr - target_sr) > tolerance:
        lo, hi = (th, th + window) if sr > target_sr else (th - window, th)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            s_, _ = run(mid)
            if abs(s_ - target_sr) <= tolerance:
                break
            if s_ > target_sr:
                lo = mid
            else:
                hi = mid
    x, (sr, notes) = min(cache.items(), key=lambda kv: abs(kv[1][0] - target_sr))
    return x, sr, notes
