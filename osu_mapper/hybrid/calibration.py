"""
Post-training calibration of systematic offsets, measured on held-out songs.

* onset_shift_ms  - constant correction of note start times. A non-zero median start error usually comes from
                    the audio decoder (MP3 encoder delay differs between decoders) rather than from the network.
* ln_length_scale - multiplier for predicted long-note lengths, so releases are neither late nor early on average.

Both are stored in the checkpoint and applied by the decoder; the network weights are not changed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint
from .metrics import calibration_samples
from .renderer import DecodeKnobs, HeadOutputs, decode_outputs

MIN_EVENTS = 200
MIN_LNS = 30
MAX_SHIFT_MS = 15


def calibrate(
    items: Iterable[Tuple[HeadOutputs, Sequence[Note], Sequence[RawTimingPoint]]],
    onset_threshold: float,
    ln_threshold: float,
) -> Dict[str, float]:
    starts, ratios = [], []
    for out, ref, tps in items:
        notes = decode_outputs(out, DecodeKnobs(onset_threshold=onset_threshold, ln_threshold=ln_threshold), tps)
        s, r = calibration_samples(notes, ref)
        starts += s
        ratios += r
    res: Dict[str, float] = {"calib_events": len(starts), "calib_long_notes": len(ratios)}
    if starts:
        med = float(np.median(starts))
        res["start_error_median_ms"] = med
        res["onset_shift_ms"] = int(np.clip(-round(med), -MAX_SHIFT_MS, MAX_SHIFT_MS)) if len(starts) >= MIN_EVENTS else 0
    else:
        res["onset_shift_ms"] = 0
    if ratios:
        med_r = float(np.median(ratios))
        res["ln_length_ratio_median"] = float(2.0 ** med_r)
        res["ln_length_scale"] = float(np.clip(2.0 ** (-med_r), 0.5, 2.0)) if len(ratios) >= MIN_LNS else 1.0
    else:
        res["ln_length_scale"] = 1.0
    return res
