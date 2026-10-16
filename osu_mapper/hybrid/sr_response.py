"""
Does the network obey the requested star rating?

Every map is generated at its own star rating and at shifted requests (e.g. -1, 0, +1 star); the star rating
of the result is measured with rosu-pp. Reported:

* ``slope``     - pooled within-map regression of achieved SR on requested SR (1.0 = every requested star
                  becomes one real star, 0 = the request is ignored),
* ``monotonic`` - share of maps whose achieved SR strictly increases with the request,
* ``sr@d``      - mean achieved SR for request = reference SR + d, ``ref`` - mean reference SR,
* ``f1``        - event F1 @16 ms against the reference at d = 0.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint
from .inference import HybridPredictor, SongInputs, star_rating_of
from .metrics import per_ms_report
from .renderer import DecodeKnobs, decode_outputs

Item = Tuple[SongInputs, List[Note], List[RawTimingPoint], float]


def sr_response(
    predictor: HybridPredictor,
    items: Sequence[Item],
    deltas: Sequence[float] = (-1.0, 0.0, 1.0),
    guidance: float = 1.0,
    threshold: Optional[float] = None,
    min_sr: float = 0.5,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, float]:
    deltas = sorted(float(d) for d in deltas)
    achieved: List[List[float]] = []
    refs, f1s = [], []
    for n, (inputs, ref_notes, tps, ref_sr) in enumerate(items, 1):
        row = []
        for d in deltas:
            req = max(min_sr, ref_sr + d)
            out = predictor.predict(inputs, req, sr_guidance=guidance)
            th = threshold if threshold is not None else predictor.threshold_for(req)
            notes = decode_outputs(out, DecodeKnobs(onset_threshold=th, ln_threshold=predictor.ln_threshold), tps)
            row.append(star_rating_of(notes, tps))
            if d == 0.0:
                f1s.append(per_ms_report(notes, ref_notes)["event_f1@16"])
        achieved.append(row)
        refs.append(ref_sr)
        if progress is not None:
            progress(n, len(items))
    if not achieved:
        return {}
    A = np.array(achieved, dtype=np.float64)                       # [maps, deltas]
    R = np.array([[max(min_sr, r + d) for d in deltas] for r in refs])
    Ac, Rc = A - A.mean(axis=1, keepdims=True), R - R.mean(axis=1, keepdims=True)
    res: Dict[str, float] = {
        "maps": float(len(achieved)),
        "slope": float((Ac * Rc).sum() / max(1e-9, (Rc ** 2).sum())),
        "monotonic": float(np.mean(np.all(np.diff(A, axis=1) > 0, axis=1))) if len(deltas) > 1 else float("nan"),
        "ref": float(np.mean(refs)),
    }
    for j, d in enumerate(deltas):
        res[f"sr@{d:+g}"] = float(A[:, j].mean())
    if f1s:
        res["f1"] = float(np.mean(f1s))
    return res


def format_response(res: Dict[str, float], label: str = "") -> str:
    if not res:
        return f"{label} no maps"
    cols = " ".join(f"{k}={res[k]:.2f}" for k in res if k.startswith("sr@"))
    return (f"{label}slope={res['slope']:.2f} mono={100 * res['monotonic']:.0f}% {cols} ref={res['ref']:.2f}"
            + (f" f1={res['f1']:.3f}" if "f1" in res else "") + f" n={int(res['maps'])}")
