"""
Exact per-millisecond comparison of a generated map with a reference map.

All metrics work on rendered notes, so they measure the final map, not the network's internals:

* event timing F1 at tolerances 0/1/2/5/10/16/32/64 ms (an event = a distinct start time),
* lane-aware note F1 at the same tolerances,
* timing MAE of matched events, chord-size accuracy, LN start/end errors, density ratio,
* ``per_ms_loss`` - mean of (1 - event F1) over all tolerances: 0.0 means bit-exact timing,
  1.0 means nothing within 64 ms. It is the single number used for checkpoint selection.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .beatmap import Note
from .params import NUM_LANES

TOLERANCES_MS: Tuple[int, ...] = (0, 1, 2, 5, 10, 16, 32, 64)


def match_times(pred: np.ndarray, ref: np.ndarray, tol: float) -> List[Tuple[int, int]]:
    """Maximum one-to-one matching of sorted 1-D times within ``tol`` (greedy is optimal here)."""
    pairs: List[Tuple[int, int]] = []
    i = j = 0
    n_p, n_r = len(pred), len(ref)
    while i < n_p and j < n_r:
        d = pred[i] - ref[j]
        if abs(d) <= tol:
            pairs.append((i, j))
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return pairs


def _f1(tp: int, n_pred: int, n_ref: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else (1.0 if n_ref == 0 else 0.0)
    r = tp / n_ref if n_ref else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def _events(notes: Sequence[Note]) -> Tuple[np.ndarray, Dict[int, List[Note]]]:
    groups: Dict[int, List[Note]] = {}
    for n in notes:
        groups.setdefault(n.start, []).append(n)
    times = np.array(sorted(groups), dtype=np.int64)
    return times, groups


def per_ms_report(pred_notes: Sequence[Note], ref_notes: Sequence[Note], tolerances: Sequence[int] = TOLERANCES_MS) -> Dict[str, float]:
    pred = sorted(pred_notes)
    ref = sorted(ref_notes)
    pt, pg = _events(pred)
    rt, rg = _events(ref)
    rep: Dict[str, float] = {
        "pred_notes": float(len(pred)),
        "ref_notes": float(len(ref)),
        "pred_events": float(len(pt)),
        "ref_events": float(len(rt)),
    }
    losses = []
    for tol in tolerances:
        pairs = match_times(pt, rt, tol)
        p, r, f = _f1(len(pairs), len(pt), len(rt))
        rep[f"event_f1@{tol}"] = f
        rep[f"event_precision@{tol}"] = p
        rep[f"event_recall@{tol}"] = r
        losses.append(1.0 - f)
        tp_notes = 0
        for c in range(NUM_LANES):
            a = np.array([n.start for n in pred if n.col == c], dtype=np.int64)
            b = np.array([n.start for n in ref if n.col == c], dtype=np.int64)
            tp_notes += len(match_times(a, b, tol))
        rep[f"note_f1@{tol}"] = _f1(tp_notes, len(pred), len(ref))[2]
    rep["per_ms_loss"] = float(np.mean(losses))

    pairs = match_times(pt, rt, max(tolerances))
    if pairs:
        diffs = np.array([pt[i] - rt[j] for i, j in pairs], dtype=np.float64)
        rep["timing_mae_ms"] = float(np.abs(diffs).mean())
        rep["timing_bias_ms"] = float(diffs.mean())
        chord_ok = [len(pg[int(pt[i])]) == len(rg[int(rt[j])]) for i, j in pairs]
        rep["chord_size_acc"] = float(np.mean(chord_ok))
        ln_end_err = []
        ln_flag_tp = ln_pred = ln_ref = 0
        for i, j in pairs:
            pn = sorted(pg[int(pt[i])], key=lambda n: n.col)
            rn = sorted(rg[int(rt[j])], key=lambda n: n.col)
            ln_pred += sum(1 for n in pn if n.ln)
            ln_ref += sum(1 for n in rn if n.ln)
            for a, b in zip(pn, rn):  # members by lane order (lane-independent)
                if a.ln and b.ln:
                    ln_flag_tp += 1
                    ln_end_err.append(abs((a.end - a.start) - (b.end - b.start)))
        rep["ln_f1"] = _f1(ln_flag_tp, ln_pred, ln_ref)[2] if (ln_pred or ln_ref) else 1.0
        rep["ln_length_mae_ms"] = float(np.mean(ln_end_err)) if ln_end_err else 0.0
    else:
        rep["timing_mae_ms"] = float("nan")
        rep["timing_bias_ms"] = float("nan")
        rep["chord_size_acc"] = 0.0
        rep["ln_f1"] = 0.0
        rep["ln_length_mae_ms"] = float("nan")

    span_ref = (rt[-1] - rt[0]) / 1000.0 if len(rt) > 1 else 1.0
    rep["density_ratio"] = (len(pred) / max(1.0, span_ref)) / max(1e-6, len(ref) / max(1.0, span_ref)) if ref else float("nan")

    # Where do surplus / missing notes come from? (events matched at 16 ms)
    pairs16 = match_times(pt, rt, 16)
    matched_p = {i for i, _ in pairs16}
    matched_r_times = np.array(sorted(int(rt[j]) for _, j in pairs16), dtype=np.int64)
    dup = iso = 0
    for i in range(len(pt)):
        if i in matched_p:
            continue
        # a false event right next to a reference event that is already covered = hedged duplicate
        if matched_r_times.size and np.min(np.abs(matched_r_times - pt[i])) <= 30:
            dup += 1
        else:
            iso += 1
    extra_chord = missing_chord = 0
    for i, j in pairs16:
        d = len(pg[int(pt[i])]) - len(rg[int(rt[j])])
        extra_chord += max(0, d)
        missing_chord += max(0, -d)
    n_ref = max(1, len(ref))
    rep["surplus_duplicate_events_pct"] = 100.0 * dup / n_ref      # extra events within 30 ms of a covered reference event
    rep["surplus_isolated_events_pct"] = 100.0 * iso / n_ref       # extra events far from any reference event
    rep["surplus_chord_notes_pct"] = 100.0 * extra_chord / n_ref   # chords with more notes than the reference
    rep["missing_chord_notes_pct"] = 100.0 * missing_chord / n_ref
    rep["missed_events_pct"] = 100.0 * (len(rt) - len(pairs16)) / max(1, len(rt))
    rep["exact_match"] = float(pred == ref)
    return rep


def summarize_reports(reports: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not reports:
        return {}
    keys = reports[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in reports], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[k] = float(vals.mean()) if vals.size else float("nan")
    return out


def calibration_samples(pred_notes: Sequence[Note], ref_notes: Sequence[Note], tol: int = 40):
    """Signed start errors of matched events and log2 length ratios of matched long notes (pred vs ref)."""
    pt, pg = _events(sorted(pred_notes))
    rt, rg = _events(sorted(ref_notes))
    starts, ln_ratios = [], []
    for i, j in match_times(pt, rt, tol):
        starts.append(float(pt[i] - rt[j]))
        pn = sorted(pg[int(pt[i])], key=lambda n: n.col)
        rn = sorted(rg[int(rt[j])], key=lambda n: n.col)
        for a, b in zip(pn, rn):
            if a.ln and b.ln and a.end > a.start and b.end > b.start:
                ln_ratios.append(float(np.log2((a.end - a.start) / (b.end - b.start))))
    return starts, ln_ratios
