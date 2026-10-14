"""
Hybrid osu!mania 4K mapper: neural network -> parameters -> deterministic algorithm -> map.

The pipeline is split into two parts that are trained/verified independently:

1. A universal deterministic algorithm (``renderer``) that turns a dense, time-indexed
   parameter track (:class:`~osu_mapper.hybrid.params.ParamTrack`) into hit objects in a
   single O(N) pass, plus its exact inverse (``solver``) that turns ANY 4K map into the
   canonical parameter track. ``render(solve(map)) == map`` holds bit-exactly (verified by
   ``hybrid_solvability_audit.py`` on the whole dataset and by fuzz tests).

2. A non-autoregressive network (``model``) that reads the whole audio through several
   complementary representations ("tokenizers") and predicts the parameter track for every
   10 ms frame in one forward pass. Training targets are produced by the solver, so the
   loss is dense, per-frame and millisecond-exact; map-level per-ms metrics (``metrics``)
   are computed on the rendered map during validation.

Long tracks are handled by window-local inference (fixed window length identical to
training, no absolute positions, length-agnostic global context) and by the renderer,
which carries its state (lane memory, long-note occupancy) across window boundaries.
"""

from .beatmap import ManiaMap, Note, RawTimingPoint, parse_osu_mania, map_to_osu_text
from .params import (
    FRAME_MS,
    LANE_OPS,
    N_LANE_CODES,
    N_LN_BUCKETS,
    PARAM_VERSION,
    ParamTrack,
    apply_lane_code,
    canonical_lane_code,
)
from .renderer import render_track
from .solver import solve_map, verify_roundtrip

__all__ = [
    "ManiaMap",
    "Note",
    "RawTimingPoint",
    "parse_osu_mania",
    "map_to_osu_text",
    "FRAME_MS",
    "LANE_OPS",
    "N_LANE_CODES",
    "N_LN_BUCKETS",
    "PARAM_VERSION",
    "ParamTrack",
    "apply_lane_code",
    "canonical_lane_code",
    "render_track",
    "solve_map",
    "verify_roundtrip",
]
