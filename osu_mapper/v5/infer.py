"""
One entry point for generating a whole song with any v5 network: the canvas (masked frames restored, the default
since the canvas rewrite) or the older autoregressive decoder - chosen by the checkpoint's config.

    notes = generate_map(model, feats_all, beat_all, timing_points, sr, style=..., candidates=4, score_fn=...)
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..hybrid.beatmap import Note

CANVAS_KEYS = ("style", "style_mask", "steps", "temperature", "cfg_scale", "candidates", "score_fn", "refine_passes",
               "refine_frac", "fixed", "regen", "seed", "device", "log", "overlap", "stats")
AR_KEYS = ("style", "style_mask", "passes", "candidates", "score_fn", "temperature", "top_p", "cfg_scale", "seed",
           "device", "log", "keep_better", "lane_predictor", "power")


def generate_map(model, feats_all: np.ndarray, beat_all: np.ndarray, timing_points, sr: float, **kw) -> List[Note]:
    if getattr(model.cfg, "decoder", "ar") == "canvas":
        from .canvas_gen import generate_canvas
        return generate_canvas(model, feats_all, beat_all, timing_points, sr,
                               **{k: v for k, v in kw.items() if k in CANVAS_KEYS and v is not None})
    from .generate import generate_passes
    return generate_passes(model, feats_all, beat_all, timing_points, sr,
                           **{k: v for k, v in kw.items() if k in AR_KEYS and v is not None})
