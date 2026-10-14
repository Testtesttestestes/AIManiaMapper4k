"""
Inference for tracks of ANY length.

The network is only ever applied to windows of exactly the training crop length. Windows overlap by
50 % and are blended with a Hann weight (time probabilities) or taken from the most central window
(all other heads), so every frame gets at least a quarter-window of left/right context and the
result does not depend on where the song starts or how long it is. The universal algorithm then
renders the whole track in one sequential pass, carrying lane memory and LN occupancy across window
boundaries. Memory is O(track length) with small constants (~0.4 KB per 10 ms frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .audio_features import BEAT_DIM, FeatureConfig, beat_features, coarse_features, compute_stacked_features, estimate_tempo
from .beatmap import ManiaMap, Note, RawTimingPoint, map_to_osu_text
from .model import HybridModelConfig, HybridParamNet
from .params import PARAM_VERSION
from .renderer import HeadOutputs
from .solver import STYLE_DIM, STYLE_KEYS

CHECKPOINT_FORMAT = "hybrid-checkpoint-v1"


def resolve_device(requested: str = "auto") -> torch.device:
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


@dataclass
class SongInputs:
    feats: np.ndarray                 # [F, D] float32 normalized stacked representations
    coarse: np.ndarray                # [F/100, D]
    beat: np.ndarray                  # [F, 13]
    duration_ms: int
    reps: Optional[Dict[str, np.ndarray]] = None

    @property
    def num_frames(self) -> int:
        return int(self.feats.shape[0])


def prepare_song_inputs(
    wave: np.ndarray,
    feature_cfg: FeatureConfig,
    timing_points: Optional[Sequence[RawTimingPoint]] = None,
    keep_reps: bool = False,
    device: str = "cpu",
) -> SongInputs:
    feats = compute_stacked_features(wave, feature_cfg, device=device)
    F = feats.shape[0]
    beat = beat_features(timing_points, 0, F, feature_cfg.frame_ms) if timing_points else np.zeros((F, BEAT_DIM), np.float32)
    duration_ms = int(round(len(wave) * 1000.0 / feature_cfg.sample_rate))
    slices = feature_cfg.slices()
    kept = {k: feats[:, slices[k][0]:slices[k][1]] for k in ("onset", "fine_env") if k in slices} if keep_reps else None
    return SongInputs(feats=feats, coarse=coarse_features(feats), beat=beat, duration_ms=duration_ms, reps=kept)


def style_from_dict(values: Optional[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """User style controls (raw units) -> normalized style vector + known mask."""
    style = np.zeros(STYLE_DIM, dtype=np.float32)
    mask = np.zeros(STYLE_DIM, dtype=np.float32)
    if not values:
        return style, mask
    for key, v in values.items():
        if key not in STYLE_KEYS:
            raise KeyError(f"unknown style key '{key}'. Valid keys: {', '.join(STYLE_KEYS)}")
        i = STYLE_KEYS.index(key)
        v = float(v)
        if key == "sr":
            v = v / 10.0
        elif key == "od":
            v = v / 10.0
        elif key == "nps":
            v = math.log1p(v) / math.log1p(30.0)
        style[i] = v
        mask[i] = 1.0
    return style, mask


class HybridPredictor:
    def __init__(self, model: HybridParamNet, feature_cfg: FeatureConfig, window_frames: int, device: torch.device,
                 decode_threshold: float = 0.5, ln_threshold: float = 0.5):
        self.model = model.to(device).eval()
        self.feature_cfg = feature_cfg
        self.window = int(window_frames)
        self.device = device
        self.decode_threshold = float(decode_threshold)
        self.ln_threshold = float(ln_threshold)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto", use_ema: bool = True) -> "HybridPredictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"'{path}' is not a hybrid checkpoint (format={ckpt.get('format')})")
        if ckpt.get("param_version") != PARAM_VERSION:
            raise ValueError(f"checkpoint parameter version {ckpt.get('param_version')} != code {PARAM_VERSION}")
        cfg = HybridModelConfig.from_dict(ckpt["model_config"])
        model = HybridParamNet(cfg)
        state = ckpt.get("ema_state") if (use_ema and ckpt.get("ema_state")) else ckpt["model_state"]
        model.load_state_dict(state)
        fcfg = FeatureConfig.from_dict(ckpt["feature_config"])
        return cls(model, fcfg, ckpt.get("crop_frames", 2048), resolve_device(device), ckpt.get("decode_threshold", 0.5), ckpt.get("ln_threshold", 0.5))

    @torch.no_grad()
    def predict(
        self,
        inputs: SongInputs,
        star_rating: float,
        style: Optional[np.ndarray] = None,
        style_mask: Optional[np.ndarray] = None,
        hop_frames: Optional[int] = None,
        batch_size: int = 8,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> HeadOutputs:
        cfg = self.model.cfg
        F, W = inputs.num_frames, self.window
        H = cfg.frame_ms
        hop = int(hop_frames or W // 2)
        starts = list(range(-W // 4, max(1, F - W // 4), hop))
        if starts[-1] + W < F:
            starts.append(F - W + W // 4)
        idx = np.arange(W)
        weight = np.sin(np.pi * (idx + 0.5) / W) ** 2 + 1e-3

        time_acc = np.zeros((F, 1 + H), dtype=np.float32)
        w_acc = np.zeros(F, dtype=np.float32)
        best_w = np.zeros(F, dtype=np.float32)
        lane = np.zeros((F, cfg.n_lane_codes), dtype=np.float32)
        chord = np.zeros((F, cfg.max_members), dtype=np.float32)
        stagger = np.zeros((F, cfg.max_members), dtype=np.int16)
        ln_logits = np.zeros((F, cfg.max_members), dtype=np.float32)
        ln_bucket = np.zeros((F, cfg.max_members), dtype=np.int16)
        ln_resid = np.zeros((F, cfg.max_members), dtype=np.float32)

        dev = self.device
        st = np.zeros(STYLE_DIM, np.float32) if style is None else np.asarray(style, np.float32)
        sm = np.zeros(STYLE_DIM, np.float32) if style_mask is None else np.asarray(style_mask, np.float32)
        coarse = torch.from_numpy(inputs.coarse.astype(np.float32)).unsqueeze(0).to(dev)

        def window_slice(arr: np.ndarray, s: int) -> np.ndarray:
            out = np.zeros((W,) + arr.shape[1:], dtype=np.float32)
            a, b = max(0, s), min(arr.shape[0], s + W)
            if b > a:
                out[a - s: b - s] = arr[a:b]
            return out

        for bi in range(0, len(starts), batch_size):
            chunk = starts[bi: bi + batch_size]
            n = len(chunk)
            feats = torch.from_numpy(np.stack([window_slice(inputs.feats, s) for s in chunk])).to(dev)
            beat = torch.from_numpy(np.stack([window_slice(inputs.beat, s) for s in chunk])).to(dev)
            out = self.model(
                feats, beat,
                torch.full((n, 1), float(star_rating), device=dev),
                torch.from_numpy(st).to(dev).expand(n, -1),
                torch.from_numpy(sm).to(dev).expand(n, -1),
                coarse.expand(n, -1, -1),
                None,
            )
            tp = torch.softmax(out["time_logits"].float(), dim=-1).cpu().numpy()
            lg = out["lane_logits"].float().cpu().numpy()
            ch = torch.log_softmax(out["chord_logits"].float(), dim=-1).cpu().numpy()
            sg = out["stagger_logits"].argmax(dim=-1).cpu().numpy()
            ll = out["ln_logits"].float().cpu().numpy()
            lb = out["ln_bucket_logits"].argmax(dim=-1).cpu().numpy()
            lr = out["ln_resid"].float().cpu().numpy()
            for j, s in enumerate(chunk):
                a, b = max(0, s), min(F, s + W)
                if b <= a:
                    continue
                wj = weight[a - s: b - s]
                time_acc[a:b] += tp[j, a - s: b - s] * wj[:, None]
                w_acc[a:b] += wj
                better = wj > best_w[a:b]
                rows = np.flatnonzero(better) + a
                src = rows - s
                lane[rows] = lg[j, src]
                chord[rows] = ch[j, src]
                stagger[rows] = sg[j, src]
                ln_logits[rows] = ll[j, src]
                ln_bucket[rows] = lb[j, src]
                ln_resid[rows] = lr[j, src]
                best_w[a:b] = np.maximum(best_w[a:b], wj)
            if progress is not None:
                progress(min(len(starts), bi + n), len(starts))

        time_probs = time_acc / np.maximum(w_acc, 1e-6)[:, None]
        return HeadOutputs(
            time_probs=time_probs,
            lane_logits=lane,
            stagger=stagger.astype(np.int64),
            ln_logits=ln_logits,
            ln_bucket=ln_bucket.astype(np.int64),
            ln_resid=ln_resid,
            frame_ms=H,
            chord_logprobs=chord,
        )


def star_rating_of(notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], od: float = 8.0) -> float:
    from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating

    if not notes:
        return 0.0
    tps = list(timing_points) or [RawTimingPoint(0.0, 500.0)]
    text = map_to_osu_text(ManiaMap(notes=list(notes), timing_points=tps, overall_difficulty=od))
    hits = [(n.col, n.start, n.end, n.ln) for n in notes]
    return float(calculate_osu_mania_star_rating(osu_text=text, hit_objects=hits, overall_difficulty=od, keys=4))


def estimated_timing_points(reps: Dict[str, np.ndarray], frame_ms: int) -> List[RawTimingPoint]:
    bpm, offset = estimate_tempo(reps, frame_ms)
    return [RawTimingPoint(time=float(offset), beat_len=60000.0 / bpm, meter=4, volume=70, uninherited=True)]
