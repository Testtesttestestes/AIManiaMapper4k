"""
Training losses of the hybrid network. Every term is a per-element mean, so all of them live on
the same O(1) scale and the total is comparable between runs (unlike the old Model 3 loss, where a
millisecond regression term dominated the sum and made "loss = 11" meaningless).

time     per-frame cross-entropy over {no event, event at ms 0..9} with a Gaussian (sigma_ms)
         spread across neighbouring milliseconds/frames: a 1 ms miss costs almost nothing,
         a 30 ms miss costs as much as a miss. This is the per-millisecond loss on note timing.
lane     cross-entropy of the canonical lane code on event frames
chord    cross-entropy of the chord size (1..4 notes) on event frames
stagger  cross-entropy of per-member ms offsets inside chords (only chords >= 2 notes)
ln_flag  BCE tap-vs-long-note per chord member
ln_len   log-bucket cross-entropy + in-bucket L1 of the LN length (LN members only)
style    smooth-L1 of the auto-style head (lets inference pick a style when the user gives none)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .params import LN_BUCKETS_PER_OCTAVE, LN_MAX_MS, MAX_MEMBERS, N_LN_BUCKETS


@dataclass
class LossWeights:
    time: float = 1.0
    lane: float = 0.5
    stagger: float = 0.1
    chord: float = 0.5
    ln_flag: float = 0.5
    ln_len: float = 0.25
    style: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def soft_time_targets(time_class: torch.Tensor, frame_ms: int, sigma_ms: float) -> torch.Tensor:
    """[B, T] time classes -> [B, T, 1 + H] soft targets with a Gaussian spread in milliseconds."""
    B, T = time_class.shape
    H = int(frame_ms)
    ev = time_class > 0
    if sigma_ms <= 0:
        return F.one_hot(time_class.clamp(min=0), 1 + H).float()
    impulses = torch.zeros(B, T * H, device=time_class.device)
    pos = (torch.arange(T, device=time_class.device) * H).view(1, T) + (time_class - 1).clamp(min=0)
    impulses.scatter_(1, pos.clamp(0, T * H - 1), ev.float())  # positions of different frames never collide
    radius = int(math.ceil(3.0 * sigma_ms))
    xs = torch.arange(-radius, radius + 1, device=time_class.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (xs / sigma_ms) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    mass = F.conv1d(impulses.unsqueeze(1), kernel, padding=radius).view(B, T, H)
    ev_mass = mass.sum(dim=-1, keepdim=True)
    none = (1.0 - ev_mass).clamp(min=0.0)
    target = torch.cat([none, mass], dim=-1)
    return target / target.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def ln_targets(ln_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    L = ln_len.float().clamp(1.0, float(LN_MAX_MS) - 1e-3)
    pos = torch.log2(L) * LN_BUCKETS_PER_OCTAVE
    bucket = pos.floor().clamp(0, N_LN_BUCKETS - 1)
    return bucket.long(), (pos - bucket).clamp(0.0, 1.0)


def compute_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: LossWeights,
    frame_ms: int,
    sigma_ms: float = 1.0,
    event_weight: float = 4.0,
    label_smoothing: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    tc = batch["time_class"]
    valid = batch["valid"].float()
    B, T = tc.shape
    logs: Dict[str, float] = {}

    # --- time (per-ms) -------------------------------------------------------------------------
    target = soft_time_targets(tc, frame_ms, sigma_ms)
    logp = F.log_softmax(out["time_logits"].float(), dim=-1)
    ce = -(target * logp).sum(dim=-1)
    w = (1.0 + (event_weight - 1.0) * (1.0 - target[..., 0])) * valid
    loss_time = (ce * w).sum() / w.sum().clamp(min=1.0)
    total = weights.time * loss_time
    logs["time"] = float(loss_time.detach())

    ev = (tc > 0) & (valid > 0)
    n_ev = int(ev.sum())
    logs["events"] = float(n_ev)
    zero = out["time_logits"].sum() * 0.0
    if n_ev > 0:
        # --- lane code -----------------------------------------------------------------------
        loss_lane = F.cross_entropy(out["lane_logits"][ev].float(), batch["lane_code"][ev], label_smoothing=label_smoothing)
        # --- members ---------------------------------------------------------------------------
        masks = batch["event_mask"]
        bits = ((masks.unsqueeze(-1) >> torch.arange(4, device=masks.device)) & 1)
        chord = bits.sum(dim=-1)
        member = (torch.arange(MAX_MEMBERS, device=masks.device).view(1, 1, -1) < chord.unsqueeze(-1)) & ev.unsqueeze(-1)
        loss_chord = F.cross_entropy(out["chord_logits"][ev].float(), (chord[ev] - 1).clamp(0, MAX_MEMBERS - 1))
        chord_member = member & (chord.unsqueeze(-1) >= 2)
        if chord_member.any():
            loss_stagger = F.cross_entropy(out["stagger_logits"][chord_member].float(), batch["stagger"][chord_member].clamp(0, frame_ms - 1))
        else:
            loss_stagger = zero
        loss_lnf = F.binary_cross_entropy_with_logits(out["ln_logits"][member].float(), batch["ln_flag"][member].float())
        ln_member = member & (batch["ln_flag"] > 0.5)
        if ln_member.any():
            b_t, r_t = ln_targets(batch["ln_len"][ln_member])
            loss_lnl = F.cross_entropy(out["ln_bucket_logits"][ln_member].float(), b_t) + F.l1_loss(out["ln_resid"][ln_member].float(), r_t)
        else:
            loss_lnl = zero
    else:
        loss_lane = loss_chord = loss_stagger = loss_lnf = loss_lnl = zero

    total = total + weights.lane * loss_lane + weights.chord * loss_chord + weights.stagger * loss_stagger + weights.ln_flag * loss_lnf + weights.ln_len * loss_lnl
    logs["lane"] = float(loss_lane.detach())
    logs["chord"] = float(loss_chord.detach())
    logs["stagger"] = float(loss_stagger.detach())
    logs["ln_flag"] = float(loss_lnf.detach())
    logs["ln_len"] = float(loss_lnl.detach())

    # --- auto-style ----------------------------------------------------------------------------
    sm = batch["style_mask"].float()
    if sm.sum() > 0:
        loss_style = (F.smooth_l1_loss(out["style_pred"].float(), batch["style"].float(), reduction="none", beta=0.1) * sm).sum() / sm.sum()
    else:
        loss_style = zero
    total = total + weights.style * loss_style
    logs["style"] = float(loss_style.detach())
    logs["total"] = float(total.detach())
    return total, logs


@torch.no_grad()
def crop_metrics(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], threshold: float = 0.5) -> Dict[str, float]:
    """Cheap frame-level diagnostics on training/validation crops (map-level metrics come from metrics.py)."""
    tc = batch["time_class"]
    valid = batch["valid"] > 0
    probs = torch.softmax(out["time_logits"].float(), dim=-1)
    pred_ev = ((1.0 - probs[..., 0]) >= threshold) & valid
    true_ev = (tc > 0) & valid
    tp = (pred_ev & true_ev).sum().item()
    p = tp / max(1, pred_ev.sum().item())
    r = tp / max(1, true_ev.sum().item())
    res = {"frame_f1": 2 * p * r / max(1e-9, p + r)}
    if true_ev.any():
        off_pred = probs[..., 1:].argmax(dim=-1) + 1
        res["offset_exact"] = (off_pred[true_ev] == tc[true_ev]).float().mean().item()
        res["offset_within1ms"] = ((off_pred[true_ev] - tc[true_ev]).abs() <= 1).float().mean().item()
        res["lane_acc"] = (out["lane_logits"].argmax(dim=-1)[true_ev] == batch["lane_code"][true_ev]).float().mean().item()
        bits = (batch["event_mask"].unsqueeze(-1) >> torch.arange(4, device=tc.device)) & 1
        res["chord_acc"] = (out["chord_logits"].argmax(dim=-1)[true_ev] == (bits.sum(-1)[true_ev] - 1)).float().mean().item()
    return res
