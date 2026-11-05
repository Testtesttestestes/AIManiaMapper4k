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

import numpy as np
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
    density: float = 0.3   # expected events / notes per window vs the reference (do not under-/over-fill)
    pair: float = 0.3      # same window at two difficulties: the density RATIO must match the references
    jack: float = 0.0      # fast transitions (< 160 ms): P(the chosen lane code hits a lane of the previous chord) vs the map
    ladder: float = 0.0    # difficulty ladder: which moments are mapped at which star rating (needs the importance head)

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
    # masked diffusion: frames given on the canvas are not predicted (time, lanes, members count only masked frames)
    valid_pred = valid * (batch["canvas_state"] == 0).float() if "canvas_state" in batch else valid
    logs: Dict[str, float] = {}

    # --- time (per-ms) -------------------------------------------------------------------------
    target = soft_time_targets(tc, frame_ms, sigma_ms)
    logp = F.log_softmax(out["time_logits"].float(), dim=-1)
    ce = -(target * logp).sum(dim=-1)
    w = (1.0 + (event_weight - 1.0) * (1.0 - target[..., 0])) * valid_pred
    loss_time = (ce * w).sum() / w.sum().clamp(min=1.0)
    total = weights.time * loss_time
    logs["time"] = float(loss_time.detach())

    ev = (tc > 0) & (valid_pred > 0)
    n_ev = int(ev.sum())
    logs["events"] = float(n_ev)
    # zero-valued terms stay connected to their heads, so every parameter receives a (zero) gradient:
    # required by DistributedDataParallel and harmless otherwise
    def zero(*keys: str) -> torch.Tensor:
        return sum(out[k].float().sum() for k in keys) * 0.0
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
            loss_stagger = zero("stagger_logits")
        loss_lnf = F.binary_cross_entropy_with_logits(out["ln_logits"][member].float(), batch["ln_flag"][member].float())
        ln_member = member & (batch["ln_flag"] > 0.5)
        if ln_member.any():
            b_t, r_t = ln_targets(batch["ln_len"][ln_member])
            loss_lnl = F.cross_entropy(out["ln_bucket_logits"][ln_member].float(), b_t) + F.l1_loss(out["ln_resid"][ln_member].float(), r_t)
        else:
            loss_lnl = zero("ln_bucket_logits", "ln_resid")
    else:
        loss_lane = zero("lane_logits")
        loss_chord = zero("chord_logits")
        loss_stagger = zero("stagger_logits")
        loss_lnf = zero("ln_logits")
        loss_lnl = zero("ln_bucket_logits", "ln_resid")

    total = total + weights.lane * loss_lane + weights.chord * loss_chord + weights.stagger * loss_stagger + weights.ln_flag * loss_lnf + weights.ln_len * loss_lnl
    logs["lane"] = float(loss_lane.detach())
    logs["chord"] = float(loss_chord.detach())
    logs["stagger"] = float(loss_stagger.detach())
    logs["ln_flag"] = float(loss_lnf.detach())
    logs["ln_len"] = float(loss_lnl.detach())

    # --- density and pairs -------------------------------------------------------------------------
    # expected number of events / notes in every window from the probabilities vs the reference counts:
    # under-filling a dense map (or over-filling an easy one) now costs something, not only timing errors
    probs = torch.softmax(out["time_logits"].float(), dim=-1)
    p_ev = (1.0 - probs[..., 0]) * valid
    sizes = torch.arange(1, MAX_MEMBERS + 1, device=tc.device, dtype=torch.float32)
    exp_chord = (torch.softmax(out["chord_logits"].float(), dim=-1) * sizes).sum(dim=-1)
    e_events = p_ev.sum(dim=1)
    e_notes = (p_ev * exp_chord).sum(dim=1)
    ev_f = ev.float()
    bits_all = ((batch["event_mask"].unsqueeze(-1) >> torch.arange(4, device=tc.device)) & 1).sum(dim=-1).float()
    n_events = ev_f.sum(dim=1)
    n_notes = (bits_all * ev_f).sum(dim=1)
    has = valid.sum(dim=1) > 0
    log_e = torch.stack([torch.log1p(e_events), torch.log1p(e_notes)], dim=-1)
    log_n = torch.stack([torch.log1p(n_events), torch.log1p(n_notes)], dim=-1)
    if weights.density > 0 and has.any():
        loss_density = F.smooth_l1_loss(log_e[has], log_n[has], beta=0.1)
    else:
        loss_density = zero("time_logits", "chord_logits")
    pair = batch.get("pair_prev")
    pidx = torch.nonzero((pair.view(-1) > 0) & (torch.arange(B, device=tc.device) >= 1)).view(-1) if pair is not None else None
    if weights.pair > 0 and pidx is not None and pidx.numel() > 0:
        loss_pair = F.smooth_l1_loss(log_e[pidx] - log_e[pidx - 1], log_n[pidx] - log_n[pidx - 1], beta=0.1)
    else:
        loss_pair = zero("time_logits", "chord_logits")
    total = total + weights.density * loss_density + weights.pair * loss_pair
    logs["density"] = float(loss_density.detach())
    logs["pair"] = float(loss_pair.detach())
    logs["pairs"] = float(pidx.numel()) if pidx is not None else 0.0

    # --- fast jacks ----------------------------------------------------------------------------
    # the network does not see the lanes it decoded before: an explicit mask hits the previous chord by chance, and on maps
    # it never saw its P(jack) at fast transitions was 0.27 against 0.09 in the human maps. With the reference's previous
    # two chords, every lane code has a known outcome; P(outcome shares a lane with the previous chord) is trained against
    # whether the map does. Any code giving a non-jack is right where the map has none (cross-entropy wants only one).
    if weights.jack > 0:
        loss_jack, n_pairs = jack_loss(out["lane_logits"], tc, batch["event_mask"], valid, frame_ms)
        total = total + weights.jack * loss_jack
        logs["jack"] = float(loss_jack.detach())
        logs["jack_pairs"] = float(n_pairs)

    # --- difficulty ladder ---------------------------------------------------------------------
    # every moment that any map of the song has (+-1 frame): mapped at star rating s_k or not, for each of the song's maps
    # -> sigma(importance - theta(s_k)); strong hits end up important at every difficulty, fills only at the hard ones
    if weights.ladder > 0 and "importance" in out and "ladder_presence" in batch:
        from .model import ladder_threshold
        pres = batch["ladder_presence"].float()                    # [B, K, T]
        srs = batch["ladder_sr"].float()                           # [B, K]
        have = (srs > 0).float().unsqueeze(-1)                     # [B, K, 1]
        known = (pres >= 0).float() * have                         # [B, K, T]
        union = ((pres * known).amax(dim=1) > 0).float() * valid  # [B, T]
        theta = ladder_threshold(out["ladder_c0"], out["ladder_c"], srs.clamp(min=0.0))  # [B, K]
        logit = out["importance"].float().unsqueeze(1) - theta.unsqueeze(-1)          # [B, K, T]
        m = known * union.unsqueeze(1)
        if m.sum() > 0:
            bce = F.binary_cross_entropy_with_logits(logit, pres.clamp(min=0.0), reduction="none")
            loss_ladder = (bce * m).sum() / m.sum()
        else:
            loss_ladder = out["importance"].float().sum() * 0.0 + out["ladder_c"].sum() * 0.0
        total = total + weights.ladder * loss_ladder
        logs["ladder"] = float(loss_ladder.detach())

    # --- auto-style ----------------------------------------------------------------------------
    sm = batch["style_mask"].float()
    if sm.sum() > 0:
        loss_style = (F.smooth_l1_loss(out["style_pred"].float(), batch["style"].float(), reduction="none", beta=0.1) * sm).sum() / sm.sum()
    else:
        loss_style = zero("style_pred")
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


_LANE_TABLE_T: Dict[str, torch.Tensor] = {}


def jack_loss(lane_logits: torch.Tensor, tc: torch.Tensor, event_mask: torch.Tensor, valid: torch.Tensor, frame_ms: int,
              jack_ms: float = 160.0):
    """BCE of P(lane code -> chord sharing a lane with the previous chord) on events less than ``jack_ms`` after the
    previous one, given the reference's previous two chords (see compute_losses)."""
    from .params import LANE_TABLE

    key = str(lane_logits.device)
    if key not in _LANE_TABLE_T:
        _LANE_TABLE_T[key] = torch.from_numpy(LANE_TABLE.astype(np.int64)).to(lane_logits.device)
    table = _LANE_TABLE_T[key]                                            # [16, 16, C]
    rows, cur, prev, prev2 = [], [], [], []
    tc_c, em_c, va_c = tc.cpu().numpy(), event_mask.cpu().numpy(), valid.cpu().numpy()
    for b in range(tc_c.shape[0]):
        fr = np.flatnonzero(tc_c[b] > 0)
        if fr.size < 3:
            continue
        ms = fr * frame_ms + tc_c[b, fr] - 1
        i = np.arange(2, fr.size)
        keep = (ms[i] - ms[i - 1] < jack_ms) & (va_c[b, fr[i]] > 0) & (em_c[b, fr[i - 1]] > 0) & (em_c[b, fr[i]] > 0)
        i = i[keep]
        rows += [(b, int(f)) for f in fr[i]]
        cur += list(em_c[b, fr[i]])
        prev += list(em_c[b, fr[i - 1]])
        prev2 += list(em_c[b, fr[i - 2]])
    if not rows:
        return lane_logits.float().sum() * 0.0, 0
    dev = lane_logits.device
    bi = torch.tensor([r[0] for r in rows], device=dev)
    fi = torch.tensor([r[1] for r in rows], device=dev)
    p = torch.tensor(prev, device=dev, dtype=torch.int64) & 15
    p2 = torch.tensor(prev2, device=dev, dtype=torch.int64) & 15
    y = ((torch.tensor(cur, device=dev, dtype=torch.int64) & p) > 0).float()
    res = table[p, p2]                                                    # [N, C] resulting masks (0 = invalid code)
    prob = torch.softmax(lane_logits[bi, fi].float(), dim=-1) * (res > 0).float()
    prob = prob / prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    q = (prob * ((res & p.unsqueeze(-1)) > 0).float()).sum(dim=-1).clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(q, y), len(rows)
