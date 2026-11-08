"""
Direct preference optimization (DPO, Rafailov et al. 2023) for the hybrid network.

Pairs (chosen, rejected) of maps of the same song and star rating come from make_self_maps.py: the best and the worst
of the network's candidates by the objective reward, and a human map against the network's own map at the human map's
star rating ("the human did better", SPIN-style - no human rating needed).

The network factorizes over frames, so the log-probability of a whole map in a window is a sum:
    log p(map) = sum_frames log p(time class) + sum_events [log p(lane code) + log p(chord size) + sum_members log p(LN flag)]
and the DPO loss on a pair is
    -log sigmoid(beta * ((log pi(c) - log ref(c)) - (log pi(r) - log ref(r))))
with ``ref`` a frozen copy of the network at the start of the stage: the network moves towards the chosen maps and away
from the rejected ones, but only as far as the KL term allows (no reward hacking by drifting away from real maps).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .params import MAX_MEMBERS, ParamTrack


def map_logprob(out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
    """[B] log-probability of the target maps under the network outputs (sum over the window, see module docstring)."""
    tc = tgt["time_class"].long()
    v = valid.float()
    lp_time = torch.log_softmax(out["time_logits"].float(), dim=-1).gather(-1, tc.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    total = (lp_time * v).sum(dim=1)
    ev = (tc > 0).float() * v
    lane = tgt["lane_code"].long()
    lp_lane = torch.log_softmax(out["lane_logits"].float(), dim=-1).gather(-1, lane.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    total = total + (lp_lane * ev * (lane >= 0).float()).sum(dim=1)
    masks = tgt["event_mask"].long()
    chord = ((masks.unsqueeze(-1) >> torch.arange(4, device=masks.device)) & 1).sum(dim=-1)
    lp_chord = torch.log_softmax(out["chord_logits"].float(), dim=-1).gather(-1, (chord - 1).clamp(0, MAX_MEMBERS - 1).unsqueeze(-1)).squeeze(-1)
    total = total + (lp_chord * ev).sum(dim=1)
    member = (torch.arange(MAX_MEMBERS, device=masks.device).view(1, 1, -1) < chord.unsqueeze(-1)).float() * ev.unsqueeze(-1)
    lnf = tgt["ln_flag"].float()
    logit = out["ln_logits"].float()
    lp_ln = -F.binary_cross_entropy_with_logits(logit, lnf, reduction="none")
    total = total + (lp_ln * member).sum(dim=(1, 2))
    return total


def dpo_loss(pol_c: torch.Tensor, pol_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor, beta: float = 0.1):
    """(loss, preference accuracy) of a batch of pairs."""
    margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    return -F.logsigmoid(margin).mean(), float((margin > 0).float().mean())


class DPOPairDataset(Dataset):
    """Windows of preference pairs: the same crop of the song's audio, the chosen and the rejected map's targets."""

    def __init__(self, index, pairs: Sequence[Dict[str, Any]], crop_frames: int, samples_per_epoch: int, seed: int = 0):
        from .dataset import SparseTrack

        self.index = index
        self.crop = int(crop_frames)
        self.pairs = list(pairs)
        self.samples = int(samples_per_epoch)
        self.seed = seed
        self.tracks = [(SparseTrack(ParamTrack.load_npz(index.root / p["chosen"])),
                        SparseTrack(ParamTrack.load_npz(index.root / p["rejected"]))) for p in self.pairs]

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        from .audio_features import beat_features
        from .dataset import _crop_rows

        rng = random.Random(self.seed * 1_000_003 + i)
        k = rng.randrange(len(self.pairs))
        p = self.pairs[k]
        ch, rj = self.tracks[k]
        ev = np.union1d(ch.event_frames(), rj.event_frames())
        lo, hi = (int(ev[0]), int(ev[-1])) if ev.size else (0, ch.num_frames)
        start = rng.randint(lo - self.crop // 2, max(lo - self.crop // 2 + 1, hi - self.crop // 2))
        feats = _crop_rows(self.index.features(p["song_id"]), start, self.crop)
        beat = beat_features(ch.timing_points, start, self.crop, ch.frame_ms)
        frames = np.arange(start, start + self.crop)
        valid = (frames >= 0) & (frames < self.index.features(p["song_id"]).shape[0])
        item = {"feats": torch.from_numpy(feats), "beat": torch.from_numpy(beat),
                "valid": torch.from_numpy(valid.astype(np.float32)),
                "sr": torch.tensor([float(p["sr"])], dtype=torch.float32),
                "coarse": torch.from_numpy(self.index.coarse(p["song_id"]))}
        for tag, tr in (("c_", ch), ("r_", rj)):
            for key, arr in tr.crop_arrays(start, self.crop).items():
                if key in ("time_class", "lane_code", "event_mask", "ln_flag"):
                    item[tag + key] = torch.from_numpy(arr)
        return item
