#!/usr/bin/env python3
"""
Trains the critic of the v5 network (osu_mapper/v5/critic.py): audio + notes of a 10 s window -> "a human made this".

    python train_critic.py --data dataset_hybrid
    python train_critic.py --data dataset_hybrid --v5 checkpoints/v5/v5_best.pt --network_windows 400   # + the network's own maps

Targets: 1 for windows of human maps; for every other window its REAL quality against the human map of the same
place (0.6 x note F1 + 0.4 x expressiveness: bursts, rhythm, density per bar, long notes) - for the same windows made
bland (everything on 1/2, bursts collapsed), broken (lanes, timing, missing / extra notes, long notes), self-generated
maps of the dataset (make_self_maps.py), and - with --v5 - windows the v5 network generates on training songs. So
the critic learns how bad the network's own maps are and where, from their actual errors (re-run after each v5
training round, as the discriminator of GAIL keeps up with the generator). Validation: error of the predicted
quality per kind of window, and its correlation with the real quality of the network's windows.
The critic is saved as <checkpoint_dir>/critic.pt and used by generate_v5.py --search.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from osu_mapper.hybrid.audio_features import beat_features
from osu_mapper.hybrid.dataset import HybridIndex, _crop_rows
from osu_mapper.hybrid.inference import resolve_device
from osu_mapper.hybrid.renderer import render_track
from osu_mapper.v5.critic import CriticConfig, CriticNet, blandify, corrupt, map_frames
from osu_mapper.v5.data import SongCache, song_inputs

KINDS = ("human", "bland", "broken", "self", "network")


class CriticData(Dataset):
    def __init__(self, index: HybridIndex, cache: SongCache, items: List[Tuple[int, int, str, list]], W: int, seed: int):
        self.index, self.cache, self.items, self.W, self.seed = index, cache, items, W, seed
        self.epoch = 0
        self._notes: Dict[int, list] = {}
        self._tps: Dict[int, list] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _map(self, mi: int):
        if mi not in self._notes:
            tr = self.index.track(mi)
            self._notes[mi] = sorted(render_track(tr))
            self._tps[mi] = list(tr.timing_points)
        return self._notes[mi], self._tps[mi]

    def __getitem__(self, i: int):
        mi, start, kind, given = self.items[i]
        rng = random.Random(self.seed * 7919 + self.epoch * 104729 + i)
        e = self.index.maps[mi]
        notes, tps = self._map(mi)
        s_ms, e_ms = start * 10, (start + self.W) * 10
        win = [n for n in notes if s_ms - 2000 <= n.start < e_ms]
        if kind == "bland":
            win = blandify(win, tps)
        elif kind == "broken":
            win = corrupt(win, rng)
        elif kind in ("self", "network"):
            win = given
        feats = self.cache.window(e["song_id"], start, self.W, tps, sections=False)   # sound + stem attacks
        beat = beat_features(tps, start, self.W, 10)
        q = 1.0 if kind == "human" else window_quality(win, [n for n in notes if s_ms <= n.start < e_ms], s_ms, e_ms, tps)
        return (torch.from_numpy(feats), torch.from_numpy(beat.astype(np.float32)),
                torch.from_numpy(map_frames(win, s_ms, self.W)), torch.tensor(q, dtype=torch.float32),
                torch.tensor(KINDS.index(kind)))


def window_quality(pred, ref, s_ms: int, e_ms: int, timing_points) -> float:
    """How good a window is against the human map of the same place: 0.6 x note F1 (32 ms) + 0.4 x expressiveness
    (bursts, rhythm, density per bar, long notes). The critic learns this number - not only "human or not" - so it
    knows roughly how bad a map of the network is, from the real errors of the network's own maps."""
    from osu_mapper.hybrid.metrics import expressive_score, per_ms_report

    pred = [n for n in pred if s_ms <= n.start < e_ms]
    if not ref:
        return 1.0 if not pred else 0.3
    rep = per_ms_report(pred, ref, tolerances=(32,), timing_points=timing_points)
    return float(np.clip(0.6 * rep.get("event_f1@32", 0.0) + 0.4 * expressive_score(rep), 0.0, 1.0))


def make_items(index: HybridIndex, maps: List[int], W: int, per_map: int, rng: random.Random, network: Dict[int, list]):
    """(map, start frame, kind, notes for self / network kinds)."""
    items = []
    by_song_self: Dict[str, List[int]] = {}
    for i, m in enumerate(index.maps):
        if m.get("self"):
            by_song_self.setdefault(m["song_id"], []).append(i)
    for mi in maps:
        n = render_track(index.track(mi))
        if len(n) < 8:
            continue
        a, b = n[0].start // 10, n[-1].start // 10
        for _ in range(per_map):
            st = rng.randint(max(0, a - W // 4), max(a, b - W // 2))
            items.append((mi, st, "human", []))
            items.append((mi, st, rng.choice(["bland", "bland", "broken"]), []))
            sid = index.maps[mi]["song_id"]
            if by_song_self.get(sid) and rng.random() < 0.5:
                si = rng.choice(by_song_self[sid])
                sn = render_track(index.track(si))
                items.append((mi, st, "self", [x for x in sn if st * 10 - 2000 <= x.start < (st + W) * 10]))
            if mi in network and rng.random() < 0.7:
                nn_ = network[mi]
                items.append((mi, st, "network", [x for x in nn_ if st * 10 - 2000 <= x.start < (st + W) * 10]))
    return items


def network_maps(index: HybridIndex, cache: SongCache, maps: List[int], ckpt: str, n_maps: int, dev, max_sec: float):
    from osu_mapper.v5.infer import generate_map
    from osu_mapper.v5.model import V5Config, V5Net

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = V5Net(V5Config.from_dict(ck["config"]))
    net.load_state_dict(ck["ema"] or ck["model"])
    net.to(dev).eval()
    out = {}
    for mi in maps[:n_maps]:
        e = index.maps[mi]
        tr = index.track(mi)
        fa = song_inputs(np.asarray(index.features(e["song_id"], 0)), cache.slices, tr.timing_points,
                         highlight=net.cfg.highlight)
        Fr = min(fa.shape[0], int(max_sec * 100))
        out[mi] = generate_map(net, fa[:Fr], beat_features(tr.timing_points, 0, Fr, 10), tr.timing_points,
                               float(e.get("sr", 0.0)), device=dev)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--checkpoint_dir", default="checkpoints/v5")
    ap.add_argument("--v5", default=None, help="v5 checkpoint whose own maps become negatives")
    ap.add_argument("--network_windows", type=int, default=200, help="songs the v5 network maps for negatives")
    ap.add_argument("--network_max_sec", type=float, default=90.0)
    ap.add_argument("--per_map", type=int, default=6, help="windows per map and epoch")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    index = HybridIndex(args.data)
    cache = SongCache(index)
    tr_idx, va_idx = index.split(0.1, args.seed)
    real = lambda i: not index.maps[i].get("derived") and not index.maps[i].get("self")
    tr_idx, va_idx = [i for i in tr_idx if real(i)], [i for i in va_idx if real(i)]
    W = 1024
    network: Dict[int, list] = {}
    if args.v5:
        t0 = time.time()
        pick = random.Random(args.seed).sample(tr_idx, min(len(tr_idx), args.network_windows)) + va_idx
        network = network_maps(index, cache, pick, args.v5, len(pick), dev, args.network_max_sec)
        print(f"[critic] the v5 network mapped {len(network)} songs for negatives ({time.time() - t0:.0f}s)")
    probe = cache.window(index.maps[tr_idx[0]]["song_id"], 0, 4, None, sections=False)
    cfg = CriticConfig(feat_dim=probe.shape[1], d_model=args.d_model, window_frames=W)
    net = CriticNet(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    rng = random.Random(args.seed)
    va_items = make_items(index, va_idx, W, 4, random.Random(args.seed + 1), network)
    va = DataLoader(CriticData(index, cache, va_items, W, args.seed + 1), batch_size=args.batch_size, num_workers=0)
    out = Path(args.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        ds = CriticData(index, cache, make_items(index, tr_idx, W, args.per_map, rng, network), W, args.seed)
        ds.epoch = ep
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for f, b, m, y, _k in dl:
            f, b, m, y = f.to(dev), b.to(dev), m.to(dev), y.to(dev)
            loss = F.binary_cross_entropy_with_logits(net(f, b, m), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        net.eval()
        preds: Dict[str, list] = {k: [] for k in KINDS}
        with torch.no_grad():
            for f, b, m, y, k in va:
                p = torch.sigmoid(net(f.to(dev), b.to(dev), m.to(dev))).cpu()
                for pi, yi, ki in zip(p.tolist(), y.tolist(), k.tolist()):
                    preds[KINDS[ki]].append((pi, yi))
        err = {k: float(np.mean([abs(a - b) for a, b in v])) for k, v in preds.items() if v}
        net_pairs = preds.get("network", []) + preds.get("self", [])
        corr = float(np.corrcoef(*zip(*net_pairs))[0, 1]) if len(net_pairs) > 3 else float("nan")
        neg = [v for k, v in err.items() if k != "human"]
        # half human, half the others: predicting one constant cannot score well
        bal = 1.0 - (0.5 * err.get("human", 1.0) + 0.5 * (float(np.mean(neg)) if neg else 1.0))
        print(f"[critic {ep}] train loss {tot / max(1, n):.4f} ({time.time() - t0:.0f}s) | held-out error of the predicted "
              f"quality: " + " ".join(f"{k} {v:.3f}" for k, v in err.items())
              + (f" | network maps: correlation with their real quality {corr:.2f}" if np.isfinite(corr) else "")
              + f" | score {bal:.3f}", flush=True)
        acc = err
        if bal > best:
            best = bal
            torch.save({"config": cfg.to_dict(), "model": net.state_dict(), "accuracy": acc}, out / "critic.pt")
            print(f"  * saved {out / 'critic.pt'}")
    if best < 0:
        sys.exit("no validation windows")


if __name__ == "__main__":
    main()
