#!/usr/bin/env python3
"""
Trains the meta model of bursts (osu_mapper/v5/meta.py): "a burst proposed here - would a human mapper burst here at
this star rating?"

    python train_meta.py --data dataset_hybrid
    python train_meta.py --data dataset_hybrid --v5 checkpoints/v5/v5_best.pt     # + bursts the v5 network proposes

Proposals: the bursts of every other difficulty of the same song (and of the v5 network's maps with --v5); label =
the human map of the target difficulty has a burst over at least half of the span. Validation on held-out songs:
accuracy, and - the point of meta-labeling - the precision of the proposals before and after the filter (how many of
the bursts that are kept are really there in the human map) and how many real ones are lost.
Saved as <checkpoint_dir>/meta.pt; generate_v5.py uses it (--meta).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from osu_mapper.hybrid.audio_features import beat_features
from osu_mapper.hybrid.dataset import HybridIndex
from osu_mapper.hybrid.inference import resolve_device
from osu_mapper.hybrid.metrics import bursts
from osu_mapper.hybrid.renderer import render_track
from osu_mapper.v5.data import SongCache, song_inputs
from osu_mapper.v5.meta import BurstMeta, MetaConfig, burst_inputs, input_columns, span_has_burst


def proposals(index: HybridIndex, maps: List[int], cache: SongCache, highlight, v5=None, dev=None, max_sec: float = 90.0,
              per_song: int = 40, seed: int = 0):
    """(frames, shape, label, source) of every proposed burst (at most ``per_song`` per song, drawn at random)."""
    rng = random.Random(seed)
    by_song: Dict[str, List[int]] = {}
    for i in maps:
        by_song.setdefault(index.maps[i]["song_id"], []).append(i)
    out = []
    for sid, ids in by_song.items():
        notes = {i: sorted(render_track(index.track(i))) for i in ids}
        tps0 = list(index.track(ids[0]).timing_points)
        feats_all = song_inputs(np.asarray(index.features(sid, 0)), cache.slices, tps0, highlight=highlight)
        beat_all = beat_features(tps0, 0, feats_all.shape[0], 10)
        bursts_of = {i: bursts(np.array(sorted({n.start for n in notes[i]}), dtype=np.float64), tps0) for i in ids}
        song_props = []
        for i in ids:                                          # target difficulty
            sr = float(index.maps[i].get("sr", 0.0))
            props = []
            for j in ids:
                if j != i:
                    props += [(sp, [n for n in notes[j] if sp[0] <= n.start <= sp[1]], "sibling") for sp in bursts_of[j]]
            if v5 is not None:
                from osu_mapper.v5.infer import generate_map
                F_ = min(feats_all.shape[0], int(max_sec * 100))
                gen = generate_map(v5, feats_all[:F_], beat_all[:F_], tps0, sr, device=dev)
                t = np.array(sorted({n.start for n in gen}), dtype=np.float64)
                props += [(sp, [n for n in gen if sp[0] <= n.start <= sp[1]], "network") for sp in bursts(t, tps0)]
            song_props.extend((i, sp, members, src, sr) for sp, members, src in props)
        cols = input_columns(cache.slices, feats_all.shape[1])
        if len(song_props) > per_song:
            song_props = rng.sample(song_props, per_song)
        for i, sp, members, src, sr in song_props:
            fr, sh = burst_inputs(feats_all, beat_all, sp, members, tps0, sr, cols)
            out.append((fr, sh, float(span_has_burst(sp, bursts_of[i])), src))
    return out, cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--checkpoint_dir", default="checkpoints/v5")
    ap.add_argument("--v5", default=None, help="v5 checkpoint: its own bursts become proposals too")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--per_song", type=int, default=40, help="proposals per song at most (memory)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    index = HybridIndex(args.data)
    tr_idx, va_idx = index.split(0.1, args.seed)
    real = lambda i: not index.maps[i].get("derived") and not index.maps[i].get("self")
    tr_idx, va_idx = [i for i in tr_idx if real(i)], [i for i in va_idx if real(i)]
    v5, highlight = None, []
    if args.v5:
        from osu_mapper.v5.model import V5Config, V5Net
        ck = torch.load(args.v5, map_location="cpu", weights_only=False)
        v5 = V5Net(V5Config.from_dict(ck["config"]))
        v5.load_state_dict(ck.get("ema") or ck["model"])
        v5.to(dev).eval()
        highlight = v5.cfg.highlight
    cache = SongCache(index, highlight=highlight)
    t0 = time.time()
    tr, cols = proposals(index, tr_idx, cache, highlight, v5, dev, per_song=args.per_song, seed=args.seed)
    va, _ = proposals(index, va_idx, cache, highlight, v5, dev, per_song=args.per_song, seed=args.seed + 1)
    if not tr:
        raise SystemExit("no burst proposals: the dataset needs songs with several difficulties that have bursts")
    pos = np.mean([y for _, _, y, _ in tr])
    print(f"[meta] {len(tr)} training / {len(va)} validation proposals ({100 * pos:.0f}% real), {time.time() - t0:.0f}s")
    D = tr[0][0].shape[1] - 13
    net = BurstMeta(MetaConfig(feat_dim=D, cols=cols)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    X = torch.from_numpy(np.stack([f for f, _, _, _ in tr]))            # float16 in memory, float32 per batch
    S = torch.from_numpy(np.stack([s for _, s, _, _ in tr]))
    Y = torch.tensor([y for _, _, y, _ in tr], dtype=torch.float32)
    pw = torch.tensor((1 - pos) / max(pos, 1e-3))
    best, out = -1.0, Path(args.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = rng.permutation(len(Y))
        for k in range(0, len(perm), args.batch_size):
            b = perm[k:k + args.batch_size]
            loss = F.binary_cross_entropy_with_logits(net(X[b].to(dev).float(), S[b].to(dev)), Y[b].to(dev), pos_weight=pw.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if not va:
            continue
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(np.stack([f for f, _, _, _ in va])).to(dev).float(),
                                  torch.from_numpy(np.stack([s for _, s, _, _ in va])).to(dev))).cpu().numpy()
        y = np.array([y for _, _, y, _ in va]) > 0.5
        keep = p >= args.threshold
        prec0 = float(y.mean())
        prec1 = float(y[keep].mean()) if keep.any() else 0.0
        lost = float((~keep & y).sum() / max(1, y.sum()))
        acc = float(((p >= 0.5) == y).mean())
        score = prec1 - prec0 - 0.5 * lost
        print(f"[meta {ep}] held-out: accuracy {acc:.3f} | precision of the proposals {prec0:.3f} -> {prec1:.3f} "
              f"after the filter, real bursts lost {100 * lost:.0f}%", flush=True)
        if score > best:
            best = score
            torch.save({"config": net.cfg.to_dict(), "model": net.state_dict(), "threshold": args.threshold,
                        "val": {"accuracy": acc, "precision_before": prec0, "precision_after": prec1, "lost": lost}},
                       out / "meta.pt")
    if not va:
        torch.save({"config": net.cfg.to_dict(), "model": net.state_dict(), "threshold": args.threshold}, out / "meta.pt")
    print(f"[meta] saved {out / 'meta.pt'}")


if __name__ == "__main__":
    main()
