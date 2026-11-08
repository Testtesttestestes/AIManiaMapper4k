#!/usr/bin/env python3
"""
Self-generated training maps chosen by an objective reward (ReST / best-of-N with relabeling).

    python make_self_maps.py --data dataset_hybrid --checkpoint checkpoints/hybrid/hybrid_best.pt
    python make_self_maps.py --data dataset_hybrid --targets 6,7,8 --candidates 6 --max_songs 200
    python make_self_maps.py --data dataset_hybrid --remove                 # back to the human (+ thinned) maps

Why: the network saturates where human maps are rare (sr_diagnose.py: ~5* on the 6-8* requests). For every song of
the dataset and every requested star rating that the song has no human map near, the network makes several
candidate maps (different guidance strengths and lane sampling temperatures, each fitted to the request like
generate_hybrid.py does), the reward (osu_mapper/hybrid/reward.py: stars, playability, patterns of real maps, no
notes in silence) picks the best, and the chosen map is added to the dataset with its REAL star rating (rosu-pp),
not the requested one - the network gets hard examples it can actually learn from, labelled honestly.

The new maps are marked "derived": true and "self": true: never used for validation, drawn only with the derived
share of training crops (train_hybrid.py --derived_weight), no style information. They belong to the song's piece, so
songs of the validation split stay out of training. Repeat after each training round (the next network makes
better candidates); make_derived_maps.py keeps them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--checkpoint", "-c", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--targets", default="5.5,6.5,7.5,8.5", help="requested star ratings")
    ap.add_argument("--candidates", type=int, default=4, help="maps generated per song and request (best one kept)")
    ap.add_argument("--min_gap", type=float, default=0.4, help="skip a request this close to a human map of the song")
    ap.add_argument("--max_error", type=float, default=0.5, help="keep only maps within this many stars of the request")
    ap.add_argument("--min_reward", type=float, default=-3.0, help="keep only maps with at least this reward")
    ap.add_argument("--max_songs", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--motifs", default=None, help="motif library for the pattern reward (default: test_NN/motifs.npz)")
    ap.add_argument("--remove", action="store_true", help="only delete the self-generated maps")
    ap.add_argument("--no_pairs", action="store_true",
                    help="do not save DPO preference pairs (best vs worst candidate, human map vs the network's map at the "
                         "human map's rating; train_hybrid.py --dpo_weight uses them)")
    args = ap.parse_args()

    from make_derived_maps import _write
    from osu_mapper.hybrid.audio_features import FeatureConfig, beat_features
    from osu_mapper.hybrid.beatmap import ManiaMap
    from osu_mapper.hybrid.dataset import split_key
    from osu_mapper.hybrid.inference import HybridPredictor, SongInputs
    from osu_mapper.hybrid.motifs import LanePredictor, load_default_motifs
    from osu_mapper.hybrid.network_fit import fit_network_map
    from osu_mapper.hybrid.params import ParamTrack
    from osu_mapper.hybrid.renderer import DecodeKnobs, render_track
    from osu_mapper.hybrid.reward import map_reward
    from osu_mapper.hybrid.solver import STYLE_DIM, SolveStats, compute_style, solve_map
    from osu_mapper.hybrid.sources import stable_id

    root = Path(args.data)
    index_path = root / "index.json"
    data: Dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
    keep = [m for m in data["maps"] if not m.get("self")]
    old = [m for m in data["maps"] if m.get("self")]
    for m in old:
        (root / m["track"]).unlink(missing_ok=True)
    if old:
        print(f"removed {len(old)} previous self-generated maps")
    for p_ in data.get("dpo_pairs", []):
        if p_.get("rejected_own", True):
            (root / p_["rejected"]).unlink(missing_ok=True)
    data["dpo_pairs"] = []
    shutil.copy2(index_path, index_path.with_suffix(".json.bak"))
    if args.remove:
        data["maps"] = keep
        _write(index_path, data)
        return

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        ckpt = Path("test_NN/hybrid_best.pt")
    predictor = HybridPredictor.from_checkpoint(ckpt, device=args.device)
    fcfg = FeatureConfig.from_dict(data["feature_config"])
    if fcfg.to_dict() != predictor.feature_cfg.to_dict():
        raise SystemExit("the dataset was prepared with another feature config than the checkpoint (e.g. --external): "
                         "use the dataset the model was trained on")
    lib = load_default_motifs(args.motifs, log=print)
    spec = fcfg.slices().get("spectral")
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    real = [m for m in keep if not m.get("derived") and m.get("roundtrip_exact")]
    by_song: Dict[str, List[Dict[str, Any]]] = {}
    for m in real:
        by_song.setdefault(m["song_id"], []).append(m)
    songs = sorted(by_song)
    if args.max_songs:
        songs = songs[:args.max_songs]
    made: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    t0 = time.time()

    def save_track(notes, tag: str, ref_map, num_frames: int):
        m_ = ManiaMap(notes=list(notes), timing_points=tps, overall_difficulty=float(ref_map.get("od", 8.0)))
        st_ = SolveStats()
        tr_ = solve_map(m_, num_frames=num_frames, stats=st_)
        if render_track(tr_) != sorted(m_.notes):
            return None, None, None
        mid_ = stable_id(ref_map["map_id"], tag)
        tr_.save_npz(root / "maps" / f"{mid_}.npz")
        return mid_, tr_, st_
    # candidate settings: classifier-free guidance (hears more notes) x lane sampling temperature (other patterns)
    variants = [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (1.5, 0.6), (1.0, 0.6), (2.5, 0.0), (2.0, 0.6), (3.0, 0.3)]
    for n_song, sid in enumerate(songs, 1):
        maps = by_song[sid]
        ref = max(maps, key=lambda m: float(m.get("sr", 0.0)))
        tr_ref = ParamTrack.load_npz(root / ref["track"])
        tps = list(tr_ref.timing_points)
        song = data["songs"][sid]
        feats = np.load(root / song["features"][0]).astype(np.float32)
        coarse = np.load(root / song["coarse"]).astype(np.float32)
        beat = beat_features(tps, 0, feats.shape[0], fcfg.frame_ms)
        inputs = SongInputs(feats, coarse, beat, int(song.get("duration_ms", feats.shape[0] * fcfg.frame_ms)))
        power = feats[:, spec[0]] if spec else None
        human = [float(m.get("sr", 0.0)) for m in maps]
        if not args.no_pairs:
            # the human map against the network's own map at the same rating: "the human did better" (SPIN-style)
            for hm in maps:
                hsr = float(hm.get("sr", 0.0))
                out = predictor.predict(inputs, hsr)
                knobs = DecodeKnobs(onset_threshold=predictor.threshold_for(hsr), ln_threshold=predictor.ln_threshold)
                fit = fit_network_map(out, knobs, tps, hsr, float(hm.get("od", 8.0)))
                if len(fit.notes) < 20:
                    continue
                mid_, _, _ = save_track(fit.notes, f"dpo-rej-human-{hsr:g}", hm, tr_ref.num_frames)
                if mid_ is not None:
                    pairs.append({"song_id": sid, "sr": hsr, "chosen": hm["track"], "rejected": f"maps/{mid_}.npz",
                                  "kind": "human>network", "rejected_own": True})
        for target in targets:
            if any(abs(h - target) < args.min_gap for h in human):
                continue
            pred = LanePredictor.cached(lib, target) if lib is not None else None
            best = None
            worst = None
            for g, temp in variants[:max(1, args.candidates)]:
                out = predictor.predict(inputs, target, sr_guidance=g)
                knobs = DecodeKnobs(onset_threshold=predictor.threshold_for(target), ln_threshold=predictor.ln_threshold,
                                    lane_temperature=temp)
                fit = fit_network_map(out, knobs, tps, target, float(ref.get("od", 8.0)))
                r = map_reward(fit.notes, tps, target, float(ref.get("od", 8.0)), pred, power, sr=float(fit.star_rating))
                if best is None or r["reward"] > best[0]["reward"]:
                    best = (r, fit.notes, g, temp)
                if worst is None or r["reward"] < worst[0]["reward"]:
                    worst = (r, fit.notes, g, temp)
            r, notes, g, temp = best
            ok = abs(r["sr"] - target) <= args.max_error and r["reward"] >= args.min_reward
            print(f"[{n_song}/{len(songs)}] {ref.get('osu_file', sid)[:44]:<44} ask {target:.1f} -> {r['sr']:.2f}* "
                  f"reward {r['reward']:+.2f} (guidance {g:g}, lanes T {temp:g}){'' if ok else '  - not kept'}", flush=True)
            if not ok:
                continue
            m = ManiaMap(notes=list(notes), timing_points=tps, overall_difficulty=float(ref.get("od", 8.0)))
            stats = SolveStats()
            tr = solve_map(m, num_frames=tr_ref.num_frames, stats=stats)
            if render_track(tr) != sorted(m.notes):
                print("  [skip] roundtrip mismatch")
                continue
            mid = stable_id(ref["map_id"], f"self-{target:g}")
            tr.save_npz(root / "maps" / f"{mid}.npz")
            style, _ = compute_style(tr, m.notes, r["sr"], m.overall_difficulty)
            made.append({
                "map_id": mid, "song_id": sid, "track": f"maps/{mid}.npz", "source": ref["source"],
                "osu_file": f"{ref.get('osu_file', '')} [self {target:g}]", "version": f"self {target:g}", "creator": "network",
                "sr": round(float(r["sr"]), 3), "od": m.overall_difficulty,
                "style": [round(float(x), 5) for x in style], "style_mask": [0.0] * STYLE_DIM,
                "notes": len(m.notes), "events": stats.events, "overflow": stats.overflow,
                "overflow_reasons": stats.overflow_reasons, "overlaps_in_ln": stats.overlaps_in_ln, "roundtrip_exact": True,
                "derived": True, "self": True, "parent": ref["map_id"], "requested_sr": target,
                "reward": {k: round(float(v), 4) for k, v in r.items()}, "piece": split_key(ref),
            })
            human.append(float(r["sr"]))
            if not args.no_pairs and worst is not None and worst[0]["reward"] < r["reward"] - 0.5:
                mid_w, _, _ = save_track(worst[1], f"dpo-rej-{target:g}", ref, tr_ref.num_frames)
                if mid_w is not None:
                    pairs.append({"song_id": sid, "sr": target, "chosen": f"maps/{mid}.npz", "rejected": f"maps/{mid_w}.npz",
                                  "kind": "best>worst", "rejected_own": True,
                                  "rewards": [round(float(r["reward"]), 3), round(float(worst[0]["reward"]), 3)]})
    data["maps"] = keep + made
    data["dpo_pairs"] = pairs
    _write(index_path, data)
    if pairs:
        print(f"DPO pairs: {len(pairs)} ({sum(p['kind'] == 'human>network' for p in pairs)} human > network, "
              f"{sum(p['kind'] == 'best>worst' for p in pairs)} best > worst)")
    bins = np.bincount([int(min(d["sr"], 9.99)) for d in made], minlength=10) if made else np.zeros(10, int)
    print(f"\nself-generated maps: {len(made)} from {len(songs)} songs in {time.time() - t0:.0f}s "
          f"(backup: {index_path.with_suffix('.json.bak')})")
    print("per star: " + " ".join(f"{b}*:{int(c)}" for b, c in enumerate(bins) if c))


if __name__ == "__main__":
    main()
