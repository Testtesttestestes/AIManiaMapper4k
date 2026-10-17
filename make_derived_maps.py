#!/usr/bin/env python3
"""
Adds easier "derived" versions of the real maps to a prepared hybrid dataset (no audio processing).

    python make_derived_maps.py --data dataset_hybrid
    python make_derived_maps.py --data dataset_hybrid --min_sr 3 --levels 0.75 0.55 0.4 0.28
    python make_derived_maps.py --data dataset_hybrid --remove        # back to the real maps only

Every real map with star rating >= --min_sr is thinned musically (osu_mapper/hybrid/thinning.py) to the
given shares of its events; each version gets its real rosu-pp star rating and an exact parameter track
(verified by re-rendering). Versions that are too close in SR to another difficulty of the same song are
skipped. Derived maps share the song's audio features, are marked "derived": true in index.json, are
never used for validation, get only a limited share of the training draws (train_hybrid.py
--derived_weight) and carry no style information (their style is mechanical, only SR -> density is taught).

Re-running replaces the previous derived maps. Run it again after prepare_hybrid_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from osu_mapper.hybrid.audio_features import FeatureConfig
from osu_mapper.hybrid.beatmap import ManiaMap, map_to_osu_text
from osu_mapper.hybrid.dataset import split_key
from osu_mapper.hybrid.params import ParamTrack
from osu_mapper.hybrid.renderer import render_track
from osu_mapper.hybrid.solver import STYLE_DIM, SolveStats, compute_style, solve_map
from osu_mapper.hybrid.sources import stable_id
from osu_mapper.hybrid.thinning import onset_curve, thin_map
from prepare_hybrid_dataset import star_rating


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--min_sr", type=float, default=2.5, help="only maps at least this hard are thinned")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.75, 0.55, 0.4, 0.28], help="shares of events to keep")
    ap.add_argument("--min_gap", type=float, default=0.35, help="skip a version closer than this (stars) to another difficulty of the song")
    ap.add_argument("--min_notes", type=int, default=80)
    ap.add_argument("--remove", action="store_true", help="only delete the derived maps")
    args = ap.parse_args()

    root = Path(args.data)
    index_path = root / "index.json"
    data: Dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
    real = [m for m in data["maps"] if not m.get("derived")]
    old = [m for m in data["maps"] if m.get("derived")]
    for m in old:
        (root / m["track"]).unlink(missing_ok=True)
    if old:
        print(f"removed {len(old)} previous derived maps")
    shutil.copy2(index_path, index_path.with_suffix(".json.bak"))
    if args.remove:
        data["maps"] = real
        _write(index_path, data)
        print(f"index: {len(real)} real maps")
        return

    fcfg = FeatureConfig.from_dict(data["feature_config"])
    onset_slice = fcfg.slices().get("onset")
    song_srs: Dict[str, List[float]] = {}
    for m in real:
        if m.get("roundtrip_exact") and float(m.get("sr", 0)) > 0:
            song_srs.setdefault(m["song_id"], []).append(float(m["sr"]))

    parents = [m for m in real if m.get("roundtrip_exact") and float(m.get("sr", 0)) >= args.min_sr]
    print(f"{len(parents)} real maps with SR >= {args.min_sr:g} will be thinned to {args.levels}")
    derived: List[Dict[str, Any]] = []
    onset_cache: Dict[str, Any] = {}
    t0 = time.time()
    for n_par, par in enumerate(parents, 1):
        track = ParamTrack.load_npz(root / par["track"])
        notes = render_track(track)
        tps = list(track.timing_points)
        sid = par["song_id"]
        if sid not in onset_cache:
            feats = np.load(root / data["songs"][sid]["features"][0], mmap_mode="r")
            onset_cache = {sid: onset_curve(feats, tuple(onset_slice) if onset_slice else None)}
        made = []
        for keep in sorted(args.levels, reverse=True):
            thin = thin_map(notes, tps, keep, onset_cache[sid], track.frame_ms)
            if len(thin) < args.min_notes:
                continue
            m = ManiaMap(notes=thin, timing_points=tps, overall_difficulty=float(par.get("od", 8.0)),
                         metadata={"Version": f"{par.get('version', '')} (thin {keep:g})", "Creator": par.get("creator", "")})
            sr = star_rating(map_to_osu_text(m), m)
            if sr <= 0:
                raise SystemExit("star rating calculation failed (rosu-pp missing?): derived maps need their real SR")
            others = song_srs.get(sid, [])
            if sr < 1.0 or sr > float(par["sr"]) - args.min_gap or any(abs(sr - o) < args.min_gap for o in others):
                continue
            stats = SolveStats()
            tr = solve_map(m, num_frames=track.num_frames, stats=stats)
            if render_track(tr) != sorted(thin):
                print(f"  [skip] roundtrip mismatch for {par['osu_file']} keep {keep:g}")
                continue
            mid = stable_id(par["map_id"], f"derived-{keep:g}")
            tr.save_npz(root / "maps" / f"{mid}.npz")
            style, _known = compute_style(tr, thin, sr, m.overall_difficulty)
            derived.append({
                "map_id": mid, "song_id": sid, "track": f"maps/{mid}.npz", "source": par["source"],
                "osu_file": f"{par['osu_file']} [thin {keep:g}]", "version": m.metadata["Version"], "creator": par.get("creator", ""),
                "sr": round(sr, 3), "od": m.overall_difficulty,
                "style": [round(float(x), 5) for x in style], "style_mask": [0.0] * STYLE_DIM,
                "notes": len(thin), "events": stats.events, "overflow": stats.overflow, "overflow_reasons": stats.overflow_reasons,
                "overlaps_in_ln": stats.overlaps_in_ln, "roundtrip_exact": True,
                "derived": True, "parent": par["map_id"], "keep": keep, "piece": split_key(par),
            })
            others.append(sr)
            song_srs[sid] = others
            made.append(sr)
        print(f"[{n_par}/{len(parents)}] {par.get('version', '')[:40]:<40} SR {float(par['sr']):.2f} -> "
              + (" ".join(f"{s:.2f}" for s in made) if made else "-"), flush=True)
    data["maps"] = real + derived
    _write(index_path, data)
    bins = np.bincount([int(min(d["sr"], 9.99)) for d in derived], minlength=10) if derived else np.zeros(10, int)
    real_bins = np.bincount([int(min(float(m.get("sr", 0)), 9.99)) for m in real if float(m.get("sr", 0)) > 0], minlength=10)
    print(f"\nderived maps: {len(derived)} from {len(parents)} parents in {time.time() - t0:.0f}s (backup: {index_path.with_suffix('.json.bak')})")
    print("real    per star: " + " ".join(f"{b}*:{int(c)}" for b, c in enumerate(real_bins) if c))
    print("derived per star: " + " ".join(f"{b}*:{int(c)}" for b, c in enumerate(bins) if c))


def _write(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
