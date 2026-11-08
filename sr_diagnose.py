#!/usr/bin/env python3
"""
Where does the star rating get lost? Requested vs achieved star rating of the network's maps.

    python sr_diagnose.py --checkpoint checkpoints/hybrid/hybrid_best.pt --songs "sample beatmaps" --max_songs 20
    python sr_diagnose.py --songs originals/ --requests 2,3,4,5,6,7,8 --out sr_diagnose.json

For every song (.osz with audio) and every requested star rating it reports three numbers:

* ``raw``      the network's own map: its decode threshold for that star rating, nothing fitted;
* ``fit``      after the star-rating fit of generate_hybrid.py (network_fit.fit_network_map);
* ``ceiling``  the hardest map the network's probabilities allow at all (lowest threshold).

Reading it: ``raw`` far below the request but ``fit`` on target -> the decoding fixes it; ``fit`` below the request
and ``ceiling`` below the request too -> the network does not hear enough notes at that difficulty (a training
problem: too few hard maps, regression to the mean); ``ceiling`` above but ``fit`` below -> the fit stops early.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np


def song_inputs(predictor, osz: Path, cache: Path):
    import torch

    from generate_hybrid import read_template
    from osu_mapper.hybrid.audio_features import load_audio
    from osu_mapper.hybrid.inference import prepare_song_inputs

    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"{osz.stem}_net.pt"
    with tempfile.TemporaryDirectory() as work:
        audio, timing, _ = read_template(osz, Path(work))
        if cf.exists():
            return torch.load(cf, map_location="cpu", weights_only=False), timing
        wave = load_audio(audio, predictor.feature_cfg.sample_rate)
    inputs = prepare_song_inputs(wave, predictor.feature_cfg, timing if any(t.uninherited for t in timing) else None,
                                 keep_reps=True, device=str(predictor.device))
    torch.save(inputs, cf)
    return inputs, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", "-c", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--songs", default="sample beatmaps", help="folder of .osz files with audio, or one .osz")
    ap.add_argument("--requests", default="2,3,4,5,6,7,8", help="requested star ratings")
    ap.add_argument("--max_songs", type=int, default=0)
    ap.add_argument("--cache", default="eval_cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write_calibration", default=None, metavar="JSON",
                    help="save the requested -> achieved curve of the raw maps; generate_hybrid.py --sr_calibration JSON "
                         "then asks the network for the rating at which it really gives the requested one")
    args = ap.parse_args()

    from osu_mapper.hybrid.inference import HybridPredictor, star_rating_of
    from osu_mapper.hybrid.network_fit import fit_network_map
    from osu_mapper.hybrid.renderer import DecodeKnobs, decode_outputs

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        ckpt = Path("test_NN/hybrid_best.pt")
    predictor = HybridPredictor.from_checkpoint(ckpt, device=args.device)
    print(f"[diagnose] {ckpt}")
    p = Path(args.songs)
    songs = [p] if p.suffix.lower() == ".osz" else sorted(p.glob("*.osz"))
    if args.max_songs:
        songs = songs[:args.max_songs]
    reqs = [float(x) for x in args.requests.split(",") if x.strip()]
    rows: List[Dict[str, float]] = []
    for osz in songs:
        try:
            inputs, timing = song_inputs(predictor, osz, Path(args.cache))
        except Exception as err:  # broken archive / no audio
            print(f"  [skip] {osz.name}: {err}")
            continue
        for req in reqs:
            out = predictor.predict(inputs, req)
            knobs = DecodeKnobs(onset_threshold=predictor.threshold_for(req), ln_threshold=predictor.ln_threshold)
            raw = star_rating_of(decode_outputs(out, knobs, timing), timing)
            low = DecodeKnobs(onset_threshold=0.15, ln_threshold=predictor.ln_threshold)
            ceiling = star_rating_of(decode_outputs(out, low, timing), timing)
            fit = fit_network_map(out, knobs, timing, req)
            rows.append(dict(song=osz.stem, request=req, raw=raw, fit=float(fit.star_rating), ceiling=ceiling))
            print(f"  {osz.stem[:28]:28s} request {req:4.1f} | raw {raw:5.2f} | fit {fit.star_rating:5.2f} | ceiling {ceiling:5.2f}",
                  flush=True)
    if not rows:
        raise SystemExit("no songs measured")
    print("\nrequest |  raw  |  fit  | ceiling | fit reached (+-0.2) | ceiling below request")
    for req in reqs:
        r = [x for x in rows if x["request"] == req]
        print(f"  {req:4.1f}  | {np.mean([x['raw'] for x in r]):5.2f} | {np.mean([x['fit'] for x in r]):5.2f} | "
              f"{np.mean([x['ceiling'] for x in r]):7.2f} | {np.mean([abs(x['fit'] - req) <= 0.2 for x in r]):18.0%} | "
              f"{np.mean([x['ceiling'] < req - 0.2 for x in r]):.0%}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if args.write_calibration:
        curve = {"checkpoint": str(ckpt), "requests": reqs,
                 "achieved": [float(np.mean([x["raw"] for x in rows if x["request"] == r])) for r in reqs]}
        Path(args.write_calibration).write_text(json.dumps(curve, indent=1), encoding="utf-8")
        print(f"[diagnose] calibration -> {args.write_calibration}")


if __name__ == "__main__":
    main()
