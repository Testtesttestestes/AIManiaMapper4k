#!/usr/bin/env python3
"""
5-15 minute smoke test of the whole hybrid pipeline on synthetic music (no dataset needed).

    python hybrid_synthetic_demo.py                 # CPU-friendly defaults
    python hybrid_synthetic_demo.py --device cuda --songs 40 --epochs 8

Steps: synthetic .osz songs -> prepare_hybrid_dataset.py -> hybrid_solvability_audit.py ->
train_hybrid.py -> generate_hybrid.py on an unseen song -> per-ms comparison with its reference map ->
hybrid_long_track_test.py on the same song repeated to ~10 minutes.
The synthetic maps are a deterministic function of the audio, so a healthy pipeline reaches a high
event F1 within a few epochs; if it does not, something is broken before you spend GPU time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent


def run(args, **kw):
    print("\n$ python " + " ".join(str(a) for a in args), flush=True)
    res = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, **kw)
    if res.returncode != 0:
        sys.exit(f"step failed: {args[0]} (exit {res.returncode})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="hybrid_demo")
    ap.add_argument("--songs", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from osu_mapper.hybrid.beatmap import parse_osu_mania
    from osu_mapper.hybrid.metrics import per_ms_report
    from osu_mapper.hybrid.synthetic import make_synthetic_dataset

    work = (REPO / args.work).resolve()
    osz_dir, test_dir = work / "osz", work / "test_osz"
    if not osz_dir.exists():
        print(f"[demo] writing {args.songs} synthetic songs to {osz_dir}")
        make_synthetic_dataset(osz_dir, args.songs, seed=1, duration_s=args.seconds)
        make_synthetic_dataset(test_dir, 1, seed=77, duration_s=args.seconds)

    run(["prepare_hybrid_dataset.py", "--input_dir", osz_dir, "--output_dir", work / "dataset", "--quality_variants", "mp3_48k"])
    run(["hybrid_solvability_audit.py", "--input_dir", osz_dir, "--out", work / "audit.json"])
    run(["train_hybrid.py", "--data", work / "dataset", "--checkpoint_dir", work / "checkpoints", "--preset", args.preset,
         "--epochs", args.epochs, "--steps_per_epoch", args.steps, "--batch_size", 8, "--crop_sec", 10.24, "--warmup_steps", 100,
         "--lr", 1e-3, "--val_maps", 6, "--val_crops", 64, "--val_fraction", 0.17, "--device", args.device])

    test_osz = sorted(test_dir.glob("*.osz"))[0]
    out_osz = work / "generated.osz"
    with zipfile.ZipFile(test_osz) as z:
        refs = {n: parse_osu_mania(z.read(n).decode("utf-8")) for n in z.namelist() if n.endswith(".osu")}
    ref = refs[[n for n in refs if "[hard]" in n][0]]
    from osu_mapper.hybrid.inference import star_rating_of
    target = star_rating_of(ref.notes, ref.timing_points)
    run(["generate_hybrid.py", "--template", test_osz, "--sr", f"{target:.2f}", "--checkpoint", work / "checkpoints" / "hybrid_best.pt",
         "--output", out_osz, "--device", args.device])
    with zipfile.ZipFile(out_osz) as z:
        gen = parse_osu_mania(z.read([n for n in z.namelist() if n.endswith(".osu")][0]).decode("utf-8"))
    rep = per_ms_report(gen.notes, ref.notes)
    print(f"\n[demo] unseen song, generated vs reference ({len(gen.notes)} vs {len(ref.notes)} notes): "
          f"event F1 @2ms {rep['event_f1@2']:.3f} @16ms {rep['event_f1@16']:.3f} | timing MAE {rep['timing_mae_ms']:.2f} ms | per-ms loss {rep['per_ms_loss']:.3f}")

    with zipfile.ZipFile(test_osz) as z:
        (work / "test_audio.wav").write_bytes(z.read("audio.wav"))
    repeats = max(2, int(round(600.0 / args.seconds)))
    run(["hybrid_long_track_test.py", "--checkpoint", work / "checkpoints" / "hybrid_best.pt", "--audio", work / "test_audio.wav",
         "--repeat", repeats, "--sr", f"{target:.2f}", "--device", args.device, "--out", work / "long_track.json"])
    print("\n[demo] all steps finished - the pipeline works end to end.")


if __name__ == "__main__":
    main()
