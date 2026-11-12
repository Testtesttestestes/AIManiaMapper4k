#!/usr/bin/env python3
"""
osu!mania 4K maps with the v5 network (autoregressive, osu_mapper/v5).

    python generate_v5.py song.mp3 --sr 4.5
    python generate_v5.py --template my_template.osz --sr 3.5,5,6.5 --search 4
    python generate_v5.py song.mp3 --sr 6 --search 6 --critic checkpoints/v5/critic.pt --why

Every 5 s phrase is sampled ``--search`` times; the critic (train_critic.py: "would a human have mapped this?") and
the per-note reward (jacks, one-hand runs, notes in silence, patterns) choose the continuation that is kept.
``--why`` prints, per phrase, which sound sources the network listened to (the source selector) and what the notes
follow (the cause head): drums, bass, other, vocals, or the beat grid.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from generate_hybrid import AUDIO_EXTS, read_template
from osu_mapper.hybrid.audio_features import beat_features, load_audio
from osu_mapper.hybrid.beatmap import ManiaMap, RawTimingPoint, map_to_osu_text
from osu_mapper.hybrid.causes import CAUSES, stem_attacks
from osu_mapper.hybrid.inference import estimated_timing_points, prepare_song_inputs, resolve_device, star_rating_of
from osu_mapper.v5.generate import generate_song
from osu_mapper.v5.model import V5Config, V5Net


def make_scorer(critic, feats_all, beat_all, timing, sr, lane_pred, power, critic_weight: float, reward_weight: float):
    from osu_mapper.hybrid.reward import note_rewards

    def score(new_notes, so_far, start_ms, end_ms):
        allnotes = sorted(list(so_far) + list(new_notes))
        s = 0.0
        if critic is not None:
            s += critic_weight * critic.score(feats_all, beat_all, allnotes, end_ms)
        if reward_weight > 0 and len(new_notes) >= 2:
            nr = note_rewards(allnotes, sr, lane_pred, power)
            sel = (nr["times"] >= start_ms) & (nr["times"] < end_ms)
            if sel.any():
                s += reward_weight * float(nr["penalty"][sel].mean())
        return s
    return score


@torch.no_grad()
def explain(model: V5Net, feats_all: np.ndarray, beat_all: np.ndarray, notes, dev, phrase_s: float = 5.0) -> None:
    """Per phrase: the source selector's mean weight of every input group, and the causes of the notes."""
    from osu_mapper.hybrid.dataset import _crop_rows

    W = model.cfg.window_frames
    names = list(model.cfg.groups) + ["beat"]
    F_ = feats_all.shape[0]
    step = int(phrase_s * 100)
    print("[why] per phrase: what the network listens to most | what its notes follow")
    for s in range(0, F_, step):
        w0 = max(0, s - (W - step) // 2)
        f = torch.from_numpy(_crop_rows(feats_all, w0, W).astype(np.float32))[None].to(dev)
        b = torch.from_numpy(_crop_rows(beat_all, w0, W).astype(np.float32))[None].to(dev)
        enc = model.encode(f, b)
        a, z = s - w0, s - w0 + step
        w = enc["source_weights"][0, a:z].mean(0).cpu().numpy()
        top = np.argsort(-w)[:3]
        ns = [n for n in notes if s * 10 <= n.start < (s + step) * 10]
        causes = []
        for t in sorted({n.start for n in ns}):
            fr = t // 10 - w0
            if 0 <= fr < W:
                causes.append(int(enc["cause_logits"][0, fr].argmax()))
        cs = np.bincount(causes, minlength=len(CAUSES)) / max(1, len(causes))
        print(f"  {s / 100:6.1f}s  listens: " + ", ".join(f"{names[i]} {100 * w[i]:.0f}%" for i in top)
              + f" | {len(ns):3d} notes follow: " + (", ".join(f"{CAUSES[k]} {100 * cs[k]:.0f}%" for k in np.argsort(-cs)[:3] if cs[k] > 0) or "-"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--sr", nargs="*", default=["4"])
    ap.add_argument("--checkpoint", "-c", default="checkpoints/v5/v5_best.pt")
    ap.add_argument("--critic", default=None, help="critic.pt (default: next to the checkpoint, if it exists)")
    ap.add_argument("--search", type=int, default=1, help="continuations sampled per phrase; the best is kept")
    ap.add_argument("--critic_weight", type=float, default=1.0)
    ap.add_argument("--reward_weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--cfg", type=float, default=1.0, help="classifier-free guidance on the star rating / style")
    ap.add_argument("--output", "-o", default="output_beatmaps/v5_map.osz")
    ap.add_argument("--title", default=None)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--creator", default="AIManiaMapper v5")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--od", type=float, default=8.0)
    ap.add_argument("--hp", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--why", action="store_true", help="print what the network listened to and what the notes follow")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    dev = resolve_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = V5Config.from_dict(ck["config"])
    model = V5Net(cfg)
    model.load_state_dict(ck.get("ema") or ck["model"])
    model.to(dev).eval()
    from osu_mapper.hybrid.audio_features import FeatureConfig
    fcfg = FeatureConfig.from_dict(ck["feature_config"])

    work = Path(tempfile.mkdtemp(prefix="v5_gen_"))
    try:
        timing, meta = [], {}
        if args.template:
            audio_path, timing, meta = read_template(Path(args.template), work)
        elif args.audio and Path(args.audio).suffix.lower() in (".osz", ".osu"):
            audio_path, timing, meta = read_template(Path(args.audio), work)
        elif args.audio and Path(args.audio).suffix.lower() in AUDIO_EXTS:
            audio_path = Path(args.audio)
        else:
            sys.exit("give an audio file or --template")
        if args.bpm is not None:
            timing = [tp for tp in timing if not tp.uninherited] + [RawTimingPoint(float(args.offset or 0.0), 60000.0 / args.bpm, 4, 0, 0, 70, True, 0)]
        t0 = time.time()
        wave = load_audio(audio_path, fcfg.sample_rate)
        inputs = prepare_song_inputs(wave, fcfg, timing or None, keep_reps=True, device="cuda" if dev.type == "cuda" else "cpu")
        if not any(tp.uninherited for tp in timing):
            timing = estimated_timing_points(inputs.reps or {}, fcfg.frame_ms) + timing
            print(f"[timing] estimated red line: {60000.0 / timing[0].beat_len:.2f} BPM, offset {timing[0].time:.0f} ms "
                  "(use --bpm/--offset or a template for exact timing)")
        feats = inputs.feats
        att = stem_attacks(feats, fcfg.slices())
        feats_all = np.concatenate([feats, att], axis=1) if att is not None else feats
        beat_all = beat_features(timing, 0, feats.shape[0], fcfg.frame_ms)
        print(f"[audio] {audio_path.name}: {inputs.duration_ms / 1000:.1f}s, features in {time.time() - t0:.1f}s")

        critic = None
        cpath = Path(args.critic) if args.critic else Path(args.checkpoint).with_name("critic.pt")
        if args.search > 1 and cpath.exists():
            from osu_mapper.v5.critic import Critic
            critic = Critic.load(cpath, dev)
            print(f"[search] {args.search} continuations per phrase, critic {cpath}")
        elif args.search > 1:
            print(f"[search] {args.search} continuations per phrase, no critic ({cpath} not found): per-note reward only")
        lib = None
        try:
            from osu_mapper.hybrid.motifs import LanePredictor, load_default_motifs
            lib = load_default_motifs(None, log=lambda *a: None)
        except Exception:
            lib = None
        spec = fcfg.slices().get("spectral")
        power = feats[:, spec[0]] if spec else None

        from osu_mapper.pipeline.packager import BeatmapMetadata, BeatmapPackager
        title = args.title or meta.get("Title") or audio_path.stem
        artist = args.artist or meta.get("Artist") or "Unknown Artist"
        items = []
        targets = [float(x) for s in args.sr for x in str(s).split(",") if x.strip()]
        for target in targets:
            t1 = time.time()
            score_fn = None
            if args.search > 1:
                lp = LanePredictor.cached(lib, target) if lib is not None else None
                score_fn = make_scorer(critic, feats_all, beat_all, timing, target, lp, power, args.critic_weight,
                                       args.reward_weight)
            notes = generate_song(model, feats_all, beat_all, timing, target, candidates=args.search, score_fn=score_fn,
                                  temperature=args.temperature, top_p=args.top_p, cfg_scale=args.cfg, seed=args.seed,
                                  device=dev)
            sr_val = star_rating_of(notes, timing, args.od)
            version = f"v5 {target:.2f}*"
            print(f"[{version}] {len(notes)} notes ({sum(n.ln for n in notes)} LN), star rating {sr_val:.2f} "
                  f"(asked {target:.2f}) in {time.time() - t1:.1f}s")
            if args.why:
                explain(model, feats_all, beat_all, notes, dev)
            m = ManiaMap(notes=notes, timing_points=timing, overall_difficulty=args.od, hp_drain=args.hp,
                         metadata={"Title": title, "Artist": artist, "Creator": args.creator, "Version": version,
                                   "AudioFilename": audio_path.name})
            items.append((BeatmapMetadata(title=title, artist=artist, creator=args.creator, version=version,
                                          audio_filename=audio_path.name), map_to_osu_text(m)))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = BeatmapPackager.package_multi_diff_osz(items, str(audio_path), args.output)
        print(f"[done] {out}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
