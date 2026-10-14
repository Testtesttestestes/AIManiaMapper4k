#!/usr/bin/env python3
"""
Shows what a trained hybrid checkpoint expects as input and how far it got.

    python inspect_hybrid_checkpoint.py                                   # checkpoints/hybrid/hybrid_best.pt
    python inspect_hybrid_checkpoint.py checkpoints/hybrid/hybrid_last.pt --data dataset_hybrid

Prints the audio representations the network reads (built-in and external neural encoders), the window
length, model size, training progress, the calibrated thresholds and the last validation metrics. With
--data it also checks that the prepared dataset was built with the same feature config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from osu_mapper.hybrid.audio_features import FeatureConfig
from osu_mapper.hybrid.model import HybridModelConfig, HybridParamNet, count_parameters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--data", default=None, help="dataset folder to compare the feature config with")
    args = ap.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        sys.exit(f"'{path}' not found")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "hybrid-checkpoint-v1":
        sys.exit(f"'{path}' is not a hybrid checkpoint (format={ck.get('format')})")
    fcfg = FeatureConfig.from_dict(ck["feature_config"])
    mcfg = HybridModelConfig.from_dict(ck["model_config"])
    model = HybridParamNet(mcfg)
    a = ck.get("args", {})

    print(f"checkpoint      : {path}")
    print(f"trained         : epoch {ck.get('epoch')}, step {ck.get('step')} of {ck.get('total_steps')}")
    print(f"model           : d_model={mcfg.d_model}, conv={mcfg.conv_layers}, attn={mcfg.attn_layers}, "
          f"{count_parameters(model):,} parameters (preset at start: {a.get('preset', '?')})")
    print(f"window          : {ck.get('crop_frames')} frames = {ck.get('crop_frames', 0) * fcfg.frame_ms / 1000:.2f} s, "
          f"audio {fcfg.sample_rate} Hz, frame {fcfg.frame_ms} ms")
    print("audio inputs    :")
    dims = fcfg.dims()
    for name in fcfg.representations:
        print(f"   built-in  {name:<14} {dims[name]:>4}-d")
    if fcfg.external:
        for name in fcfg.external:
            print(f"   EXTERNAL  {name:<14} {dims[name]:>4}-d   (neural encoder; needed at generation time too)")
    else:
        print("   external neural encoders: NONE (EnCodec / MERT / Demucs were not used)")
    print(f"   total input  {fcfg.total_dim()}-d per 10 ms frame (+13-d beat grid, star rating, style)")
    print(f"thresholds      : event {ck.get('decode_threshold')}, long note {ck.get('ln_threshold', 0.5)}")
    vs = ck.get("val_summary") or {}
    if vs:
        def g(k: str) -> str:
            v = vs.get(k)
            return "n/a" if v is None else f"{v:.3f}"
        print("last validation : event F1 @0ms {} @2ms {} @16ms {} | timing MAE {} ms | chord acc {} | LN F1 {} | SR error {} | density x{}".format(
            g("event_f1@0"), g("event_f1@2"), g("event_f1@16"), g("timing_mae_ms"), g("chord_size_acc"),
            g("ln_f1_at_best_ln_threshold"), g("sr_mae"), g("density_ratio")))
    hist = ck.get("history") or []
    if hist:
        print("history (F1@16ms per epoch): " + " ".join(f"{h.get('map_event_f1@16', float('nan')):.3f}" for h in hist))

    if args.data:
        idx = Path(args.data) / "index.json"
        data_cfg = FeatureConfig.from_dict(json.loads(idx.read_text(encoding="utf-8"))["feature_config"]).to_dict()
        same = data_cfg == fcfg.to_dict()
        print(f"dataset         : {idx} -> feature config {'MATCHES' if same else 'DIFFERS'}"
              + ("" if same else f" (dataset external={data_cfg.get('external')}, checkpoint external={list(fcfg.external)})"))


if __name__ == "__main__":
    main()
