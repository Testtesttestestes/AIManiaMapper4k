"""
Tests of the hybrid (neural -> parameters -> universal algorithm) pipeline.
"""

import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

from osu_mapper.hybrid.beatmap import ManiaMap, Note, RawTimingPoint, map_to_osu_text, parse_osu_mania
from osu_mapper.hybrid.metrics import match_times, per_ms_report
from osu_mapper.hybrid.params import (
    LANE_TABLE,
    N_LANE_CODES,
    N_OPS,
    ParamTrack,
    apply_lane_code,
    canonical_lane_code,
    ln_bucket_to_length,
    ln_length_to_bucket,
)
from osu_mapper.hybrid.renderer import DecodeKnobs, GridSnapper, decode_outputs, oracle_outputs, render_track, resolve_lane_conflicts
from osu_mapper.hybrid.solver import STYLE_DIM, compute_style, solve_map, verify_roundtrip

REPO = Path(__file__).resolve().parents[1]


def random_map(rng: random.Random, n: int, degenerate: bool = False) -> ManiaMap:
    notes = []
    t = rng.randint(0, 3000)
    for _ in range(n):
        t += rng.choice([0, 1, 2, 3, 5, 9, 10, 11, 23, 37, 75, 150, 300, 2500])
        for c in rng.sample(range(4), rng.choice([1, 1, 1, 2, 2, 3, 4])):
            s = t + (rng.choice([0, 1, 2, 7]) if rng.random() < 0.15 else 0)
            if rng.random() < 0.25:
                notes.append(Note(s, c, s + rng.choice([1, 9, 30, 120, 480, 5000]), True))
            else:
                notes.append(Note(s, c, s, False))
    if degenerate:
        notes += [Note(-7, 2, -7, False), Note(50, 1, 50, True), Note(60, 0, 61, False), Note(70, 3, 70, False), Note(72, 3, 72, False)]
    tps = [RawTimingPoint(12.5, 333.333, 4, 2, 1, 60, True, 0), RawTimingPoint(4000, -80.0, 4, 2, 0, 50, False, 1)]
    return ManiaMap(notes=notes, timing_points=tps)


class TestUniversalAlgorithm(unittest.TestCase):
    def test_exact_roundtrip_fuzz(self):
        rng = random.Random(1234)
        for i in range(250):
            m = random_map(rng, rng.randint(0, 300), degenerate=(i % 4 == 0))
            ok, track, rendered = verify_roundtrip(m)
            self.assertTrue(ok, f"map {i}: render(solve(map)) != map")
            if i % 4 == 0:
                self.assertGreater(len(track.overflow), 0)

    def test_osu_text_roundtrip(self):
        rng = random.Random(7)
        m = random_map(rng, 200, degenerate=True)
        m.notes = [n for n in m.notes if n.ln or n.end == n.start]  # a tap cannot carry an end time in .osu text
        m2 = parse_osu_mania(map_to_osu_text(m))
        self.assertEqual(sorted(m2.notes), sorted(m.notes))
        self.assertEqual([t.as_tuple() for t in m2.timing_points], [t.as_tuple() for t in m.timing_points])
        ok, _, _ = verify_roundtrip(m2)
        self.assertTrue(ok)

    def test_non_mania_rejected(self):
        text = map_to_osu_text(ManiaMap(notes=[Note(0, 0, 0)])).replace("Mode: 3", "Mode: 0")
        self.assertIsNone(parse_osu_mania(text))
        text7 = map_to_osu_text(ManiaMap(notes=[Note(0, 0, 0)])).replace("CircleSize:4", "CircleSize:7")
        self.assertIsNone(parse_osu_mania(text7))

    def test_lane_codes_are_universal_and_canonical(self):
        for prev in range(16):
            for prev2 in range(16):
                for mask in range(1, 16):
                    code = canonical_lane_code(mask, prev, prev2)
                    self.assertEqual(apply_lane_code(code, prev, prev2), mask)
                    for earlier in range(min(code, N_OPS)):
                        self.assertNotEqual(LANE_TABLE[prev, prev2, earlier], mask)
        self.assertEqual(apply_lane_code(0, 0b0010, 0), 0b0010)            # SAME = jack
        self.assertEqual(apply_lane_code(1, 0b0010, 0b0100), 0b0100)       # ALT = trill
        self.assertEqual(apply_lane_code(2, 0b1000, 0), 0)                 # SHIFT_R off the edge is invalid
        self.assertEqual(apply_lane_code(4, 0b1000, 0), 0b0001)            # ROT_R wraps
        self.assertEqual(apply_lane_code(9, 0b0011, 0), 0b1100)            # COMPLEMENT
        self.assertEqual(N_LANE_CODES, N_OPS + 15)

    def test_ln_buckets_exact(self):
        L = np.arange(1, 70000, 7)
        b, r = ln_length_to_bucket(L)
        self.assertTrue(np.array_equal(ln_bucket_to_length(b, r)[L < 65536], L[L < 65536]))

    def test_params_json_file_roundtrip(self):
        """hybrid_params.py: map -> JSON parameters -> algorithm reading only the JSON -> identical map."""
        import json
        sys.path.insert(0, str(REPO))
        from hybrid_params import compare, json_to_map, track_to_json

        m = random_map(random.Random(11), 200, degenerate=True)
        m.notes = [n for n in m.notes if n.ln or n.end == n.start]
        text = json.dumps(track_to_json(solve_map(m), m, "test"))
        rendered = parse_osu_mania(map_to_osu_text(json_to_map(json.loads(text))))
        ok, msg = compare(m, rendered)
        self.assertTrue(ok, msg)

    def test_track_npz_roundtrip(self):
        m = random_map(random.Random(3), 120, degenerate=True)
        track = solve_map(m)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.npz"
            track.save_npz(p)
            t2 = ParamTrack.load_npz(p)
        self.assertEqual(render_track(t2), sorted(m.notes))
        self.assertEqual([t.as_tuple() for t in t2.timing_points], [t.as_tuple() for t in m.timing_points])

    def test_oracle_decode_equals_exact_render(self):
        rng = random.Random(99)
        knobs = DecodeKnobs(merge_ms=0, ln_conflict="truncate", ln_release_gap_ms=0, ln_min_ms=1)
        for _ in range(60):
            clean = resolve_lane_conflicts(random_map(rng, 150).notes, release_gap_ms=0, ln_min_ms=1)
            track = solve_map(clean)
            expected = sorted(set(clean) - set(track.overflow))
            self.assertEqual(decode_outputs(oracle_outputs(track), knobs), expected)

    def test_decoder_respects_held_long_notes(self):
        track = solve_map([Note(0, 0, 2000, True), Note(500, 1, 500, False)])
        out = oracle_outputs(track)
        out.lane_logits[50] = -30.0
        out.lane_logits[50, N_OPS + 0b0001 - 1] = 30.0   # network wants lane 0 while it is held
        out.lane_logits[50, N_OPS + 0b0100 - 1] = 10.0   # second choice lane 2
        notes = decode_outputs(out, DecodeKnobs())
        self.assertIn(Note(500, 2, 500, False), notes)
        self.assertIn(Note(0, 0, 2000, True), notes)

    def test_grid_snapper(self):
        snap = GridSnapper([RawTimingPoint(100.0, 400.0)])
        self.assertEqual(list(snap.snap(np.array([103, 199, 233, 160]), 5)), [100, 200, 233, 160])
        self.assertEqual(list(snap.snap(np.array([233]), 5)), [233])
        self.assertEqual(int(snap.snap(np.array([234]), 5)[0]), 233)  # 1/3 beat after 100 = 233.33

    def test_style_vector(self):
        m = random_map(random.Random(5), 100)
        track = solve_map(m)
        style, known = compute_style(track, m.notes, 4.2, 8.0)
        self.assertEqual(style.shape, (STYLE_DIM,))
        self.assertTrue(np.all(known == 1.0))
        self.assertAlmostEqual(float(style[0]), 0.42, places=5)


class TestPerMsMetrics(unittest.TestCase):
    def test_greedy_matching_is_optimal_on_a_line(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = np.sort(rng.choice(200, size=rng.integers(0, 15), replace=False))
            b = np.sort(rng.choice(200, size=rng.integers(0, 15), replace=False))
            tol = int(rng.integers(0, 6))
            greedy = len(match_times(a, b, tol))
            # brute-force maximum matching via augmenting paths
            adj = [[j for j in range(len(b)) if abs(int(a[i]) - int(b[j])) <= tol] for i in range(len(a))]
            match_b = [-1] * len(b)

            def aug(i, seen):
                for j in adj[i]:
                    if not seen[j]:
                        seen[j] = True
                        if match_b[j] < 0 or aug(match_b[j], seen):
                            match_b[j] = i
                            return True
                return False

            best = sum(aug(i, [False] * len(b)) for i in range(len(a)))
            self.assertEqual(greedy, best)

    def test_report(self):
        ref = [Note(1000, 0, 1000), Note(1500, 1, 1800, True), Note(2000, 2, 2000), Note(2000, 3, 2000)]
        rep = per_ms_report(ref, ref)
        self.assertEqual(rep["per_ms_loss"], 0.0)
        self.assertEqual(rep["exact_match"], 1.0)
        shifted = [Note(n.start + 3, n.col, n.end + 3, n.ln) for n in ref]
        rep = per_ms_report(shifted, ref)
        self.assertEqual(rep["event_f1@2"], 0.0)
        self.assertEqual(rep["event_f1@5"], 1.0)
        self.assertAlmostEqual(rep["timing_mae_ms"], 3.0)
        self.assertEqual(rep["ln_length_mae_ms"], 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestAudioFrontEnd(unittest.TestCase):
    def test_millisecond_alignment_and_blocking(self):
        from osu_mapper.hybrid.audio_features import FeatureConfig, compute_features

        cfg = FeatureConfig(block_frames=700)
        sr = cfg.sample_rate
        x = 0.001 * np.random.default_rng(0).standard_normal(sr * 12).astype(np.float32)
        clicks = [1234, 5003, 7777, 10001]
        n = np.arange(int(0.01 * sr))
        for c in clicks:
            i = int(c * sr / 1000)
            x[i:i + n.size] += 0.8 * np.exp(-n / (0.0015 * sr)) * np.sin(2 * np.pi * 3000 * n / sr)
        a = compute_features(x, cfg, normalize=False)
        b = compute_features(x, FeatureConfig(block_frames=100000), normalize=False)
        for name in a:
            self.assertEqual(a[name].shape, (1200, cfg.dims()[name]))
            np.testing.assert_allclose(a[name], b[name], rtol=1e-4, atol=1e-3, err_msg=name)
        fe = a["fine_env"][:, :10].reshape(-1)  # 1 ms energy envelope
        for c in clicks:
            self.assertEqual(int(np.argmax(fe[c - 8: c + 9])) - 8, 0, f"click at {c} ms")

    def test_external_encoder_plumbing(self):
        """Neural tokenizers (EnCodec/MERT/Demucs) plug in through the registry; tested with a stand-in."""
        from osu_mapper.hybrid.audio_features import FeatureConfig, compute_features
        from osu_mapper.hybrid.external_reps import EXTERNAL_REGISTRY, ExternalRep, register_external

        def energy75(w, sr):
            hop = sr / 75.0
            n = int(np.ceil(w.size / hop))
            e = np.array([np.sqrt((w[int(i * hop):int((i + 1) * hop)] ** 2).mean() + 1e-9) for i in range(n)])
            return np.stack([e, -e], axis=1)

        register_external(ExternalRep("test_energy75", 2, 75.0, energy75, chunk_s=3.0, overlap_s=0.5))
        try:
            cfg = FeatureConfig(external=("test_energy75",))
            sr = cfg.sample_rate
            x = np.zeros(sr * 10, np.float32)
            x[int(sr * 6.0): int(sr * 6.1)] = 0.5
            reps = compute_features(x, cfg, normalize=False)
            self.assertEqual(reps["test_energy75"].shape, (1000, 2))
            hot = np.flatnonzero(reps["test_energy75"][:, 0] > 0.25)
            self.assertTrue(598 <= hot[0] <= 601 and 608 <= hot[-1] <= 611, hot)
            self.assertEqual(cfg.total_dim(), FeatureConfig().total_dim() + 2)
            self.assertEqual(FeatureConfig.from_dict(cfg.to_dict()), cfg)
        finally:
            EXTERNAL_REGISTRY.pop("test_energy75", None)

    def test_beat_features(self):
        from osu_mapper.hybrid.audio_features import beat_features

        bf = beat_features([RawTimingPoint(0.0, 500.0)], 0, 100, 10)
        self.assertEqual(bf.shape, (100, 13))
        self.assertTrue(np.all(bf[:, -1] == 1.0))
        self.assertTrue(np.all(beat_features([], 0, 10, 10) == 0.0))

    def test_tempo_estimate(self):
        from osu_mapper.hybrid.audio_features import FeatureConfig, compute_features, estimate_tempo

        cfg = FeatureConfig()
        sr = cfg.sample_rate
        x = 0.001 * np.random.default_rng(1).standard_normal(sr * 30).astype(np.float32)
        n = np.arange(int(0.05 * sr))
        for k in range(70):
            i = int((300 + k * 400) * sr / 1000)
            x[i:i + n.size] += 0.5 * np.exp(-n / (0.01 * sr)) * np.sin(2 * np.pi * 80 * n / sr)
        bpm, offset = estimate_tempo(compute_features(x, cfg))
        self.assertAlmostEqual(bpm, 150.0, delta=0.5)
        self.assertLess(min(abs(offset - 300) % 400, 400 - abs(offset - 300) % 400), 4)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestHybridNetwork(unittest.TestCase):
    def _model(self):
        from osu_mapper.hybrid.audio_features import FeatureConfig
        from osu_mapper.hybrid.model import HybridParamNet, build_config

        fcfg = FeatureConfig()
        torch.manual_seed(0)
        return fcfg, HybridParamNet(build_config(fcfg.dims(), fcfg.total_dim(), "tiny"))

    def test_forward_backward_and_losses(self):
        from osu_mapper.hybrid.losses import LossWeights, compute_losses, soft_time_targets

        fcfg, model = self._model()
        B, T = 2, 256
        track = solve_map(random_map(random.Random(1), 60))
        crops = [track.crop_arrays(0, T), track.crop_arrays(100, T)]
        batch = {k: torch.from_numpy(np.stack([c[k] for c in crops])) for k in crops[0]}
        batch.update(valid=torch.ones(B, T), style=torch.rand(B, STYLE_DIM), style_mask=torch.ones(B, STYLE_DIM))
        out = model(torch.randn(B, T, fcfg.total_dim()), torch.zeros(B, T, 13), torch.tensor([[3.0], [6.0]]),
                    batch["style"], batch["style_mask"], torch.randn(B, 30, fcfg.total_dim()), None)
        self.assertEqual(out["time_logits"].shape, (B, T, 11))
        self.assertEqual(out["stagger_logits"].shape, (B, T, 4, 10))
        loss, logs = compute_losses(out, batch, LossWeights(), 10, sigma_ms=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        tgt = soft_time_targets(torch.tensor([[0, 3, 0]]), 10, 0.0)
        self.assertEqual(tgt[0, 1].argmax().item(), 3)
        self.assertTrue(torch.allclose(soft_time_targets(torch.tensor([[0, 5, 0]]), 10, 1.0).sum(-1), torch.ones(1, 3)))

    def test_window_inference_is_position_independent(self):
        """The same audio gives the same prediction no matter how long the track is or where it sits."""
        from osu_mapper.hybrid.inference import HybridPredictor, SongInputs

        fcfg, model = self._model()
        W = 256
        pred = HybridPredictor(model, fcfg, W, torch.device("cpu"))
        rng = np.random.default_rng(0)
        D = fcfg.total_dim()
        a = rng.standard_normal((5 * W, D)).astype(np.float32)
        b = rng.standard_normal((9 * W, D)).astype(np.float32)
        coarse = rng.standard_normal((20, D)).astype(np.float32)
        beat = np.zeros((14 * W, 13), np.float32)
        short = pred.predict(SongInputs(a, coarse, beat[: 5 * W], 0), 4.0)
        long_ = pred.predict(SongInputs(np.concatenate([a, b]), coarse, beat, 0), 4.0)
        safe = 5 * W - W  # frames whose windows never reach past the end of `a`
        np.testing.assert_allclose(short.time_probs[:safe], long_.time_probs[:safe], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(short.lane_logits[:safe], long_.lane_logits[:safe], rtol=1e-4, atol=1e-4)

    def test_sixty_minute_track_is_fully_covered(self):
        """A 60-minute input gets predictions for every frame (no truncation) in linear time."""
        from osu_mapper.hybrid.inference import HybridPredictor, SongInputs

        fcfg, model = self._model()
        W = 2048
        pred = HybridPredictor(model, fcfg, W, torch.device("cpu"))
        F = 60 * 60 * 100
        D = fcfg.total_dim()
        rng = np.random.default_rng(1)
        pattern = rng.standard_normal((4 * W, D)).astype(np.float16)
        feats = np.tile(pattern, (F // (4 * W) + 1, 1))[:F].astype(np.float32)
        out = pred.predict(SongInputs(feats, rng.standard_normal((F // 100, D)).astype(np.float32), np.zeros((F, 13), np.float32), F * 10), 4.0, batch_size=16)
        self.assertEqual(out.time_probs.shape[0], F)
        self.assertTrue(np.isfinite(out.time_probs).all())
        self.assertTrue(np.all(np.abs(out.time_probs.sum(axis=1) - 1.0) < 1e-3))
        p = 1.0 - out.time_probs[:, 0]
        period = 4 * W
        middle = slice(10 * period, 11 * period)
        later = slice(40 * period, 41 * period)
        np.testing.assert_allclose(p[middle], p[later], rtol=1e-4, atol=1e-5)  # same audio -> same output, anywhere


def _ffmpeg_available() -> bool:
    try:
        from osu_mapper.dataset.feature_extractor import find_ffmpeg
        return find_ffmpeg() is not None
    except Exception:
        return False


@unittest.skipUnless(TORCH_AVAILABLE and _ffmpeg_available(), "needs PyTorch and FFmpeg")
class TestEndToEndSmoke(unittest.TestCase):
    def test_prepare_train_generate(self):
        from osu_mapper.hybrid.synthetic import make_synthetic_dataset

        work = Path(tempfile.mkdtemp(prefix="hybrid_e2e_"))
        try:
            make_synthetic_dataset(work / "osz", 3, seed=3, duration_s=16)
            env = dict(os.environ, PYTHONPATH=str(REPO))
            run = lambda *a: subprocess.run([sys.executable, *a], cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
            r = run("prepare_hybrid_dataset.py", "--input_dir", str(work / "osz"), "--output_dir", str(work / "ds"))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("exact roundtrip failures: 0", r.stdout)
            r = run("train_hybrid.py", "--data", str(work / "ds"), "--checkpoint_dir", str(work / "ck"), "--preset", "tiny",
                    "--epochs", "1", "--steps_per_epoch", "3", "--batch_size", "2", "--crop_sec", "5.12", "--val_maps", "1",
                    "--val_crops", "2", "--val_fraction", "0.34", "--warmup_steps", "1")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((work / "ck" / "hybrid_best.pt").exists())
            r = run("train_hybrid.py", "--data", str(work / "ds"), "--checkpoint_dir", str(work / "ck"), "--epochs", "1",
                    "--steps_per_epoch", "2", "--batch_size", "2", "--val_maps", "1", "--val_crops", "2", "--val_fraction", "0.34", "--resume")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("[resume]", r.stdout)
            r = run("-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "train_hybrid.py", "--data", str(work / "ds"),
                    "--checkpoint_dir", str(work / "ck_ddp"), "--preset", "tiny", "--epochs", "1", "--steps_per_epoch", "2",
                    "--batch_size", "2", "--crop_sec", "5.12", "--val_maps", "1", "--val_crops", "2", "--val_fraction", "0.34", "--device", "cpu")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("x 2 GPU(s)", r.stdout)
            self.assertTrue((work / "ck_ddp" / "hybrid_last.pt").exists())
            osz = sorted((work / "osz").glob("*.osz"))[0]
            r = run("generate_hybrid.py", "--template", str(osz), "--sr", "2.5,4", "--checkpoint", str(work / "ck" / "hybrid_best.pt"),
                    "--output", str(work / "out.osz"), "--threshold", "0.05")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            import zipfile
            with zipfile.ZipFile(work / "out.osz") as z:
                osus = [n for n in z.namelist() if n.endswith(".osu")]
                self.assertEqual(len(osus), 2)
                for n in osus:
                    self.assertIsNotNone(parse_osu_mania(z.read(n).decode("utf-8")))
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
