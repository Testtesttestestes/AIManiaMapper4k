"""
Unit tests for the training pipeline, models, and feature extraction.
"""

import unittest
import os
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from osu_mapper.models import (
        SyncBPMOffsetModel,
        DynamicsSVModel,
        ManiaPatternTransformer,
        DifficultyEvaluatorModel,
        TransformerConfig,
        keys_to_token,
        token_to_keys,
    )
    from osu_mapper.dataset.feature_extractor import (
        create_mel_filterbank,
        compute_mel_spectrogram,
        find_ffmpeg,
    )
else:
    from osu_mapper.dataset.feature_extractor import find_ffmpeg


class TestTrainingPipeline(unittest.TestCase):
    def test_find_ffmpeg_detection(self):
        """Verify find_ffmpeg correctly returns None or a valid binary without crashing."""
        res = find_ffmpeg()
        if res is not None:
            self.assertTrue(isinstance(res, str))
            self.assertTrue(len(res) > 0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_token_encoding_decoding(self):
        """Verify 16-chord token bijection for 4K mania."""
        for token in range(16):
            k0, k1, k2, k3 = token_to_keys(token)
            re_token = keys_to_token(k0, k1, k2, k3)
            self.assertEqual(token, re_token)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_mel_filterbank_and_spectrogram(self):
        """Verify Mel filterbank generation and spectrogram computation."""
        fb = create_mel_filterbank(sr=16000, n_fft=512, n_mels=80)
        self.assertEqual(fb.shape, (80, 257))

        dummy_audio = torch.randn(1, 16000)  # 1 sec
        mel = compute_mel_spectrogram(dummy_audio, sr=16000, n_mels=80, filterbank=fb)
        self.assertEqual(mel.shape[-1], 80)
        self.assertGreater(mel.shape[0], 0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_all_four_models_forward(self):
        """Test forward passes for all 4 AI models."""
        B, T = 2, 64
        mel = torch.randn(B, T, 80)
        sr = torch.tensor([[3.5], [6.8]])
        tokens = torch.randint(0, 16, (B, T))

        # Model 1: Sync
        m1 = SyncBPMOffsetModel(audio_dim=80)
        bpm, offset = m1(mel)
        self.assertEqual(bpm.shape, (B, 1))
        self.assertEqual(offset.shape, (B, 1))

        # Model 2: Dynamics
        m2 = DynamicsSVModel(audio_dim=80)
        sv, trig = m2(mel)
        self.assertEqual(sv.shape, (B, T, 1))
        self.assertEqual(trig.shape, (B, T, 1))

        # Model 3: Transformer
        cfg = TransformerConfig(audio_dim=80, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
        m3 = ManiaPatternTransformer(cfg)
        chord_logits, ln_logits = m3(mel, star_rating=sr, prev_tokens=tokens)
        self.assertEqual(chord_logits.shape, (B, T, 16))
        self.assertEqual(ln_logits.shape, (B, T, 4))

        # Model 4: Evaluator
        m4 = DifficultyEvaluatorModel(num_classes=16, token_dim=32, hidden_dim=64)
        pred_sr = m4(tokens)
        self.assertEqual(pred_sr.shape, (B, 1))


    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in test environment")
    def test_slice_beatmap_timing_sources(self):
        """Test slice_beatmap_into_musical_chunks with both 'map' (ground truth) and 'model1' (override)."""
        from osu_mapper.dataset.feature_extractor import (
            ParsedBeatmapDetails,
            slice_beatmap_into_musical_chunks,
        )

        dummy_details = ParsedBeatmapDetails(
            osu_file_name="test.osu",
            version="4K Insane",
            mode=3,
            circle_size=4.0,
            overall_difficulty=8.0,
            audio_filename="audio.mp3",
            bpm=150.0,
            offset_ms=100.0,
            star_rating=5.0,
            timing_points=[(100.0, 400.0, True)],
            hit_objects=[(0, 100, 100, False), (1, 500, 500, False)],
        )

        mel = torch.randn(1000, 80)
        tokens = torch.zeros(1000, dtype=torch.int64)
        ln_active = torch.zeros(1000, 4, dtype=torch.float32)
        sv_curve = torch.ones(1000, dtype=torch.float32)

        # 1. Ground Truth from Map (.osu)
        map_chunks = slice_beatmap_into_musical_chunks(
            mel, tokens, ln_active, sv_curve, dummy_details,
            chunk_size=256, align_mode="beat", timing_source="map"
        )
        self.assertGreater(len(map_chunks), 0)
        self.assertEqual(map_chunks[0]["timing_source"], "map")
        self.assertEqual(float(map_chunks[0]["bpm"].item()), 150.0)
        self.assertEqual(float(map_chunks[0]["offset_ms"].item()), 100.0)

        # 2. Inferred from Model 1 (AI timing override)
        model1_chunks = slice_beatmap_into_musical_chunks(
            mel, tokens, ln_active, sv_curve, dummy_details,
            chunk_size=256, align_mode="beat", timing_source="model1",
            timing_override=(175.0, 30.0)
        )
        self.assertGreater(len(model1_chunks), 0)
        self.assertEqual(model1_chunks[0]["timing_source"], "model1")
        self.assertEqual(float(model1_chunks[0]["bpm"].item()), 175.0)
        self.assertEqual(float(model1_chunks[0]["offset_ms"].item()), 30.0)


if __name__ == "__main__":
    unittest.main()
