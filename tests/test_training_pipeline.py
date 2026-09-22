"""
Unit tests for the training pipeline, models, and feature extraction.
"""

import unittest
import torch
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
)


class TestTrainingPipeline(unittest.TestCase):
    def test_token_encoding_decoding(self):
        """Verify 16-chord token bijection for 4K mania."""
        for token in range(16):
            k0, k1, k2, k3 = token_to_keys(token)
            re_token = keys_to_token(k0, k1, k2, k3)
            self.assertEqual(token, re_token)

    def test_mel_filterbank_and_spectrogram(self):
        """Verify Mel filterbank generation and spectrogram computation."""
        fb = create_mel_filterbank(sr=16000, n_fft=512, n_mels=80)
        self.assertEqual(fb.shape, (80, 257))

        dummy_audio = torch.randn(1, 16000)  # 1 sec
        mel = compute_mel_spectrogram(dummy_audio, sr=16000, n_mels=80, filterbank=fb)
        self.assertEqual(mel.shape[-1], 80)
        self.assertGreater(mel.shape[0], 0)

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


if __name__ == "__main__":
    unittest.main()
