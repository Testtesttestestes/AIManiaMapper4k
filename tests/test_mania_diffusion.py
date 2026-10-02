"""
Unit tests for ManiaDiffusion 1D Beatmap Latent Diffusion Architecture.
"""

import unittest
import torch

from osu_mapper.dataset.audio_tokenizers import (
    AudioConditioningEncoder,
    extract_multimodal_audio_features,
)
from osu_mapper.models.diffusion_scheduler import DiffusionScheduler
from osu_mapper.models.diffusion_unet import ManiaDiffusionUNet
from osu_mapper.models.mania_diffusion import (
    ManiaDiffusion,
    beatmap_tensor_to_hit_objects,
)
from osu_mapper.pipeline.timing import MusicalGridTick


def test_multimodal_audio_tokenizers():
    B, T = 2, 64
    audio_mel = torch.randn(B, T, 80)
    grid_metrics = torch.randn(B, T, 8)

    acoustic_tokens = extract_multimodal_audio_features(audio_mel, grid_metrics)
    assert acoustic_tokens.shape == (B, T, 128), f"Expected [2, 64, 128], got {acoustic_tokens.shape}"


def test_audio_conditioning_encoder():
    B, T = 2, 64
    acoustic_tokens = torch.randn(B, T, 128)
    encoder = AudioConditioningEncoder(in_dim=128, d_model=256, num_layers=2)
    context = encoder(acoustic_tokens)
    assert context.shape == (B, T, 256), f"Expected [2, 64, 256], got {context.shape}"


def test_diffusion_scheduler():
    scheduler = DiffusionScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    x_0 = torch.randn(2, 12, 64)
    noise = torch.randn_like(x_0)
    t = torch.tensor([100, 500])

    noisy = scheduler.add_noise(x_0, noise, t)
    assert noisy.shape == x_0.shape

    # Test DDIM step
    denoised_step = scheduler.step_ddim(
        model_output=noise,
        timestep=500,
        prev_timestep=450,
        sample=noisy,
    )
    assert denoised_step.shape == x_0.shape


def test_mania_diffusion_forward_and_sample():
    B, T = 2, 64
    model = ManiaDiffusion(
        audio_in_dim=128,
        audio_d_model=128,
        unet_base_channels=64,
        channel_multipliers=(1, 2),
        time_dim=128,
        in_channels=12,
    )

    beatmap_tensor = torch.randn(B, 12, T)
    audio_mel = torch.randn(B, T, 80)
    grid_metrics = torch.randn(B, T, 8)
    star_rating = torch.tensor([[3.5], [4.5]], dtype=torch.float32)

    # 1. Forward training loss
    loss, loss_dict = model(
        beatmap_tensor=beatmap_tensor,
        audio_mel=audio_mel,
        grid_metrics=grid_metrics,
        star_rating=star_rating,
    )
    assert torch.is_tensor(loss)
    assert loss.item() > 0
    assert "loss_diffusion" in loss_dict

    # 2. Fast DDIM sample (5 steps for test)
    model.eval()
    sample = model.sample(
        audio_mel=audio_mel[:1],
        grid_metrics=grid_metrics[:1],
        star_rating=3.5,
        num_inference_steps=5,
        cfg_scale=2.0,
    )
    assert sample.shape == (1, 12, T)

    # 3. Decoding to HitObjects
    ticks = [
        MusicalGridTick(
            time_ms=float(i * 125),
            bpm=120.0,
            beat_len_ms=500.0,
            meter=4,
            divisor=4,
            divisor_fraction=0.25,
            subdivision_index=i % 4,
            subdivisions_per_beat=4,
            beat_in_measure=i % 4,
            bar_phase=float(i % 16) / 16.0,
            is_downbeat=(i % 16 == 0),
            is_beat_start=(i % 4 == 0),
            is_ternary=False,
        )
        for i in range(T)
    ]
    objs = beatmap_tensor_to_hit_objects(
        denoised_tensor=sample[0],
        ticks=ticks,
        duration_ms=float(T * 125),
        beat_len_ms=500.0,
    )
    assert isinstance(objs, list)
