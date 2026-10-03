"""
Unified ManiaDiffusion Model for osu!mania 4K Beatmap Generation.

Integrates:
- Multimodal Audio Tokenizer & Feature Extraction Pipeline
- Deep Acoustic Conditioning Backbone with Model 3 Checkpoint Weight Transfer
- 1D Diffusion UNet
- DDPM / DDIM Scheduler with Classifier-Free Guidance (CFG)
- Beatmap Piano-Roll Tensor <-> ManiaHitObject Conversion
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from osu_mapper.dataset.audio_tokenizers import (
    AudioConditioningEncoder,
    extract_multimodal_audio_features,
)
from osu_mapper.models.diffusion_scheduler import DiffusionScheduler
from osu_mapper.models.diffusion_unet import ManiaDiffusionUNet


if TORCH_AVAILABLE and torch is not None:
    no_grad = torch.no_grad
else:
    def no_grad():
        def decorator(func):
            return func
        return decorator


if TORCH_AVAILABLE and nn is not None:
    BaseModule = nn.Module
else:
    class BaseModule:  # type: ignore
        pass


class ManiaDiffusion(BaseModule):
    """
    Complete Diffusion Model for generating osu!mania 4K beatmaps in a single non-autoregressive pass.
    """
    def __init__(
        self,
        audio_in_dim: int = 128,
        audio_d_model: int = 256,
        unet_base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        in_channels: int = 12,  # 4 tap + 4 LN start + 4 LN sustain
        cfg_dropout_prob: float = 0.15,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cfg_dropout_prob = cfg_dropout_prob

        # 1. Acoustic Conditioning Backbone
        self.audio_encoder = AudioConditioningEncoder(
            in_dim=audio_in_dim,
            d_model=audio_d_model,
            num_layers=3,
            dropout=0.1,
        )

        # 2. 1D Diffusion UNet
        self.unet = ManiaDiffusionUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=unet_base_channels,
            channel_multipliers=channel_multipliers,
            audio_dim=audio_d_model,
            time_dim=time_dim,
            num_res_blocks=2,
        )

        # 3. Noise Scheduler
        self.scheduler = DiffusionScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="cosine",
        )

        # Learnable null conditioning for Classifier-Free Guidance (CFG)
        self.null_audio_context = nn.Parameter(torch.zeros(1, 1, audio_d_model))
        nn.init.normal_(self.null_audio_context, std=0.02)

    def load_from_model3_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Transfers learned audio representation weights from an existing Model 3 checkpoint.
        Ensures the training time spent on acoustic understanding is not wasted.
        """
        p = Path(checkpoint_path)
        if not p.exists():
            return False

        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            state_dict = None
            for k in ("model_state", "model_state_dict", "state_dict"):
                if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
                    state_dict = ckpt[k]
                    break
            if state_dict is None and isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt

            if state_dict is None:
                return False

            transferred_count = 0
            audio_state = {}
            for name, param in state_dict.items():
                if name.startswith("audio_encoder."):
                    sub_name = name[len("audio_encoder."):]
                    audio_state[sub_name] = param

            if audio_state:
                # Load compatible weights into our audio encoder
                missing, unexpected = self.audio_encoder.load_state_dict(audio_state, strict=False)
                transferred_count = len(audio_state) - len(missing)
                print(f"  ✓ Transferred {transferred_count} acoustic weights from Model 3 checkpoint: {p.name}")
                return True
        except Exception as e:
            print(f"  [Notice] Could not transfer weights from {p.name}: {e}")
        return False

    def forward(
        self,
        beatmap_tensor: torch.Tensor,
        audio_mel: torch.Tensor,
        grid_metrics: Optional[torch.Tensor],
        star_rating: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Training forward pass: computes MSE noise prediction loss.
        Args:
            beatmap_tensor: [B, 12, T] Ground truth beatmap piano-roll tensor (values in [-1, 1]).
            audio_mel: [B, T, 80] Log-mel spectrogram.
            grid_metrics: [B, T, 8] Musical grid metrics.
            star_rating: [B, 1] Target difficulty.
        """
        B, C, T = beatmap_tensor.shape
        device = beatmap_tensor.device

        # 1. Multi-tokenizer acoustic extraction & projection
        acoustic_tokens = extract_multimodal_audio_features(audio_mel, grid_metrics)
        audio_context = self.audio_encoder(acoustic_tokens)  # [B, T, audio_d_model]

        # 2. Classifier-Free Guidance (CFG) conditioning dropout during training
        if self.training and self.cfg_dropout_prob > 0.0:
            mask = (torch.rand(B, 1, 1, device=device) < self.cfg_dropout_prob)
            audio_context = torch.where(mask, self.null_audio_context.expand(B, T, -1), audio_context)

        # 3. Sample random diffusion timesteps for each batch element
        if timesteps is None:
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device).long()

        # 4. Sample Gaussian noise and corrupt the clean beatmap tensor
        if noise is None:
            noise = torch.randn_like(beatmap_tensor)
        noisy_beatmap = self.scheduler.add_noise(beatmap_tensor, noise, timesteps)

        # 5. Predict the noise using 1D UNet
        predicted_noise = self.unet(noisy_beatmap, timesteps, audio_context, star_rating)

        # 6. Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        loss_dict = {
            "loss_diffusion": loss.item(),
            "l1_error": F.l1_loss(predicted_noise, noise).item(),
        }
        return loss, loss_dict

    @no_grad()
    def sample(
        self,
        audio_mel: torch.Tensor,
        grid_metrics: Optional[torch.Tensor],
        star_rating: Union[float, torch.Tensor],
        num_inference_steps: int = 30,
        cfg_scale: float = 3.0,
        eta: float = 0.0,
        progress_callback: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Generates a full beatmap piano-roll tensor [B, 12, T] from pure Gaussian noise using fast DDIM.
        Allows user to choose number of steps (e.g. 20, 30, 50) and CFG scale, exactly like Stable Diffusion 1.5!
        """
        B, T, n_mels = audio_mel.shape
        device = audio_mel.device

        if isinstance(star_rating, (int, float)):
            sr_tensor = torch.tensor([[float(star_rating)]], device=device, dtype=torch.float32).expand(B, 1)
        else:
            sr_tensor = star_rating.view(B, 1).to(device)

        # 1. Multi-tokenizer acoustic features
        acoustic_tokens = extract_multimodal_audio_features(audio_mel, grid_metrics)
        audio_context = self.audio_encoder(acoustic_tokens)  # [B, T, audio_d_model]
        uncond_audio_context = self.null_audio_context.expand(B, T, -1)

        # 2. Initialize pure standard Gaussian noise x_T ~ N(0, I)
        shape = (B, self.in_channels, T)
        sample = torch.randn(shape, device=device)

        # 3. Get DDIM inference timesteps
        timesteps = self.scheduler.get_timesteps(num_inference_steps, device)

        # 4. DDIM Denoising Loop
        for i, t in enumerate(timesteps):
            t_int = int(t.item())
            prev_t_int = int(timesteps[i + 1].item()) if (i + 1) < len(timesteps) else -1
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

            if cfg_scale > 1.0:
                # Batched CFG: evaluate conditioned and unconditioned together
                combined_sample = torch.cat([sample, sample], dim=0)
                combined_t = torch.cat([t_batch, t_batch], dim=0)
                combined_audio = torch.cat([audio_context, uncond_audio_context], dim=0)
                combined_sr = torch.cat([sr_tensor, sr_tensor], dim=0)

                noise_pred = self.unet(combined_sample, combined_t, combined_audio, combined_sr)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)

                # Classifier-Free Guidance formula: eps_hat = eps_uncond + s * (eps_cond - eps_uncond)
                guided_noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                guided_noise_pred = self.unet(sample, t_batch, audio_context, sr_tensor)

            # Reverse DDIM Step
            sample = self.scheduler.step_ddim(
                model_output=guided_noise_pred,
                timestep=t_int,
                prev_timestep=prev_t_int,
                sample=sample,
                eta=eta,
            )

            if progress_callback:
                progress_callback(i + 1, num_inference_steps)

        # Denoised beatmap tensor in [-1, 1]
        return sample


def beatmap_tensor_to_hit_objects(
    denoised_tensor: torch.Tensor,
    ticks: List[MusicalGridTick],
    duration_ms: float,
    beat_len_ms: float,
    tap_threshold: float = 0.05,
    ln_threshold: float = 0.15,
    min_ln_ms: Optional[float] = None,
) -> List[ManiaHitObject]:
    """
    Decodes the 12-channel continuous diffusion tensor [12, T] into validated discrete ManiaHitObjects.
    Channels 0..3: Tap activations
    Channels 4..7: LN Start activations
    Channels 8..11: LN Sustain holds
    """
    if min_ln_ms is None:
        min_ln_ms = max(110.0, beat_len_ms * 0.35)

    arr = denoised_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]  # [12, T]

    num_channels, num_ticks = arr.shape
    num_ticks = min(num_ticks, len(ticks))
    hit_objects: List[ManiaHitObject] = []

    # Track active LN starts per column
    active_ln_starts: Dict[int, Optional[int]] = {0: None, 1: None, 2: None, 3: None}
    last_note_ms: Dict[int, int] = {0: -9999, 1: -9999, 2: -9999, 3: -9999}

    for t_idx in range(num_ticks):
        t_obj = ticks[t_idx]
        cur_ms = int(round(t_obj.time_ms))
        if cur_ms >= duration_ms - 50:
            break

        for c in range(4):
            tap_val = float(arr[c, t_idx])
            ln_start_val = float(arr[c + 4, t_idx]) if num_channels >= 8 else -1.0
            ln_hold_val = float(arr[c + 8, t_idx]) if num_channels >= 12 else -1.0

            # 1. Check if column was holding an LN: should it be released?
            if active_ln_starts[c] is not None:
                hold_duration = cur_ms - active_ln_starts[c]
                # Release condition: hold activation dropped, or another note starts, or held > 4 beats
                should_release = (ln_hold_val < (ln_threshold * 0.5)) or (tap_val > tap_threshold) or (hold_duration >= beat_len_ms * 4.0)
                if should_release:
                    start_t = active_ln_starts[c]
                    if hold_duration >= min_ln_ms:
                        end_t = max(start_t + int(round(min_ln_ms)), cur_ms - 25)
                        hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=end_t, is_ln=True))
                    else:
                        hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=start_t, is_ln=False))
                    active_ln_starts[c] = None
                    last_note_ms[c] = cur_ms

            # 2. Check if a new hit object starts at this tick
            if active_ln_starts[c] is None:
                is_tap = (tap_val > tap_threshold)
                is_ln = (ln_start_val > ln_threshold)
                gap_ok = (cur_ms - last_note_ms[c]) >= 40  # Physical minimum gap per lane

                if is_ln and gap_ok:
                    active_ln_starts[c] = cur_ms
                elif is_tap and gap_ok:
                    hit_objects.append(ManiaHitObject(column=c, start_time=cur_ms, end_time=cur_ms, is_ln=False))
                    last_note_ms[c] = cur_ms

    # Close any lingering sliders
    final_ms = min(int(round(duration_ms)), int(round(ticks[num_ticks - 1].time_ms)))
    for c in range(4):
        if active_ln_starts[c] is not None:
            start_t = active_ln_starts[c]
            hold_len = final_ms - start_t
            if hold_len >= min_ln_ms:
                hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=final_ms, is_ln=True))
            else:
                hit_objects.append(ManiaHitObject(column=c, start_time=start_t, end_time=start_t, is_ln=False))
            active_ln_starts[c] = None

    hit_objects.sort(key=lambda o: (o.start_time, o.column))
    return hit_objects
