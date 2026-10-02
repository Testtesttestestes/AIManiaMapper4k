"""
DDPM / DDIM Diffusion Scheduler for ManiaDiffusion.

Implements:
1. Cosine & Linear beta schedules for smooth non-destructive noise injection.
2. Fast DDIM deterministic/stochastic sampling (20-50 steps, matching Stable Diffusion 1.5).
3. Classifier-Free Guidance (CFG) blending.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in Nichol & Dhariwal 2021.
    Prevents abrupt noise injection at small t compared to linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class DiffusionScheduler:
    """
    Handles forward noise addition (q_sample) and reverse DDIM sampling.
    """
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.5,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        if beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_train_timesteps)
        else:
            self.betas = linear_beta_schedule(num_train_timesteps, beta_start, beta_end)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "DiffusionScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise
        """
        device = original_samples.device
        self.to(device)

        # Broadcast over [B, 1, 1]
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step_ddim(
        self,
        model_output: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM reverse step from timestep to prev_timestep.
        eta = 0.0 corresponds to deterministic DDIM sampling (standard in Stable Diffusion).
        """
        device = sample.device
        self.to(device)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=device)

        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        # 1. Predict x_0 from current sample and predicted noise epsilon
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)

        # 2. Compute variance (sigma_t for stochasticity, 0 for eta=0)
        variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * torch.sqrt(torch.clamp(variance, min=0.0))

        # 3. Direction pointing to x_t
        pred_sample_direction = torch.sqrt(torch.clamp(beta_prod_t_prev - std_dev_t ** 2, min=0.0)) * model_output

        # 4. Compute x_{t-prev}
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        if eta > 0.0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample

    def get_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """
        Generates evenly spaced DDIM inference timesteps (e.g. [980, 960, ..., 0]).
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps, device=device) * step_ratio
        timesteps = torch.flip(timesteps, dims=[0])
        return timesteps
