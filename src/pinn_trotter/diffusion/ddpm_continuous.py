"""Continuous DDPM for time-step allocation (continuous component of strategy)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pinn_trotter.diffusion.transition_matrix import _make_beta_schedule


class ContinuousDDPM:
    """Standard DDPM for a continuous vector (the K-dim time-step allocation).

    Handles the τ part of the strategy: τ ∈ R^K (normalized, sums to 1).

    Args:
        T:             Total diffusion timesteps.
        beta_schedule: 'linear' or 'cosine'.
        beta_start:    Start value.
        beta_end:      End value.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.T = T

        betas = _make_beta_schedule(beta_schedule, T, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        # Posterior variance β̃_t = β_t (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp(min=1e-10)
        )

    def to(self, device: torch.device | str) -> "ContinuousDDPM":
        for attr in [
            "betas", "alphas", "alpha_bar", "alpha_bar_prev",
            "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def forward_sample(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I).

        Args:
            x0: Clean time-step vectors, shape (B, K).
            t:  Integer timesteps, shape (B,).

        Returns:
            (x_t, noise) both shape (B, K).
        """
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)           # (B, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)  # (B, 1)
        x_t = sqrt_ab * x0 + sqrt_one_minus * noise
        return x_t, noise

    def ddpm_loss(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and actual noise.

        Args:
            noise_pred: Predicted noise, shape (B, K).
            noise_true: True noise used in forward_sample, shape (B, K).

        Returns:
            Scalar MSE loss.
        """
        return nn.functional.mse_loss(noise_pred, noise_true)

    def reverse_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t).

        Args:
            x_t:        Noisy tensor, shape (B, K).
            noise_pred: Predicted noise ε_θ, shape (B, K).
            t:          Current timestep, shape (B,).

        Returns:
            x_{t-1}, shape (B, K).
        """
        alpha_t = self.alphas[t].unsqueeze(-1)          # (B, 1)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)   # (B, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)

        # Predicted x_0
        sqrt_recip_alpha = (1.0 / alpha_t.sqrt())
        coeff = self.betas[t].unsqueeze(-1) / sqrt_one_minus
        mean = sqrt_recip_alpha * (x_t - coeff * noise_pred)

        # Variance
        var = self.posterior_variance[t].unsqueeze(-1).clamp(min=1e-8)
        noise = torch.randn_like(x_t)
        # Don't add noise at t=0
        t0_mask = (t == 0).float().unsqueeze(-1)
        return mean + (1.0 - t0_mask) * var.sqrt() * noise
