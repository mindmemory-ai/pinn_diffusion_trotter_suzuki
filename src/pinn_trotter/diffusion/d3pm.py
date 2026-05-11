"""D3PM training loss for discrete diffusion."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix


def d3pm_loss(
    model_output: torch.Tensor,
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    transition_matrix: UniformTransitionMatrix,
) -> torch.Tensor:
    """D3PM ELBO loss: -log p_θ(x_0|x_t) + KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)).

    Args:
        model_output: Predicted x_0 logits from denoising model, shape (B, M, K).
        x_0:          True grouping labels, shape (B, M), integers in [0, K-1].
        x_t:          Noisy labels at timestep t, shape (B, M).
        t:            Diffusion timesteps, shape (B,), integers in [0, T-1].
        transition_matrix: UniformTransitionMatrix instance.

    Returns:
        Scalar mean loss.
    """
    B, M, K = model_output.shape

    # --- Reconstruction term: -log p_θ(x_0 | x_t) ---
    log_probs = F.log_softmax(model_output, dim=-1)  # (B, M, K)
    nll = F.nll_loss(
        log_probs.reshape(B * M, K),
        x_0.reshape(B * M),
        reduction="mean",
    )

    # --- KL term: KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)) ---
    # Posterior q(x_{t-1}|x_t, x_0): using true x_0
    x_0_oh = F.one_hot(x_0, num_classes=K).float()
    x_0_logits_true = torch.log(x_0_oh.clamp(min=1e-8))
    q_posterior_log = transition_matrix.compute_posterior_logits(x_t, x_0_logits_true, t)

    # Predicted posterior p_θ(x_{t-1}|x_t): using model's x_0 distribution
    p_posterior_log = transition_matrix.compute_posterior_logits(x_t, model_output, t)

    # KL(q || p) = Σ_k q_k * (log q_k - log p_k)
    q_probs = torch.softmax(q_posterior_log, dim=-1).clamp(min=1e-8)
    p_log = F.log_softmax(p_posterior_log, dim=-1).clamp(min=math.log(1e-8))
    kl = (q_probs * (torch.log(q_probs) - p_log)).sum(dim=-1).mean()

    # At t=0, only use reconstruction loss
    t_is_zero = (t == 0).float().mean()
    loss = (1.0 - t_is_zero) * (nll + kl) + t_is_zero * nll
    return loss


def d3pm_reverse_step(
    x_t: torch.Tensor,
    model_output: torch.Tensor,
    t: torch.Tensor,
    transition_matrix: UniformTransitionMatrix,
) -> torch.Tensor:
    """Sample x_{t-1} ~ p_θ(x_{t-1} | x_t) using predicted x_0 logits.

    Args:
        x_t:          Noisy labels, shape (B, M).
        model_output: Predicted x_0 logits, shape (B, M, K).
        t:            Current timestep, shape (B,).
        transition_matrix: UniformTransitionMatrix instance.

    Returns:
        x_{t-1} samples, shape (B, M).
    """
    B, M = x_t.shape
    K = transition_matrix.K

    logits = transition_matrix.compute_posterior_logits(x_t, model_output, t)  # (B, M, K)
    probs = torch.softmax(logits, dim=-1).clamp(min=1e-10)

    flat = probs.reshape(B * M, K)
    samples = torch.multinomial(flat, num_samples=1).squeeze(1)
    return samples.reshape(B, M)
