"""Gradient utilities for policy optimization: Gumbel-Softmax, STE, REINFORCE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """Gumbel-Softmax estimator with optional straight-through hard samples.

    Forward pass: hard one-hot if hard=True (argmax), else soft categorical.
    Backward pass: gradients flow through soft distribution regardless.

    Args:
        logits:      Shape (..., K), unnormalized log-probabilities.
        temperature: τ → 0 approaches argmax; τ → ∞ approaches uniform.
        hard:        If True, return hard one-hot with STE backward.

    Returns:
        Shape (..., K). One-hot (hard=True) or soft (hard=False).
    """
    # Sample Gumbel noise: -log(-log(U)), U ~ Uniform(0, 1)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-20)))
    perturbed = (logits + gumbel_noise) / temperature
    soft = F.softmax(perturbed, dim=-1)

    if hard:
        # Straight-through: forward uses hard one-hot, backward uses soft
        idx = soft.argmax(dim=-1, keepdim=True)
        hard_out = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
        return (hard_out - soft).detach() + soft  # STE
    return soft


def straight_through_estimator(
    x_hard: torch.Tensor,
    x_soft: torch.Tensor,
) -> torch.Tensor:
    """Straight-through estimator wrapper.

    Forward: x_hard (discrete/quantized values).
    Backward: gradient flows through x_soft.

    Args:
        x_hard: Discretized forward value, shape (...).
        x_soft: Soft approximation for gradient, same shape.

    Returns:
        x_hard in forward, x_soft gradient in backward.
    """
    return (x_hard - x_soft).detach() + x_soft


def compute_policy_log_prob(
    grouping_labels: torch.Tensor,
    order_labels: torch.Tensor,
    grouping_logits: torch.Tensor,
    order_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute log p_θ(strategy) for REINFORCE.

    log p = Σ_m log p(grouping_m) + Σ_k log p(order_k)

    Args:
        grouping_labels: Integer labels (B, M).
        order_labels:    Integer labels (B, K), values 0/1/2.
        grouping_logits: Predicted x_0 logits (B, M, K_groups).
        order_logits:    Predicted x_0 logits (B, K, 3).

    Returns:
        Log-probabilities per sample, shape (B,).
    """
    B, M, Kg = grouping_logits.shape
    _, K, Ko = order_logits.shape

    # Grouping: NLL summed over M tokens
    log_p_g = F.log_softmax(grouping_logits, dim=-1)  # (B, M, Kg)
    g_selected = log_p_g.gather(
        -1, grouping_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, M)

    # Orders: NLL summed over K groups
    log_p_o = F.log_softmax(order_logits, dim=-1)  # (B, K, 3)
    o_selected = log_p_o.gather(
        -1, order_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, K)

    return g_selected.sum(dim=-1) + o_selected.sum(dim=-1)  # (B,)
