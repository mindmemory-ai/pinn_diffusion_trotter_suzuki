"""Mixed-space denoising model: D3PM (grouping+orders) + DDPM (time steps).

Architecture (see spec §4.3):
  Inputs (all noisy):
    grouping_noisy:    (B, M)       integer class labels
    time_steps_noisy:  (B, K)       continuous values
    orders_noisy:      (B, K, 3)    one-hot Suzuki orders
    t_diff:            (B,)         diffusion timestep integer
    condition:         (B, 512)     GNN condition vector (zeros for CFG)

  Condition encoding:
    condition → Linear(512→256) → SiLU
    t_diff    → sinusoidal_embedding(256) → Linear(256→256) → SiLU
    fusion    → element-wise add → (B, 256)

  Grouping branch  (D3PM):
    grouping_onehot (B,M,K) + fused cond → Transformer(4 layers) → Linear→softmax

  Time-step branch (DDPM):
    time_steps (B,K) + fused cond → MLP(3 layers) → noise prediction (B,K)

  Order branch (D3PM):
    orders (B,K,3) + fused cond → Transformer(2 layers) → Linear→softmax

  Joint loss:
    L = L_D3PM_grouping + 0.5 * L_DDPM_timesteps + 0.3 * L_D3PM_orders
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn_trotter.diffusion.d3pm import d3pm_loss, d3pm_reverse_step
from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal timestep embedding.

    Args:
        t:   Integer or float timesteps, shape (B,).
        dim: Output dimension (must be even).

    Returns:
        Embeddings, shape (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Transformer block (lightweight, used for grouping/order branches)
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(x2))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# MixedDiffusionModel
# ---------------------------------------------------------------------------

class MixedDiffusionModel(nn.Module):
    """Joint denoising model for Trotter strategy (grouping + time steps + orders).

    Args:
        max_groups:       K, maximum number of groups.
        n_terms:          M, number of Pauli terms (for grouping labels).
        condition_dim:    GNN output dimension (512 default).
        fused_dim:        Internal fused condition dimension (256 default).
        time_embed_dim:   Sinusoidal timestep embedding dimension.
        grouping_layers:  Transformer layers for grouping branch.
        order_layers:     Transformer layers for order branch.
        ts_mlp_layers:    MLP hidden layers for time-step branch.
        dropout:          Dropout rate.
        p_cond_drop:      CFG training dropout probability.
    """

    def __init__(
        self,
        max_groups: int = 8,
        n_terms: int = 8,
        condition_dim: int = 512,
        fused_dim: int = 256,
        time_embed_dim: int = 256,
        grouping_layers: int = 4,
        order_layers: int = 2,
        ts_mlp_layers: int = 3,
        dropout: float = 0.1,
        p_cond_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_groups = max_groups
        self.n_terms = n_terms
        self.p_cond_drop = p_cond_drop
        K = max_groups

        # --- Condition encoding ---
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, fused_dim),
            nn.SiLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, fused_dim),
            nn.SiLU(),
        )
        self.time_embed_dim = time_embed_dim

        # --- Grouping branch (D3PM) ---
        # Input: grouping one-hot (B, M, K) + condition injection
        g_dim = K + fused_dim
        self.grouping_transformer = nn.Sequential(
            *[_TransformerBlock(g_dim, n_heads=max(1, g_dim // 32), dropout=dropout)
              for _ in range(grouping_layers)]
        )
        self.grouping_head = nn.Linear(g_dim, K)

        # --- Time-step branch (DDPM) ---
        # Input: (B, K) + (B, fused_dim) → concatenated (B, K+fused_dim)
        ts_in = K + fused_dim
        ts_layers_list: list[nn.Module] = []
        d = ts_in
        hidden = 512
        for i in range(ts_mlp_layers):
            ts_layers_list += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        ts_layers_list.append(nn.Linear(d, K))
        self.timestep_mlp = nn.Sequential(*ts_layers_list)

        # --- Order branch (D3PM) ---
        # Input: orders one-hot (B, K, 3) + condition injection
        o_dim = 3 + fused_dim
        _o_heads = max(1, o_dim // 32)
        while o_dim % _o_heads != 0 and _o_heads > 1:
            _o_heads -= 1
        self.order_transformer = nn.Sequential(
            *[_TransformerBlock(o_dim, n_heads=_o_heads, dropout=dropout)
              for _ in range(order_layers)]
        )
        self.order_head = nn.Linear(o_dim, 3)

    def _fuse_condition(
        self,
        condition: torch.Tensor,
        t_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse GNN condition and diffusion timestep into a (B, fused_dim) vector."""
        c = self.cond_proj(condition)
        te = sinusoidal_embedding(t_diff, self.time_embed_dim)
        tp = self.time_proj(te)
        return c + tp  # (B, fused_dim)

    def forward(
        self,
        grouping_noisy: torch.Tensor,
        time_steps_noisy: torch.Tensor,
        orders_noisy: torch.Tensor,
        t_diff: torch.Tensor,
        condition: torch.Tensor,
        drop_condition: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict denoised components.

        Args:
            grouping_noisy:   (B, M) integer labels.
            time_steps_noisy: (B, K) continuous.
            orders_noisy:     (B, K, 3) one-hot.
            t_diff:           (B,) diffusion timestep integers.
            condition:        (B, 512) GNN condition (zeros → unconditional).
            drop_condition:   If True, zero out condition (CFG inference).

        Returns:
            (grouping_logits, ts_noise_pred, order_logits)
              grouping_logits: (B, M, K)
              ts_noise_pred:   (B, K)
              order_logits:    (B, K, 3)
        """
        B = grouping_noisy.shape[0]
        K = self.max_groups

        if drop_condition:
            condition = torch.zeros_like(condition)

        fused = self._fuse_condition(condition, t_diff)  # (B, fused_dim)

        # --- Grouping branch ---
        g_oh = F.one_hot(grouping_noisy, num_classes=K).float()  # (B, M, K)
        fused_g = fused.unsqueeze(1).expand(-1, self.n_terms, -1)  # (B, M, fused_dim)
        g_in = torch.cat([g_oh, fused_g], dim=-1)  # (B, M, K+fused_dim)
        g_out = self.grouping_transformer(g_in)       # (B, M, K+fused_dim)
        grouping_logits = self.grouping_head(g_out)   # (B, M, K)

        # --- Time-step branch ---
        fused_ts = torch.cat([time_steps_noisy, fused], dim=-1)  # (B, K+fused_dim)
        ts_noise_pred = self.timestep_mlp(fused_ts)               # (B, K)

        # --- Order branch ---
        orders_f = orders_noisy.float()               # (B, K, 3)
        fused_o = fused.unsqueeze(1).expand(-1, K, -1)  # (B, K, fused_dim)
        o_in = torch.cat([orders_f, fused_o], dim=-1)   # (B, K, 3+fused_dim)
        o_out = self.order_transformer(o_in)             # (B, K, 3+fused_dim)
        order_logits = self.order_head(o_out)            # (B, K, 3)

        return grouping_logits, ts_noise_pred, order_logits

    def compute_loss(
        self,
        grouping_logits: torch.Tensor,
        ts_noise_pred: torch.Tensor,
        order_logits: torch.Tensor,
        x0_grouping: torch.Tensor,
        x_t_grouping: torch.Tensor,
        t_diff: torch.Tensor,
        ts_noise_true: torch.Tensor,
        x0_orders: torch.Tensor,
        x_t_orders: torch.Tensor,
        transition_matrix: UniformTransitionMatrix,
        order_transition_matrix: UniformTransitionMatrix,
        ddpm: ContinuousDDPM,
        lambda_ts: float = 0.5,
        lambda_ord: float = 0.3,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute joint loss.

        Returns:
            (total_loss, loss_dict) with keys 'grouping', 'timestep', 'order'.
        """
        l_group = d3pm_loss(grouping_logits, x0_grouping, x_t_grouping, t_diff, transition_matrix)
        l_ts = ddpm.ddpm_loss(ts_noise_pred, ts_noise_true)
        l_ord = d3pm_loss(order_logits, x0_orders, x_t_orders, t_diff, order_transition_matrix)

        total = l_group + lambda_ts * l_ts + lambda_ord * l_ord
        return total, {"grouping": l_group, "timestep": l_ts, "order": l_ord}


# ---------------------------------------------------------------------------
# EMA wrapper
# ---------------------------------------------------------------------------

class EMAWrapper:
    """Exponential moving average of model parameters.

    Args:
        model:  The model to track.
        decay:  EMA decay factor (0.9999 recommended).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def get_model(self) -> nn.Module:
        return self.shadow


# ---------------------------------------------------------------------------
# Guided sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_sample(
    model: MixedDiffusionModel,
    condition: torch.Tensor,
    n_terms: int,
    max_groups: int,
    transition_matrix: UniformTransitionMatrix,
    order_transition_matrix: UniformTransitionMatrix,
    ddpm: ContinuousDDPM,
    guidance_scale: float = 2.0,
    n_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classifier-free guided reverse diffusion sampling.

    Args:
        model:             MixedDiffusionModel (eval mode).
        condition:         GNN condition vector, shape (1, 512) or (B, 512).
        n_terms:           M, number of Pauli terms.
        max_groups:        K, max groups.
        transition_matrix: For grouping labels.
        order_transition_matrix: For order labels.
        ddpm:              ContinuousDDPM for time steps.
        guidance_scale:    w_1 for CFG interpolation.
        n_steps:           Number of reverse steps (default: T from ddpm).
        device:            Computation device.

    Returns:
        (grouping, time_steps, orders) at t=0:
            grouping:    (B, M) integer labels
            time_steps:  (B, K) continuous
            orders:      (B, K) integer indices 0/1/2 → Suzuki order 1/2/4
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    T = ddpm.T if n_steps is None else n_steps
    B = condition.shape[0]
    K = max_groups
    M = n_terms

    # Initialize from noise
    g_t = torch.randint(0, K, (B, M), device=device)
    ts_t = torch.randn(B, K, device=device)
    ord_t = torch.randint(0, 3, (B, K), device=device)

    condition = condition.to(device)
    transition_matrix.to(device)
    order_transition_matrix.to(device)
    ddpm.to(device)

    for t_idx in reversed(range(T)):
        t_batch = torch.full((B,), t_idx, dtype=torch.long, device=device)
        ord_oh = F.one_hot(ord_t, num_classes=3).float()

        # Conditional prediction
        g_logits, ts_noise, o_logits = model(
            g_t, ts_t, ord_oh, t_batch, condition, drop_condition=False
        )
        # Unconditional prediction (CFG)
        g_logits_unc, ts_noise_unc, o_logits_unc = model(
            g_t, ts_t, ord_oh, t_batch, condition, drop_condition=True
        )

        # CFG interpolation
        g_logits_guided = g_logits_unc + guidance_scale * (g_logits - g_logits_unc)
        ts_noise_guided = ts_noise_unc + guidance_scale * (ts_noise - ts_noise_unc)
        o_logits_guided = o_logits_unc + guidance_scale * (o_logits - o_logits_unc)

        # Reverse steps
        g_t = d3pm_reverse_step(g_t, g_logits_guided, t_batch, transition_matrix)
        ts_t = ddpm.reverse_step(ts_t, ts_noise_guided, t_batch)
        ord_t = d3pm_reverse_step(
            ord_t, o_logits_guided, t_batch, order_transition_matrix
        )

    return g_t, ts_t, ord_t
