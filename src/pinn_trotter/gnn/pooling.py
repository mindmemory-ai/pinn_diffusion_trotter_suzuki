"""Attention pooling for graph-level readout."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Soft attention pooling: c = Σ_i α_i h_i, α_i = softmax(MLP(h_i)).

    Args:
        in_dim: Node embedding dimension.
        out_dim: Output graph-level vector dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1),
        )
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool node embeddings to graph-level vector.

        Args:
            x: Node embeddings, shape (N, in_dim).
            batch: Optional batch assignment vector, shape (N,).
                   If None, all nodes are treated as one graph.

        Returns:
            Graph-level vectors, shape (B, out_dim) or (1, out_dim).
        """
        scores = self.score_mlp(x)  # (N, 1)

        if batch is None:
            # Single graph
            alpha = torch.softmax(scores, dim=0)  # (N, 1)
            pooled = (alpha * x).sum(dim=0, keepdim=True)  # (1, in_dim)
        else:
            # Batched graphs: scatter softmax per graph
            pooled = _batched_attention_pool(x, scores, batch)  # (B, in_dim)

        return self.proj(pooled)  # (B, out_dim)


def _batched_attention_pool(
    x: torch.Tensor,
    scores: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Vectorized scatter-based attention pooling for batched graphs."""
    try:
        from torch_scatter import scatter_softmax, scatter_add
        s = scores.squeeze(-1)  # (N,)
        alpha = scatter_softmax(s, batch, dim=0)  # (N,) per-graph softmax
        B = int(batch.max().item()) + 1
        return scatter_add(alpha.unsqueeze(1) * x, batch, dim=0, out=torch.zeros(B, x.size(1), device=x.device, dtype=x.dtype))
    except ImportError:
        pass

    # Pure-PyTorch fallback: vectorized scatter without Python loops.
    B = int(batch.max().item()) + 1
    N, D = x.shape
    s = scores.squeeze(-1)  # (N,)

    # Per-graph softmax via scatter: shift by per-graph max for numerical stability.
    g_max = torch.full((B,), float("-inf"), device=x.device, dtype=s.dtype)
    g_max.scatter_reduce_(0, batch, s, reduce="amax", include_self=True)
    s_shifted = s - g_max[batch]
    exp_s = torch.exp(s_shifted)
    sum_exp = torch.zeros(B, device=x.device, dtype=s.dtype)
    sum_exp.scatter_add_(0, batch, exp_s)
    alpha = (exp_s / sum_exp[batch]).unsqueeze(1)  # (N, 1)

    out = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(1).expand(N, D), alpha * x)
    return out
