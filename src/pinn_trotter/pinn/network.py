"""PINN network: Fourier Feature MLP for quantum state evolution."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    """Maps scalar t to sin/cos Fourier features: γ(t) ∈ R^{2m}.

    B_k ~ N(0, σ²) is sampled once and frozen (not trained).
    γ(t) = [sin(2π B_1 t), cos(2π B_1 t), ..., sin(2π B_m t), cos(2π B_m t)]
    """

    def __init__(self, m: int, sigma: float) -> None:
        super().__init__()
        self.m = m
        B = torch.randn(m) * sigma
        self.register_buffer("B", B)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,) → (..., 2m)
        t_flat = t.unsqueeze(-1)  # (..., 1)
        angles = 2 * np.pi * t_flat * self.B  # (..., m)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    @property
    def out_dim(self) -> int:
        return 2 * self.m


class PINNNetwork(nn.Module):
    """Physics-Informed Neural Network for quantum state evolution.

    Architecture:
        t ∈ R  →  Fourier Features (2m)  →  3×[Linear(d→d) + LayerNorm + Tanh]
               →  Linear(d → 2·2^n)  →  reshape (2^n, 2)

    The output (2^n, 2) represents [Re(ψ), Im(ψ)] for each basis state.

    Args:
        n_qubits: Number of qubits. Hilbert space dimension = 2^n_qubits.
        fourier_m: Number of Fourier feature pairs (output dim = 2*fourier_m).
        hidden_dim: MLP hidden dimension.
        hamiltonian_norm: ‖H‖ used to set Fourier σ = ‖H‖/(2π).
    """

    def __init__(
        self,
        n_qubits: int,
        fourier_m: int = 256,
        hidden_dim: int = 512,
        hamiltonian_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits

        sigma = hamiltonian_norm / (2 * np.pi)
        self.embedding = FourierFeatureEmbedding(fourier_m, sigma)

        in_dim = self.embedding.out_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2 * self.dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return ψ(t) as real-imaginary tensor.

        Args:
            t: Scalar or batch of time values, shape (...,).

        Returns:
            Tensor of shape (..., 2^n, 2) where [..., :, 0] = Re(ψ), [..., :, 1] = Im(ψ).
        """
        feats = self.embedding(t)           # (..., 2m)
        out = self.mlp(feats)               # (..., 2·dim)
        shape = t.shape + (self.dim, 2)
        return out.view(shape)

    def as_complex(self, t: torch.Tensor) -> torch.Tensor:
        """Return ψ(t) as complex tensor of shape (..., 2^n)."""
        ri = self.forward(t)  # (..., dim, 2)
        return torch.view_as_complex(ri.contiguous())

    def normalization_penalty(self, t: torch.Tensor) -> torch.Tensor:
        """L2 penalty encouraging ‖ψ(t)‖ = 1 at given time points.

        Args:
            t: Shape (N,) batch of time values.

        Returns:
            Scalar mean-squared deviation from unit norm.
        """
        psi = self.as_complex(t)  # (N, dim)
        norms_sq = (psi.abs() ** 2).sum(dim=-1)  # (N,)
        return ((norms_sq - 1.0) ** 2).mean()
