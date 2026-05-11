"""Fidelity regression head for GNN proxy pre-training (4-B-1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FidelityRegressionHead(nn.Module):
    """Maps GNN condition vector (+ optional strategy features) → predicted fidelity in [0, 1].

    Architecture: Linear → LayerNorm → SiLU → Dropout → Linear → LayerNorm → SiLU → Dropout
                  → Linear → SiLU → Linear → Sigmoid

    Args:
        input_dim: Dimension of input (GNN output_dim, or GNN + strategy features when concatenated).
    """

    def __init__(self, input_dim: int = 512) -> None:
        super().__init__()
        h1 = min(512, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: shape (B, input_dim).

        Returns:
            Predicted fidelity, shape (B, 1).
        """
        return self.net(condition)

    def loss(
        self,
        condition: torch.Tensor,
        target_fidelity: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and target fidelity.

        Args:
            condition:       shape (B, input_dim).
            target_fidelity: shape (B,) or (B, 1), values in [0, 1].

        Returns:
            Scalar MSE loss.
        """
        pred = self.forward(condition).squeeze(-1)
        target = target_fidelity.squeeze(-1).to(pred.device)
        return nn.functional.mse_loss(pred, target)
