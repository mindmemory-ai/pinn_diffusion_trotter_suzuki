"""PINNTrainer: training loop with early stopping and warm-start support."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pinn_trotter.pinn.loss import compute_pinn_loss
from pinn_trotter.pinn.network import PINNNetwork

log = logging.getLogger(__name__)


class PINNTrainer:
    """Trains a PINNNetwork to fit the Schrödinger equation for a given Hamiltonian.

    Args:
        pinn: PINNNetwork to train.
        H_matrix: Hamiltonian matrix, shape (dim, dim), complex128.
        psi_0: Initial state vector, shape (dim,), complex128.
        t_total: Total evolution time.
        n_colloc: Number of collocation time points sampled per step.
        lr: Adam learning rate.
        max_steps: Maximum number of gradient steps.
        early_stop_patience: Steps over which to check relative L_pde change.
        early_stop_tol: Relative L_pde change threshold for early stopping.
        loss_weights: Optional dict with keys 'ic', 'pde', 'circuit', 'norm'.
        device: Torch device string or object.
    """

    def __init__(
        self,
        pinn: PINNNetwork,
        H_matrix: torch.Tensor,
        psi_0: torch.Tensor,
        t_total: float,
        n_colloc: int = 64,
        lr: float = 5e-4,
        max_steps: int = 2000,
        early_stop_patience: int = 10,
        early_stop_tol: float = 1e-5,
        loss_weights: Optional[dict[str, float]] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.pinn = pinn
        self.t_total = t_total
        self.n_colloc = n_colloc
        self.max_steps = max_steps
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol
        self.loss_weights = loss_weights

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.pinn = pinn.to(self.device)

        if not H_matrix.is_complex():
            H_matrix = H_matrix.to(dtype=torch.complex128)
        self.H = H_matrix.to(self.device)

        if not psi_0.is_complex():
            psi_0 = psi_0.to(dtype=torch.complex128)
        self.psi_0 = psi_0.to(self.device)

        self.optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=50)

        # Circuit checkpoints: filled via set_circuit_targets()
        self._t_circuit: Optional[torch.Tensor] = None
        self._psi_circuit: Optional[torch.Tensor] = None

    def set_circuit_targets(
        self,
        t_circuit: torch.Tensor,
        psi_circuit: torch.Tensor,
    ) -> None:
        """Supply Trotter circuit reference states for L_circuit term.

        Args:
            t_circuit: Shape (K,) checkpoint times.
            psi_circuit: Shape (K, dim) complex target states.
        """
        self._t_circuit = t_circuit.to(self.device)
        if not psi_circuit.is_complex():
            psi_circuit = psi_circuit.to(dtype=torch.complex128)
        self._psi_circuit = psi_circuit.to(self.device)

    def train(self) -> dict[str, list[float]]:
        """Run training loop.

        Returns:
            History dict with keys 'total', 'ic', 'pde', 'circuit', 'norm',
            each a list of loss values (one per step).
        """
        self.pinn.train()
        history: dict[str, list[float]] = {
            k: [] for k in ['total', 'ic', 'pde', 'circuit', 'norm']
        }
        pde_window: list[float] = []

        for step in range(self.max_steps):
            # Sample collocation points uniformly in [0, t_total]
            t_c = torch.rand(self.n_colloc, device=self.device) * self.t_total

            self.optimizer.zero_grad()
            total, losses = compute_pinn_loss(
                self.pinn,
                self.H,
                self.psi_0,
                t_c,
                self._t_circuit,
                self._psi_circuit,
                self.loss_weights,
            )
            total.backward()
            nn.utils.clip_grad_norm_(self.pinn.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            history['total'].append(float(total))
            for k in ['ic', 'pde', 'circuit', 'norm']:
                history[k].append(float(losses[k]))

            # Early stopping on relative L_pde change
            pde_val = float(losses['pde'])
            pde_window.append(pde_val)
            if len(pde_window) > self.early_stop_patience:
                pde_window.pop(0)
            if len(pde_window) == self.early_stop_patience:
                rel_change = abs(pde_window[-1] - pde_window[0]) / (abs(pde_window[0]) + 1e-12)
                if rel_change < self.early_stop_tol:
                    log.info("Early stopping at step %d (rel L_pde change=%.2e)", step, rel_change)
                    break

            if step % 100 == 0:
                log.debug(
                    "step=%d total=%.4e ic=%.4e pde=%.4e",
                    step, float(total), float(losses['ic']), float(losses['pde']),
                )

        return history

    def warm_start(self, state_dict: dict) -> None:
        """Load a previously trained PINN state for warm starting.

        Args:
            state_dict: Output of pinn.state_dict() from a prior run.
        """
        self.pinn.load_state_dict(state_dict)
        log.info("Warm-started PINN from provided state dict.")
