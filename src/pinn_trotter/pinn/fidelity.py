"""Fidelity proxy evaluation using the trained PINN."""

from __future__ import annotations

import numpy as np
import torch


def evaluate_fidelity_proxy(
    pinn: "torch.nn.Module",
    psi_target: torch.Tensor,
    t_eval: float,
) -> float:
    """Compute fidelity F = |⟨ψ_target | ψ_pinn(t_eval)⟩|² using PINN.

    This is a differentiable proxy for the true circuit fidelity.

    Args:
        pinn: Trained PINNNetwork.
        psi_target: Target state vector at time t_eval, shape (2^n,), complex.
        t_eval: Time at which to evaluate the PINN.

    Returns:
        Scalar fidelity value in [0, 1].
    """
    pinn.eval()
    with torch.no_grad():
        t = torch.tensor([t_eval], dtype=torch.float32, device=next(pinn.parameters()).device)
        psi_pred = pinn.as_complex(t)[0]  # (dim,)

        if not psi_target.is_complex():
            psi_tgt = psi_target.to(dtype=torch.complex64)
        else:
            psi_tgt = psi_target.to(dtype=torch.complex64, device=psi_pred.device)

        overlap = (psi_tgt.conj() @ psi_pred).abs()
        norm_pred = psi_pred.norm()
        norm_tgt = psi_tgt.norm()
        fidelity = (overlap / (norm_pred * norm_tgt + 1e-12)).item() ** 2

    return float(np.clip(fidelity, 0.0, 1.0))


def fidelity_from_states(psi_pred: torch.Tensor, psi_target: torch.Tensor) -> torch.Tensor:
    """Compute fidelity tensor (differentiable) for use in training loops.

    Args:
        psi_pred:   shape (..., dim), complex.
        psi_target: shape (..., dim), complex.

    Returns:
        Fidelity values, shape (...,).
    """
    overlap = (psi_target.conj() * psi_pred).sum(dim=-1).abs()  # (...,)
    norm_pred = psi_pred.norm(dim=-1)
    norm_tgt = psi_target.norm(dim=-1)
    return (overlap / (norm_pred * norm_tgt + 1e-12)) ** 2
