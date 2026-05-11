"""PINN-based and exact fidelity evaluators for the closed-loop optimizer.

Two concrete evaluators are provided:

* ExactFidelityEvaluator  — uses exact quantum state evolution (scipy expm / RK45).
  Correct for any Hamiltonian; practical for n_qubits ≤ 8.  No pre-trained model
  needed.  Use this as the default for PoC experiments.

* PINNEvaluator  — uses a trained PINNNetwork as a cheap proxy for the exact
  evolution.  The PINN must have been trained on the same Hamiltonian that will
  be passed at evaluation time.  Use this when n_qubits > 8 makes exact
  simulation too expensive.

Both expose the same callable interface expected by ClosedLoopOptimizer:

    fidelity: float = evaluator(H_graph, grouping_t, timesteps_t, orders_t)

where:
    H_graph      : HamiltonianGraph instance
    grouping_t   : (1, M)    int64 tensor  — group index per Pauli term
    timesteps_t  : (1, K)    float32 tensor — time steps normalised to sum = 1
    orders_t     : (1, K, 3) float32 tensor — one-hot Trotter orders {1, 2, 4}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared decoding helper
# ---------------------------------------------------------------------------

def _decode_strategy(H_graph, grouping_t: torch.Tensor, timesteps_t: torch.Tensor,
                     orders_t: torch.Tensor, t_total: float):
    """Decode diffusion-model output tensors → TrotterStrategy.

    Args:
        H_graph:     HamiltonianGraph (used for n_qubits and n_terms).
        grouping_t:  (1, M) or (M,) integer tensor.
        timesteps_t: (1, K) or (K,) float tensor, normalised sum = 1.
        orders_t:    (1, K, 3) or (K, 3) float tensor, one-hot.
        t_total:     Total evolution time (seconds).

    Returns:
        TrotterStrategy with time_steps rescaled to sum = t_total.
    """
    from pinn_trotter.strategy.encoding import tensor_to_strategy

    g = grouping_t.squeeze(0) if grouping_t.dim() == 2 else grouping_t
    ts = timesteps_t.squeeze(0) if timesteps_t.dim() == 2 else timesteps_t
    o = orders_t.squeeze(0) if orders_t.dim() in (2, 3) else orders_t
    # guided_sample emits integer order ids (0/1/2). Convert to one-hot to
    # keep tensor_to_strategy's expected `(K, 3)` interface.
    if o.dim() == 1:
        o = torch.nn.functional.one_hot(o.long(), num_classes=3).float()

    return tensor_to_strategy(g, o, ts, H_graph.n_qubits, t_total)


# ---------------------------------------------------------------------------
# ExactFidelityEvaluator
# ---------------------------------------------------------------------------

class ExactFidelityEvaluator:
    """Exact fidelity evaluator using exact quantum state evolution.

    Computes F = |⟨ψ_exact(T) | ψ_Trotter(T)⟩|² via scipy.linalg.expm (n ≤ 8)
    or scipy.integrate.solve_ivp / RK45 (n > 8).

    This is the ground-truth evaluator for the PoC (4-qubit TFIM).  It is
    accurate for any Hamiltonian passed at call time.

    Args:
        t_total:  Total evolution time T.
        psi_0:    Initial state as a numpy array of shape (2^n,).  Defaults to
                  the computational-basis state |0⟩^⊗n when None (determined
                  from H_graph at first call).
    """

    def __init__(
        self,
        t_total: float,
        psi_0: Optional[np.ndarray] = None,
    ) -> None:
        self.t_total = t_total
        self._psi_0 = psi_0

    def _default_psi0(self, n_qubits: int) -> np.ndarray:
        """Return computational-basis |0⟩^⊗n."""
        psi = np.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
        return psi

    def __call__(
        self,
        H_graph,
        grouping_t: torch.Tensor,
        timesteps_t: torch.Tensor,
        orders_t: torch.Tensor,
    ) -> float:
        from pinn_trotter.data.generator import compute_exact_fidelity_from_hamiltonian

        psi_0 = self._psi_0 if self._psi_0 is not None else self._default_psi0(H_graph.n_qubits)

        try:
            strategy = _decode_strategy(H_graph, grouping_t, timesteps_t, orders_t, self.t_total)
            fidelity, _ = compute_exact_fidelity_from_hamiltonian(H_graph, strategy, psi_0)
            return float(np.clip(fidelity, 0.0, 1.0))
        except Exception as exc:
            log.warning("ExactFidelityEvaluator failed: %s", exc)
            return 0.0

    def circuit_depth(
        self,
        H_graph,
        grouping_t: torch.Tensor,
        timesteps_t: torch.Tensor,
        orders_t: torch.Tensor,
    ) -> int:
        """Return circuit depth estimate for the decoded strategy."""
        try:
            strategy = _decode_strategy(H_graph, grouping_t, timesteps_t, orders_t, self.t_total)
            return strategy.circuit_depth_estimate()
        except Exception:
            return int(grouping_t.max().item() + 1) * H_graph.n_terms


# ---------------------------------------------------------------------------
# PINNEvaluator
# ---------------------------------------------------------------------------

class PINNEvaluator:
    """PINN-proxy fidelity evaluator.

    Evaluates F = |⟨ψ_PINN(T) | ψ_Trotter(T)⟩|² where ψ_PINN is the
    prediction of a pre-trained PINNNetwork and ψ_Trotter is the numerically
    simulated Trotter evolution.

    Use this when exact evolution is too expensive (n_qubits > 8).  The PINN
    must have been trained on the same physical system that will be evaluated;
    passing a different Hamiltonian gives meaningless fidelity values.

    Args:
        pinn:        Trained PINNNetwork (already on the correct device).
        t_total:     Total evolution time T.
        psi_0:       Initial state as numpy array, shape (2^n,).  Defaults to |0⟩^⊗n.
        device:      Torch device for PINN inference.
        fallback_exact: If True and evaluation fails, fall back to exact simulation.
    """

    def __init__(
        self,
        pinn: "torch.nn.Module",
        t_total: float,
        psi_0: Optional[np.ndarray] = None,
        device: Optional[str | torch.device] = None,
        fallback_exact: bool = True,
    ) -> None:
        self.pinn = pinn
        self.t_total = t_total
        self._psi_0_np = psi_0
        self.fallback_exact = fallback_exact

        if device is None:
            device = next(pinn.parameters()).device
        self.device = torch.device(device)

        # Cache PINN prediction at t_total (constant across calls)
        self._psi_pinn: Optional[torch.Tensor] = None

    def _get_pinn_state(self) -> torch.Tensor:
        """Lazy-compute and cache ψ_PINN(T)."""
        if self._psi_pinn is None:
            self.pinn.eval()
            with torch.no_grad():
                t = torch.tensor([self.t_total], dtype=torch.float32, device=self.device)
                self._psi_pinn = self.pinn.as_complex(t)[0].to(torch.complex64)  # (dim,)
        return self._psi_pinn

    def _default_psi0(self, n_qubits: int) -> np.ndarray:
        psi = np.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
        return psi

    def __call__(
        self,
        H_graph,
        grouping_t: torch.Tensor,
        timesteps_t: torch.Tensor,
        orders_t: torch.Tensor,
    ) -> float:
        from pinn_trotter.pinn.fidelity import fidelity_from_states
        from pinn_trotter.data.generator import apply_trotter_from_hamiltonian

        psi_0 = self._psi_0_np if self._psi_0_np is not None else self._default_psi0(H_graph.n_qubits)

        try:
            strategy = _decode_strategy(H_graph, grouping_t, timesteps_t, orders_t, self.t_total)

            # Trotter-evolved state
            psi_trotter_np = apply_trotter_from_hamiltonian(H_graph, strategy, psi_0)
            psi_trotter = torch.tensor(psi_trotter_np, dtype=torch.complex64, device=self.device)

            # PINN-predicted state (cached)
            psi_pinn = self._get_pinn_state()

            # F = |⟨ψ_PINN | ψ_Trotter⟩|²
            f = fidelity_from_states(
                psi_pinn.unsqueeze(0), psi_trotter.unsqueeze(0)
            )[0].item()
            return float(np.clip(f, 0.0, 1.0))

        except Exception as exc:
            log.warning("PINNEvaluator failed: %s", exc)
            if self.fallback_exact:
                log.debug("Falling back to ExactFidelityEvaluator")
                return ExactFidelityEvaluator(self.t_total, psi_0)(
                    H_graph, grouping_t, timesteps_t, orders_t
                )
            return 0.0

    def circuit_depth(
        self,
        H_graph,
        grouping_t: torch.Tensor,
        timesteps_t: torch.Tensor,
        orders_t: torch.Tensor,
    ) -> int:
        try:
            strategy = _decode_strategy(H_graph, grouping_t, timesteps_t, orders_t, self.t_total)
            return strategy.circuit_depth_estimate()
        except Exception:
            return int(grouping_t.max().item() + 1) * H_graph.n_terms


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_evaluator(
    t_total: float,
    n_qubits: int,
    pinn: Optional["torch.nn.Module"] = None,
    pinn_ckpt: Optional[str | Path] = None,
    psi_0: Optional[np.ndarray] = None,
    device: Optional[str | torch.device] = None,
    exact_threshold: int = 8,
) -> ExactFidelityEvaluator | PINNEvaluator:
    """Build the appropriate evaluator for the given system size.

    For n_qubits <= exact_threshold: returns ExactFidelityEvaluator.
    For n_qubits >  exact_threshold: returns PINNEvaluator (requires pinn or pinn_ckpt).

    Args:
        t_total:         Total evolution time.
        n_qubits:        Number of qubits.
        pinn:            Pre-built PINNNetwork (optional).
        pinn_ckpt:       Path to a saved PINN state dict (optional, used when pinn=None).
        psi_0:           Initial state (optional, defaults to |0⟩^⊗n).
        device:          Torch device.
        exact_threshold: Use exact evaluator for systems up to this size.

    Returns:
        An ExactFidelityEvaluator or PINNEvaluator instance.
    """
    if n_qubits <= exact_threshold:
        log.info(
            "make_evaluator: n_qubits=%d ≤ %d → ExactFidelityEvaluator",
            n_qubits, exact_threshold,
        )
        return ExactFidelityEvaluator(t_total=t_total, psi_0=psi_0)

    if pinn is None and pinn_ckpt is not None:
        from pinn_trotter.pinn.network import PINNNetwork
        _dev = torch.device(device) if device else torch.device("cpu")
        state = torch.load(pinn_ckpt, map_location=_dev, weights_only=True)
        pinn = PINNNetwork(n_qubits=n_qubits)
        pinn.load_state_dict(state)
        pinn = pinn.to(_dev)

    if pinn is None:
        log.warning(
            "make_evaluator: n_qubits=%d > %d but no PINN provided "
            "— falling back to ExactFidelityEvaluator (may be slow).",
            n_qubits, exact_threshold,
        )
        return ExactFidelityEvaluator(t_total=t_total, psi_0=psi_0)

    log.info(
        "make_evaluator: n_qubits=%d > %d → PINNEvaluator",
        n_qubits, exact_threshold,
    )
    return PINNEvaluator(pinn=pinn, t_total=t_total, psi_0=psi_0, device=device)
