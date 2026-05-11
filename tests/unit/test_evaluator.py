"""Unit tests for PINNEvaluator and ExactFidelityEvaluator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_trotter.benchmarks.hamiltonians import make_tfim
from pinn_trotter.pinn.evaluator import (
    ExactFidelityEvaluator,
    PINNEvaluator,
    _decode_strategy,
    make_evaluator,
)


N_QUBITS = 4
N_TERMS = 2 * N_QUBITS
MAX_GROUPS = 4
T_TOTAL = 1.0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def H_tfim():
    return make_tfim(N_QUBITS, J=1.0, h=0.5, boundary="periodic")


@pytest.fixture(scope="module")
def strategy_tensors():
    """Minimal valid strategy: all terms in group 0, order 1, uniform time step."""
    g = torch.zeros(1, N_TERMS, dtype=torch.long)
    ts = torch.full((1, MAX_GROUPS), 1.0 / MAX_GROUPS)
    o = torch.zeros(1, MAX_GROUPS, 3)
    o[:, :, 0] = 1.0  # all order-1
    return g, ts, o


# ---------------------------------------------------------------------------
# _decode_strategy
# ---------------------------------------------------------------------------

class TestDecodeStrategy:
    def test_returns_valid_strategy(self, H_tfim, strategy_tensors):
        g, ts, o = strategy_tensors
        strat = _decode_strategy(H_tfim, g, ts, o, T_TOTAL)
        assert strat.n_terms == N_TERMS
        assert strat.t_total == pytest.approx(T_TOTAL)

    def test_time_steps_sum_to_t_total(self, H_tfim, strategy_tensors):
        g, ts, o = strategy_tensors
        strat = _decode_strategy(H_tfim, g, ts, o, T_TOTAL)
        assert sum(strat.time_steps) == pytest.approx(T_TOTAL, abs=1e-5)

    def test_accepts_unbatched_tensors(self, H_tfim, strategy_tensors):
        g, ts, o = strategy_tensors
        strat = _decode_strategy(H_tfim, g.squeeze(0), ts.squeeze(0), o.squeeze(0), T_TOTAL)
        assert strat.n_terms == N_TERMS

    def test_accepts_order_indices_from_guided_sample(self, H_tfim, strategy_tensors):
        g, ts, _ = strategy_tensors
        # guided_sample returns (B, K) integer order ids 0/1/2.
        order_idx = torch.zeros(1, MAX_GROUPS, dtype=torch.long)
        strat = _decode_strategy(H_tfim, g, ts, order_idx, T_TOTAL)
        assert strat.n_terms == N_TERMS
        assert all(o in (1, 2, 4) for o in strat.orders)


# ---------------------------------------------------------------------------
# ExactFidelityEvaluator
# ---------------------------------------------------------------------------

class TestExactFidelityEvaluator:
    def test_returns_float_in_range(self, H_tfim, strategy_tensors):
        g, ts, o = strategy_tensors
        ev = ExactFidelityEvaluator(t_total=T_TOTAL)
        f = ev(H_tfim, g, ts, o)
        assert isinstance(f, float)
        assert 0.0 <= f <= 1.0

    def test_fidelity_nonzero(self, H_tfim, strategy_tensors):
        """Short time / simple strategy should give non-negligible fidelity."""
        g, ts, o = strategy_tensors
        ev = ExactFidelityEvaluator(t_total=0.1)
        f = ev(H_tfim, g, ts, o)
        assert f > 0.0

    def test_custom_psi0(self, H_tfim, strategy_tensors):
        g, ts, o = strategy_tensors
        psi_0 = np.zeros(2**N_QUBITS, dtype=complex)
        psi_0[0] = 1.0
        ev = ExactFidelityEvaluator(t_total=T_TOTAL, psi_0=psi_0)
        f = ev(H_tfim, g, ts, o)
        assert 0.0 <= f <= 1.0

    def test_circuit_depth_positive(self, H_tfim, strategy_tensors):
        g, ts, o = strategy_tensors
        ev = ExactFidelityEvaluator(t_total=T_TOTAL)
        d = ev.circuit_depth(H_tfim, g, ts, o)
        assert isinstance(d, int)
        assert d > 0

    def test_short_evolution_high_fidelity(self, H_tfim, strategy_tensors):
        """At very short time, Trotter ≈ exact → fidelity close to 1."""
        g, ts, o = strategy_tensors
        ev = ExactFidelityEvaluator(t_total=0.001)
        f = ev(H_tfim, g, ts, o)
        assert f > 0.9, f"Expected high fidelity for very short time, got {f:.4f}"


# ---------------------------------------------------------------------------
# PINNEvaluator
# ---------------------------------------------------------------------------

class TestPINNEvaluator:
    @pytest.fixture(scope="class")
    def tiny_pinn(self):
        from pinn_trotter.pinn.network import PINNNetwork
        return PINNNetwork(n_qubits=N_QUBITS, fourier_m=8, hidden_dim=16)

    def test_returns_float_in_range(self, H_tfim, strategy_tensors, tiny_pinn):
        g, ts, o = strategy_tensors
        ev = PINNEvaluator(pinn=tiny_pinn, t_total=T_TOTAL)
        f = ev(H_tfim, g, ts, o)
        assert isinstance(f, float)
        assert 0.0 <= f <= 1.0

    def test_circuit_depth_positive(self, H_tfim, strategy_tensors, tiny_pinn):
        g, ts, o = strategy_tensors
        ev = PINNEvaluator(pinn=tiny_pinn, t_total=T_TOTAL)
        d = ev.circuit_depth(H_tfim, g, ts, o)
        assert isinstance(d, int)
        assert d > 0

    def test_pinn_state_cached(self, H_tfim, strategy_tensors, tiny_pinn):
        """Second call should reuse cached PINN state (no re-forward)."""
        g, ts, o = strategy_tensors
        ev = PINNEvaluator(pinn=tiny_pinn, t_total=T_TOTAL)
        ev(H_tfim, g, ts, o)
        cached = ev._psi_pinn
        ev(H_tfim, g, ts, o)
        assert ev._psi_pinn is cached, "PINN state should be cached after first call"

    def test_fallback_on_bad_strategy(self, H_tfim, tiny_pinn):
        """Invalid grouping should not raise; fallback returns [0, 1]."""
        g = torch.randint(0, MAX_GROUPS, (1, N_TERMS))
        ts = torch.rand(1, MAX_GROUPS)
        o = torch.zeros(1, MAX_GROUPS, 3)
        o[:, :, 0] = 1.0
        ev = PINNEvaluator(pinn=tiny_pinn, t_total=T_TOTAL, fallback_exact=True)
        f = ev(H_tfim, g, ts, o)
        assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# make_evaluator
# ---------------------------------------------------------------------------

class TestMakeEvaluator:
    def test_small_system_returns_exact(self):
        ev = make_evaluator(t_total=1.0, n_qubits=4, exact_threshold=8)
        assert isinstance(ev, ExactFidelityEvaluator)

    def test_large_system_no_pinn_returns_exact_with_warning(self):
        # n_qubits > threshold but no PINN → falls back to Exact with a warning
        ev = make_evaluator(t_total=1.0, n_qubits=10, exact_threshold=8)
        assert isinstance(ev, ExactFidelityEvaluator)

    def test_large_system_with_pinn_returns_pinn_evaluator(self):
        from pinn_trotter.pinn.network import PINNNetwork
        pinn = PINNNetwork(n_qubits=10, fourier_m=8, hidden_dim=16)
        ev = make_evaluator(t_total=1.0, n_qubits=10, pinn=pinn, exact_threshold=8)
        assert isinstance(ev, PINNEvaluator)
