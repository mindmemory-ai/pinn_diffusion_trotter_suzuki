"""Unit tests for Qiskit / Cirq / TKET / PennyLane baseline wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis.evolution import SuzukiTrotter

from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
from pinn_trotter.benchmarks.baseline_adapters import (
    BASELINE_REGISTRY,
    CirqTrotterBaseline,
    PennyLaneTrotterBaseline,
    TketTrotterBaseline,
)
from pinn_trotter.benchmarks.hamiltonians import make_tfim


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.abs(np.vdot(a, b)) ** 2, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Qiskit (kept: 4th-order is now the canonical Qiskit baseline)
# ---------------------------------------------------------------------------


def test_generate_strategy_sets_qiskit_reps_metadata():
    h = make_tfim(4, J=1.0, h=0.5)
    baseline = QiskitTrotterBaseline()
    s = baseline.generate_strategy(h, t_final=1.2, order=4, n_steps=5, reps=2)

    assert s.grouping == [list(range(h.n_terms))]
    assert s.orders == [4]
    assert s.time_steps == [pytest.approx(1.2)]
    assert s.metadata["reps"] == 10


def test_qiskit_fourth_order_matches_manual_evolution():
    n_qubits = 4
    t_final = 0.7
    n_steps = 3
    h = make_tfim(n_qubits, J=1.1, h=0.4)
    baseline = QiskitTrotterBaseline()
    strategy = baseline.generate_strategy(
        hamiltonian=h, t_final=t_final, order=4, n_steps=n_steps
    )

    circuit_baseline = baseline._strategy_to_qiskit_circuit(strategy, h)
    circuit_baseline = transpile(
        circuit_baseline,
        basis_gates=["cx", "rx", "ry", "rz", "h", "x", "y", "z"],
        optimization_level=0,
    )

    op = SparsePauliOp([p[::-1] for p in h.pauli_strings], [float(c) for c in h.coefficients])
    manual = PauliEvolutionGate(op, time=t_final, synthesis=SuzukiTrotter(order=4, reps=n_steps))
    from qiskit import QuantumCircuit

    circuit_manual = QuantumCircuit(n_qubits)
    circuit_manual.append(manual, range(n_qubits))
    circuit_manual = transpile(
        circuit_manual,
        basis_gates=["cx", "rx", "ry", "rz", "h", "x", "y", "z"],
        optimization_level=0,
    )

    psi0 = np.zeros(2**n_qubits, dtype=complex)
    psi0[0] = 1.0
    psi0_le = baseline._swap_endian(psi0, n_qubits)
    psi_baseline = baseline._swap_endian(
        Statevector(psi0_le).evolve(circuit_baseline).data, n_qubits
    )
    psi_manual = baseline._swap_endian(
        Statevector(psi0_le).evolve(circuit_manual).data, n_qubits
    )

    assert _fidelity(psi_baseline, psi_manual) > 1 - 1e-12


def test_sweep_n_steps_returns_valid_fidelity_range():
    h = make_tfim(4, J=0.9, h=0.6)
    baseline = QiskitTrotterBaseline()
    results = baseline.sweep_n_steps(h, t_final=0.5, order=4, n_steps_range=[1, 2])

    assert len(results) == 2
    for strategy, fidelity in results:
        assert strategy.metadata["baseline"] == "qiskit_trotter"
        assert 0.0 <= fidelity <= 1.0


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


def test_baseline_registry_contains_expected_adapters():
    assert set(BASELINE_REGISTRY.keys()) == {"cirq", "tket", "pennylane"}


# ---------------------------------------------------------------------------
# Cirq adapter
# ---------------------------------------------------------------------------


def test_cirq_adapter_returns_valid_metrics():
    h = make_tfim(4, J=1.0, h=0.3)
    adapter = CirqTrotterBaseline(n_steps=3)
    res = adapter.evaluate(h, t_total=1.0)

    assert "fidelity" in res and "strategy" in res
    assert 0.0 <= res["fidelity"] <= 1.0
    assert res["strategy"].metadata["baseline"] == "cirq"
    assert res["strategy"].orders == [4]


def test_cirq_high_n_steps_improves_fidelity():
    """Sanity: more Trotter steps should not degrade fidelity meaningfully."""
    h = make_tfim(4, J=1.0, h=0.3)
    f_low = CirqTrotterBaseline(n_steps=2).evaluate(h, t_total=1.0)["fidelity"]
    f_high = CirqTrotterBaseline(n_steps=10).evaluate(h, t_total=1.0)["fidelity"]
    assert f_high >= f_low - 1e-3


# ---------------------------------------------------------------------------
# TKET adapter
# ---------------------------------------------------------------------------


def test_tket_adapter_returns_valid_metrics():
    h = make_tfim(4, J=1.0, h=0.3)
    adapter = TketTrotterBaseline(n_steps=3)
    res = adapter.evaluate(h, t_total=1.0)

    assert "fidelity" in res and "strategy" in res
    assert 0.0 <= res["fidelity"] <= 1.0
    assert res["strategy"].metadata["baseline"] == "tket"
    assert res["strategy"].orders == [4]


# ---------------------------------------------------------------------------
# PennyLane adapter
# ---------------------------------------------------------------------------


def test_pennylane_adapter_returns_valid_metrics():
    h = make_tfim(4, J=1.0, h=0.3)
    adapter = PennyLaneTrotterBaseline(n_steps=3)
    res = adapter.evaluate(h, t_total=1.0)

    assert "fidelity" in res and "strategy" in res
    assert 0.0 <= res["fidelity"] <= 1.0
    assert res["strategy"].metadata["baseline"] == "pennylane"
    assert res["strategy"].orders == [4]


# ---------------------------------------------------------------------------
# Cross-baseline agreement (smoke test)
# ---------------------------------------------------------------------------


def test_external_baselines_agree_within_tolerance():
    """Cirq, TKET, PennyLane should give similar fidelities for same problem."""
    h = make_tfim(4, J=1.0, h=0.3)
    n_steps = 5
    f_cirq = CirqTrotterBaseline(n_steps=n_steps).evaluate(h, t_total=1.0)["fidelity"]
    f_tket = TketTrotterBaseline(n_steps=n_steps).evaluate(h, t_total=1.0)["fidelity"]
    f_pl = PennyLaneTrotterBaseline(n_steps=n_steps).evaluate(h, t_total=1.0)["fidelity"]

    # All three should be within 5% of each other (different sub-step orderings allowed)
    fids = [f_cirq, f_tket, f_pl]
    assert max(fids) - min(fids) < 0.05, f"Cross-baseline disagreement: {fids}"
