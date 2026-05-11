"""Tests for circuit_builder: strategy_to_circuit and compute_trotter_statevector."""

from __future__ import annotations

import numpy as np
import pytest

from pinn_trotter.benchmarks.hamiltonians import make_tfim
from pinn_trotter.data.generator import apply_trotter_from_hamiltonian
from pinn_trotter.strategy.circuit_builder import (
    _swap_endian,
    compute_trotter_statevector,
    strategy_to_circuit,
)
from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tfim4_hamiltonian():
    return make_tfim(4, J=1.0, h=0.5, boundary="periodic")


@pytest.fixture()
def standard_order2_strategy(tfim4_hamiltonian):
    """Second-order strategy: all TFIM terms in one group."""
    H = tfim4_hamiltonian
    n = H.n_terms
    return TrotterStrategy(
        grouping=[list(range(n))],
        orders=[2],
        time_steps=[0.5],
        n_qubits=H.n_qubits,
        n_terms=n,
        t_total=0.5,
    )


@pytest.fixture()
def psi_0_ground():
    """Computational basis ground state |0000>."""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    return psi


# ---------------------------------------------------------------------------
# Tests for _swap_endian
# ---------------------------------------------------------------------------


def test_swap_endian_selfinverse():
    """Applying _swap_endian twice returns the original array."""
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    assert np.allclose(_swap_endian(_swap_endian(psi, 4), 4), psi)


def test_swap_endian_n2():
    """Explicit 2-qubit check: state |10> (big-endian index 2) maps to little-endian index 1."""
    psi = np.zeros(4, dtype=complex)
    psi[2] = 1.0  # |10> in big-endian (qubit 0 = 1)
    sv = _swap_endian(psi, 2)
    # In little-endian, qubit 0 = 1 means index 1
    assert sv[1] == pytest.approx(1.0)
    assert np.sum(np.abs(sv) ** 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests for strategy_to_circuit
# ---------------------------------------------------------------------------


def test_strategy_to_circuit_returns_circuit(tfim4_hamiltonian, standard_order2_strategy):
    from qiskit import QuantumCircuit

    qc = strategy_to_circuit(standard_order2_strategy, tfim4_hamiltonian)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4


def test_strategy_to_circuit_orders(tfim4_hamiltonian):
    """Circuits for all supported Suzuki orders can be built without error."""
    H = tfim4_hamiltonian
    n = H.n_terms
    for order in (1, 2, 4):
        strat = TrotterStrategy(
            grouping=[list(range(n))],
            orders=[order],
            time_steps=[0.3],
            n_qubits=H.n_qubits,
            n_terms=n,
            t_total=0.3,
        )
        qc = strategy_to_circuit(strat, H)
        assert qc.num_qubits == 4


# ---------------------------------------------------------------------------
# Test 3-C-3: Qiskit Statevector vs sparse matrix (error < 1e-10)
# ---------------------------------------------------------------------------


def test_statevector_matches_sparse_order2(
    tfim4_hamiltonian, standard_order2_strategy, psi_0_ground
):
    """TFIM 4-qubit order-2: Qiskit Statevector vs sparse matrix, error < 1e-10."""
    H = tfim4_hamiltonian
    strategy = standard_order2_strategy
    psi_0 = psi_0_ground

    psi_sparse = apply_trotter_from_hamiltonian(H, strategy, psi_0)
    psi_qiskit = compute_trotter_statevector(strategy, H, psi_0, H.n_qubits)

    err = np.max(np.abs(psi_qiskit - psi_sparse))
    assert err < 1e-10, f"Max element-wise error {err:.2e} exceeds 1e-10"


def test_statevector_matches_sparse_order1(tfim4_hamiltonian, psi_0_ground):
    """TFIM 4-qubit order-1: Qiskit vs sparse matrix, error < 1e-10."""
    H = tfim4_hamiltonian
    n = H.n_terms
    strategy = TrotterStrategy(
        grouping=[list(range(n))],
        orders=[1],
        time_steps=[0.3],
        n_qubits=H.n_qubits,
        n_terms=n,
        t_total=0.3,
    )
    psi_sparse = apply_trotter_from_hamiltonian(H, strategy, psi_0_ground)
    psi_qiskit = compute_trotter_statevector(strategy, H, psi_0_ground, H.n_qubits)
    err = np.max(np.abs(psi_qiskit - psi_sparse))
    assert err < 1e-10, f"Order-1 error {err:.2e} exceeds 1e-10"


def test_statevector_circuit_input(
    tfim4_hamiltonian, standard_order2_strategy, psi_0_ground
):
    """compute_trotter_statevector accepts a pre-built QuantumCircuit."""
    H = tfim4_hamiltonian
    qc = strategy_to_circuit(standard_order2_strategy, H)
    psi_from_circuit = compute_trotter_statevector(qc, H, psi_0_ground, H.n_qubits)
    psi_from_strategy = compute_trotter_statevector(
        standard_order2_strategy, H, psi_0_ground, H.n_qubits
    )
    assert np.allclose(psi_from_circuit, psi_from_strategy, atol=1e-14)


def test_statevector_preserves_norm(tfim4_hamiltonian, standard_order2_strategy, psi_0_ground):
    """Output statevector should be normalised."""
    H = tfim4_hamiltonian
    psi_out = compute_trotter_statevector(
        standard_order2_strategy, H, psi_0_ground, H.n_qubits
    )
    norm = np.linalg.norm(psi_out)
    assert abs(norm - 1.0) < 1e-12, f"Norm {norm} deviates from 1"
