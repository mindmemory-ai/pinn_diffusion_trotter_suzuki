"""Qiskit circuit building and statevector computation for Trotter strategies."""

from __future__ import annotations

import numpy as np

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


def strategy_to_circuit(
    strategy: TrotterStrategy,
    hamiltonian: object,
) -> "qiskit.QuantumCircuit":
    """Build a Qiskit QuantumCircuit implementing the Trotter decomposition.

    Uses PauliEvolutionGate with LieTrotter/SuzukiTrotter synthesis.
    The returned circuit contains undecomposed PauliEvolutionGates suitable
    for downstream transpilation (depth/cx_count analysis).

    Args:
        strategy: TrotterStrategy to compile.
        hamiltonian: HamiltonianGraph instance.

    Returns:
        QuantumCircuit implementing the Trotter evolution.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis.evolution import LieTrotter, SuzukiTrotter
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

    H: HamiltonianGraph = hamiltonian  # type: ignore[assignment]
    n = H.n_qubits
    qc = QuantumCircuit(n)

    synthesis_map = {
        1: LieTrotter(reps=1),
        2: SuzukiTrotter(order=2, reps=1),
        4: SuzukiTrotter(order=4, reps=1),
    }

    for group_indices, order, tau in zip(
        strategy.grouping, strategy.orders, strategy.time_steps
    ):
        # Our strings are big-endian (qubit 0 leftmost);
        # Qiskit SparsePauliOp needs little-endian (qubit 0 rightmost).
        pauli_list = [H.pauli_strings[idx][::-1] for idx in group_indices]
        coeffs = [float(H.coefficients[idx]) for idx in group_indices]
        op = SparsePauliOp(pauli_list, coeffs)
        gate = PauliEvolutionGate(op, time=float(tau), synthesis=synthesis_map[order])
        qc.append(gate, range(n))

    return qc


def compute_trotter_statevector(
    circuit_or_strategy: "qiskit.QuantumCircuit | TrotterStrategy",
    hamiltonian: object,
    psi_0: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Compute the final state after Trotter evolution.

    Uses Qiskit Statevector for n_qubits <= 12, sparse matrix otherwise.

    The output state vector is in the same big-endian qubit convention as psi_0
    (qubit 0 = most-significant bit of the state index).

    Args:
        circuit_or_strategy: A compiled QuantumCircuit or a TrotterStrategy.
        hamiltonian: HamiltonianGraph instance.
        psi_0: Initial state vector, shape (2^n,), big-endian convention.
        n_qubits: Number of qubits.

    Returns:
        Final state vector, shape (2^n,), complex, big-endian convention.
    """
    from qiskit import QuantumCircuit
    from pinn_trotter.data.generator import apply_trotter_from_hamiltonian

    is_circuit = isinstance(circuit_or_strategy, QuantumCircuit)

    if n_qubits <= 12:
        from qiskit import transpile
        from qiskit.quantum_info import Statevector

        if is_circuit:
            circuit = circuit_or_strategy
        else:
            circuit = strategy_to_circuit(circuit_or_strategy, hamiltonian)

        # Decompose PauliEvolutionGates into primitive basis gates
        circuit_decomposed = transpile(
            circuit,
            basis_gates=["cx", "rx", "ry", "rz", "h", "x", "y", "z"],
            optimization_level=0,
        )

        # psi_0 is big-endian; Qiskit Statevector uses little-endian.
        psi_0_le = _swap_endian(psi_0.astype(complex), n_qubits)
        sv = Statevector(psi_0_le).evolve(circuit_decomposed)
        # Convert result back to big-endian convention
        return _swap_endian(sv.data, n_qubits)
    else:
        if is_circuit:
            raise ValueError(
                "Need TrotterStrategy (not QuantumCircuit) for n_qubits > 12; "
                "sparse matrix simulation requires explicit strategy."
            )
        return apply_trotter_from_hamiltonian(
            hamiltonian, circuit_or_strategy, psi_0
        )


def _swap_endian(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """Swap qubit ordering between big-endian and little-endian conventions.

    This operation is self-inverse: applying it twice returns the original array.
    """
    return (
        psi.reshape([2] * n_qubits)
        .transpose(range(n_qubits - 1, -1, -1))
        .reshape(-1)
    )
