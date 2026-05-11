"""Baseline wrappers for fair comparison against external tools."""

from __future__ import annotations

import numpy as np

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


class QiskitTrotterBaseline:
    """Wrap Qiskit's built-in Trotter synthesis as framework strategies."""

    def generate_strategy(
        self,
        hamiltonian,
        t_final: float,
        order: int = 2,
        n_steps: int = 10,
        reps: int = 1,
    ) -> TrotterStrategy:
        """Create a baseline strategy compatible with this codebase.

        Notes:
            - Strategy keeps a single group containing all Pauli terms.
            - Repeated Trotter steps are carried by `metadata["reps"]` and
              consumed in `strategy_to_circuit`.
        """
        if order not in (1, 2, 4):
            raise ValueError(f"order must be one of {{1,2,4}}, got {order}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}")

        effective_reps = int(n_steps) * int(reps)
        return TrotterStrategy(
            grouping=[list(range(hamiltonian.n_terms))],
            orders=[int(order)],
            time_steps=[float(t_final)],
            n_qubits=int(hamiltonian.n_qubits),
            n_terms=int(hamiltonian.n_terms),
            t_total=float(t_final),
            metadata={
                "baseline": "qiskit_trotter",
                "n_steps": int(n_steps),
                "reps": int(effective_reps),
            },
        )

    def sweep_n_steps(
        self,
        hamiltonian,
        t_final: float,
        order: int = 2,
        n_steps_range: list[int] | None = None,
    ) -> list[tuple[TrotterStrategy, float]]:
        """Sweep over step counts and return (strategy, exact_fidelity)."""
        if n_steps_range is None:
            n_steps_range = [1, 2, 5, 10, 20, 50]

        out: list[tuple[TrotterStrategy, float]] = []
        for n_steps in n_steps_range:
            strategy = self.generate_strategy(
                hamiltonian=hamiltonian,
                t_final=t_final,
                order=order,
                n_steps=n_steps,
            )
            fidelity = self._exact_fidelity_from_circuit(hamiltonian, strategy)
            out.append((strategy, fidelity))
        return out

    def evaluate(
        self,
        hamiltonian,
        t_total: float,
        order: int = 4,
        n_steps: int = 5,
    ) -> dict:
        """Run Qiskit Suzuki–Trotter and compute exact fidelity.

        This mirrors the `evaluate()` interface of the external baseline
        adapters (cirq/tket/pennylane) so that the benchmark loop can treat
        all baselines uniformly. Fidelity is computed via Qiskit's
        Statevector simulation against ``expm(-iHt)|0>``.
        """
        strategy = self.generate_strategy(
            hamiltonian=hamiltonian,
            t_final=t_total,
            order=order,
            n_steps=n_steps,
        )
        fidelity = self._exact_fidelity_from_circuit(hamiltonian, strategy)
        return {
            "fidelity": float(fidelity),
            "strategy": strategy,
            "circuit": None,
            "n_steps": int(n_steps),
        }

    @staticmethod
    def _default_psi0(n_qubits: int) -> np.ndarray:
        psi = np.zeros(2**n_qubits, dtype=complex)
        psi[0] = 1.0
        return psi

    def _exact_fidelity_from_circuit(self, hamiltonian, strategy: TrotterStrategy) -> float:
        """Compute exact fidelity against exp(-iHt) using the generated circuit."""
        from scipy.linalg import expm
        from qiskit.quantum_info import Statevector
        from qiskit import transpile

        n_qubits = int(hamiltonian.n_qubits)
        psi_0 = self._default_psi0(n_qubits)
        H_dense = hamiltonian.to_dense_matrix()
        U_exact = expm(-1j * H_dense * float(strategy.t_total))
        psi_exact = U_exact @ psi_0

        circuit = self._strategy_to_qiskit_circuit(strategy, hamiltonian)
        circuit = transpile(
            circuit,
            basis_gates=["cx", "rx", "ry", "rz", "h", "x", "y", "z"],
            optimization_level=0,
        )
        psi_0_le = self._swap_endian(psi_0, n_qubits)
        psi_trotter_le = Statevector(psi_0_le).evolve(circuit).data
        psi_trotter = self._swap_endian(psi_trotter_le, n_qubits)

        overlap = np.vdot(psi_exact, psi_trotter)
        return float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))

    @staticmethod
    def _swap_endian(psi: np.ndarray, n_qubits: int) -> np.ndarray:
        return (
            psi.reshape([2] * n_qubits)
            .transpose(range(n_qubits - 1, -1, -1))
            .reshape(-1)
        )

    @staticmethod
    def _strategy_to_qiskit_circuit(strategy: TrotterStrategy, hamiltonian):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis.evolution import LieTrotter, SuzukiTrotter

        reps = int(strategy.metadata.get("reps", 1))
        if reps < 1:
            reps = 1

        qc = QuantumCircuit(hamiltonian.n_qubits)
        for group_indices, order, tau in zip(
            strategy.grouping, strategy.orders, strategy.time_steps
        ):
            pauli_list = [hamiltonian.pauli_strings[idx][::-1] for idx in group_indices]
            coeffs = [float(hamiltonian.coefficients[idx]) for idx in group_indices]
            op = SparsePauliOp(pauli_list, coeffs)
            if order == 1:
                synthesis = LieTrotter(reps=reps)
            else:
                synthesis = SuzukiTrotter(order=order, reps=reps)
            gate = PauliEvolutionGate(op, time=float(tau), synthesis=synthesis)
            qc.append(gate, range(hamiltonian.n_qubits))
        return qc
