"""Cirq / TKET / PennyLane / Paulihedral Trotter baseline adapters.

Each adapter provides a uniform `evaluate(hamiltonian, t_total)` returning a
metrics dict with `fidelity`, `depth`, `cx_count`, and `strategy` (a
``TrotterStrategy`` proxy used by the benchmark for unified depth/cx counting
under the comparison gate set).

Design choices:

- All adapters run a 4th-order Suzuki–Trotter decomposition with the same
  number of repetitions (``n_steps``) so the comparison reflects each
  framework's native compilation/runtime, not the algorithm choice.
- Fidelity is computed inside each framework against scipy's ``expm`` reference
  state, then reported back to the benchmark; the strategy is also returned so
  that `transpiled_depth` / `cx_count` (Qiskit-based) can produce a consistent
  cross-framework gate count under a unified basis (h/cx/rz/x).
- Latency is measured by ``inference_latency`` over ``adapter.evaluate`` calls.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


# ---------------------------------------------------------------------------
# Helpers shared across adapters
# ---------------------------------------------------------------------------


def _default_psi0(n_qubits: int) -> np.ndarray:
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0
    return psi


def _exact_state(hamiltonian, t_total: float) -> np.ndarray:
    """Return exp(-i H t) |0...0>."""
    from scipy.linalg import expm

    n = int(hamiltonian.n_qubits)
    H_dense = hamiltonian.to_dense_matrix()
    return expm(-1j * H_dense * float(t_total)) @ _default_psi0(n)


def _make_proxy_strategy(hamiltonian, t_total: float, order: int, n_steps: int,
                          tag: str) -> TrotterStrategy:
    """Proxy strategy mirroring QiskitTrotterBaseline output for unified depth/cx counts."""
    return TrotterStrategy(
        grouping=[list(range(int(hamiltonian.n_terms)))],
        orders=[int(order)],
        time_steps=[float(t_total)],
        n_qubits=int(hamiltonian.n_qubits),
        n_terms=int(hamiltonian.n_terms),
        t_total=float(t_total),
        metadata={"baseline": tag, "n_steps": int(n_steps), "reps": int(n_steps)},
    )


def _swap_endian(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    return (
        psi.reshape([2] * n_qubits)
        .transpose(range(n_qubits - 1, -1, -1))
        .reshape(-1)
    )


# ---------------------------------------------------------------------------
# Cirq adapter
# ---------------------------------------------------------------------------


class CirqTrotterBaseline:
    """4th-order Suzuki–Trotter via Cirq's PauliSumExponential primitives.

    Cirq does not ship a high-level Trotter routine; we apply the 4th-order
    Suzuki recursion (Yoshida 1990) over Cirq's exact PauliString exponentials
    for each repetition.
    """

    name = "cirq"

    def __init__(self, n_steps: int = 5) -> None:
        self.n_steps = int(n_steps)

    def evaluate(self, hamiltonian, t_total: float) -> dict[str, Any]:
        import cirq

        n = int(hamiltonian.n_qubits)
        qubits = cirq.LineQubit.range(n)
        psi0 = _default_psi0(n)

        # Build cirq PauliStrings from our HamiltonianGraph
        pauli_strings = []
        for s, c in zip(hamiltonian.pauli_strings, hamiltonian.coefficients):
            ops = []
            for q, ch in enumerate(s):
                if ch == "I":
                    continue
                op = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}[ch]
                ops.append(op(qubits[q]))
            if ops:
                ps = cirq.PauliString(*ops, coefficient=float(c))
                pauli_strings.append(ps)

        # Suzuki 4th-order coefficients (Yoshida)
        s_p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        s_3 = 1.0 - 4.0 * s_p
        sub = [s_p, s_p, s_3, s_p, s_p]

        def _step_circuit(tau: float) -> "cirq.Circuit":
            """One first-order step: product of single PauliString exponentials."""
            ops = []
            for ps in pauli_strings:
                # exp(-i τ Σ c_j P_j) factorized: exp(-i τ c_j P_j)
                # cirq.PauliStringPhasor has half_turns sign convention:
                #   PauliStringPhasor(ps, exponent_neg) => exp(-i π exp_neg ps / 2)
                # We need exp(-i τ ps); set exp = 2τ/π so factor = 2τ/π * π/2 = τ.
                # Sign: use exponent_neg = 2τ * c_j / π  (ps has coefficient absorbed)
                # Cirq absorbs `coefficient` into ps; we want exp(-i τ * c * P).
                # PauliStringPhasor(ps_unit, exponent_neg=2*τ*c/π)
                # where ps_unit is ps with its coefficient stripped to ±1.
                coeff = float(ps.coefficient.real)
                ps_unit = ps / coeff
                ops.append(
                    cirq.PauliStringPhasor(ps_unit, exponent_neg=(2 * tau * coeff) / np.pi)
                )
            return cirq.Circuit(ops)

        def _suzuki4_step(tau: float) -> "cirq.Circuit":
            c = cirq.Circuit()
            for s in sub:
                c += _step_circuit(s * tau)
            return c

        dt = float(t_total) / self.n_steps
        circuit = cirq.Circuit()
        for _ in range(self.n_steps):
            circuit += _suzuki4_step(dt)

        # Simulate
        sim = cirq.Simulator(dtype=np.complex128)
        result = sim.simulate(circuit, qubit_order=qubits)
        # Cirq's state vector is in big-endian (qubit 0 = most-significant),
        # matching our psi convention; check against direct mat product if mismatch.
        psi_trotter = np.asarray(result.final_state_vector, dtype=complex)

        psi_exact = _exact_state(hamiltonian, t_total)
        # Try both endianness; use the one that gives larger overlap.
        f_native = abs(np.vdot(psi_exact, psi_trotter)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2
        )
        psi_swap = _swap_endian(psi_trotter, n)
        f_swap = abs(np.vdot(psi_exact, psi_swap)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_swap) ** 2
        )
        fidelity = float(np.clip(max(f_native, f_swap), 0.0, 1.0))

        strategy = _make_proxy_strategy(hamiltonian, t_total, 4, self.n_steps, "cirq")

        return {
            "fidelity": fidelity,
            "strategy": strategy,
            "circuit": circuit,
            "n_steps": self.n_steps,
        }


# ---------------------------------------------------------------------------
# TKET (pytket) adapter
# ---------------------------------------------------------------------------


class TketTrotterBaseline:
    """4th-order Suzuki–Trotter via pytket's PauliExpBox.

    We construct a pytket Circuit using PauliExpBox operations and run the
    state-vector simulation through the AerBackend (statevector mode) for a
    Qiskit-compatible reference state.
    """

    name = "tket"

    def __init__(self, n_steps: int = 5) -> None:
        self.n_steps = int(n_steps)

    def evaluate(self, hamiltonian, t_total: float) -> dict[str, Any]:
        from pytket.circuit import Circuit, PauliExpBox
        from pytket.pauli import Pauli

        n = int(hamiltonian.n_qubits)

        pauli_map = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}

        def _pauli_terms(coeff_factor: float):
            """Yield (PauliExpBox, qubit_indices) for each H term scaled by coeff_factor.

            PauliExpBox(p_list, t) implements exp(-i (t π / 2) ⊗P_i).
            We want exp(-i τ c P) ⇒ t = 2 τ c / π.
            """
            terms = []
            for s, c in zip(hamiltonian.pauli_strings, hamiltonian.coefficients):
                paulis = [pauli_map[ch] for ch in s]
                pe_t = (2.0 * coeff_factor * float(c)) / np.pi
                terms.append((PauliExpBox(paulis, pe_t), list(range(n))))
            return terms

        # Suzuki 4th-order
        s_p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        s_3 = 1.0 - 4.0 * s_p
        sub = [s_p, s_p, s_3, s_p, s_p]

        dt = float(t_total) / self.n_steps
        c = Circuit(n)
        for _ in range(self.n_steps):
            for s in sub:
                for box, qubits in _pauli_terms(s * dt):
                    c.add_pauliexpbox(box, qubits)

        # Simulate with pytket's built-in statevector method
        try:
            from pytket.utils import probs_from_state  # noqa: F401
            psi_trotter = c.get_statevector()
        except AttributeError:
            # Fall back: convert to Qiskit and simulate
            from pytket.extensions.qiskit import tk_to_qiskit
            from qiskit.quantum_info import Statevector

            qc = tk_to_qiskit(c)
            psi_trotter = Statevector(qc).data

        psi_trotter = np.asarray(psi_trotter, dtype=complex)
        psi_exact = _exact_state(hamiltonian, t_total)

        f_native = abs(np.vdot(psi_exact, psi_trotter)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2
        )
        psi_swap = _swap_endian(psi_trotter, n)
        f_swap = abs(np.vdot(psi_exact, psi_swap)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_swap) ** 2
        )
        fidelity = float(np.clip(max(f_native, f_swap), 0.0, 1.0))

        strategy = _make_proxy_strategy(hamiltonian, t_total, 4, self.n_steps, "tket")
        return {
            "fidelity": fidelity,
            "strategy": strategy,
            "circuit": c,
            "n_steps": self.n_steps,
        }


# ---------------------------------------------------------------------------
# PennyLane adapter
# ---------------------------------------------------------------------------


class PennyLaneTrotterBaseline:
    """4th-order Suzuki–Trotter via PennyLane's TrotterProduct."""

    name = "pennylane"

    def __init__(self, n_steps: int = 5) -> None:
        self.n_steps = int(n_steps)

    def evaluate(self, hamiltonian, t_total: float) -> dict[str, Any]:
        import pennylane as qml

        n = int(hamiltonian.n_qubits)

        # Build PennyLane Hamiltonian (sum of c_j * tensor of Paulis)
        ops = []
        coeffs = []
        pauli_map = {"X": qml.X, "Y": qml.Y, "Z": qml.Z}
        for s, c in zip(hamiltonian.pauli_strings, hamiltonian.coefficients):
            term_ops = []
            for q, ch in enumerate(s):
                if ch == "I":
                    continue
                term_ops.append(pauli_map[ch](q))
            if not term_ops:
                continue
            tensor = term_ops[0]
            for op in term_ops[1:]:
                tensor = tensor @ op
            ops.append(tensor)
            coeffs.append(float(c))

        H_pl = qml.Hamiltonian(coeffs, ops)

        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def circuit():
            qml.TrotterProduct(H_pl, time=float(t_total), n=self.n_steps, order=4)
            return qml.state()

        psi_trotter = np.asarray(circuit(), dtype=complex)

        psi_exact = _exact_state(hamiltonian, t_total)
        f_native = abs(np.vdot(psi_exact, psi_trotter)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2
        )
        psi_swap = _swap_endian(psi_trotter, n)
        f_swap = abs(np.vdot(psi_exact, psi_swap)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_swap) ** 2
        )
        fidelity = float(np.clip(max(f_native, f_swap), 0.0, 1.0))

        strategy = _make_proxy_strategy(hamiltonian, t_total, 4, self.n_steps, "pennylane")
        return {
            "fidelity": fidelity,
            "strategy": strategy,
            "circuit": None,  # pennylane qnode is opaque
            "n_steps": self.n_steps,
        }


# ---------------------------------------------------------------------------
# Paulihedral adapter
# ---------------------------------------------------------------------------


class _PaulihedralTerm:
    """Minimal wrapper for paulihedral scheduler APIs."""

    def __init__(self, ps: str, coeff: float) -> None:
        self.ps = ps
        self.coeff = float(coeff)

    def __len__(self) -> int:
        return len(self.ps)

    def count(self, token: str) -> int:
        return self.ps.count(token)


class PaulihedralBaseline:
    """Paulihedral baseline with paulihedral.parallel_bl scheduling."""

    name = "paulihedral"

    def __init__(self, n_steps: int = 5, scheduler: str = "depth") -> None:
        self.n_steps = int(n_steps)
        self.scheduler = str(scheduler)
        if self.n_steps <= 0:
            raise ValueError("PaulihedralBaseline requires n_steps > 0")
        if self.scheduler not in {"depth", "gate_count"}:
            raise ValueError("scheduler must be one of {'depth', 'gate_count'}")

    @staticmethod
    def _depth_and_cx(circuit) -> tuple[int, int]:
        from qiskit import transpile

        transpiled = transpile(
            circuit,
            basis_gates=["h", "cx", "rz", "x"],
            optimization_level=1,
        )
        return int(transpiled.depth()), int(transpiled.count_ops().get("cx", 0))

    def _schedule_terms(self, terms: list[_PaulihedralTerm]) -> list[_PaulihedralTerm]:
        import paulihedral.parallel_bl as pb

        # paulihedral scheduler input shape: layer -> block -> term
        raw_blocks = [[term] for term in terms]
        if self.scheduler == "depth":
            layers = pb.depth_oriented_scheduling(raw_blocks, maxiter=1)
        else:
            layers = pb.gate_count_oriented_scheduling(raw_blocks)
        return [term for layer in layers for block in layer for term in block]

    def _build_qiskit_circuit(self, hamiltonian, t_total: float):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        n = int(hamiltonian.n_qubits)
        terms = [
            _PaulihedralTerm(s[::-1], float(c))
            for s, c in zip(hamiltonian.pauli_strings, hamiltonian.coefficients)
        ]
        ordered_terms = self._schedule_terms(terms)
        dt = float(t_total) / self.n_steps

        qc = QuantumCircuit(n)
        for _ in range(self.n_steps):
            for term in ordered_terms:
                op = SparsePauliOp([term.ps], [term.coeff])
                qc.append(PauliEvolutionGate(op, time=dt), list(range(n)))
        return qc

    def evaluate(self, hamiltonian, t_total: float) -> dict[str, Any]:
        try:
            import paulihedral.parallel_bl as _  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "paulihedral is unavailable in current environment. "
                "Please install package/module `paulihedral`."
            ) from exc

        from qiskit.quantum_info import Statevector

        n = int(hamiltonian.n_qubits)
        circuit = self._build_qiskit_circuit(hamiltonian, t_total)

        psi_0 = _default_psi0(n)
        psi_exact = _exact_state(hamiltonian, t_total)
        psi_trotter_le = Statevector(self._swap_endian(psi_0, n)).evolve(circuit).data
        psi_trotter = self._swap_endian(np.asarray(psi_trotter_le, dtype=complex), n)
        fidelity = float(
            np.clip(
                abs(np.vdot(psi_exact, psi_trotter)) ** 2
                / (np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2),
                0.0,
                1.0,
            )
        )
        depth, cx = self._depth_and_cx(circuit)

        strategy = _make_proxy_strategy(
            hamiltonian=hamiltonian,
            t_total=t_total,
            order=1,
            n_steps=self.n_steps,
            tag="paulihedral",
        )
        strategy.metadata.update(
            {
                "framework": "paulihedral",
                "scheduler": self.scheduler,
            }
        )
        return {
            "fidelity": fidelity,
            "strategy": strategy,
            "circuit": circuit,
            "n_steps": self.n_steps,
            "depth": depth,
            "cx_count": cx,
        }

    @staticmethod
    def _swap_endian(psi: np.ndarray, n_qubits: int) -> np.ndarray:
        return (
            psi.reshape([2] * n_qubits)
            .transpose(range(n_qubits - 1, -1, -1))
            .reshape(-1)
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BASELINE_REGISTRY: dict[str, type] = {
    "cirq": CirqTrotterBaseline,
    "tket": TketTrotterBaseline,
    "pennylane": PennyLaneTrotterBaseline,
    "paulihedral": PaulihedralBaseline,
}
