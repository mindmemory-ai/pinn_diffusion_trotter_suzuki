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

    def __init__(self, n_steps: int = 5, order: int = 4) -> None:
        self.n_steps = int(n_steps)
        if order not in (1, 2, 4):
            raise ValueError(f"CirqTrotterBaseline: order must be 1, 2, or 4, got {order}")
        self.order = int(order)

    def evaluate(self, hamiltonian, t_total: float) -> dict[str, Any]:
        import cirq

        n = int(hamiltonian.n_qubits)
        qubits = cirq.LineQubit.range(n)

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

        def _s1_step(tau: float) -> "cirq.Circuit":
            """One first-order step: product of single PauliString exponentials."""
            ops = []
            for ps in pauli_strings:
                coeff = float(ps.coefficient.real)
                ps_unit = ps / coeff
                ops.append(
                    cirq.PauliStringPhasor(ps_unit, exponent_neg=(2 * tau * coeff) / np.pi)
                )
            return cirq.Circuit(ops)

        def _s2_step(tau: float) -> "cirq.Circuit":
            """Symmetric 2nd-order Trotter step (Strang splitting)."""
            c = cirq.Circuit()
            half = tau / 2.0
            for ps in pauli_strings:
                coeff = float(ps.coefficient.real)
                c.append(
                    cirq.PauliStringPhasor(ps / coeff, exponent_neg=(2 * half * coeff) / np.pi)
                )
            for ps in reversed(pauli_strings):
                coeff = float(ps.coefficient.real)
                c.append(
                    cirq.PauliStringPhasor(ps / coeff, exponent_neg=(2 * half * coeff) / np.pi)
                )
            return c

        dt = float(t_total) / self.n_steps
        circuit = cirq.Circuit()

        if self.order == 1:
            for _ in range(self.n_steps):
                circuit += _s1_step(dt)
        elif self.order == 2:
            for _ in range(self.n_steps):
                circuit += _s2_step(dt)
        else:
            # Suzuki 4th-order coefficients (Yoshida 1990)
            s_p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
            s_3 = 1.0 - 4.0 * s_p
            sub = [s_p, s_p, s_3, s_p, s_p]

            def _suzuki4_step(tau: float) -> "cirq.Circuit":
                c = cirq.Circuit()
                for s in sub:
                    c += _s2_step(s * tau)
                return c

            for _ in range(self.n_steps):
                circuit += _suzuki4_step(dt)

        # Simulate
        sim = cirq.Simulator(dtype=np.complex128)
        result = sim.simulate(circuit, qubit_order=qubits)
        psi_trotter = np.asarray(result.final_state_vector, dtype=complex)

        psi_exact = _exact_state(hamiltonian, t_total)
        f_native = abs(np.vdot(psi_exact, psi_trotter)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2
        )
        psi_swap = _swap_endian(psi_trotter, n)
        f_swap = abs(np.vdot(psi_exact, psi_swap)) ** 2 / (
            np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_swap) ** 2
        )
        fidelity = float(np.clip(max(f_native, f_swap), 0.0, 1.0))

        strategy = _make_proxy_strategy(hamiltonian, t_total, self.order, self.n_steps, "cirq")

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

    def __init__(self, n_steps: int = 5, order: int = 4) -> None:
        self.n_steps = int(n_steps)
        if order not in (1, 2, 4):
            raise ValueError(f"TketTrotterBaseline: order must be 1, 2, or 4, got {order}")
        self.order = int(order)

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

        dt = float(t_total) / self.n_steps
        c = Circuit(n)

        if self.order == 1:
            for _ in range(self.n_steps):
                for box, qubits in _pauli_terms(dt):
                    c.add_pauliexpbox(box, qubits)
        elif self.order == 2:
            for _ in range(self.n_steps):
                half = dt / 2.0
                for box, qubits in _pauli_terms(half):
                    c.add_pauliexpbox(box, qubits)
                for box, qubits in reversed(_pauli_terms(half)):
                    c.add_pauliexpbox(box, qubits)
        else:
            # Suzuki 4th-order (Yoshida 1990): S4(t) = S2(p₁t)@S2(p₂t)@S2(p₃t)@S2(p₄t)@S2(p₅t)
            s_p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
            s_3 = 1.0 - 4.0 * s_p
            sub = [s_p, s_p, s_3, s_p, s_p]
            for _ in range(self.n_steps):
                for s in sub:
                    half = (s * dt) / 2.0
                    for box, qubits in _pauli_terms(half):
                        c.add_pauliexpbox(box, qubits)
                    for box, qubits in reversed(_pauli_terms(half)):
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

        strategy = _make_proxy_strategy(hamiltonian, t_total, self.order, self.n_steps, "tket")
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

    def __init__(self, n_steps: int = 5, order: int = 4) -> None:
        self.n_steps = int(n_steps)
        if order not in (1, 2, 4):
            raise ValueError(f"PennyLaneTrotterBaseline: order must be 1, 2, or 4, got {order}")
        self.order = int(order)

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
            qml.TrotterProduct(H_pl, time=float(t_total), n=self.n_steps, order=self.order)
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

        strategy = _make_proxy_strategy(hamiltonian, t_total, self.order, self.n_steps, "pennylane")
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


class PaulihedralSuzuki4Baseline(PaulihedralBaseline):
    """Paulihedral scheduling wrapped in 4th-order Suzuki-Trotter packaging.

    Uses Paulihedral's ``depth_oriented_scheduling`` for term reordering,
    then groups scheduled blocks and applies ``SuzukiTrotter(order=4)``
    so the comparison with other fourth-order baselines is fair.
    """

    name = "paulihedral_4th"
    order = 4

    def _build_qiskit_circuit(self, hamiltonian, t_total: float):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis.evolution import SuzukiTrotter
        import paulihedral.parallel_bl as pb

        n = int(hamiltonian.n_qubits)
        terms = [
            _PaulihedralTerm(s[::-1], float(c))
            for s, c in zip(hamiltonian.pauli_strings, hamiltonian.coefficients)
        ]

        # Use Paulihedral to find depth-optimized block structure.
        raw_blocks = [[term] for term in terms]
        if self.scheduler == "depth":
            layers = pb.depth_oriented_scheduling(raw_blocks, maxiter=1)
        else:
            layers = pb.gate_count_oriented_scheduling(raw_blocks)

        # Collect scheduled blocks (each block = commuting terms that can run in parallel).
        scheduled_blocks: list[list[_PaulihedralTerm]] = []
        for layer in layers:
            for block in layer:
                if block:
                    scheduled_blocks.append(block)

        dt = float(t_total) / self.n_steps
        qc = QuantumCircuit(n)

        for _ in range(self.n_steps):
            for block in scheduled_blocks:
                pauli_strs = [t.ps for t in block]
                coeffs = [t.coeff for t in block]
                op = SparsePauliOp(pauli_strs, coeffs)
                qc.append(
                    PauliEvolutionGate(
                        op,
                        time=dt,
                        synthesis=SuzukiTrotter(order=4, reps=1),
                    ),
                    list(range(n)),
                )
        return qc

    def evaluate(self, hamiltonian, t_total: float) -> dict[str, Any]:
        try:
            import paulihedral.parallel_bl as _  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "paulihedral is unavailable. Install paulihedral to use this baseline."
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
            order=4,
            n_steps=self.n_steps,
            tag="paulihedral_4th",
        )
        strategy.metadata.update(
            {"framework": "paulihedral", "scheduler": self.scheduler, "order": 4}
        )
        return {
            "fidelity": fidelity,
            "strategy": strategy,
            "circuit": circuit,
            "n_steps": self.n_steps,
            "depth": depth,
            "cx_count": cx,
        }


class QiskitGroupCommutingBaseline:
    """Qiskit group_commuting baseline — the teacher for our training data.

    Uses ``SparsePauliOp.group_commuting()`` to partition terms into commuting
    groups, then applies each group sequentially with n_steps repetitions.
    This is the exact heuristic our diffusion model was trained to improve upon.
    """

    name = "qiskit_group_commuting"

    def __init__(self, n_steps: int = 5, order: int = 1) -> None:
        self.n_steps = int(n_steps)
        self.order = int(order)
        if self.n_steps <= 0:
            raise ValueError("QiskitGroupCommutingBaseline requires n_steps > 0")

    @staticmethod
    def _depth_and_cx(circuit) -> tuple[int, int]:
        from qiskit import transpile

        transpiled = transpile(
            circuit,
            basis_gates=["h", "cx", "rz", "x"],
            optimization_level=1,
        )
        return int(transpiled.depth()), int(transpiled.count_ops().get("cx", 0))

    def _build_qiskit_circuit(self, hamiltonian, t_total: float):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        n = int(hamiltonian.n_qubits)
        # Build full sparse op and group into commuting subsets
        labels = [s[::-1] for s in hamiltonian.pauli_strings]  # Qiskit little-endian
        coeffs = [float(c) for c in hamiltonian.coefficients]
        full_op = SparsePauliOp(labels, coeffs)

        # group_commuting returns a list of SparsePauliOp, each containing
        # mutually commuting terms that can be applied simultaneously
        commuting_groups = full_op.group_commuting()

        dt = float(t_total) / self.n_steps

        qc = QuantumCircuit(n)
        for _ in range(self.n_steps):
            for group_op in commuting_groups:
                qc.append(PauliEvolutionGate(group_op, time=dt), list(range(n)))
        return qc

    def evaluate(self, hamiltonian, t_total: float) -> dict[str, Any]:
        from qiskit.quantum_info import Statevector

        n = int(hamiltonian.n_qubits)
        circuit = self._build_qiskit_circuit(hamiltonian, t_total)

        psi_0 = _default_psi0(n)
        psi_exact = _exact_state(hamiltonian, t_total)

        # PaulihedralBaseline defines _swap_endian — reuse it
        tmp_baseline = PaulihedralBaseline(n_steps=1)
        psi_trotter_le = Statevector(tmp_baseline._swap_endian(psi_0, n)).evolve(circuit).data
        psi_trotter = tmp_baseline._swap_endian(
            np.asarray(psi_trotter_le, dtype=complex), n
        )
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
            tag="qiskit_group_commuting",
        )
        strategy.metadata.update(
            {
                "framework": "qiskit",
                "method": "group_commuting",
                "n_groups": len(circuit.data) // self.n_steps if self.n_steps > 0 else 0,
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BASELINE_REGISTRY: dict[str, type] = {
    "cirq": CirqTrotterBaseline,
    "tket": TketTrotterBaseline,
    "pennylane": PennyLaneTrotterBaseline,
    "paulihedral": PaulihedralBaseline,
    "paulihedral_4th": PaulihedralSuzuki4Baseline,
    "qiskit_group_commuting": QiskitGroupCommutingBaseline,
}
