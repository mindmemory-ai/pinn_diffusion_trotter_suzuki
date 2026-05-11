"""Standard benchmark Hamiltonian factories."""

from __future__ import annotations

from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph


def make_tfim(
    n_qubits: int,
    J: float = 1.0,
    h: float = 0.5,
    boundary: str = "periodic",
) -> HamiltonianGraph:
    """Transverse-field Ising model: H = -J Σ_{<ij>} Z_i Z_j - h Σ_i X_i.

    Args:
        n_qubits: Number of qubits (sites).
        J: ZZ coupling strength.
        h: Transverse field strength.
        boundary: 'periodic' or 'open'.
    """
    pauli_strings: list[str] = []
    coefficients: list[float] = []

    # ZZ interaction terms
    n_bonds = n_qubits if boundary == "periodic" else n_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % n_qubits
        s = ["I"] * n_qubits
        s[i] = "Z"
        s[j] = "Z"
        pauli_strings.append("".join(s))
        coefficients.append(-J)

    # Transverse field terms
    for i in range(n_qubits):
        s = ["I"] * n_qubits
        s[i] = "X"
        pauli_strings.append("".join(s))
        coefficients.append(-h)

    return HamiltonianGraph(pauli_strings, coefficients, n_qubits)


def make_heisenberg(
    n_qubits: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    boundary: str = "periodic",
) -> HamiltonianGraph:
    """Heisenberg model: H = Σ_{<ij>} (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j).

    Args:
        n_qubits: Number of qubits.
        Jx, Jy, Jz: Coupling strengths for XX, YY, ZZ interactions.
        boundary: 'periodic' or 'open'.
    """
    pauli_strings: list[str] = []
    coefficients: list[float] = []

    n_bonds = n_qubits if boundary == "periodic" else n_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % n_qubits
        for pauli, coeff in [("X", Jx), ("Y", Jy), ("Z", Jz)]:
            s = ["I"] * n_qubits
            s[i] = pauli
            s[j] = pauli
            pauli_strings.append("".join(s))
            coefficients.append(coeff)

    return HamiltonianGraph(pauli_strings, coefficients, n_qubits)


def generate_h2_hamiltonian(bond_length: float, basis: str = "sto-3g") -> HamiltonianGraph:
    """Generate H₂ Hamiltonian via PySCF + Qiskit Nature + Jordan-Wigner mapping.

    Args:
        bond_length: H-H bond length in Angstrom.
        basis: Basis set string for PySCF.

    Returns:
        HamiltonianGraph for the H₂ electronic Hamiltonian.
    """
    pyscf = __import__("pyscf")
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    return HamiltonianGraph.from_sparse_pauli_op(qubit_op)


def generate_lih_hamiltonian(bond_length: float, basis: str = "sto-3g") -> HamiltonianGraph:
    """Generate LiH Hamiltonian via PySCF + Qiskit Nature + Jordan-Wigner mapping.

    Args:
        bond_length: Li-H bond length in Angstrom.
        basis: Basis set string for PySCF.

    Returns:
        HamiltonianGraph for the LiH electronic Hamiltonian.
    """
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit

    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {bond_length}",
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    return HamiltonianGraph.from_sparse_pauli_op(qubit_op)
