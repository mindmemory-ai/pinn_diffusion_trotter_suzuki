"""Validation tests for molecular Hamiltonian generators."""

from __future__ import annotations

import numpy as np

from pinn_trotter.benchmarks.hamiltonians import generate_h2_hamiltonian


def test_h2_ground_state_energy_close_to_reference():
    """H2@0.74A (STO-3G) total ground energy should be around -1.117 Ha."""
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit

    bond_length = 0.74
    h_graph = generate_h2_hamiltonian(bond_length=bond_length, basis="sto-3g")
    e_electronic = float(np.linalg.eigvalsh(h_graph.to_dense_matrix()).min())

    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    qubit_op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    e_electronic_ref = float(np.linalg.eigvalsh(qubit_op.to_matrix()).min())
    e_nuclear = float(problem.hamiltonian.constants.get("nuclear_repulsion_energy", 0.0))
    e_total = e_electronic + e_nuclear

    # Generator should match direct Qiskit-Nature mapping.
    assert abs(e_electronic - e_electronic_ref) < 1e-8
    # Reference in TODO is ~-1.117 Ha; allow a reasonable chemistry tolerance.
    assert abs(e_total - (-1.117)) < 0.03
