"""Numerical consistency checks for exact fidelity backends."""

from __future__ import annotations

import numpy as np

from pinn_trotter.benchmarks.hamiltonians import make_tfim
from pinn_trotter.data.generator import compute_exact_fidelity_from_hamiltonian
from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


def test_expm_and_rk45_fidelity_match_on_tfim_4q():
    h = make_tfim(n_qubits=4, J=1.0, h=0.5, boundary="periodic")
    strategy = TrotterStrategy(
        grouping=[list(range(h.n_terms))],
        orders=[2],
        time_steps=[0.8],
        n_qubits=h.n_qubits,
        n_terms=h.n_terms,
        t_total=0.8,
    )
    psi_0 = np.zeros(2**h.n_qubits, dtype=complex)
    psi_0[0] = 1.0

    f_expm, _ = compute_exact_fidelity_from_hamiltonian(
        hamiltonian=h,
        strategy=strategy,
        psi_0=psi_0,
        exact_method="expm",
    )
    f_rk45, _ = compute_exact_fidelity_from_hamiltonian(
        hamiltonian=h,
        strategy=strategy,
        psi_0=psi_0,
        exact_method="rk45",
    )

    assert abs(f_expm - f_rk45) < 1e-8
