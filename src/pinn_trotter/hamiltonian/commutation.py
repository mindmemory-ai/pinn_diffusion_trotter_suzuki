"""Batch computation of commutator norm matrices for Pauli Hamiltonian terms."""

from __future__ import annotations

import numpy as np

from pinn_trotter.hamiltonian.pauli_utils import pauli_commutator_norm


def compute_commutator_norm_matrix(
    pauli_strings: list[str],
    coefficients: list[float],
) -> np.ndarray:
    """Compute M×M matrix of weighted commutator norms ‖[c_j H_j, c_k H_k]‖.

    Args:
        pauli_strings: List of M Pauli strings (e.g. ['XZ', 'ZI', ...]).
        coefficients: List of M real coefficients.

    Returns:
        (M, M) float64 ndarray where entry [j,k] = |c_j||c_k| * ‖[P_j, P_k]‖.
        Diagonal is always zero. Matrix is symmetric.

    Note:
        Performance: O(M^2 * n) where n = qubit count.
        For M ≤ 100, n ≤ 16: completes in < 100ms on modern hardware.
    """
    M = len(pauli_strings)
    result = np.zeros((M, M), dtype=np.float64)
    for j in range(M):
        for k in range(j + 1, M):
            norm = pauli_commutator_norm(pauli_strings[j], pauli_strings[k])
            weighted = abs(coefficients[j]) * abs(coefficients[k]) * norm
            result[j, k] = weighted
            result[k, j] = weighted
    return result
