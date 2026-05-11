"""Pauli operator utility functions."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULI_MAP: dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

# Pauli multiplication table: P_a @ P_b = coeff * P_c
# Stored as (result_char, complex_coeff)
_PAULI_PRODUCT: dict[tuple[str, str], tuple[str, complex]] = {
    ("I", "I"): ("I", 1),
    ("I", "X"): ("X", 1),
    ("I", "Y"): ("Y", 1),
    ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1),
    ("X", "X"): ("I", 1),
    ("X", "Y"): ("Z", 1j),
    ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Y"): ("I", 1),
    ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1),
    ("Z", "X"): ("Y", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "Z"): ("I", 1),
}


def pauli_string_to_matrix(s: str) -> np.ndarray:
    """Convert a Pauli string (e.g. 'XIZI') to a 2^n × 2^n dense matrix.

    The leftmost character acts on qubit 0 (most significant qubit).
    """
    result = _PAULI_MAP[s[0]]
    for char in s[1:]:
        result = np.kron(result, _PAULI_MAP[char])
    return result


def pauli_to_sparse(s: str, coeff: float) -> sp.csr_matrix:
    """Convert a weighted Pauli string to a sparse CSR matrix."""
    return sp.csr_matrix(coeff * pauli_string_to_matrix(s))


def _multiply_pauli_strings(s1: str, s2: str) -> tuple[str, complex]:
    """Multiply two Pauli strings site-wise, return (result_string, overall_phase)."""
    result_chars = []
    phase: complex = 1.0
    for a, b in zip(s1, s2):
        char, c = _PAULI_PRODUCT[(a, b)]
        result_chars.append(char)
        phase *= c
    return "".join(result_chars), phase


def pauli_commutator_norm(s1: str, s2: str) -> float:
    """Compute ‖[P1, P2]‖ using Pauli multiplication rules in O(n) time.

    Since [A, B] = AB - BA, and both products are Pauli strings (up to phase),
    the commutator is either 0 (if they commute) or 2*|coeff|*(Pauli string).
    The operator norm of a Pauli string is 1, so ‖[P1, P2]‖ = 0 or 2.
    """
    _, phase_ab = _multiply_pauli_strings(s1, s2)
    _, phase_ba = _multiply_pauli_strings(s2, s1)
    # [P1,P2] = 0 iff phase_ab == phase_ba (i.e., they commute)
    if abs(phase_ab - phase_ba) < 1e-12:
        return 0.0
    # ‖[P1,P2]‖ = |phase_ab - phase_ba| (operator norm of result Pauli string is 1)
    return float(abs(phase_ab - phase_ba))


def locality(s: str) -> int:
    """Count the number of non-identity factors in a Pauli string."""
    return sum(1 for c in s if c != "I")
