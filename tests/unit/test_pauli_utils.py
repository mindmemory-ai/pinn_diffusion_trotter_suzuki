"""Unit tests for Pauli utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from pinn_trotter.hamiltonian.pauli_utils import (
    locality,
    pauli_commutator_norm,
    pauli_string_to_matrix,
    pauli_to_sparse,
)


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


class TestPauliStringToMatrix:
    def test_single_x(self) -> None:
        np.testing.assert_allclose(pauli_string_to_matrix("X"), _X)

    def test_single_z(self) -> None:
        np.testing.assert_allclose(pauli_string_to_matrix("Z"), _Z)

    def test_xz_kron(self) -> None:
        expected = np.kron(_X, _Z)
        np.testing.assert_allclose(pauli_string_to_matrix("XZ"), expected)

    def test_identity(self) -> None:
        np.testing.assert_allclose(pauli_string_to_matrix("II"), np.eye(4, dtype=complex))

    def test_shape(self) -> None:
        assert pauli_string_to_matrix("XIZI").shape == (16, 16)


class TestPauliToSparse:
    def test_sparse_equals_dense(self) -> None:
        s = "XZ"
        coeff = 2.5
        dense = coeff * pauli_string_to_matrix(s)
        sparse = pauli_to_sparse(s, coeff).toarray()
        np.testing.assert_allclose(sparse, dense)


class TestCommutatorNorm:
    def test_xz_commutator_norm(self) -> None:
        # [X, Z] = XZ - ZX = (-iY)(2) → ‖[X,Z]‖ = 2
        norm = pauli_commutator_norm("X", "Z")
        assert abs(norm - 2.0) < 1e-12

    def test_commuting_gives_zero(self) -> None:
        # [X, X] = 0
        assert pauli_commutator_norm("X", "X") == 0.0

    def test_diagonal_paulis_commute(self) -> None:
        # [Z, Z] = 0
        assert pauli_commutator_norm("Z", "Z") == 0.0
        # [I, Z] = 0
        assert pauli_commutator_norm("I", "Z") == 0.0

    def test_xz_on_separate_qubits_commute(self) -> None:
        # XI and IZ act on different qubits → commute
        assert pauli_commutator_norm("XI", "IZ") == 0.0

    def test_anti_commuting_pair(self) -> None:
        # XY anticommute: [X,Y] = 2iZ → norm = 2
        norm = pauli_commutator_norm("X", "Y")
        assert abs(norm - 2.0) < 1e-12

    def test_verify_against_matrix(self) -> None:
        # [XZ, ZX] computed analytically vs matrix commutator
        s1, s2 = "XZ", "ZX"
        norm_fast = pauli_commutator_norm(s1, s2)
        M1 = pauli_string_to_matrix(s1)
        M2 = pauli_string_to_matrix(s2)
        comm_matrix = M1 @ M2 - M2 @ M1
        # Operator norm of commutator = max singular value
        norm_matrix = np.linalg.norm(comm_matrix, ord=2)
        assert abs(norm_fast - norm_matrix) < 1e-10


class TestLocality:
    def test_all_identity(self) -> None:
        assert locality("III") == 0

    def test_single_x(self) -> None:
        assert locality("XIZ") == 2  # X and Z are non-identity

    def test_full_locality(self) -> None:
        assert locality("XXXX") == 4

    def test_mixed(self) -> None:
        assert locality("XIZI") == 2
