"""Unit tests for commutation.py and pauli_commutator_norm (TODO 10-1).

Verifies analytic known values:
  - [X, Z] anti-commutes → norm = 2
  - [Z, Z] commutes → norm = 0
  - [I, P] always commutes → norm = 0
  - compute_commutator_norm_matrix: diagonal zero, symmetric, weighted correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn_trotter.hamiltonian.commutation import compute_commutator_norm_matrix
from pinn_trotter.hamiltonian.pauli_utils import pauli_commutator_norm


# ---------------------------------------------------------------------------
# pauli_commutator_norm
# ---------------------------------------------------------------------------

class TestPauliCommutatorNorm:
    def test_X_Z_anticommutes(self):
        """[X, Z] should yield norm 2 (they anti-commute, so [X,Z] = 2i*Y)."""
        assert pauli_commutator_norm("X", "Z") == pytest.approx(2.0)

    def test_Z_X_anticommutes_symmetric(self):
        """Norm is symmetric: ‖[X,Z]‖ == ‖[Z,X]‖."""
        assert pauli_commutator_norm("Z", "X") == pytest.approx(2.0)

    def test_X_Y_anticommutes(self):
        """[X, Y] anti-commutes → norm = 2."""
        assert pauli_commutator_norm("X", "Y") == pytest.approx(2.0)

    def test_Y_Z_anticommutes(self):
        """[Y, Z] anti-commutes → norm = 2."""
        assert pauli_commutator_norm("Y", "Z") == pytest.approx(2.0)

    def test_X_X_commutes(self):
        """[X, X] = 0 (same operator)."""
        assert pauli_commutator_norm("X", "X") == pytest.approx(0.0)

    def test_Z_Z_commutes(self):
        """[Z, Z] = 0 (diagonal Pauli commutes with itself)."""
        assert pauli_commutator_norm("Z", "Z") == pytest.approx(0.0)

    def test_I_any_commutes(self):
        """[I, P] = 0 for any single-qubit Pauli P."""
        for p in ("X", "Y", "Z", "I"):
            assert pauli_commutator_norm("I", p) == pytest.approx(0.0), f"[I, {p}] should be 0"

    def test_ZI_IZ_commutes(self):
        """ZI and IZ act on different qubits → commute → norm = 0."""
        assert pauli_commutator_norm("ZI", "IZ") == pytest.approx(0.0)

    def test_XI_IX_commutes(self):
        """XI and IX act on different qubits → commute → norm = 0."""
        assert pauli_commutator_norm("XI", "IX") == pytest.approx(0.0)

    def test_XI_ZI_anticommutes(self):
        """XI and ZI: X and Z anti-commute on qubit 0 → total norm = 2."""
        assert pauli_commutator_norm("XI", "ZI") == pytest.approx(2.0)

    def test_XX_ZZ_commutes(self):
        """XX and ZZ: each qubit contributes a sign flip, two sign flips → commute."""
        # XX·ZZ = (XZ)(XZ) = (-iY)(-iY) → phase product
        # ZZ·XX = (ZX)(ZX) = (iY)(iY)   → phase product
        # net: they commute (two anti-commuting pairs cancel)
        assert pauli_commutator_norm("XX", "ZZ") == pytest.approx(0.0)

    def test_XZ_ZX_commutes(self):
        """XZ and ZX: anti-commuting on both qubits cancels → commute overall."""
        # XZ·ZX = (X·Z)(Z·X) = (-iY)(iY) = I  (phase = +1)
        # ZX·XZ = (Z·X)(X·Z) = (iY)(-iY) = I  (phase = +1)
        # Same phase → commute
        assert pauli_commutator_norm("XZ", "ZX") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_commutator_norm_matrix
# ---------------------------------------------------------------------------

class TestComputeCommutatorNormMatrix:
    def test_shape(self):
        pauli_strings = ["XI", "IZ", "XX"]
        coefficients = [1.0, 1.0, 1.0]
        M = compute_commutator_norm_matrix(pauli_strings, coefficients)
        assert M.shape == (3, 3)

    def test_diagonal_zero(self):
        """Diagonal must always be zero ([P, P] = 0)."""
        pauli_strings = ["X", "Y", "Z", "I"]
        coefficients = [1.0, 2.0, 0.5, 1.0]
        M = compute_commutator_norm_matrix(pauli_strings, coefficients)
        np.testing.assert_array_almost_equal(np.diag(M), 0.0)

    def test_symmetric(self):
        """Matrix must be symmetric."""
        pauli_strings = ["X", "Y", "Z"]
        coefficients = [1.0, 2.0, 3.0]
        M = compute_commutator_norm_matrix(pauli_strings, coefficients)
        np.testing.assert_array_almost_equal(M, M.T)

    def test_weighted_correctly(self):
        """Entry [j,k] = |c_j| * |c_k| * ‖[P_j, P_k]‖."""
        pauli_strings = ["X", "Z"]
        c = [2.0, 3.0]
        M = compute_commutator_norm_matrix(pauli_strings, c)
        # ‖[X,Z]‖ = 2, weighted: 2.0 * 3.0 * 2 = 12
        assert M[0, 1] == pytest.approx(12.0)
        assert M[1, 0] == pytest.approx(12.0)

    def test_commuting_pair_zero(self):
        """Commuting pair produces zero entry."""
        pauli_strings = ["ZI", "IZ"]  # act on different qubits
        coefficients = [5.0, 7.0]
        M = compute_commutator_norm_matrix(pauli_strings, coefficients)
        assert M[0, 1] == pytest.approx(0.0)

    def test_negative_coefficients_use_abs(self):
        """Negative coefficients are treated with |c_j|."""
        pauli_strings = ["X", "Z"]
        c_pos = [2.0, 3.0]
        c_neg = [-2.0, -3.0]
        M_pos = compute_commutator_norm_matrix(pauli_strings, c_pos)
        M_neg = compute_commutator_norm_matrix(pauli_strings, c_neg)
        np.testing.assert_array_almost_equal(M_pos, M_neg)

    def test_single_term(self):
        """Single-term Hamiltonian → 1×1 zero matrix."""
        M = compute_commutator_norm_matrix(["X"], [1.0])
        assert M.shape == (1, 1)
        assert M[0, 0] == pytest.approx(0.0)

    def test_all_commuting_tfim_transverse_terms(self):
        """IZ, ZI: transverse-field terms acting on separate qubits commute."""
        pauli_strings = ["IZ", "ZI"]
        M = compute_commutator_norm_matrix(pauli_strings, [1.0, 1.0])
        np.testing.assert_array_almost_equal(M, 0.0)
