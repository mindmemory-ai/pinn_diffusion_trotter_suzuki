"""Unit tests for HamiltonianGraph."""

from __future__ import annotations

import numpy as np
import pytest

from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph


def make_tfim_4q(J: float = 1.0, h: float = 0.5) -> HamiltonianGraph:
    """TFIM Hamiltonian for 4 qubits: H = -J Σ ZZ - h Σ X."""
    pauli_strings = [
        "ZZII", "IZZI", "IIZZ", "ZIIZ",  # ZZ terms (periodic)
        "XIII", "IXII", "IIXI", "IIIX",  # X terms
    ]
    coefficients = [-J, -J, -J, -J, -h, -h, -h, -h]
    return HamiltonianGraph(pauli_strings, coefficients, n_qubits=4)


class TestHamiltonianGraphConstruction:
    def test_from_dict(self) -> None:
        H = HamiltonianGraph.from_dict({"ZZ": 1.0, "XI": -0.5}, n_qubits=2)
        assert H.n_terms == 2
        assert H.n_qubits == 2

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(AssertionError):
            HamiltonianGraph(["XXX", "Z"], [1.0, 1.0], n_qubits=3)


class TestCommutationMatrix:
    def test_zz_x_commutation(self) -> None:
        H = make_tfim_4q()
        comm = H.compute_commutation_matrix()
        assert comm.shape == (8, 8)
        # ZZ terms commute with each other
        for i in range(4):
            for j in range(4):
                assert comm[i, j], f"ZZ terms {i},{j} should commute"
        # X terms commute with each other
        for i in range(4, 8):
            for j in range(4, 8):
                assert comm[i, j], f"X terms {i},{j} should commute"

    def test_norm_matrix_symmetry(self) -> None:
        H = make_tfim_4q()
        norms = H.commutator_norms()
        np.testing.assert_allclose(norms, norms.T)

    def test_norm_matrix_diagonal_zero(self) -> None:
        H = make_tfim_4q()
        norms = H.commutator_norms()
        np.testing.assert_allclose(np.diag(norms), 0.0)


class TestPyGData:
    def test_node_shape(self) -> None:
        pytest.importorskip("torch_geometric")
        H = make_tfim_4q()
        data = H.to_pyg_data()
        # n_qubits + 2 features per node
        assert data.x.shape == (8, 4 + 2)

    def test_edge_index_shape(self) -> None:
        pytest.importorskip("torch_geometric")
        H = make_tfim_4q()
        data = H.to_pyg_data()
        assert data.edge_index.shape[0] == 2

    def test_edge_attr_shape(self) -> None:
        pytest.importorskip("torch_geometric")
        H = make_tfim_4q()
        data = H.to_pyg_data()
        n_edges = data.edge_index.shape[1]
        assert data.edge_attr.shape == (n_edges, 3)


class TestSparseMatrix:
    def test_sparse_matches_qiskit(self) -> None:
        pytest.importorskip("qiskit")
        from qiskit.quantum_info import SparsePauliOp
        # Small 2-qubit system
        H_dict = {"ZZ": 1.0, "XI": -0.5, "IX": -0.5}
        H = HamiltonianGraph.from_dict(H_dict, n_qubits=2)
        H_sparse = H.to_sparse_matrix().toarray()

        # Build reference via Qiskit
        # Qiskit uses little-endian, so reverse Pauli strings
        pauli_list = [(k[::-1], v) for k, v in H_dict.items()]
        op = SparsePauliOp.from_list(pauli_list)
        H_ref = op.to_matrix()
        np.testing.assert_allclose(H_sparse, H_ref, atol=1e-10)

    def test_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="≤ 16"):
            H = HamiltonianGraph(["X" * 17], [1.0], n_qubits=17)
            H.to_sparse_matrix()

    def test_dense_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="≤ 12"):
            H = HamiltonianGraph(["X" * 13], [1.0], n_qubits=13)
            H.to_dense_matrix()
