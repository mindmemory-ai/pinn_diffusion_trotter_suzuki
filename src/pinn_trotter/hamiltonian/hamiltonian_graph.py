"""HamiltonianGraph: core class representing a Pauli Hamiltonian as a graph."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from pinn_trotter.hamiltonian.commutation import compute_commutator_norm_matrix
from pinn_trotter.hamiltonian.pauli_utils import (
    locality,
    pauli_string_to_matrix,
    pauli_to_sparse,
)


class HamiltonianGraph:
    """Represents a Pauli Hamiltonian H = Σ_j c_j P_j as a commutation graph.

    Attributes:
        pauli_strings: List of M Pauli strings.
        coefficients: Array of M real coefficients.
        n_qubits: Number of qubits.
        n_terms: M, number of Pauli terms.
    """

    def __init__(
        self,
        pauli_strings: list[str],
        coefficients: list[float] | np.ndarray,
        n_qubits: int,
    ) -> None:
        self.pauli_strings = pauli_strings
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.n_qubits = n_qubits
        self.n_terms = len(pauli_strings)
        assert len(self.coefficients) == self.n_terms
        assert all(len(s) == n_qubits for s in pauli_strings), (
            "All Pauli strings must have length equal to n_qubits"
        )
        self._comm_matrix: np.ndarray | None = None
        self._norm_matrix: np.ndarray | None = None

    @classmethod
    def from_sparse_pauli_op(cls, op: object) -> "HamiltonianGraph":
        """Construct from a Qiskit SparsePauliOp."""
        # op.paulis.to_labels() returns strings in little-endian (qubit 0 rightmost)
        # We store big-endian (qubit 0 leftmost) for consistency
        pauli_list = [str(p)[::-1] for p in op.paulis]  # type: ignore[attr-defined]
        coeffs = [float(c.real) for c in op.coeffs]  # type: ignore[attr-defined]
        n_qubits = op.num_qubits  # type: ignore[attr-defined]
        return cls(pauli_list, coeffs, n_qubits)

    @classmethod
    def from_dict(cls, pauli_dict: dict[str, float], n_qubits: int) -> "HamiltonianGraph":
        """Construct from a dict mapping Pauli string -> coefficient."""
        pauli_strings = list(pauli_dict.keys())
        coefficients = [pauli_dict[s] for s in pauli_strings]
        return cls(pauli_strings, coefficients, n_qubits)

    def commutator_norms(self) -> np.ndarray:
        """Return (M, M) float64 array of weighted commutator norms.

        Cached after first call.
        """
        if self._norm_matrix is None:
            self._norm_matrix = compute_commutator_norm_matrix(
                self.pauli_strings, self.coefficients.tolist()
            )
        return self._norm_matrix

    def compute_commutation_matrix(self, threshold: float = 1e-10) -> np.ndarray:
        """Return (M, M) boolean array: True where terms commute (norm < threshold).

        Cached after first call.
        """
        if self._comm_matrix is None:
            norms = self.commutator_norms()
            self._comm_matrix = norms < threshold
        return self._comm_matrix

    def to_pyg_data(self) -> "Data":
        """Convert to a PyTorch Geometric Data object for GNN input.

        Node features (per Pauli term):
            - coefficient (1)
            - locality (1)
            - support one-hot: which qubits are non-identity (n_qubits)
          Total: n_qubits + 2 features per node.

        Edge index: all non-commuting pairs (i, j) and (j, i).

        Edge features (per non-commuting pair):
            - commutator_norm
            - shared qubit count (number of qubits where both are non-I)
            - commuting flag (always 0 for edges, 1 for commuting — but we only
              add edges for non-commuting pairs; commuting pairs have no edge)
        """
        from torch_geometric.data import Data
        M = self.n_terms
        n = self.n_qubits

        # Node features
        node_feats = np.zeros((M, n + 2), dtype=np.float32)
        for i, (s, c) in enumerate(zip(self.pauli_strings, self.coefficients)):
            node_feats[i, 0] = float(c)
            node_feats[i, 1] = float(locality(s))
            for q, char in enumerate(s):
                node_feats[i, 2 + q] = 0.0 if char == "I" else 1.0

        # Edges: non-commuting pairs
        norm_matrix = self.commutator_norms()
        comm_matrix = self.compute_commutation_matrix()

        src_list, dst_list, edge_feat_list = [], [], []
        for i in range(M):
            for j in range(i + 1, M):
                if not comm_matrix[i, j]:
                    # Count shared non-I qubits
                    shared = sum(
                        1
                        for si, sj in zip(self.pauli_strings[i], self.pauli_strings[j])
                        if si != "I" and sj != "I"
                    )
                    feat = [norm_matrix[i, j], float(shared), 0.0]
                    # Add both directions
                    src_list += [i, j]
                    dst_list += [j, i]
                    edge_feat_list += [feat, feat]

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)

        return Data(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_qubits=n,
            n_terms=M,
        )

    def to_sparse_matrix(self) -> sp.csr_matrix:
        """Build sparse Hamiltonian matrix H = Σ_j c_j P_j.

        Only supported for n_qubits ≤ 16.
        """
        if self.n_qubits > 16:
            raise ValueError(
                f"to_sparse_matrix only supports n_qubits ≤ 16, got {self.n_qubits}"
            )
        dim = 2**self.n_qubits
        result = sp.csr_matrix((dim, dim), dtype=complex)
        for s, c in zip(self.pauli_strings, self.coefficients):
            result = result + pauli_to_sparse(s, c)
        return result

    def to_dense_matrix(self) -> np.ndarray:
        """Build dense Hamiltonian matrix. Only supported for n_qubits ≤ 12."""
        if self.n_qubits > 12:
            raise ValueError(
                f"to_dense_matrix only supports n_qubits ≤ 12, got {self.n_qubits}"
            )
        dim = 2**self.n_qubits
        result = np.zeros((dim, dim), dtype=complex)
        for s, c in zip(self.pauli_strings, self.coefficients):
            result += c * pauli_string_to_matrix(s)
        return result
