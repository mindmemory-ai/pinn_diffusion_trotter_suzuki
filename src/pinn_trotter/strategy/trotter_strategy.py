"""TrotterStrategy dataclass: core data structure for Trotter-Suzuki decomposition strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np


@dataclass
class TrotterStrategy:
    """Represents a Trotter-Suzuki decomposition strategy π = (G, k, τ).

    Attributes:
        grouping: List of groups, each group is a list of Pauli term indices.
        orders: Suzuki order for each group (must be 1, 2, or 4).
        time_steps: Time duration for each group step (must sum to t_total).
        n_qubits: Number of qubits in the system.
        n_terms: Total number of Pauli terms in the Hamiltonian.
        t_total: Total evolution time.
        metadata: Optional metadata dict for storing extra info.
    """

    grouping: list[list[int]]
    orders: list[int]
    time_steps: list[float]
    n_qubits: int
    n_terms: int
    t_total: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K = len(self.grouping)
        assert len(self.orders) == K, (
            f"orders length {len(self.orders)} must match grouping length {K}"
        )
        assert len(self.time_steps) == K, (
            f"time_steps length {len(self.time_steps)} must match grouping length {K}"
        )
        assert all(o in (1, 2, 4) for o in self.orders), (
            f"All orders must be in {{1, 2, 4}}, got {self.orders}"
        )
        assert abs(sum(self.time_steps) - self.t_total) < 1e-8, (
            f"time_steps sum {sum(self.time_steps):.6f} must equal t_total {self.t_total:.6f}"
        )
        all_indices = sorted(idx for group in self.grouping for idx in group)
        assert all_indices == list(range(self.n_terms)), (
            f"Grouping must partition all {self.n_terms} Pauli indices; "
            f"got {all_indices}"
        )

    def circuit_depth_estimate(self) -> int:
        """Estimate circuit depth based on grouping structure and Suzuki orders.

        Returns:
            Estimated circuit depth (number of two-qubit gate layers).
        """
        total_depth = 0
        for group, order in zip(self.grouping, self.orders):
            m = len(group)
            if order == 1:
                # First-order: one sweep through group
                total_depth += m
            elif order == 2:
                # Second-order symmetric: forward + backward sweep, minus shared boundary
                total_depth += 2 * m - 1
            elif order == 4:
                # Fourth-order Suzuki: 5 sub-steps of 2nd order
                # p1 = p2 = p4 = p5 = 1/(4-4^(1/3)), p3 = 1 - 4*p1
                # Each sub-step applies 2nd-order with depth 2m-1
                total_depth += 5 * (2 * m - 1)
        return total_depth


def to_hdf5(group: h5py.Group, strategy: TrotterStrategy) -> None:
    """Serialize a TrotterStrategy to an open HDF5 group.

    Schema:
        grouping/         dataset (ragged via variable-length dtype)
        orders            dataset shape (K,) int32
        time_steps        dataset shape (K,) float64
        n_qubits          attribute int
        n_terms           attribute int
        t_total           attribute float
        metadata/         subgroup (key-value string attributes)
    """
    K = len(strategy.grouping)
    # Store flat grouping + offsets (compressed ragged array)
    flat = np.array([idx for g in strategy.grouping for idx in g], dtype=np.int32)
    offsets = np.cumsum([0] + [len(g) for g in strategy.grouping], dtype=np.int32)
    group.create_dataset("grouping_flat", data=flat)
    group.create_dataset("grouping_offsets", data=offsets)
    group.create_dataset("orders", data=np.array(strategy.orders, dtype=np.int32))
    group.create_dataset("time_steps", data=np.array(strategy.time_steps, dtype=np.float64))
    group.attrs["n_qubits"] = strategy.n_qubits
    group.attrs["n_terms"] = strategy.n_terms
    group.attrs["t_total"] = strategy.t_total
    group.attrs["K"] = K

    meta_grp = group.require_group("metadata")
    for k, v in strategy.metadata.items():
        meta_grp.attrs[k] = str(v)


def from_hdf5(group: h5py.Group) -> TrotterStrategy:
    """Deserialize a TrotterStrategy from an open HDF5 group."""
    flat = group["grouping_flat"][:]
    offsets = group["grouping_offsets"][:]
    grouping = [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    orders = group["orders"][:].tolist()
    time_steps = group["time_steps"][:].tolist()
    n_qubits = int(group.attrs["n_qubits"])
    n_terms = int(group.attrs["n_terms"])
    t_total = float(group.attrs["t_total"])

    metadata: dict[str, Any] = {}
    if "metadata" in group:
        metadata = {k: v for k, v in group["metadata"].attrs.items()}

    return TrotterStrategy(
        grouping=grouping,
        orders=orders,
        time_steps=time_steps,
        n_qubits=n_qubits,
        n_terms=n_terms,
        t_total=t_total,
        metadata=metadata,
    )
