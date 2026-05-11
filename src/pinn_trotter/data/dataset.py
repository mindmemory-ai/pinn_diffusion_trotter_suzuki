"""PyTorch Dataset for Trotter strategy data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pinn_trotter.strategy.encoding import strategy_to_tensor
from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


class TrotterDataset(Dataset):
    """HDF5-backed PyTorch Dataset for Trotter strategy samples.

    Each item returns:
        (pyg_data_or_node_feats, strategy_tensor, fidelity, circuit_depth)

    When torch_geometric is available, pyg_data is a torch_geometric.data.Data
    object. When not available, it returns a plain dict with keys 'pauli_strings',
    'coefficients', 'n_qubits', 'n_terms'.

    Args:
        dataset_path: Path to HDF5 dataset file.
        max_groups: Maximum number of groups for strategy tensor encoding.
        augment: If True, apply random Pauli permutation augmentation.
        n_qubits_filter: If set, only load samples with this many qubits.
        h_max_filter: If set, only load samples where the transverse field h ≤ this value.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        max_groups: int = 8,
        augment: bool = False,
        n_qubits_filter: Optional[int] = None,
        h_max_filter: Optional[float] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.max_groups = max_groups
        self.augment = augment
        self.n_qubits_filter = n_qubits_filter
        self.h_max_filter = h_max_filter

        # Build index of valid sample IDs
        self._sample_ids: list[str] = []
        with h5py.File(self.dataset_path, "r") as f:
            samples = f.get("dataset_v1/samples", {})
            for sid in sorted(samples.keys()):
                h_grp = samples[sid]["hamiltonian"]
                if n_qubits_filter is not None:
                    n_q = int(h_grp.attrs["n_qubits"])
                    if n_q != n_qubits_filter:
                        continue
                if h_max_filter is not None:
                    h_val = float(h_grp.attrs.get("h", float("nan")))
                    if h_val > h_max_filter:
                        continue
                self._sample_ids.append(sid)

        try:
            import torch_geometric  # noqa: F401
            self._has_pyg = True
        except ImportError:
            self._has_pyg = False

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> tuple:
        sid = self._sample_ids[idx]
        with h5py.File(self.dataset_path, "r") as f:
            grp = f[f"dataset_v1/samples/{sid}"]

            # Hamiltonian
            pauli_strings = [s.decode() if isinstance(s, bytes) else s
                             for s in grp["hamiltonian"]["pauli_strings"][:]]
            coefficients = grp["hamiltonian"]["coefficients"][:].astype(np.float64)
            n_qubits = int(grp["hamiltonian"].attrs["n_qubits"])
            n_terms = int(grp["hamiltonian"].attrs["n_terms"])

            # Strategy
            grouping_flat = grp["strategy"]["grouping_flat"][:].astype(np.int64)
            n_groups = int(grp["strategy"].attrs["n_groups"])
            orders = grp["strategy"]["orders"][:].astype(np.int64)
            time_steps = grp["strategy"]["time_steps"][:].astype(np.float64)
            t_total = float(grp["strategy"].attrs["t_total"])

            # Labels
            fidelity = float(grp["labels"].attrs["fidelity"])
            circuit_depth = int(grp["labels"].attrs["circuit_depth"])

            # Hamiltonian physics params (for direct feature use in regression)
            J_val = float(grp["hamiltonian"].attrs.get("J", float("nan")))
            h_val = float(grp["hamiltonian"].attrs.get("h", float("nan")))

        # Reconstruct TrotterStrategy from flat grouping
        grouping: list[list[int]] = [[] for _ in range(n_groups)]
        for term_idx, g_idx in enumerate(grouping_flat):
            grouping[g_idx].append(term_idx)
        grouping = [g for g in grouping if g]  # remove empty groups

        strategy = TrotterStrategy(
            grouping=grouping,
            orders=orders.tolist(),
            time_steps=time_steps.tolist(),
            n_qubits=n_qubits,
            n_terms=n_terms,
            t_total=t_total,
        )

        if self.augment:
            pauli_strings, coefficients, strategy = _permute_pauli_labels(
                pauli_strings, coefficients, strategy
            )

        # Strategy tensor
        grouping_labels, orders_onehot, time_steps_norm = strategy_to_tensor(
            strategy, self.max_groups
        )

        # Graph data
        if self._has_pyg:
            from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph
            H = HamiltonianGraph(pauli_strings, coefficients, n_qubits)
            graph_data = H.to_pyg_data()
        else:
            graph_data = {
                "pauli_strings": pauli_strings,
                "coefficients": torch.tensor(coefficients, dtype=torch.float64),
                "n_qubits": n_qubits,
                "n_terms": n_terms,
            }

        # Physics params packed as a small tensor: [t_total, log(t_total), J, h, J*h]
        # These are critical for fidelity prediction since Trotter error ∝ t_total² × J × h
        ham_params = torch.tensor(
            [t_total,
             float(np.log(max(t_total, 1e-6))),
             J_val,
             h_val,
             J_val * h_val],
            dtype=torch.float32,
        )
        strategy_tensor = (grouping_labels, orders_onehot, time_steps_norm, ham_params)
        return graph_data, strategy_tensor, torch.tensor(fidelity, dtype=torch.float32), circuit_depth


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _permute_pauli_labels(
    pauli_strings: list[str],
    coefficients: np.ndarray,
    strategy: TrotterStrategy,
) -> tuple[list[str], np.ndarray, TrotterStrategy]:
    """Randomly permute Pauli term indices while preserving grouping structure.

    This is a data augmentation: the same Hamiltonian with terms reordered
    should yield an equivalent strategy (same groups, just different indexing).

    Returns:
        (permuted_pauli_strings, permuted_coefficients, new_strategy)
    """
    M = len(pauli_strings)
    perm = np.random.permutation(M)
    inv_perm = np.argsort(perm)

    new_pauli = [pauli_strings[perm[i]] for i in range(M)]
    new_coeffs = coefficients[perm]

    # Rebuild grouping with new indices
    new_grouping: list[list[int]] = []
    for group in strategy.grouping:
        new_group = [int(inv_perm[j]) for j in group]
        new_grouping.append(sorted(new_group))

    new_strategy = TrotterStrategy(
        grouping=new_grouping,
        orders=strategy.orders,
        time_steps=strategy.time_steps,
        n_qubits=strategy.n_qubits,
        n_terms=strategy.n_terms,
        t_total=strategy.t_total,
    )
    return new_pauli, new_coeffs, new_strategy


def permute_pauli_labels(
    strategy_tensor: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply random Pauli label permutation to an already-encoded strategy tensor.

    This version operates on the encoded tensor (grouping_labels, orders_onehot,
    time_steps_normalized) rather than raw data structures, so it can be used
    as a collate-time augmentation.

    The grouping_labels are permuted: if grouping_labels[j] = k, after permutation
    term perm[j] belongs to group k. This is equivalent to shuffling the Pauli terms.

    Returns:
        Augmented (grouping_labels, orders_onehot, time_steps_normalized).
    """
    grouping_labels, orders_onehot, time_steps_norm = strategy_tensor
    M = grouping_labels.shape[0]
    perm = torch.randperm(M)
    return grouping_labels[perm], orders_onehot, time_steps_norm
