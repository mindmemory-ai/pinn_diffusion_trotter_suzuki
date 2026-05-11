"""Encoding and decoding between TrotterStrategy and tensor representations."""

from __future__ import annotations

import numpy as np
import torch

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


_ORDER_TO_IDX = {1: 0, 2: 1, 4: 2}
_IDX_TO_ORDER = {0: 1, 1: 2, 2: 4}


def strategy_to_tensor(
    strategy: TrotterStrategy,
    max_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a TrotterStrategy into three tensors.

    Args:
        strategy: The strategy to encode.
        max_groups: Maximum number of groups K (for padding).

    Returns:
        grouping_labels: shape (M,) int64 — group assignment for each Pauli term.
        orders_onehot: shape (K, 3) float32 — one-hot encoding of orders {1,2,4}.
        time_steps_normalized: shape (K,) float32 — time steps normalized to sum=1.
    """
    M = strategy.n_terms
    K = len(strategy.grouping)

    # Group assignment labels
    grouping_labels = torch.zeros(M, dtype=torch.long)
    for group_idx, group in enumerate(strategy.grouping):
        for term_idx in group:
            grouping_labels[term_idx] = group_idx

    # Orders one-hot, padded to max_groups
    orders_onehot = torch.zeros(max_groups, 3, dtype=torch.float32)
    for i, order in enumerate(strategy.orders):
        orders_onehot[i, _ORDER_TO_IDX[order]] = 1.0

    # Time steps normalized, padded
    time_steps_normalized = torch.zeros(max_groups, dtype=torch.float32)
    ts = torch.tensor(strategy.time_steps, dtype=torch.float32)
    ts_norm = ts / ts.sum()
    time_steps_normalized[:K] = ts_norm

    return grouping_labels, orders_onehot, time_steps_normalized


def tensor_to_strategy(
    grouping_labels: torch.Tensor,
    orders_onehot: torch.Tensor,
    time_steps: torch.Tensor,
    n_qubits: int,
    t_total: float,
) -> TrotterStrategy:
    """Decode tensors back to a TrotterStrategy.

    Args:
        grouping_labels: shape (M,) int64 — group assignment for each Pauli term.
        orders_onehot: shape (K, 3) float32 — one-hot orders.
        time_steps: shape (K,) float32 — normalized time steps (must sum to 1).
        n_qubits: Number of qubits.
        t_total: Total evolution time.

    Returns:
        A valid TrotterStrategy after fixing any invalid groupings.
    """
    M = int(grouping_labels.shape[0])
    K_max = int(orders_onehot.shape[0])

    labels = grouping_labels.cpu().numpy()
    used_groups = sorted(set(int(l) for l in labels))
    n_used = len(used_groups)

    # Remap labels to contiguous 0..n_used-1 (canonical ordering)
    remap = {old: new for new, old in enumerate(used_groups)}
    canonical_labels = np.array([remap[int(l)] for l in labels])

    # Sort groups so group 0 contains the smallest Pauli index
    group_min_idx = [M] * n_used
    for term_idx, g in enumerate(canonical_labels):
        if term_idx < group_min_idx[g]:
            group_min_idx[g] = term_idx
    sorted_groups = sorted(range(n_used), key=lambda g: group_min_idx[g])
    final_remap = {old: new for new, old in enumerate(sorted_groups)}
    final_labels = np.array([final_remap[int(l)] for l in canonical_labels])

    # Build grouping list
    grouping: list[list[int]] = [[] for _ in range(n_used)]
    for term_idx, g in enumerate(final_labels):
        grouping[g].append(term_idx)

    # Extract orders (argmax of one-hot)
    orders_arr = orders_onehot[:n_used].cpu().numpy()
    orders = [_IDX_TO_ORDER[int(np.argmax(orders_arr[i]))] for i in range(n_used)]

    # Extract and rescale time steps
    ts = time_steps[:n_used].cpu().numpy().astype(np.float64)
    ts = np.abs(ts)  # ensure non-negative
    ts_sum = ts.sum()
    if ts_sum < 1e-10:
        ts = np.ones(n_used) / n_used
    else:
        ts = ts / ts_sum
    time_steps_final = (ts * t_total).tolist()

    return TrotterStrategy(
        grouping=grouping,
        orders=orders,
        time_steps=time_steps_final,
        n_qubits=n_qubits,
        n_terms=M,
        t_total=t_total,
    )
