"""Unit tests for strategy encoding/decoding."""

from __future__ import annotations

import torch
import pytest

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy
from pinn_trotter.strategy.encoding import strategy_to_tensor, tensor_to_strategy


def make_strategy() -> TrotterStrategy:
    return TrotterStrategy(
        grouping=[[0, 2, 4], [1, 3, 5]],
        orders=[2, 4],
        time_steps=[1.5, 0.5],
        n_qubits=4,
        n_terms=6,
        t_total=2.0,
    )


class TestStrategyToTensor:
    def test_shapes(self) -> None:
        s = make_strategy()
        labels, orders, ts = strategy_to_tensor(s, max_groups=4)
        assert labels.shape == (6,)
        assert orders.shape == (4, 3)
        assert ts.shape == (4,)

    def test_grouping_labels_correct(self) -> None:
        s = make_strategy()
        labels, _, _ = strategy_to_tensor(s, max_groups=4)
        # Terms 0,2,4 → group 0; terms 1,3,5 → group 1
        for i in [0, 2, 4]:
            assert labels[i].item() == 0
        for i in [1, 3, 5]:
            assert labels[i].item() == 1

    def test_orders_onehot(self) -> None:
        s = make_strategy()
        _, orders, _ = strategy_to_tensor(s, max_groups=4)
        # Group 0: order=2 → index 1
        assert orders[0, 1].item() == 1.0
        assert orders[0, 0].item() == 0.0
        # Group 1: order=4 → index 2
        assert orders[1, 2].item() == 1.0

    def test_time_steps_normalized(self) -> None:
        s = make_strategy()
        _, _, ts = strategy_to_tensor(s, max_groups=4)
        # Normalized: [1.5/2.0, 0.5/2.0] = [0.75, 0.25]
        assert abs(ts[0].item() - 0.75) < 1e-6
        assert abs(ts[1].item() - 0.25) < 1e-6
        assert abs(ts[2].item()) < 1e-6  # padding


class TestTensorToStrategy:
    def test_roundtrip(self) -> None:
        original = make_strategy()
        labels, orders, ts = strategy_to_tensor(original, max_groups=4)
        recovered = tensor_to_strategy(labels, orders, ts, n_qubits=4, t_total=2.0)

        assert recovered.n_terms == original.n_terms
        assert recovered.n_qubits == original.n_qubits
        assert abs(recovered.t_total - original.t_total) < 1e-6

        # Check grouping sets match (groups may be reordered)
        orig_groups = [set(g) for g in original.grouping]
        rec_groups = [set(g) for g in recovered.grouping]
        assert sorted(str(g) for g in orig_groups) == sorted(str(g) for g in rec_groups)

    def test_empty_group_repair(self) -> None:
        """Labels with a gap (empty group) should be repaired."""
        labels = torch.tensor([0, 2, 0, 2, 0, 2])  # group 1 is empty
        orders = torch.zeros(4, 3)
        orders[0, 1] = 1.0  # group 0 → order 2
        orders[1, 1] = 1.0  # group 1 (will be remapped from 2) → order 2
        ts = torch.tensor([0.5, 0.5, 0.0, 0.0])
        s = tensor_to_strategy(labels, orders, ts, n_qubits=4, t_total=2.0)
        # Should produce 2 groups, not 3
        assert len(s.grouping) == 2

    def test_all_same_group(self) -> None:
        """All terms in one group."""
        labels = torch.zeros(4, dtype=torch.long)
        orders = torch.zeros(4, 3)
        orders[0, 1] = 1.0
        ts = torch.tensor([1.0, 0.0, 0.0, 0.0])
        s = tensor_to_strategy(labels, orders, ts, n_qubits=4, t_total=3.0)
        assert len(s.grouping) == 1
        assert set(s.grouping[0]) == {0, 1, 2, 3}

    def test_negative_time_steps_repaired(self) -> None:
        """Negative time steps should be made absolute."""
        labels = torch.tensor([0, 0, 1, 1])
        orders = torch.zeros(4, 3)
        orders[0, 1] = 1.0
        orders[1, 1] = 1.0
        ts = torch.tensor([-0.3, 0.7, 0.0, 0.0])  # negative ts
        s = tensor_to_strategy(labels, orders, ts, n_qubits=4, t_total=2.0)
        for t in s.time_steps:
            assert t >= 0.0
