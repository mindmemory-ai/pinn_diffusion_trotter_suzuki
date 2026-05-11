"""Unit tests for TrotterStrategy dataclass."""

from __future__ import annotations

import io
import pytest
import numpy as np
import h5py

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy, to_hdf5, from_hdf5


def make_valid_strategy() -> TrotterStrategy:
    return TrotterStrategy(
        grouping=[[0, 2], [1, 3]],
        orders=[2, 2],
        time_steps=[1.0, 1.0],
        n_qubits=4,
        n_terms=4,
        t_total=2.0,
    )


class TestTrotterStrategyValidation:
    def test_valid_construction(self) -> None:
        s = make_valid_strategy()
        assert s.n_terms == 4

    def test_orders_length_mismatch(self) -> None:
        with pytest.raises(AssertionError, match="orders length"):
            TrotterStrategy(
                grouping=[[0, 2], [1, 3]],
                orders=[2],  # wrong length
                time_steps=[1.0, 1.0],
                n_qubits=4, n_terms=4, t_total=2.0,
            )

    def test_time_steps_length_mismatch(self) -> None:
        with pytest.raises(AssertionError, match="time_steps length"):
            TrotterStrategy(
                grouping=[[0, 2], [1, 3]],
                orders=[2, 2],
                time_steps=[2.0],  # wrong length
                n_qubits=4, n_terms=4, t_total=2.0,
            )

    def test_invalid_order_value(self) -> None:
        with pytest.raises(AssertionError, match="orders must be in"):
            TrotterStrategy(
                grouping=[[0, 2], [1, 3]],
                orders=[2, 3],  # 3 is invalid
                time_steps=[1.0, 1.0],
                n_qubits=4, n_terms=4, t_total=2.0,
            )

    def test_time_steps_sum_mismatch(self) -> None:
        with pytest.raises(AssertionError, match="time_steps sum"):
            TrotterStrategy(
                grouping=[[0, 2], [1, 3]],
                orders=[2, 2],
                time_steps=[1.0, 0.5],  # sum = 1.5 != 2.0
                n_qubits=4, n_terms=4, t_total=2.0,
            )

    def test_grouping_incomplete(self) -> None:
        with pytest.raises(AssertionError, match="partition all"):
            TrotterStrategy(
                grouping=[[0, 2], [1]],  # missing index 3
                orders=[2, 2],
                time_steps=[1.0, 1.0],
                n_qubits=4, n_terms=4, t_total=2.0,
            )


class TestCircuitDepthEstimate:
    def test_order1_depth(self) -> None:
        s = TrotterStrategy(
            grouping=[[0, 1, 2]],
            orders=[1],
            time_steps=[1.0],
            n_qubits=4, n_terms=3, t_total=1.0,
        )
        assert s.circuit_depth_estimate() == 3

    def test_order2_depth(self) -> None:
        s = TrotterStrategy(
            grouping=[[0, 1, 2]],
            orders=[2],
            time_steps=[1.0],
            n_qubits=4, n_terms=3, t_total=1.0,
        )
        assert s.circuit_depth_estimate() == 2 * 3 - 1  # = 5

    def test_order4_depth(self) -> None:
        s = TrotterStrategy(
            grouping=[[0, 1, 2]],
            orders=[4],
            time_steps=[1.0],
            n_qubits=4, n_terms=3, t_total=1.0,
        )
        assert s.circuit_depth_estimate() == 5 * (2 * 3 - 1)  # = 25

    def test_mixed_order_depth(self) -> None:
        s = make_valid_strategy()  # 2 groups of size 2, both order 2
        expected = 2 * (2 * 2 - 1)  # 2 groups, each depth=3
        assert s.circuit_depth_estimate() == expected


class TestHDF5Serialization:
    def test_roundtrip(self) -> None:
        original = make_valid_strategy()
        original.metadata["test_key"] = "hello"

        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            to_hdf5(f.require_group("strategy"), original)

        buf.seek(0)
        with h5py.File(buf, "r") as f:
            loaded = from_hdf5(f["strategy"])

        assert loaded.grouping == original.grouping
        assert loaded.orders == original.orders
        assert np.allclose(loaded.time_steps, original.time_steps)
        assert loaded.n_qubits == original.n_qubits
        assert loaded.n_terms == original.n_terms
        assert abs(loaded.t_total - original.t_total) < 1e-10
        assert loaded.metadata.get("test_key") == "hello"
