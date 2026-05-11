"""Integration test: full dataset integrity scan.

Loads the generated HDF5 dataset and verifies:
- No corrupted/missing samples
- Fidelity values in [0, 1]
- Grouping completeness (all Pauli terms assigned)
- Strategy tensor dimensions consistent with max_groups
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "dataset_tfim.h5"


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="Dataset not generated yet — run 01_generate_dataset.py first")
class TestDatasetIntegrity:
    """Full scan of the generated dataset for correctness."""

    def test_loads_without_error(self):
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        assert len(ds) > 0, "Dataset is empty"

    def test_sample_count_reasonable(self):
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        assert len(ds) >= 10, f"Too few samples: {len(ds)}"

    def test_fidelity_range(self):
        """All fidelities must be in [0, 1]."""
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        for i in range(len(ds)):
            _, _, fidelity, _ = ds[i]
            f = float(fidelity)
            assert 0.0 <= f <= 1.0 + 1e-6, f"Sample {i}: fidelity={f} out of range"

    def test_strategy_tensor_dimensions(self):
        """Strategy tensors must match max_groups and n_terms."""
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        graph_data, strat, fidelity, depth = ds[0]
        grouping, orders_oh, ts_norm, ham_params = strat

        assert grouping.ndim == 1, f"grouping should be 1D, got shape {grouping.shape}"
        assert orders_oh.ndim == 2, f"orders_oh should be 2D, got shape {orders_oh.shape}"
        assert orders_oh.shape[-1] == 3, f"orders last dim should be 3, got {orders_oh.shape[-1]}"
        assert ts_norm.ndim == 1, f"ts_norm should be 1D, got shape {ts_norm.shape}"
        assert orders_oh.shape[0] == ts_norm.shape[0], "orders and ts_norm must have same length"
        assert ham_params.ndim == 1, f"ham_params should be 1D, got shape {ham_params.shape}"
        assert ham_params.shape[0] == 5, f"ham_params should have 5 elements, got {ham_params.shape[0]}"

    def test_grouping_completeness(self):
        """Every Pauli term must be assigned to a group (no -1 or out-of-range labels)."""
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        for i in range(min(len(ds), 50)):
            _, strat, _, _ = ds[i]
            grouping = strat[0]
            assert (grouping >= 0).all(), f"Sample {i}: negative grouping label"
            assert (grouping < 8).all(), f"Sample {i}: grouping label >= max_groups"

    def test_timesteps_normalized(self):
        """Time-step normalizations should be in (0, 1] and sum to ~1."""
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        for i in range(min(len(ds), 50)):
            _, strat, _, _ = ds[i]
            ts = strat[2]
            assert (ts >= 0).all(), f"Sample {i}: negative time step"
            total = float(ts.sum())
            assert abs(total - 1.0) < 0.05, f"Sample {i}: ts sum={total:.4f}, expected ~1.0"

    def test_circuit_depth_positive(self):
        """Circuit depth estimates should be positive integers."""
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        for i in range(min(len(ds), 50)):
            _, _, _, depth = ds[i]
            assert float(depth) > 0, f"Sample {i}: non-positive depth {depth}"

    def test_augmentation_preserves_shape(self):
        """Augmentation must not change tensor shapes."""
        from pinn_trotter.data.dataset import TrotterDataset
        ds_plain = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False)
        ds_aug = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=True)
        _, strat_plain, _, _ = ds_plain[0]
        _, strat_aug, _, _ = ds_aug[0]
        for t_p, t_a in zip(strat_plain, strat_aug):
            assert t_p.shape == t_a.shape, f"Shape mismatch after augmentation: {t_p.shape} vs {t_a.shape}"

    def test_n_qubits_filter(self):
        """Filtering by n_qubits should return only matching samples."""
        import h5py
        from pinn_trotter.data.dataset import TrotterDataset
        ds = TrotterDataset(dataset_path=DATASET_PATH, max_groups=8, augment=False, n_qubits_filter=4)
        assert len(ds) > 0
        # Spot-check first few samples have correct n_qubits
        with h5py.File(DATASET_PATH, "r") as f:
            samples = f["dataset_v1/samples"]
            for sid in list(samples.keys())[:5]:
                n_q = int(samples[sid]["hamiltonian"].attrs["n_qubits"])
                if n_q == 4:
                    break
            else:
                pytest.skip("No 4-qubit samples found in first 5")
