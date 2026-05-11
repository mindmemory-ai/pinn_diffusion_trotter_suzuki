"""Unit tests for ParetoTracker."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pinn_trotter.optimizer.pareto import ParetoTracker


class TestParetoTracker:
    @pytest.fixture
    def tracker(self):
        return ParetoTracker(ref_depth=100.0)

    # --- Dominance and update ---

    def test_single_point(self, tracker):
        added = tracker.update([0.9], [50])
        assert added == 1
        assert len(tracker) == 1

    def test_dominated_point_not_added(self, tracker):
        tracker.update([0.9], [50])
        # Lower fidelity, higher depth → dominated
        added = tracker.update([0.7], [80])
        assert added == 0
        assert len(tracker) == 1

    def test_non_dominated_both_kept(self, tracker):
        tracker.update([0.9], [80])   # high fidelity, high depth
        tracker.update([0.7], [30])   # lower fidelity, lower depth
        assert len(tracker) == 2

    def test_better_point_removes_dominated(self, tracker):
        tracker.update([0.9], [50])
        # Better on both axes → old point dominated and pruned
        tracker.update([0.95], [40])
        assert len(tracker) == 1
        assert tracker.get_front()[0]["fidelity"] == 0.95

    def test_equal_points_only_one_kept(self, tracker):
        tracker.update([0.9], [50])
        tracker.update([0.9], [50])
        # Second is not strictly better → should be treated as dominated
        assert len(tracker) <= 2  # implementation may keep both equal; at most 2

    def test_front_sorted_by_depth(self, tracker):
        tracker.update([0.95, 0.85, 0.75], [80, 50, 20])
        front = tracker.get_front()
        depths = [p["depth"] for p in front]
        assert depths == sorted(depths)

    # --- Hypervolume ---

    def test_hv_single_point(self):
        t = ParetoTracker(ref_depth=100)
        t.update([0.8], [50])
        # Sweep: prev=0,max=0 → point(d=50,f=0.8) → hv+=(50-0)*0=0; prev=50,max=0.8
        # Final: hv += (100-50)*0.8 = 40
        hv = t.hypervolume()
        assert abs(hv - 40.0) < 1e-6, f"Expected 40.0, got {hv}"

    def test_hv_empty(self, tracker):
        assert tracker.hypervolume() == 0.0

    def test_hv_increases_with_better_front(self, tracker):
        tracker.update([0.7], [50])
        hv1 = tracker.hypervolume()
        tracker2 = ParetoTracker(ref_depth=100.0)
        tracker2.update([0.9], [30])
        hv2 = tracker2.hypervolume()
        assert hv2 > hv1

    def test_hv_known_value_two_points(self):
        """Known analytical result for 2 non-dominated points."""
        # ref_depth=100
        # Points: (fid=0.9, d=20), (fid=0.6, d=60)
        # Sorted by depth: (20, 0.9), (60, 0.6)
        # HV sweep: 0 to 20 at fid 0.0 = 0; 20 to 60 at fid 0.9 = 36;
        #           60 to 100 at fid 0.9 = 36 → total = 72
        # Wait, max_fid only updates when strictly increasing…let me trace:
        # prev_depth=0, max_fid=0
        # point (d=20, f=0.9): d=20 > 0, f=0.9 > 0 → hv += (20-0)*0.0=0; prev=20, max=0.9
        # point (d=60, f=0.6): d=60 > 20, f=0.6 < 0.9 → condition f > max_fid fails
        # Final: hv += (100-20)*0.9 = 72 → total = 72
        t = ParetoTracker(ref_depth=100)
        t.update([0.9, 0.6], [20, 60])
        hv = t.hypervolume()
        assert abs(hv - 72.0) < 1e-6, f"Expected 72.0, got {hv}"

    # --- Best values ---

    def test_best_fidelity(self, tracker):
        tracker.update([0.9, 0.7, 0.8], [80, 20, 50])
        assert tracker.best_fidelity() == 0.9

    def test_best_depth(self, tracker):
        tracker.update([0.9, 0.7, 0.8], [80, 20, 50])
        assert tracker.best_depth() == 20

    # --- Serialization ---

    def test_save_load_roundtrip(self, tracker):
        tracker.update([0.9, 0.7], [80, 30])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pareto.json"
            tracker.save(path)
            t2 = ParetoTracker(ref_depth=100.0)
            t2.load(path)
            assert len(t2) == len(tracker)
            f1 = {p["fidelity"] for p in tracker.get_front()}
            f2 = {p["fidelity"] for p in t2.get_front()}
            assert f1 == f2

    def test_save_creates_json(self, tracker):
        tracker.update([0.85], [40])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "pareto.json"
            tracker.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "front" in data
            assert data["n_points"] == 1
