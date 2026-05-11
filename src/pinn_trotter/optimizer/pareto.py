"""Pareto frontier tracker with hypervolume computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ParetoTracker:
    """Track non-dominated (fidelity, depth) points across iterations.

    Uses the 2-objective dominance criterion:
        Point A dominates B if A.fidelity >= B.fidelity AND A.depth <= B.depth
        with at least one strict inequality.

    Hypervolume is computed against a fixed reference point (0.0, D_ref)
    where D_ref is set at construction time.

    Args:
        ref_depth: Reference depth for hypervolume computation.
                   Should be max expected circuit depth (e.g., Qiskit baseline × 2).
    """

    def __init__(self, ref_depth: float = 1000.0) -> None:
        self.ref_depth = ref_depth
        # Each entry: {'fidelity': float, 'depth': int, 'metadata': dict}
        self._points: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        fidelities: list[float] | np.ndarray,
        depths: list[int] | np.ndarray,
        metadata: list[dict] | None = None,
    ) -> int:
        """Insert new points and prune dominated ones.

        Args:
            fidelities: Fidelity values for each new strategy.
            depths:     Circuit depths for each new strategy.
            metadata:   Optional per-point metadata dicts.

        Returns:
            Number of new non-dominated points added.
        """
        if metadata is None:
            metadata = [{} for _ in fidelities]

        added = 0
        for f, d, m in zip(fidelities, depths, metadata):
            point = {"fidelity": float(f), "depth": int(d), "metadata": m}
            if not self._is_dominated(float(f), int(d)):
                self._points.append(point)
                added += 1

        self._prune()
        return added

    def get_front(self) -> list[dict[str, Any]]:
        """Return current Pareto-optimal points sorted by ascending depth."""
        return sorted(self._points, key=lambda p: p["depth"])

    def hypervolume(self, ref_fidelity: float = 0.0) -> float:
        """Compute 2D hypervolume dominated by the Pareto front.

        Reference point: (ref_fidelity, ref_depth).
        HV = area dominated by the front above ref_fidelity and below ref_depth.

        Args:
            ref_fidelity: Lower bound on fidelity axis (default 0.0).

        Returns:
            Hypervolume value ≥ 0.
        """
        if not self._points:
            return 0.0

        front = self.get_front()
        # WFG/sweep algorithm for 2D: sort by depth ascending, sweep fidelity
        ref_depth = self.ref_depth
        hv = 0.0
        prev_depth = 0
        max_fid = 0.0

        # Sort by depth ascending; at each depth, track max fidelity seen so far
        for p in front:
            d = min(p["depth"], ref_depth)
            f = max(p["fidelity"] - ref_fidelity, 0.0)
            if d > prev_depth and f > max_fid:
                hv += (d - prev_depth) * max_fid
                prev_depth = d
                max_fid = f

        # Final segment to ref_depth
        if prev_depth < ref_depth:
            hv += (ref_depth - prev_depth) * max_fid

        return float(hv)

    def hypervolume_pymoo(self, ref_fidelity: float = 0.0) -> float:
        """Compute hypervolume via pymoo (more accurate for >2 objectives).

        Falls back to self.hypervolume() if pymoo is not installed.
        """
        try:
            from pymoo.indicators.hv import HV
            front = self.get_front()
            if not front:
                return 0.0
            # pymoo minimizes, so negate fidelity (we maximize fidelity, minimize depth)
            F = np.array([[-p["fidelity"], p["depth"]] for p in front])
            ref_point = np.array([-ref_fidelity, self.ref_depth])
            ind = HV(ref_point=ref_point)
            return float(ind(F))
        except ImportError:
            return self.hypervolume(ref_fidelity)

    def best_fidelity(self) -> float:
        """Return the maximum fidelity across the Pareto front."""
        if not self._points:
            return 0.0
        return max(p["fidelity"] for p in self._points)

    def best_depth(self) -> int:
        """Return the minimum depth across the Pareto front."""
        if not self._points:
            return 0
        return min(p["depth"] for p in self._points)

    def __len__(self) -> int:
        return len(self._points)

    def save(self, path: str | Path) -> None:
        """Serialize Pareto front to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ref_depth": self.ref_depth,
            "n_points": len(self._points),
            "front": self.get_front(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load Pareto front from JSON (replaces current state)."""
        with open(path) as f:
            data = json.load(f)
        self.ref_depth = data["ref_depth"]
        self._points = data["front"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_dominated(self, fidelity: float, depth: int) -> bool:
        """Check if (fidelity, depth) is dominated by any existing point."""
        for p in self._points:
            if p["fidelity"] >= fidelity and p["depth"] <= depth:
                if p["fidelity"] > fidelity or p["depth"] < depth:
                    return True
        return False

    def _prune(self) -> None:
        """Remove dominated points from the front."""
        non_dom = []
        for i, p in enumerate(self._points):
            dominated = False
            for j, q in enumerate(self._points):
                if i == j:
                    continue
                if q["fidelity"] >= p["fidelity"] and q["depth"] <= p["depth"]:
                    if q["fidelity"] > p["fidelity"] or q["depth"] < p["depth"]:
                        dominated = True
                        break
            if not dominated:
                non_dom.append(p)
        self._points = non_dom
