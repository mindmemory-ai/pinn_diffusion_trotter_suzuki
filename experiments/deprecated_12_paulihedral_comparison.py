"""Paulihedral Trotter-order comparison: 1st-order vs 4th-order Suzuki.

Compares three methods on 50 n=4 TFIM Hamiltonians with the same experimental
standard as the main benchmark (t_total=2.0, n_steps=5, depth via
basis_gates=["h","cx","rz","x"], optimization_level=1).

Output: experiments/benchmark_results/paulihedral_order_comparison.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pinn_trotter.benchmarks.baseline_adapters import (
    PaulihedralBaseline,
    PaulihedralSuzuki4Baseline,
)
from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
from pinn_trotter.benchmarks.metrics import transpiled_depth, cx_count


def _make_tfim_hamiltonians(
    n_hams: int = 50,
    n_qubits: int = 4,
    seed: int = 99,
    j_min: float = 0.5,
    j_max: float = 2.0,
    h_min: float = 0.1,
    h_max: float = 0.5,
) -> list:
    from pinn_trotter.benchmarks.hamiltonians import make_tfim

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_hams):
        jv = float(rng.uniform(j_min, j_max))
        hv = float(rng.uniform(h_min, h_max))
        out.append(make_tfim(n_qubits, jv, hv, "periodic"))
    return out


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = float(statistics.fmean(values))
    s = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": m, "std": s}


def main() -> None:
    t_total = 2.0
    n_steps = 5
    n_hams = 50

    print(f"Generating {n_hams} n=4 TFIM Hamiltonians ...")
    hamiltonians = _make_tfim_hamiltonians(n_hams=n_hams, seed=99)

    methods: dict[str, Any] = {
        "paulihedral_1st": PaulihedralBaseline(n_steps=n_steps),
        "paulihedral_4th": PaulihedralSuzuki4Baseline(n_steps=n_steps),
        "qiskit_4th": QiskitTrotterBaseline(),
    }
    qiskit_n_steps = n_steps  # passed to .evaluate() below

    results: dict[str, Any] = {
        "config": {"t_total": t_total, "n_steps": n_steps, "n_hamiltonians": n_hams},
        "methods": {},
    }

    for name, baseline in methods.items():
        print(f"\n--- {name} ---")
        fids, depths, cxs = [], [], []
        for i, H in enumerate(hamiltonians):
            if name == "qiskit_4th":
                out = baseline.evaluate(H, t_total, order=4, n_steps=qiskit_n_steps)
                fids.append(out["fidelity"])
                depths.append(transpiled_depth(out["strategy"], H))
                cxs.append(cx_count(out["strategy"], H))
            else:
                out = baseline.evaluate(H, t_total)
                fids.append(out["fidelity"])
                depths.append(out["depth"])
                cxs.append(out["cx_count"])
            if (i + 1) % 10 == 0:
                fid_val = fids[-1]
                dep_val = depths[-1]
                print(f"  {i + 1}/{n_hams}  fid={fid_val:.4f}  depth={dep_val}")

        results["methods"][name] = {
            "fidelity": _summarize(fids),
            "depth": _summarize(depths),
            "cx_count": _summarize(cxs),
            "per_sample": [
                {"fid": f, "depth": d, "cx": c}
                for f, d, c in zip(fids, depths, cxs)
            ],
        }
        print(f"  => fid={_summarize(fids)}, depth={_summarize(depths)}, cx={_summarize(cxs)}")

    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paulihedral_order_comparison.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nResults written to {out_path}")

    # Summary comparison.
    p1 = results["methods"]["paulihedral_1st"]
    p4 = results["methods"]["paulihedral_4th"]
    q4 = results["methods"]["qiskit_4th"]
    print("\n=== Summary ===")
    print(f"{'Method':<22} {'Fidelity':>18} {'Depth':>12} {'CX':>10}")
    print("-" * 64)
    for label, r in [("paulihedral_1st", p1), ("paulihedral_4th", p4), ("qiskit_4th", q4)]:
        print(
            f"{label:<22} {r['fidelity']['mean']:>8.4f} ± {r['fidelity']['std']:.4f}"
            f"  {r['depth']['mean']:>6.1f}  {r['cx_count']['mean']:>6.1f}"
        )


if __name__ == "__main__":
    main()
