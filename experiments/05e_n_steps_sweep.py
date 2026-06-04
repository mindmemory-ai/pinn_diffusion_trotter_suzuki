"""n_steps sweep for ALL baseline methods.

Sweeps Trotter step count across baselines (qiskit_4th, cirq, tket, pennylane,
paulihedral, paulihedral_4th) and reports fidelity, depth, CX count at each
n_steps. Provides the baseline calibration curves needed for fidelity-matched
comparison.

Usage:
    python experiments/05e_n_steps_sweep.py \
        ++benchmark.n_test_hamiltonians=30 \
        ++benchmark.n_steps_list="[1,2,3,4,5,6,8,10,12,16]" \
        ++benchmark.methods=qiskit_4th,cirq,tket,pennylane,paulihedral,paulihedral_4th \
        ++benchmark.trotter_order=4 ++experiment.t_total=2.0
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_hamiltonians(n: int, seed: int, h_min=0.1, h_max=0.5,
                         tfim_ratio=0.4, random_ratio=0.2):
    """Generate n mixed-type Hamiltonians."""
    from pinn_trotter.benchmarks.hamiltonians import make_heisenberg, make_tfim
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

    rng = np.random.default_rng(seed)
    nq_choices = [4, 6, 8]
    nq_probs = [0.6, 0.3, 0.1]
    pauli_chars = ["I", "X", "Y", "Z"]
    hamiltonians = []
    for _ in range(n):
        nq = int(rng.choice(nq_choices, p=nq_probs))
        r = rng.random()
        if r < random_ratio:
            n_terms = int(rng.integers(4, 17))
            seen: set[str] = set()
            paulis, coeffs = [], []
            for _ in range(n_terms):
                while True:
                    s = "".join(rng.choice(pauli_chars, size=nq))
                    if s != "I" * nq and s not in seen:
                        seen.add(s)
                        break
                paulis.append(s)
                coeffs.append(float(np.exp(rng.uniform(np.log(0.1), np.log(5.0)))))
            hamiltonians.append(HamiltonianGraph(paulis, coeffs, nq))
        elif r < random_ratio + (1.0 - random_ratio) * tfim_ratio:
            j_val = float(rng.uniform(0.5, 2.0))
            h_val = float(rng.uniform(h_min, h_max))
            hamiltonians.append(make_tfim(nq, j_val, h_val))
        else:
            J = float(rng.uniform(0.5, 2.0))
            hamiltonians.append(make_heisenberg(nq, J, J, J))
    return hamiltonians


def _instantiate_baseline(name: str, n_steps: int, order: int, scheduler: str = "depth"):
    """Create a single baseline adapter instance."""
    from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
    from pinn_trotter.benchmarks.baseline_adapters import BASELINE_REGISTRY

    if name == "qiskit_4th":
        return name, QiskitTrotterBaseline()
    cls = BASELINE_REGISTRY[name]
    if name in ("paulihedral", "paulihedral_4th"):
        return name, cls(n_steps=n_steps, scheduler=scheduler)
    else:
        return name, cls(n_steps=n_steps, order=order)


def _evaluate_baseline(name: str, adapter, hamiltonian, t_total: float,
                       n_steps: int, order: int, fidelity_only: bool):
    """Evaluate one baseline at a specific n_steps."""
    from pinn_trotter.benchmarks.metrics import cx_count, transpiled_depth

    if name == "qiskit_4th":
        res = adapter.evaluate(
            hamiltonian=hamiltonian, t_total=t_total,
            order=order, n_steps=n_steps,
        )
    elif name in ("paulihedral", "paulihedral_4th"):
        adapter.n_steps = n_steps
        res = adapter.evaluate(hamiltonian, t_total)
    else:
        adapter.n_steps = n_steps
        res = adapter.evaluate(hamiltonian, t_total)

    fid = res["fidelity"]
    if fidelity_only:
        strategy = res.get("strategy")
        depth = strategy.circuit_depth_estimate() if strategy is not None else 0
        cx = 0
    else:
        if "depth" in res and "cx_count" in res:
            depth = int(res["depth"])
            cx = int(res["cx_count"])
        else:
            strategy = res["strategy"]
            depth = int(transpiled_depth(strategy, hamiltonian))
            cx = int(cx_count(strategy, hamiltonian))
    return {"fidelity": fid, "depth": depth, "cx_count": cx, "n_steps": n_steps}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="experiment/tfim_4q_poc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    bench_cfg = cfg.get("benchmark", {})
    exp_cfg = cfg.get("experiment", {})
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ---- Config ----
    t_total = float(exp_cfg.get("t_total", 2.0))
    n_test_h = int(bench_cfg.get("n_test_hamiltonians", 30))
    n_steps_list = list(bench_cfg.get("n_steps_list", [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]))
    trotter_order = int(bench_cfg.get("trotter_order", 4))
    seed0 = int(bench_cfg.get("seed", 42))
    fidelity_only = bool(bench_cfg.get("fidelity_only", False))
    h_min = float(bench_cfg.get("h_min", 0.1))
    h_max = float(bench_cfg.get("h_max", 0.5))
    tfim_ratio = float(bench_cfg.get("tfim_ratio", 0.4))
    random_ratio = float(bench_cfg.get("random_ratio", 0.2))
    paulihedral_scheduler = str(bench_cfg.get("paulihedral_scheduler", "depth"))

    # Parse baseline methods
    raw_methods = bench_cfg.get("methods", None)
    if raw_methods is None:
        baseline_methods = ["qiskit_4th", "cirq", "tket", "pennylane",
                           "paulihedral", "paulihedral_4th"]
    elif isinstance(raw_methods, str):
        baseline_methods = [m.strip() for m in raw_methods.split(",") if m.strip()]
    else:
        baseline_methods = [str(m).strip() for m in raw_methods if str(m).strip()]
    log.info("Baseline methods: %s", baseline_methods)
    log.info("n_steps sweep: %s", n_steps_list)

    # ---- Sample Hamiltonians ----
    hamiltonians = _sample_hamiltonians(n_test_h, seed0, h_min=h_min, h_max=h_max,
                                        tfim_ratio=tfim_ratio, random_ratio=random_ratio)
    log.info("Sampled %d Hamiltonians", len(hamiltonians))

    # ---- Evaluate each baseline at each n_steps ----
    # Structure: results[method][n_steps] = list of {fidelity, depth, cx_count}
    all_results: dict[str, dict[int, list[dict]]] = {
        bname: {ns: [] for ns in n_steps_list}
        for bname in baseline_methods
    }

    t0 = time.time()
    for bname in baseline_methods:
        log.info("=== Baseline: %s ===", bname)
        # Create fresh adapter for each baseline (n_steps will be overridden per call)
        _, adapter = _instantiate_baseline(bname, n_steps_list[0], trotter_order,
                                           paulihedral_scheduler)

        for idx, H in enumerate(hamiltonians):
            for ns in n_steps_list:
                res = _evaluate_baseline(bname, adapter, H, t_total, ns,
                                        trotter_order, fidelity_only)
                all_results[bname][ns].append(res)

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                log.info("  %s %d/%d (%.1fs)", bname, idx + 1, n_test_h, elapsed)

    total_elapsed = time.time() - t0
    log.info("All sweeps done in %.1fs", total_elapsed)

    # ---- Aggregate per (method, n_steps) ----
    def _summarize(values):
        if not values:
            return {"mean": 0, "std": 0}
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    summary: dict[str, dict[int, dict]] = {}
    for bname in baseline_methods:
        summary[bname] = {}
        for ns in n_steps_list:
            entries = all_results[bname][ns]
            fids = [e["fidelity"] for e in entries]
            depths = [e["depth"] for e in entries]
            cxs = [e["cx_count"] for e in entries]
            summary[bname][ns] = {
                "fidelity": _summarize(fids),
                "depth": _summarize(depths),
                "cx_count": _summarize(cxs),
                "n_samples": len(entries),
            }

    # ---- Report ----
    print("\n" + "=" * 100)
    print("n_steps SWEEP — ALL BASELINES")
    print(f"{n_test_h} Hamiltonians, order={trotter_order}, t_total={t_total}")
    print("=" * 100)

    for bname in baseline_methods:
        print(f"\n{'─' * 100}")
        print(f"  {bname}")
        print(f"{'─' * 100}")
        header = f"{'n_steps':>8s} | {'Fidelity':>20s} | {'Depth':>16s} | {'CX Count':>16s}"
        print(header)
        print("-" * 100)
        for ns in n_steps_list:
            s = summary[bname][ns]
            fid_str = f"{s['fidelity']['mean']:.6f}±{s['fidelity']['std']:.4f}"
            dep_str = f"{s['depth']['mean']:.0f}±{s['depth']['std']:.0f}"
            cx_str = f"{s['cx_count']['mean']:.0f}±{s['cx_count']['std']:.0f}"
            print(f"{ns:8d} | {fid_str:>20s} | {dep_str:>16s} | {cx_str:>16s}")

    # ---- Fidelity comparison at key n_steps ----
    print(f"\n{'═' * 100}")
    print("CROSS-BASELINE FIDELITY COMPARISON")
    print(f"{'═' * 100}")
    key_steps = [ns for ns in [1, 3, 5, 8, 10] if ns in n_steps_list]
    for ns in key_steps:
        print(f"\n  n_steps = {ns}:")
        for bname in baseline_methods:
            s = summary[bname][ns]
            print(f"    {bname:>20s}: fid={s['fidelity']['mean']:.6f}  depth={s['depth']['mean']:.0f}  cx={s['cx_count']['mean']:.0f}")

    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # ---- Save ----
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "n_steps_sweep_all_baselines.json"

    # Convert int keys to str for JSON
    json_summary = {
        bname: {str(ns): v for ns, v in s.items()}
        for bname, s in summary.items()
    }

    report = {
        "config": {
            "n_test_hamiltonians": n_test_h,
            "n_steps_list": n_steps_list,
            "trotter_order": trotter_order,
            "t_total": t_total,
            "baseline_methods": baseline_methods,
            "tfim_ratio": tfim_ratio,
            "random_ratio": random_ratio,
        },
        "summary": json_summary,
        "elapsed_s": total_elapsed,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
