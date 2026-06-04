"""Analyze fidelity-matched results across ALL baselines.

Reads the output JSON from 05d_fidelity_matched_all_baselines.py and generates:
1. Paper-ready comparison table (all methods at each threshold)
2. Depth/CX reduction ratios relative to each baseline
3. Reachability summary
4. Optional: N-sensitivity from per-candidate data

Usage:
    python experiments/analyze_all_baselines.py
    python experiments/analyze_all_baselines.py --per-candidate  # if per_candidate saved
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "benchmark_results"
INPUT_FILE = "fidelity_matched_all_baselines_per_candidate.json"
FALLBACK_FILE = "fidelity_matched_all_baselines.json"

METHOD_NAMES = {
    "ours": "Ours (best-of-N)",
    "qiskit_4th": "Qiskit 4th",
    "cirq": "Cirq",
    "tket": "TKET",
    "pennylane": "PennyLane",
    "paulihedral": "Paulihedral (1st)",
    "paulihedral_4th": "Paulihedral 4th",
}

METHOD_ORDER = ["ours", "paulihedral_4th", "paulihedral", "qiskit_4th", "cirq", "tket", "pennylane"]


def fmt_val(mean: float, std: float, decimals: int = 2) -> str:
    """Format mean±std with consistent precision."""
    if decimals == 0:
        return f"{mean:.0f}±{std:.0f}"
    elif decimals == 2:
        return f"{mean:.2f}±{std:.2f}"
    elif decimals == 4:
        return f"{mean:.4f}±{std:.3f}"
    return f"{mean}±{std}"


def print_header(title: str, width: int = 110):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def main():
    # Try per-candidate file first, fall back to summary-only
    per_candidate = "--per-candidate" in sys.argv
    input_path = RESULTS_DIR / (INPUT_FILE if per_candidate else FALLBACK_FILE)
    if per_candidate and not input_path.exists():
        print(f"Per-candidate file not found: {input_path}")
        print(f"Falling back to: {RESULTS_DIR / FALLBACK_FILE}")
        input_path = RESULTS_DIR / FALLBACK_FILE
    elif not input_path.exists():
        input_path = RESULTS_DIR / INPUT_FILE
        if not input_path.exists():
            input_path = RESULTS_DIR / FALLBACK_FILE

    if not input_path.exists():
        print(f"No results found in {RESULTS_DIR}")
        print("Run 05d_fidelity_matched_all_baselines.py first.")
        sys.exit(1)

    print(f"Reading: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    summary = data["summary"]["thresholds"]
    config = data.get("config", {})
    n_hams = config.get("n_test_hamiltonians", "?")
    n_candidates = config.get("n_candidates", "?")
    order = config.get("trotter_order", "?")
    methods_in_file = config.get("baseline_methods", [])

    # Collect all methods present
    all_methods = ["ours"] + methods_in_file
    thresholds = sorted([float(t) for t in summary.keys()])

    # =========================================================================
    # TABLE 1: Main comparison — all methods at each threshold
    # =========================================================================
    print_header(f"FIDELITY-MATCHED COMPARISON — ALL BASELINES  ({n_hams} Hams × {n_candidates} candidates, order={order})")

    for T in thresholds:
        T_str = str(T)
        entry = summary[T_str]

        print(f"\n{'─' * 110}")
        print(f"  Threshold T = {T:.2f}")
        print(f"{'─' * 110}")

        # Header
        print(f"{'Method':>22s} │ {'Reachable':>10s} │ {'Fidelity':>18s} │ {'Depth':>14s} │ {'CX':>14s} │ {'Depth↓':>10s}")
        print("─" * 110)

        for method in METHOD_ORDER:
            if method not in entry:
                continue
            m = entry[method]
            reach = f"{m['reachable']}/{n_hams}" if m.get('reachable', 0) > 0 else f"0/{n_hams}"
            fid_s = fmt_val(m["fidelity"]["mean"], m["fidelity"]["std"], 4)
            dep_s = fmt_val(m["depth"]["mean"], m["depth"]["std"], 0)
            cx_s = fmt_val(m["cx_count"]["mean"], m["cx_count"]["std"], 0)

            # Depth reduction relative to this baseline
            dr_key = f"depth_reduction_vs_{method}"
            if method != "ours" and dr_key in entry:
                dr = f"{entry[dr_key]:.1f}×"
            else:
                dr = "—"

            print(f"{METHOD_NAMES.get(method, method):>22s} │ {reach:>10s} │ {fid_s:>18s} │ {dep_s:>14s} │ {cx_s:>14s} │ {dr:>10s}")

    # =========================================================================
    # TABLE 2: Depth reduction matrix (ours vs each baseline at each threshold)
    # =========================================================================
    print_header("DEPTH REDUCTION RATIO MATRIX (ours depth vs baseline depth)")

    bl_methods = [m for m in METHOD_ORDER if m != "ours" and m in all_methods]
    header = f"{'Threshold':>12s} │ " + " │ ".join(f"{METHOD_NAMES.get(m, m):>22s}" for m in bl_methods)
    print(header)
    print("─" * len(header))

    for T in thresholds:
        T_str = str(T)
        entry = summary[T_str]
        parts = [f"T={T:.2f}".rjust(12)]
        for bm in bl_methods:
            dr_key = f"depth_reduction_vs_{bm}"
            if dr_key in entry and entry[dr_key] > 0:
                parts.append(f"{entry[dr_key]:.1f}×".rjust(22))
            else:
                parts.append("N/A".rjust(22))
        print(" │ ".join(parts))

    # =========================================================================
    # TABLE 3: CX reduction matrix
    # =========================================================================
    print_header("CX COUNT REDUCTION RATIO MATRIX (ours CX vs baseline CX)")

    header = f"{'Threshold':>12s} │ " + " │ ".join(f"{METHOD_NAMES.get(m, m):>22s}" for m in bl_methods)
    print(header)
    print("─" * len(header))

    for T in thresholds:
        T_str = str(T)
        entry = summary[T_str]
        parts = [f"T={T:.2f}".rjust(12)]
        for bm in bl_methods:
            cr_key = f"cx_reduction_vs_{bm}"
            if cr_key in entry and entry[cr_key] > 0:
                parts.append(f"{entry[cr_key]:.1f}×".rjust(22))
            else:
                parts.append("N/A".rjust(22))
        print(" │ ".join(parts))

    # =========================================================================
    # TABLE 4: Reachability summary
    # =========================================================================
    print_header("REACHABILITY SUMMARY")

    header = f"{'Method':>22s} │ " + " │ ".join(f"T={T:.2f}".rjust(12) for T in thresholds)
    print(header)
    print("─" * len(header))

    for method in METHOD_ORDER:
        if method not in summary[str(thresholds[0])]:
            continue
        parts = [METHOD_NAMES.get(method, method).rjust(22)]
        for T in thresholds:
            T_str = str(T)
            m = summary[T_str][method]
            pct = m["reachable"] / n_hams * 100 if isinstance(n_hams, int) and n_hams > 0 else 0
            parts.append(f"{m['reachable']}/{n_hams} ({pct:.0f}%)".rjust(12))
        print(" │ ".join(parts))

    # =========================================================================
    # TABLE 5: Paper-ready concise table (for LaTeX)
    # =========================================================================
    print_header("PAPER-READY TABLE (T=0.95, primary result)")

    T = 0.95
    T_str = str(T)
    if T_str in summary:
        entry = summary[T_str]
        print(f"{'Method':>22s} │ {'Reachable':>10s} │ {'Fidelity':>18s} │ {'Depth':>14s} │ {'CX Count':>12s} │ {'Depth↓ vs ours':>14s}")
        print("─" * 100)

        for method in METHOD_ORDER:
            if method not in entry:
                continue
            m = entry[method]
            reach = f"{m['reachable']}/{n_hams}"
            fid_s = f"{m['fidelity']['mean']:.4f}"
            dep_s = f"{m['depth']['mean']:.0f}"
            cx_s = f"{m['cx_count']['mean']:.0f}"

            if method == "ours":
                dr = "—"
            else:
                dr_key = f"depth_reduction_vs_{method}"
                dr = f"{entry[dr_key]:.1f}×" if dr_key in entry else "N/A"

            print(f"{METHOD_NAMES.get(method, method):>22s} │ {reach:>10s} │ {fid_s:>18s} │ {dep_s:>14s} │ {cx_s:>12s} │ {dr:>14s}")

    # =========================================================================
    # N-Sensitivity (if per-candidate data available)
    # =========================================================================
    ours_results = data.get("ours_results", [])
    if ours_results and "candidates" in ours_results[0]:
        print_header("N-SENSITIVITY ANALYSIS (from per-candidate data)")

        N_VALUES = [1, 2, 4, 8, 16, 32, 64, 100]
        N_BOOTSTRAP = 50
        rng = np.random.default_rng(42)

        ham_fidelities = []
        for r in ours_results:
            fids = [c["fidelity"] for c in r["candidates"]]
            ham_fidelities.append(np.array(fids))

        n_hams_actual = len(ham_fidelities)

        print(f"{'N':>6s} │ {'Reach@0.90':>16s} │ {'Reach@0.95':>16s} │ {'Reach@0.99':>16s} │ {'Best Fid':>16s}")
        print("─" * 85)

        for N in N_VALUES:
            if N > len(ham_fidelities[0]):
                continue
            reach_counts = {T: [] for T in [0.90, 0.95, 0.99]}
            best_fids_all = []

            for _ in range(N_BOOTSTRAP):
                reach_trial = {T: 0 for T in [0.90, 0.95, 0.99]}
                best_fids_trial = []

                for fids in ham_fidelities:
                    if N >= len(fids):
                        sampled = fids
                    else:
                        idx = rng.choice(len(fids), size=N, replace=False)
                        sampled = fids[idx]
                    best_fid = float(np.max(sampled))
                    best_fids_trial.append(best_fid)
                    for T in [0.90, 0.95, 0.99]:
                        if best_fid >= T:
                            reach_trial[T] += 1

                for T in [0.90, 0.95, 0.99]:
                    reach_counts[T].append(reach_trial[T] / n_hams_actual)
                best_fids_all.append(np.mean(best_fids_trial))

            parts = [f"{N:6d}"]
            for T in [0.90, 0.95, 0.99]:
                mean_r = np.mean(reach_counts[T])
                std_r = np.std(reach_counts[T])
                parts.append(f"{mean_r:.1%}±{std_r:.1%}".rjust(16))
            mean_fid = np.mean(best_fids_all)
            std_fid = np.std(best_fids_all)
            parts.append(f"{mean_fid:.4f}±{std_fid:.3f}".rjust(16))
            print(" │ ".join(parts))

    # =========================================================================
    # Key findings summary
    # =========================================================================
    print_header("KEY FINDINGS")

    # Find best baseline at T=0.95
    if "0.95" in summary:
        entry = summary["0.95"]
        ours_depth = entry.get("ours", {}).get("depth", {}).get("mean", 0)

        best_bl = None
        best_bl_depth = float("inf")
        for bm in bl_methods:
            d = entry.get(bm, {}).get("depth", {}).get("mean", float("inf"))
            if d < best_bl_depth:
                best_bl_depth = d
                best_bl = bm

        if best_bl and ours_depth > 0:
            dr = best_bl_depth / ours_depth
            print(f"  Best baseline: {METHOD_NAMES.get(best_bl, best_bl)} (depth={best_bl_depth:.0f})")
            print(f"  Ours depth: {ours_depth:.0f}")
            print(f"  Depth reduction vs best baseline: {dr:.1f}×")

        # Compare vs qiskit_4th (no-grouping reference)
        qiskit_depth = entry.get("qiskit_4th", {}).get("depth", {}).get("mean", 0)
        if qiskit_depth > 0 and ours_depth > 0:
            print(f"  Depth reduction vs Qiskit 4th (no grouping): {qiskit_depth / ours_depth:.1f}×")

        # Grouping-only benefit (qiskit_4th → paulihedral_4th)
        ph4_depth = entry.get("paulihedral_4th", {}).get("depth", {}).get("mean", 0)
        if ph4_depth > 0 and qiskit_depth > 0:
            print(f"  Paulihedral grouping benefit (qiskit→ph4): {qiskit_depth / ph4_depth:.1f}×")
            if ours_depth > 0:
                print(f"  Learning benefit (ph4→ours): {ph4_depth / ours_depth:.1f}×")

    print(f"\n{'═' * 110}")


if __name__ == "__main__":
    main()
