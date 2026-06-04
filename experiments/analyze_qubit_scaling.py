"""Analyze performance scaling by qubit count (P6).

Post-processes fidelity-matched per-candidate data to reveal how
reachability, depth, and fidelity scale with system size.

Usage:
    python experiments/analyze_qubit_scaling.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    per_candidate_path = Path(__file__).parent / "benchmark_results" / \
        "fidelity_matched_all_baselines_per_candidate.json"

    with open(per_candidate_path) as f:
        d = json.load(f)

    # Group ours results by n_qubits
    by_q = defaultdict(lambda: defaultdict(list))
    n_hams_per_q = defaultdict(int)
    for ham in d["ours_results"]:
        nq = ham["n_qubits"]
        n_hams_per_q[nq] += 1
        for t_str in ["0.9", "0.95", "0.99"]:
            td = ham["thresholds"][t_str]
            if td is not None:
                by_q[nq][float(t_str)].append(td)

    print("=" * 80)
    print("P6: QUBIT SCALING ANALYSIS")
    print(f"Data: P1 fidelity-matched, {len(d['ours_results'])} Hams x "
          f"{d['config']['n_candidates']} candidates")
    print("=" * 80)

    # Summary table
    thresholds = [0.9, 0.95, 0.99]
    print(f"\n{'Qubits':>8}  {'Hams':>5}  ", end="")
    for t in thresholds:
        print(f"{'T='+str(t):>20}  ", end="")
    print(f"{'All-cand':>18}")
    print(f"{'':>8}  {'':>5}  ", end="")
    for t in thresholds:
        print(f"{'Reach':>6} {'Fid':>6} {'Depth':>6}  ", end="")
    print(f"{'Depth':>8} {'Median':>8}")
    print("-" * 80)

    for nq in sorted(by_q.keys()):
        total = n_hams_per_q[nq]
        print(f"{nq:>8}  {total:>5}  ", end="")
        for t in thresholds:
            entries = by_q[nq].get(t, [])
            if entries:
                fids = [e["fidelity"] for e in entries]
                depths = [e["depth"] for e in entries]
                print(f"{len(entries):>3}/{total:<2} "
                      f"{np.mean(fids):.3f} {np.mean(depths):>5.0f}  ", end="")
            else:
                print(f"{0:>3}/{total:<2} {'N/A':>6} {'N/A':>6}  ", end="")

        # All-candidate stats
        all_depths = []
        for ham in d["ours_results"]:
            if ham["n_qubits"] != nq:
                continue
            for cand in ham["candidates"]:
                if cand["depth"] is not None:
                    all_depths.append(cand["depth"])
        if all_depths:
            print(f"{np.mean(all_depths):>8.0f} {np.median(all_depths):>8.0f}")
        else:
            print()

    # Baseline comparison (if available)
    print(f"\n--- Baseline Depth by Qubits ---")
    for nq in sorted(n_hams_per_q.keys()):
        ham_indices = [h["hamiltonian_idx"] for h in d["ours_results"]
                       if h["n_qubits"] == nq]
        print(f"  nq={nq}: {len(ham_indices)} Hamiltonians (indices: "
              f"{min(ham_indices)}..{max(ham_indices)})")

    print(f"\nInterpretation:")
    if len(by_q) >= 2:
        nqs = sorted(by_q.keys())
        t = 0.95
        r_small = len(by_q[nqs[0]].get(t, [])) / n_hams_per_q[nqs[0]]
        r_large = len(by_q[nqs[-1]].get(t, [])) / n_hams_per_q[nqs[-1]]
        drop = (r_small - r_large) / r_small * 100 if r_small > 0 else 0
        print(f"  Reachability drop from {nqs[0]}→{nqs[-1]} qubits @ T={t}: "
              f"{100*r_small:.0f}% → {100*r_large:.0f}% ({drop:.0f}% relative drop)")

    out = {
        "config": {"source": str(per_candidate_path)},
        "per_qubit": {},
    }
    for nq in sorted(by_q.keys()):
        out["per_qubit"][str(nq)] = {
            "n_hamiltonians": n_hams_per_q[nq],
            "thresholds": {},
        }
        for t in thresholds:
            entries = by_q[nq].get(t, [])
            if entries:
                fids = [e["fidelity"] for e in entries]
                depths = [e["depth"] for e in entries]
                out["per_qubit"][str(nq)]["thresholds"][str(t)] = {
                    "reachable": len(entries),
                    "total": n_hams_per_q[nq],
                    "fidelity_mean": float(np.mean(fids)),
                    "fidelity_std": float(np.std(fids)),
                    "depth_mean": float(np.mean(depths)),
                    "depth_std": float(np.std(depths)),
                }

    out_path = Path(__file__).parent / "benchmark_results" / "qubit_scaling.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
