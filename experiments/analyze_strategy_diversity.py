"""Analyze strategy diversity across best-of-N candidates (P15).

Loads the Phase 4 model, samples 100 candidates per Hamiltonian,
and computes grouping diversity, order entropy, and time allocation variance.

Usage:
    python experiments/analyze_strategy_diversity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pinn_trotter.utils.model_builder import build_models_from_checkpoint
from pinn_trotter.diffusion.mixed_model import guided_sample
from pinn_trotter.pinn.evaluator import _decode_strategy


def jaccard_distance(partition_a: list[set], partition_b: list[set]) -> float:
    n = max(len(partition_a), len(partition_b))
    dists = []
    for i in range(n):
        sa = partition_a[i] if i < len(partition_a) else set()
        sb = partition_b[i] if i < len(partition_b) else set()
        if not sa and not sb:
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        dists.append(1.0 - inter / union if union > 0 else 0.0)
    return float(np.mean(dists)) if dists else 0.0


def grouping_to_partitions(grouping: list[int]) -> list[set[int]]:
    groups: dict[int, set[int]] = {}
    for term_idx, g in enumerate(grouping):
        groups.setdefault(int(g), set()).add(term_idx)
    return [groups[k] for k in sorted(groups.keys())]


def _sample_hamiltonians(n_h, seed, **kw):
    from pinn_trotter.benchmarks.hamiltonians import make_heisenberg, make_tfim
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

    rng = np.random.default_rng(seed)
    hams = []
    nq_list = [4, 6, 8]
    for _ in range(n_h):
        nq = int(rng.choice(nq_list))
        r = rng.random()
        if r < 0.4:  # TFIM
            hams.append(make_tfim(nq, float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.1, 0.5))))
        elif r < 0.6:  # Heisenberg
            hams.append(make_heisenberg(nq, float(rng.uniform(0.5, 2.0)),
                                        float(rng.uniform(0.5, 2.0)),
                                        float(rng.uniform(0.5, 2.0))))
        else:  # Random
            paulis, coeffs = [], []
            seen = set()
            for _ in range(int(rng.integers(3, 9))):
                while True:
                    s = "".join(rng.choice(["I", "X", "Y", "Z"], size=nq))
                    if s != "I" * nq and s not in seen:
                        seen.add(s)
                        break
                paulis.append(s)
                coeffs.append(float(np.exp(rng.uniform(np.log(0.1), np.log(5.0)))))
            hams.append(HamiltonianGraph(paulis, coeffs, nq))
    return hams


def _encode(gnn, H, device, max_n_q, disable_gnn=False, pauli_enc=False):
    from pinn_trotter.hamiltonian.pauli_utils import encode_pauli_types, locality

    if disable_gnn:
        return torch.zeros((1, gnn.output_dim), device=device)

    n = H.n_qubits
    if pauli_enc:
        feat_dim = 4 + 3 * max(n, max_n_q)
    else:
        feat_dim = max(n, max_n_q) + 2
    x = torch.zeros((H.n_terms, feat_dim), device=device)
    for i, (s, c) in enumerate(zip(H.pauli_strings, H.coefficients)):
        x[i, 0] = float(c)
        if pauli_enc:
            type_vec, counts = encode_pauli_types(s, max(n, max_n_q))
            x[i, 1] = float(counts[0])
            x[i, 2] = float(counts[1])
            x[i, 3] = float(counts[2])
            x[i, 4:] = torch.tensor(type_vec, device=device)
        else:
            x[i, 1] = float(locality(s))
            for q in range(n):
                x[i, 2 + q] = 1.0 if s[q] != "I" else 0.0
    edge_dim = int(getattr(gnn, "edge_feat_dim", 2))
    ei = torch.zeros((2, 0), dtype=torch.long, device=device)
    ea = torch.zeros((0, edge_dim), device=device)
    return gnn(x, ei, ea)


def main():
    CKPT = Path(__file__).parent / "closed_loop_checkpoints" / "diffusion_best_20260603_005153_HV9995.5852.pt"
    N_HAMS = 15
    N_CANDIDATES = 100
    T_TOTAL = 2.0
    GUIDANCE_SCALE = 3.0
    SEED = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    models = build_models_from_checkpoint(CKPT, max_groups=8, device=device,
        gnn_hidden_dim=512, gnn_output_dim=768, gnn_n_layers=6,
        diffusion_fused_dim=512, diffusion_time_embed_dim=256,
        diffusion_grouping_layers=8, diffusion_order_layers=4, diffusion_ts_mlp_layers=4)
    gnn = models["gnn"]
    diffusion = models["diffusion"]
    tm = models["tm"]
    order_tm = models["order_tm"]
    ddpm = models["ddpm"]
    max_groups = models["max_groups"]
    max_n_q = models["max_n_qubits"]
    pauli_enc = models.get("pauli_encoding", True)

    hamiltonians = _sample_hamiltonians(N_HAMS, SEED)
    print(f"Sampled {len(hamiltonians)} Hamiltonians")

    # Per-Ham metrics
    jaccard_all = []
    n_unique_all = []
    order_entropy_all = []
    time_cv_all = []

    t0 = time.time()
    for idx, H in enumerate(hamiltonians):
        with torch.no_grad():
            cond = _encode(gnn, H, device, max_n_q, pauli_enc=pauli_enc)
            g_all, ts_all, o_all = guided_sample(
                model=diffusion, condition=cond.repeat(N_CANDIDATES, 1),
                n_terms=H.n_terms, max_groups=max_groups,
                transition_matrix=tm, order_transition_matrix=order_tm,
                ddpm=ddpm, guidance_scale=GUIDANCE_SCALE, device=device,
            )

        # Extract groupings
        group_matrices = [g_all[k].cpu().tolist() for k in range(N_CANDIDATES)]
        order_matrices = [o_all[k].cpu().tolist() for k in range(N_CANDIDATES)]
        time_matrices = [ts_all[k].cpu().tolist() for k in range(N_CANDIDATES)]

        # 1. Pairwise Jaccard
        partitions = [grouping_to_partitions(gm) for gm in group_matrices]
        pair_dists = []
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                pair_dists.append(jaccard_distance(partitions[i], partitions[j]))
        jaccard_all.append(float(np.mean(pair_dists)) if pair_dists else 0.0)

        # 2. Unique patterns
        gm_tuples = [tuple(int(x) for x in gm) for gm in group_matrices]
        n_unique_all.append(len(set(gm_tuples)))

        # 3. Order entropy and time CV
        n_used_groups = max(max(gm) for gm in group_matrices) + 1
        orders_per_g = {k: [] for k in range(n_used_groups)}
        times_per_g = {k: [] for k in range(n_used_groups)}

        for k in range(N_CANDIDATES):
            gm = group_matrices[k]
            om = order_matrices[k]
            tm_data = time_matrices[k]
            for ti in range(H.n_terms):
                g = int(gm[ti])
                if ti < len(om):
                    orders_per_g.setdefault(g, []).append(int(om[ti]))
            for gi in range(min(len(tm_data), n_used_groups)):
                times_per_g.setdefault(gi, []).append(float(tm_data[gi]))

        # Order entropy
        entropies = []
        for g, orders in orders_per_g.items():
            if len(orders) < 2:
                continue
            vals, counts = np.unique(orders, return_counts=True)
            probs = counts / counts.sum()
            entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
        order_entropy_all.append(float(np.mean(entropies)) if entropies else 0.0)

        # Time CV
        cvs = []
        for g, times in times_per_g.items():
            if len(times) < 2:
                continue
            m = np.mean(times)
            if m == 0:
                continue
            cvs.append(float(np.std(times) / m))
        time_cv_all.append(float(np.mean(cvs)) if cvs else 0.0)

        elapsed = time.time() - t0
        print(f"  [{idx+1}/{N_HAMS}] nq={H.n_qubits} M={H.n_terms} "
              f"Jaccard={jaccard_all[-1]:.3f} unique={n_unique_all[-1]}/{N_CANDIDATES} "
              f"order_H={order_entropy_all[-1]:.3f} time_CV={time_cv_all[-1]:.3f} "
              f"({elapsed:.0f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("STRATEGY DIVERSITY ANALYSIS (P15)")
    print(f"  Hamiltonians: {N_HAMS}, Candidates per Ham: {N_CANDIDATES}")
    print(f"  Total pairwise comparisons: {N_HAMS * N_CANDIDATES * (N_CANDIDATES-1) // 2}")
    print()

    jd_mean, jd_std = np.mean(jaccard_all), np.std(jaccard_all)
    nu_mean, nu_std = np.mean(n_unique_all), np.std(n_unique_all)
    oe_mean, oe_std = np.mean(order_entropy_all), np.std(order_entropy_all)
    tc_mean, tc_std = np.mean(time_cv_all), np.std(time_cv_all)

    print(f"  Pairwise Jaccard Distance (0=identical, 1=completely different):")
    print(f"    Mean={jd_mean:.4f}  Std={jd_std:.4f}  Min={np.min(jaccard_all):.4f}  Max={np.max(jaccard_all):.4f}")
    print()
    print(f"  Unique Grouping Patterns (out of {N_CANDIDATES} candidates):")
    print(f"    Mean={nu_mean:.1f}  Std={nu_std:.1f}  Min={min(n_unique_all)}  Max={max(n_unique_all)}")
    print(f"    % unique: {nu_mean/N_CANDIDATES*100:.1f}%")
    print()
    print(f"  Order Entropy (higher = more diverse order choices):")
    print(f"    Mean={oe_mean:.4f}  Std={oe_std:.4f}")
    print()
    print(f"  Time CV (higher = more diverse time allocation):")
    print(f"    Mean={tc_mean:.4f}  Std={tc_std:.4f}")
    print()

    # Interpretation
    print("INTERPRETATION:")
    if jd_mean > 0.3:
        print(f"  ✓ Jaccard={jd_mean:.3f}: Grouping partitions are substantially diverse")
    elif jd_mean > 0.1:
        print(f"  ~ Jaccard={jd_mean:.3f}: Moderate grouping diversity")
    else:
        print(f"  ✗ Jaccard={jd_mean:.3f}: Groupings are nearly identical — best-of-N adds little")

    if nu_mean > 50:
        print(f"  ✓ {nu_mean:.0f} unique patterns: Best-of-N explores many distinct strategies")
    elif nu_mean > 10:
        print(f"  ~ {nu_mean:.0f} unique patterns: Moderate strategy exploration")
    else:
        print(f"  ✗ {nu_mean:.0f} unique patterns: Strategies collapse to a few modes")

    if oe_mean > 0.5:
        print(f"  ✓ Order entropy={oe_mean:.3f}: Different order choices across candidates")
    else:
        print(f"  ✗ Order entropy={oe_mean:.3f}: Orders are deterministic")

    if tc_mean > 0.2:
        print(f"  ✓ Time CV={tc_mean:.3f}: Time allocation varies substantially across candidates")
    else:
        print(f"  ✗ Time CV={tc_mean:.3f}: Time allocation is nearly uniform")

    print("=" * 70)

    # Save
    out = {
        "config": {"n_hamiltonians": N_HAMS, "n_candidates": N_CANDIDATES, "seed": SEED,
                   "guidance_scale": GUIDANCE_SCALE},
        "summary": {
            "jaccard_mean": jd_mean, "jaccard_std": jd_std,
            "n_unique_mean": nu_mean, "n_unique_std": nu_std,
            "order_entropy_mean": oe_mean, "order_entropy_std": oe_std,
            "time_cv_mean": tc_mean, "time_cv_std": tc_std,
        },
    }
    out_path = Path(__file__).parent / "benchmark_results" / "strategy_diversity.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
