"""Qubit-count scaling experiment: Heisenberg n ∈ {4, 6, 8, 10, 12}.

Runs closed-loop optimization for each (n_qubits, Jx, Jy, Jz) point and
compares final fidelity/depth against the Qiskit-4th-order baseline.

For n ≥ 10 exact simulation requires diagonalizing a D×D matrix with
D = 2^n (1024 for n=10, 4096 for n=12), so per-evaluation cost is
substantial. Set --max-n or --samples-per-n to reduce runtime.

Output: experiments/benchmark_results/heisenberg_scaling_extended.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_components(n_qubits: int, n_terms: int, max_groups: int, device: torch.device):
    from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
    from pinn_trotter.diffusion.mixed_model import MixedDiffusionModel
    from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
    from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder

    T = 200
    gnn = HamiltonianGNNEncoder(
        node_feat_dim=n_qubits + 2,
        edge_feat_dim=3,
        hidden_dim=256,
        output_dim=512,
        n_layers=4,
    ).to(device)
    diffusion = MixedDiffusionModel(
        max_groups=max_groups,
        n_terms=n_terms,
        condition_dim=512,
        fused_dim=256,
        time_embed_dim=128,
        grouping_layers=4,
        order_layers=2,
        ts_mlp_layers=3,
        p_cond_drop=0.1,
    ).to(device)
    tm = UniformTransitionMatrix(K=max_groups, T=T, beta_schedule="cosine").to(device)
    order_tm = UniformTransitionMatrix(K=3, T=T, beta_schedule="cosine").to(device)
    ddpm = ContinuousDDPM(T=T, beta_schedule="cosine").to(device)
    return gnn, diffusion, tm, order_tm, ddpm


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = float(statistics.fmean(values))
    s = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": m, "std": s}


# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------

def _run_single_case(
    *,
    n_qubits: int,
    jx: float,
    jy: float,
    jz: float,
    t_total: float,
    n_iterations: int,
    batch_size: int,
    guidance_scale: float,
    lambda_weight: float,
    max_groups: int,
    device: torch.device,
) -> dict:
    from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
    from pinn_trotter.benchmarks.hamiltonians import make_heisenberg
    from pinn_trotter.benchmarks.metrics import exact_fidelity, transpiled_depth
    from pinn_trotter.diffusion.mixed_model import guided_sample
    from pinn_trotter.optimizer.closed_loop import ClosedLoopOptimizer
    from pinn_trotter.pinn.evaluator import _decode_strategy, make_evaluator

    hamiltonian = make_heisenberg(n_qubits=n_qubits, Jx=jx, Jy=jy, Jz=jz, boundary="periodic")
    n_terms = hamiltonian.n_terms
    gnn, diffusion, tm, order_tm, ddpm = _build_components(
        n_qubits, n_terms, max_groups, device
    )

    evaluator = make_evaluator(t_total=t_total, n_qubits=n_qubits, exact_threshold=8)

    optimizer = ClosedLoopOptimizer(
        diffusion_model=diffusion,
        gnn_encoder=gnn,
        transition_matrix=tm,
        order_transition_matrix=order_tm,
        ddpm=ddpm,
        pinn_evaluator=evaluator,
        lambda_weight=lambda_weight,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        n_terms=n_terms,
        max_groups=max_groups,
        checkpoint_interval=max(1, n_iterations),
        device=device,
    )

    def sampler():
        return [hamiltonian for _ in range(batch_size)]

    t0 = time.time()
    history = optimizer.train(n_iterations=n_iterations, hamiltonian_sampler=sampler)
    elapsed = time.time() - t0

    # Final sample from the optimized model
    with torch.no_grad():
        cond = optimizer._encode_hamiltonians([hamiltonian])
        g, ts, o = guided_sample(
            model=optimizer.diffusion_model,
            condition=cond,
            n_terms=n_terms,
            max_groups=max_groups,
            transition_matrix=optimizer.transition_matrix,
            order_transition_matrix=optimizer.order_transition_matrix,
            ddpm=optimizer.ddpm,
            guidance_scale=guidance_scale,
            device=device,
        )
    ours_strategy = _decode_strategy(hamiltonian, g, ts, o, t_total=t_total)
    ours_fid = exact_fidelity(ours_strategy, hamiltonian)
    ours_depth = transpiled_depth(ours_strategy, hamiltonian)

    # Qiskit 4th-order baseline
    base_strategy = QiskitTrotterBaseline().generate_strategy(
        hamiltonian=hamiltonian, t_final=t_total, order=4, n_steps=5,
    )
    base_fid = exact_fidelity(base_strategy, hamiltonian)
    base_depth = transpiled_depth(base_strategy, hamiltonian)

    return {
        "n_qubits": n_qubits,
        "Jx": jx, "Jy": jy, "Jz": jz,
        "n_iterations": n_iterations,
        "elapsed_s": round(elapsed, 1),
        "history": {
            "mean_fidelity": history["mean_fidelity"],
            "mean_depth": history["mean_depth"],
            "pareto_hv": history["pareto_hv"],
        },
        "ours": {"fidelity": ours_fid, "depth": ours_depth},
        "qiskit_4th": {"fidelity": base_fid, "depth": base_depth},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Heisenberg qubit-count scaling")
    parser.add_argument("--n-qubits-list", default="4,6,8,10,12")
    parser.add_argument("--samples-per-n", type=int, default=20)
    parser.add_argument("--param-min", type=float, default=0.2)
    parser.add_argument("--param-max", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-total", type=float, default=2.0)
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--lambda-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="Skip to this case index (for resuming)")
    parser.add_argument("--output",
                        default="experiments/benchmark_results/heisenberg_scaling_extended.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    n_list = [int(x.strip()) for x in args.n_qubits_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    # Build case list
    all_cases: list[tuple[int, float, float, float]] = []
    for n_qubits in n_list:
        for _ in range(args.samples_per_n):
            jx = float(rng.uniform(args.param_min, args.param_max))
            jy = float(rng.uniform(args.param_min, args.param_max))
            jz = float(rng.uniform(args.param_min, args.param_max))
            all_cases.append((n_qubits, jx, jy, jz))

    total = len(all_cases)
    print(f"Total cases: {total} ({len(n_list)} qubit sizes × {args.samples_per_n} samples)")
    print(f"Iterations per case: {args.n_iterations}")
    print()

    results: list[dict] = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for idx in range(args.start_idx, total):
        n_qubits, jx, jy, jz = all_cases[idx]
        print(f"[{idx + 1}/{total}] n={n_qubits} J=({jx:.3f}, {jy:.3f}, {jz:.3f})", end=" ", flush=True)

        try:
            case_result = _run_single_case(
                n_qubits=n_qubits, jx=jx, jy=jy, jz=jz,
                t_total=args.t_total,
                n_iterations=args.n_iterations,
                batch_size=args.batch_size,
                guidance_scale=args.guidance_scale,
                lambda_weight=args.lambda_weight,
                max_groups=args.max_groups,
                device=device,
            )
            results.append(case_result)
            print(f" ours_fid={case_result['ours']['fidelity']:.4f} "
                  f"base_fid={case_result['qiskit_4th']['fidelity']:.4f} "
                  f"({case_result['elapsed_s']:.0f}s)")
        except Exception as exc:
            print(f" FAILED: {exc}")
            results.append({
                "n_qubits": n_qubits, "Jx": jx, "Jy": jy, "Jz": jz,
                "error": str(exc),
            })

        # Incremental save every 5 cases.
        if (idx + 1) % 5 == 0 or idx == total - 1:
            summary: dict = {}
            for nq in n_list:
                group = [r for r in results if r.get("n_qubits") == nq and "ours" in r]
                if group:
                    ours_fids = [r["ours"]["fidelity"] for r in group]
                    base_fids = [r["qiskit_4th"]["fidelity"] for r in group]
                    ours_deps = [r["ours"]["depth"] for r in group]
                    base_deps = [r["qiskit_4th"]["depth"] for r in group]
                    summary[str(nq)] = {
                        "count": len(group),
                        "ours_fidelity": _summarize(ours_fids),
                        "ours_depth": _summarize(ours_deps),
                        "qiskit_4th_fidelity": _summarize(base_fids),
                        "qiskit_4th_depth": _summarize(base_deps),
                    }

            with open(out_path, "w") as fh:
                json.dump({"config": vars(args), "summary": summary, "results": results},
                          fh, indent=2)

    # Final summary
    print("\n=== Scaling Summary ===")
    print(f"{'n':>4}  {'Ours Fid':>12}  {'Ours Depth':>10}  {'Q4 Fid':>12}  {'Q4 Depth':>10}")
    print("-" * 62)
    for nq in n_list:
        group = [r for r in results if r.get("n_qubits") == nq and "ours" in r]
        if group:
            of = _summarize([r["ours"]["fidelity"] for r in group])
            od = _summarize([r["ours"]["depth"] for r in group])
            qf = _summarize([r["qiskit_4th"]["fidelity"] for r in group])
            qd = _summarize([r["qiskit_4th"]["depth"] for r in group])
            print(f"{nq:>4}  {of['mean']:>7.4f}±{of['std']:.4f}  {od['mean']:>6.1f}±{od['std']:.1f}  "
                  f"{qf['mean']:>7.4f}±{qf['std']:.4f}  {qd['mean']:>6.1f}±{qd['std']:.1f}")


if __name__ == "__main__":
    main()
