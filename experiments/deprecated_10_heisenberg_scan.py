"""Heisenberg scan for n=4/6/8 over random coupling points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _build_components(n_qubits: int, n_terms: int, max_groups: int, device: torch.device):
    from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
    from pinn_trotter.diffusion.mixed_model import MixedDiffusionModel
    from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
    from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder

    t_diffusion = 200
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
    tm = UniformTransitionMatrix(K=max_groups, T=t_diffusion, beta_schedule="cosine").to(device)
    order_tm = UniformTransitionMatrix(K=3, T=t_diffusion, beta_schedule="cosine").to(device)
    ddpm = ContinuousDDPM(T=t_diffusion, beta_schedule="cosine").to(device)
    return gnn, diffusion, tm, order_tm, ddpm


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
    gnn_checkpoint: str | None = None,
    diffusion_checkpoint: str | None = None,
    device: torch.device,
) -> dict:
    from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
    from pinn_trotter.benchmarks.hamiltonians import make_heisenberg
    from pinn_trotter.benchmarks.metrics import exact_fidelity, transpiled_depth
    from pinn_trotter.diffusion.mixed_model import guided_sample
    from pinn_trotter.optimizer.closed_loop import ClosedLoopOptimizer
    from pinn_trotter.pinn.evaluator import _decode_strategy, make_evaluator

    hamiltonian = make_heisenberg(n_qubits=n_qubits, Jx=jx, Jy=jy, Jz=jz, boundary="periodic")
    gnn, diffusion, tm, order_tm, ddpm = _build_components(
        hamiltonian.n_qubits, hamiltonian.n_terms, max_groups, device
    )
    if gnn_checkpoint:
        gnn.load_state_dict(torch.load(gnn_checkpoint, map_location=device))
    if diffusion_checkpoint:
        diffusion.load_state_dict(torch.load(diffusion_checkpoint, map_location=device))
    evaluator = make_evaluator(t_total=t_total, n_qubits=hamiltonian.n_qubits, exact_threshold=8)
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
        n_terms=hamiltonian.n_terms,
        max_groups=max_groups,
        checkpoint_interval=max(1, n_iterations),
        device=device,
    )

    def sampler():
        return [hamiltonian for _ in range(batch_size)]

    history = optimizer.train(n_iterations=n_iterations, hamiltonian_sampler=sampler)

    with torch.no_grad():
        cond = optimizer._encode_hamiltonians([hamiltonian])
        g, ts, o = guided_sample(
            model=optimizer.diffusion_model,
            condition=cond,
            n_terms=hamiltonian.n_terms,
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

    baseline = QiskitTrotterBaseline().generate_strategy(
        hamiltonian=hamiltonian,
        t_final=t_total,
        order=4,
        n_steps=5,
    )
    base_fid = exact_fidelity(baseline, hamiltonian)
    base_depth = transpiled_depth(baseline, hamiltonian)

    return {
        "n_qubits": n_qubits,
        "Jx": jx,
        "Jy": jy,
        "Jz": jz,
        "history": {
            "mean_fidelity": history["mean_fidelity"],
            "mean_depth": history["mean_depth"],
            "pareto_hv": history["pareto_hv"],
        },
        "ours": {"fidelity": ours_fid, "depth": ours_depth},
        "qiskit_4th": {"fidelity": base_fid, "depth": base_depth},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits-list", default="4,6,8")
    parser.add_argument("--samples-per-n", type=int, default=50)
    parser.add_argument("--param-min", type=float, default=0.2)
    parser.add_argument("--param-max", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-total", type=float, default=2.0)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--lambda-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--gnn-checkpoint", default=None)
    parser.add_argument("--diffusion-checkpoint", default=None)
    parser.add_argument(
        "--output",
        default="experiments/benchmark_results/heisenberg_scan.json",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    n_list = [int(x.strip()) for x in args.n_qubits_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    all_cases: list[tuple[int, float, float, float]] = []
    for n_qubits in n_list:
        for _ in range(args.samples_per_n):
            jx = float(rng.uniform(args.param_min, args.param_max))
            jy = float(rng.uniform(args.param_min, args.param_max))
            jz = float(rng.uniform(args.param_min, args.param_max))
            all_cases.append((n_qubits, jx, jy, jz))

    if args.max_cases is not None:
        all_cases = all_cases[: args.max_cases]

    results = []
    for n_qubits, jx, jy, jz in all_cases:
        results.append(
            _run_single_case(
                n_qubits=n_qubits,
                jx=jx,
                jy=jy,
                jz=jz,
                t_total=args.t_total,
                n_iterations=args.n_iterations,
                batch_size=args.batch_size,
                guidance_scale=args.guidance_scale,
                lambda_weight=args.lambda_weight,
                max_groups=args.max_groups,
                gnn_checkpoint=args.gnn_checkpoint,
                diffusion_checkpoint=args.diffusion_checkpoint,
                device=device,
            )
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved Heisenberg scan results to {out_path}")


if __name__ == "__main__":
    main()
