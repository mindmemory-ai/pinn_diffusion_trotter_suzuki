"""LiH bond-length scan with lightweight closed-loop proxy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


class DepthProxyEvaluator:
    """Cheap evaluator for large molecular systems where exact evolution is expensive."""

    def __init__(self, t_total: float, depth_scale: float = 20000.0) -> None:
        self.t_total = t_total
        self.depth_scale = depth_scale

    def __call__(self, hamiltonian, grouping_t, timesteps_t, orders_t) -> float:
        from pinn_trotter.pinn.evaluator import _decode_strategy

        strategy = _decode_strategy(hamiltonian, grouping_t, timesteps_t, orders_t, self.t_total)
        depth = max(strategy.circuit_depth_estimate(), 1)
        return float(np.exp(-depth / self.depth_scale))

    def circuit_depth(self, hamiltonian, grouping_t, timesteps_t, orders_t) -> int:
        from pinn_trotter.pinn.evaluator import _decode_strategy

        strategy = _decode_strategy(hamiltonian, grouping_t, timesteps_t, orders_t, self.t_total)
        return strategy.circuit_depth_estimate()

    def proxy_from_depth(self, depth: int) -> float:
        depth = max(int(depth), 1)
        return float(np.exp(-depth / self.depth_scale))


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


def _run_single_bond(
    bond_length: float,
    *,
    basis: str,
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
    from pinn_trotter.benchmarks.hamiltonians import generate_lih_hamiltonian
    from pinn_trotter.diffusion.mixed_model import guided_sample
    from pinn_trotter.optimizer.closed_loop import ClosedLoopOptimizer
    from pinn_trotter.pinn.evaluator import _decode_strategy

    hamiltonian = generate_lih_hamiltonian(bond_length=bond_length, basis=basis)
    gnn, diffusion, tm, order_tm, ddpm = _build_components(
        hamiltonian.n_qubits, hamiltonian.n_terms, max_groups, device
    )
    if gnn_checkpoint:
        gnn.load_state_dict(torch.load(gnn_checkpoint, map_location=device))
    if diffusion_checkpoint:
        diffusion.load_state_dict(torch.load(diffusion_checkpoint, map_location=device))
    evaluator = DepthProxyEvaluator(t_total=t_total)
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
    ours_depth = ours_strategy.circuit_depth_estimate()
    ours_proxy_fidelity = evaluator(
        hamiltonian,
        g,
        ts,
        torch.nn.functional.one_hot(o, num_classes=3).float() if o.dim() == 2 else o,
    )

    baseline = QiskitTrotterBaseline().generate_strategy(
        hamiltonian=hamiltonian,
        t_final=t_total,
        order=4,
        n_steps=5,
    )
    base_depth = baseline.circuit_depth_estimate()
    base_proxy_fidelity = evaluator.proxy_from_depth(base_depth)

    return {
        "bond_length": bond_length,
        "n_qubits": hamiltonian.n_qubits,
        "n_terms": hamiltonian.n_terms,
        "history": {
            "mean_proxy_fidelity": history["mean_fidelity"],
            "mean_depth": history["mean_depth"],
            "pareto_hv": history["pareto_hv"],
        },
        "ours": {"proxy_fidelity": ours_proxy_fidelity, "depth": ours_depth},
        "qiskit_4th": {"proxy_fidelity": base_proxy_fidelity, "depth": base_depth},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--bond-start", type=float, default=1.0)
    parser.add_argument("--bond-end", type=float, default=4.0)
    parser.add_argument("--bond-step", type=float, default=0.1)
    parser.add_argument("--t-total", type=float, default=2.0)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--lambda-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-bonds", type=int, default=None)
    parser.add_argument("--gnn-checkpoint", default=None)
    parser.add_argument("--diffusion-checkpoint", default=None)
    parser.add_argument("--output", default="experiments/benchmark_results/lih_bond_scan.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    bonds = np.arange(args.bond_start, args.bond_end + 1e-9, args.bond_step).round(3).tolist()
    if args.max_bonds is not None:
        bonds = bonds[: args.max_bonds]

    results = []
    for b in bonds:
        results.append(
            _run_single_bond(
                b,
                basis=args.basis,
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
    print(f"Saved LiH scan results to {out_path}")


if __name__ == "__main__":
    main()
