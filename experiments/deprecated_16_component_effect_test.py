"""Experiment 16: Test whether GNN/CFG actually change the generated strategies.

For fixed random seeds, compare strategies from:
  a. full_model:  GNN on,  CFG w=3.0
  b. no_gnn:      GNN off, CFG w=3.0
  c. no_cfg:      GNN on,  CFG w=1.0

If strategies are identical → the component doesn't affect generation.
If strategies differ → the component works but doesn't change fidelity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _encode_hamiltonian(gnn, hamiltonian, device, max_n_qubits: int = 8, disable_gnn: bool = False):
    if disable_gnn:
        output_dim = int(getattr(gnn, "output_dim", 512))
        return torch.zeros((1, output_dim), device=device)
    try:
        data = hamiltonian.to_pyg_data(max_n_qubits=max_n_qubits)
        return gnn(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    except Exception:
        from pinn_trotter.hamiltonian.pauli_utils import locality

        n = hamiltonian.n_qubits
        m = hamiltonian.n_terms
        feat_dim = max(n, max_n_qubits) + 2
        node_feats = np.zeros((m, feat_dim), dtype=np.float32)
        for i, (s, c) in enumerate(zip(hamiltonian.pauli_strings, hamiltonian.coefficients)):
            node_feats[i, 0] = float(c)
            node_feats[i, 1] = float(locality(s))
            for q, ch in enumerate(s):
                node_feats[i, 2 + q] = 0.0 if ch == "I" else 1.0
        x = torch.tensor(node_feats, device=device)
        ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        ea = torch.zeros(0, 3, device=device)
        return gnn(x, ei, ea)


def _exact_state(hamiltonian, t_total: float) -> np.ndarray:
    import scipy.linalg

    n = hamiltonian.n_qubits
    H_dense = hamiltonian.to_dense_matrix()
    U = scipy.linalg.expm(-1j * H_dense * t_total)
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0
    return U @ psi0


def _compute_fidelity(strategy, hamiltonian, psi_exact) -> float:
    from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
    from pinn_trotter.pinn.evaluator import _decode_strategy
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis.evolution import LieTrotter, SuzukiTrotter
    from qiskit import QuantumCircuit

    reps = int(strategy.metadata.get("reps", 1))
    if reps < 1:
        reps = 1
    qc = QuantumCircuit(hamiltonian.n_qubits)
    for group_indices, order, tau in zip(strategy.grouping, strategy.orders, strategy.time_steps):
        pauli_list = [hamiltonian.pauli_strings[idx][::-1] for idx in group_indices]
        coeffs = [float(hamiltonian.coefficients[idx]) for idx in group_indices]
        op = SparsePauliOp(pauli_list, coeffs)
        synthesis = LieTrotter(reps=reps) if order == 1 else SuzukiTrotter(order=order, reps=reps)
        qc.append(PauliEvolutionGate(op, time=float(tau), synthesis=synthesis),
                   range(hamiltonian.n_qubits))

    psi0 = np.zeros(2**hamiltonian.n_qubits, dtype=complex)
    psi0[0] = 1.0
    psi_trotter = Statevector(
        QiskitTrotterBaseline._swap_endian(psi0, hamiltonian.n_qubits)
    ).evolve(qc).data
    psi_trotter = QiskitTrotterBaseline._swap_endian(psi_trotter, hamiltonian.n_qubits)
    fid = float(np.clip(
        abs(np.vdot(psi_exact, psi_trotter)) ** 2
        / (np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2),
        0.0, 1.0,
    ))
    return fid


def main() -> None:
    n_qubits = 4
    t_total = 2.0
    n_hams = 30
    n_seeds = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    from pinn_trotter.utils.model_builder import build_models_from_checkpoint

    ckpt_path = Path(__file__).parent / "closed_loop_checkpoints" / "diffusion_best.pt"
    models = build_models_from_checkpoint(ckpt_path, max_groups=8, device=device)
    gnn = models["gnn"]
    diffusion = models["diffusion"]
    tm = models["tm"]
    order_tm = models["order_tm"]
    ddpm = models["ddpm"]
    max_groups = models["max_groups"]
    max_n_q = models["max_n_qubits"]

    from pinn_trotter.diffusion.mixed_model import guided_sample
    from pinn_trotter.pinn.evaluator import _decode_strategy

    # Generate Hamiltonians
    from pinn_trotter.benchmarks.hamiltonians import make_tfim

    rng = np.random.default_rng(42)
    hamiltonians = []
    for _ in range(n_hams):
        j_val = float(rng.uniform(0.5, 2.0))
        h_val = float(rng.uniform(0.1, 0.5))
        hamiltonians.append(make_tfim(n_qubits, j_val, h_val, "periodic"))

    results: dict[str, Any] = {"config": {"n_qubits": n_qubits, "n_hams": n_hams, "n_seeds": n_seeds}, "cases": []}

    for i, H in enumerate(hamiltonians):
        psi_exact = _exact_state(H, t_total)

        case = {"ham_idx": i, "J": float(H.coefficients[0]), "h": float(H.coefficients[-1]), "seeds": []}

        # Encode condition once for GNN-on cases
        cond_normal = _encode_hamiltonian(gnn, H, device, max_n_qubits=max_n_q, disable_gnn=False)
        cond_zero = _encode_hamiltonian(gnn, H, device, max_n_qubits=max_n_q, disable_gnn=True)

        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            seed_data = {"seed": seed, "variants": {}}

            # --- full_model: GNN on, CFG w=3.0 ---
            torch.manual_seed(seed)
            with torch.no_grad():
                g_a, ts_a, o_a = guided_sample(
                    diffusion, cond_normal, H.n_terms, max_groups,
                    tm, order_tm, ddpm, guidance_scale=3.0, device=device,
                )
            strat_a = _decode_strategy(H, g_a, ts_a, o_a, t_total)
            fid_a = _compute_fidelity(strat_a, H, psi_exact)
            seed_data["variants"]["full_model"] = {
                "grouping": g_a.cpu().tolist(),
                "orders": o_a.cpu().tolist(),
                "fidelity": fid_a,
            }

            # --- no_gnn: GNN off, CFG w=3.0, SAME seed ---
            torch.manual_seed(seed)
            with torch.no_grad():
                g_b, ts_b, o_b = guided_sample(
                    diffusion, cond_zero, H.n_terms, max_groups,
                    tm, order_tm, ddpm, guidance_scale=3.0, device=device,
                )
            strat_b = _decode_strategy(H, g_b, ts_b, o_b, t_total)
            fid_b = _compute_fidelity(strat_b, H, psi_exact)

            # Compare strategies
            g_match = (g_a == g_b).all().item()
            o_match = (o_a == o_b).all().item()
            ts_match = torch.allclose(ts_a, ts_b, atol=1e-6) if ts_a.shape == ts_b.shape else ts_a.shape == ts_b.shape
            seed_data["variants"]["no_gnn"] = {
                "grouping": g_b.cpu().tolist(),
                "orders": o_b.cpu().tolist(),
                "fidelity": fid_b,
                "identical_to_full": bool(g_match and o_match),
            }

            # --- no_cfg: GNN on, CFG w=1.0, SAME seed ---
            torch.manual_seed(seed)
            with torch.no_grad():
                g_c, ts_c, o_c = guided_sample(
                    diffusion, cond_normal, H.n_terms, max_groups,
                    tm, order_tm, ddpm, guidance_scale=1.0, device=device,
                )
            strat_c = _decode_strategy(H, g_c, ts_c, o_c, t_total)
            fid_c = _compute_fidelity(strat_c, H, psi_exact)

            g_match_c = (g_a == g_c).all().item()
            o_match_c = (o_a == o_c).all().item()
            seed_data["variants"]["no_cfg"] = {
                "grouping": g_c.cpu().tolist(),
                "orders": o_c.cpu().tolist(),
                "fidelity": fid_c,
                "identical_to_full": bool(g_match_c and o_match_c),
            }

            case["seeds"].append(seed_data)

        # Summary for this Hamiltonian
        n_gnn_diff = sum(1 for s in case["seeds"] if not s["variants"]["no_gnn"]["identical_to_full"])
        n_cfg_diff = sum(1 for s in case["seeds"] if not s["variants"]["no_cfg"]["identical_to_full"])
        print(f"H[{i}] J={case['J']:.2f} h={case['h']:.2f} | "
              f"full_fid={case['seeds'][0]['variants']['full_model']['fidelity']:.4f} | "
              f"no_gnn differs in {n_gnn_diff}/{n_seeds} seeds | "
              f"no_cfg differs in {n_cfg_diff}/{n_seeds} seeds")
        results["cases"].append(case)

    # Aggregate
    total_gnn_diff = sum(
        1 for c in results["cases"] for s in c["seeds"]
        if not s["variants"]["no_gnn"]["identical_to_full"]
    )
    total_cfg_diff = sum(
        1 for c in results["cases"] for s in c["seeds"]
        if not s["variants"]["no_cfg"]["identical_to_full"]
    )
    total = n_hams * n_seeds
    print(f"\n=== Summary ===")
    print(f"no_gnn differs from full_model in {total_gnn_diff}/{total} cases ({100*total_gnn_diff/total:.1f}%)")
    print(f"no_cfg differs from full_model in {total_cfg_diff}/{total} cases ({100*total_cfg_diff/total:.1f}%)")

    # Save
    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "component_effect_test.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
