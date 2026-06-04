"""CFG (Classifier-Free Guidance) strength w sweep.

Evaluates the impact of guidance strength w ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
on fidelity and depth using the main diffusion checkpoint.

Uses the same 50 n=4 TFIM Hamiltonians as the Paulihedral comparison test.
Results are averaged across 5 seeds per Hamiltonian.

Output: experiments/benchmark_results/cfg_sweep_results.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
    return [
        make_tfim(n_qubits, float(rng.uniform(j_min, j_max)), float(rng.uniform(h_min, h_max)), "periodic")
        for _ in range(n_hams)
    ]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    m = float(statistics.fmean(values))
    s = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": m, "std": s}


def _encode(gnn, hamiltonian, device: torch.device, max_n_qubits: int = 8) -> torch.Tensor:
    try:
        data = hamiltonian.to_pyg_data(max_n_qubits=max_n_qubits)
        return gnn(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    except Exception:
        n = hamiltonian.n_qubits
        m = hamiltonian.n_terms
        from pinn_trotter.hamiltonian.pauli_utils import locality

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


def _depth_cx(circuit) -> tuple[int, int]:
    from qiskit import transpile

    t = transpile(circuit, basis_gates=["h", "cx", "rz", "x"], optimization_level=1)
    return int(t.depth()), int(t.count_ops().get("cx", 0))


def _strategy_to_circuit(strategy, hamiltonian):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis.evolution import LieTrotter, SuzukiTrotter

    reps = int(strategy.metadata.get("reps", 1))
    if reps < 1:
        reps = 1
    qc = QuantumCircuit(hamiltonian.n_qubits)
    for group_indices, order, tau in zip(strategy.grouping, strategy.orders, strategy.time_steps):
        pauli_list = [hamiltonian.pauli_strings[idx][::-1] for idx in group_indices]
        coeffs = [float(hamiltonian.coefficients[idx]) for idx in group_indices]
        op = SparsePauliOp(pauli_list, coeffs)
        synthesis = LieTrotter(reps=reps) if order == 1 else SuzukiTrotter(order=order, reps=reps)
        qc.append(PauliEvolutionGate(op, time=float(tau), synthesis=synthesis), range(hamiltonian.n_qubits))
    return qc


def main() -> None:
    n_qubits = 4
    t_total = 2.0
    n_seeds = 5
    n_hams = 50
    w_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from pinn_trotter.utils.model_builder import build_models_from_checkpoint
    from pinn_trotter.diffusion.mixed_model import guided_sample as guided_sample_fn

    ckpt_path = Path(__file__).parent / "closed_loop_checkpoints" / "diffusion_best.pt"
    models = build_models_from_checkpoint(ckpt_path, max_groups=8, device=device)
    gnn = models["gnn"]
    diffusion = models["diffusion"]
    tm = models["tm"]
    order_tm = models["order_tm"]
    ddpm = models["ddpm"]
    max_groups = models["max_groups"]
    max_n_q = models["max_n_qubits"]

    print("Generating Hamiltonians ...")
    hamiltonians = _make_tfim_hamiltonians(n_hams=n_hams, n_qubits=n_qubits, seed=99)

    from pinn_trotter.pinn.evaluator import _decode_strategy

    results: dict = {
        "config": {"n_qubits": n_qubits, "t_total": t_total, "n_hamiltonians": n_hams, "n_seeds": n_seeds},
        "w_values": {},
    }

    for w in w_values:
        print(f"\n=== w = {w} ===")
        fids, depths, cxs = [], [], []

        for i, H in enumerate(hamiltonians):
            c_vec = _encode(gnn, H, device, max_n_qubits=max_n_q)
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                g, ts, o = guided_sample_fn(
                    diffusion, c_vec, H.n_terms, max_groups,
                    tm, order_tm, ddpm,
                    guidance_scale=w,
                    n_steps=100,  # subsample 1000→100 for speed
                )
                strategy = _decode_strategy(H, g, ts, o, t_total)
                circuit = _strategy_to_circuit(strategy, H)
                depth, cx = _depth_cx(circuit)

                psi_exact = _exact_state(H, t_total)
                from qiskit.quantum_info import Statevector

                psi0 = np.zeros(2**n_qubits, dtype=complex)
                psi0[0] = 1.0
                from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline

                psi_trotter = Statevector(QiskitTrotterBaseline._swap_endian(psi0, n_qubits)).evolve(circuit).data
                psi_trotter = QiskitTrotterBaseline._swap_endian(psi_trotter, n_qubits)
                fid = float(np.clip(
                    abs(np.vdot(psi_exact, psi_trotter)) ** 2
                    / (np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2),
                    0.0, 1.0,
                ))

                fids.append(fid)
                depths.append(float(depth))
                cxs.append(float(cx))

            if (i + 1) % 10 == 0:
                recent = fids[-n_seeds:]
                print(f"  {i + 1}/{n_hams}  fid={statistics.fmean(recent):.4f}")

        results["w_values"][str(w)] = {
            "fidelity": _summarize(fids),
            "depth": _summarize(depths),
            "cx_count": _summarize(cxs),
        }
        print(f"  => fid={_summarize(fids)}, depth={_summarize(depths)}")

    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cfg_sweep_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
