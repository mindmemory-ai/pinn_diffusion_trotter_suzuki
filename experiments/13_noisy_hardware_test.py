"""Noisy hardware evaluation: all methods under standard depolarizing noise.

Compares Ours / Qiskit-4th / Qiskit-opt / Cirq / TKET / PennyLane / Paulihedral
under a custom NISQ depolarizing noise model (1q err=0.001, 2q err=0.005, RO=2%).

Output: experiments/benchmark_results/noisy_hardware_results.json
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

from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
from pinn_trotter.benchmarks.baseline_adapters import (
    BASELINE_REGISTRY,
    CirqTrotterBaseline,
    PaulihedralBaseline,
    PennyLaneTrotterBaseline,
    TketTrotterBaseline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tfim_hamiltonians(
    n_hams: int = 30,
    n_qubits: int = 4,
    seed: int = 42,
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


def _noisy_fidelity(
    circuit,
    psi_exact: np.ndarray,
    n_qubits: int,
    shots: int = 8192,
    transpile_opt: int = 1,
) -> tuple[float, int, int]:
    from qiskit import transpile

    t_qc = transpile(circuit, basis_gates=["h", "cx", "rz", "x"], optimization_level=transpile_opt)
    depth = t_qc.depth()
    cx = t_qc.count_ops().get("cx", 0)

    exact_probs = np.abs(psi_exact) ** 2
    exact_probs = np.clip(exact_probs, 0, 1)
    exact_probs /= exact_probs.sum()

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.noise.errors import depolarizing_error

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.001, 1), ["u1", "u2", "u3", "rz", "sx", "x", "h"],
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.005, 2), ["cx"],
        )
        noise_model.add_all_qubit_readout_error([[0.98, 0.02], [0.02, 0.98]])

        sim = AerSimulator(noise_model=noise_model, method="automatic")

        meas_qc = t_qc.copy()
        meas_qc.measure_all()
        job = sim.run(meas_qc, shots=shots)
        counts = job.result().get_counts()

        noisy_probs = np.zeros(2**n_qubits)
        for bitstr, count in counts.items():
            idx = int(bitstr, 2)
            noisy_probs[idx] = count / shots
        noisy_probs = np.clip(noisy_probs, 0, 1)
        noisy_probs /= noisy_probs.sum()

        fid = float(np.sum(np.sqrt(exact_probs * noisy_probs)) ** 2)
        return np.clip(fid, 0.0, 1.0), depth, cx

    except ImportError:
        pass

    cx_count = t_qc.count_ops().get("cx", 0)
    survival = (1 - 0.005) ** cx_count * (1 - 0.001) ** (depth - cx_count)
    return float(max(0.0, survival)), depth, cx_count


def _exact_state(hamiltonian, t_total: float) -> np.ndarray:
    import scipy.linalg

    n = hamiltonian.n_qubits
    H_dense = hamiltonian.to_dense_matrix()
    U = scipy.linalg.expm(-1j * H_dense * t_total)
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0
    return U @ psi0


# ---------------------------------------------------------------------------
# Ours (diffusion model) helpers
# ---------------------------------------------------------------------------


def _encode_hamiltonian(gnn, hamiltonian, device: torch.device, max_n_qubits: int = 8,
                         pauli_enc: bool = True) -> torch.Tensor:
    try:
        data = hamiltonian.to_pyg_data(max_n_qubits=max_n_qubits)
        edge_dim = int(getattr(gnn, "edge_feat_dim", 2))
        if data.edge_attr.shape[1] != edge_dim:
            if data.edge_attr.shape[1] < edge_dim:
                padding = torch.zeros(data.edge_attr.shape[0], edge_dim - data.edge_attr.shape[1],
                                      device=device)
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
            else:
                data.edge_attr = data.edge_attr[:, :edge_dim]
        return gnn(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    except Exception:
        from pinn_trotter.hamiltonian.pauli_utils import encode_pauli_types, locality
        n = hamiltonian.n_qubits
        m = hamiltonian.n_terms
        if pauli_enc:
            feat_dim = 4 + 3 * max(n, max_n_qubits)
        else:
            feat_dim = max(n, max_n_qubits) + 2
        node_feats = np.zeros((m, feat_dim), dtype=np.float32)
        for i, (s, c) in enumerate(zip(hamiltonian.pauli_strings, hamiltonian.coefficients)):
            node_feats[i, 0] = float(c)
            if pauli_enc:
                type_vec, counts = encode_pauli_types(s, max(n, max_n_qubits))
                node_feats[i, 1] = float(counts[0])
                node_feats[i, 2] = float(counts[1])
                node_feats[i, 3] = float(counts[2])
                node_feats[i, 4:] = np.array(type_vec, dtype=np.float32)
            else:
                node_feats[i, 1] = float(locality(s))
                for q, ch in enumerate(s):
                    node_feats[i, 2 + q] = 0.0 if ch == "I" else 1.0
        x = torch.tensor(node_feats, device=device)
        ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_dim = int(getattr(gnn, "edge_feat_dim", 2))
        ea = torch.zeros(0, edge_dim, device=device)
        return gnn(x, ei, ea)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    n_qubits = 4
    t_total = 2.0
    n_steps = 5
    n_hams = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Using standard NISQ depolarizing noise model (1q err=0.001, 2q err=0.005, RO=2%).")

    # --- Load diffusion checkpoint for Ours ---
    from pinn_trotter.utils.model_builder import build_models_from_checkpoint

    ckpt_path = Path(__file__).parent / "closed_loop_checkpoints" / "diffusion_best_20260603_005153_HV9995.5852.pt"
    models = build_models_from_checkpoint(ckpt_path, max_groups=8, device=device,
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

    from pinn_trotter.diffusion.mixed_model import guided_sample
    from pinn_trotter.pinn.evaluator import _decode_strategy

    # --- Build baseline instances ---
    qiskit_4th = QiskitTrotterBaseline()
    cirq = CirqTrotterBaseline(n_steps=n_steps)
    tket = TketTrotterBaseline(n_steps=n_steps)
    pennylane = PennyLaneTrotterBaseline(n_steps=n_steps)
    paulihedral = PaulihedralBaseline(n_steps=n_steps, scheduler="depth")

    methods: dict[str, Any] = {
        "ours": None,             # handled separately
        "qiskit_4th": qiskit_4th,
        "qiskit_opt": qiskit_4th,  # same baseline, different transpile opt
        "cirq": cirq,
        "tket": tket,
        "pennylane": pennylane,
        "paulihedral": paulihedral,
    }

    # --- Generate Hamiltonians ---
    print(f"Generating {n_hams} n={n_qubits} TFIM Hamiltonians ...")
    hamiltonians = _make_tfim_hamiltonians(n_hams=n_hams, n_qubits=n_qubits, seed=42)

    results: dict = {
        "config": {
            "n_qubits": n_qubits,
            "t_total": t_total,
            "n_steps": n_steps,
            "n_hamiltonians": n_hams,
            "noise_model": {"1q_error": 0.001, "2q_error": 0.005, "readout_error": 0.02},
        },
        "methods": {},
    }

    for name in methods:
        print(f"\n--- {name} ---")
        fids: list[float] = []
        nfids: list[float] = []
        depths: list[int] = []
        cxs: list[int] = []

        for i, H in enumerate(hamiltonians):
            psi_exact = _exact_state(H, t_total)

            if name == "ours":
                # Diffusion sampling → strategy → circuit
                c_vec = _encode_hamiltonian(gnn, H, device, max_n_qubits=max_n_q, pauli_enc=pauli_enc)
                with torch.no_grad():
                    torch.manual_seed(42)
                    g, ts, o = guided_sample(
                        diffusion, c_vec, H.n_terms, max_groups,
                        tm, order_tm, ddpm,
                        guidance_scale=3.0,
                        device=device,
                    )
                strategy = _decode_strategy(H, g, ts, o, t_total)
                circuit = _strategy_to_circuit(strategy, H)
                # Noiseless fidelity (via Statevector, not exact diag — must be consistent)
                from qiskit.quantum_info import Statevector

                psi0 = np.zeros(2**n_qubits, dtype=complex)
                psi0[0] = 1.0
                psi_trotter = Statevector(
                    QiskitTrotterBaseline._swap_endian(psi0, n_qubits)
                ).evolve(circuit).data
                psi_trotter = QiskitTrotterBaseline._swap_endian(psi_trotter, n_qubits)
                fid = float(np.clip(
                    abs(np.vdot(psi_exact, psi_trotter)) ** 2
                    / (np.linalg.norm(psi_exact) ** 2 * np.linalg.norm(psi_trotter) ** 2),
                    0.0, 1.0,
                ))
                opt_level = 1
            elif name == "qiskit_4th":
                out = qiskit_4th.evaluate(H, t_total, order=4, n_steps=n_steps)
                circuit = qiskit_4th._strategy_to_qiskit_circuit(out["strategy"], H)
                fid = out["fidelity"]
                opt_level = 1
            elif name == "qiskit_opt":
                out = qiskit_4th.evaluate(H, t_total, order=4, n_steps=n_steps)
                circuit = qiskit_4th._strategy_to_qiskit_circuit(out["strategy"], H)
                fid = out["fidelity"]
                opt_level = 3
            elif name == "paulihedral":
                out = paulihedral.evaluate(H, t_total)
                circuit = _strategy_to_circuit(out["strategy"], H)
                fid = out["fidelity"]
                opt_level = 1
            else:
                # cirq / tket / pennylane
                baseline = methods[name]
                out = baseline.evaluate(H, t_total)
                circuit = _strategy_to_circuit(out["strategy"], H)
                fid = out["fidelity"]
                opt_level = 1

            nfid, depth, cx = _noisy_fidelity(
                circuit, psi_exact, n_qubits, transpile_opt=opt_level,
            )

            fids.append(fid)
            nfids.append(nfid)
            depths.append(depth)
            cxs.append(cx)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n_hams}  noiseless_fid={fid:.4f}  noise_fid={nfid:.4f}  depth={depth}")

        results["methods"][name] = {
            "fidelity": _summarize(fids),
            "noise_fidelity": _summarize(nfids),
            "depth": _summarize([float(d) for d in depths]),
            "cx_count": _summarize([float(c) for c in cxs]),
        }
        print(
            f"  => fid={_summarize(fids)['mean']:.4f}, "
            f"noise_fid={_summarize(nfids)['mean']:.4f}, "
            f"depth={_summarize([float(d) for d in depths])['mean']:.1f}"
        )

    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "noisy_hardware_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
