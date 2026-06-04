"""Quick diagnostic: test CFG guidance_scale fidelity-depth tradeoff.

Loads a Phase 4 checkpoint, runs inference on 20 Hamiltonians with
4 different guidance_scale values at n_steps=100 (fast mode).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pinn_trotter.benchmarks.metrics import exact_fidelity
from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
from pinn_trotter.diffusion.mixed_model import MixedDiffusionModel, guided_sample
from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder
from pinn_trotter.pinn.evaluator import _decode_strategy


def load_old_checkpoint(ckpt_path: str, device: torch.device):
    """Load a Phase 4 checkpoint saved with old key names, inferring dimensions."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    gnn_sd = state.get("gnn_state") or state.get("gnn_encoder_state")
    ema_sd = state.get("ema_state") or state.get("ema_model_state")
    diff_sd = state.get("diffusion_state") or state.get("diffusion_model_state")

    # Infer from checkpoint weights
    max_n_q = gnn_sd["input_proj.weight"].shape[1] - 2
    max_groups = ema_sd["grouping_head.weight"].shape[0]
    fused_dim = ema_sd["grouping_head.weight"].shape[1] - max_groups
    time_embed_dim = ema_sd["time_proj.0.weight"].shape[1]
    gnn_n_layers = sum(1 for k in gnn_sd if k.startswith("layers.") and k.endswith(".msg_mlp.0.weight"))
    diff_grouping_layers = sum(1 for k in ema_sd if k.startswith("grouping_transformer.") and k.endswith(".attn.out_proj.weight"))
    diff_order_layers = sum(1 for k in ema_sd if k.startswith("order_transformer.") and k.endswith(".attn.out_proj.weight"))
    _ts_linear_count = sum(1 for k in ema_sd if k.startswith("timestep_mlp.") and k.endswith(".weight"))
    ts_mlp_layers = max(1, _ts_linear_count - 1)

    print(f"  Inferred: max_n_q={max_n_q}, K={max_groups}, fused={fused_dim}, "
          f"temb={time_embed_dim}, gnn_L={gnn_n_layers}, "
          f"gL={diff_grouping_layers}, oL={diff_order_layers}, tsL={ts_mlp_layers}")

    T = 1000
    max_M = int(state.get("max_M", 24))

    gnn = HamiltonianGNNEncoder(
        node_feat_dim=max_n_q + 2, edge_feat_dim=3,
        hidden_dim=256, output_dim=512, n_layers=gnn_n_layers,
    ).to(device)
    gnn.load_state_dict(gnn_sd)
    gnn.eval()

    diff = MixedDiffusionModel(
        max_groups=max_groups, n_terms=max_M, condition_dim=512,
        fused_dim=fused_dim, time_embed_dim=time_embed_dim,
        grouping_layers=diff_grouping_layers, order_layers=diff_order_layers,
        ts_mlp_layers=ts_mlp_layers, dropout=0.1, p_cond_drop=0.1,
    ).to(device)
    diff.load_state_dict(ema_sd)
    diff.eval()

    tm = UniformTransitionMatrix(K=max_groups, T=T)
    order_tm = UniformTransitionMatrix(K=3, T=T)
    ddpm = ContinuousDDPM(T=T)

    return {"gnn": gnn, "diffusion": diff, "tm": tm, "order_tm": order_tm, "ddpm": ddpm,
            "max_n_qubits": max_n_q, "max_M": max_M, "max_groups": max_groups}


def sample_hamiltonians(n: int, seed: int = 42):
    from pinn_trotter.benchmarks.hamiltonians import make_heisenberg, make_tfim
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

    rng = np.random.default_rng(seed)
    pauli_chars = ["I", "X", "Y", "Z"]
    out = []
    for _ in range(n):
        nq = 4  # model trained with 4-qubit Hamiltonians
        r = rng.random()
        if r < 0.2:
            n_terms = int(rng.integers(4, 17))
            seen = set()
            paulis, coeffs = [], []
            for _ in range(n_terms):
                while True:
                    s = "".join(rng.choice(pauli_chars, size=nq))
                    if s != "I" * nq and s not in seen:
                        seen.add(s)
                        break
                paulis.append(s)
                coeffs.append(float(np.exp(rng.uniform(np.log(0.1), np.log(5.0)))))
            out.append(HamiltonianGraph(paulis, coeffs, nq))
        elif r < 0.6:
            J = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            h = float(np.exp(rng.uniform(np.log(0.1), np.log(0.5))))
            out.append(make_tfim(nq, J, h))
        else:
            J = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            out.append(make_heisenberg(nq, J, J, J))
    return out


def encode(gnn, hamiltonian, device, max_n_qubits):
    from pinn_trotter.hamiltonian.pauli_utils import locality

    try:
        data = hamiltonian.to_pyg_data(max_n_qubits=max_n_qubits)
        return gnn(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    except Exception:
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = "experiments/checkpoints/ckpt_iter_000049_fid0.9999_depth0023.pt"
    print(f"Loading: {ckpt_path}")
    models = load_old_checkpoint(ckpt_path, device)

    gnn, diffusion = models["gnn"], models["diffusion"]
    tm, order_tm, ddpm = models["tm"], models["order_tm"], models["ddpm"]
    max_n_q, max_groups = models["max_n_qubits"], models["max_groups"]
    tm.to(device); order_tm.to(device); ddpm.to(device)

    t_total = 2.0
    N_HAMILTONIANS = 20
    N_STEPS = 100  # fast mode for diagnostic

    hamiltonians = sample_hamiltonians(N_HAMILTONIANS, seed=123)
    print(f"Sampled {len(hamiltonians)} Hamiltonians (all 4-qubit)")

    guidance_values = [1.0, 2.0, 3.0, 5.0, 10.0]

    print(f"\n{'='*90}")
    print(f"{'Config':<20} {'Fidelity (mean±std)':<28} {'Depth (mean±std)':<25} {'Time':<10}")
    print(f"{'='*90}")

    results = {}
    for w in guidance_values:
        fid_list, depth_list = [], []
        t0 = time.time()

        for h in hamiltonians:
            with torch.no_grad():
                cond = encode(gnn, h, device, max_n_q)
                g, ts, o = guided_sample(
                    model=diffusion, condition=cond, n_terms=h.n_terms,
                    max_groups=max_groups, transition_matrix=tm,
                    order_transition_matrix=order_tm, ddpm=ddpm,
                    guidance_scale=w, n_steps=N_STEPS, device=device,
                )
            strategy = _decode_strategy(h, g, ts, o, t_total=t_total)
            fid_list.append(exact_fidelity(strategy, h))
            depth_list.append(strategy.circuit_depth_estimate())

        elapsed = time.time() - t0
        fid_mean = statistics.fmean(fid_list)
        fid_std = statistics.pstdev(fid_list) if len(fid_list) > 1 else 0.0
        depth_mean = statistics.fmean(depth_list)
        depth_std = statistics.pstdev(depth_list) if len(depth_list) > 1 else 0.0
        print(f"guidance={w:<13.1f} {fid_mean:.4f} ± {fid_std:.4f}          "
              f"{depth_mean:.1f} ± {depth_std:.1f}             {elapsed:.1f}s")
        results[f"guidance={w}"] = {
            "fidelity_mean": fid_mean, "fidelity_std": fid_std,
            "depth_mean": depth_mean, "depth_std": depth_std,
            "fidelity_values": fid_list, "depth_values": depth_list,
        }

    print(f"{'='*90}")

    # Tradeoff summary
    print("\nFidelity-Depth tradeoff:")
    prev_fid, prev_depth = None, None
    for w in guidance_values:
        r = results[f"guidance={w}"]
        delta_fid = ""
        delta_depth = ""
        if prev_fid is not None:
            delta_fid = f" (Δ={r['fidelity_mean']-prev_fid:+.4f})"
            delta_depth = f" (Δ={r['depth_mean']-prev_depth:+.1f})"
        print(f"  guidance={w:.1f}: fid={r['fidelity_mean']:.4f}{delta_fid}, "
              f"depth={r['depth_mean']:.1f}{delta_depth}")
        prev_fid = r['fidelity_mean']
        prev_depth = r['depth_mean']

    # Save
    output_dir = Path("experiments/benchmark_results")
    output_dir.mkdir(exist_ok=True)
    report = {
        "checkpoint": ckpt_path,
        "n_hamiltonians": N_HAMILTONIANS,
        "n_steps": N_STEPS,
        "results": {k: {"fidelity_mean": v["fidelity_mean"], "fidelity_std": v["fidelity_std"],
                         "depth_mean": v["depth_mean"], "depth_std": v["depth_std"]}
                    for k, v in results.items()},
    }
    out_path = output_dir / "diagnostic_guidance_scale.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
