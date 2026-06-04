"""Fidelity-matched circuit cost comparison — ALL baselines.

Extends 05c to compare against cirq, tket, pennylane, paulihedral, and
paulihedral_4th (not just qiskit_4th). Each baseline sweeps n_steps to find
the shallowest circuit meeting each fidelity threshold.

Usage:
    python experiments/05d_fidelity_matched_all_baselines.py \
        benchmark.model_ckpt=... \
        benchmark.n_test_hamiltonians=30 \
        benchmark.n_candidates=100 \
        ++benchmark.methods=ours,qiskit_4th,cirq,tket,pennylane,paulihedral,paulihedral_4th
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with 05c_fidelity_matched.py)
# ---------------------------------------------------------------------------

def _sample_hamiltonians(n: int, seed: int, h_min=0.1, h_max=0.5,
                         tfim_ratio=0.4, random_ratio=0.2):
    """Generate n mixed-type Hamiltonians."""
    from pinn_trotter.benchmarks.hamiltonians import make_heisenberg, make_tfim
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

    rng = np.random.default_rng(seed)
    nq_choices = [4, 6, 8]
    nq_probs = [0.6, 0.3, 0.1]
    pauli_chars = ["I", "X", "Y", "Z"]
    hamiltonians = []
    for _ in range(n):
        nq = int(rng.choice(nq_choices, p=nq_probs))
        r = rng.random()
        if r < random_ratio:
            n_terms = int(rng.integers(4, 17))
            seen: set[str] = set()
            paulis, coeffs = [], []
            for _ in range(n_terms):
                while True:
                    s = "".join(rng.choice(pauli_chars, size=nq))
                    if s != "I" * nq and s not in seen:
                        seen.add(s)
                        break
                paulis.append(s)
                coeffs.append(float(np.exp(rng.uniform(np.log(0.1), np.log(5.0)))))
            hamiltonians.append(HamiltonianGraph(paulis, coeffs, nq))
        elif r < random_ratio + (1.0 - random_ratio) * tfim_ratio:
            j_val = float(rng.uniform(0.5, 2.0))
            h_val = float(rng.uniform(h_min, h_max))
            hamiltonians.append(make_tfim(nq, j_val, h_val))
        else:
            J = float(rng.uniform(0.5, 2.0))
            hamiltonians.append(make_heisenberg(nq, J, J, J))
    return hamiltonians


def _pad_edge_attr(edge_attr: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad edge_attr to target_dim if needed (backward compat: Phase 3 uses dim=3)."""
    if edge_attr.numel() == 0:
        return torch.zeros((0, target_dim), device=edge_attr.device, dtype=edge_attr.dtype)
    cur = edge_attr.shape[1]
    if cur >= target_dim:
        return edge_attr[:, :target_dim]
    pad = torch.zeros((edge_attr.shape[0], target_dim - cur), device=edge_attr.device, dtype=edge_attr.dtype)
    return torch.cat([edge_attr, pad], dim=1)


def _encode_hamiltonian(gnn, hamiltonian, device, max_n_qubits=8,
                        disable_gnn_encoder=False, pauli_encoding=False):
    """Encode HamiltonianGraph -> (1, D) condition vector."""
    if disable_gnn_encoder:
        output_dim = int(getattr(gnn, "output_dim", 512))
        return torch.zeros((1, output_dim), device=device)
    try:
        data = hamiltonian.to_pyg_data(max_n_qubits=max_n_qubits,
                                       pauli_encoding=pauli_encoding)
        expected_dim = int(gnn.input_proj.in_features)
        if data.x.shape[1] != expected_dim:
            raise ValueError(
                f"to_pyg_data returned dim={data.x.shape[1]}, "
                f"expected={expected_dim}, pauli_encoding={pauli_encoding}"
            )
        # Ensure edge_attr dim matches GNN expectation (Phase 3 used 3, Phase 4 uses 2)
        gnn_edge_dim = int(getattr(gnn, "edge_feat_dim", 2))
        if data.edge_attr.shape[1] != gnn_edge_dim:
            data.edge_attr = _pad_edge_attr(data.edge_attr, gnn_edge_dim)
        return gnn(data.x.to(device), data.edge_index.to(device),
                   data.edge_attr.to(device))
    except Exception:
        from pinn_trotter.hamiltonian.pauli_utils import encode_pauli_types, locality
        n = hamiltonian.n_qubits
        if pauli_encoding:
            feat_dim = 4 + 3 * max(n, max_n_qubits)
        else:
            feat_dim = max(n, max_n_qubits) + 2
        x = torch.zeros((hamiltonian.n_terms, feat_dim), device=device)
        for i, (s, c) in enumerate(zip(hamiltonian.pauli_strings,
                                        hamiltonian.coefficients)):
            x[i, 0] = float(c)
            if pauli_encoding:
                type_vec, counts = encode_pauli_types(s, max(n, max_n_qubits))
                x[i, 1] = float(counts[0])
                x[i, 2] = float(counts[1])
                x[i, 3] = float(counts[2])
                x[i, 4:] = torch.tensor(type_vec, device=device)
            else:
                x[i, 1] = float(locality(s))
                for q in range(n):
                    x[i, 2 + q] = 1.0 if s[q] != "I" else 0.0
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_dim = int(getattr(gnn, "edge_feat_dim", 2))
        ea = torch.zeros((0, edge_dim), device=device)
        return gnn(x, ei, ea)


def _instantiate_baseline(name: str, n_steps: int, order: int, scheduler: str = "depth"):
    """Create a single baseline adapter instance."""
    from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
    from pinn_trotter.benchmarks.baseline_adapters import BASELINE_REGISTRY

    if name == "qiskit_4th":
        return name, QiskitTrotterBaseline()
    cls = BASELINE_REGISTRY[name]
    if name in ("paulihedral", "paulihedral_4th"):
        return name, cls(n_steps=n_steps, scheduler=scheduler)
    else:
        return name, cls(n_steps=n_steps, order=order)


def _evaluate_baseline(name: str, adapter, hamiltonian, t_total: float,
                       n_steps: int, order: int, fidelity_only: bool):
    """Evaluate one baseline at a specific n_steps."""
    from pinn_trotter.benchmarks.metrics import cx_count, transpiled_depth

    if name == "qiskit_4th":
        res = adapter.evaluate(
            hamiltonian=hamiltonian, t_total=t_total,
            order=order, n_steps=n_steps,
        )
    elif name in ("paulihedral", "paulihedral_4th"):
        adapter.n_steps = n_steps
        res = adapter.evaluate(hamiltonian, t_total)
    else:
        adapter.n_steps = n_steps
        res = adapter.evaluate(hamiltonian, t_total)

    fid = res["fidelity"]
    if fidelity_only:
        depth = res.get("strategy", None)
        if depth is not None:
            depth = depth.circuit_depth_estimate()
        else:
            depth = 0
        cx = 0
    else:
        if "depth" in res and "cx_count" in res:
            depth = int(res["depth"])
            cx = int(res["cx_count"])
        else:
            strategy = res["strategy"]
            depth = int(transpiled_depth(strategy, hamiltonian))
            cx = int(cx_count(strategy, hamiltonian))
    return {"fidelity": fid, "depth": depth, "cx_count": cx, "n_steps": n_steps}


# ---------------------------------------------------------------------------
# Architecture auto-detection (for Phase 3 vs Phase 4 compatibility)
# ---------------------------------------------------------------------------

def _detect_architecture(checkpoint_path: str) -> dict:
    """Auto-detect GNN and diffusion architecture from checkpoint state dict.

    Returns:
        Dict with keys: gnn_hidden_dim, gnn_output_dim, gnn_n_layers,
                        diffusion_fused_dim, diffusion_time_embed_dim,
                        diffusion_grouping_layers, diffusion_order_layers,
                        diffusion_ts_mlp_layers
    """
    import re

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gnn_sd = ckpt["gnn_state"]
    diff_sd = ckpt["diffusion_state"]

    # GNN
    gnn_hidden_dim = int(gnn_sd["input_proj.weight"].shape[0])
    gnn_output_dim = int(gnn_sd["pooling.proj.weight"].shape[0])
    gnn_layers = set()
    for k in gnn_sd:
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            gnn_layers.add(int(m.group(1)))
    gnn_n_layers = len(gnn_layers)

    # Diffusion
    # grouped_head: (K, K+fused_dim) → fused_dim = shape[1] - max_groups
    max_groups_from_ckpt = int(ckpt.get("max_groups", 8))
    fused_dim = int(diff_sd["grouping_head.weight"].shape[1]) - max_groups_from_ckpt
    time_embed_dim = int(diff_sd["time_proj.0.weight"].shape[1])
    grouping_layers = set()
    order_layers = set()
    ts_linear = 0
    for k in diff_sd:
        gm = re.search(r"grouping_transformer\.(\d+)\.", k)
        if gm:
            grouping_layers.add(int(gm.group(1)))
        om = re.search(r"order_transformer\.(\d+)\.", k)
        if om:
            order_layers.add(int(om.group(1)))
        if re.search(r"timestep_mlp\.\d+\.weight", k):
            ts_linear += 1
    diffusion_grouping_layers = len(grouping_layers)
    diffusion_order_layers = len(order_layers)
    diffusion_ts_mlp_layers = max(1, ts_linear - 1)  # N linear layers = ts_mlp_layers + 1

    return {
        "gnn_hidden_dim": gnn_hidden_dim,
        "gnn_output_dim": gnn_output_dim,
        "gnn_n_layers": gnn_n_layers,
        "diffusion_fused_dim": fused_dim,
        "diffusion_time_embed_dim": time_embed_dim,
        "diffusion_grouping_layers": diffusion_grouping_layers,
        "diffusion_order_layers": diffusion_order_layers,
        "diffusion_ts_mlp_layers": diffusion_ts_mlp_layers,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="experiment/tfim_4q_poc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from pinn_trotter.diffusion.mixed_model import guided_sample
    from pinn_trotter.benchmarks.metrics import cx_count, exact_fidelity, transpiled_depth
    from pinn_trotter.pinn.evaluator import _decode_strategy
    from pinn_trotter.utils.model_builder import build_models_from_checkpoint

    bench_cfg = cfg.get("benchmark", {})
    exp_cfg = cfg.get("experiment", {})
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ---- Config ----
    t_total = float(exp_cfg.get("t_total", 2.0))
    n_test_h = int(bench_cfg.get("n_test_hamiltonians", 30))
    n_candidates = int(bench_cfg.get("n_candidates", 100))
    fidelity_thresholds = list(bench_cfg.get("fidelity_thresholds", [0.90, 0.95, 0.99]))
    baseline_n_steps = list(bench_cfg.get("baseline_n_steps", [1, 2, 3, 4, 5, 6, 8, 10]))
    trotter_order = int(bench_cfg.get("trotter_order", 4))
    seed0 = int(bench_cfg.get("seed", 42))
    guidance_scale = float(bench_cfg.get("guidance_scale", 3.0))
    inference_steps = int(bench_cfg.get("inference_steps", 0)) or None
    fidelity_only = bool(bench_cfg.get("fidelity_only", False))
    disable_gnn_encoder = bool(bench_cfg.get("disable_gnn_encoder", False))
    fixed_order = bool(bench_cfg.get("fixed_order", False))
    uniform_time = bool(bench_cfg.get("uniform_time", False))
    save_per_candidate = bool(bench_cfg.get("save_per_candidate", False))
    h_min = float(bench_cfg.get("h_min", 0.1))
    h_max = float(bench_cfg.get("h_max", 0.5))
    tfim_ratio = float(bench_cfg.get("tfim_ratio", 0.4))
    random_ratio = float(bench_cfg.get("random_ratio", 0.2))
    model_ckpt = str(bench_cfg.get("model_ckpt", ""))
    paulihedral_scheduler = str(bench_cfg.get("paulihedral_scheduler", "depth"))
    output_filename = str(bench_cfg.get("output_filename", ""))
    output_prefix = str(bench_cfg.get("output_prefix", ""))

    # Parse baseline methods
    raw_methods = bench_cfg.get("methods", None)
    if raw_methods is None:
        baseline_methods = ["qiskit_4th", "qiskit_group_commuting", "cirq", "tket",
                           "pennylane", "paulihedral", "paulihedral_4th"]
    elif isinstance(raw_methods, str):
        baseline_methods = [m.strip() for m in raw_methods.split(",") if m.strip()]
    else:
        baseline_methods = [str(m).strip() for m in raw_methods if str(m).strip()]

    # Filter: "ours" is handled separately, baselines are the rest
    baseline_methods = [m for m in baseline_methods if m != "ours"]
    log.info("Baseline methods: %s", baseline_methods)

    device_str = str(bench_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    # ---- Load model ----
    max_groups_cfg = int(bench_cfg.get("n_groups_max", 8))
    arch = _detect_architecture(model_ckpt)
    log.info("Auto-detected architecture: %s", arch)
    models = build_models_from_checkpoint(
        model_ckpt, max_groups=max_groups_cfg, device=device,
        gnn_hidden_dim=arch["gnn_hidden_dim"],
        gnn_output_dim=arch["gnn_output_dim"],
        gnn_n_layers=arch["gnn_n_layers"],
        diffusion_fused_dim=arch["diffusion_fused_dim"],
        diffusion_time_embed_dim=arch["diffusion_time_embed_dim"],
        diffusion_grouping_layers=arch["diffusion_grouping_layers"],
        diffusion_order_layers=arch["diffusion_order_layers"],
        diffusion_ts_mlp_layers=arch["diffusion_ts_mlp_layers"],
    )
    gnn = models["gnn"]
    diffusion = models["diffusion"]
    tm = models["tm"]
    order_tm = models["order_tm"]
    ddpm = models["ddpm"]
    max_n_q = models["max_n_qubits"]
    max_groups = models["max_groups"]
    pauli_enc = models.get("pauli_encoding", False)
    log.info("Loaded model: max_n_qubits=%d max_M=%d pauli_enc=%s",
             max_n_q, models["max_M"], pauli_enc)

    # ---- Instantiate all baselines once ----
    baseline_adapters: dict[str, Any] = {}
    for bname in baseline_methods:
        _, adapter = _instantiate_baseline(bname, baseline_n_steps[0],
                                           trotter_order, paulihedral_scheduler)
        baseline_adapters[bname] = adapter
        log.info("Baseline %s: ready", bname)

    # ---- Sample Hamiltonians ----
    hamiltonians = _sample_hamiltonians(n_test_h, seed0, h_min=h_min, h_max=h_max,
                                        tfim_ratio=tfim_ratio, random_ratio=random_ratio)
    log.info("Sampled %d Hamiltonians", len(hamiltonians))

    # ---- Evaluate ours: N candidates per Hamiltonian ----
    log.info("Evaluating ours with n_candidates=%d...", n_candidates)
    ours_results: list[dict] = []

    t0 = time.time()
    for idx, H in enumerate(hamiltonians):
        with torch.no_grad():
            cond = _encode_hamiltonian(gnn, H, device, max_n_qubits=max_n_q,
                                       disable_gnn_encoder=disable_gnn_encoder,
                                       pauli_encoding=pauli_enc)
            cond_r = cond.repeat(n_candidates, 1)
            g_all, ts_all, o_all = guided_sample(
                model=diffusion, condition=cond_r,
                n_terms=H.n_terms, max_groups=max_groups,
                transition_matrix=tm, order_transition_matrix=order_tm,
                ddpm=ddpm, guidance_scale=guidance_scale,
                n_steps=inference_steps, device=device,
            )

        # Ablation overrides (P14): fix order and/or time allocation
        # o_all: integer labels, 0=order1, 1=order2, 2=order4
        if fixed_order:
            o_all[:] = 2  # all groups use Suzuki-4 (index 2)
        if uniform_time:
            ts_all[:] = t_total / max_groups

        candidates = []
        for k in range(n_candidates):
            s = _decode_strategy(H, g_all[k:k+1], ts_all[k:k+1], o_all[k:k+1],
                                 t_total=t_total)
            fid = exact_fidelity(s, H, psi_0=None)
            if fidelity_only:
                depth = s.circuit_depth_estimate()
                cx = 0
            else:
                depth = int(transpiled_depth(s, H))
                cx = int(cx_count(s, H))
            candidates.append({"fidelity": fid, "depth": depth, "cx_count": cx})

        per_threshold = {}
        for T in fidelity_thresholds:
            meeting = [c for c in candidates if c["fidelity"] >= T]
            if meeting:
                best = min(meeting, key=lambda c: c["depth"])
                per_threshold[str(T)] = {
                    "fidelity": best["fidelity"],
                    "depth": best["depth"],
                    "cx_count": best["cx_count"],
                }
            else:
                per_threshold[str(T)] = None

        ours_results.append({
            "hamiltonian_idx": idx,
            "n_qubits": H.n_qubits,
            "n_terms": H.n_terms,
            "thresholds": per_threshold,
            **({"candidates": candidates} if save_per_candidate else {}),
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            log.info("  ours %d/%d (%.1fs)", idx + 1, n_test_h, elapsed)

    ours_elapsed = time.time() - t0
    log.info("Ours done in %.1fs (%.1fs/ham)", ours_elapsed, ours_elapsed / n_test_h)

    # ---- Evaluate ALL baselines: n_steps sweep ----
    all_baseline_results: dict[str, list] = {bname: [] for bname in baseline_methods}

    t0_bl = time.time()
    for bname in baseline_methods:
        adapter = baseline_adapters[bname]
        log.info("Evaluating baseline: %s", bname)

        for idx, H in enumerate(hamiltonians):
            per_threshold = {}
            for T in fidelity_thresholds:
                best_depth = float("inf")
                best_info = None
                for ns in baseline_n_steps:
                    res = _evaluate_baseline(bname, adapter, H, t_total, ns,
                                             trotter_order, fidelity_only)
                    if res["fidelity"] >= T:
                        if res["depth"] < best_depth:
                            best_depth = res["depth"]
                            best_info = {
                                "fidelity": res["fidelity"],
                                "depth": res["depth"],
                                "cx_count": res["cx_count"],
                                "n_steps": ns,
                            }
                per_threshold[str(T)] = best_info

            all_baseline_results[bname].append({
                "hamiltonian_idx": idx,
                "n_qubits": H.n_qubits,
                "n_terms": H.n_terms,
                "thresholds": per_threshold,
            })

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0_bl
                log.info("  %s %d/%d (%.1fs)", bname, idx + 1, n_test_h, elapsed)

    bl_elapsed = time.time() - t0_bl
    log.info("All baselines done in %.1fs", bl_elapsed)

    # ---- Aggregate ----
    def _summarize(values):
        if not values:
            return {"mean": 0, "std": 0}
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    summary: dict[str, Any] = {"thresholds": {}}
    for T in fidelity_thresholds:
        T_str = str(T)
        entry: dict[str, Any] = {}

        # Ours
        ours_depths, ours_fids, ours_cxs = [], [], []
        ours_miss = 0
        for r in ours_results:
            tinfo = r["thresholds"].get(T_str)
            if tinfo is not None:
                ours_depths.append(tinfo["depth"])
                ours_fids.append(tinfo["fidelity"])
                ours_cxs.append(tinfo["cx_count"])
            else:
                ours_miss += 1
        entry["ours"] = {
            "reachable": n_test_h - ours_miss,
            "miss": ours_miss,
            "depth": _summarize(ours_depths),
            "fidelity": _summarize(ours_fids),
            "cx_count": _summarize(ours_cxs),
        }

        # Each baseline
        for bname in baseline_methods:
            bres = all_baseline_results[bname]
            b_depths, b_fids, b_cxs = [], [], []
            b_miss = 0
            for r in bres:
                tinfo = r["thresholds"].get(T_str)
                if tinfo is not None:
                    b_depths.append(tinfo["depth"])
                    b_fids.append(tinfo["fidelity"])
                    b_cxs.append(tinfo["cx_count"])
                else:
                    b_miss += 1
            entry[bname] = {
                "reachable": n_test_h - b_miss,
                "miss": b_miss,
                "depth": _summarize(b_depths),
                "fidelity": _summarize(b_fids),
                "cx_count": _summarize(b_cxs),
            }

            # Depth/CX reduction vs this baseline
            if ours_depths and b_depths:
                ours_mean = np.mean(ours_depths)
                bl_mean = np.mean(b_depths)
                entry[f"depth_reduction_vs_{bname}"] = (
                    float(bl_mean / ours_mean) if ours_mean > 0 else 0.0
                )
                bl_cx_mean = np.mean(b_cxs)
                ours_cx_mean = np.mean(ours_cxs)
                entry[f"cx_reduction_vs_{bname}"] = (
                    float(bl_cx_mean / ours_cx_mean) if ours_cx_mean > 0 else 0.0
                )

        summary["thresholds"][T_str] = entry

    # ---- Report ----
    print("\n" + "=" * 120)
    print("FIDELITY-MATCHED DEPTH COMPARISON — ALL BASELINES")
    print(f"{n_test_h} Hamiltonians x {n_candidates} candidates, order={trotter_order}")
    print("=" * 120)

    for T in fidelity_thresholds:
        T_str = str(T)
        entry = summary["thresholds"][T_str]
        print(f"\n{'─' * 120}")
        print(f"  Threshold T = {T}")
        print(f"{'─' * 120}")
        header = f"{'Method':>18s} | {'Reach':>8s} | {'Fidelity':>16s} | {'Depth':>16s} | {'CX Count':>16s} | {'Depth↓':>10s} | {'CX↓':>10s}"
        print(header)
        print("-" * 120)

        all_method_names = ["ours"] + baseline_methods
        for method in all_method_names:
            m = entry[method]
            reach_pct = f"{m['reachable']}/{n_test_h}"
            fid_str = f"{m['fidelity']['mean']:.4f}±{m['fidelity']['std']:.3f}"
            dep_str = f"{m['depth']['mean']:.0f}±{m['depth']['std']:.0f}"
            cx_str = f"{m['cx_count']['mean']:.0f}±{m['cx_count']['std']:.0f}"

            dr_key = f"depth_reduction_vs_{method}"
            cr_key = f"cx_reduction_vs_{method}"
            if method != "ours" and dr_key in entry:
                dr = f"{entry[dr_key]:.1f}x"
                cr = f"{entry[cr_key]:.1f}x"
            else:
                dr = "-"
                cr = "-"

            print(f"{method:>18s} | {reach_pct:>8s} | {fid_str:>16s} | {dep_str:>16s} | {cx_str:>16s} | {dr:>10s} | {cr:>10s}")

    print(f"\n{'═' * 120}")
    print(f"Ours elapsed: {ours_elapsed:.0f}s | Baselines elapsed: {bl_elapsed:.0f}s")
    print(f"{'═' * 120}")

    # ---- Save ----
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_filename:
        fname = output_filename if output_filename.endswith(".json") else output_filename + ".json"
    elif output_prefix:
        base = "fidelity_matched_all_baselines_per_candidate" if save_per_candidate else "fidelity_matched_all_baselines"
        fname = f"{output_prefix}_{base}.json"
    else:
        fname = "fidelity_matched_all_baselines_per_candidate.json" if save_per_candidate else "fidelity_matched_all_baselines.json"
    output_path = output_dir / fname
    report = {
        "config": {
            "n_test_hamiltonians": n_test_h,
            "n_candidates": n_candidates,
            "fidelity_thresholds": fidelity_thresholds,
            "baseline_n_steps": baseline_n_steps,
            "trotter_order": trotter_order,
            "baseline_methods": baseline_methods,
            "model_ckpt": model_ckpt,
            "guidance_scale": guidance_scale,
            "disable_gnn_encoder": disable_gnn_encoder,
            "fixed_order": fixed_order,
            "uniform_time": uniform_time,
            "tfim_ratio": tfim_ratio,
            "random_ratio": random_ratio,
            "output_prefix": output_prefix,
            "seed": seed0,
        },
        "summary": summary,
        "ours_elapsed_s": ours_elapsed,
        "baselines_elapsed_s": bl_elapsed,
        **({"ours_results": ours_results} if save_per_candidate else {}),
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
