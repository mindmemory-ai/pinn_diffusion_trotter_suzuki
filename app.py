#!/usr/bin/env python3
"""Interactive Trotter Strategy Inference & Comparison Tool.

Given Pauli strings and coefficients, uses a trained GNN + diffusion model to
generate a Trotter-Suzuki decomposition strategy, evaluates its fidelity and
circuit depth, and optionally compares against Qiskit baselines.

Architecture is auto-detected from the checkpoint — no need to specify
GNN/diffusion hyperparameters manually.

Usage:
    python app.py --ckpt experiments/closed_loop_checkpoints/diffusion_best.pt \\
                  --pauli "XXII,IZII,IIZI,IIIZ" --coeffs "1.0,1.0,-0.5,-0.5" \\
                  --n-qubits 4 --compare qiskit group_commuting --n-samples 8

    # Fast inference with DDIM acceleration
    python app.py --ckpt ckpt.pt --pauli "ZI,IZ,XX" --coeffs "1.0,0.5,0.3" \\
                  --n-qubits 2 --n-steps 50 --n-samples 8

    # With fidelity threshold highlighting
    python app.py --ckpt ckpt.pt --pauli "XXII,IZII,IIZI,IIIZ" \\
                  --coeffs "1.0,1.0,-0.5,-0.5" --n-qubits 4 \\
                  --fidelity-threshold 0.95 --n-samples 16
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(_WORKSPACE / "src"))

from pinn_trotter.utils.model_builder import build_models_from_checkpoint
from pinn_trotter.diffusion.mixed_model import guided_sample
from pinn_trotter.strategy.encoding import tensor_to_strategy
from pinn_trotter.data.generator import compute_exact_fidelity_from_hamiltonian
from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

ORDER_MAP = {0: 1, 1: 2, 2: 4}

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trotter Strategy Inference & Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference (architecture auto-detected)
  python app.py --ckpt ckpt.pt --pauli "XXII,IZII,IIZI,IIIZ" \\
                --coeffs "1.0,1.0,-0.5,-0.5" --n-qubits 4

  # With group_commuting comparison + 8 samples
  python app.py --ckpt ckpt.pt --pauli "ZI,IZ,XX" --coeffs "1.0,0.5,0.3" \\
                --n-qubits 2 --compare qiskit group_commuting --n-samples 8

  # Fast mode (DDIM 50 steps) + fidelity threshold
  python app.py --ckpt ckpt.pt --pauli "XXII,IZII,IIZI,IIIZ" \\
                --coeffs "1.0,1.0,-0.5,-0.5" --n-qubits 4 \\
                --n-steps 50 --fidelity-threshold 0.95 --n-samples 16
        """,
    )
    # Model
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to Phase 3/4 checkpoint (.pt) — architecture auto-detected")
    p.add_argument("--max-groups", type=int, default=None,
                   help="Max Suzuki-Trotter groups [auto-detect from checkpoint]")
    p.add_argument("--guidance-scale", type=float, default=3.0,
                   help="CFG guidance scale [3.0]")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Reverse diffusion steps (DDIM) [50]")

    # Architecture overrides (all optional — auto-detected from checkpoint)
    p.add_argument("--gnn-hidden-dim", type=int, default=None)
    p.add_argument("--gnn-output-dim", type=int, default=None)
    p.add_argument("--gnn-n-layers", type=int, default=None)
    p.add_argument("--diff-fused-dim", type=int, default=None)
    p.add_argument("--diff-time-embed-dim", type=int, default=None)
    p.add_argument("--diff-grouping-layers", type=int, default=None)
    p.add_argument("--diff-order-layers", type=int, default=None)
    p.add_argument("--diff-ts-mlp-layers", type=int, default=None)

    # Hamiltonian
    p.add_argument("--pauli", type=str, required=True,
                   help="Comma-separated Pauli strings, e.g. 'XXII,IZII'")
    p.add_argument("--coeffs", type=str, required=True,
                   help="Comma-separated coefficients, e.g. '1.0,-0.5'")
    p.add_argument("--n-qubits", type=int, required=True,
                   help="Number of qubits")
    p.add_argument("--t-total", type=float, default=1.0,
                   help="Total evolution time [1.0]")

    # Solver
    p.add_argument("--solver", type=str, default="exact",
                   help="'exact' (default) or path to PINN checkpoint (.pt)")

    # Comparison
    p.add_argument("--compare", type=str, nargs="*", default=[],
                   choices=["qiskit", "qiskit_opt", "group_commuting"],
                   help="Baselines: qiskit, qiskit_opt, group_commuting")

    # Sampling
    p.add_argument("--n-samples", type=int, default=1,
                   help="Number of diffusion samples (Best-of-N) [1]")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed [42]")
    p.add_argument("--fidelity-threshold", type=float, default=None,
                   help="Highlight samples meeting this fidelity [none]")

    # Output
    p.add_argument("--quiet", action="store_true",
                   help="Suppress strategy details, show only summary")

    return p.parse_args()


# ── Architecture auto-detection ──────────────────────────────────────────────

def _detect_architecture(checkpoint_path: str) -> dict:
    """Auto-detect GNN and diffusion architecture from checkpoint state dict.

    Returns dict with gnn_hidden_dim, gnn_output_dim, gnn_n_layers,
    diffusion_fused_dim, diffusion_time_embed_dim, diffusion_grouping_layers,
    diffusion_order_layers, diffusion_ts_mlp_layers.
    """
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
    diffusion_ts_mlp_layers = max(1, ts_linear - 1)

    return {
        "gnn_hidden_dim": gnn_hidden_dim,
        "gnn_output_dim": gnn_output_dim,
        "gnn_n_layers": gnn_n_layers,
        "diffusion_fused_dim": fused_dim,
        "diffusion_time_embed_dim": time_embed_dim,
        "diffusion_grouping_layers": diffusion_grouping_layers,
        "diffusion_order_layers": diffusion_order_layers,
        "diffusion_ts_mlp_layers": diffusion_ts_mlp_layers,
        "max_groups": max_groups_from_ckpt,
    }


# ── Inference runner ─────────────────────────────────────────────────────────

class InferenceRunner:
    """Loads models once, runs repeated inference on a single Hamiltonian."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Auto-detect architecture
        print(f"Loading checkpoint: {args.ckpt}")
        t0 = time.time()
        arch = _detect_architecture(args.ckpt)
        print(f"  Auto-detected: GNN({arch['gnn_hidden_dim']}/{arch['gnn_output_dim']}/"
              f"{arch['gnn_n_layers']})  "
              f"Diff(fused={arch['diffusion_fused_dim']},"
              f"time_emb={arch['diffusion_time_embed_dim']},"
              f"g={arch['diffusion_grouping_layers']},"
              f"o={arch['diffusion_order_layers']},"
              f"t={arch['diffusion_ts_mlp_layers']})  "
              f"K={arch['max_groups']}")

        # Use explicit args if provided, otherwise auto-detected
        gnn_hidden_dim = args.gnn_hidden_dim or arch["gnn_hidden_dim"]
        gnn_output_dim = args.gnn_output_dim or arch["gnn_output_dim"]
        gnn_n_layers = args.gnn_n_layers or arch["gnn_n_layers"]
        diff_fused_dim = args.diff_fused_dim or arch["diffusion_fused_dim"]
        diff_time_embed_dim = args.diff_time_embed_dim or arch["diffusion_time_embed_dim"]
        diff_grouping_layers = args.diff_grouping_layers or arch["diffusion_grouping_layers"]
        diff_order_layers = args.diff_order_layers or arch["diffusion_order_layers"]
        diff_ts_mlp_layers = args.diff_ts_mlp_layers or arch["diffusion_ts_mlp_layers"]
        max_groups = args.max_groups or arch["max_groups"]

        ckpt = build_models_from_checkpoint(
            args.ckpt,
            max_groups=max_groups,
            device=self.device,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_n_layers=gnn_n_layers,
            diffusion_fused_dim=diff_fused_dim,
            diffusion_time_embed_dim=diff_time_embed_dim,
            diffusion_grouping_layers=diff_grouping_layers,
            diffusion_order_layers=diff_order_layers,
            diffusion_ts_mlp_layers=diff_ts_mlp_layers,
        )
        self.gnn = ckpt["gnn"]
        self.diffusion = ckpt["diffusion"]
        self.tm = ckpt["tm"]
        self.order_tm = ckpt["order_tm"]
        self.ddpm = ckpt["ddpm"]
        self.max_M = ckpt["max_M"]
        self.max_groups = ckpt["max_groups"]
        self.max_n_q = ckpt["max_n_qubits"]
        self.pauli_encoding = ckpt.get("pauli_encoding", False)
        print(f"  Loaded in {time.time() - t0:.1f}s  "
              f"(max_n_qubits={ckpt['max_n_qubits']}, max_M={self.max_M}, "
              f"K={self.max_groups}, pauli_enc={self.pauli_encoding})")

        # PINN evaluator (optional)
        self.pinn = None
        if args.solver != "exact":
            pinn_path = Path(args.solver)
            if pinn_path.exists():
                from pinn_trotter.pinn.network import PINNNetwork
                state = torch.load(pinn_path, map_location=self.device, weights_only=True)
                self.pinn = PINNNetwork(n_qubits=args.n_qubits).to(self.device)
                self.pinn.load_state_dict(state)
                self.pinn.eval()
                print(f"  PINN loaded: {args.solver}")

        # Build Hamiltonian
        paulis = [s.strip() for s in args.pauli.split(",")]
        coeffs = [float(c.strip()) for c in args.coeffs.split(",")]
        self.hamiltonian = HamiltonianGraph(paulis, coeffs, args.n_qubits)
        print(f"Hamiltonian: {args.n_qubits} qubits, {self.hamiltonian.n_terms} terms")
        for i, (s, c) in enumerate(zip(paulis, coeffs)):
            print(f"  [{i}] {s}  coeff={c:.4f}")

        if self.hamiltonian.n_terms > self.max_M:
            sys.exit(f"ERROR: n_terms={self.hamiltonian.n_terms} exceeds "
                     f"checkpoint max_M={self.max_M}")

        # Initial state
        self.psi_0 = np.zeros(2 ** args.n_qubits, dtype=complex)
        self.psi_0[0] = 1.0

        # GNN encode — must pad to checkpoint's max_n_qubits
        data = self.hamiltonian.to_pyg_data(max_n_qubits=self.max_n_q,
                                            pauli_encoding=self.pauli_encoding)
        self.condition = self.gnn.forward(
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.edge_attr.to(self.device),
        )

    def sample_one(self, seed: int) -> dict:
        """Run one guided_sample → decode → evaluate cycle."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        grouping, ts, orders = guided_sample(
            model=self.diffusion,
            condition=self.condition,
            n_terms=self.hamiltonian.n_terms,
            max_groups=self.max_groups,
            transition_matrix=self.tm,
            order_transition_matrix=self.order_tm,
            ddpm=self.ddpm,
            guidance_scale=self.args.guidance_scale,
            n_steps=self.args.n_steps,
            device=self.device,
        )

        strategy = tensor_to_strategy(
            grouping_labels=grouping[0],
            orders_onehot=F.one_hot(orders[0], num_classes=3).float(),
            time_steps=ts[0],
            n_qubits=self.args.n_qubits,
            t_total=self.args.t_total,
        )

        # Evaluate fidelity
        if self.pinn is not None:
            from pinn_trotter.pinn.evaluator import PINNEvaluator
            evaluator = PINNEvaluator(self.pinn, self.args.t_total, psi_0=self.psi_0,
                                       device=self.device, fallback_exact=True)
            g_eval = grouping[0:1, :self.hamiltonian.n_terms]
            o_eval = F.one_hot(orders[0:1], num_classes=3).float()
            fidelity = evaluator(self.hamiltonian, g_eval, ts[0:1], o_eval)
        else:
            fidelity, _ = compute_exact_fidelity_from_hamiltonian(
                self.hamiltonian, strategy, self.psi_0,
            )
        fidelity = float(np.clip(fidelity, 0.0, 1.0))

        depth = strategy.circuit_depth_estimate()

        return {
            "strategy": strategy,
            "fidelity": fidelity,
            "depth": depth,
            "grouping": grouping[0, :self.hamiltonian.n_terms].cpu().tolist(),
            "orders": orders[0].cpu().tolist(),
            "time_steps": ts[0].cpu().tolist(),
        }

    def run(self) -> dict:
        """Run n_samples inferences and return aggregated results."""
        results = []
        for i in range(self.args.n_samples):
            seed = self.args.seed + i
            t0 = time.time()
            r = self.sample_one(seed)
            r["time_s"] = time.time() - t0
            results.append(r)

        fids = [r["fidelity"] for r in results]
        depths = [r["depth"] for r in results]

        # Best: highest fidelity; tie-break by depth
        best_idx = max(range(len(results)), key=lambda i: (
            results[i]["fidelity"], -results[i]["depth"]
        ))

        # Count samples meeting threshold
        meet_threshold = None
        if self.args.fidelity_threshold is not None:
            meet_threshold = sum(
                1 for r in results if r["fidelity"] >= self.args.fidelity_threshold
            )

        return {
            "results": results,
            "best_idx": best_idx,
            "fid_best": max(fids),
            "fid_mean": statistics.fmean(fids),
            "fid_std": statistics.stdev(fids) if len(fids) > 1 else 0.0,
            "depth_best": results[best_idx]["depth"],
            "depth_mean": statistics.fmean(depths),
            "depth_std": statistics.stdev(depths) if len(depths) > 1 else 0.0,
            "meet_threshold": meet_threshold,
        }


# ── Baselines ────────────────────────────────────────────────────────────────

def run_qiskit_baseline(hamiltonian: HamiltonianGraph, t_total: float,
                         psi_0: np.ndarray, optimization_level: int = 0) -> dict:
    """Compute Qiskit SuzukiTrotter(order=4) baseline (no grouping)."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.synthesis.evolution import SuzukiTrotter

    n = hamiltonian.n_qubits

    pauli_le = [s[::-1] for s in hamiltonian.pauli_strings]
    op = SparsePauliOp(pauli_le, hamiltonian.coefficients.tolist())

    qc = QuantumCircuit(n)
    gate = PauliEvolutionGate(op, time=t_total,
                               synthesis=SuzukiTrotter(order=4, reps=1))
    qc.append(gate, range(n))

    qc_decomp = transpile(qc, basis_gates=["cx", "rx", "ry", "rz", "h", "x", "y", "z"],
                           optimization_level=0)
    if optimization_level > 0:
        qc_decomp = transpile(qc_decomp,
                              basis_gates=["cx", "rx", "ry", "rz", "h", "x", "y", "z"],
                              optimization_level=optimization_level)

    from pinn_trotter.strategy.circuit_builder import _swap_endian
    psi_0_le = _swap_endian(psi_0.astype(complex), n)
    sv = Statevector(psi_0_le).evolve(qc_decomp)
    psi_trotter_be = _swap_endian(sv.data, n)

    from scipy.linalg import expm
    H_mat = hamiltonian.to_sparse_matrix().toarray()
    psi_exact = expm(-1j * t_total * H_mat) @ psi_0
    fidelity = float(np.abs(np.vdot(psi_exact, psi_trotter_be)) ** 2)
    fidelity = float(np.clip(fidelity / (np.linalg.norm(psi_exact)**2 *
                                          np.linalg.norm(psi_trotter_be)**2), 0.0, 1.0))

    depth = qc_decomp.depth()
    cx_count = qc_decomp.count_ops().get("cx", 0)
    total_gates = sum(qc_decomp.count_ops().values())

    return {
        "label": f"Qiskit 4th (opt={optimization_level})",
        "fidelity": fidelity,
        "depth": depth,
        "cx_count": cx_count,
        "total_gates": total_gates,
    }


def run_group_commuting_baseline(hamiltonian: HamiltonianGraph, t_total: float,
                                  psi_0: np.ndarray) -> dict:
    """Compute Qiskit SparsePauliOp.group_commuting() baseline.

    Groups commuting terms via Qiskit's built-in heuristic, applies each
    group's exponential sequentially (1st-order), and measures fidelity.
    This is the teacher strategy our model was trained to improve upon.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.synthesis.evolution import SuzukiTrotter

    n = hamiltonian.n_qubits

    pauli_le = [s[::-1] for s in hamiltonian.pauli_strings]
    op = SparsePauliOp(pauli_le, hamiltonian.coefficients.tolist())

    # Group commuting terms
    groups = op.group_commuting()
    n_groups = len(groups)

    qc = QuantumCircuit(n)
    tau = t_total  # Single Trotter step: all groups applied once
    for grp in groups:
        if len(grp) == 0:
            continue
        gate = PauliEvolutionGate(grp, time=tau, synthesis=SuzukiTrotter(order=1, reps=1))
        qc.append(gate, range(n))

    qc_decomp = transpile(qc, basis_gates=["cx", "rx", "ry", "rz", "h", "x", "y", "z"],
                           optimization_level=0)

    from pinn_trotter.strategy.circuit_builder import _swap_endian
    psi_0_le = _swap_endian(psi_0.astype(complex), n)
    sv = Statevector(psi_0_le).evolve(qc_decomp)
    psi_trotter_be = _swap_endian(sv.data, n)

    from scipy.linalg import expm
    H_mat = hamiltonian.to_sparse_matrix().toarray()
    psi_exact = expm(-1j * t_total * H_mat) @ psi_0
    fidelity = float(np.abs(np.vdot(psi_exact, psi_trotter_be)) ** 2)
    fidelity = float(np.clip(fidelity / (np.linalg.norm(psi_exact)**2 *
                                          np.linalg.norm(psi_trotter_be)**2), 0.0, 1.0))

    depth = qc_decomp.depth()
    cx_count = qc_decomp.count_ops().get("cx", 0)
    total_gates = sum(qc_decomp.count_ops().values())

    return {
        "label": f"Qiskit GC ({n_groups} groups)",
        "fidelity": fidelity,
        "depth": depth,
        "cx_count": cx_count,
        "total_gates": total_gates,
    }


# ── Rendering ────────────────────────────────────────────────────────────────

def render(results: dict, runner: InferenceRunner, baselines: list[dict],
           args: argparse.Namespace) -> None:
    """Print formatted results."""
    H = runner.hamiltonian
    best = results["results"][results["best_idx"]]
    strategy = best["strategy"]

    # ── Strategy details ──
    if not args.quiet:
        print(f"\n{'='*70}")
        print("STRATEGY DETAILS (best sample)")
        print(f"{'='*70}")
        effective_groups = [g for g in strategy.grouping if len(g) > 0]
        for k, group_indices in enumerate(effective_groups):
            paulis_in_group = [H.pauli_strings[idx] for idx in group_indices]
            order = strategy.orders[k]
            tau = strategy.time_steps[k]
            print(f"  Group {k}: order={ORDER_MAP.get(order, order)}, "
                  f"τ={tau:.4f}, size={len(group_indices)} → {paulis_in_group}")
        print(f"  Total groups: {len(effective_groups)} (non-empty)")
        print(f"  Total depth:  {best['depth']}")
        print(f"  Total time:   {sum(strategy.time_steps):.6f}")

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    n_qubits = args.n_qubits
    pauli_short = ", ".join(H.pauli_strings[:3])
    if len(H.pauli_strings) > 3:
        pauli_short += f", ... ({H.n_terms} total)"
    print(f"  Hamiltonian: {pauli_short}")
    print(f"  Qubits: {n_qubits}, t_total: {args.t_total}, "
          f"samples: {args.n_samples}, steps: {args.n_steps}, GS: {args.guidance_scale}")
    print()

    header = f"  {'Method':<30} {'Fidelity':>10} {'Depth':>8} {'CX':>8} {'Gates':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Our results
    our_label = f"Ours (best/{args.n_samples})" if args.n_samples > 1 else "Ours"
    our_cx = _estimate_cx(strategy, H)
    print(f"  {our_label:<30} {best['fidelity']:>10.4f} "
          f"{best['depth']:>8} {our_cx:>8} {'-':>8}")

    if args.n_samples > 1:
        print(f"  {'  (mean ± std)':<30} "
              f"{results['fid_mean']:>10.4f} {results['depth_mean']:>8.0f} "
              f"{'-':>8} {'-':>8}")

    # Fidelity threshold
    if results["meet_threshold"] is not None:
        meet = results["meet_threshold"]
        total = args.n_samples
        rate = meet / total * 100
        print(f"  {'  ≥' + str(args.fidelity_threshold):<30} "
              f"{f'{meet}/{total} ({rate:.0f}%)':>10} {'-':>8} {'-':>8} {'-':>8}")

    # Baselines
    for bl in baselines:
        bl_cx = bl.get("cx_count", "-")
        bl_gates = bl.get("total_gates", "-")
        print(f"  {bl['label']:<30} {bl['fidelity']:>10.4f} "
              f"{bl['depth']:>8} {bl_cx:>8} {bl_gates:>8}")

    # Depth comparison
    if baselines:
        print()
        for bl in baselines:
            if bl["depth"] > 0 and best["depth"] > 0:
                ratio = bl["depth"] / best["depth"]
                print(f"  Depth vs {bl['label']}: {bl['depth']}/{best['depth']} "
                      f"= {ratio:.1f}× reduction")

    print()

    # ── Per-sample stats ──
    if args.n_samples > 1 and not args.quiet:
        print("  Per-sample:")
        for i, r in enumerate(results["results"]):
            marker = " ← best" if i == results["best_idx"] else ""
            threshold_mark = ""
            if args.fidelity_threshold is not None:
                threshold_mark = " ✓" if r["fidelity"] >= args.fidelity_threshold else ""
            print(f"    [{i}] fid={r['fidelity']:.4f}  depth={r['depth']:3d}  "
                  f"time={r['time_s']:.2f}s{marker}{threshold_mark}")


def _estimate_cx(strategy, H: HamiltonianGraph) -> int:
    """Estimate CX count from strategy structure (upper bound)."""
    total = 0
    for group_indices, order in zip(strategy.grouping, strategy.orders):
        m = len(group_indices)
        n_two_qubit = sum(1 for idx in group_indices
                          if H.pauli_strings[idx].count('I') < H.n_qubits - 1)
        if order == 1:
            reps = 1
        elif order == 2:
            reps = 2
        else:
            reps = 5
        total += n_two_qubit * 2 * reps
    return total


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Validate
    paulis = [s.strip() for s in args.pauli.split(",")]
    coeffs = [float(c.strip()) for c in args.coeffs.split(",")]
    if len(paulis) != len(coeffs):
        sys.exit(f"ERROR: {len(paulis)} Pauli strings vs {len(coeffs)} coefficients")
    for s in paulis:
        if len(s) != args.n_qubits:
            sys.exit(f"ERROR: Pauli string '{s}' length {len(s)} != n_qubits {args.n_qubits}")
        if not all(c in "IXYZ" for c in s):
            sys.exit(f"ERROR: Invalid characters in '{s}' (allowed: I, X, Y, Z)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run inference
    runner = InferenceRunner(args)
    t0 = time.time()
    results = runner.run()
    elapsed = time.time() - t0
    print(f"\nInference done: {args.n_samples} samples in {elapsed:.1f}s "
          f"({elapsed/args.n_samples:.1f}s/sample)")

    # Baselines
    baselines = []
    for method in args.compare:
        print(f"Running baseline: {method} ...")
        if method == "qiskit":
            bl = run_qiskit_baseline(runner.hamiltonian, args.t_total,
                                      runner.psi_0, optimization_level=0)
            baselines.append(bl)
        elif method == "qiskit_opt":
            bl = run_qiskit_baseline(runner.hamiltonian, args.t_total,
                                      runner.psi_0, optimization_level=3)
            baselines.append(bl)
        elif method == "group_commuting":
            bl = run_group_commuting_baseline(runner.hamiltonian, args.t_total,
                                               runner.psi_0)
            baselines.append(bl)

    render(results, runner, baselines, args)


if __name__ == "__main__":
    main()
