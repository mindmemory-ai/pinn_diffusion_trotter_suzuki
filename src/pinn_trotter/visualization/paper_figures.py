"""Paper figure generation for PINN-Trotter project.

Implements all 9 figures required for the paper (阶段 9-B-1 ~ 9-B-9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pinn_trotter.visualization.style import (
    COLORS,
    FIG_DOUBLE,
    FIG_SINGLE,
    FIG_SQUARE,
    FIG_WIDE,
    MARKERS,
    save_figure,
)


# ---------------------------------------------------------------------------
# 9-B-1: Pareto front (fidelity vs depth)
# ---------------------------------------------------------------------------

def plot_pareto_front(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot Pareto front: fidelity vs circuit depth.

    Data source: benchmark_evaluation_results.json
    Shows: Ours and all framework baselines on Pareto plane.
    """
    bench = data["sources"]["benchmark_evaluation_results.json"]
    summary = bench["summary"]["methods"]
    per_seed = bench.get("per_seed", [])

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    method_order = [
        ("ours", "Ours"),
        ("qiskit_4th", "Qiskit-4th"),
        ("cirq", "Cirq"),
        ("tket", "TKET"),
        ("pennylane", "PennyLane"),
        ("paulihedral", "Paulihedral"),
    ]
    color_key = {"qiskit_4th": "qiskit4"}
    marker_key = {"qiskit_4th": "qiskit4"}

    for method, label in method_order:
        if method not in summary:
            continue
        metric = summary[method]
        x_mean = float(metric["depth"]["mean"])
        y_mean = float(metric["fidelity"]["mean"])
        x_std = float(metric["depth"]["std"])
        y_std = float(metric["fidelity"]["std"])
        c_key = color_key.get(method, method)

        # Per-seed trajectory (if available), used as a lightweight "curve" view.
        seed_points = []
        for seed_payload in per_seed:
            m = seed_payload.get("methods", {}).get(method)
            if m is None:
                continue
            seed_points.append(
                (
                    float(m["depth"]["mean"]),
                    float(m["fidelity"]["mean"]),
                )
            )
        if len(seed_points) >= 2:
            seed_points = sorted(seed_points, key=lambda p: p[0])
            xs = [p[0] for p in seed_points]
            ys = [p[1] for p in seed_points]
            ax.plot(xs, ys, color=COLORS.get(c_key, COLORS["neutral"]), alpha=0.4, linewidth=1.2)

        ax.scatter(
            x_mean,
            y_mean,
            color=COLORS.get(c_key, COLORS["neutral"]),
            marker=MARKERS.get(marker_key.get(method, method), MARKERS["baseline"]),
            s=90,
            label=label,
            zorder=3,
        )
        ax.errorbar(
            x_mean,
            y_mean,
            xerr=x_std,
            yerr=y_std,
            fmt="none",
            color=COLORS.get(c_key, COLORS["neutral"]),
            alpha=0.45,
            capsize=3,
        )

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Fidelity")
    ax.set_title("Pareto Front: Fidelity vs Circuit Depth")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return save_figure(fig, "fig1_pareto_front", output_dir)


# ---------------------------------------------------------------------------
# 9-B-2: PINN accuracy (proxy error vs PDE residual)
# ---------------------------------------------------------------------------

def plot_pinn_accuracy(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot PINN validation: proxy error and PDE residual.

    Data source: pinn_checkpoints_phase3e2/pinn_pretrain_report_4q.json
    Shows: Mean proxy error, PDE residual, and optional external benchmark panel.
    """
    # Load PINN report from alternative path
    import json
    pinn_report_path = Path("experiments/pinn_checkpoints_phase3e2/pinn_pretrain_report_4q.json")
    if not pinn_report_path.exists():
        raise FileNotFoundError(f"PINN report not found: {pinn_report_path}")

    with open(pinn_report_path) as f:
        pinn_data = json.load(f)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    # Left: PDE residual
    pde_residuals = pinn_data["pde_residuals"]
    ax1.bar(range(len(pde_residuals)), pde_residuals, color=COLORS["ours"], alpha=0.7)
    ax1.axhline(1e-4, color="red", linestyle="--", linewidth=1.5, label="Threshold (1e-4)")
    ax1.set_xlabel("Hamiltonian Index")
    ax1.set_ylabel("PDE Residual")
    ax1.set_title("PINN PDE Residual")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Proxy error
    mean_error = pinn_data["mean_proxy_abs_error"]
    ax2.bar([0], [mean_error], color=COLORS["ours"], alpha=0.7, width=0.5)
    ax2.axhline(0.01, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.01)")
    ax2.set_ylabel("Mean Proxy Absolute Error")
    ax2.set_title("PINN Proxy Accuracy")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Mean Error"])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Right: Optional external test panel (prefer latest GPU benchmark report)
    benchmark_candidates = [
        Path("experiments/benchmark_results/benchmark_evaluation_results_paulihedral_gpu.json"),
        Path("experiments/benchmark_results/benchmark_evaluation_results.json"),
    ]
    benchmark_path = next((p for p in benchmark_candidates if p.exists()), None)
    if benchmark_path is not None:
        with open(benchmark_path) as f:
            benchmark = json.load(f)
        summary = benchmark.get("summary", {}).get("methods", {})
        method_label_pairs = [
            ("ours", "Ours"),
            ("qiskit_4th", "Qiskit-4th"),
            ("paulihedral", "Paulihedral"),
        ]
        methods = [m for m, _ in method_label_pairs if m in summary]
        labels = [lab for m, lab in method_label_pairs if m in summary]
        if methods:
            vals = [float(summary[m]["fidelity"]["mean"]) for m in methods]
            stds = [float(summary[m]["fidelity"]["std"]) for m in methods]
            colors = [COLORS.get("qiskit4" if m == "qiskit_4th" else m, COLORS["neutral"]) for m in methods]
            x = np.arange(len(methods))
            ax3.bar(x, vals, yerr=stds, capsize=4, color=colors, alpha=0.8)
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels, rotation=15, ha="right")
            ax3.set_ylim([0.0, 1.05])
            ax3.set_ylabel("Fidelity")
            ax3.set_title("External Test Set (Benchmark)")
            ax3.grid(True, alpha=0.3, axis="y")
        else:
            ax3.text(0.5, 0.5, "No benchmark methods found", ha="center", va="center", color="gray")
            ax3.set_axis_off()
    else:
        ax3.text(0.5, 0.5, "Benchmark result not found", ha="center", va="center", color="gray")
        ax3.set_axis_off()

    fig.suptitle("PINN Evaluator Validation (M2 Milestone + External Test)", y=1.04)
    fig.tight_layout()

    return save_figure(fig, "fig2_pinn_accuracy", output_dir)


# ---------------------------------------------------------------------------
# 9-B-3: Training convergence (closed-loop fidelity over iterations)
# ---------------------------------------------------------------------------

def plot_training_convergence(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot closed-loop training convergence.

    Data source: 6d_poc_results.json (6d1_trend)
    Shows: Fidelity evolution over 50 iterations.
    """
    import json
    poc_path = Path("experiments/benchmark_results/6d_poc_results.json")
    if not poc_path.exists():
        raise FileNotFoundError(f"PoC results not found: {poc_path}")

    with open(poc_path) as f:
        poc_data = json.load(f)

    trend = poc_data["6d1_trend"]
    fidelities = trend["all_fidelities"]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    ax.plot(fidelities, color=COLORS["ours"], linewidth=2, label="Fidelity")
    ax.axhline(trend["min_final_fidelity_threshold"], color="red", linestyle="--",
               linewidth=1.5, label=f"Threshold ({trend['min_final_fidelity_threshold']})")

    # Highlight initial and final windows
    window_size = 10
    ax.axvspan(0, window_size, alpha=0.1, color="green", label="Initial window")
    ax.axvspan(len(fidelities) - window_size, len(fidelities), alpha=0.1, color="blue", label="Final window")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Fidelity")
    ax.set_title("Closed-Loop Training Convergence (6-D-1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return save_figure(fig, "fig3_training_convergence", output_dir)


# ---------------------------------------------------------------------------
# 9-B-4: Method comparison (bar chart: fidelity, depth, latency)
# ---------------------------------------------------------------------------

def plot_method_comparison(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot method comparison: fidelity, depth, latency.

    Data source: benchmark_evaluation_results.json
    Shows: Bar chart comparing Ours vs Qiskit-4th/Cirq/TKET/PennyLane.
    """
    summary = data["sources"]["benchmark_evaluation_results.json"]["summary"]["methods"]

    method_label_pairs = [
        ("ours", "Ours"),
        ("qiskit_4th", "Qiskit-4th"),
        ("cirq", "Cirq"),
        ("tket", "TKET"),
        ("pennylane", "PennyLane"),
        ("paulihedral", "Paulihedral"),
    ]
    methods = [m for m, _ in method_label_pairs if m in summary]
    labels = [lab for m, lab in method_label_pairs if m in summary]

    fidelities = [summary[m]["fidelity"]["mean"] for m in methods]
    depths = [summary[m]["depth"]["mean"] for m in methods]
    latencies = [summary[m]["latency"]["mean"] for m in methods]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    x = np.arange(len(methods))
    width = 0.6

    # Fidelity
    bars1 = ax1.bar(
        x,
        fidelities,
        width,
        color=[COLORS.get("qiskit4" if m == "qiskit_4th" else m, COLORS["neutral"]) for m in methods],
    )
    ax1.set_ylabel("Fidelity")
    ax1.set_title("Fidelity Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis="y")

    # Depth
    bars2 = ax2.bar(
        x,
        depths,
        width,
        color=[COLORS.get("qiskit4" if m == "qiskit_4th" else m, COLORS["neutral"]) for m in methods],
    )
    ax2.set_ylabel("Circuit Depth")
    ax2.set_title("Circuit Depth Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Latency (log scale)
    bars3 = ax3.bar(
        x,
        latencies,
        width,
        color=[COLORS.get("qiskit4" if m == "qiskit_4th" else m, COLORS["neutral"]) for m in methods],
    )
    ax3.set_ylabel("Latency (s)")
    ax3.set_title("Latency Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Method Comparison (500 Test Hamiltonians)", y=1.02)
    fig.tight_layout()

    return save_figure(fig, "fig4_method_comparison", output_dir)


# ---------------------------------------------------------------------------
# 9-B-5: Ablation study (bar chart: fidelity drop per component)
# ---------------------------------------------------------------------------

def plot_ablation_study(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot ablation study: fidelity drop when removing each component.

    Data source: ablation_summary.json
    Shows: Bar chart of fidelity for each ablation profile.
    """
    import json
    ablation_path = Path("experiments/benchmark_results/ablation_summary.json")
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation = json.load(f)["results"]
    else:
        ablation = data["sources"].get("ablation_summary.json", {}).get("results", {})

    profiles = [
        "full_model",
        "no_pinn_guidance",
        "no_cfg",
        "no_structured_matrix",
        "no_gnn_encoder",
        "no_gumbel_estimator",
    ]
    labels = [
        "Full Model",
        "No PINN\nGuidance",
        "No CFG",
        "No Structured\nMatrix",
        "No GNN\nEncoder",
        "No Gumbel\nEstimator",
    ]

    fidelities = [ablation[p]["summary"]["methods"]["ours"]["fidelity"]["mean"] for p in profiles]
    stds = [ablation[p]["summary"]["methods"]["ours"]["fidelity"]["std"] for p in profiles]

    fig, ax = plt.subplots(figsize=FIG_WIDE)

    x = np.arange(len(profiles))
    bars = ax.bar(x, fidelities, yerr=stds, capsize=5, color=COLORS["ours"], alpha=0.7)

    # Highlight full model
    bars[0].set_color(COLORS["baseline"])
    bars[0].set_alpha(1.0)

    ax.set_ylabel("Mean Fidelity")
    ax.set_title("Ablation Study: Component Contribution (100 iterations)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylim([0, max(fidelities) * 1.2])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, fidelities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + stds[i] + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    return save_figure(fig, "fig5_ablation_study", output_dir)


# ---------------------------------------------------------------------------
# 9-B-6: Error scaling (fidelity vs n_qubits, if Heisenberg data available)
# ---------------------------------------------------------------------------

def plot_error_scaling(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot error scaling: fidelity vs n_qubits.

    Data source: heisenberg_scan.json (if available)
    Shows: Fidelity degradation as system size increases.
    """
    # Check if Heisenberg data exists
    import json
    heisenberg_path = Path("experiments/benchmark_results/heisenberg_scan.json")
    if not heisenberg_path.exists():
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        ax.text(0.5, 0.5, "Heisenberg scan data not available\n(Stage 8-3 pending)",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis("off")
        return save_figure(fig, "fig6_error_scaling", output_dir)

    with open(heisenberg_path) as f:
        heisenberg_data = json.load(f)

    results = heisenberg_data["results"]

    # Group by n_qubits
    n_qubits_list = sorted(set(r["n_qubits"] for r in results))
    ours_fid_by_n = {n: [] for n in n_qubits_list}
    base_fid_by_n = {n: [] for n in n_qubits_list}

    for r in results:
        n = r["n_qubits"]
        ours_fid_by_n[n].append(r["ours"]["fidelity"])
        base_fid_by_n[n].append(r["qiskit_4th"]["fidelity"])

    ours_means = [np.mean(ours_fid_by_n[n]) for n in n_qubits_list]
    ours_stds = [np.std(ours_fid_by_n[n]) for n in n_qubits_list]
    base_means = [np.mean(base_fid_by_n[n]) for n in n_qubits_list]
    base_stds = [np.std(base_fid_by_n[n]) for n in n_qubits_list]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    ax.errorbar(n_qubits_list, ours_means, yerr=ours_stds, marker=MARKERS["ours"],
                color=COLORS["ours"], label="Ours", capsize=4, linewidth=2)
    ax.errorbar(n_qubits_list, base_means, yerr=base_stds, marker=MARKERS["qiskit4"],
                color=COLORS["qiskit4"], label="Qiskit-4th", capsize=4, linewidth=2)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Mean Fidelity")
    ax.set_title("Error Scaling: Heisenberg Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return save_figure(fig, "fig6_error_scaling", output_dir)


# ---------------------------------------------------------------------------
# 9-B-7: Grouping heatmap (strategy visualization)
# ---------------------------------------------------------------------------

def plot_grouping_heatmap(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot grouping heatmap: visualize a sample strategy.

    Data source: benchmark_evaluation_results.json (sample_strategies if available)
    Shows: Heatmap of Pauli term grouping.
    """
    # For now, create a synthetic example since sample strategies may not be in JSON
    fig, ax = plt.subplots(figsize=FIG_SQUARE)

    # Synthetic 8-term, 4-group example
    M, K = 8, 4
    grouping = [[0, 3], [1, 4, 6], [2, 5], [7]]
    orders = [2, 4, 2, 1]

    grid = np.zeros((M, K))
    for g_idx, group in enumerate(grouping):
        for term_idx in group:
            grid[term_idx, g_idx] = g_idx + 1

    cmap = plt.get_cmap("tab10")
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=9, origin="upper")

    # Annotate orders
    for g_idx, order in enumerate(orders):
        ax.text(g_idx, -0.5, f"k={order}", ha="center", va="top", fontsize=10, fontweight="bold")

    ax.set_xlabel("Group Index")
    ax.set_ylabel("Pauli Term Index")
    ax.set_title("Strategy Grouping Heatmap (Example)")
    ax.set_xticks(range(K))
    ax.set_yticks(range(M))

    plt.colorbar(im, ax=ax, label="Group ID")

    return save_figure(fig, "fig7_grouping_heatmap", output_dir)


# ---------------------------------------------------------------------------
# 9-B-8: Molecular generalization (H2/LiH bond scan)
# ---------------------------------------------------------------------------

def plot_molecular_generalization(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot molecular generalization: H2 and LiH bond scans.

    Data source: h2_bond_scan.json, lih_bond_scan.json
    Shows: Fidelity vs bond length for both molecules.
    """
    import json

    h2_path = Path("experiments/benchmark_results/h2_bond_scan.json")
    lih_path = Path("experiments/benchmark_results/lih_bond_scan.json")

    if not h2_path.exists() or not lih_path.exists():
        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        ax.text(0.5, 0.5, "Molecular scan data not available\n(Stage 8-1, 8-2 pending)",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis("off")
        return save_figure(fig, "fig8_molecular_generalization", output_dir)

    with open(h2_path) as f:
        h2_data = json.load(f)
    with open(lih_path) as f:
        lih_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # H2
    h2_results = h2_data["results"]
    h2_bonds = [r["bond_length"] for r in h2_results]
    h2_ours_fid = [r["ours"]["fidelity"] for r in h2_results]
    h2_base_fid = [r["qiskit_4th"]["fidelity"] for r in h2_results]

    ax1.plot(h2_bonds, h2_ours_fid, marker=MARKERS["ours"], color=COLORS["ours"], label="Ours", linewidth=2)
    ax1.plot(h2_bonds, h2_base_fid, marker=MARKERS["qiskit4"], color=COLORS["qiskit4"], label="Qiskit-4th", linewidth=2)
    ax1.set_xlabel("Bond Length (Å)")
    ax1.set_ylabel("Fidelity")
    ax1.set_title("H$_2$ Bond Scan (STO-3G)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LiH (proxy fidelity)
    lih_results = lih_data["results"]
    lih_bonds = [r["bond_length"] for r in lih_results]
    lih_ours_proxy = [r["ours"]["proxy_fidelity"] for r in lih_results]
    lih_base_proxy = [r["qiskit_4th"]["proxy_fidelity"] for r in lih_results]

    ax2.plot(lih_bonds, lih_ours_proxy, marker=MARKERS["ours"], color=COLORS["ours"], label="Ours", linewidth=2)
    ax2.plot(lih_bonds, lih_base_proxy, marker=MARKERS["qiskit4"], color=COLORS["qiskit4"], label="Qiskit-4th", linewidth=2)
    ax2.set_xlabel("Bond Length (Å)")
    ax2.set_ylabel("Proxy Fidelity (depth-based)")
    ax2.set_title("LiH Bond Scan (STO-3G)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Molecular Generalization (Stage 8)", y=1.02)
    fig.tight_layout()

    return save_figure(fig, "fig8_molecular_generalization", output_dir)


# ---------------------------------------------------------------------------
# 9-B-9: Dataset statistics (distribution of n_qubits, J, h)
# ---------------------------------------------------------------------------

def plot_dataset_statistics(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot dataset statistics: distribution of n_qubits, J, h.

    Data source: dataset_tfim.h5 metadata or dataset_report.json
    Shows: Histograms of dataset parameters.
    """
    # Load dataset metadata
    import h5py
    dataset_path = Path("data/processed/dataset_tfim.h5")

    if not dataset_path.exists():
        # Placeholder
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        ax.text(0.5, 0.5, "Dataset file not available",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis("off")
        return save_figure(fig, "fig9_dataset_statistics", output_dir)

    with h5py.File(dataset_path, "r") as f:
        n_samples = len(f.keys())
        n_qubits_list = []
        J_list = []
        h_list = []

        for key in list(f.keys())[:min(1000, n_samples)]:  # Sample first 1000
            sample = f[key]
            n_qubits_list.append(sample.attrs.get("n_qubits", 4))
            J_list.append(sample.attrs.get("J", 1.0))
            h_list.append(sample.attrs.get("h", 0.5))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    # n_qubits distribution
    ax1.hist(n_qubits_list, bins=np.arange(3.5, 9.5, 1), color=COLORS["ours"], alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Count")
    ax1.set_title("n_qubits Distribution")
    ax1.grid(True, alpha=0.3, axis="y")

    # J distribution
    ax2.hist(J_list, bins=20, color=COLORS["ours"], alpha=0.7, edgecolor="black")
    ax2.set_xlabel("J (coupling strength)")
    ax2.set_ylabel("Count")
    ax2.set_title("J Distribution")
    ax2.grid(True, alpha=0.3, axis="y")

    # h distribution
    ax3.hist(h_list, bins=20, color=COLORS["ours"], alpha=0.7, edgecolor="black")
    ax3.set_xlabel("h (transverse field)")
    ax3.set_ylabel("Count")
    ax3.set_title("h Distribution")
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Dataset Statistics (n={n_samples} samples)", y=1.02)
    fig.tight_layout()

    return save_figure(fig, "fig9_dataset_statistics", output_dir)


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

FIGURE_REGISTRY = {
    "pareto": plot_pareto_front,
    "pinn": plot_pinn_accuracy,
    "training": plot_training_convergence,
    "comparison": plot_method_comparison,
    "ablation": plot_ablation_study,
    "scaling": plot_error_scaling,
    "grouping": plot_grouping_heatmap,
    "molecular": plot_molecular_generalization,
    "dataset": plot_dataset_statistics,
}
