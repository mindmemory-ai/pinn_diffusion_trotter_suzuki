"""Paper figure generation for PINN-Trotter Phase 4 revision.

Implements 7 figures: 5 core + 2 optional.
All data is loaded directly from JSON files in experiments/benchmark_results/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/benchmark_results")

THRESHOLDS = ["0.9", "0.95", "0.99"]
THRESHOLD_FLOATS = [0.90, 0.95, 0.99]

METHOD_LABELS: dict[str, str] = {
    "ours": "P-GONE (Ours)",
    "qiskit_4th": "Qiskit-4th",
    "cirq": "Cirq",
    "tket": "TKET",
    "pennylane": "PennyLane",
    "paulihedral": "Paulihedral",
    "paulihedral_4th": "Paulihedral+4th*",
    "qiskit_group_commuting": "Qiskit GC",
}

METHOD_COLORS: dict[str, str] = {
    "ours": COLORS["ours"],
    "qiskit_4th": COLORS["qiskit4"],
    "cirq": COLORS["cirq"],
    "tket": COLORS["tket"],
    "pennylane": COLORS["pennylane"],
    "paulihedral": COLORS["paulihedral"],
    "paulihedral_4th": "#F0E442",
    "qiskit_group_commuting": "#009E73",
}

BASELINE_METHODS = ["qiskit_4th", "cirq", "tket", "pennylane", "paulihedral"]
FIG1_METHODS = ["ours", "qiskit_group_commuting", "paulihedral", "paulihedral_4th", "pennylane", "qiskit_4th", "cirq", "tket"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(name: str) -> dict[str, Any]:
    """Load a JSON file from the benchmark results directory."""
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _get_method_label(method: str) -> str:
    """Return display label for a method key."""
    return METHOD_LABELS.get(method, method)


def _get_method_color(method: str) -> str:
    """Return color for a method key."""
    return METHOD_COLORS.get(method, COLORS["neutral"])


def _safe_mean_std(metric_dict: dict | None, key: str = "depth") -> tuple[float, float]:
    """Extract (mean, std) from a nested metric dict. Returns (0, 0) on failure."""
    if metric_dict is None:
        return 0.0, 0.0
    inner = metric_dict.get(key, {})
    if isinstance(inner, dict):
        return float(inner.get("mean", 0)), float(inner.get("std", 0))
    return 0.0, 0.0


def _get_ours_depth_for_threshold(summary: dict, threshold: str) -> float:
    """Return ours depth mean at a threshold."""
    t_data = summary.get("thresholds", {}).get(threshold, {})
    ours = t_data.get("ours", {})
    if ours is None:
        return 0.0
    return float(ours.get("depth", {}).get("mean", 0))


# ---------------------------------------------------------------------------
# Figure 1: Fidelity-matched circuit depth (CORE)
# ---------------------------------------------------------------------------

def plot_fidelity_matched_depth(
    data: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Grouped bar chart: circuit depth for ours vs baselines at 3 fidelity thresholds.

    Uses fidelity_matched_all_baselines_20260604.json (unified 8-baseline, 30 Hams × 100 candidates).
    """
    d_all = _load_json("fidelity_matched_all_baselines_20260604.json")

    # All 8 baselines in a single unified experiment — no merge needed
    thresholds_data = d_all["summary"]["thresholds"]

    methods_to_show = FIG1_METHODS

    fig, ax = plt.subplots(figsize=FIG_WIDE)

    x = np.arange(len(THRESHOLDS))
    n_methods = len(methods_to_show)
    total_width = 0.90
    bar_width = total_width / n_methods

    for i, method in enumerate(methods_to_show):
        means = []
        stds = []
        for t_key in THRESHOLDS:
            t_data = thresholds_data.get(t_key, {})
            m_data = t_data.get(method)
            if m_data is None or m_data.get("reachable", 0) == 0:
                means.append(0)
                stds.append(0)
            else:
                depth_mean, depth_std = _safe_mean_std(m_data, "depth")
                means.append(depth_mean)
                stds.append(depth_std)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            color=_get_method_color(method),
            label=_get_method_label(method),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fidelity $\\geq$ {t}" for t in THRESHOLD_FLOATS])
    ax.set_ylabel("Circuit Depth")
    ax.set_title("Circuit Depth at Fidelity Thresholds")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return save_figure(fig, "fig3_fidelity_matched_depth", output_dir)


# ---------------------------------------------------------------------------
# Figure 2: Best-of-N sensitivity (CORE)
# ---------------------------------------------------------------------------

def plot_best_of_n(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Reachability vs number of candidates N (log scale), with fidelity on twin axis.

    Uses n_sensitivity_results.json.
    """
    d = _load_json("n_sensitivity_results.json")
    results = d["results"]
    n_values = sorted(int(k) for k in results.keys())

    n_array = np.array(n_values)

    fig, ax1 = plt.subplots(figsize=FIG_SINGLE)

    threshold_styles = {
        "0.9": {"color": COLORS["ours"], "marker": "o", "linestyle": "-"},
        "0.95": {"color": COLORS["qiskit4"], "marker": "^", "linestyle": "--"},
        "0.99": {"color": COLORS["pennylane"], "marker": "s", "linestyle": "-."},
    }

    for t_key, style in threshold_styles.items():
        reach_means = []
        reach_stds = []
        for n in n_values:
            n_key = str(n)
            t_data = results.get(n_key, {}).get(t_key, {})
            reach_means.append(t_data.get("reachability_mean", 0))
            reach_stds.append(t_data.get("reachability_std", 0))

        reach_means = np.array(reach_means)
        reach_stds = np.array(reach_stds)

        ax1.plot(
            n_array, reach_means,
            color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], linewidth=2,
            label=f"Reachability (fid $\geq$ {THRESHOLD_FLOATS[THRESHOLDS.index(t_key)]})",
        )
        ax1.fill_between(
            n_array,
            np.clip(reach_means - reach_stds, 0, 1),
            np.clip(reach_means + reach_stds, 0, 1),
            color=style["color"], alpha=0.12,
        )

    # Annotate N=32
    n32_reach = results.get("32", {}).get("0.9", {}).get("reachability_mean", 0)
    ax1.annotate(
        f"N=32: {n32_reach:.1%}",
        xy=(32, n32_reach),
        xytext=(32, n32_reach + 0.08),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9,
        ha="center",
    )

    ax1.set_xlabel("Number of Candidates (N)")
    ax1.set_ylabel("Reachability")
    ax1.set_xscale("log")
    ax1.set_xticks(n_values)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Twin axis for best fidelity
    ax2 = ax1.twinx()
    best_fids = []
    best_fid_stds = []
    for n in n_values:
        bf = results.get(str(n), {}).get("_best_fid", {})
        best_fids.append(bf.get("mean", 0))
        best_fid_stds.append(bf.get("std", 0))

    ax2.plot(
        n_array, best_fids,
        color=COLORS["neutral"], marker="D", linestyle=":",
        linewidth=1.5, label="Best Fidelity",
    )
    ax2.set_ylabel("Best Fidelity", color=COLORS["neutral"])
    ax2.tick_params(axis="y", labelcolor=COLORS["neutral"])
    ax2.set_ylim(0, 1.05)

    ax1.set_title("Best-of-N: Reachability vs Candidate Count")

    fig.tight_layout()
    return save_figure(fig, "fig2_best_of_n", output_dir)


# ---------------------------------------------------------------------------
# Figure 3: Component ablation — two panels (CORE)
# ---------------------------------------------------------------------------

def plot_component_ablation(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Two-panel ablation: (left) branch ablation, (right) CFG ablation.

    Left: Full model vs fixed_order vs uniform_time (reachability at thresholds).
    Right: Full model vs cfg_gs1 (reachability at thresholds).

    Note: branch_ablation_both_fixed.json does not exist, so we skip it.
    """
    # Load data
    full = _load_json("fidelity_matched_all_baselines_20260604.json")
    fixed_order = _load_json("branch_ablation_fixed_order.json")
    uniform_time = _load_json("branch_ablation_uniform_time.json")
    cfg_gs1 = _load_json("cfg_ablation_gs1.json")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Left: Branch ablation ---
    ablations_left = [
        ("Full Model", full, "ours"),
        ("Fixed Order", fixed_order, "ours"),
        ("Uniform Time", uniform_time, "ours"),
    ]
    left_colors = [COLORS["ours"], COLORS["qiskit4"], COLORS["pennylane"]]

    x = np.arange(len(THRESHOLDS))
    n_bars = len(ablations_left)
    bar_width = 0.8 / n_bars

    for i, (label, dset, method_key) in enumerate(ablations_left):
        reach = []
        ts = dset["summary"]["thresholds"]
        for t_key in THRESHOLDS:
            m = ts.get(t_key, {}).get(method_key)
            if m is None:
                reach.append(0)
            else:
                n_total = dset["config"].get("n_test_hamiltonians", 1)
                reachable = m.get("reachable", 0)
                reach.append(reachable / n_total if n_total > 0 else 0)

        offset = (i - n_bars / 2 + 0.5) * bar_width
        ax_left.bar(
            x + offset, reach, bar_width,
            color=left_colors[i], label=label, alpha=0.85,
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"Fid $\geq$ {t}" for t in THRESHOLD_FLOATS])
    ax_left.set_ylabel("Reachability")
    ax_left.set_title("Branch Ablation")
    ax_left.set_ylim(0, 1.05)
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3, axis="y")

    # --- Right: CFG ablation ---
    ablations_right = [
        ("Full Model (CFG)", full, "ours"),
        ("No CFG (gs=1.0)", cfg_gs1, "ours"),
    ]
    right_colors = [COLORS["ours"], COLORS["cirq"]]

    bar_width_r = 0.8 / len(ablations_right)

    for i, (label, dset, method_key) in enumerate(ablations_right):
        reach = []
        ts = dset["summary"]["thresholds"]
        for t_key in THRESHOLDS:
            m = ts.get(t_key, {}).get(method_key)
            if m is None:
                reach.append(0)
            else:
                n_total = dset["config"].get("n_test_hamiltonians", 1)
                reachable = m.get("reachable", 0)
                reach.append(reachable / n_total if n_total > 0 else 0)

        offset = (i - len(ablations_right) / 2 + 0.5) * bar_width_r
        ax_right.bar(
            x + offset, reach, bar_width_r,
            color=right_colors[i], label=label, alpha=0.85,
        )

    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"Fid $\geq$ {t}" for t in THRESHOLD_FLOATS])
    ax_right.set_ylabel("Reachability")
    ax_right.set_title("CFG Ablation")
    ax_right.set_ylim(0, 1.05)
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Component Ablation Study", y=1.01)
    fig.tight_layout()
    return save_figure(fig, "fig4_component_ablation", output_dir)


# ---------------------------------------------------------------------------
# Figure 4: Per-type boundary analysis (CORE)
# ---------------------------------------------------------------------------

def plot_per_type_boundary(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Two panels: (left) reachability by Hamiltonian type at 3 thresholds,
    (right) depth reduction ratio vs Paulihedral-4th.

    Uses per_type_tfim.json, per_type_heisenberg.json, per_type_random.json.
    """
    per_type_data = {
        "TFIM": _load_json("per_type_tfim.json"),
        "Heisenberg": _load_json("per_type_heisenberg.json"),
        "Random": _load_json("per_type_random.json"),
    }

    type_colors = {"TFIM": COLORS["ours"], "Heisenberg": COLORS["qiskit4"], "Random": COLORS["pennylane"]}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Left: Reachability by type ---
    x = np.arange(len(THRESHOLDS))
    n_types = len(per_type_data)
    bar_width = 0.8 / n_types

    for i, (htype, dset) in enumerate(per_type_data.items()):
        reach = []
        ts = dset["summary"]["thresholds"]
        for t_key in THRESHOLDS:
            m = ts.get(t_key, {}).get("ours")
            if m is None:
                reach.append(0)
            else:
                n_total = dset["config"].get("n_test_hamiltonians", 1)
                reachable = m.get("reachable", 0)
                reach.append(reachable / n_total if n_total > 0 else 0)

        offset = (i - n_types / 2 + 0.5) * bar_width
        ax_left.bar(
            x + offset, reach, bar_width,
            color=type_colors[htype], label=htype, alpha=0.85,
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"Fid $\geq$ {t}" for t in THRESHOLD_FLOATS])
    ax_left.set_ylabel("Reachability")
    ax_left.set_title("Reachability by Hamiltonian Type")
    ax_left.set_ylim(0, 1.05)
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3, axis="y")

    # --- Right: Depth reduction vs Paulihedral-4th ---
    x2 = np.arange(len(THRESHOLDS))
    bar_width_r = 0.8 / n_types

    for i, (htype, dset) in enumerate(per_type_data.items()):
        ratios = []
        ts = dset["summary"]["thresholds"]
        for t_key in THRESHOLDS:
            ratio = ts.get(t_key, {}).get("depth_reduction_vs_paulihedral_4th", 0)
            if ratio is None:
                ratio = 0
            ratios.append(float(ratio))

        offset = (i - n_types / 2 + 0.5) * bar_width_r
        ax_right.bar(
            x2 + offset, ratios, bar_width_r,
            color=type_colors[htype], label=htype, alpha=0.85,
        )

    ax_right.set_xticks(x2)
    ax_right.set_xticklabels([f"Fid $\geq$ {t}" for t in THRESHOLD_FLOATS])
    ax_right.set_ylabel("Depth Reduction vs Paulihedral-4th")
    ax_right.set_title("Depth Reduction Ratio")
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.3, axis="y")
    # Use log scale for potentially large ratios
    ax_right.set_yscale("log")

    fig.suptitle("Per-Type Boundary Analysis", y=1.01)
    fig.tight_layout()
    return save_figure(fig, "fig6_per_type_boundary", output_dir)


# ---------------------------------------------------------------------------
# Figure 5: Noisy hardware comparison (CORE)
# ---------------------------------------------------------------------------

def plot_noisy_hardware(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Grouped bar: noiseless vs noisy fidelity for ours, paulihedral, qiskit_4th.

    Annotates circuit depth on noisy bars.

    Uses noisy_hardware_results.json.
    """
    d = _load_json("noisy_hardware_results.json")
    methods_data = d["methods"]

    methods_to_show = ["ours", "paulihedral", "qiskit_4th"]
    labels = [_get_method_label(m) for m in methods_to_show]
    colors = [_get_method_color(m) for m in methods_to_show]

    x = np.arange(len(methods_to_show))
    bar_width = 0.35

    noiseless_fids = []
    noiseless_stds = []
    noisy_fids = []
    noisy_stds = []
    depths = []

    for method in methods_to_show:
        m = methods_data.get(method, {})
        fid = m.get("fidelity", {})
        noise_fid = m.get("noise_fidelity", {})
        depth_val = m.get("depth", {}).get("mean", 0)

        noiseless_fids.append(float(fid.get("mean", 0)))
        noiseless_stds.append(float(fid.get("std", 0)))
        noisy_fids.append(float(noise_fid.get("mean", 0)))
        noisy_stds.append(float(noise_fid.get("std", 0)))
        depths.append(float(depth_val))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    bars1 = ax.bar(
        x - bar_width / 2, noiseless_fids, bar_width,
        yerr=noiseless_stds, capsize=4,
        color=[_lighten(c, 0.5) for c in colors],
        label="Noiseless", alpha=0.85,
    )

    bars2 = ax.bar(
        x + bar_width / 2, noisy_fids, bar_width,
        yerr=noisy_stds, capsize=4,
        color=colors,
        label="Noisy (IBM Jakarta)", alpha=0.85,
    )

    # Annotate depth on noisy bars
    for i, (bar, depth) in enumerate(zip(bars2, depths)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(height + noisy_stds[i], height * 1.05) + 0.02,
            f"d={depth:.0f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fidelity")
    ax.set_title("Noisy Hardware: Fidelity Comparison")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return save_figure(fig, "fig7_noisy_hardware", output_dir)


def _lighten(hex_color: str, factor: float = 0.5) -> str:
    """Lighten a hex color by mixing with white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Figure 6: Strategy diversity (OPTIONAL)
# ---------------------------------------------------------------------------

def plot_strategy_diversity(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Three panels: unique ratio, Jaccard distance, order/time diversity.

    Uses strategy_diversity.json (aggregate summary statistics only).
    Since per-hamiltonian distributions are not available, shows summary bars.
    """
    d = _load_json("strategy_diversity.json")
    s = d["summary"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: Unique patterns
    n_unique_mean = s.get("n_unique_mean", 0)
    n_unique_std = s.get("n_unique_std", 0)
    n_total = d["config"].get("n_candidates", 100)
    unique_ratio = n_unique_mean / n_total if n_total > 0 else 0
    unique_ratio_std = n_unique_std / n_total if n_total > 0 else 0

    ax1.bar(
        [0], [unique_ratio * 100],
        yerr=[unique_ratio_std * 100],
        color=COLORS["ours"], alpha=0.7, capsize=5, width=0.4,
    )
    ax1.set_ylabel("Unique Patterns (%)")
    ax1.set_title("Strategy Uniqueness")
    ax1.set_xticks([0])
    ax1.set_xticklabels([f"N={n_total}"])
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Jaccard distance + Order entropy
    metrics = [
        ("Jaccard\nDistance", s.get("jaccard_mean", 0), s.get("jaccard_std", 0)),
        ("Order\nEntropy", s.get("order_entropy_mean", 0), s.get("order_entropy_std", 0)),
        ("Time\nCV", s.get("time_cv_mean", 0), s.get("time_cv_std", 0)),
    ]
    metric_labels = [m[0] for m in metrics]
    metric_means = [m[1] for m in metrics]
    metric_stds = [m[2] for m in metrics]
    metric_colors = [COLORS["ours"], COLORS["qiskit4"], COLORS["pennylane"]]

    x2 = np.arange(len(metrics))
    ax2.bar(x2, metric_means, yerr=metric_stds, capsize=5, color=metric_colors, alpha=0.7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylabel("Value")
    ax2.set_title("Diversity Metrics")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Text summary
    ax3.axis("off")
    summary_lines = [
        f"Jaccard distance: {s.get('jaccard_mean', 0):.3f} +/- {s.get('jaccard_std', 0):.3f}",
        f"Unique patterns: {n_unique_mean:.1f} / {n_total} ({unique_ratio*100:.1f}%)",
        f"Order entropy: {s.get('order_entropy_mean', 0):.3f} +/- {s.get('order_entropy_std', 0):.3f}",
        f"Time CV: {s.get('time_cv_mean', 0):.3f} +/- {s.get('time_cv_std', 0):.3f}",
        f"",
        f"Hamiltonians: {d['config'].get('n_hamiltonians', '?')}",
        f"Guidance scale: {d['config'].get('guidance_scale', '?')}",
    ]
    for i, line in enumerate(summary_lines):
        ax3.text(0.1, 0.9 - i * 0.08, line, fontsize=10, fontfamily="monospace",
                 transform=ax3.transAxes, va="top")
    ax3.set_title("Summary Statistics")

    fig.suptitle("Strategy Diversity Analysis", y=1.02)
    fig.tight_layout()
    return save_figure(fig, "fig5_strategy_diversity", output_dir)


# ---------------------------------------------------------------------------
# Figure 7: REINFORCE training progress (OPTIONAL)
# ---------------------------------------------------------------------------

def plot_reinforce_training(data: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot REINFORCE training effect using Pareto front and Phase 3 vs Phase 4 comparison.

    Uses closed_loop_checkpoints/pareto_summary.json (Pareto front of fidelity vs depth)
    and phase3_vs_phase4.json (comparison before/after REINFORCE).
    """
    # Load Pareto front from closed-loop checkpoint directory
    pareto_path = Path("experiments/closed_loop_checkpoints/pareto_summary.json")
    if not pareto_path.exists():
        raise FileNotFoundError(f"Pareto summary not found: {pareto_path}")

    with open(pareto_path) as f:
        pareto = json.load(f)

    # Load Phase 3 vs Phase 4 comparison
    p3v4 = _load_json("phase3_vs_phase4.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # --- Left: Pareto front ---
    front = pareto.get("front", [])
    if front:
        fidelities = [p["fidelity"] for p in front]
        depths = [p["depth"] for p in front]

        ax1.plot(
            depths, fidelities,
            color=COLORS["ours"], marker="o", linewidth=2, markersize=8,
            label="REINFORCE Pareto Front",
        )
        # Annotate best point
        best_idx = np.argmax(fidelities)
        ax1.annotate(
            f"Best: fid={fidelities[best_idx]:.4f}, d={depths[best_idx]}",
            xy=(depths[best_idx], fidelities[best_idx]),
            xytext=(depths[best_idx] + 3, fidelities[best_idx] - 0.05),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=8,
        )

    ax1.set_xlabel("Circuit Depth")
    ax1.set_ylabel("Fidelity")
    ax1.set_title("REINFORCE Pareto Front")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Right: Phase 3 vs Phase 4 ---
    ts = p3v4["summary"]["thresholds"]
    methods_to_show = [
        ("ours", "Phase 4 (REINFORCE)"),
    ]
    phase3_key = "ours_phase3"  # Check if this key exists
    if "ours_phase3" not in ts.get("0.9", {}):
        # Try alternative keys
        for possible_key in ["phase3", "ours_pretrain"]:
            if possible_key in ts.get("0.9", {}):
                phase3_key = possible_key
                break

    x3 = np.arange(len(THRESHOLDS))
    bar_width = 0.3

    for i, (key, label) in enumerate(methods_to_show):
        depths_list = []
        for t_key in THRESHOLDS:
            m = ts.get(t_key, {}).get(key)
            if m and m.get("reachable", 0) > 0:
                depths_list.append(float(m.get("depth", {}).get("mean", 0)))
            else:
                depths_list.append(0)
        ax2.bar(x3 + (i - 0.5) * bar_width, depths_list, bar_width,
                color=COLORS["ours"], label=label, alpha=0.85)

    ax2.set_xticks(x3)
    ax2.set_xticklabels([f"Fid $\geq$ {t}" for t in THRESHOLD_FLOATS])
    ax2.set_ylabel("Circuit Depth")
    ax2.set_title("Phase 4: Depth at Thresholds")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("REINFORCE Training Progress", y=1.01)
    fig.tight_layout()
    return save_figure(fig, "fig1_reinforce_training", output_dir)


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

FIGURE_REGISTRY: dict[str, Any] = {
    "fidelity_depth": plot_fidelity_matched_depth,
    "best_of_n": plot_best_of_n,
    "component_ablation": plot_component_ablation,
    "per_type_boundary": plot_per_type_boundary,
    "noisy_hardware": plot_noisy_hardware,
    "strategy_diversity": plot_strategy_diversity,
    "reinforce_training": plot_reinforce_training,
}
