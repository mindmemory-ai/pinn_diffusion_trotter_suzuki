"""Matplotlib-based plotting utilities for PINN-Trotter diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list[float]],
    save_path: Optional[str | Path] = None,
    title: str = "Closed-loop training history",
) -> "matplotlib.figure.Figure":
    """Plot training history dict produced by ClosedLoopOptimizer.train().

    Expected keys (any subset): 'policy_loss', 'mean_fidelity',
    'mean_depth', 'pareto_hv'.

    Args:
        history:    Dict mapping metric name → list of per-iteration values.
        save_path:  If given, save figure to this path.
        title:      Figure suptitle.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    metrics = [k for k in ["policy_loss", "mean_fidelity", "mean_depth", "pareto_hv"]
               if k in history and history[k]]
    n = len(metrics)
    if n == 0:
        raise ValueError("history contains no recognised metric keys")

    labels = {
        "policy_loss": "Policy loss",
        "mean_fidelity": "Mean fidelity",
        "mean_depth": "Mean circuit depth",
        "pareto_hv": "Pareto hypervolume",
    }

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, metrics):
        vals = history[key]
        ax.plot(vals, linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(labels[key])
        ax.set_title(labels[key])
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def plot_pareto_front(
    pareto_tracker,
    save_path: Optional[str | Path] = None,
    title: str = "Pareto front: fidelity vs circuit depth",
    baseline_depth: Optional[float] = None,
    baseline_fidelity: Optional[float] = None,
) -> "matplotlib.figure.Figure":
    """Scatter-plot the Pareto front from a ParetoTracker.

    Args:
        pareto_tracker:    ParetoTracker instance.
        save_path:         If given, save figure here.
        title:             Axes title.
        baseline_depth:    If set, draw a vertical dashed line (reference depth).
        baseline_fidelity: If set, draw a horizontal dashed line (reference fidelity).

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    front = pareto_tracker.get_front()
    if not front:
        raise ValueError("Pareto front is empty")

    depths = [p["depth"] for p in front]
    fidelities = [p["fidelity"] for p in front]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(depths, fidelities, color="tab:blue", zorder=3, label="Pareto front")

    # Connect front points with staircase line
    sorted_pts = sorted(zip(depths, fidelities))
    xs = [p[0] for p in sorted_pts]
    ys = [p[1] for p in sorted_pts]
    ax.step(xs, ys, where="post", color="tab:blue", alpha=0.5, linewidth=1.2)

    if baseline_depth is not None:
        ax.axvline(baseline_depth, color="tab:red", linestyle="--",
                   linewidth=1.2, label=f"Baseline depth = {baseline_depth}")
    if baseline_fidelity is not None:
        ax.axhline(baseline_fidelity, color="tab:orange", linestyle="--",
                   linewidth=1.2, label=f"Baseline fidelity = {baseline_fidelity:.3f}")

    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def plot_strategy_comparison(
    strategies: list,
    labels: list[str],
    save_path: Optional[str | Path] = None,
    title: str = "Strategy comparison",
) -> "matplotlib.figure.Figure":
    """Visualise a set of TrotterStrategy objects side by side.

    Each strategy is shown as a grid: rows = Pauli terms, columns = time steps,
    coloured by group assignment.  An additional row shows the Trotter order per group.

    Args:
        strategies: List of TrotterStrategy objects.
        labels:     Display label for each strategy (same length as strategies).
        save_path:  If given, save figure here.
        title:      Figure suptitle.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(strategies) != len(labels):
        raise ValueError("strategies and labels must have the same length")

    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")

    for ax, strat, label in zip(axes, strategies, labels):
        M = strat.n_terms
        K = len(strat.grouping)

        # Build (M+1) x K grid: rows 0..M-1 = grouping, row M = order
        grid = np.zeros((M + 1, K), dtype=float)
        for g_idx, group in enumerate(strat.grouping):
            for term_idx in group:
                grid[term_idx, g_idx] = g_idx + 1  # group colour

        # Order row
        for g_idx, order in enumerate(strat.orders):
            grid[M, g_idx] = order

        ax.imshow(grid[:-1, :], aspect="auto", cmap=cmap, vmin=0, vmax=9, origin="upper")

        # Time-step bar
        ts = np.array(strat.time_steps)
        ts_norm = ts / ts.sum()
        for g_idx, w in enumerate(ts_norm):
            ax.add_patch(
                plt.Rectangle((g_idx - 0.5, M - 0.5), 1, w * 2, color="black", alpha=0.3)
            )

        # Annotate Trotter orders
        for g_idx, order in enumerate(strat.orders):
            ax.text(g_idx, M - 0.2, f"k={order}", ha="center", va="bottom",
                    fontsize=7, color="white" if order > 1 else "black")

        ax.set_xlabel("Group index")
        ax.set_ylabel("Pauli term")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(K))
        ax.set_yticks(range(M))

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
