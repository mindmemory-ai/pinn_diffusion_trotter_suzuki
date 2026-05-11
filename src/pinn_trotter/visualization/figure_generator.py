"""Figure generator: generate all paper figures from result files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pinn_trotter.visualization.data_loader import load_results
from pinn_trotter.visualization.paper_figures import FIGURE_REGISTRY


_FIGURES_NEEDING_BENCHMARK = {"pareto", "comparison", "ablation"}
_FIGURES_NEEDING_MOLECULAR = {"molecular", "scaling"}
_FIGURES_STANDALONE = {"pinn", "training", "grouping", "dataset"}


def _load_data_for_figure(fig_name: str, results_dir: Path) -> dict[str, Any] | None:
    """Load data payload for a figure. Returns None if required files are missing."""
    # Figures that use the data_loader with benchmark results
    if fig_name in _FIGURES_NEEDING_BENCHMARK:
        loader_key = "ablation" if fig_name == "ablation" else "comparison"
        try:
            return load_results(results_dir, loader_key)
        except FileNotFoundError:
            return None

    # Figures that are standalone (load their own files internally)
    if fig_name in _FIGURES_STANDALONE or fig_name in _FIGURES_NEEDING_MOLECULAR:
        # Pass an empty dict; the plot function handles file loading internally
        return {"sources": {}, "figure_name": fig_name}

    return {"sources": {}, "figure_name": fig_name}


def generate_figures(
    figures: list[str],
    results_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, bool]:
    """Generate the requested paper figures.

    Args:
        figures:     List of figure keys (from FIGURE_REGISTRY) or ["all"].
        results_dir: Directory containing benchmark result JSON files.
        output_dir:  Directory to write PDF + PNG outputs.

    Returns:
        Dict mapping figure key → True (success) / False (skipped/failed).
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "all" in figures:
        figures = list(FIGURE_REGISTRY.keys())

    status: dict[str, bool] = {}
    for fig_name in figures:
        if fig_name not in FIGURE_REGISTRY:
            print(f"  [skip] Unknown figure: '{fig_name}'")
            status[fig_name] = False
            continue

        print(f"  [gen]  {fig_name}...", end=" ", flush=True)
        data = _load_data_for_figure(fig_name, results_dir)
        if data is None:
            print("SKIPPED (missing data)")
            status[fig_name] = False
            continue

        try:
            pdf_path, png_path = FIGURE_REGISTRY[fig_name](data, output_dir)
            print(f"OK → {png_path.name}")
            status[fig_name] = True
        except Exception as exc:
            print(f"ERROR: {exc}")
            status[fig_name] = False

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument(
        "--results-dir",
        default="experiments/benchmark_results",
        help="Directory with benchmark JSON result files.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/paper_figures",
        help="Output directory for PDF and PNG figures.",
    )
    parser.add_argument(
        "--figures",
        default="all",
        help=(
            "Comma-separated list of figure keys to generate, or 'all'. "
            f"Available: {', '.join(FIGURE_REGISTRY.keys())}"
        ),
    )
    args = parser.parse_args()

    figures = [f.strip() for f in args.figures.split(",")]

    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Figures     : {figures}")
    print()

    status = generate_figures(
        figures=figures,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    n_ok = sum(v for v in status.values())
    n_total = len(status)
    print(f"\nDone: {n_ok}/{n_total} figures generated successfully.")

    if n_ok < n_total:
        skipped = [k for k, v in status.items() if not v]
        print(f"Skipped/failed: {skipped}")


if __name__ == "__main__":
    main()
