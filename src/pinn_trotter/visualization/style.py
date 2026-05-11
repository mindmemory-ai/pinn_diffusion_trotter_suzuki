"""Central plotting style and figure saving utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl

# Color-blind-friendly palette
COLORS: dict[str, str] = {
    "ours": "#0072B2",
    "qiskit4": "#009E73",
    "cirq": "#E69F00",
    "tket": "#CC79A7",
    "pennylane": "#D55E00",
    "baseline": "#6C757D",
    "neutral": "#6C757D",
}

MARKERS: dict[str, str] = {
    "ours": "o",
    "qiskit4": "^",
    "cirq": "s",
    "tket": "D",
    "pennylane": "v",
    "baseline": "X",
}

# Figure size presets (inches)
FIG_SINGLE = (6.0, 4.0)
FIG_DOUBLE = (12.0, 4.5)
FIG_SQUARE = (6.0, 6.0)
FIG_WIDE = (10.0, 4.0)

_LATEX_AVAILABLE = shutil.which("latex") is not None


def apply_plot_style() -> None:
    """Apply project-wide matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": _LATEX_AVAILABLE,
        }
    )


def save_figure(fig, name: str, output_dir: str | Path) -> tuple[Path, Path]:
    """Save a figure to PDF and PNG(300DPI)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(name).stem
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    return pdf_path, png_path


apply_plot_style()

