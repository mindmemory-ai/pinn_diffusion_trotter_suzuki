"""Unit tests for visualization style helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from pinn_trotter.visualization.style import (
    COLORS,
    FIG_DOUBLE,
    FIG_SINGLE,
    FIG_SQUARE,
    FIG_WIDE,
    MARKERS,
    apply_plot_style,
    save_figure,
)


def test_style_dicts_and_presets_exist():
    assert "ours" in COLORS
    assert "qiskit4" in COLORS
    assert "ours" in MARKERS
    assert len(FIG_SINGLE) == 2
    assert len(FIG_DOUBLE) == 2
    assert len(FIG_SQUARE) == 2
    assert len(FIG_WIDE) == 2


def test_apply_plot_style_sets_pdf_fonttype_42():
    apply_plot_style()
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_save_figure_writes_pdf_and_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    pdf_path, png_path = save_figure(fig, "demo_plot", tmp_path)
    assert isinstance(pdf_path, Path)
    assert isinstance(png_path, Path)
    assert pdf_path.exists()
    assert png_path.exists()
