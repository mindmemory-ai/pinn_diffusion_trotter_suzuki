"""Result loading helpers for figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Figure key -> candidate result files (searched in order).
_FIGURE_SOURCE_MAP: dict[str, list[str]] = {
    "pareto": ["benchmark_evaluation_results.json"],
    "pinn": ["pinn_pretrain_results.json", "pinn_metrics.json"],
    "training": ["closed_loop_training_log.json", "benchmark_evaluation_results.json"],
    "comparison": ["benchmark_evaluation_results.json"],
    "scaling": ["heisenberg_scan.json"],
    "graph": ["graph_examples.json"],
    "ablation": ["ablation_summary.json"],
    "generalization": ["h2_bond_scan.json", "lih_bond_scan.json"],
    "dataset": ["dataset_report.json", "dataset_tfim.h5"],
}

_ALIASES: dict[str, str] = {
    "pareto_plots": "pareto",
    "pinn_plots": "pinn",
    "training_plots": "training",
    "comparison_plots": "comparison",
    "scaling_plots": "scaling",
    "graph_plots": "graph",
    "ablation_plots": "ablation",
    "generalization_plots": "generalization",
    "dataset_plots": "dataset",
}


def _read_hdf5_group(group: h5py.Group) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        if isinstance(value, np.ndarray):
            out[f"@attr:{key}"] = value.tolist()
        else:
            out[f"@attr:{key}"] = value.item() if hasattr(value, "item") else value

    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if isinstance(data, np.ndarray):
                out[key] = data.tolist()
            else:
                out[key] = data.item() if hasattr(data, "item") else data
        elif isinstance(item, h5py.Group):
            out[key] = _read_hdf5_group(item)
    return out


def _resolve_figure_key(figure_name: str) -> str:
    key = figure_name.strip().lower()
    key = key.replace(".py", "")
    if key in _FIGURE_SOURCE_MAP:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    if key.endswith("_plots") and key[:-6] in _FIGURE_SOURCE_MAP:
        return key[:-6]
    raise ValueError(f"Unsupported figure_name: {figure_name}")


def load_results(results_dir: str | Path, figure_name: str) -> dict[str, Any]:
    """Load result payloads needed by a given figure family."""
    base = Path(results_dir)
    fig_key = _resolve_figure_key(figure_name)
    candidates = _FIGURE_SOURCE_MAP[fig_key]

    loaded: dict[str, Any] = {}
    missing: list[str] = []
    for rel in candidates:
        path = base / rel
        if not path.exists():
            missing.append(rel)
            continue
        if path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                loaded[rel] = json.load(f)
        elif path.suffix.lower() in (".h5", ".hdf5"):
            with h5py.File(path, "r") as f:
                loaded[rel] = _read_hdf5_group(f)
        else:
            raise ValueError(f"Unsupported results file type: {path}")

    if not loaded:
        expected = ", ".join(candidates)
        raise FileNotFoundError(f"No result files found for '{fig_key}' in {base}: {expected}")

    return {"figure_name": fig_key, "results_dir": str(base), "sources": loaded, "missing": missing}

