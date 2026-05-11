"""Unit tests for visualization result data loader."""

from __future__ import annotations

import json

import h5py

from pinn_trotter.visualization.data_loader import load_results


def test_load_results_json_for_pareto(tmp_path):
    path = tmp_path / "benchmark_evaluation_results.json"
    path.write_text(json.dumps({"summary": {"ok": True}}), encoding="utf-8")

    payload = load_results(tmp_path, "pareto")
    assert payload["figure_name"] == "pareto"
    assert "benchmark_evaluation_results.json" in payload["sources"]
    assert payload["sources"]["benchmark_evaluation_results.json"]["summary"]["ok"] is True


def test_load_results_alias_and_missing_list(tmp_path):
    path = tmp_path / "ablation_summary.json"
    path.write_text(json.dumps({"results": {}}), encoding="utf-8")

    payload = load_results(tmp_path, "ablation_plots")
    assert payload["figure_name"] == "ablation"
    assert payload["sources"]["ablation_summary.json"]["results"] == {}


def test_load_results_hdf5_dataset(tmp_path):
    path = tmp_path / "dataset_tfim.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=[1, 2, 3])
        f.attrs["version"] = 1

    payload = load_results(tmp_path, "dataset")
    source = payload["sources"]["dataset_tfim.h5"]
    assert source["x"] == [1, 2, 3]
    assert source["@attr:version"] == 1
