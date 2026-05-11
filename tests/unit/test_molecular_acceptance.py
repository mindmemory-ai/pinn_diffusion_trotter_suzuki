"""Unit tests for molecular acceptance evaluator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "experiments" / "11_molecular_acceptance.py"
    spec = importlib.util.spec_from_file_location("molecular_acceptance", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_dataset_passes_with_better_scores(tmp_path):
    mod = _load_module()
    p = tmp_path / "scan.json"
    p.write_text(
        json.dumps(
            {
                "results": [
                    {"bond_length": 0.7, "ours": {"fidelity": 0.91}, "qiskit_4th": {"fidelity": 0.90}},
                    {"bond_length": 1.2, "ours": {"fidelity": 0.88}, "qiskit_4th": {"fidelity": 0.87}},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = mod._evaluate_dataset(p, "H2")
    assert result["status"] is True
    assert result["failed"] == 0


def test_evaluate_dataset_handles_proxy_scores(tmp_path):
    mod = _load_module()
    p = tmp_path / "scan.json"
    p.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "bond_length": 2.0,
                        "ours": {"proxy_fidelity": 0.81},
                        "qiskit_4th": {"proxy_fidelity": 0.80},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = mod._evaluate_dataset(p, "LiH")
    assert result["status"] is True
    assert result["checked"] == 1
