"""Unit tests for ablation runner helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ablation_runner_dry_run_executes():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "07_ablation_runner.py"
    cmd = [
        sys.executable,
        str(script),
        "--dry-run",
        "--profiles",
        "full_model",
        "no_cfg",
        "--train-iters",
        "1",
        "--n-test-hamiltonians",
        "1",
        "--n-seeds",
        "1",
        "--latency-trials",
        "1",
        "--device",
        "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Running ablation: full_model" in proc.stdout
    assert "Running ablation: no_cfg" in proc.stdout
    assert "Saved ablation summary" in proc.stdout
