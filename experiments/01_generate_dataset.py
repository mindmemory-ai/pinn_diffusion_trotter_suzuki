"""Phase 1 dataset generation script.

Generates a Trotter strategy dataset for TFIM Hamiltonians.
Supports breakpoint resume: re-running this script will skip already-generated samples.

Usage:
    python experiments/01_generate_dataset.py
    python experiments/01_generate_dataset.py training=phase1_dataset
    python experiments/01_generate_dataset.py training.n_samples=1000 training.n_workers=4
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure src is on path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="experiment/tfim_4q_poc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from pinn_trotter.data.generator import generate_dataset, generate_dataset_report

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    training_cfg = cfg.get("training", {})
    n_samples = int(training_cfg.get("n_samples", 10000))
    n_workers = int(training_cfg.get("n_workers", 4))
    output_path = Path(training_cfg.get("output_path", "data/processed/dataset_tfim.h5"))
    random_seed = int(training_cfg.get("random_seed", 42))
    n_groups_max = int(training_cfg.get("n_groups_max", 8))

    # Make path relative to project root (parent of experiments/)
    project_root = Path(__file__).parent.parent
    if not output_path.is_absolute():
        output_path = project_root / output_path

    config_dict = {
        "n_groups_max": n_groups_max,
        "random_seed": random_seed,
        "J_range": list(training_cfg.get("J_range", [0.1, 5.0])),
        "h_range": list(training_cfg.get("h_range", [0.1, 5.0])),
        "t_final_range": list(training_cfg.get("t_final_range", [0.5, 10.0])),
        "n_qubits": training_cfg.get("n_qubits", None),
    }

    log.info(
        "Generating dataset: n_samples=%d, output=%s, n_workers=%d",
        n_samples, output_path, n_workers,
    )

    t0 = time.time()
    written = generate_dataset(n_samples, output_path, n_workers, config_dict)
    elapsed = time.time() - t0

    log.info("Written %d new samples in %.1fs → %s", written, elapsed, output_path)

    if written > 0 or output_path.exists():
        log.info("Generating dataset report...")
        report = generate_dataset_report(output_path)
        log.info(
            "Report: n_samples=%d, fidelity mean=%.4f±%.4f, depth mean=%.1f",
            report["n_samples"],
            report["fidelity"]["mean"],
            report["fidelity"]["std"],
            report["circuit_depth"]["mean"],
        )


if __name__ == "__main__":
    main()
