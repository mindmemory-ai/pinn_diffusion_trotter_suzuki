"""Phase 2 PINN pretraining experiment script.

Trains a PINNNetwork to solve the Schrödinger equation for a given Hamiltonian.
Supports warm-start from a previous checkpoint.

Usage:
    python experiments/02_pretrain_pinn.py
    python experiments/02_pretrain_pinn.py experiment=tfim_4q_poc
    python experiments/02_pretrain_pinn.py training.phase2_pinn_pretrain.max_epochs=500
"""

from __future__ import annotations

import logging
import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


def _build_tfim_hamiltonian(
    n_qubits: int,
    J: float,
    h: float,
    boundary: str,
) -> "HamiltonianGraph":
    """Build TFIM Hamiltonian: H = -J Σ ZZ - h Σ X."""
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

    pauli_strings = []
    coefficients = []

    # ZZ interaction terms
    n_bonds = n_qubits if boundary == "periodic" else n_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % n_qubits
        s = ["I"] * n_qubits
        s[i] = "Z"
        s[j] = "Z"
        pauli_strings.append("".join(s))
        coefficients.append(-J)

    # Transverse field X terms
    for i in range(n_qubits):
        s = ["I"] * n_qubits
        s[i] = "X"
        pauli_strings.append("".join(s))
        coefficients.append(-h)

    return HamiltonianGraph(pauli_strings, coefficients, n_qubits)


@hydra.main(config_path="../configs", config_name="experiment/tfim_4q_poc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from pinn_trotter.pinn.network import PINNNetwork
    from pinn_trotter.pinn.trainer import PINNTrainer
    from pinn_trotter.pinn.fidelity import evaluate_fidelity_proxy
    from pinn_trotter.data.generator import (
        apply_trotter_from_hamiltonian,
        compute_exact_fidelity_from_hamiltonian,
    )
    from pinn_trotter.data.sampling import sample_random_strategy

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    exp_cfg = cfg.get("experiment", {})
    train_cfg = cfg.get("training", {})

    n_qubits = int(exp_cfg.get("n_qubits", 4))
    J = float(exp_cfg.get("J", 1.0))
    h_field = float(exp_cfg.get("h", 0.5))
    boundary = str(exp_cfg.get("boundary", "periodic"))
    t_total = float(exp_cfg.get("t_total", 2.0))

    max_steps = int(train_cfg.get("max_epochs", 2000))
    n_colloc = int(train_cfg.get("n_collocation_points", 64))
    lr = float(train_cfg.get("optimizer", {}).get("lr", 5e-4))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 10))
    early_stop_tol = float(train_cfg.get("early_stop_threshold", 1e-5))
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "experiments/pinn_checkpoints"))
    warmstart_steps = int(train_cfg.get("warmstart_steps", 0))

    loss_weights_cfg = train_cfg.get("loss_weights", {})
    loss_weights = {
        "pde": float(loss_weights_cfg.get("pde", 1.0)),
        "ic": float(loss_weights_cfg.get("ic", 10.0)),
        "circuit": float(loss_weights_cfg.get("circuit", 5.0)),
        "norm": 0.1,
    }

    # Warm start/checkpoint roots
    project_root = Path(__file__).parent.parent
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3-E-2 acceptance mode (default enabled): 10 Hamiltonians + 500 validation strategies.
    n_hamiltonians = int(train_cfg.get("n_hamiltonians_eval", 10))
    n_validation_samples = int(train_cfg.get("n_validation_samples", 500))
    rng = np.random.default_rng(int(train_cfg.get("random_seed", 42)))

    # Build 10 parameter combinations around the TFIM range.
    ham_params: list[tuple[float, float]] = []
    for _ in range(n_hamiltonians):
        j_val = float(rng.uniform(0.5, 2.0))
        h_val = float(rng.uniform(0.1, 1.5))
        ham_params.append((j_val, h_val))

    pde_residuals: list[float] = []
    proxy_abs_errors: list[float] = []
    dim = 2**n_qubits
    psi_0_np = np.ones(dim, dtype=complex) / np.sqrt(dim)
    psi_0_t = torch.tensor(psi_0_np, dtype=torch.complex64)

    for idx, (j_val, h_val) in enumerate(ham_params):
        H_graph = _build_tfim_hamiltonian(n_qubits, j_val, h_val, boundary)
        H_dense = H_graph.to_dense_matrix()
        H_tensor = torch.tensor(H_dense, dtype=torch.complex64)
        eigvals = np.linalg.eigvalsh(H_dense.real)
        h_norm = float(np.max(np.abs(eigvals)))

        pinn = PINNNetwork(
            n_qubits=n_qubits,
            fourier_m=256,
            hidden_dim=512,
            hamiltonian_norm=h_norm,
        )

        ckpt_path = checkpoint_dir / f"pinn_tfim_{n_qubits}q_{idx:02d}.pt"
        if ckpt_path.exists() and warmstart_steps > 0:
            log.info("Warm-starting from checkpoint: %s", ckpt_path)
            state = torch.load(ckpt_path, weights_only=True)
            pinn.load_state_dict(state)

        trainer = PINNTrainer(
            pinn=pinn,
            H_matrix=H_tensor,
            psi_0=psi_0_t,
            t_total=t_total,
            n_colloc=n_colloc,
            lr=lr,
            max_steps=max_steps,
            early_stop_patience=early_stop_patience,
            early_stop_tol=early_stop_tol,
            loss_weights=loss_weights,
        )

        log.info(
            "[%d/%d] Training PINN: J=%.3f h=%.3f max_steps=%d",
            idx + 1, n_hamiltonians, j_val, h_val, max_steps,
        )
        t0 = time.time()
        history = trainer.train()
        elapsed = time.time() - t0
        final_pde = float(history["pde"][-1])
        pde_residuals.append(final_pde)
        log.info(
            "[%d/%d] Done in %.1fs: final_pde=%.4e",
            idx + 1, n_hamiltonians, elapsed, final_pde,
        )

        # Validation: compare proxy fidelity to exact fidelity.
        for _ in range(n_validation_samples):
            strategy = sample_random_strategy(
                n_terms=H_graph.n_terms,
                n_groups_max=int(train_cfg.get("n_groups_max", 8)),
                t_total=t_total,
                n_qubits=n_qubits,
                rng=rng,
            )
            psi_trotter_np = apply_trotter_from_hamiltonian(H_graph, strategy, psi_0_np)
            psi_trotter_t = torch.tensor(psi_trotter_np, dtype=torch.complex64)
            f_proxy = evaluate_fidelity_proxy(pinn, psi_trotter_t, t_total)
            f_exact, _ = compute_exact_fidelity_from_hamiltonian(H_graph, strategy, psi_0_np)
            proxy_abs_errors.append(abs(f_proxy - float(f_exact)))

        torch.save(pinn.state_dict(), ckpt_path)
        log.info("Checkpoint saved: %s", ckpt_path)

    mean_pde = float(np.mean(pde_residuals)) if pde_residuals else float("inf")
    mean_proxy_err = float(np.mean(proxy_abs_errors)) if proxy_abs_errors else float("inf")
    report = {
        "n_hamiltonians": n_hamiltonians,
        "n_validation_samples_per_hamiltonian": n_validation_samples,
        "pde_residuals": pde_residuals,
        "mean_pde_residual": mean_pde,
        "mean_proxy_abs_error": mean_proxy_err,
        "acceptance": {
            "pde_lt_1e-4": mean_pde < 1e-4,
            "proxy_error_lt_0.01": mean_proxy_err < 0.01,
        },
    }
    report_path = checkpoint_dir / f"pinn_pretrain_report_{n_qubits}q.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("Report saved: %s", report_path)
    log.info(
        "3-E-2 summary: mean_pde=%.4e, mean_proxy_abs_error=%.4e",
        mean_pde,
        mean_proxy_err,
    )


if __name__ == "__main__":
    main()
