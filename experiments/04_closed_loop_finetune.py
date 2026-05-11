"""Phase 4 closed-loop fine-tuning experiment script.

Runs REINFORCE-based joint optimization of the diffusion model conditioned
on Hamiltonian GNN embeddings, guided by PINN fidelity proxy.

Usage:
    python experiments/04_closed_loop_finetune.py
    python experiments/04_closed_loop_finetune.py training.phase4_closed_loop.n_iter=500
    python experiments/04_closed_loop_finetune.py +resume_ckpt=path/to/ckpt.pt
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


def _build_components(cfg: DictConfig, n_qubits: int, n_terms: int):
    """Build all model components from config."""
    from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
    from pinn_trotter.diffusion.mixed_model import MixedDiffusionModel
    from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
    from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder

    diff_cfg = cfg.get("model", {})
    gnn_cfg = cfg.get("model", {})

    T = int(diff_cfg.get("T", 1000))
    max_groups = int(cfg.get("training", {}).get("n_groups_max", 8))

    gnn = HamiltonianGNNEncoder(
        node_feat_dim=n_qubits + 2,
        edge_feat_dim=3,
        hidden_dim=int(gnn_cfg.get("hidden_dim", 256)),
        output_dim=int(gnn_cfg.get("output_dim", 512)),
        n_layers=int(gnn_cfg.get("n_layers", 4)),
    )
    diffusion = MixedDiffusionModel(
        max_groups=max_groups,
        n_terms=n_terms,
        condition_dim=int(gnn_cfg.get("output_dim", 512)),
        fused_dim=int(diff_cfg.get("condition_dim", 256)),
        time_embed_dim=int(diff_cfg.get("time_embed_dim", 128)),
        grouping_layers=int(diff_cfg.get("grouping_transformer_layers", 4)),
        order_layers=int(diff_cfg.get("order_transformer_layers", 2)),
        ts_mlp_layers=int(diff_cfg.get("timestep_mlp_layers", 3)),
        p_cond_drop=float(diff_cfg.get("p_cond_drop", 0.1)),
    )
    tm = UniformTransitionMatrix(
        K=max_groups, T=T,
        beta_schedule=str(diff_cfg.get("beta_schedule", "cosine")),
    )
    order_tm = UniformTransitionMatrix(K=3, T=T, beta_schedule="cosine")
    ddpm = ContinuousDDPM(T=T, beta_schedule=str(diff_cfg.get("beta_schedule", "cosine")))

    return gnn, diffusion, tm, order_tm, ddpm, max_groups


@hydra.main(config_path="../configs", config_name="experiment/tfim_4q_poc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from pinn_trotter.benchmarks.hamiltonians import make_tfim
    from pinn_trotter.optimizer.closed_loop import ClosedLoopOptimizer
    from pinn_trotter.pinn.evaluator import make_evaluator

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    exp_cfg = cfg.get("experiment", {})
    loop_cfg = cfg.get("training", {})

    n_qubits = int(exp_cfg.get("n_qubits", 4))
    J = float(exp_cfg.get("J", 1.0))
    h_field = float(exp_cfg.get("h", 0.5))
    boundary = str(exp_cfg.get("boundary", "periodic"))
    t_total = float(exp_cfg.get("t_total", 2.0))
    n_terms = 2 * n_qubits  # TFIM: n ZZ + n X

    n_iter = int(loop_cfg.get("n_iterations", 100))
    batch_size = int(loop_cfg.get("batch_size_hamiltonians", 4))
    lambda_values = loop_cfg.get("lambda_weight_sweep", [0.1])
    if isinstance(lambda_values, (int, float)):
        lambda_values = [lambda_values]
    lambda_values = [float(v) for v in lambda_values]
    guidance_scale = float(loop_cfg.get("guidance_scale", 2.0))
    disable_gnn_encoder = bool(loop_cfg.get("disable_gnn_encoder", False))
    disable_pinn_guidance = bool(loop_cfg.get("disable_pinn_guidance", False))

    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / "experiments" / "closed_loop_checkpoints"

    gnn, diffusion, tm, order_tm, ddpm, max_groups = _build_components(cfg, n_qubits, n_terms)

    # Warm-start from Phase 3 diffusion pretraining checkpoint if provided.
    pretrain_ckpt = cfg.get("pretrain_ckpt", None)
    if pretrain_ckpt is not None:
        pretrain_path = project_root / pretrain_ckpt
        state = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        diffusion.load_state_dict(state["diffusion_state"])
        gnn.load_state_dict(state["gnn_state"])
        log.info("Warm-started GNN + diffusion from Phase 3 checkpoint: %s", pretrain_path)

    # Build fidelity evaluator.
    # For n_qubits <= 8 (PoC): ExactFidelityEvaluator uses exact quantum simulation.
    # For n_qubits > 8: pass pinn_ckpt to use a pre-trained PINN as proxy.
    pinn_ckpt = cfg.get("pinn_ckpt", None)
    if pinn_ckpt is not None:
        pinn_ckpt = project_root / pinn_ckpt
    pinn_evaluator = make_evaluator(
        t_total=t_total,
        n_qubits=n_qubits,
        pinn_ckpt=pinn_ckpt,
        exact_threshold=8,
    )
    log.info("Using evaluator: %s", type(pinn_evaluator).__name__)

    # Hamiltonian sampler: random TFIM variants around base parameters
    rng = np.random.default_rng(42)

    def hamiltonian_sampler():
        result = []
        for _ in range(batch_size):
            J_s = float(rng.uniform(0.5, 2.0))
            h_s = float(rng.uniform(0.1, 1.5))
            result.append(make_tfim(n_qubits, J_s, h_s, boundary))
        return result

    optimizer = ClosedLoopOptimizer(
        diffusion_model=diffusion,
        gnn_encoder=gnn,
        transition_matrix=tm,
        order_transition_matrix=order_tm,
        ddpm=ddpm,
        pinn_evaluator=pinn_evaluator,
        lambda_weight=lambda_values[0],
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        n_terms=n_terms,
        max_groups=max_groups,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=max(n_iter // 5, 1),
        disable_gnn_encoder=disable_gnn_encoder,
        disable_pinn_guidance=disable_pinn_guidance,
    )

    # Resume from checkpoint if provided
    resume_ckpt = cfg.get("resume_ckpt", None)
    start_iter = 0
    if resume_ckpt:
        start_iter = optimizer.load_checkpoint(resume_ckpt) + 1
        log.info("Resuming from iteration %d", start_iter)

    if len(lambda_values) > 1:
        log.info("Running lambda sweep: %s", lambda_values)
        t0 = time.time()
        results = optimizer.lambda_sweep(lambda_values, n_iter, hamiltonian_sampler)
        elapsed = time.time() - t0
        for lam, res in results.items():
            log.info(
                "λ=%.3f → HV=%.4f, best_fid=%.4f",
                lam, res["hypervolume"], max((p["fidelity"] for p in res["pareto_front"]), default=0.0),
            )
        log.info("Lambda sweep complete in %.1fs", elapsed)
    else:
        t0 = time.time()
        history = optimizer.train(n_iter, hamiltonian_sampler, start_iteration=start_iter)
        elapsed = time.time() - t0
        log.info(
            "Training complete: %d iterations in %.1fs | final_fid=%.4f | HV=%.4f",
            n_iter, elapsed,
            history["mean_fidelity"][-1] if history["mean_fidelity"] else 0.0,
            history["pareto_hv"][-1] if history["pareto_hv"] else 0.0,
        )


if __name__ == "__main__":
    main()
