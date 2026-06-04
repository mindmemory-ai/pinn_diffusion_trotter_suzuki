"""Shared model builder for Phase 4 / benchmark / ablation scripts.

Reads Phase 3 checkpoint metadata and constructs correctly-sized
GNN + diffusion models, then loads the pretrained weights.
"""

from __future__ import annotations

import torch
from pathlib import Path


def build_models_from_checkpoint(
    checkpoint_path: str | Path,
    max_groups: int,
    device: torch.device | None = None,
    gnn_hidden_dim: int = 256,
    gnn_output_dim: int = 512,
    gnn_n_layers: int = 4,
    diffusion_fused_dim: int = 256,
    diffusion_time_embed_dim: int = 256,
    diffusion_grouping_layers: int = 4,
    diffusion_order_layers: int = 2,
    diffusion_ts_mlp_layers: int = 3,
    diffusion_dropout: float = 0.1,
    diffusion_p_cond_drop: float = 0.1,
    T: int = 1000,
    beta_schedule: str = "cosine",
) -> dict:
    """Load a Phase 3 checkpoint and build matching models.

    Reads max_n_qubits and max_M from the checkpoint metadata, constructs
    GNN + diffusion + transition matrices with the correct input dimensions,
    and loads the pretrained weights.

    Args:
        checkpoint_path: Path to Phase 3 diffusion checkpoint (.pt).
        max_groups: K, maximum number of Suzuki-Trotter groups.
        device: Torch device (auto-detected if None).
        gnn_*: GNN architecture hyperparams (must match the saved checkpoint).
        diffusion_*: Diffusion architecture hyperparams (must match the saved checkpoint).
        T: Number of diffusion timesteps.
        beta_schedule: DDPM noise schedule type.

    Returns:
        Dict with keys: gnn, diffusion, tm, order_tm, ddpm,
                        max_n_qubits, max_M, max_groups
    """
    from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
    from pinn_trotter.diffusion.mixed_model import MixedDiffusionModel
    from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
    from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    max_n_q = int(state["max_n_qubits"])
    max_M = int(state["max_M"])
    checkpoint_max_groups = int(state.get("max_groups", max_groups))
    pauli_enc = bool(state.get("pauli_encoding", False))
    if "node_feat_dim" in state:
        node_feat_dim = int(state["node_feat_dim"])
    elif pauli_enc:
        node_feat_dim = 4 + 3 * max_n_q
    else:
        node_feat_dim = max_n_q + 2

    # Auto-detect edge_feat_dim from checkpoint GNN state
    # (Phase 3 used edge_feat_dim=3, Phase 4+ uses 2)
    gnn_state = state["gnn_state"]
    edge_feat_dim = 2  # default
    for k in gnn_state:
        if "msg_mlp.0.weight" in k:
            w = gnn_state[k]
            edge_feat_dim = int(w.shape[1] - 2 * w.shape[0])
            break

    gnn = HamiltonianGNNEncoder(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=gnn_hidden_dim,
        output_dim=gnn_output_dim,
        n_layers=gnn_n_layers,
        pauli_encoding=pauli_enc,
    ).to(device)
    # Load GNN weights with backward-compat for older checkpoints lacking input_norm
    missing = set(gnn.state_dict().keys()) - set(gnn_state.keys())
    if missing <= {"input_norm.weight", "input_norm.bias"}:
        gnn.load_state_dict(gnn_state, strict=False)
    else:
        gnn.load_state_dict(gnn_state)

    diffusion = MixedDiffusionModel(
        max_groups=checkpoint_max_groups,
        n_terms=max_M,
        condition_dim=gnn_output_dim,
        fused_dim=diffusion_fused_dim,
        time_embed_dim=diffusion_time_embed_dim,
        grouping_layers=diffusion_grouping_layers,
        order_layers=diffusion_order_layers,
        ts_mlp_layers=diffusion_ts_mlp_layers,
        dropout=diffusion_dropout,
        p_cond_drop=diffusion_p_cond_drop,
    ).to(device)
    diffusion.load_state_dict(state["diffusion_state"])

    tm = UniformTransitionMatrix(
        K=checkpoint_max_groups,
        T=T,
        beta_schedule=beta_schedule,
    )
    order_tm = UniformTransitionMatrix(K=3, T=T, beta_schedule="cosine")
    ddpm = ContinuousDDPM(T=T, beta_schedule=beta_schedule)

    return {
        "gnn": gnn,
        "diffusion": diffusion,
        "tm": tm,
        "order_tm": order_tm,
        "ddpm": ddpm,
        "max_n_qubits": max_n_q,
        "max_M": max_M,
        "max_groups": checkpoint_max_groups,
        "pauli_encoding": pauli_enc,
        "node_feat_dim": node_feat_dim,
    }
