"""Phase 3 GNN + diffusion joint supervised pre-training.

Trains the MixedDiffusionModel (conditioned on HamiltonianGNNEncoder embeddings)
on the labelled strategy dataset generated in Phase 1.  Uses DDPM/D3PM losses for
the continuous time-step branch and the discrete grouping/order branches.

Usage:
    python experiments/03_pretrain_diffusion.py
    python experiments/03_pretrain_diffusion.py training.phase3_joint_train.max_epochs=200
    python experiments/03_pretrain_diffusion.py +resume_ckpt=path/to/ckpt.pt
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def _collate(batch):
    """Collate a list of (graph_data, strategy_tensor, fidelity, depth) items.

    graph_data items are returned as a list so the encoder can handle them
    individually (graphs may have variable node/edge counts).
    """
    graph_list, strat_list, fids, depths = zip(*batch)
    groupings = torch.stack([s[0] for s in strat_list])   # (B, M)
    orders_oh = torch.stack([s[1] for s in strat_list])   # (B, K, 3)
    ts_norm = torch.stack([s[2] for s in strat_list])     # (B, K)
    # ham_params: (B, 5) = [t_total, log_t_total, J, h, J*h]
    # Older dataset items may not have this 4th element; fall back to zeros.
    if len(strat_list[0]) >= 4:
        ham_params = torch.stack([s[3] for s in strat_list])  # (B, 5)
    else:
        ham_params = torch.zeros(len(strat_list), 5)
    fidelity = torch.stack(fids)                           # (B,)
    return list(graph_list), groupings, orders_oh, ts_norm, ham_params, fidelity


# ---------------------------------------------------------------------------
# Hamiltonian encoding helper
# ---------------------------------------------------------------------------

def _encode_single(gd, gnn, device):
    """Encode one graph_data item (PyG Data or plain dict) → (1, output_dim)."""
    if hasattr(gd, "x"):
        return gnn(gd.x.to(device), gd.edge_index.to(device), gd.edge_attr.to(device))

    from pinn_trotter.hamiltonian.pauli_utils import locality

    n = int(gd["n_qubits"])
    pauli_strings = gd["pauli_strings"]
    coeffs = gd["coefficients"]
    if hasattr(coeffs, "numpy"):
        coeffs = coeffs.numpy()
    node_feats = torch.zeros(len(pauli_strings), n + 2, device=device)
    for i, (s, c_val) in enumerate(zip(pauli_strings, coeffs)):
        node_feats[i, 0] = float(c_val)
        node_feats[i, 1] = float(locality(s))
        for q, ch in enumerate(s):
            node_feats[i, 2 + q] = 0.0 if ch == "I" else 1.0
    ei = torch.zeros(2, 0, dtype=torch.long, device=device)
    ea = torch.zeros(0, 3, device=device)
    return gnn(node_feats, ei, ea)


def _encode_batch(graph_list, gnn, device):
    """Encode a list of graph_data → (B, output_dim) tensor.

    Uses PyG batching for a single forward pass when all items are PyG Data objects,
    otherwise falls back to individual encoding.
    Returns (embedding, batched_graph_or_None).
    """
    if hasattr(graph_list[0], "x"):
        from torch_geometric.data import Batch
        batched = Batch.from_data_list(graph_list)
        emb = gnn(
            batched.x.to(device),
            batched.edge_index.to(device),
            batched.edge_attr.to(device),
            batched.batch.to(device),
        )
        return emb, batched
    emb = torch.cat([_encode_single(gd, gnn, device) for gd in graph_list], dim=0)
    return emb, None


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R^2 without extra dependencies."""
    y_true = y_true.detach().float()
    y_pred = y_pred.detach().float()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if float(ss_tot) < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _strategy_features(
    groupings: torch.Tensor,
    orders_oh: torch.Tensor,
    ts_norm: torch.Tensor,
    ham_params: torch.Tensor,
    max_groups: int,
) -> torch.Tensor:
    """Build full strategy + Hamiltonian features for supervised GNN pretraining."""
    # groupings one-hot flatten: (B, M, max_groups) -> (B, M*max_groups)
    g_flat = F.one_hot(groupings.long(), num_classes=max_groups).float().reshape(groupings.shape[0], -1)
    # orders one-hot flatten: (B, K, 3) -> (B, K*3)
    o_flat = orders_oh.float().reshape(orders_oh.shape[0], -1)
    # normalized timesteps: (B, K)
    t_feat = ts_norm.float()
    # ham_params: (B, 5) = [t_total, log_t_total, J, h, J*h]  ← key physics features
    h_feat = ham_params.float()
    return torch.cat([g_flat, o_flat, t_feat, h_feat], dim=1)


def _trotter_proxy(
    graph_list: list,
    groupings: torch.Tensor,
    ts_norm: torch.Tensor,
    ham_params: torch.Tensor,
    device: torch.device,
    batched_graph=None,
) -> torch.Tensor:
    """Compute inter-group commutator sum as Trotter error proxy.

    For each sample:
        proxy = Σ_{(i,j) non-commuting, group(i)≠group(j)} comm_norm(i,j) × ts_actual[g_i] × ts_actual[g_j]

    where ts_actual = ts_norm * t_total (absolute timesteps in physical time units).
    Returns shape (B, 1).
    """
    B = groupings.shape[0]
    # t_total is ham_params[:, 0]
    t_total = ham_params[:, 0].float().cpu()  # (B,) on CPU for scatter ops

    if batched_graph is not None and hasattr(batched_graph, "ptr"):
        edge_index = batched_graph.edge_index  # (2, E_total) CPU
        edge_attr = batched_graph.edge_attr    # (E_total, 3) CPU
        batch_ptr = batched_graph.ptr          # (B+1,) CPU
        batch_assign = batched_graph.batch     # (N_total,) CPU

        if edge_index.shape[1] == 0:
            return torch.zeros(B, 1, device=device)

        src, dst = edge_index[0], edge_index[1]  # CPU long tensors
        comm_norms = edge_attr[:, 0].float()       # (E_total,)

        graph_idx = batch_assign[src]           # (E_total,)
        node_off = batch_ptr[graph_idx]         # (E_total,) offset per edge

        local_src = (src - node_off).long()
        local_dst = (dst - node_off).long()

        g_src = groupings[graph_idx, local_src]   # (E_total,) group of src node
        g_dst = groupings[graph_idx, local_dst]   # (E_total,) group of dst node
        inter_mask = (g_src != g_dst)

        if inter_mask.sum() == 0:
            return torch.zeros(B, 1, device=device)

        gi_inter = graph_idx[inter_mask]
        # Use absolute timesteps: ts_actual = ts_norm * t_total
        t_total_per_edge = t_total[gi_inter]  # (E_inter,)
        ts_src = ts_norm[gi_inter, g_src[inter_mask]].float() * t_total_per_edge
        ts_dst = ts_norm[gi_inter, g_dst[inter_mask]].float() * t_total_per_edge
        proxy_vals = comm_norms[inter_mask] * ts_src * ts_dst

        proxies = torch.zeros(B, dtype=torch.float32)
        proxies.scatter_add_(0, gi_inter, proxy_vals)
        return proxies.unsqueeze(1).to(device)

    # Fallback: Python loop over individual graphs
    proxies = []
    for i in range(B):
        g_i = groupings[i]
        ts_i = ts_norm[i]
        t_i = float(t_total[i])
        graph = graph_list[i]
        edge_index = graph.edge_index
        comm_norms = graph.edge_attr[:, 0]
        if edge_index.shape[1] == 0:
            proxies.append(torch.tensor(0.0))
            continue
        src, dst = edge_index[0], edge_index[1]
        g_src = g_i[src]
        g_dst = g_i[dst]
        inter_mask = (g_src != g_dst)
        if inter_mask.sum() == 0:
            proxies.append(torch.tensor(0.0))
            continue
        # Use absolute timesteps
        ts_src = ts_i[g_src[inter_mask]].float() * t_i
        ts_dst = ts_i[g_dst[inter_mask]].float() * t_i
        proxy = (comm_norms[inter_mask].float() * ts_src * ts_dst).sum()
        proxies.append(proxy)
    return torch.stack(proxies).unsqueeze(1).to(device)


# ---------------------------------------------------------------------------
# One forward pass producing (total_loss, loss_dict)
# ---------------------------------------------------------------------------

def _step(graph_list, groupings, orders_oh, ts_norm, gnn, diffusion, tm, order_tm, ddpm,
          T, device, lambda_ts, lambda_ord):
    B = groupings.shape[0]
    condition, _ = _encode_batch(graph_list, gnn, device)   # (B, output_dim)

    t_diff = torch.randint(0, T, (B,), device=device)

    grouping_noisy = tm.forward_sample(groupings, t_diff)          # (B, M)
    ts_noisy, ts_noise = ddpm.forward_sample(ts_norm, t_diff)      # (B, K), (B, K)
    orders_int = orders_oh.argmax(dim=-1)                          # (B, K)
    orders_noisy_int = order_tm.forward_sample(orders_int, t_diff) # (B, K)
    orders_noisy_oh = F.one_hot(orders_noisy_int, num_classes=3).float()

    g_logits, ts_noise_pred, o_logits = diffusion(
        grouping_noisy, ts_noisy, orders_noisy_oh, t_diff, condition
    )
    return diffusion.compute_loss(
        g_logits, ts_noise_pred, o_logits,
        groupings, grouping_noisy, t_diff,
        ts_noise, orders_int, orders_noisy_int,
        tm, order_tm, ddpm,
        lambda_ts=lambda_ts, lambda_ord=lambda_ord,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="experiment/tfim_4q_poc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from pinn_trotter.data.dataset import TrotterDataset
    from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
    from pinn_trotter.diffusion.mixed_model import EMAWrapper, MixedDiffusionModel
    from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
    from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder
    from pinn_trotter.gnn.head import FidelityRegressionHead

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    exp_cfg = cfg.get("experiment", {})
    train_cfg = cfg.get("training", {})
    diff_cfg = cfg.get("model", {})
    gnn_cfg = cfg.get("model", {})

    n_qubits = int(exp_cfg.get("n_qubits", 4))
    project_root = Path(__file__).parent.parent

    _ds_override = train_cfg.get("dataset_path", None)
    dataset_path = Path(_ds_override) if _ds_override else project_root / "data" / "processed" / "dataset_tfim.h5"
    if not dataset_path.exists():
        log.error("Dataset not found at %s — run 01_generate_dataset.py first.", dataset_path)
        return

    max_groups = int(cfg.get("training", {}).get("n_groups_max", 8))
    batch_size = int(train_cfg.get("batch_size", 128))
    max_epochs = int(train_cfg.get("max_epochs", 500))
    augment = bool(train_cfg.get("augment", True))
    lr = float(train_cfg.get("optimizer", {}).get("lr", 1e-4))
    weight_decay = float(train_cfg.get("optimizer", {}).get("weight_decay", 1e-2))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    train_split = float(train_cfg.get("train_split", 0.9))
    log_interval = int(train_cfg.get("log_interval", 50))
    save_interval = int(train_cfg.get("save_interval", 10))
    lambda_ts = float(train_cfg.get("loss_weights", {}).get("timesteps", 0.5))
    lambda_ord = float(train_cfg.get("loss_weights", {}).get("orders", 0.3))
    gnn_pretrain_epochs = int(train_cfg.get("gnn_pretrain_epochs", 30))
    gnn_pretrain_lr = float(train_cfg.get("gnn_pretrain_lr", 3e-4))
    gnn_pretrain_log_interval = int(train_cfg.get("gnn_pretrain_log_interval", 10))
    gnn_pretrain_use_strategy_features = bool(train_cfg.get("gnn_pretrain_use_strategy_features", False))
    h_max_filter_raw = train_cfg.get("h_max_filter", None)
    h_max_filter = float(h_max_filter_raw) if h_max_filter_raw is not None else None
    T = int(diff_cfg.get("T", 1000))

    checkpoint_dir = project_root / train_cfg.get(
        "checkpoint_dir", "experiments/diffusion_checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device_cfg = str(train_cfg.get("device", "auto"))
    if device_cfg == "auto":
        # Use CUDA if available; verify with a small kernel to catch unsupported arch
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda") + 1
                device = torch.device("cuda")
            except Exception:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_cfg)
    log.info("Device: %s", device)

    # Dataset / DataLoaders
    dataset = TrotterDataset(
        dataset_path=dataset_path,
        max_groups=max_groups,
        augment=augment,
        n_qubits_filter=n_qubits,
        h_max_filter=h_max_filter,
    )
    n_train = max(1, int(len(dataset) * train_split))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    log.info("Dataset: %d train / %d val", n_train, n_val)

    # drop_last=True only when we have enough samples to form at least 2 full batches;
    # otherwise with small datasets (e.g. 127 samples, batch=128) we'd get 0 batches.
    drop_last = n_train >= 2 * batch_size
    _n_workers = min(4, int(train_cfg.get("n_workers", 4)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, drop_last=drop_last,
                              num_workers=_n_workers, persistent_workers=_n_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate,
                            num_workers=_n_workers, persistent_workers=_n_workers > 0)

    n_terms = 2 * n_qubits  # TFIM: n ZZ + n X

    # Build models
    gnn = HamiltonianGNNEncoder(
        node_feat_dim=n_qubits + 2,
        edge_feat_dim=3,
        hidden_dim=int(gnn_cfg.get("hidden_dim", 256)),
        output_dim=int(gnn_cfg.get("output_dim", 512)),
        n_layers=int(gnn_cfg.get("n_layers", 4)),
    ).to(device)

    # ---- Stage 1: GNN supervised pre-training (4-B-2/4-B-3) ----
    gnn_out_dim = int(gnn_cfg.get("output_dim", 512))
    # strategy_feat_dim: groupings one-hot + orders one-hot + ts_norm + ham_params
    # ham_params = [t_total, log_t_total, J, h, J*h] → 5 extra features
    strategy_feat_dim = n_terms * max_groups + max_groups * 3 + max_groups + 5
    # +1 for absolute Trotter error proxy scalar
    pretrain_head_in_dim = gnn_out_dim + (strategy_feat_dim + 1 if gnn_pretrain_use_strategy_features else 0)
    regression_head = FidelityRegressionHead(input_dim=pretrain_head_in_dim).to(device)
    gnn_optimizer = torch.optim.AdamW(
        list(gnn.parameters()) + list(regression_head.parameters()),
        lr=gnn_pretrain_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    gnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        gnn_optimizer,
        T_max=max(gnn_pretrain_epochs, 1),
        eta_min=1e-6,
    )
    gnn_best_r2 = -float("inf")
    gnn_best_rmse = float("inf")
    gnn_best_epoch = -1

    # Two-phase training: freeze GNN for first half to stabilize head learning on
    # strategy/physics features, then unfreeze for joint fine-tuning.
    gnn_freeze_epochs = gnn_pretrain_epochs // 2

    if gnn_pretrain_epochs > 0:
        log.info(
            "Stage-1 GNN pretrain: epochs=%d lr=%.2e (GNN frozen for first %d epochs)",
            gnn_pretrain_epochs, gnn_pretrain_lr, gnn_freeze_epochs,
        )
    for epoch in range(max(gnn_pretrain_epochs, 0)):
        t0 = time.time()

        # Phase transition: freeze → unfreeze GNN at midpoint
        if epoch == 0:
            for p in gnn.parameters():
                p.requires_grad_(False)
            log.info("GNN frozen (phase-1)")
        elif epoch == gnn_freeze_epochs:
            for p in gnn.parameters():
                p.requires_grad_(True)
            # Re-initialise optimiser so stale Adam states don't bias GNN updates
            gnn_optimizer = torch.optim.AdamW(
                list(gnn.parameters()) + list(regression_head.parameters()),
                lr=gnn_pretrain_lr * 0.3,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
            gnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                gnn_optimizer,
                T_max=max(gnn_pretrain_epochs - gnn_freeze_epochs, 1),
                eta_min=1e-6,
            )
            log.info("GNN unfrozen (phase-2), lr reset to %.2e", gnn_pretrain_lr * 0.3)

        gnn.train()
        regression_head.train()
        train_losses: list[float] = []
        for graph_list, groupings, orders_oh, ts_norm, ham_params, fid in train_loader:
            fid = fid.to(device)
            cond, batched_graph = _encode_batch(graph_list, gnn, device)
            if gnn_pretrain_use_strategy_features:
                s_feat = _strategy_features(
                    groupings=groupings.to(device),
                    orders_oh=orders_oh.to(device),
                    ts_norm=ts_norm.to(device),
                    ham_params=ham_params.to(device),
                    max_groups=max_groups,
                )
                t_proxy = _trotter_proxy(graph_list, groupings, ts_norm, ham_params, device, batched_graph)
                cond = torch.cat([cond, s_feat, t_proxy], dim=1)
            pred = regression_head(cond).squeeze(-1)
            loss = F.mse_loss(pred, fid)
            gnn_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(gnn.parameters()) + list(regression_head.parameters()), max_norm=1.0)
            gnn_optimizer.step()
            train_losses.append(loss.detach().item())

        gnn.eval()
        regression_head.eval()
        val_losses: list[float] = []
        val_true: list[torch.Tensor] = []
        val_pred: list[torch.Tensor] = []
        with torch.no_grad():
            for graph_list, groupings, orders_oh, ts_norm, ham_params, fid in val_loader:
                fid = fid.to(device)
                cond, batched_graph = _encode_batch(graph_list, gnn, device)
                if gnn_pretrain_use_strategy_features:
                    s_feat = _strategy_features(
                        groupings=groupings.to(device),
                        orders_oh=orders_oh.to(device),
                        ts_norm=ts_norm.to(device),
                        ham_params=ham_params.to(device),
                        max_groups=max_groups,
                    )
                    t_proxy = _trotter_proxy(graph_list, groupings, ts_norm, ham_params, device, batched_graph)
                    cond = torch.cat([cond, s_feat, t_proxy], dim=1)
                pred = regression_head(cond).squeeze(-1)
                loss = F.mse_loss(pred, fid)
                val_losses.append(float(loss))
                val_true.append(fid.detach().cpu())
                val_pred.append(pred.detach().cpu())

        y_true = torch.cat(val_true, dim=0) if val_true else torch.zeros(1)
        y_pred = torch.cat(val_pred, dim=0) if val_pred else torch.zeros(1)
        val_r2 = _r2_score(y_true, y_pred)
        val_rmse = float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))
        if val_r2 > gnn_best_r2:
            gnn_best_r2 = val_r2
            gnn_best_rmse = val_rmse
            gnn_best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "gnn_state": gnn.state_dict(),
                    "head_state": regression_head.state_dict(),
                    "val_r2": val_r2,
                    "val_rmse": val_rmse,
                },
                checkpoint_dir / "gnn_pretrain_best.pt",
            )
        if epoch % gnn_pretrain_log_interval == 0 or epoch == gnn_pretrain_epochs - 1:
            mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
            mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
            log.info(
                "gnn-epoch=%d train_mse=%.4e val_mse=%.4e val_r2=%.4f val_rmse=%.4e lr=%.2e %.1fs",
                epoch, mean_train, mean_val, val_r2, val_rmse,
                gnn_optimizer.param_groups[0]["lr"], time.time() - t0,
            )
        gnn_scheduler.step()

    gnn_report = {
        "gnn_pretrain_epochs": gnn_pretrain_epochs,
        "gnn_pretrain_use_strategy_features": gnn_pretrain_use_strategy_features,
        "best_val_r2": gnn_best_r2,
        "best_val_rmse": gnn_best_rmse,
        "best_epoch": gnn_best_epoch,
        "acceptance_r2_gt_0_8": gnn_best_r2 > 0.8,
    }
    report_path = checkpoint_dir / "gnn_pretrain_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(gnn_report, f, ensure_ascii=False, indent=2)
    log.info("GNN pretrain report saved: %s", report_path)
    log.info("GNN pretrain best val R2=%.4f (epoch=%d)", gnn_best_r2, gnn_best_epoch)

    # Continue with stage-2 diffusion training
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
    ).to(device)

    tm = UniformTransitionMatrix(K=max_groups, T=T,
                                 beta_schedule=str(diff_cfg.get("beta_schedule", "cosine"))).to(device)
    order_tm = UniformTransitionMatrix(K=3, T=T, beta_schedule="cosine").to(device)
    ddpm = ContinuousDDPM(T=T, beta_schedule=str(diff_cfg.get("beta_schedule", "cosine"))).to(device)
    ema = EMAWrapper(diffusion, decay=ema_decay)

    optimizer = torch.optim.AdamW(
        list(diffusion.parameters()) + list(gnn.parameters()),
        lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay,
    )

    # Optional resume
    start_epoch = 0
    resume_ckpt = cfg.get("resume_ckpt", None)
    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location=device, weights_only=False)
        diffusion.load_state_dict(state["diffusion_state"])
        gnn.load_state_dict(state["gnn_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        log.info("Resumed from epoch %d", start_epoch)

    step_kwargs = dict(tm=tm, order_tm=order_tm, ddpm=ddpm, T=T, device=device,
                       lambda_ts=lambda_ts, lambda_ord=lambda_ord)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        # ---- Training ----
        diffusion.train()
        gnn.train()
        train_losses: list[float] = []

        for graph_list, groupings, orders_oh, ts_norm, _ham_params, _fid in train_loader:
            groupings = groupings.to(device)
            orders_oh = orders_oh.to(device)
            ts_norm = ts_norm.to(device)

            total_loss, _ = _step(graph_list, groupings, orders_oh, ts_norm,
                                   gnn, diffusion, **step_kwargs)
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(diffusion.parameters()) + list(gnn.parameters()), max_norm=1.0
            )
            optimizer.step()
            ema.update(diffusion)
            train_losses.append(float(total_loss))

        # ---- Validation ----
        diffusion.eval()
        gnn.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for graph_list, groupings, orders_oh, ts_norm, _ham_params, _fid in val_loader:
                groupings = groupings.to(device)
                orders_oh = orders_oh.to(device)
                ts_norm = ts_norm.to(device)
                total_loss, _ = _step(graph_list, groupings, orders_oh, ts_norm,
                                       gnn, diffusion, **step_kwargs)
                val_losses.append(float(total_loss))

        mean_train = sum(train_losses) / max(len(train_losses), 1)
        mean_val = sum(val_losses) / max(len(val_losses), 1)
        elapsed = time.time() - t0

        if epoch % log_interval == 0 or epoch == max_epochs - 1:
            log.info("epoch=%d  train=%.4e  val=%.4e  %.1fs",
                     epoch, mean_train, mean_val, elapsed)

        # ---- Checkpoint ----
        is_best = mean_val < best_val_loss
        if is_best:
            best_val_loss = mean_val
        if is_best or (epoch + 1) % save_interval == 0:
            ckpt_name = "diffusion_best.pt" if is_best else f"diffusion_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "diffusion_state": diffusion.state_dict(),
                    "ema_state": ema.shadow.state_dict(),
                    "gnn_state": gnn.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": mean_val,
                },
                checkpoint_dir / ckpt_name,
            )
            log.info("Saved%s: %s", " best" if is_best else "", ckpt_name)

    log.info("Phase 3 complete. Best val loss: %.4e", best_val_loss)


if __name__ == "__main__":
    main()
