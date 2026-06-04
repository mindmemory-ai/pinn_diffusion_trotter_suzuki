"""Full-dataset per-type R² analysis for Phase 3 v3 (per-type heads)."""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pinn_trotter.data.dataset import TrotterDataset
from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder
from pinn_trotter.gnn.head import TypeConditionedFidelityHead
from torch.utils.data import DataLoader

CKPT_PATH = "/home/ga4ss/pdts/phase3_v3_checkpoints/gnn_pretrain_best_20260602_162343.pt"
DATASET_PATH = "/home/ga4ss/pdts/data/dataset_f0.1_s1.0_m0.0.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048


def _collate(batch, M_target: int, K: int):
    graph_list, strat_list, fids, depths, htype_indices = zip(*batch)
    B = len(strat_list)
    # group_labels: (M,) long tensor → pad to (B, M_target)
    glist, rmlist = [], []
    for s in strat_list:
        g = s[0]  # (M_real,) long
        M_real = g.shape[0]
        if M_real < M_target:
            g_pad = torch.cat([g, torch.zeros(M_target - M_real, dtype=g.dtype)])
            mask = torch.cat([
                torch.ones(M_real, dtype=torch.float32),
                torch.zeros(M_target - M_real, dtype=torch.float32),
            ])
        else:
            g_pad = g[:M_target]
            mask = torch.ones(M_target, dtype=torch.float32)
        glist.append(g_pad)
        rmlist.append(mask)
    groupings = torch.stack(glist)          # (B, M_target)
    real_mask = torch.stack(rmlist)         # (B, M_target)
    orders_oh = torch.stack([s[1] for s in strat_list])   # (B, K, 3)
    ts_norm = torch.stack([s[2] for s in strat_list])     # (B, K)
    ham_params = torch.stack([s[3] for s in strat_list])  # (B, 5)
    fidelity = torch.stack(fids)
    htype_idx = torch.tensor(htype_indices, dtype=torch.long)
    return list(graph_list), groupings, orders_oh, ts_norm, ham_params, fidelity, real_mask, htype_idx


def _strategy_features(groupings, orders_oh, ts_norm, ham_params, max_groups, group_mask=None):
    g_flat = F.one_hot(groupings.long(), num_classes=max_groups).float().reshape(groupings.shape[0], -1)
    if group_mask is not None:
        mask_flat = group_mask.unsqueeze(-1).expand(-1, -1, max_groups).reshape(groupings.shape[0], -1)
        g_flat = g_flat * mask_flat
    o_flat = orders_oh.float().reshape(orders_oh.shape[0], -1)
    t_feat = ts_norm.float()
    h_feat = ham_params.float()
    return torch.cat([g_flat, o_flat, t_feat, h_feat], dim=1)


def _trotter_proxy(graph_list, groupings, ts_norm, ham_params):
    B = groupings.shape[0]
    t_total = ham_params[:, 0].float()
    proxies = []
    for i in range(B):
        g_i = groupings[i].cpu()
        ts_i = ts_norm[i].cpu()
        t_i = float(t_total[i].cpu())
        graph = graph_list[i]
        if hasattr(graph, 'edge_index') and graph.edge_index.shape[1] > 0:
            ei = graph.edge_index.cpu()
            src, dst = ei[0], ei[1]
            g_src = g_i[src]
            g_dst = g_i[dst]
            inter_mask = (g_src != g_dst)
            if inter_mask.sum() == 0:
                proxies.append(torch.tensor(0.0))
                continue
            comm_norms = graph.edge_attr[:, 0].cpu()
            ts_src = ts_i[g_src[inter_mask]].float() * t_i
            ts_dst = ts_i[g_dst[inter_mask]].float() * t_i
            proxy = (comm_norms[inter_mask].float() * ts_src * ts_dst).sum()
            proxies.append(proxy)
        else:
            proxies.append(torch.tensor(0.0))
    return torch.stack(proxies).unsqueeze(1)


def _r2_score(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-10))


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    max_n_qubits = ckpt["max_n_qubits"]
    max_M = ckpt["max_M"]
    max_groups = ckpt["max_groups"]
    print(f"  max_n_qubits={max_n_qubits}, max_M={max_M}, max_groups={max_groups}")

    # Build model
    gnn = HamiltonianGNNEncoder(
        node_feat_dim=max_n_qubits + 2,
        hidden_dim=512,
        output_dim=768,
        n_layers=6,
    ).to(DEVICE)

    # Strategy feat dim = M_target*K (one-hot grouping) + K*3 (orders) + K (ts) + 5 (ham_params)
    strat_feat_dim = max_M * max_groups + max_groups * 3 + max_groups + 5
    head_input_dim = 768 + strat_feat_dim + 1  # +1 for trotter proxy
    print(f"  strat_feat_dim={strat_feat_dim}, head_input_dim={head_input_dim}")

    head = TypeConditionedFidelityHead(
        input_dim=head_input_dim,
        hidden_dims=[256, 128, 64],
        dropout=0.1,
    ).to(DEVICE)

    gnn.load_state_dict(ckpt["gnn_state"])
    head.load_state_dict_per_type(ckpt["head_state"])
    gnn.eval()
    head.eval()

    print("Loading full dataset...")
    dataset = TrotterDataset(
        DATASET_PATH,
        max_groups=max_groups,
        include_type=True,
    )
    print(f"Dataset size: {len(dataset)}")

    from functools import partial
    collate_fn = partial(_collate, M_target=max_M, K=max_groups)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    all_preds, all_fids, all_htypes = [], [], []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (graph_list, groupings, orders_oh, ts_norm, ham_params, fidelity, real_mask, htype_idx) in enumerate(loader):
            fidelity = fidelity.to(DEVICE)
            htype_idx = htype_idx.to(DEVICE)

            # GNN encoding
            if hasattr(graph_list[0], 'x'):
                from torch_geometric.data import Batch
                batched = Batch.from_data_list(graph_list).to(DEVICE)
                cond = gnn(batched.x, batched.edge_index, batched.edge_attr, batched.batch.to(DEVICE))
            else:
                from pinn_trotter.hamiltonian.pauli_utils import locality
                clist = []
                for gd in graph_list:
                    n = int(gd["n_qubits"])
                    ps = gd["pauli_strings"]
                    coeffs = gd["coefficients"]
                    if hasattr(coeffs, "numpy"):
                        coeffs = coeffs.numpy()
                    nf = torch.zeros(len(ps), max(max_n_qubits, n) + 2, device=DEVICE)
                    for i, (s, c) in enumerate(zip(ps, coeffs)):
                        nf[i, 0] = float(c)
                        nf[i, 1] = float(locality(s))
                        for q, ch in enumerate(s):
                            nf[i, 2 + q] = 0.0 if ch == "I" else 1.0
                    ei = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
                    ea = torch.zeros(0, 2, device=DEVICE)
                    clist.append(gnn(nf, ei, ea))
                cond = torch.cat(clist, dim=0)

            # Strategy features + Trotter proxy → condition
            groupings = groupings.to(DEVICE)
            orders_oh = orders_oh.to(DEVICE)
            ts_norm = ts_norm.to(DEVICE)
            ham_params = ham_params.to(DEVICE)
            real_mask = real_mask.to(DEVICE)

            s_feat = _strategy_features(groupings, orders_oh, ts_norm, ham_params, max_groups, real_mask)
            t_proxy = _trotter_proxy(graph_list, groupings, ts_norm, ham_params).to(DEVICE)
            cond_full = torch.cat([cond, s_feat, t_proxy], dim=1)

            pred = head(cond_full, htype_idx).squeeze(-1)

            all_preds.append(pred.cpu())
            all_fids.append(fidelity.cpu())
            all_htypes.append(htype_idx.cpu())

            if batch_idx % 3 == 0:
                print(f"  batch {batch_idx}/{len(loader)}")

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_fids)
    y_htype = torch.cat(all_htypes)

    overall_r2 = _r2_score(y_true, y_pred)
    rmse = float(torch.sqrt(F.mse_loss(y_pred, y_true)))

    print()
    print("=" * 66)
    print("FULL DATASET PER-TYPE R² ANALYSIS (v3 — Per-Type Heads)")
    print("=" * 66)
    print(f"Overall R²:  {overall_r2:.4f}   RMSE: {rmse:.4f}")
    print()
    print(f"{'Type':<16} {'Count':>6} {'R²':>10} {'RMSE':>10} {'Mean F':>10} {'Std F':>10}")
    print("-" * 66)

    IDX_TO_TYPE = {0: "tfim", 1: "heisenberg", 2: "random_pauli"}
    per_type_results = {}
    for idx in range(3):
        mask = y_htype == idx
        n = mask.sum().item()
        if n > 1:
            r2 = _r2_score(y_true[mask], y_pred[mask])
            rmse_i = float(np.sqrt(F.mse_loss(y_pred[mask], y_true[mask]).item()))
            mean_f = y_true[mask].mean().item()
            std_f = y_true[mask].std().item()
            tname = IDX_TO_TYPE[idx]
            per_type_results[tname] = (n, r2, rmse_i, mean_f, std_f)
            print(f"{tname:<16} {n:>6} {r2:>10.4f} {rmse_i:>10.4f} {mean_f:>10.4f} {std_f:>10.4f}")

    print()
    print("=== COMPARISON TO BASELINE (single-head) ===")
    print(f"{'Type':<16} {'Baseline R²':>12} {'v3 R²':>10} {'Delta':>10}")
    print("-" * 52)
    baseline = {"tfim": 0.56, "heisenberg": 0.37, "random_pauli": 0.31, "overall": 0.54}
    for tname in ["tfim", "heisenberg", "random_pauli"]:
        if tname in per_type_results:
            _, r2, _, _, _ = per_type_results[tname]
            delta = r2 - baseline[tname]
            print(f"{tname:<16} {baseline[tname]:>12.4f} {r2:>10.4f} {delta:>+10.4f}")
    delta_overall = overall_r2 - baseline["overall"]
    print(f"{'overall':<16} {baseline['overall']:>12.4f} {overall_r2:>10.4f} {delta_overall:>+10.4f}")


if __name__ == "__main__":
    main()
