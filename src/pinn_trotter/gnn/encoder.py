"""HamiltonianGNNEncoder: MPNN-based encoder for Pauli Hamiltonians.

Works in two modes:
  - PyG mode (torch_geometric available): uses MessagePassing base class.
  - Fallback mode: pure-PyTorch dense MPNN using adjacency tensors.

Both modes expose the same interface and produce identical outputs when
given the same inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn_trotter.gnn.pooling import AttentionPooling


# ---------------------------------------------------------------------------
# Pure-PyTorch MPNN layer (used when torch_geometric is absent)
# ---------------------------------------------------------------------------

class _MPNNLayer(nn.Module):
    """Single message-passing layer with edge features (dense adjacency).

    Message: m_{ij} = φ_msg([h_i || h_j || e_{ij}])
    Update:  h_i'   = φ_upd([h_i || Σ_{j∈N(i)} m_{ij}])
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        msg_in = 2 * node_dim + edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        upd_in = node_dim + hidden_dim
        self.upd_mlp = nn.Sequential(
            nn.Linear(upd_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, node_dim)
            edge_index: (2, E) — directed edges [src; dst]
            edge_attr:  (E, edge_dim)

        Returns:
            Updated node embeddings (N, node_dim).
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Messages
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)  # (E, *)
        msgs = self.msg_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate (sum) per destination node
        agg = torch.zeros(N, msgs.shape[-1], device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)

        # Update
        upd_input = torch.cat([x, agg], dim=-1)
        return x + self.upd_mlp(upd_input)  # residual


# ---------------------------------------------------------------------------
# PyG MessagePassing layer (used when torch_geometric is available)
# ---------------------------------------------------------------------------

def _make_pyg_layer(node_dim: int, edge_dim: int, hidden_dim: int, dropout: float):
    """Build a PyG-based MPNN layer. Only called when PyG is installed."""
    from torch_geometric.nn import MessagePassing

    class _PyGMPNNLayer(MessagePassing):
        def __init__(self):
            super().__init__(aggr="sum")
            msg_in = 2 * node_dim + edge_dim
            self.msg_mlp = nn.Sequential(
                nn.Linear(msg_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            upd_in = node_dim + hidden_dim
            self.upd_mlp = nn.Sequential(
                nn.Linear(upd_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, node_dim),
            )

        def forward(self, x, edge_index, edge_attr):
            agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            return x + self.upd_mlp(torch.cat([x, agg], dim=-1))

        def message(self, x_i, x_j, edge_attr):
            return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    return _PyGMPNNLayer()


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class HamiltonianGNNEncoder(nn.Module):
    """GNN encoder that maps a Pauli Hamiltonian graph to a condition vector.

    Architecture:
        node_feat → Linear(node_feat_dim → hidden_dim)
        → 4× MPNNLayer(hidden_dim, edge_feat_dim, hidden_dim)
        → AttentionPooling(hidden_dim → output_dim)

    Works with or without torch_geometric. When PyG is available, uses
    MessagePassing for memory-efficient sparse message passing. Otherwise
    falls back to a functionally equivalent dense implementation.

    Args:
        node_feat_dim: Input node feature dimension (coefficient + locality
                       + support one-hot = n_qubits + 2).
        edge_feat_dim: Edge feature dimension (3 by default).
        hidden_dim:    Internal MPNN hidden dimension.
        output_dim:    Final condition vector dimension (512 by default).
        n_layers:      Number of MPNN layers (4 by default).
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        node_feat_dim: int = 16,
        edge_feat_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        try:
            import torch_geometric  # noqa: F401
            self._use_pyg = True
        except ImportError:
            self._use_pyg = False

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        if self._use_pyg:
            self.layers = nn.ModuleList([
                _make_pyg_layer(hidden_dim, edge_feat_dim, hidden_dim, dropout)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                _MPNNLayer(hidden_dim, edge_feat_dim, hidden_dim, dropout)
                for _ in range(n_layers)
            ])

        self.pooling = AttentionPooling(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a Hamiltonian graph to a condition vector.

        Args:
            x:          Node features, shape (N, node_feat_dim).
            edge_index: Edge index, shape (2, E).
            edge_attr:  Edge features, shape (E, edge_feat_dim).
            batch:      Batch vector for PyG batching, shape (N,). Optional.

        Returns:
            Condition vector, shape (1, output_dim) or (B, output_dim).
        """
        h = F.silu(self.input_proj(x))  # (N, hidden_dim)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.pooling(h, batch)  # (1 or B, output_dim)

    def encode_hamiltonian_graph(self, H_graph) -> torch.Tensor:
        """Convenience wrapper: encode a HamiltonianGraph object directly.

        Requires torch_geometric to build the graph data; otherwise raises.

        Args:
            H_graph: HamiltonianGraph instance.

        Returns:
            Condition vector, shape (1, output_dim).
        """
        data = H_graph.to_pyg_data()
        device = next(self.parameters()).device
        return self.forward(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )
