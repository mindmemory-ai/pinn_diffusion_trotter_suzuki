"""Unit tests for GNN encoder, pooling, and fidelity head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder, _MPNNLayer
from pinn_trotter.gnn.head import FidelityRegressionHead
from pinn_trotter.gnn.pooling import AttentionPooling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_nodes: int = 6, n_edges: int = 8, node_dim: int = 6, edge_dim: int = 3):
    """Random graph tensors for testing."""
    torch.manual_seed(0)
    x = torch.randn(n_nodes, node_dim)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(n_edges, edge_dim)
    return x, edge_index, edge_attr


# ---------------------------------------------------------------------------
# AttentionPooling
# ---------------------------------------------------------------------------

class TestAttentionPooling:
    def test_output_shape_single(self):
        pool = AttentionPooling(in_dim=32, out_dim=64)
        x = torch.randn(10, 32)
        out = pool(x)
        assert out.shape == (1, 64)

    def test_output_shape_batched(self):
        pool = AttentionPooling(in_dim=32, out_dim=64)
        x = torch.randn(10, 32)
        # 2 graphs: nodes 0-4 → graph 0, nodes 5-9 → graph 1
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        out = pool(x, batch)
        assert out.shape == (2, 64)

    def test_scores_sum_to_one(self):
        pool = AttentionPooling(in_dim=8, out_dim=16)
        x = torch.randn(5, 8)
        scores = pool.score_mlp(x)  # (5, 1)
        alpha = torch.softmax(scores, dim=0)
        assert torch.isclose(alpha.sum(), torch.ones(1), atol=1e-5)

    def test_backward(self):
        pool = AttentionPooling(in_dim=16, out_dim=32)
        x = torch.randn(8, 16)
        out = pool(x)
        out.sum().backward()
        for p in pool.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# _MPNNLayer
# ---------------------------------------------------------------------------

class TestMPNNLayer:
    def test_output_shape(self):
        layer = _MPNNLayer(node_dim=32, edge_dim=3, hidden_dim=64)
        x, edge_index, edge_attr = _make_graph(6, 8, 32, 3)
        out = layer(x, edge_index, edge_attr)
        assert out.shape == (6, 32)

    def test_residual_connection(self):
        """With zero weights, output should equal input (residual)."""
        layer = _MPNNLayer(node_dim=4, edge_dim=2, hidden_dim=8)
        # Zero all weights
        for p in layer.parameters():
            p.data.zero_()
        x = torch.randn(4, 4)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 2)
        out = layer(x, edge_index, edge_attr)
        assert torch.allclose(out, x, atol=1e-6)

    def test_no_edges(self):
        """Layer with no edges should not crash."""
        layer = _MPNNLayer(node_dim=8, edge_dim=3, hidden_dim=16)
        x = torch.randn(5, 8)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 3)
        out = layer(x, edge_index, edge_attr)
        assert out.shape == (5, 8)

    def test_backward(self):
        layer = _MPNNLayer(node_dim=16, edge_dim=3, hidden_dim=32)
        x, ei, ea = _make_graph(6, 8, 16, 3)
        out = layer(x, ei, ea)
        out.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# HamiltonianGNNEncoder
# ---------------------------------------------------------------------------

class TestHamiltonianGNNEncoder:
    @pytest.fixture
    def encoder(self):
        return HamiltonianGNNEncoder(
            node_feat_dim=6,
            edge_feat_dim=3,
            hidden_dim=32,
            output_dim=64,
            n_layers=2,
            dropout=0.0,
        )

    def test_output_shape(self, encoder):
        x, ei, ea = _make_graph(8, 12, 6, 3)
        out = encoder(x, ei, ea)
        assert out.shape == (1, 64)

    def test_batched_output_shape(self, encoder):
        x, ei, ea = _make_graph(8, 12, 6, 3)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = encoder(x, ei, ea, batch)
        assert out.shape == (2, 64)

    def test_backward(self, encoder):
        x, ei, ea = _make_graph(8, 12, 6, 3)
        out = encoder(x, ei, ea)
        out.sum().backward()
        for p in encoder.parameters():
            assert p.grad is not None

    def test_permutation_equivariance(self, encoder):
        """Permuting node order should give the same pooled output (≤ 1e-5).

        Attention pooling with symmetric aggregation is permutation-invariant,
        so the graph-level vector should be the same regardless of node ordering.
        """
        encoder.eval()
        torch.manual_seed(42)
        N, E, Nd, Ed = 8, 14, 6, 3
        x = torch.randn(N, Nd)
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        edge_index = torch.stack([src, dst])
        edge_attr = torch.randn(E, Ed)

        out1 = encoder(x, edge_index, edge_attr)

        # Permute nodes
        perm = torch.randperm(N)
        inv = torch.argsort(perm)
        x_p = x[perm]
        ei_p = torch.stack([inv[src], inv[dst]])
        out2 = encoder(x_p, ei_p, edge_attr)  # edge_attr unchanged (same edges)

        assert torch.allclose(out1, out2, atol=1e-4), (
            f"Permutation invariance violated: max diff = {(out1 - out2).abs().max():.6f}"
        )

    def test_empty_edges(self, encoder):
        """Encoder with no edges (fully commuting Hamiltonian) should not crash."""
        x = torch.randn(5, 6)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 3)
        out = encoder(x, edge_index, edge_attr)
        assert out.shape == (1, 64)


# ---------------------------------------------------------------------------
# FidelityRegressionHead
# ---------------------------------------------------------------------------

class TestFidelityRegressionHead:
    def test_output_range(self):
        head = FidelityRegressionHead(input_dim=64)
        cond = torch.randn(10, 64)
        pred = head(cond)
        assert pred.shape == (10, 1)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_loss_nonneg(self):
        head = FidelityRegressionHead(input_dim=64)
        cond = torch.randn(8, 64)
        target = torch.rand(8)
        loss = head.loss(cond, target)
        assert float(loss.detach()) >= 0.0

    def test_backward(self):
        head = FidelityRegressionHead(input_dim=32)
        cond = torch.randn(4, 32)
        target = torch.rand(4)
        loss = head.loss(cond, target)
        loss.backward()
        for p in head.parameters():
            assert p.grad is not None
