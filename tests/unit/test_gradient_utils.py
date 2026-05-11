"""Unit tests for gradient utilities: Gumbel-Softmax, STE, policy log-prob."""

from __future__ import annotations

import pytest
import torch

from pinn_trotter.optimizer.gradient_utils import (
    compute_policy_log_prob,
    gumbel_softmax,
    straight_through_estimator,
)


class TestGumbelSoftmax:
    def test_output_shape(self):
        logits = torch.randn(8, 4)
        out = gumbel_softmax(logits, temperature=1.0, hard=True)
        assert out.shape == (8, 4)

    def test_hard_is_one_hot(self):
        logits = torch.randn(16, 6)
        out = gumbel_softmax(logits, temperature=1.0, hard=True)
        assert (out.sum(dim=-1) - 1.0).abs().max() < 1e-5
        assert ((out == 0) | (out == 1)).all()

    def test_soft_is_distribution(self):
        logits = torch.randn(8, 4)
        out = gumbel_softmax(logits, temperature=1.0, hard=False)
        assert (out >= 0).all()
        assert (out.sum(dim=-1) - 1.0).abs().max() < 1e-5

    def test_gradient_flows_hard(self):
        logits = torch.randn(4, 3, requires_grad=True)
        out = gumbel_softmax(logits, temperature=1.0, hard=True)
        # out.sum() is constant (softmax rows sum to 1), so use a non-uniform
        # aggregation that actually depends on logits via the STE soft path.
        out[:, 0].sum().backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_low_temperature_approaches_argmax(self):
        """At very low temperature, hard sample should equal argmax of logits."""
        torch.manual_seed(0)
        logits = torch.tensor([[0.0, 10.0, 0.0]])  # clear winner at index 1
        results = []
        for _ in range(20):
            out = gumbel_softmax(logits, temperature=0.01, hard=True)
            results.append(out.argmax(dim=-1).item())
        # With very low temperature, should almost always pick index 1
        assert results.count(1) >= 18, f"Expected argmax=1 most of the time, got {results}"

    def test_batch_dimensions(self):
        logits = torch.randn(3, 5, 4)
        out = gumbel_softmax(logits, temperature=1.0, hard=True)
        assert out.shape == (3, 5, 4)
        assert (out.sum(dim=-1) - 1.0).abs().max() < 1e-5


class TestStraightThroughEstimator:
    def test_forward_uses_hard(self):
        x_hard = torch.tensor([1.0, 0.0, 0.0])
        x_soft = torch.tensor([0.7, 0.2, 0.1])
        out = straight_through_estimator(x_hard, x_soft)
        assert torch.allclose(out, x_hard)

    def test_backward_uses_soft(self):
        x_soft = torch.tensor([0.7, 0.2, 0.1], requires_grad=True)
        x_hard = torch.tensor([1.0, 0.0, 0.0])
        out = straight_through_estimator(x_hard, x_soft)
        out.sum().backward()
        assert x_soft.grad is not None
        # Gradient w.r.t. x_soft should be all-ones (d(sum)/d(x_soft))
        assert torch.allclose(x_soft.grad, torch.ones(3))

    def test_no_grad_through_hard(self):
        x_hard = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        x_soft = torch.tensor([0.7, 0.2, 0.1], requires_grad=True)
        out = straight_through_estimator(x_hard, x_soft)
        out.sum().backward()
        # x_hard.grad should be zero (detached in forward)
        assert x_hard.grad is None or x_hard.grad.abs().sum() == 0


class TestComputePolicyLogProb:
    def test_shape(self):
        B, M, K, Kg = 4, 6, 8, 8
        grouping_labels = torch.randint(0, Kg, (B, M))
        order_labels = torch.randint(0, 3, (B, K))
        g_logits = torch.randn(B, M, Kg)
        o_logits = torch.randn(B, K, 3)
        log_p = compute_policy_log_prob(grouping_labels, order_labels, g_logits, o_logits)
        assert log_p.shape == (B,)

    def test_log_prob_nonpositive(self):
        """Log-probabilities of valid samples must be ≤ 0."""
        B, M, K = 4, 6, 4
        g = torch.randint(0, K, (B, M))
        o = torch.randint(0, 3, (B, K))
        g_logits = torch.randn(B, M, K)
        o_logits = torch.randn(B, K, 3)
        log_p = compute_policy_log_prob(g, o, g_logits, o_logits)
        assert (log_p <= 0).all(), f"log_prob > 0: {log_p}"

    def test_high_confidence_higher_log_prob(self):
        """Confident correct predictions should yield higher log-prob."""
        B, M, K = 2, 4, 4
        g = torch.zeros(B, M, dtype=torch.long)
        o = torch.zeros(B, K, dtype=torch.long)

        # High confidence on correct class
        g_conf = torch.full((B, M, K), -100.0)
        g_conf[..., 0] = 100.0
        o_conf = torch.full((B, K, 3), -100.0)
        o_conf[..., 0] = 100.0

        # Random logits
        g_rand = torch.randn(B, M, K)
        o_rand = torch.randn(B, K, 3)

        lp_conf = compute_policy_log_prob(g, o, g_conf, o_conf)
        lp_rand = compute_policy_log_prob(g, o, g_rand, o_rand)
        assert (lp_conf > lp_rand).all()

    def test_gradients_flow(self):
        B, M, K = 3, 5, 4
        g = torch.randint(0, K, (B, M))
        o = torch.randint(0, 3, (B, K))
        g_logits = torch.randn(B, M, K, requires_grad=True)
        o_logits = torch.randn(B, K, 3, requires_grad=True)
        log_p = compute_policy_log_prob(g, o, g_logits, o_logits)
        log_p.sum().backward()
        assert g_logits.grad is not None
        assert o_logits.grad is not None
