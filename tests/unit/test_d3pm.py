"""Unit tests for D3PM discrete diffusion."""

from __future__ import annotations

import pytest
import torch

from pinn_trotter.diffusion.d3pm import d3pm_loss, d3pm_reverse_step
from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix


@pytest.fixture
def tm():
    return UniformTransitionMatrix(K=4, T=100, beta_schedule="linear")


class TestUniformTransitionMatrix:
    def test_Q_bar_shape(self, tm):
        assert tm.Q_bar.shape == (100, 4, 4)

    def test_Q_bar_row_sums_to_one(self, tm):
        row_sums = tm.Q_bar.sum(dim=-1)  # (T, K)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_Q_bar_at_T_approaches_uniform(self):
        """At large t with aggressive schedule, Q̄_t should approach 1/K."""
        # Use high beta_end so alpha_bar → 0 within T=100 steps
        tm_agg = UniformTransitionMatrix(K=4, T=100, beta_schedule="linear",
                                         beta_start=0.01, beta_end=0.5)
        Q_last = tm_agg.Q_bar[-1]  # (K, K)
        expected = torch.full((4, 4), 1.0 / 4)
        assert torch.allclose(Q_last, expected, atol=0.05), (
            f"Q̄_T not close to uniform:\n{Q_last}"
        )

    def test_get_Q_bar_batch(self, tm):
        t = torch.tensor([0, 10, 50, 99])
        Q = tm.get_Q_bar(t)
        assert Q.shape == (4, 4, 4)

    def test_forward_sample_shape(self, tm):
        x0 = torch.randint(0, 4, (8, 6))
        t = torch.randint(0, 100, (8,))
        xt = tm.forward_sample(x0, t)
        assert xt.shape == (8, 6)
        assert (xt >= 0).all() and (xt < 4).all()

    def test_forward_sample_t0_mostly_clean(self):
        """At t=0 (minimal noise), x_t should still be valid labels."""
        tm_small = UniformTransitionMatrix(K=4, T=100, beta_schedule="linear",
                                           beta_start=1e-6, beta_end=1e-5)
        x0 = torch.zeros(50, 8, dtype=torch.long)  # all zeros
        t = torch.zeros(50, dtype=torch.long)
        xt = tm_small.forward_sample(x0, t)
        # With very low noise, most should remain 0
        frac_unchanged = (xt == 0).float().mean().item()
        assert frac_unchanged > 0.9, f"Expected mostly clean at t=0, got {frac_unchanged:.2f}"

    def test_forward_sample_marginal_uniform_at_T(self):
        """At t=T-1 with aggressive schedule, marginal ≈ uniform over K classes."""
        torch.manual_seed(0)
        tm_agg = UniformTransitionMatrix(K=4, T=100, beta_schedule="linear",
                                         beta_start=0.01, beta_end=0.5)
        x0 = torch.zeros(500, 10, dtype=torch.long)
        t = torch.full((500,), 99, dtype=torch.long)
        xt = tm_agg.forward_sample(x0, t)
        for k in range(4):
            frac = (xt == k).float().mean().item()
            assert abs(frac - 0.25) < 0.05, f"class {k} freq={frac:.3f}, expected ~0.25"

    def test_posterior_logits_shape(self, tm):
        x_t = torch.randint(0, 4, (4, 6))
        x0_logits = torch.randn(4, 6, 4)
        t = torch.randint(1, 99, (4,))
        logits = tm.compute_posterior_logits(x_t, x0_logits, t)
        assert logits.shape == (4, 6, 4)

    def test_posterior_normalized(self, tm):
        """Posterior probabilities should sum to 1 per token."""
        x_t = torch.randint(0, 4, (3, 5))
        x0_logits = torch.randn(3, 5, 4)
        t = torch.randint(1, 99, (3,))
        logits = tm.compute_posterior_logits(x_t, x0_logits, t)
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestD3PMReverseStep:
    def test_shape(self, tm):
        x_t = torch.randint(0, 4, (8, 6))
        logits = torch.randn(8, 6, 4)
        t = torch.randint(1, 99, (8,))
        out = d3pm_reverse_step(x_t, logits, t, tm)
        assert out.shape == (8, 6)
        assert (out >= 0).all() and (out < 4).all()


class TestD3PMLoss:
    def test_loss_nonneg(self, tm):
        B, M, K = 4, 6, 4
        model_output = torch.randn(B, M, K)
        x_0 = torch.randint(0, K, (B, M))
        x_t = torch.randint(0, K, (B, M))
        t = torch.randint(0, 100, (B,))
        loss = d3pm_loss(model_output, x_0, x_t, t, tm)
        assert float(loss) >= 0.0

    def test_loss_backward(self, tm):
        B, M, K = 4, 6, 4
        model_output = torch.randn(B, M, K, requires_grad=True)
        x_0 = torch.randint(0, K, (B, M))
        x_t = torch.randint(0, K, (B, M))
        t = torch.randint(1, 99, (B,))
        loss = d3pm_loss(model_output, x_0, x_t, t, tm)
        loss.backward()
        assert model_output.grad is not None

    def test_perfect_prediction_low_loss(self, tm):
        """When model predicts true x_0 with high confidence, loss should be small."""
        B, M, K = 4, 6, 4
        x_0 = torch.randint(0, K, (B, M))
        x_t = torch.randint(0, K, (B, M))
        t = torch.randint(1, 99, (B,))
        # High-confidence correct prediction
        model_output = torch.full((B, M, K), -100.0)
        for b in range(B):
            for m in range(M):
                model_output[b, m, x_0[b, m]] = 100.0
        loss_good = d3pm_loss(model_output, x_0, x_t, t, tm)

        # Random prediction
        model_output_rand = torch.randn(B, M, K)
        loss_rand = d3pm_loss(model_output_rand, x_0, x_t, t, tm)

        assert float(loss_good) < float(loss_rand)
