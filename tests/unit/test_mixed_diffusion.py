"""Unit tests for MixedDiffusionModel, EMAWrapper, DDPM, and guided_sample."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
from pinn_trotter.diffusion.mixed_model import (
    EMAWrapper,
    MixedDiffusionModel,
    guided_sample,
    sinusoidal_embedding,
)
from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model():
    return MixedDiffusionModel(
        max_groups=4,
        n_terms=6,
        condition_dim=32,
        fused_dim=16,
        time_embed_dim=16,
        grouping_layers=1,
        order_layers=1,
        ts_mlp_layers=1,
        dropout=0.0,
        p_cond_drop=0.1,
    )


@pytest.fixture
def tm():
    return UniformTransitionMatrix(K=4, T=20, beta_schedule="linear")


@pytest.fixture
def ddpm():
    return ContinuousDDPM(T=20, beta_schedule="linear")


# ---------------------------------------------------------------------------
# Sinusoidal embedding
# ---------------------------------------------------------------------------

class TestSinusoidalEmbedding:
    def test_shape(self):
        t = torch.randint(0, 100, (8,))
        out = sinusoidal_embedding(t, dim=64)
        assert out.shape == (8, 64)

    def test_different_t_different_emb(self):
        t1 = torch.tensor([0])
        t2 = torch.tensor([100])
        e1 = sinusoidal_embedding(t1, 32)
        e2 = sinusoidal_embedding(t2, 32)
        assert not torch.allclose(e1, e2)


# ---------------------------------------------------------------------------
# ContinuousDDPM
# ---------------------------------------------------------------------------

class TestContinuousDDPM:
    def test_forward_sample_shape(self, ddpm):
        x0 = torch.randn(8, 4)
        t = torch.randint(0, 20, (8,))
        xt, noise = ddpm.forward_sample(x0, t)
        assert xt.shape == (8, 4)
        assert noise.shape == (8, 4)

    def test_reverse_step_shape(self, ddpm):
        xt = torch.randn(8, 4)
        noise_pred = torch.randn(8, 4)
        t = torch.randint(1, 20, (8,))
        out = ddpm.reverse_step(xt, noise_pred, t)
        assert out.shape == (8, 4)

    def test_no_noise_at_t0(self, ddpm):
        """At t=0, reverse_step should return deterministic mean."""
        torch.manual_seed(0)
        xt = torch.randn(4, 4)
        noise_pred = torch.randn(4, 4)
        t = torch.zeros(4, dtype=torch.long)
        out1 = ddpm.reverse_step(xt, noise_pred, t)
        out2 = ddpm.reverse_step(xt, noise_pred, t)
        assert torch.allclose(out1, out2)

    def test_ddpm_loss_nonneg(self, ddpm):
        pred = torch.randn(8, 4)
        true = torch.randn(8, 4)
        loss = ddpm.ddpm_loss(pred, true)
        assert float(loss) >= 0.0

    def test_schedule_betas_increasing_linear(self):
        d = ContinuousDDPM(T=100, beta_schedule="linear", beta_start=1e-4, beta_end=0.02)
        assert (d.betas[1:] >= d.betas[:-1]).all()

    def test_alpha_bar_decreasing(self, ddpm):
        assert (ddpm.alpha_bar[1:] <= ddpm.alpha_bar[:-1]).all()


# ---------------------------------------------------------------------------
# MixedDiffusionModel
# ---------------------------------------------------------------------------

class TestMixedDiffusionModel:
    def test_output_shapes(self, small_model):
        B, M, K = 4, 6, 4
        grouping_noisy = torch.randint(0, K, (B, M))
        time_steps_noisy = torch.randn(B, K)
        orders_noisy = F.one_hot(torch.randint(0, 3, (B, K)), num_classes=3).float()
        t_diff = torch.randint(0, 20, (B,))
        condition = torch.randn(B, 32)

        g_logits, ts_pred, o_logits = small_model(
            grouping_noisy, time_steps_noisy, orders_noisy, t_diff, condition
        )
        assert g_logits.shape == (B, M, K)
        assert ts_pred.shape == (B, K)
        assert o_logits.shape == (B, K, 3)

    def test_cfg_drop_condition(self, small_model):
        """Drop_condition=True should give different output than False."""
        B, M, K = 2, 6, 4
        g = torch.randint(0, K, (B, M))
        ts = torch.randn(B, K)
        o = F.one_hot(torch.randint(0, 3, (B, K)), num_classes=3).float()
        t = torch.randint(0, 20, (B,))
        cond = torch.randn(B, 32)

        out_cond = small_model(g, ts, o, t, cond, drop_condition=False)
        out_uncond = small_model(g, ts, o, t, cond, drop_condition=True)
        # Outputs should differ (condition was non-zero)
        assert not torch.allclose(out_cond[0], out_uncond[0])

    def test_zero_condition_is_unconditional(self, small_model):
        """Passing zeros as condition should equal drop_condition=True."""
        B, M, K = 2, 6, 4
        g = torch.randint(0, K, (B, M))
        ts = torch.randn(B, K)
        o = F.one_hot(torch.randint(0, 3, (B, K)), num_classes=3).float()
        t = torch.randint(0, 20, (B,))
        cond = torch.randn(B, 32)
        zeros = torch.zeros(B, 32)

        out_drop = small_model(g, ts, o, t, cond, drop_condition=True)
        out_zero = small_model(g, ts, o, t, zeros, drop_condition=False)
        assert torch.allclose(out_drop[0], out_zero[0], atol=1e-5)

    def test_backward(self, small_model, tm, ddpm):
        B, M, K = 4, 6, 4
        order_tm = UniformTransitionMatrix(K=3, T=20, beta_schedule="linear")

        g = torch.randint(0, K, (B, M))
        ts_clean = torch.rand(B, K)
        o_clean = torch.randint(0, 3, (B, K))
        t = torch.randint(1, 19, (B,))
        cond = torch.randn(B, 32)

        g_t = tm.forward_sample(g, t)
        ts_t, noise_true = ddpm.forward_sample(ts_clean, t)
        o_t = order_tm.forward_sample(o_clean, t)
        o_t_oh = F.one_hot(o_t, num_classes=3).float()

        g_logits, ts_pred, o_logits = small_model(g_t, ts_t, o_t_oh, t, cond)
        total, _ = small_model.compute_loss(
            g_logits, ts_pred, o_logits,
            g, g_t, t,
            noise_true, o_clean, o_t,
            tm, order_tm, ddpm,
        )
        total.backward()
        for p in small_model.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# EMAWrapper
# ---------------------------------------------------------------------------

class TestEMAWrapper:
    def test_shadow_tracks_model(self, small_model):
        ema = EMAWrapper(small_model, decay=0.9)
        # Modify model weights
        with torch.no_grad():
            for p in small_model.parameters():
                p.add_(1.0)
        shadow_before = {n: p.clone() for n, p in ema.shadow.named_parameters()}
        ema.update(small_model)
        # Shadow should have moved toward new model params
        for name, p_after in ema.shadow.named_parameters():
            p_before = shadow_before[name]
            assert not torch.allclose(p_after, p_before), (
                f"EMA shadow did not update for param {name}"
            )

    def test_shadow_no_grad(self, small_model):
        ema = EMAWrapper(small_model, decay=0.9999)
        for p in ema.shadow.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# guided_sample (smoke test)
# ---------------------------------------------------------------------------

class TestGuidedSample:
    def test_output_shapes(self, small_model, tm, ddpm):
        small_model.eval()
        order_tm = UniformTransitionMatrix(K=3, T=20, beta_schedule="linear")
        condition = torch.randn(1, 32)
        g, ts, o = guided_sample(
            model=small_model,
            condition=condition,
            n_terms=6,
            max_groups=4,
            transition_matrix=tm,
            order_transition_matrix=order_tm,
            ddpm=ddpm,
            guidance_scale=1.5,
            n_steps=5,  # fast smoke test
        )
        assert g.shape == (1, 6)
        assert ts.shape == (1, 4)
        assert o.shape == (1, 4)
        assert (g >= 0).all() and (g < 4).all()
        assert (o >= 0).all() and (o < 3).all()
