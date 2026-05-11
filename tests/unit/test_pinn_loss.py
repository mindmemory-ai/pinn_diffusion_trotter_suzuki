"""Unit tests for PINN loss, network, and fidelity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinn_trotter.pinn.fidelity import evaluate_fidelity_proxy, fidelity_from_states
from pinn_trotter.pinn.loss import _batch_jacobian, compute_pinn_loss
from pinn_trotter.pinn.network import FourierFeatureEmbedding, PINNNetwork


# ---------------------------------------------------------------------------
# FourierFeatureEmbedding
# ---------------------------------------------------------------------------

class TestFourierFeatureEmbedding:
    def test_output_shape(self):
        emb = FourierFeatureEmbedding(m=32, sigma=1.0)
        t = torch.linspace(0, 1, 10)
        out = emb(t)
        assert out.shape == (10, 64)

    def test_scalar_input(self):
        emb = FourierFeatureEmbedding(m=16, sigma=1.0)
        t = torch.tensor(0.5)
        out = emb(t)
        assert out.shape == (32,)

    def test_B_frozen(self):
        """B is a buffer (not a parameter), so it has no grad and won't be updated."""
        emb = FourierFeatureEmbedding(m=8, sigma=1.0)
        B_before = emb.B.clone()
        # B must not appear in parameters()
        param_names = [n for n, _ in emb.named_parameters()]
        assert "B" not in param_names
        # B must not have requires_grad
        assert not emb.B.requires_grad
        # Value unchanged after forward pass
        emb(torch.tensor(0.5))
        assert torch.allclose(emb.B, B_before)


# ---------------------------------------------------------------------------
# PINNNetwork
# ---------------------------------------------------------------------------

class TestPINNNetwork:
    @pytest.fixture
    def net_1q(self):
        return PINNNetwork(n_qubits=1, fourier_m=8, hidden_dim=32, hamiltonian_norm=1.0)

    @pytest.fixture
    def net_2q(self):
        return PINNNetwork(n_qubits=2, fourier_m=16, hidden_dim=64, hamiltonian_norm=2.0)

    def test_output_shape_1q(self, net_1q):
        t = torch.tensor(0.5)
        out = net_1q(t)
        assert out.shape == (2, 2)  # (dim, 2) for re/im

    def test_output_shape_2q_batch(self, net_2q):
        t = torch.linspace(0, 1, 5)
        out = net_2q(t)
        assert out.shape == (5, 4, 2)

    def test_as_complex_shape(self, net_2q):
        t = torch.linspace(0, 1, 3)
        psi = net_2q.as_complex(t)
        assert psi.shape == (3, 4)
        assert psi.is_complex()

    def test_normalization_penalty_untrained(self, net_1q):
        t = torch.linspace(0, 1, 10)
        penalty = net_1q.normalization_penalty(t)
        assert penalty.shape == ()  # scalar
        assert penalty.item() >= 0.0

    def test_output_smooth_in_t(self, net_1q):
        """Network should be smooth: close t values give close outputs."""
        t1 = torch.tensor(0.5)
        t2 = torch.tensor(0.5001)
        out1 = net_1q(t1)
        out2 = net_1q(t2)
        assert torch.allclose(out1, out2, atol=0.1)


# ---------------------------------------------------------------------------
# batch Jacobian
# ---------------------------------------------------------------------------

class TestBatchJacobian:
    def test_known_gradient(self):
        """d(sin(t*k)) / dt = k*cos(t*k)."""
        N, D = 5, 3
        t = torch.rand(N, requires_grad=True)
        k = torch.arange(1, D + 1, dtype=torch.float32)
        output = torch.sin(t.unsqueeze(1) * k)  # (N, D)
        jac = _batch_jacobian(output, t)          # (N, D)
        expected = torch.cos(t.detach().unsqueeze(1) * k) * k
        assert torch.allclose(jac.detach(), expected, atol=1e-5)

    def test_shape(self):
        t = torch.rand(8, requires_grad=True)
        out = t.unsqueeze(1).expand(-1, 4) ** 2
        jac = _batch_jacobian(out, t)
        assert jac.shape == (8, 4)


# ---------------------------------------------------------------------------
# compute_pinn_loss
# ---------------------------------------------------------------------------

class TestComputePinnLoss:
    @pytest.fixture
    def simple_setup(self):
        """1-qubit Hamiltonian: H = σ_z = diag(1, -1). Exact solution known."""
        n = 1
        dim = 2
        H = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
        psi_0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)  # |0⟩
        net = PINNNetwork(n_qubits=n, fourier_m=8, hidden_dim=32, hamiltonian_norm=1.0)
        return net, H, psi_0, dim

    def test_loss_keys(self, simple_setup):
        net, H, psi_0, _ = simple_setup
        t_c = torch.linspace(0.01, 1.0, 5)
        total, losses = compute_pinn_loss(net, H, psi_0, t_c)
        assert set(losses.keys()) == {'ic', 'pde', 'circuit', 'norm'}

    def test_total_is_positive(self, simple_setup):
        net, H, psi_0, _ = simple_setup
        t_c = torch.linspace(0.01, 1.0, 5)
        total, _ = compute_pinn_loss(net, H, psi_0, t_c)
        assert float(total.detach()) >= 0.0

    def test_circuit_term_zero_when_absent(self, simple_setup):
        net, H, psi_0, _ = simple_setup
        t_c = torch.linspace(0.01, 1.0, 5)
        _, losses = compute_pinn_loss(net, H, psi_0, t_c)
        assert float(losses['circuit'].detach()) == 0.0

    def test_circuit_term_active(self, simple_setup):
        net, H, psi_0, dim = simple_setup
        t_c = torch.linspace(0.01, 1.0, 5)
        t_circ = torch.tensor([0.5, 1.0])
        psi_circ = torch.zeros(2, dim, dtype=torch.complex128)
        psi_circ[0, 0] = 1.0
        psi_circ[1, 0] = 1.0
        _, losses = compute_pinn_loss(net, H, psi_0, t_c, t_circ, psi_circ)
        assert float(losses['circuit'].detach()) >= 0.0

    def test_backward_runs(self, simple_setup):
        """Full backward pass should not raise."""
        net, H, psi_0, _ = simple_setup
        t_c = torch.linspace(0.01, 1.0, 4)
        total, _ = compute_pinn_loss(net, H, psi_0, t_c)
        total.backward()
        for p in net.parameters():
            assert p.grad is not None

    def test_real_H_accepted(self, simple_setup):
        """Real (float64) Hamiltonian should be cast internally."""
        net, _, psi_0, _ = simple_setup
        H_real = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        t_c = torch.linspace(0.01, 1.0, 4)
        total, _ = compute_pinn_loss(net, H_real, psi_0, t_c)
        assert float(total.detach()) >= 0.0


# ---------------------------------------------------------------------------
# fidelity utilities
# ---------------------------------------------------------------------------

class TestFidelity:
    def test_perfect_fidelity(self):
        psi = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex128)
        f = fidelity_from_states(psi.unsqueeze(0), psi.unsqueeze(0))
        assert torch.isclose(f, torch.ones(1, dtype=torch.float64), atol=1e-6)

    def test_orthogonal_zero_fidelity(self):
        psi_a = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        psi_b = torch.tensor([0.0, 1.0], dtype=torch.complex128)
        f = fidelity_from_states(psi_a.unsqueeze(0), psi_b.unsqueeze(0))
        assert torch.isclose(f, torch.zeros(1, dtype=torch.float64), atol=1e-6)

    def test_batch_fidelity_shape(self):
        psi = torch.randn(5, 4, dtype=torch.complex64)
        f = fidelity_from_states(psi, psi)
        assert f.shape == (5,)
        assert (f >= 0).all()

    def test_evaluate_fidelity_proxy(self):
        n = 1
        net = PINNNetwork(n_qubits=n, fourier_m=8, hidden_dim=32)
        psi_target = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        f = evaluate_fidelity_proxy(net, psi_target, t_eval=0.5)
        assert 0.0 <= f <= 1.0
