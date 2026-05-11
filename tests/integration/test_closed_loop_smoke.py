"""Integration smoke test: end-to-end closed-loop optimization pipeline.

Runs a minimal version of the full pipeline:
  GNN encode → diffusion sample → evaluate → REINFORCE update → Pareto update.

Does NOT require a dataset on disk; Hamiltonians are synthesised from the
benchmark factory.  Designed to catch cross-module wiring regressions.
"""

from __future__ import annotations

import math

import pytest
import torch

from pinn_trotter.benchmarks.hamiltonians import make_tfim
from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
from pinn_trotter.diffusion.mixed_model import MixedDiffusionModel
from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder
from pinn_trotter.optimizer.closed_loop import ClosedLoopOptimizer


N_QUBITS = 4
N_TERMS = 2 * N_QUBITS       # TFIM: 4 ZZ + 4 X
MAX_GROUPS = 4
T_DIFFUSION = 10             # tiny T for speed
BATCH_SIZE = 2
N_ITER = 3


@pytest.fixture(scope="module")
def components():
    gnn = HamiltonianGNNEncoder(
        node_feat_dim=N_QUBITS + 2,
        edge_feat_dim=3,
        hidden_dim=32,
        output_dim=64,
        n_layers=2,
    )
    diffusion = MixedDiffusionModel(
        max_groups=MAX_GROUPS,
        n_terms=N_TERMS,
        condition_dim=64,
        fused_dim=32,
        time_embed_dim=16,
        grouping_layers=1,
        order_layers=1,
        ts_mlp_layers=2,
        p_cond_drop=0.1,
    )
    tm = UniformTransitionMatrix(K=MAX_GROUPS, T=T_DIFFUSION, beta_schedule="cosine")
    order_tm = UniformTransitionMatrix(K=3, T=T_DIFFUSION, beta_schedule="cosine")
    ddpm = ContinuousDDPM(T=T_DIFFUSION, beta_schedule="cosine")
    return gnn, diffusion, tm, order_tm, ddpm


@pytest.fixture(scope="module")
def optimizer_and_sampler(components, tmp_path_factory):
    gnn, diffusion, tm, order_tm, ddpm = components
    tmp = tmp_path_factory.mktemp("ckpts")

    def _dummy_pinn(H_graph, grouping, timesteps, orders):
        return float(torch.rand(1).item() * 0.5 + 0.5)

    opt = ClosedLoopOptimizer(
        diffusion_model=diffusion,
        gnn_encoder=gnn,
        transition_matrix=tm,
        order_transition_matrix=order_tm,
        ddpm=ddpm,
        pinn_evaluator=_dummy_pinn,
        lambda_weight=0.05,
        batch_size=BATCH_SIZE,
        n_terms=N_TERMS,
        max_groups=MAX_GROUPS,
        checkpoint_dir=tmp,
        checkpoint_interval=100,   # no checkpoint during short smoke test
        device="cpu",
    )

    rng_state = [0]

    def sampler():
        rng_state[0] += 1
        torch.manual_seed(rng_state[0])
        return [make_tfim(N_QUBITS, J=float(torch.rand(1) + 0.5),
                          h=float(torch.rand(1) + 0.1))
                for _ in range(BATCH_SIZE)]

    return opt, sampler


def test_train_runs(optimizer_and_sampler):
    """Training loop executes without error and returns history dicts."""
    opt, sampler = optimizer_and_sampler
    history = opt.train(N_ITER, sampler)
    assert len(history["policy_loss"]) == N_ITER
    assert len(history["mean_fidelity"]) == N_ITER
    assert len(history["pareto_hv"]) == N_ITER


def test_loss_is_finite(optimizer_and_sampler):
    """Policy loss values are finite after training."""
    opt, sampler = optimizer_and_sampler
    history = opt.train(2, sampler)
    for loss in history["policy_loss"]:
        assert math.isfinite(loss), f"Non-finite policy loss: {loss}"


def test_fidelity_in_range(optimizer_and_sampler):
    """Sampled fidelities are in [0, 1]."""
    opt, sampler = optimizer_and_sampler
    history = opt.train(2, sampler)
    for f in history["mean_fidelity"]:
        assert 0.0 <= f <= 1.0, f"Fidelity out of range: {f}"


def test_pareto_updated(optimizer_and_sampler):
    """Pareto front has at least one point after training."""
    opt, sampler = optimizer_and_sampler
    # pareto may already have points from previous fixtures sharing the same opt
    assert len(opt.pareto) >= 1


def test_lambda_sweep_returns_all_lambdas(components, tmp_path):
    """Lambda sweep returns a result for every lambda value."""
    gnn, diffusion, tm, order_tm, ddpm = components

    def _pinn(H, g, ts, o):
        return 0.7

    opt = ClosedLoopOptimizer(
        diffusion_model=diffusion,
        gnn_encoder=gnn,
        transition_matrix=tm,
        order_transition_matrix=order_tm,
        ddpm=ddpm,
        pinn_evaluator=_pinn,
        lambda_weight=0.1,
        batch_size=BATCH_SIZE,
        n_terms=N_TERMS,
        max_groups=MAX_GROUPS,
        checkpoint_dir=tmp_path,
        checkpoint_interval=100,
        device="cpu",
    )

    rng = [0]

    def sampler():
        rng[0] += 1
        return [make_tfim(N_QUBITS) for _ in range(BATCH_SIZE)]

    lambdas = [0.0, 0.1, 0.5]
    results = opt.lambda_sweep(lambdas, n_iterations=2, hamiltonian_sampler=sampler)
    assert set(results.keys()) == set(lambdas)
    for lam, res in results.items():
        assert "history" in res
        assert "pareto_front" in res
        assert "hypervolume" in res
