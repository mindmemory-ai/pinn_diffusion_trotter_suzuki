"""Unit tests for benchmark metrics."""

from __future__ import annotations

import numpy as np
import pytest

from pinn_trotter.benchmarks.hamiltonians import make_tfim
from pinn_trotter.benchmarks.metrics import (
    cx_count,
    exact_fidelity,
    inference_latency,
    pareto_hypervolume,
    transpiled_depth,
    wilcoxon_test,
)
from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


@pytest.fixture()
def tfim4():
    return make_tfim(4, J=1.0, h=0.5, boundary="periodic")


@pytest.fixture()
def simple_strategy(tfim4):
    return TrotterStrategy(
        grouping=[list(range(tfim4.n_terms))],
        orders=[2],
        time_steps=[0.2],
        n_qubits=tfim4.n_qubits,
        n_terms=tfim4.n_terms,
        t_total=0.2,
    )


def test_exact_fidelity_in_unit_interval(tfim4, simple_strategy):
    f = exact_fidelity(simple_strategy, tfim4)
    assert 0.0 <= f <= 1.0


def test_transpiled_depth_and_cx_count_non_negative(tfim4, simple_strategy):
    depth = transpiled_depth(simple_strategy, tfim4)
    cxs = cx_count(simple_strategy, tfim4)
    assert isinstance(depth, int)
    assert isinstance(cxs, int)
    assert depth >= 0
    assert cxs >= 0


def test_inference_latency_positive(tfim4):
    def dummy_model(H):
        return H.n_terms

    lat = inference_latency(dummy_model, tfim4, n_trials=5)
    assert lat >= 0.0


def test_pareto_hypervolume_non_negative():
    hv = pareto_hypervolume(
        fidelities=[0.7, 0.9, 0.8],
        depths=[70, 40, 50],
        ref_point=(0.0, 100.0),
    )
    assert hv >= 0.0


def test_wilcoxon_test_returns_valid_tuple():
    our = [0.91, 0.88, 0.95, 0.93, 0.90]
    base = [0.85, 0.84, 0.90, 0.89, 0.87]
    stat, p = wilcoxon_test(our, base)
    assert stat >= 0.0
    assert 0.0 <= p <= 1.0


def test_wilcoxon_shape_mismatch_raises():
    with pytest.raises(ValueError):
        wilcoxon_test([0.9, 0.8], [0.7])
