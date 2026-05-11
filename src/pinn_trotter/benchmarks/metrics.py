"""Benchmark metric helpers for baseline and model evaluation."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from pinn_trotter.strategy.circuit_builder import strategy_to_circuit
from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


def _default_psi0(n_qubits: int) -> np.ndarray:
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0
    return psi


def exact_fidelity(
    strategy: TrotterStrategy,
    hamiltonian,
    psi_0: np.ndarray | None = None,
) -> float:
    """Compute exact fidelity F = |<psi_exact | psi_trotter>|^2."""
    from pinn_trotter.data.generator import compute_exact_fidelity_from_hamiltonian

    psi = _default_psi0(hamiltonian.n_qubits) if psi_0 is None else psi_0
    fidelity, _ = compute_exact_fidelity_from_hamiltonian(hamiltonian, strategy, psi)
    return float(np.clip(fidelity, 0.0, 1.0))


def transpiled_depth(strategy: TrotterStrategy, hamiltonian) -> int:
    """Transpiled circuit depth under unified comparison gate set."""
    from qiskit import transpile

    circuit = strategy_to_circuit(strategy, hamiltonian)
    transpiled = transpile(
        circuit,
        basis_gates=["h", "cx", "rz", "x"],
        optimization_level=1,
    )
    return int(transpiled.depth())


def cx_count(strategy: TrotterStrategy, hamiltonian) -> int:
    """Count CX gates after transpilation under unified basis."""
    from qiskit import transpile

    circuit = strategy_to_circuit(strategy, hamiltonian)
    transpiled = transpile(
        circuit,
        basis_gates=["h", "cx", "rz", "x"],
        optimization_level=1,
    )
    return int(transpiled.count_ops().get("cx", 0))


def inference_latency(
    model: Callable,
    hamiltonian,
    n_trials: int = 100,
) -> float:
    """Average model inference latency in seconds (excluding warm-up call)."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    # Warm-up pass (JIT/cache/etc.) not counted.
    _ = model(hamiltonian)

    timings: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = model(hamiltonian)
        timings.append(time.perf_counter() - t0)
    return float(np.mean(timings))


def pareto_hypervolume(
    fidelities: list[float] | np.ndarray,
    depths: list[int] | np.ndarray,
    ref_point: tuple[float, float],
) -> float:
    """Compute Pareto hypervolume in (maximize fidelity, minimize depth) space."""
    f = np.asarray(fidelities, dtype=float)
    d = np.asarray(depths, dtype=float)
    if f.shape != d.shape:
        raise ValueError("fidelities and depths must have the same shape")
    if f.size == 0:
        return 0.0

    ref_fid, ref_depth = ref_point
    try:
        from pymoo.indicators.hv import HV

        # pymoo assumes minimization: maximize fidelity -> minimize -fidelity.
        points = np.stack([-f, d], axis=1)
        hv = HV(ref_point=np.array([-ref_fid, ref_depth], dtype=float))
        return float(hv(points))
    except ImportError:
        # Fallback: 2D sweep using a Pareto front tracker implementation.
        from pinn_trotter.optimizer.pareto import ParetoTracker

        tracker = ParetoTracker(ref_depth=float(ref_depth))
        tracker.update(f.tolist(), d.astype(int).tolist())
        return float(tracker.hypervolume(ref_fidelity=float(ref_fid)))


def wilcoxon_test(
    our_scores: list[float] | np.ndarray,
    baseline_scores: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test (alternative: our_scores > baseline_scores)."""
    from scipy.stats import wilcoxon

    ours = np.asarray(our_scores, dtype=float)
    base = np.asarray(baseline_scores, dtype=float)
    if ours.shape != base.shape:
        raise ValueError("our_scores and baseline_scores must have the same shape")
    if ours.size == 0:
        raise ValueError("scores must be non-empty")

    stat, p_value = wilcoxon(ours, base, alternative="greater")
    return float(stat), float(p_value)
