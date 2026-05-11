"""Optimizer module: gradient utilities, Pareto tracking, closed-loop training."""

from pinn_trotter.optimizer.closed_loop import ClosedLoopOptimizer
from pinn_trotter.optimizer.gradient_utils import (
    compute_policy_log_prob,
    gumbel_softmax,
    straight_through_estimator,
)
from pinn_trotter.optimizer.pareto import ParetoTracker

__all__ = [
    "gumbel_softmax",
    "straight_through_estimator",
    "compute_policy_log_prob",
    "ParetoTracker",
    "ClosedLoopOptimizer",
]
