"""Strategy and Hamiltonian parameter sampling functions."""

from __future__ import annotations

import numpy as np

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


def sample_tfim_params(
    n_samples: int,
    rng: np.random.Generator,
    h_min: float = 0.1,
    h_max: float = 5.0,
    j_min: float = 0.1,
    j_max: float = 5.0,
    n_qubits_fixed: int | None = None,
    t_min: float = 0.5,
    t_max: float = 10.0,
) -> list[dict]:
    """Sample TFIM Hamiltonian parameters.

    n_qubits distribution: 4:6:8 = 6:3:1 (or fixed if n_qubits_fixed is set)
    J ~ LogUniform(j_min, j_max)
    h ~ LogUniform(h_min, h_max)
    t_final ~ Uniform(t_min, t_max)
    """
    params = []
    n_qubit_choices = [4, 6, 8]
    n_qubit_probs = [0.6, 0.3, 0.1]

    for _ in range(n_samples):
        if n_qubits_fixed is not None:
            n_qubits = int(n_qubits_fixed)
        else:
            n_qubits = int(rng.choice(n_qubit_choices, p=n_qubit_probs))
        J = float(np.exp(rng.uniform(np.log(j_min), np.log(j_max))))
        h = float(np.exp(rng.uniform(np.log(h_min), np.log(h_max))))
        t_final = float(rng.uniform(t_min, t_max))
        params.append({
            "hamiltonian_type": "tfim",
            "n_qubits": n_qubits,
            "J": J,
            "h": h,
            "t_final": t_final,
            "boundary": "periodic",
        })
    return params


def sample_random_strategy(
    n_terms: int,
    n_groups_max: int,
    t_total: float,
    n_qubits: int,
    rng: np.random.Generator,
) -> TrotterStrategy:
    """Sample a random TrotterStrategy.

    Grouping: each term randomly assigned to one of K groups (K ~ Uniform[1, n_groups_max]).
    Time steps: Dirichlet(1) normalized to sum to t_total.
    Orders: randomly drawn from {1, 2, 4} with probs [0.2, 0.6, 0.2].

    Args:
        n_terms: Number of Pauli terms in the Hamiltonian.
        n_groups_max: Maximum number of groups K.
        t_total: Total evolution time.
        n_qubits: Number of qubits.
        rng: Random number generator.
    """
    K = int(rng.integers(1, n_groups_max + 1))

    # Random assignment ensuring all groups non-empty:
    # First assign one term to each group (shuffle), then assign remainder randomly
    order_choices = [1, 2, 4]
    order_probs = [0.2, 0.6, 0.2]

    perm = rng.permutation(n_terms).tolist()
    grouping: list[list[int]] = [[] for _ in range(K)]
    # Seed one term per group
    for g in range(min(K, n_terms)):
        grouping[g].append(perm[g])
    # Distribute remaining
    for term_idx in perm[K:]:
        g = int(rng.integers(0, K))
        grouping[g].append(term_idx)

    # Remove empty groups (shouldn't happen with above logic, but defensive)
    grouping = [g for g in grouping if g]
    K = len(grouping)

    # Time steps via Dirichlet(1) = uniform on simplex
    ts = rng.dirichlet(np.ones(K))
    time_steps = (ts * t_total).tolist()

    # Fix floating point precision
    time_steps[-1] = t_total - sum(time_steps[:-1])

    # Orders
    orders = [
        int(rng.choice(order_choices, p=order_probs))
        for _ in range(K)
    ]

    return TrotterStrategy(
        grouping=grouping,
        orders=orders,
        time_steps=time_steps,
        n_qubits=n_qubits,
        n_terms=n_terms,
        t_total=t_total,
    )


def _detect_commuting_groups(pauli_strings: list[str]) -> list[list[int]]:
    """Partition Pauli terms into maximally commuting groups using a greedy graph colouring.

    Two Pauli strings P and Q commute iff the number of positions where they differ
    (both non-identity but different) is even.
    """
    n = len(pauli_strings)

    def commutes(p: str, q: str) -> bool:
        anticommute_count = sum(
            1 for a, b in zip(p, q)
            if a != 'I' and b != 'I' and a != b
        )
        return anticommute_count % 2 == 0

    # Build adjacency (two terms are in conflict = anticommute → different groups)
    groups: list[list[int]] = []
    term_group: list[int] = [-1] * n

    for i in range(n):
        placed = False
        for g_idx, group in enumerate(groups):
            if all(commutes(pauli_strings[i], pauli_strings[j]) for j in group):
                group.append(i)
                term_group[i] = g_idx
                placed = True
                break
        if not placed:
            term_group[i] = len(groups)
            groups.append([i])

    return groups


def sample_smart_strategy(
    pauli_strings: list[str],
    n_groups_max: int,
    t_total: float,
    n_qubits: int,
    rng: np.random.Generator,
    split_prob: float = 0.3,
) -> TrotterStrategy:
    """Sample a physics-informed TrotterStrategy using commutation structure.

    Groups are formed by greedily partitioning terms into commuting subsets.
    To reduce inter-group Trotter error, each commuting group is then further
    split into single-term sub-groups (maximising K) before optionally merging
    some back to add training diversity.

    Args:
        pauli_strings: Pauli string representation of the Hamiltonian.
        n_groups_max: Maximum number of groups.
        t_total: Total evolution time.
        n_qubits: Number of qubits.
        rng: Random number generator.
        split_prob: Probability of randomly merging two groups for diversity.
    """
    n_terms = len(pauli_strings)

    # Start from maximally granular grouping: each term in its own group.
    # This minimises per-step time and thus Trotter error.
    fine_groups: list[list[int]] = [[i] for i in range(n_terms)]

    # Optionally merge a random pair to add training diversity
    if rng.random() < split_prob and len(fine_groups) > 1:
        idx_a = int(rng.integers(0, len(fine_groups)))
        idx_b = int(rng.integers(0, len(fine_groups)))
        while idx_b == idx_a:
            idx_b = int(rng.integers(0, len(fine_groups)))
        merged = fine_groups.pop(max(idx_a, idx_b)) + fine_groups.pop(min(idx_a, idx_b))
        fine_groups.append(merged)

    # Ensure K ≤ n_groups_max (merge smallest groups if needed)
    while len(fine_groups) > n_groups_max:
        fine_groups.sort(key=len)
        merged = fine_groups.pop(0) + fine_groups.pop(0)
        fine_groups.append(merged)

    K = len(fine_groups)

    # Near-uniform time steps (Dirichlet with higher concentration → peaks at uniform)
    alpha = float(rng.choice([1.0, 3.0, 5.0]))
    ts = rng.dirichlet(np.full(K, alpha))
    time_steps = (ts * t_total).tolist()
    time_steps[-1] = t_total - sum(time_steps[:-1])

    # Favour higher-order Trotter for better accuracy
    order_choices = [1, 2, 4]
    order_probs = [0.05, 0.55, 0.40]
    orders = [int(rng.choice(order_choices, p=order_probs)) for _ in range(K)]

    return TrotterStrategy(
        grouping=fine_groups,
        orders=orders,
        time_steps=time_steps,
        n_qubits=n_qubits,
        n_terms=n_terms,
        t_total=t_total,
    )

