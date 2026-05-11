"""Dataset generation: exact fidelity computation and Trotter evolution."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Literal

import h5py
import numpy as np
import scipy.linalg
import scipy.integrate
import scipy.sparse as sp
from tqdm import tqdm

from pinn_trotter.strategy.trotter_strategy import TrotterStrategy


def _apply_trotter_step_order1(
    H_terms: list[sp.csr_matrix],
    tau: float,
    psi: np.ndarray,
) -> np.ndarray:
    """Apply one first-order Trotter step: ∏_j exp(-i H_j τ)."""
    for H_j in H_terms:
        # exp(-i H_j τ) via Padé approximation (expm on sparse → dense)
        dim = H_j.shape[0]
        U = scipy.linalg.expm(-1j * tau * H_j.toarray())
        psi = U @ psi
    return psi


def _apply_trotter_step_order2(
    H_terms: list[sp.csr_matrix],
    tau: float,
    psi: np.ndarray,
) -> np.ndarray:
    """Apply one second-order symmetric Trotter step S₂(τ)."""
    half = tau / 2.0
    for H_j in H_terms:
        U = scipy.linalg.expm(-1j * half * H_j.toarray())
        psi = U @ psi
    for H_j in reversed(H_terms):
        U = scipy.linalg.expm(-1j * half * H_j.toarray())
        psi = U @ psi
    return psi


def _apply_trotter_step_order4(
    H_terms: list[sp.csr_matrix],
    tau: float,
    psi: np.ndarray,
) -> np.ndarray:
    """Apply one fourth-order Suzuki step S₄(τ) using the recursive construction."""
    p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    sub_steps = [p, p, 1.0 - 4.0 * p, p, p]
    for s in sub_steps:
        psi = _apply_trotter_step_order2(H_terms, s * tau, psi)
    return psi


def apply_trotter_sparse(
    H_sparse,
    strategy: TrotterStrategy,
    psi_0: np.ndarray,
) -> np.ndarray:
    """Not implemented: sparse-only path requires term-by-term HamiltonianGraph.

    Raises:
        NotImplementedError: Always. Use apply_trotter_from_hamiltonian instead.
    """
    raise NotImplementedError(
        "apply_trotter_sparse requires per-term matrices from HamiltonianGraph. "
        "Use apply_trotter_from_hamiltonian(hamiltonian, strategy, psi_0) instead."
    )


def apply_trotter_from_hamiltonian(
    hamiltonian: object,
    strategy: TrotterStrategy,
    psi_0: np.ndarray,
) -> np.ndarray:
    """Simulate Trotter evolution given a HamiltonianGraph.

    Args:
        hamiltonian: HamiltonianGraph instance.
        strategy: The Trotter decomposition strategy.
        psi_0: Initial state, shape (2^n,).

    Returns:
        Final state after Trotter evolution.
    """
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph
    H: HamiltonianGraph = hamiltonian  # type: ignore[assignment]

    psi = psi_0.astype(complex).copy()

    for group, order, tau in zip(strategy.grouping, strategy.orders, strategy.time_steps):
        H_terms = [
            sp.csr_matrix(H.coefficients[idx] * _pauli_to_matrix(H.pauli_strings[idx]))
            for idx in group
        ]
        if order == 1:
            psi = _apply_trotter_step_order1(H_terms, tau, psi)
        elif order == 2:
            psi = _apply_trotter_step_order2(H_terms, tau, psi)
        elif order == 4:
            psi = _apply_trotter_step_order4(H_terms, tau, psi)

    return psi


def _pauli_to_matrix(s: str) -> np.ndarray:
    from pinn_trotter.hamiltonian.pauli_utils import pauli_string_to_matrix
    return pauli_string_to_matrix(s)


def compute_exact_fidelity(
    H_sparse,
    strategy: TrotterStrategy,
    psi_0: np.ndarray,
    n_qubits: int,
) -> tuple[float, float]:
    """Deprecated: this path cannot compute Trotter evolution without HamiltonianGraph.

    Raises:
        NotImplementedError: Always. Use compute_exact_fidelity_from_hamiltonian.
    """
    raise NotImplementedError(
        "compute_exact_fidelity cannot apply Trotter steps without per-term matrices. "
        "Use compute_exact_fidelity_from_hamiltonian(hamiltonian, strategy, psi_0) instead."
    )


def compute_exact_fidelity_from_hamiltonian(
    hamiltonian: object,
    strategy: TrotterStrategy,
    psi_0: np.ndarray,
    exact_method: Literal["auto", "expm", "rk45"] = "auto",
) -> tuple[float, float]:
    """Compute exact fidelity given a HamiltonianGraph.

    Args:
        hamiltonian: HamiltonianGraph instance.
        strategy: Trotter strategy to evaluate.
        psi_0: Initial state.

    Returns:
        (fidelity, trotter_norm): Fidelity F = |⟨ψ_exact|ψ_Trotter⟩|² and Trotter state norm.
    """
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph
    H: HamiltonianGraph = hamiltonian  # type: ignore[assignment]

    n = H.n_qubits
    T = strategy.t_total
    psi_0_c = psi_0.astype(complex)
    H_sparse = H.to_sparse_matrix()

    # Exact evolution
    method = exact_method
    if method == "auto":
        method = "expm" if n <= 8 else "rk45"

    if method == "expm":
        U_exact = scipy.linalg.expm(-1j * T * H_sparse.toarray())
        psi_exact = U_exact @ psi_0_c
    elif method == "rk45":
        dim = 2**n
        H_arr = H_sparse.toarray()

        def schrodinger(t: float, y: np.ndarray) -> np.ndarray:
            psi = y[:dim] + 1j * y[dim:]
            dpsi = -1j * (H_arr @ psi)
            return np.concatenate([dpsi.real, dpsi.imag])

        y0 = np.concatenate([psi_0_c.real, psi_0_c.imag])
        sol = scipy.integrate.solve_ivp(
            schrodinger, [0, T], y0, method="RK45", rtol=1e-10, atol=1e-12
        )
        psi_exact = sol.y[:dim, -1] + 1j * sol.y[dim:, -1]
    else:
        raise ValueError(f"Unknown exact_method: {exact_method}")

    # Trotter evolution
    psi_trotter = apply_trotter_from_hamiltonian(hamiltonian, strategy, psi_0)
    trotter_norm = float(np.linalg.norm(psi_trotter))

    # Fidelity
    overlap = np.vdot(psi_exact, psi_trotter)
    fidelity = float(abs(overlap) ** 2 / (np.linalg.norm(psi_exact) ** 2 * trotter_norm**2))

    return fidelity, trotter_norm


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0"
_GENERATOR_VERSION = "0.1.0"


def write_sample_to_hdf5(
    h5file: h5py.File,
    sample_id: int,
    hamiltonian: object,
    strategy: TrotterStrategy,
    psi_0: np.ndarray,
    labels: dict[str, Any],
    hamiltonian_params: dict[str, Any] | None = None,
) -> None:
    """Write one sample to an open HDF5 file under /dataset_v1/samples/{sample_id:07d}/.

    Args:
        h5file: Open h5py File in write/append mode.
        sample_id: Integer sample index (zero-padded to 7 digits as group name).
        hamiltonian: HamiltonianGraph instance.
        strategy: TrotterStrategy instance.
        psi_0: Initial state vector, shape (2^n,).
        labels: Dict with keys: fidelity, circuit_depth, cx_count,
                total_gate_count, has_negative_timestep.
        hamiltonian_params: Optional dict with keys: hamiltonian_type, J, h,
                            bond_length, boundary.
    """
    from pinn_trotter.hamiltonian.hamiltonian_graph import HamiltonianGraph

    H: HamiltonianGraph = hamiltonian  # type: ignore[assignment]
    grp_name = f"{sample_id:07d}"
    samples_grp = h5file.require_group("dataset_v1/samples")
    grp = samples_grp.require_group(grp_name)

    # /hamiltonian/
    h_grp = grp.require_group("hamiltonian")
    dt = h5py.special_dtype(vlen=str)
    h_grp.create_dataset(
        "pauli_strings",
        data=np.array(H.pauli_strings, dtype=object),
        dtype=dt,
    )
    h_grp.create_dataset("coefficients", data=H.coefficients.astype(np.float64))
    h_grp.attrs["n_qubits"] = np.int16(H.n_qubits)
    h_grp.attrs["n_terms"] = np.int16(H.n_terms)
    if hamiltonian_params is not None:
        h_grp.attrs["hamiltonian_type"] = hamiltonian_params.get("hamiltonian_type", "unknown")
        h_grp.attrs["J"] = float(hamiltonian_params.get("J", float("nan")))
        h_grp.attrs["h"] = float(hamiltonian_params.get("h", float("nan")))
        h_grp.attrs["bond_length"] = float(hamiltonian_params.get("bond_length", float("nan")))
        h_grp.attrs["boundary"] = hamiltonian_params.get("boundary", "periodic")

    # /strategy/
    K = len(strategy.grouping)
    M = strategy.n_terms
    grouping_flat = np.zeros(M, dtype=np.int16)
    for g_idx, group in enumerate(strategy.grouping):
        for term_idx in group:
            grouping_flat[term_idx] = g_idx

    s_grp = grp.require_group("strategy")
    s_grp.create_dataset("grouping_flat", data=grouping_flat)
    s_grp.attrs["n_groups"] = np.int16(K)
    s_grp.create_dataset("orders", data=np.array(strategy.orders, dtype=np.int8))
    s_grp.create_dataset("time_steps", data=np.array(strategy.time_steps, dtype=np.float64))
    s_grp.attrs["t_total"] = float(strategy.t_total)

    # /initial_state/
    i_grp = grp.require_group("initial_state")
    i_grp.create_dataset("psi_0_real", data=psi_0.real.astype(np.float64))
    i_grp.create_dataset("psi_0_imag", data=psi_0.imag.astype(np.float64))
    i_grp.attrs["state_type"] = "computational_basis_0"

    # /labels/
    l_grp = grp.require_group("labels")
    l_grp.attrs["fidelity"] = float(labels["fidelity"])
    l_grp.attrs["circuit_depth"] = int(labels.get("circuit_depth", -1))
    l_grp.attrs["cx_count"] = int(labels.get("cx_count", -1))
    l_grp.attrs["total_gate_count"] = int(labels.get("total_gate_count", -1))
    l_grp.attrs["has_negative_timestep"] = bool(labels.get("has_negative_timestep", False))


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

_MAX_CIRCUIT_DEPTH = 10_000


def _passes_quality_filter(fidelity: float, circuit_depth: int) -> bool:
    """Return True if this sample should be kept in the dataset."""
    if fidelity < 0.001:
        return False
    if circuit_depth > _MAX_CIRCUIT_DEPTH:
        return False
    return True


# ---------------------------------------------------------------------------
# Worker function (top-level for pickling)
# ---------------------------------------------------------------------------

def _generate_single_sample(args: tuple) -> dict | None:
    """Worker: generate one sample and return a dict or None if filtered out.

    Args:
        args: (sample_id, ham_params, n_groups_max, seed)

    Returns:
        Dict with all sample data, or None if quality filter rejects it.
    """
    sample_id, ham_params, n_groups_max, seed = args
    rng = np.random.default_rng(seed)

    from pinn_trotter.benchmarks.hamiltonians import make_tfim, make_heisenberg
    from pinn_trotter.data.sampling import sample_random_strategy, sample_smart_strategy

    ham_type = ham_params["hamiltonian_type"]
    n_qubits = ham_params["n_qubits"]
    t_total = ham_params["t_final"]

    if ham_type == "tfim":
        H = make_tfim(n_qubits, J=ham_params["J"], h=ham_params["h"],
                      boundary=ham_params.get("boundary", "periodic"))
    elif ham_type == "heisenberg":
        H = make_heisenberg(n_qubits, Jx=ham_params.get("Jx", 1.0),
                            Jy=ham_params.get("Jy", 1.0), Jz=ham_params.get("Jz", 1.0),
                            boundary=ham_params.get("boundary", "periodic"))
    else:
        raise ValueError(f"Unknown hamiltonian_type: {ham_type}")

    # 80% chance of using physics-informed (commutation-based) strategy: concentrates
    # fidelity in the high-fidelity range, reducing label noise for regression.
    if rng.random() < 0.8:
        strategy = sample_smart_strategy(H.pauli_strings, n_groups_max, t_total, n_qubits, rng)
    else:
        strategy = sample_random_strategy(H.n_terms, n_groups_max, t_total, n_qubits, rng)

    has_negative_timestep = any(ts < 0 for ts in strategy.time_steps)

    psi_0 = np.zeros(2**n_qubits, dtype=complex)
    psi_0[0] = 1.0

    try:
        fidelity, _ = compute_exact_fidelity_from_hamiltonian(H, strategy, psi_0)
    except Exception:
        return None

    circuit_depth = strategy.circuit_depth_estimate()

    if not _passes_quality_filter(fidelity, circuit_depth):
        return None

    # Compute flat grouping for return
    M = H.n_terms
    grouping_flat = np.zeros(M, dtype=np.int16)
    for g_idx, group in enumerate(strategy.grouping):
        for term_idx in group:
            grouping_flat[term_idx] = g_idx

    return {
        "sample_id": sample_id,
        "pauli_strings": H.pauli_strings,
        "coefficients": H.coefficients.copy(),
        "n_qubits": n_qubits,
        "n_terms": H.n_terms,
        "ham_params": ham_params,
        "grouping_flat": grouping_flat,
        "n_groups": len(strategy.grouping),
        "orders": list(strategy.orders),
        "time_steps": list(strategy.time_steps),
        "t_total": strategy.t_total,
        "psi_0_real": psi_0.real.copy(),
        "psi_0_imag": psi_0.imag.copy(),
        "fidelity": fidelity,
        "circuit_depth": circuit_depth,
        "cx_count": -1,
        "total_gate_count": -1,
        "has_negative_timestep": has_negative_timestep,
    }


def _write_result_to_hdf5(h5file: h5py.File, result: dict) -> None:
    """Write a pre-computed result dict to HDF5."""
    sample_id = result["sample_id"]
    grp_name = f"{sample_id:07d}"
    samples_grp = h5file.require_group("dataset_v1/samples")
    grp = samples_grp.require_group(grp_name)

    dt = h5py.special_dtype(vlen=str)
    h_grp = grp.require_group("hamiltonian")
    h_grp.create_dataset(
        "pauli_strings",
        data=np.array(result["pauli_strings"], dtype=object),
        dtype=dt,
    )
    h_grp.create_dataset("coefficients", data=result["coefficients"].astype(np.float64))
    h_grp.attrs["n_qubits"] = np.int16(result["n_qubits"])
    h_grp.attrs["n_terms"] = np.int16(result["n_terms"])
    p = result["ham_params"]
    h_grp.attrs["hamiltonian_type"] = p.get("hamiltonian_type", "unknown")
    h_grp.attrs["J"] = float(p.get("J", float("nan")))
    h_grp.attrs["h"] = float(p.get("h", float("nan")))
    h_grp.attrs["bond_length"] = float(p.get("bond_length", float("nan")))
    h_grp.attrs["boundary"] = p.get("boundary", "periodic")

    s_grp = grp.require_group("strategy")
    s_grp.create_dataset("grouping_flat", data=result["grouping_flat"])
    s_grp.attrs["n_groups"] = np.int16(result["n_groups"])
    s_grp.create_dataset("orders", data=np.array(result["orders"], dtype=np.int8))
    s_grp.create_dataset("time_steps", data=np.array(result["time_steps"], dtype=np.float64))
    s_grp.attrs["t_total"] = float(result["t_total"])

    i_grp = grp.require_group("initial_state")
    i_grp.create_dataset("psi_0_real", data=result["psi_0_real"])
    i_grp.create_dataset("psi_0_imag", data=result["psi_0_imag"])
    i_grp.attrs["state_type"] = "computational_basis_0"

    l_grp = grp.require_group("labels")
    l_grp.attrs["fidelity"] = float(result["fidelity"])
    l_grp.attrs["circuit_depth"] = int(result["circuit_depth"])
    l_grp.attrs["cx_count"] = int(result["cx_count"])
    l_grp.attrs["total_gate_count"] = int(result["total_gate_count"])
    l_grp.attrs["has_negative_timestep"] = bool(result["has_negative_timestep"])


# ---------------------------------------------------------------------------
# Main dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int,
    output_path: str | Path,
    n_workers: int,
    config: dict[str, Any],
) -> int:
    """Generate a Trotter strategy dataset and save to HDF5.

    Supports breakpoint resume: samples whose IDs already exist in the output
    file are skipped.

    Args:
        n_samples: Target number of samples in the final dataset.
        output_path: Path to the output HDF5 file.
        n_workers: Number of parallel worker processes.
        config: Config dict with keys: n_groups_max, random_seed,
                n_qubits_distribution (optional), J_range (optional), etc.

    Returns:
        Number of samples written in this run.
    """
    from pinn_trotter.data.sampling import sample_tfim_params

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_base = int(config.get("random_seed", 42))
    n_groups_max = int(config.get("n_groups_max", 8))
    j_range = list(config.get("J_range", [0.1, 5.0]))
    h_range = list(config.get("h_range", [0.1, 5.0]))
    t_range = list(config.get("t_final_range", [0.5, 10.0]))
    n_qubits_fixed_raw = config.get("n_qubits", None)
    n_qubits_fixed = int(n_qubits_fixed_raw) if n_qubits_fixed_raw is not None else None

    # Check existing samples for resume support
    existing_ids: set[int] = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with h5py.File(output_path, "r") as f:
                if "dataset_v1/samples" in f:
                    existing_ids = {int(k) for k in f["dataset_v1/samples"].keys()}
        except OSError:
            pass  # corrupt or non-HDF5 file — start fresh

    remaining = n_samples - len(existing_ids)
    if remaining <= 0:
        print(f"Dataset already has {len(existing_ids)} samples, nothing to do.")
        return 0

    # Generate parameter list for all remaining samples
    rng_params = np.random.default_rng(seed_base)
    all_ids = [i for i in range(n_samples) if i not in existing_ids]
    ham_params_list = sample_tfim_params(
        len(all_ids), rng_params,
        h_min=h_range[0], h_max=h_range[1],
        j_min=j_range[0], j_max=j_range[1],
        n_qubits_fixed=n_qubits_fixed,
        t_min=t_range[0], t_max=t_range[1],
    )

    # Build task list
    tasks = [
        (all_ids[i], ham_params_list[i], n_groups_max, seed_base + all_ids[i] + 1)
        for i in range(len(all_ids))
    ]

    written = 0
    with h5py.File(output_path, "a") as h5file:
        # Write metadata
        meta = h5file.require_group("dataset_v1/metadata")
        if "created_at" not in meta.attrs:
            meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["generator_version"] = _GENERATOR_VERSION
        meta.attrs["schema_version"] = _SCHEMA_VERSION
        meta.attrs["n_samples_target"] = n_samples

        if n_workers <= 1:
            # Serial mode
            for task in tqdm(tasks, desc="Generating samples"):
                result = _generate_single_sample(task)
                if result is not None:
                    _write_result_to_hdf5(h5file, result)
                    written += 1
        else:
            with mp.Pool(processes=n_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_generate_single_sample, tasks, chunksize=10),
                    total=len(tasks),
                    desc="Generating samples",
                ):
                    if result is not None:
                        _write_result_to_hdf5(h5file, result)
                        written += 1

        meta.attrs["n_samples_written"] = len(existing_ids) + written

    return written


# ---------------------------------------------------------------------------
# Dataset statistics report
# ---------------------------------------------------------------------------

def generate_dataset_report(dataset_path: str | Path) -> dict[str, Any]:
    """Analyze dataset and write dataset_report.json alongside it.

    Args:
        dataset_path: Path to the HDF5 dataset file.

    Returns:
        Report dict (also written to JSON).
    """
    dataset_path = Path(dataset_path)
    fidelities: list[float] = []
    depths: list[int] = []
    order_counts = {1: 0, 2: 0, 4: 0}
    n_qubit_counts: dict[int, int] = {}
    n_groups_counts: dict[int, int] = {}
    n_negative_ts = 0

    with h5py.File(dataset_path, "r") as f:
        samples_grp = f.get("dataset_v1/samples", {})
        for sid in samples_grp:
            grp = samples_grp[sid]
            fidelities.append(float(grp["labels"].attrs["fidelity"]))
            depths.append(int(grp["labels"].attrs["circuit_depth"]))
            if grp["labels"].attrs.get("has_negative_timestep", False):
                n_negative_ts += 1
            orders = list(grp["strategy"]["orders"][:])
            for o in orders:
                order_counts[int(o)] = order_counts.get(int(o), 0) + 1
            n_q = int(grp["hamiltonian"].attrs["n_qubits"])
            n_qubit_counts[n_q] = n_qubit_counts.get(n_q, 0) + 1
            n_g = int(grp["strategy"].attrs["n_groups"])
            n_groups_counts[n_g] = n_groups_counts.get(n_g, 0) + 1

    fidelities_arr = np.array(fidelities)
    depths_arr = np.array(depths)

    report: dict[str, Any] = {
        "n_samples": len(fidelities),
        "fidelity": {
            "mean": float(np.mean(fidelities_arr)),
            "std": float(np.std(fidelities_arr)),
            "min": float(np.min(fidelities_arr)),
            "max": float(np.max(fidelities_arr)),
            "p25": float(np.percentile(fidelities_arr, 25)),
            "p50": float(np.percentile(fidelities_arr, 50)),
            "p75": float(np.percentile(fidelities_arr, 75)),
        },
        "circuit_depth": {
            "mean": float(np.mean(depths_arr)),
            "std": float(np.std(depths_arr)),
            "min": int(np.min(depths_arr)),
            "max": int(np.max(depths_arr)),
        },
        "order_distribution": {str(k): v for k, v in order_counts.items()},
        "n_qubits_distribution": {str(k): v for k, v in n_qubit_counts.items()},
        "n_groups_distribution": {str(k): v for k, v in n_groups_counts.items()},
        "n_negative_timestep_samples": n_negative_ts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = dataset_path.with_name("dataset_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    return report
