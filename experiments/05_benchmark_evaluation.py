"""Phase 5 benchmark evaluation across methods and random seeds.

Evaluates multiple strategy generators on sampled Hamiltonians and reports:
fidelity, transpiled depth, CX count, and inference latency statistics.

Usage:
    python experiments/05_benchmark_evaluation.py
    python experiments/05_benchmark_evaluation.py benchmark.n_test_hamiltonians=10 benchmark.n_seeds=2
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


def _build_components(cfg: DictConfig, n_qubits: int, n_terms: int):
    """Build GNN+diffusion modules used by the 'ours' method."""
    from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
    from pinn_trotter.diffusion.mixed_model import MixedDiffusionModel
    from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
    from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder

    diff_cfg = cfg.get("model", {})
    gnn_cfg = cfg.get("model", {})

    T = int(diff_cfg.get("T", 1000))
    max_groups = int(cfg.get("benchmark", {}).get("n_groups_max", 8))

    gnn = HamiltonianGNNEncoder(
        node_feat_dim=n_qubits + 2,
        edge_feat_dim=3,
        hidden_dim=int(gnn_cfg.get("hidden_dim", 256)),
        output_dim=int(gnn_cfg.get("output_dim", 512)),
        n_layers=int(gnn_cfg.get("n_layers", 4)),
    )
    diffusion = MixedDiffusionModel(
        max_groups=max_groups,
        n_terms=n_terms,
        condition_dim=int(gnn_cfg.get("output_dim", 512)),
        fused_dim=int(diff_cfg.get("condition_dim", 256)),
        time_embed_dim=int(diff_cfg.get("time_embed_dim", 128)),
        grouping_layers=int(diff_cfg.get("grouping_transformer_layers", 4)),
        order_layers=int(diff_cfg.get("order_transformer_layers", 2)),
        ts_mlp_layers=int(diff_cfg.get("timestep_mlp_layers", 3)),
        p_cond_drop=float(diff_cfg.get("p_cond_drop", 0.1)),
    )
    tm = UniformTransitionMatrix(
        K=max_groups,
        T=T,
        beta_schedule=str(diff_cfg.get("beta_schedule", "cosine")),
    )
    order_tm = UniformTransitionMatrix(K=3, T=T, beta_schedule="cosine")
    ddpm = ContinuousDDPM(T=T, beta_schedule=str(diff_cfg.get("beta_schedule", "cosine")))
    return gnn, diffusion, tm, order_tm, ddpm, max_groups


def _encode_hamiltonian(
    gnn,
    hamiltonian,
    device: torch.device,
    disable_gnn_encoder: bool = False,
) -> torch.Tensor:
    """Encode one HamiltonianGraph into conditioning vector (1, D)."""
    if disable_gnn_encoder:
        output_dim = int(getattr(gnn, "output_dim", 512))
        return torch.zeros((1, output_dim), device=device)
    try:
        data = hamiltonian.to_pyg_data()
        return gnn(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )
    except Exception:
        # Keep same fallback behavior as closed_loop.
        from pinn_trotter.hamiltonian.pauli_utils import locality

        n = hamiltonian.n_qubits
        m = hamiltonian.n_terms
        node_feats = np.zeros((m, n + 2), dtype=np.float32)
        for i, (s, c) in enumerate(zip(hamiltonian.pauli_strings, hamiltonian.coefficients)):
            node_feats[i, 0] = float(c)
            node_feats[i, 1] = float(locality(s))
            for q, ch in enumerate(s):
                node_feats[i, 2 + q] = 0.0 if ch == "I" else 1.0
        x = torch.tensor(node_feats, device=device)
        ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        ea = torch.zeros(0, 3, device=device)
        return gnn(x, ei, ea)


def _sample_hamiltonians(
    n_qubits: int,
    boundary: str,
    n_samples: int,
    seed: int,
    j_min: float = 0.5,
    j_max: float = 2.0,
    h_min: float = 0.1,
    h_max: float = 0.5,
) -> list:
    from pinn_trotter.benchmarks.hamiltonians import make_tfim

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        j_val = float(rng.uniform(j_min, j_max))
        h_val = float(rng.uniform(h_min, h_max))
        out.append(make_tfim(n_qubits, j_val, h_val, boundary))
    return out


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std}


def _safe_wandb_log(payload: dict[str, float]) -> None:
    try:
        import wandb

        wandb.log(payload)
    except Exception:
        pass


def _parse_methods(raw_methods) -> list[str]:
    """Normalize benchmark methods config to a validated list."""
    allowed = {"ours", "qiskit_4th", "cirq", "tket", "pennylane", "paulihedral"}
    if raw_methods is None:
        methods = ["ours", "qiskit_4th", "cirq", "tket", "pennylane", "paulihedral"]
    elif isinstance(raw_methods, str):
        methods = [m.strip() for m in raw_methods.split(",") if m.strip()]
    else:
        methods = [str(m).strip() for m in raw_methods if str(m).strip()]
    invalid = [m for m in methods if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported benchmark methods: {invalid}, allowed={sorted(allowed)}")
    if not methods:
        raise ValueError("benchmark.methods cannot be empty")
    return methods


@hydra.main(config_path="../configs", config_name="experiment/tfim_4q_poc", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from pinn_trotter.benchmarks.baselines import QiskitTrotterBaseline
    from pinn_trotter.benchmarks.baseline_adapters import BASELINE_REGISTRY
    from pinn_trotter.benchmarks.metrics import (
        cx_count,
        exact_fidelity,
        inference_latency,
        transpiled_depth,
    )
    from pinn_trotter.diffusion.mixed_model import guided_sample
    from pinn_trotter.pinn.evaluator import _decode_strategy

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    exp_cfg = cfg.get("experiment", {})
    bench_cfg = cfg.get("benchmark", {})

    n_qubits = int(exp_cfg.get("n_qubits", 4))
    boundary = str(exp_cfg.get("boundary", "periodic"))
    t_total = float(exp_cfg.get("t_total", 2.0))
    n_terms = 2 * n_qubits

    n_test_h = int(bench_cfg.get("n_test_hamiltonians", 100))
    n_seeds = int(bench_cfg.get("n_seeds", 5))
    seed0 = int(bench_cfg.get("seed", 42))
    n_trials_latency = int(bench_cfg.get("n_trials_latency", 10))
    guidance_scale = float(bench_cfg.get("guidance_scale", 2.0))
    use_wandb = bool(bench_cfg.get("use_wandb", False))
    disable_gnn_encoder = bool(bench_cfg.get("disable_gnn_encoder", False))
    fidelity_only = bool(bench_cfg.get("fidelity_only", False))
    h_min = float(bench_cfg.get("h_min", 0.1))
    h_max = float(bench_cfg.get("h_max", 0.5))
    all_methods = _parse_methods(bench_cfg.get("methods", None))

    device_str = str(bench_cfg.get("device", "cpu"))
    if device_str == "cuda" and not torch.cuda.is_available():
        log.warning("benchmark.device=cuda but CUDA unavailable, falling back to cpu")
        device_str = "cpu"
    elif device_str == "cuda":
        try:
            torch.zeros(1, device="cuda") + 1
        except Exception:
            log.warning("benchmark.device=cuda but CUDA kernel failed, falling back to cpu")
            device_str = "cpu"
    device = torch.device(device_str)
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "experiments" / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build "ours" components.
    gnn, diffusion, tm, order_tm, ddpm, max_groups = _build_components(cfg, n_qubits, n_terms)
    gnn = gnn.to(device).eval()
    diffusion = diffusion.to(device).eval()
    tm = tm.to(device)
    order_tm = order_tm.to(device)
    ddpm = ddpm.to(device)

    model_ckpt = cfg.get("benchmark", {}).get("model_ckpt", None)
    if model_ckpt:
        ckpt_path = project_root / str(model_ckpt)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Prefer EMA weights for inference — they yield smoother generation.
        if "ema_state" in state:
            diffusion.load_state_dict(state["ema_state"])
        elif "diffusion_state" in state:
            diffusion.load_state_dict(state["diffusion_state"])
        elif "diffusion_model_state" in state:
            diffusion.load_state_dict(state["diffusion_model_state"])
        else:
            raise ValueError(f"No diffusion state found in checkpoint: {ckpt_path}")

        if "gnn_state" in state:
            gnn.load_state_dict(state["gnn_state"])
        elif "gnn_encoder_state" in state:
            gnn.load_state_dict(state["gnn_encoder_state"])
        else:
            raise ValueError(f"No GNN state found in checkpoint: {ckpt_path}")
        log.info("Loaded model checkpoint: %s", ckpt_path)

    qiskit_baseline = QiskitTrotterBaseline()

    # External Trotter baselines (cirq/tket/pennylane/paulihedral) — instantiated once per run
    baseline_n_steps = int(bench_cfg.get("baseline_n_steps", 5))
    external_baselines = {}
    for name, cls in BASELINE_REGISTRY.items():
        if name == "paulihedral":
            external_baselines[name] = cls(
                n_steps=baseline_n_steps,
                scheduler=str(bench_cfg.get("paulihedral_scheduler", "depth")),
            )
        else:
            external_baselines[name] = cls(n_steps=baseline_n_steps)

    aggregate_values: dict[str, dict[str, list[float]]] = {
        m: {"fidelity": [], "depth": [], "cx_count": [], "latency": []}
        for m in all_methods
    }
    per_seed_results: list[dict[str, Any]] = []

    if use_wandb:
        import wandb

        wandb.init(
            project=str(bench_cfg.get("wandb_project", "pinn-trotter-benchmark")),
            name=str(bench_cfg.get("wandb_run_name", "benchmark_eval")),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    for seed_idx in range(n_seeds):
        seed = seed0 + seed_idx
        hamiltonians = _sample_hamiltonians(n_qubits, boundary, n_test_h, seed,
                                            h_min=h_min, h_max=h_max)
        seed_payload: dict[str, Any] = {"seed": seed, "methods": {}}

        for method in all_methods:
            fidelity_values: list[float] = []
            depth_values: list[float] = []
            cx_values: list[float] = []

            for hamiltonian in hamiltonians:
                # Framework baselines (qiskit_4th/cirq/tket/pennylane) — each computes
                # its own fidelity via the framework-native simulator. Depth/cx_count
                # use the proxy strategy under the unified comparison gate set.
                if method in external_baselines:
                    adapter = external_baselines[method]
                    res = adapter.evaluate(hamiltonian, t_total)
                    fidelity_values.append(float(res["fidelity"]))
                    if not fidelity_only:
                        if "depth" in res and "cx_count" in res:
                            depth_values.append(float(res["depth"]))
                            cx_values.append(float(res["cx_count"]))
                        else:
                            strategy = res["strategy"]
                            depth_values.append(float(transpiled_depth(strategy, hamiltonian)))
                            cx_values.append(float(cx_count(strategy, hamiltonian)))
                    continue
                if method == "qiskit_4th":
                    res = qiskit_baseline.evaluate(
                        hamiltonian=hamiltonian,
                        t_total=t_total,
                        order=4,
                        n_steps=int(bench_cfg.get("qiskit_4th_n_steps", 5)),
                    )
                    fidelity_values.append(float(res["fidelity"]))
                    if not fidelity_only:
                        strategy = res["strategy"]
                        depth_values.append(float(transpiled_depth(strategy, hamiltonian)))
                        cx_values.append(float(cx_count(strategy, hamiltonian)))
                    continue
                else:
                    with torch.no_grad():
                        cond = _encode_hamiltonian(
                            gnn, hamiltonian, device, disable_gnn_encoder=disable_gnn_encoder
                        )
                        g, ts, o = guided_sample(
                            model=diffusion,
                            condition=cond,
                            n_terms=hamiltonian.n_terms,
                            max_groups=max_groups,
                            transition_matrix=tm,
                            order_transition_matrix=order_tm,
                            ddpm=ddpm,
                            guidance_scale=guidance_scale,
                            device=device,
                        )
                    strategy = _decode_strategy(hamiltonian, g, ts, o, t_total=t_total)

                fidelity_values.append(exact_fidelity(strategy, hamiltonian, psi_0=None))
                if not fidelity_only:
                    depth_values.append(float(transpiled_depth(strategy, hamiltonian)))
                    cx_values.append(float(cx_count(strategy, hamiltonian)))

            if not fidelity_values:
                seed_payload["methods"][method] = {
                    "n_samples": 0,
                    "fidelity": {"mean": 0.0, "std": 0.0},
                    "depth": {"mean": 0.0, "std": 0.0},
                    "cx_count": {"mean": 0.0, "std": 0.0},
                    "latency": {"mean": 0.0, "std": 0.0},
                }
                continue

            if n_trials_latency > 0:
                # Latency: evaluate generator API repeatedly on one representative Hamiltonian.
                if method in external_baselines:
                    _adapter = external_baselines[method]
                    latency_model = lambda hh: _adapter.evaluate(hh, t_total)  # noqa: E731
                elif method == "qiskit_4th":
                    latency_model = lambda hh: qiskit_baseline.generate_strategy(  # noqa: E731
                        hamiltonian=hh,
                        t_final=t_total,
                        order=4,
                        n_steps=int(bench_cfg.get("qiskit_4th_n_steps", 5)),
                    )
                else:
                    def latency_model(hh):
                        with torch.no_grad():
                            cond = _encode_hamiltonian(
                                gnn, hh, device, disable_gnn_encoder=disable_gnn_encoder
                            )
                            g, ts, o = guided_sample(
                                model=diffusion,
                                condition=cond,
                                n_terms=hh.n_terms,
                                max_groups=max_groups,
                                transition_matrix=tm,
                                order_transition_matrix=order_tm,
                                ddpm=ddpm,
                                guidance_scale=guidance_scale,
                                device=device,
                            )
                        return _decode_strategy(hh, g, ts, o, t_total=t_total)

                latency = inference_latency(
                    latency_model,
                    hamiltonians[0],
                    n_trials=n_trials_latency,
                )
            else:
                latency = 0.0

            aggregate_values[method]["fidelity"].extend(fidelity_values)
            aggregate_values[method]["depth"].extend(depth_values)
            aggregate_values[method]["cx_count"].extend(cx_values)
            aggregate_values[method]["latency"].append(latency)

            seed_payload["methods"][method] = {
                "n_samples": len(fidelity_values),
                "fidelity": _summarize(fidelity_values),
                "depth": _summarize(depth_values),
                "cx_count": _summarize(cx_values),
                "latency": _summarize([latency]),
            }

            if use_wandb:
                _safe_wandb_log(
                    {
                        f"{method}/seed_fidelity_mean": seed_payload["methods"][method]["fidelity"]["mean"],
                        f"{method}/seed_depth_mean": seed_payload["methods"][method]["depth"]["mean"],
                        f"{method}/seed_cx_mean": seed_payload["methods"][method]["cx_count"]["mean"],
                        f"{method}/seed_latency_mean": latency,
                        "seed": float(seed),
                    }
                )

        per_seed_results.append(seed_payload)
        log.info("Completed seed %d/%d (seed=%d)", seed_idx + 1, n_seeds, seed)

    summary: dict[str, Any] = {"methods": {}}
    for method, vals in aggregate_values.items():
        summary["methods"][method] = {
            "n_samples": len(vals["fidelity"]),
            "fidelity": _summarize(vals["fidelity"]),
            "depth": _summarize(vals["depth"]),
            "cx_count": _summarize(vals["cx_count"]),
            "latency": _summarize(vals["latency"]),
        }

    report = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "summary": summary,
        "per_seed": per_seed_results,
    }
    output_filename = str(bench_cfg.get("output_filename", "benchmark_evaluation_results.json"))
    out_path = output_dir / output_filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("Saved benchmark report: %s", out_path)

    if use_wandb:
        _safe_wandb_log(
            {
                f"summary/{method}/fidelity_mean": vals["fidelity"]["mean"]
                for method, vals in summary["methods"].items()
            }
        )
        try:
            import wandb

            wandb.save(str(out_path))
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
