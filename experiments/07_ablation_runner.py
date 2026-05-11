"""Run ablation studies by retraining and evaluating each variant."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AblationProfile:
    name: str
    train_overrides: tuple[str, ...]
    eval_overrides: tuple[str, ...]
    note: str


def get_profiles() -> dict[str, AblationProfile]:
    """Ablation profiles aligned with TODO 7-C-2."""
    return {
        "full_model": AblationProfile(
            name="full_model",
            train_overrides=(),
            eval_overrides=(),
            note="Reference configuration without ablation.",
        ),
        "no_pinn_guidance": AblationProfile(
            name="no_pinn_guidance",
            train_overrides=("+training.disable_pinn_guidance=true",),
            eval_overrides=(),
            note="Disable PINN-guided reward signal in closed-loop training.",
        ),
        "no_cfg": AblationProfile(
            name="no_cfg",
            train_overrides=("model.p_cond_drop=0.0", "training.guidance_scale=1.0"),
            eval_overrides=("benchmark.guidance_scale=1.0",),
            note="Disable classifier-free guidance by removing condition dropout.",
        ),
        "no_structured_matrix": AblationProfile(
            name="no_structured_matrix",
            train_overrides=(),
            eval_overrides=(),
            note=(
                "Structured transition matrix path is not implemented yet; "
                "kept as no-op control run."
            ),
        ),
        "no_gnn_encoder": AblationProfile(
            name="no_gnn_encoder",
            train_overrides=("+training.disable_gnn_encoder=true",),
            eval_overrides=("+benchmark.disable_gnn_encoder=true",),
            note="Replace GNN condition vectors with zeros during train/eval.",
        ),
        "no_gumbel_estimator": AblationProfile(
            name="no_gumbel_estimator",
            train_overrides=(),
            eval_overrides=(),
            note="Current closed-loop implementation does not use Gumbel estimator.",
        ),
    }


def _latest_checkpoint(ckpt_dir: Path, since_ts: float) -> Path | None:
    candidates = [p for p in ckpt_dir.glob("*.pt") if p.stat().st_mtime >= since_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_command(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation retrain+evaluation loops.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=[
            "full_model",
            "no_pinn_guidance",
            "no_cfg",
            "no_structured_matrix",
            "no_gnn_encoder",
            "no_gumbel_estimator",
        ],
        help="Ablation profiles to run.",
    )
    parser.add_argument("--train-iters", type=int, default=100, help="Closed-loop iterations.")
    parser.add_argument("--n-test-hamiltonians", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--latency-trials", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_dir = root / "experiments" / "closed_loop_checkpoints"
    results_dir = root / "experiments" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    profiles = get_profiles()
    selected: list[AblationProfile] = []
    for name in args.profiles:
        if name not in profiles:
            raise ValueError(f"Unknown profile: {name}")
        selected.append(profiles[name])

    summary: dict[str, dict] = {}
    for profile in selected:
        start_ts = time.time()
        print(f"\n=== Running ablation: {profile.name} ===")
        print(profile.note)

        train_cmd = [
            sys.executable,
            str(root / "experiments" / "04_closed_loop_finetune.py"),
            f"training.n_iterations={args.train_iters}",
            "+training.batch_size_hamiltonians=2",
            "+training.lambda_weight_sweep=[0.1]",
            "+training.guidance_scale=2.0",
            "experiment.n_qubits=4",
            "experiment.t_total=2.0",
            "model.T=50",
        ]
        train_cmd.extend(profile.train_overrides)
        _run_command(train_cmd, args.dry_run)

        ckpt_path = None if args.dry_run else _latest_checkpoint(ckpt_dir, start_ts)
        eval_filename = f"ablation_{profile.name}.json"
        eval_cmd = [
            sys.executable,
            str(root / "experiments" / "05_benchmark_evaluation.py"),
            f"benchmark.n_test_hamiltonians={args.n_test_hamiltonians}",
            f"benchmark.n_seeds={args.n_seeds}",
            f"benchmark.n_trials_latency={args.latency_trials}",
            "benchmark.use_wandb=false",
            f"benchmark.device={args.device}",
            f"benchmark.output_filename={eval_filename}",
        ]
        if ckpt_path is not None:
            eval_cmd.append(f"benchmark.model_ckpt={ckpt_path.relative_to(root)}")
        eval_cmd.extend(profile.eval_overrides)
        _run_command(eval_cmd, args.dry_run)

        if args.dry_run:
            summary[profile.name] = {
                "note": profile.note,
                "status": "dry_run",
                "output_file": eval_filename,
            }
            continue

        out_path = results_dir / eval_filename
        if not out_path.exists():
            raise FileNotFoundError(f"Expected evaluation output not found: {out_path}")
        with open(out_path, encoding="utf-8") as f:
            report = json.load(f)
        summary[profile.name] = {
            "note": profile.note,
            "checkpoint": str(ckpt_path.relative_to(root)) if ckpt_path else None,
            "summary": report.get("summary", {}),
            "output_file": str(out_path.relative_to(root)),
        }

    final_report = {
        "profiles": args.profiles,
        "train_iters": args.train_iters,
        "n_test_hamiltonians": args.n_test_hamiltonians,
        "n_seeds": args.n_seeds,
        "latency_trials": args.latency_trials,
        "results": summary,
    }
    merged_path = results_dir / "ablation_summary.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved ablation summary: {merged_path}")


if __name__ == "__main__":
    main()
