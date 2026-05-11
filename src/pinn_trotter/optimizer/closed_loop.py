"""ClosedLoopOptimizer: REINFORCE-based joint training of diffusion + GNN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pinn_trotter.diffusion.ddpm_continuous import ContinuousDDPM
from pinn_trotter.diffusion.mixed_model import EMAWrapper, MixedDiffusionModel, guided_sample
from pinn_trotter.diffusion.transition_matrix import UniformTransitionMatrix
from pinn_trotter.gnn.encoder import HamiltonianGNNEncoder
from pinn_trotter.optimizer.gradient_utils import compute_policy_log_prob
from pinn_trotter.optimizer.pareto import ParetoTracker

log = logging.getLogger(__name__)


class ClosedLoopOptimizer:
    """Joint closed-loop optimization of diffusion + GNN via REINFORCE.

    Training loop (per iteration):
      1. Sample a batch of Hamiltonians from the dataset.
      2. Encode each Hamiltonian with the GNN encoder.
      3. Sample strategies from the diffusion model (guided).
      4. Evaluate strategies via PINN fidelity proxy.
      5. Compute REINFORCE loss with mean baseline.
      6. Backprop and update diffusion model (+ EMA).
      7. Update Pareto tracker.
      8. Save checkpoint every `checkpoint_interval` iterations.

    Args:
        diffusion_model:  MixedDiffusionModel to train.
        gnn_encoder:      HamiltonianGNNEncoder (frozen or fine-tuned).
        transition_matrix: For grouping labels.
        order_transition_matrix: For order labels.
        ddpm:             ContinuousDDPM for time steps.
        pinn_evaluator:   Callable (hamiltonian, strategy_tensors) → fidelity float.
                          Signature: (H_graph, grouping, timesteps, orders) → float.
        lambda_weight:    Reward = fidelity - lambda * depth.
        lr:               Adam learning rate for diffusion model.
        guidance_scale:   CFG guidance scale during sampling.
        batch_size:       Hamiltonians per iteration.
        n_terms:          M, number of Pauli terms (fixed for PoC).
        max_groups:       K, max groups.
        ema_decay:        EMA decay for diffusion model shadow.
        checkpoint_dir:   Where to save checkpoints.
        checkpoint_interval: Save every N iterations.
        ref_depth:        Pareto hypervolume reference depth.
        device:           Torch device.
    """

    def __init__(
        self,
        diffusion_model: MixedDiffusionModel,
        gnn_encoder: HamiltonianGNNEncoder,
        transition_matrix: UniformTransitionMatrix,
        order_transition_matrix: UniformTransitionMatrix,
        ddpm: ContinuousDDPM,
        pinn_evaluator,
        lambda_weight: float = 0.1,
        lr: float = 1e-4,
        guidance_scale: float = 2.0,
        batch_size: int = 8,
        n_terms: int = 8,
        max_groups: int = 8,
        ema_decay: float = 0.9999,
        checkpoint_dir: str | Path = "experiments/checkpoints",
        checkpoint_interval: int = 100,
        ref_depth: float = 1000.0,
        disable_gnn_encoder: bool = False,
        disable_pinn_guidance: bool = False,
        device: Optional[str | torch.device] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.diffusion_model = diffusion_model.to(self.device)
        self.gnn_encoder = gnn_encoder.to(self.device)
        self.transition_matrix = transition_matrix.to(self.device)
        self.order_transition_matrix = order_transition_matrix.to(self.device)
        self.ddpm = ddpm.to(self.device)
        self.pinn_evaluator = pinn_evaluator

        self.lambda_weight = lambda_weight
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.n_terms = n_terms
        self.max_groups = max_groups
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.disable_gnn_encoder = disable_gnn_encoder
        self.disable_pinn_guidance = disable_pinn_guidance

        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)
        self.ema = EMAWrapper(diffusion_model, decay=ema_decay)
        self.pareto = ParetoTracker(ref_depth=ref_depth)

        self._history: dict[str, list] = {
            "policy_loss": [], "mean_fidelity": [], "mean_depth": [], "pareto_hv": []
        }
        self._ckpt_registry: list[dict] = []  # track saved checkpoints

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_iterations: int,
        hamiltonian_sampler,
        start_iteration: int = 0,
    ) -> dict[str, list]:
        """Run the closed-loop optimization loop.

        Args:
            n_iterations:       Total iterations to run.
            hamiltonian_sampler: Callable() → list of HamiltonianGraph objects
                                 (length = batch_size).
            start_iteration:    Resume from this iteration count.

        Returns:
            Training history dict.
        """
        self.diffusion_model.train()
        self.gnn_encoder.eval()  # GNN frozen during closed-loop

        for iteration in range(start_iteration, start_iteration + n_iterations):
            # 1. Sample Hamiltonians
            hamiltonians = hamiltonian_sampler()

            # 2. Encode with GNN
            conditions = self._encode_hamiltonians(hamiltonians)  # (B, 512)

            # 3. Sample strategies via guided diffusion (no grad for sampling)
            with torch.no_grad():
                grouping_samples, ts_samples, order_samples = guided_sample(
                    model=self.diffusion_model,
                    condition=conditions,
                    n_terms=self.n_terms,
                    max_groups=self.max_groups,
                    transition_matrix=self.transition_matrix,
                    order_transition_matrix=self.order_transition_matrix,
                    ddpm=self.ddpm,
                    guidance_scale=self.guidance_scale,
                    device=self.device,
                )

            # 4. Evaluate fidelity + depth
            fidelities, depths = self._evaluate_batch(
                hamiltonians, grouping_samples, ts_samples, order_samples
            )

            # 5. REINFORCE with mean baseline
            rewards = (
                torch.tensor(fidelities, device=self.device)
                - self.lambda_weight * torch.tensor(depths, dtype=torch.float32, device=self.device)
            )
            baseline = rewards.mean()
            advantages = (rewards - baseline).detach()

            # 6. Re-compute log-prob of sampled strategies (with grad)
            log_probs = self._compute_log_probs(
                conditions, grouping_samples, ts_samples, order_samples
            )
            policy_loss = -(advantages * log_probs).mean()

            self.optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.ema.update(self.diffusion_model)

            # 7. Pareto update
            self.pareto.update(fidelities, depths)

            # 8. Log
            hv = self.pareto.hypervolume()
            self._history["policy_loss"].append(float(policy_loss.detach()))
            self._history["mean_fidelity"].append(float(np.mean(fidelities)))
            self._history["mean_depth"].append(float(np.mean(depths)))
            self._history["pareto_hv"].append(hv)

            if iteration % 10 == 0:
                log.info(
                    "iter=%d loss=%.4e fid=%.4f depth=%.1f hv=%.2f",
                    iteration, float(policy_loss.detach()),
                    float(np.mean(fidelities)), float(np.mean(depths)), hv,
                )

            # 9. Checkpoint
            if (iteration + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(iteration, float(np.mean(fidelities)), hv)

        return self._history

    # ------------------------------------------------------------------
    # Lambda sweep
    # ------------------------------------------------------------------

    def lambda_sweep(
        self,
        lambda_values: list[float],
        n_iterations: int,
        hamiltonian_sampler,
    ) -> dict[float, dict]:
        """Run independent training runs for each lambda value.

        Returns:
            Dict mapping lambda → training history.
        """
        results = {}
        original_lambda = self.lambda_weight
        original_state = {k: v.cpu() for k, v in self.diffusion_model.state_dict().items()}

        for lam in lambda_values:
            log.info("Lambda sweep: λ=%.3f", lam)
            # Reset model, optimizer, pareto, and history for each lambda run
            self.diffusion_model.load_state_dict(
                {k: v.to(self.device) for k, v in original_state.items()}
            )
            self.optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=1e-4)
            self.pareto = ParetoTracker(ref_depth=self.pareto.ref_depth)
            self._history = {"policy_loss": [], "mean_fidelity": [], "mean_depth": [], "pareto_hv": []}
            self.lambda_weight = lam

            history = self.train(n_iterations, hamiltonian_sampler)
            results[lam] = {
                "history": history,
                "pareto_front": self.pareto.get_front(),
                "hypervolume": self.pareto.hypervolume(),
            }

        self.lambda_weight = original_lambda
        return results

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int, mean_fidelity: float, hv: float) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        depth_int = int(self.pareto.best_depth())
        name = f"ckpt_iter_{iteration:06d}_fid{mean_fidelity:.4f}_depth{depth_int:04d}.pt"
        path = self.checkpoint_dir / name

        state = {
            "iteration": iteration,
            "diffusion_model_state": self.diffusion_model.state_dict(),
            "ema_model_state": self.ema.shadow.state_dict(),
            "gnn_encoder_state": self.gnn_encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "pareto_front": self.pareto.get_front(),
            "mean_fidelity": mean_fidelity,
            "hypervolume": hv,
        }
        torch.save(state, path)

        self._ckpt_registry.append({
            "path": str(path), "iteration": iteration,
            "fidelity": mean_fidelity, "hv": hv,
        })
        self._cleanup_checkpoints()
        log.info("Checkpoint saved: %s", path)

    def _cleanup_checkpoints(self) -> None:
        """Keep only: best fidelity, best HV, and 3 most recent."""
        if len(self._ckpt_registry) <= 5:
            return

        best_fid = max(self._ckpt_registry, key=lambda x: x["fidelity"])
        best_hv = max(self._ckpt_registry, key=lambda x: x["hv"])
        recent_3 = sorted(self._ckpt_registry, key=lambda x: x["iteration"])[-3:]

        keep = {best_fid["path"], best_hv["path"]}
        keep.update(c["path"] for c in recent_3)

        to_delete = [c for c in self._ckpt_registry if c["path"] not in keep]
        for c in to_delete:
            p = Path(c["path"])
            if p.exists():
                p.unlink()
        self._ckpt_registry = [c for c in self._ckpt_registry if c["path"] in keep]

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return the saved iteration number."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.diffusion_model.load_state_dict(state["diffusion_model_state"])
        self.ema.shadow.load_state_dict(state["ema_model_state"])
        self.gnn_encoder.load_state_dict(state["gnn_encoder_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        # Restore Pareto front
        for p in state["pareto_front"]:
            self.pareto._points.append(p)
        log.info("Loaded checkpoint from %s (iter=%d)", path, state["iteration"])
        return int(state["iteration"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_hamiltonians(self, hamiltonians) -> torch.Tensor:
        """Encode a list of HamiltonianGraph objects → (B, output_dim) tensor."""
        if self.disable_gnn_encoder:
            output_dim = int(getattr(self.gnn_encoder, "output_dim", 512))
            return torch.zeros((len(hamiltonians), output_dim), device=self.device)
        conds = []
        with torch.no_grad():
            for H in hamiltonians:
                try:
                    data = H.to_pyg_data()
                    c = self.gnn_encoder(
                        data.x.to(self.device),
                        data.edge_index.to(self.device),
                        data.edge_attr.to(self.device),
                    )
                except Exception:
                    # Fallback: build plain tensors from hamiltonian
                    c = self._encode_fallback(H)
                conds.append(c)
        return torch.cat(conds, dim=0)  # (B, output_dim)

    def _encode_fallback(self, H) -> torch.Tensor:
        """Build node/edge tensors directly without PyG and encode."""
        import numpy as np
        n = H.n_qubits
        M = H.n_terms
        node_feats = np.zeros((M, n + 2), dtype=np.float32)
        from pinn_trotter.hamiltonian.pauli_utils import locality
        for i, (s, c) in enumerate(zip(H.pauli_strings, H.coefficients)):
            node_feats[i, 0] = float(c)
            node_feats[i, 1] = float(locality(s))
            for q, ch in enumerate(s):
                node_feats[i, 2 + q] = 0.0 if ch == "I" else 1.0
        x = torch.tensor(node_feats, device=self.device)
        ei = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        ea = torch.zeros(0, 3, device=self.device)
        return self.gnn_encoder(x, ei, ea)

    def _evaluate_batch(
        self, hamiltonians, grouping, timesteps, orders
    ) -> tuple[list[float], list[int]]:
        """Evaluate a batch of strategies and return (fidelities, depths)."""
        import torch.nn.functional as F_func
        from pinn_trotter.pinn.evaluator import _decode_strategy

        fidelities, depths = [], []
        for i, H in enumerate(hamiltonians):
            g = grouping[i:i+1]
            ts = timesteps[i:i+1]
            o = orders[i:i+1]
            o_eval = F_func.one_hot(o, num_classes=3).float() if o.dim() == 2 else o
            if self.disable_pinn_guidance:
                f = 0.0
            else:
                try:
                    f = self.pinn_evaluator(H, g, ts, o_eval)
                except Exception:
                    f = 0.0
            # Depth: use strategy.circuit_depth_estimate() when pinn_evaluator
            # exposes a circuit_depth() helper; otherwise fall back to the
            # structural estimate from the decoded strategy.
            if hasattr(self.pinn_evaluator, "circuit_depth"):
                try:
                    d = self.pinn_evaluator.circuit_depth(H, g, ts, o_eval)
                except Exception:
                    d = int(g.max().item() + 1) * H.n_terms
            else:
                try:
                    t_total = getattr(self.pinn_evaluator, "t_total", 1.0)
                    strategy = _decode_strategy(H, g, ts, o_eval, t_total)
                    d = strategy.circuit_depth_estimate()
                except Exception:
                    d = int(g.max().item() + 1) * H.n_terms
            fidelities.append(float(f))
            depths.append(d)
        return fidelities, depths

    def _compute_log_probs(
        self, conditions, grouping, timesteps, orders
    ) -> torch.Tensor:
        """Forward pass through diffusion model to get strategy log-probs."""
        import torch.nn.functional as F_func
        B = conditions.shape[0]

        # Run one forward pass at t=0 (near-clean prediction)
        t_zero = torch.zeros(B, dtype=torch.long, device=self.device)
        orders_oh = F_func.one_hot(orders, num_classes=3).float()

        g_logits, _, o_logits = self.diffusion_model(
            grouping, timesteps, orders_oh, t_zero, conditions
        )
        return compute_policy_log_prob(grouping, orders, g_logits, o_logits)
