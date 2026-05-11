"""D3PM transition matrices for discrete diffusion over grouping labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class UniformTransitionMatrix:
    """Uniform D3PM transition matrix: noisy state approaches uniform over K classes.

    For each step t, the transition matrix is:
        Q_t[i,j] = β_t/K  (i ≠ j)
        Q_t[i,i] = 1 - β_t*(K-1)/K

    The cumulative product Q̄_t = Q_1 @ Q_2 @ ... @ Q_t satisfies:
        Q̄_t[i,j] = (1 - ᾱ_t)/K  (i ≠ j)
        Q̄_t[i,i] = ᾱ_t + (1 - ᾱ_t)/K

    where ᾱ_t = ∏_s (1 - β_s * (K-1)/K).

    Args:
        K:             Number of discrete categories (max_groups).
        T:             Total diffusion timesteps.
        beta_schedule: 'linear' or 'cosine'.
        beta_start:    Start value for linear schedule.
        beta_end:      End value for linear schedule.
    """

    def __init__(
        self,
        K: int,
        T: int,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.K = K
        self.T = T

        betas = _make_beta_schedule(beta_schedule, T, beta_start, beta_end)  # (T,)

        # Effective noise per step for K classes
        beta_eff = betas * (K - 1) / K  # (T,)

        # Cumulative alpha: α̅_t = Π_{s=1}^{t} (1 - β_eff_s)
        alpha_bar = torch.cumprod(1.0 - beta_eff, dim=0)  # (T,)

        # Precompute Q̄_t as (T, K, K)
        # Off-diagonal: (1 - α̅_t) / K
        # Diagonal:      α̅_t + (1 - α̅_t) / K
        Q_bar = torch.zeros(T, K, K)
        off_diag = (1.0 - alpha_bar) / K  # (T,)
        for t in range(T):
            Q_bar[t] = off_diag[t]
            Q_bar[t].fill_diagonal_(alpha_bar[t] + off_diag[t])

        self.register_buffer_dict = {
            "Q_bar": Q_bar,
            "alpha_bar": alpha_bar,
            "betas": betas,
        }
        # Store as plain tensors (not nn.Module, no CUDA tracking needed here)
        self.Q_bar = Q_bar        # (T, K, K)
        self.alpha_bar = alpha_bar  # (T,)
        self.betas = betas          # (T,)

    def to(self, device: torch.device | str) -> "UniformTransitionMatrix":
        self.Q_bar = self.Q_bar.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.betas = self.betas.to(device)
        return self

    def get_Q_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return cumulative transition matrices for a batch of timesteps.

        Args:
            t: Integer timesteps, shape (B,), values in [0, T-1].

        Returns:
            Q̄_t matrices, shape (B, K, K).
        """
        return self.Q_bar[t]  # (B, K, K)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) = Cat(x_0 @ Q̄_t^T).

        Args:
            x0: Integer class labels, shape (B, M).
            t:  Timesteps, shape (B,), values in [0, T-1].

        Returns:
            Noisy labels x_t, shape (B, M).
        """
        B, M = x0.shape
        K = self.K

        # One-hot encode x0: (B, M, K)
        x0_oh = F.one_hot(x0, num_classes=K).float()

        # Q̄_t: (B, K, K)
        Q_bar_t = self.get_Q_bar(t)  # (B, K, K)

        # q(x_t | x_0) = x0_oh @ Q̄_t: (B, M, K) @ (B, 1, K, K)
        # We want prob[b, m, :] = x0_oh[b, m, :] @ Q_bar_t[b]
        Q_b = Q_bar_t.unsqueeze(1)         # (B, 1, K, K)
        probs = (x0_oh.unsqueeze(2) @ Q_b).squeeze(2)  # (B, M, K)

        # Sample from categorical
        flat = probs.reshape(B * M, K)
        samples = torch.multinomial(flat.clamp(min=1e-10), num_samples=1).squeeze(1)
        return samples.reshape(B, M)

    def compute_posterior_logits(
        self,
        x_t: torch.Tensor,
        x0_logits: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for q(x_{t-1} | x_t, x_0) using predicted x_0 distribution.

        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) q(x_{t-1} | x_0)

        In the uniform case this simplifies to:
            posterior ∝ Q_t[x_{t-1}, x_t] * (x0_probs @ Q̄_{t-1})

        Args:
            x_t:       Noisy labels, shape (B, M), integers in [0, K-1].
            x0_logits: Model-predicted x_0 logits, shape (B, M, K).
            t:         Timesteps, shape (B,).

        Returns:
            Posterior logits shape (B, M, K).
        """
        B, M = x_t.shape
        K = self.K

        x0_probs = torch.softmax(x0_logits, dim=-1).clamp(min=1e-8)  # (B, M, K)

        # Q̄_{t-1}: use t=0 as identity when t==0
        t_prev = (t - 1).clamp(min=0)
        Q_bar_t = self.get_Q_bar(t)       # (B, K, K)
        Q_bar_prev = self.get_Q_bar(t_prev)  # (B, K, K)

        # q(x_{t-1} | x_0): (B, M, K)  = x0_probs @ Q̄_{t-1}
        q_prev = (x0_probs.unsqueeze(2) @ Q_bar_prev.unsqueeze(1)).squeeze(2)

        # q(x_t | x_{t-1}): we need Q_t[x_{t-1}, x_t]
        # single-step Q_t ≈ Q_bar_t @ inv(Q_bar_{t-1}) — approximate with Q_bar
        # For uniform matrix use: Q_t = Q̄_t / Q̄_{t-1} elementwise (diagonal approx)
        # Simpler: use q(x_t=xt | x_{t-1}=k) = Q̄_t[k, x_t] -- this is the full posterior
        x_t_oh = F.one_hot(x_t, num_classes=K).float()  # (B, M, K)
        # q(x_t | x_{t-1}): slice Q̄_t at column x_t → (B, M, K)
        q_t_given_prev = (Q_bar_t.unsqueeze(1) @ x_t_oh.unsqueeze(-1)).squeeze(-1)
        # q_t_given_prev[b, m, k] = Q̄_t[b, k, x_t[b,m]]  — shape (B, M, K)

        # posterior ∝ q_t_given_prev * q_prev
        posterior = (q_t_given_prev * q_prev).clamp(min=1e-8)
        return torch.log(posterior)  # return log-probs (logits)


def _make_beta_schedule(
    schedule: str,
    T: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":
        # Cosine schedule from Nichol & Dhariwal 2021
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(min=1e-5, max=0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule!r}")
