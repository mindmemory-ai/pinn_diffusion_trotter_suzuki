"""PINN loss computation for quantum dynamics."""

from __future__ import annotations

from typing import Optional

import torch


def scipy_sparse_to_torch_csr(
    H_scipy: "scipy.sparse.spmatrix",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse CSR tensor.

    Args:
        H_scipy: scipy sparse matrix (any format; will be converted to CSR).
        device: Target device. Defaults to CPU.
        dtype: Target complex dtype.

    Returns:
        torch.Tensor in sparse_csr layout.
    """
    import scipy.sparse as sp

    csr = H_scipy.tocsr().astype(complex)
    crow = torch.tensor(csr.indptr, dtype=torch.int32)
    col = torch.tensor(csr.indices, dtype=torch.int32)
    val = torch.tensor(csr.data, dtype=dtype)
    shape = csr.shape
    t = torch.sparse_csr_tensor(crow, col, val, size=shape, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _H_matmul(H: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """H @ psi, supporting both dense and sparse CSR H."""
    if H.layout == torch.sparse_csr:
        # psi: (dim, N_c) — sparse.mm requires sparse @ dense
        return torch.sparse.mm(H, psi)
    return H @ psi


def compute_pinn_loss(
    pinn: "torch.nn.Module",
    H_matrix: torch.Tensor,
    psi_0: torch.Tensor,
    t_colloc: torch.Tensor,
    t_circuit: Optional[torch.Tensor] = None,
    psi_circuit: Optional[torch.Tensor] = None,
    weights: Optional[dict[str, float]] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute composite PINN loss for the Schrödinger equation.

    The three loss components:
        L_ic:      ‖ψ(0) - ψ_0‖² — initial condition residual
        L_pde:     mean ‖iℏ dψ/dt - H ψ‖² over collocation points
        L_circuit: mean ‖ψ(T_k) - ψ_circuit_k‖² over circuit checkpoints
        L_norm:    mean (‖ψ(t)‖ - 1)² normalization penalty

    Args:
        pinn: PINNNetwork instance.
        H_matrix: Dense Hamiltonian matrix, shape (2^n, 2^n), complex128 or float64.
            If float64 (real symmetric), it is cast to complex128 internally.
        psi_0: Initial state vector, shape (2^n,), complex128.
        t_colloc: Collocation time points for PDE residual, shape (N_c,).
            Must require_grad or will be detached/re-created.
        t_circuit: Optional circuit checkpoint times, shape (N_T,).
        psi_circuit: Optional target states at circuit checkpoints, shape (N_T, 2^n), complex.
        weights: Dict with optional keys 'ic', 'pde', 'circuit', 'norm'.
            Defaults: {'ic': 1.0, 'pde': 1.0, 'circuit': 1.0, 'norm': 0.1}.

    Returns:
        (total_loss, loss_dict) where loss_dict has keys 'ic', 'pde', 'circuit', 'norm'.
    """
    w = {'ic': 1.0, 'pde': 1.0, 'circuit': 1.0, 'norm': 0.1}
    if weights is not None:
        w.update(weights)

    # Work in complex64 throughout (matches network output dtype)
    cdtype = torch.complex64
    if not H_matrix.is_complex():
        H = H_matrix.to(dtype=cdtype)
    else:
        H = H_matrix.to(dtype=cdtype)

    if not psi_0.is_complex():
        psi_0_c = psi_0.to(dtype=cdtype)
    else:
        psi_0_c = psi_0.to(dtype=cdtype)

    losses: dict[str, torch.Tensor] = {}

    # --- L_ic: initial condition ---
    t0 = torch.zeros(1, dtype=t_colloc.dtype, device=t_colloc.device)
    psi_t0 = pinn.as_complex(t0)[0]  # (dim,)
    losses['ic'] = (psi_t0 - psi_0_c).abs().pow(2).mean()

    # --- L_pde: PDE residual via autograd ---
    # Need t to require grad for computing dψ/dt
    t_c = t_colloc.detach().requires_grad_(True)
    psi_c_ri = pinn.forward(t_c)  # (N_c, dim, 2) real-imag split

    psi_c_complex = torch.view_as_complex(psi_c_ri.contiguous())  # (N_c, dim)

    # iℏ dψ/dt = H ψ  →  residual = iℏ dψ/dt - H ψ
    dpsi_dt_real = _batch_jacobian(psi_c_ri[..., 0], t_c)  # (N_c, dim)
    dpsi_dt_imag = _batch_jacobian(psi_c_ri[..., 1], t_c)  # (N_c, dim)
    dpsi_dt = torch.complex(dpsi_dt_real, dpsi_dt_imag)     # (N_c, dim)

    # iℏ dψ/dt  (ℏ=1 in natural units)
    lhs = 1j * dpsi_dt  # (N_c, dim)

    # H ψ: (N_c, dim) via batched matmul (supports dense and sparse CSR)
    rhs = _H_matmul(H, psi_c_complex.T).T  # (N_c, dim)

    pde_residual = (lhs - rhs).abs().pow(2)  # (N_c, dim)
    losses['pde'] = pde_residual.mean()

    # --- L_norm ---
    losses['norm'] = pinn.normalization_penalty(t_colloc)

    # --- L_circuit ---
    if t_circuit is not None and psi_circuit is not None:
        psi_pred = pinn.as_complex(t_circuit)  # (N_T, dim)
        if not psi_circuit.is_complex():
            psi_circ_c = psi_circuit.to(dtype=cdtype)
        else:
            psi_circ_c = psi_circuit.to(dtype=cdtype)
        losses['circuit'] = (psi_pred - psi_circ_c).abs().pow(2).mean()
    else:
        losses['circuit'] = torch.zeros(1, device=t_colloc.device)[0]

    total = (
        w['ic'] * losses['ic']
        + w['pde'] * losses['pde']
        + w['circuit'] * losses['circuit']
        + w['norm'] * losses['norm']
    )
    return total, losses


def _batch_jacobian(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian diag: d(output[n, :]) / d(input[n]) for each n.

    Args:
        output: shape (N, D), depends on input.
        input:  shape (N,), requires_grad=True.

    Returns:
        Jacobian of shape (N, D).
    """
    N, D = output.shape
    jac_rows = []
    for d in range(D):
        grad = torch.autograd.grad(
            output[:, d].sum(), input, create_graph=True, retain_graph=True
        )[0]  # (N,)
        jac_rows.append(grad)
    return torch.stack(jac_rows, dim=1)  # (N, D)
