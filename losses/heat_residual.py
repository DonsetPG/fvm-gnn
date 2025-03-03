from typing import Literal

import torch

_EPS = 1e-9


def _ensure_col(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.view(-1, 1)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def _clamp_boundary(
    pred_T: torch.Tensor, T_b: torch.Tensor, boundary_mask: torch.Tensor
) -> torch.Tensor:
    clamped = pred_T.clone()
    if boundary_mask.any():
        clamped[boundary_mask] = T_b[boundary_mask]
    return clamped


def build_edge_weights(
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    edge_deltas: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    edge_lengths = _ensure_col(edge_lengths)
    edge_deltas = torch.clamp(_ensure_col(edge_deltas), min=_EPS)
    weights = edge_lengths / edge_deltas
    return alpha * weights


def apply_graph_laplacian(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    sym_norm: bool = False,
) -> torch.Tensor:
    x = _ensure_col(x)
    edge_weights = _ensure_col(edge_weights)
    row, col = edge_index

    if sym_norm:
        num_nodes = x.shape[0]
        degrees = torch.zeros((num_nodes, 1), device=x.device, dtype=x.dtype)
        degrees.scatter_add_(0, row.unsqueeze(1), edge_weights)
        degrees = torch.clamp(degrees, min=_EPS)
        norm_weights = edge_weights / (degrees[row] * degrees[col]).sqrt()
        effective = norm_weights
    else:
        effective = edge_weights

    sum_w_xn = torch.zeros_like(x)
    sum_w_xn.scatter_add_(0, row.unsqueeze(1), effective * x[col])

    degrees_in = torch.zeros_like(x)
    degrees_in.scatter_add_(0, row.unsqueeze(1), effective)

    return sum_w_xn - degrees_in * x


def heat_residual(
    T_pred: torch.Tensor,
    T_n: torch.Tensor,
    S_n: torch.Tensor,
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    edge_deltas: torch.Tensor,
    boundary_mask: torch.Tensor,
    dt: float,
    alpha: float = 1.0,
    sym_norm: bool = False,
) -> torch.Tensor:

    T_pred = _ensure_col(T_pred)
    T_n = _ensure_col(T_n)
    S_n = _ensure_col(S_n)
    if boundary_mask.ndim > 1:
        boundary_mask = boundary_mask.view(-1)
    boundary_mask = boundary_mask.bool()

    T_b = T_n
    T_pred_eff = _clamp_boundary(T_pred, T_b, boundary_mask)

    edge_weights = build_edge_weights(
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        edge_deltas=edge_deltas,
        alpha=alpha,
    )
    laplacian_T = apply_graph_laplacian(
        T_pred_eff, edge_index=edge_index, edge_weights=edge_weights, sym_norm=sym_norm
    )

    residual = torch.zeros_like(T_pred)
    interior_mask = ~boundary_mask
    if interior_mask.any():
        residual[interior_mask] = (
            (T_pred[interior_mask] - T_n[interior_mask]) / dt
            - laplacian_T[interior_mask]
            - S_n[interior_mask]
        )

    return residual


def pde_residual_loss(
    T_pred_raw: torch.Tensor,
    T_n: torch.Tensor,
    S_n: torch.Tensor,
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    edge_deltas: torch.Tensor,
    boundary_mask: torch.Tensor,
    dt: float,
    alpha: float = 1.0,
    sym_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if boundary_mask.ndim > 1:
        boundary_mask = boundary_mask.view(-1)
    boundary_mask = boundary_mask.bool()

    residual = heat_residual(
        T_pred=T_pred_raw,
        T_n=T_n,
        S_n=S_n,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        edge_deltas=edge_deltas,
        boundary_mask=boundary_mask,
        dt=dt,
        alpha=alpha,
        sym_norm=sym_norm,
    )

    interior_mask = ~boundary_mask
    if interior_mask.any():
        residual_mse = (residual[interior_mask] ** 2).mean()
    else:
        residual_mse = residual.sum() * 0.0

    T_b = _ensure_col(T_n)
    T_pred_raw = _ensure_col(T_pred_raw)
    if boundary_mask.any():
        bc_mse = ((T_pred_raw[boundary_mask] - T_b[boundary_mask]) ** 2).mean()
    else:
        bc_mse = residual.sum() * 0.0

    return residual_mse, bc_mse, residual
