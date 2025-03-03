import torch


def fvm_1step(
    T_n: torch.Tensor,
    S_n: torch.Tensor,
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    edge_deltas: torch.Tensor,
    node_volumes: torch.Tensor,
    boundary_mask: torch.Tensor,
    alpha: float,
    dt: float,
) -> torch.Tensor:
    num_nodes = T_n.size(0)
    T_nplus1 = T_n.clone()

    interior_mask = ~boundary_mask

    row, col = edge_index
    valid_edges_mask = interior_mask[row]
    valid_row = row[valid_edges_mask]
    valid_col = col[valid_edges_mask]

    delta_PN = torch.clamp(edge_deltas[valid_edges_mask], min=1e-9)
    A_f = torch.clamp(edge_lengths[valid_edges_mask], min=1e-9)
    geom_conductivity = alpha * A_f / delta_PN

    temp_diff = T_n[valid_col] - T_n[valid_row]
    flux_terms = geom_conductivity * temp_diff

    total_flux_into_P = torch.zeros(num_nodes, 1, device=T_n.device)
    total_flux_into_P.scatter_add_(
        0, valid_row.unsqueeze(1).expand_as(flux_terms), flux_terms
    )

    V_P_interior = torch.clamp(node_volumes[interior_mask], min=1e-9)
    S_n_interior = S_n[interior_mask]

    T_nplus1[interior_mask] = T_n[interior_mask] + dt * (
        total_flux_into_P[interior_mask] / V_P_interior + S_n_interior
    )

    return T_nplus1
