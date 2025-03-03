import warnings
from typing import Dict, List, Set, Tuple

import numpy as np
import torch


def calculate_centroids_volumes_2d_tri(
    points: np.ndarray,
    cells: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:

    triangles = points[cells]
    centroids = np.mean(triangles, axis=1)

    term1 = triangles[:, 0, 0] * (triangles[:, 1, 1] - triangles[:, 2, 1])
    term2 = triangles[:, 1, 0] * (triangles[:, 2, 1] - triangles[:, 0, 1])
    term3 = triangles[:, 2, 0] * (triangles[:, 0, 1] - triangles[:, 1, 1])
    areas = 0.5 * np.abs(term1 + term2 + term3)

    return (
        torch.tensor(centroids, dtype=torch.float),
        torch.tensor(areas, dtype=torch.float).unsqueeze(1),
    )


def find_mesh_connectivity_2d_tri(
    points: np.ndarray,
    cells: np.ndarray,
) -> Tuple[
    torch.Tensor, torch.Tensor, Dict[Tuple[int, int], Tuple[int, int]], torch.Tensor
]:

    num_cells = len(cells)
    face_to_cells: Dict[Tuple[int, int], List[int]] = {}

    for cell_idx, vertices in enumerate(cells):
        for i in range(3):
            face = tuple(sorted((vertices[i], vertices[(i + 1) % 3])))
            face_to_cells.setdefault(face, []).append(cell_idx)

    edge_list: List[List[int]] = []
    cell_adj_faces_dict: Dict[Tuple[int, int], Tuple[int, int]] = {}
    boundary_cells: Set[int] = set()

    for face, adjacent_cells in face_to_cells.items():
        if len(adjacent_cells) == 2:
            c1, c2 = adjacent_cells
            edge_list.append([c1, c2])
            edge_list.append([c2, c1])
            cell_adj_faces_dict[tuple(sorted((c1, c2)))] = face
        elif len(adjacent_cells) == 1:
            boundary_cells.add(adjacent_cells[0])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        face_vertices = torch.tensor(
            [
                list(
                    cell_adj_faces_dict[
                        tuple(
                            sorted((edge_index[0, i].item(), edge_index[1, i].item()))
                        )
                    ]
                )
                for i in range(edge_index.size(1))
            ],
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        face_vertices = torch.empty((0, 2), dtype=torch.long)

    boundary_mask = torch.zeros(num_cells, dtype=torch.bool)
    if boundary_cells:
        boundary_mask[list(boundary_cells)] = True

    return edge_index, face_vertices, cell_adj_faces_dict, boundary_mask


def calculate_edge_geometry_2d(
    points: np.ndarray,
    centroids: torch.Tensor,
    edge_index: torch.Tensor,
    face_vertex_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if edge_index.numel() == 0:
        return torch.empty((0, 1)), torch.empty((0, 1))

    centroids_p = centroids[edge_index[0]]
    centroids_n = centroids[edge_index[1]]
    delta_PN = torch.norm(centroids_p - centroids_n, dim=1, keepdim=True)
    edge_deltas = torch.clamp(delta_PN, min=1e-9)

    points_tensor = torch.tensor(points, dtype=torch.float)
    v1 = points_tensor[face_vertex_indices[:, 0]]
    v2 = points_tensor[face_vertex_indices[:, 1]]
    face_lengths = torch.norm(v1 - v2, dim=1, keepdim=True)
    face_lengths = torch.clamp(face_lengths, min=1e-9)

    return face_lengths, edge_deltas


def calculate_masked_flux_term_edge_attr(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    edge_lengths: torch.Tensor,
    edge_deltas: torch.Tensor,
) -> torch.Tensor:

    if edge_index.numel() == 0:
        return torch.empty((0, 1), dtype=x.dtype, device=x.device)

    sender_indices = edge_index[1]
    receiver_indices = edge_index[0]

    T_senders = x[sender_indices, 0:1]
    T_receivers = x[receiver_indices, 0:1]
    temp_diff = T_senders - T_receivers

    edge_deltas_clamped = torch.clamp(edge_deltas, min=1e-9)
    geom_coeff = alpha * torch.clamp(edge_lengths, min=1e-9) / edge_deltas_clamped
    flux_term = geom_coeff * temp_diff

    sender_is_boundary = x[sender_indices, 1:2] >= 0.5
    masked_flux = flux_term.masked_fill(sender_is_boundary, 0.0)

    return masked_flux
