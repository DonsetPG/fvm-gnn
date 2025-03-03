import random
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .fvm_step import fvm_1step
from .geometry import (
    calculate_centroids_volumes_2d_tri,
    calculate_edge_geometry_2d,
    calculate_masked_flux_term_edge_attr,
    find_mesh_connectivity_2d_tri,
)

import meshio
import pygmsh


@dataclass
class ProcessedTopology:
    pos: torch.Tensor
    edge_index: torch.Tensor
    volumes: torch.Tensor
    edge_lengths: torch.Tensor
    edge_deltas: torch.Tensor
    boundary_mask: torch.Tensor
    num_nodes: int


def assemble_node_features(
    temperatures: torch.Tensor,
    boundary_mask: torch.Tensor,
    source_scaled: torch.Tensor,
    inv_volumes: torch.Tensor,
) -> torch.Tensor:
    boundary_feature = boundary_mask.float().unsqueeze(1)
    return torch.cat(
        [temperatures, boundary_feature, source_scaled, inv_volumes], dim=1
    )


def sample_random_source(
    boundary_mask: torch.Tensor,
    num_nodes: int,
    source_value_min: float,
    source_value_max: float,
) -> torch.Tensor:
    source = torch.zeros(num_nodes, 1)
    interior_indices = torch.where(~boundary_mask)[0]
    if interior_indices.numel():
        picked = random.choice(interior_indices.tolist())
        source[picked] = random.uniform(source_value_min, source_value_max)
    return source


def _build_minimal_adjacent_topology(
    *,
    jitter: bool = False,
    jitter_scale: float = 0,
    rng: Optional[np.random.Generator] = None,
) -> ProcessedTopology:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.5, 1.0],
            [1.5, 1.0],
            [1.0, 2.0],
        ]
    )
    cells = np.array(
        [
            [0, 1, 3],
            [1, 4, 3],
            [1, 2, 4],
            [3, 4, 5],
        ]
    )

    centroids = None
    volumes = None
    if jitter:
        if rng is None:
            rng = np.random.default_rng(random.getrandbits(32))

        base_points = points.copy()
        for _ in range(10):
            jitter_offsets = rng.uniform(
                -jitter_scale, jitter_scale, size=base_points.shape
            )
            candidate_points = base_points + jitter_offsets
            centroids_candidate, volumes_candidate = calculate_centroids_volumes_2d_tri(
                candidate_points, cells
            )
            if torch.all(volumes_candidate > 1e-9):
                points = candidate_points
                centroids = centroids_candidate
                volumes = volumes_candidate
                break

    if centroids is None or volumes is None:
        centroids, volumes = calculate_centroids_volumes_2d_tri(points, cells)
    edge_index, face_verts, _, boundary_mask = find_mesh_connectivity_2d_tri(
        points, cells
    )
    edge_lengths, edge_deltas = calculate_edge_geometry_2d(
        points, centroids, edge_index, face_verts
    )

    return ProcessedTopology(
        pos=centroids,
        edge_index=edge_index,
        volumes=volumes,
        edge_lengths=edge_lengths,
        edge_deltas=edge_deltas,
        boundary_mask=boundary_mask,
        num_nodes=len(cells),
    )


def generate_minimal_graphs(
    num_samples_per_type: int = 50,
    alpha: float = 1.0,
    dt: float = 0.1,
    source_value_min: float = 5.0,
    source_value_max: float = 15.0,
) -> List[Data]:
    data_list: List[Data] = []
    topology_factories: Dict[str, Callable[[], ProcessedTopology]] = {
        "minimal_2_interior_adj": lambda: _build_minimal_adjacent_topology(jitter=True)
    }

    for name, build_topology in topology_factories.items():
        for _ in range(num_samples_per_type):
            geom = build_topology()
            inv_volumes = dt / torch.clamp(geom.volumes, min=1e-9)

            temperatures = torch.rand(geom.num_nodes, 1) * 10.0
            temperatures[geom.boundary_mask] = 0.0

            S_n = sample_random_source(
                geom.boundary_mask,
                geom.num_nodes,
                source_value_min,
                source_value_max,
            )
            source_feat = dt * S_n
            x = assemble_node_features(
                temperatures, geom.boundary_mask, source_feat, inv_volumes
            )

            edge_attr = calculate_masked_flux_term_edge_attr(
                x, geom.edge_index, alpha, geom.edge_lengths, geom.edge_deltas
            )

            T_np1 = fvm_1step(
                x[:, 0:1],
                S_n,
                geom.edge_index,
                geom.edge_lengths,
                geom.edge_deltas,
                geom.volumes,
                geom.boundary_mask,
                alpha,
                dt,
            )

            data_list.append(
                Data(
                    x=x,
                    edge_index=geom.edge_index,
                    edge_attr=edge_attr,
                    y=T_np1[:, 0:1],
                    pos=geom.pos,
                    volumes=geom.volumes,
                    edge_lengths=geom.edge_lengths,
                    edge_deltas=geom.edge_deltas,
                    boundary_mask=geom.boundary_mask,
                    graph_type=name,
                )
            )

    return data_list


def generate_ood_unstructured_graphs(
    num_samples: int = 50,
    alpha: float = 1.0,
    dt: float = 0.1,
    source_value_min: float = 5.0,
    source_value_max: float = 15.0,
    mesh_max_edge_size: float = 0.1,
) -> List[Data]:
    data_list: List[Data] = []
    for i in range(num_samples):
        with pygmsh.geo.Geometry() as geom:
            circle = geom.add_circle([0.0, 0.0, 0.0], 1.0, mesh_size=mesh_max_edge_size)
            geom.add_physical(circle.curve_loop.curves, label="boundary")
            geom.add_physical(geom.add_surface(circle.curve_loop), label="domain")
            mesh = geom.generate_mesh(dim=2, verbose=False)

        points = mesh.points[:, :2]
        cells = None
        for block in mesh.cells:
            if block.type == "triangle":
                cells = block.data
                break

        if cells is None or len(cells) == 0:
            warnings.warn(
                "No triangle cells in generated mesh; skipping sample.", RuntimeWarning
            )
            continue

        centroids, volumes = calculate_centroids_volumes_2d_tri(points, cells)
        inv_volumes = dt / torch.clamp(volumes, min=1e-9)
        edge_index, face_verts, _, boundary_mask = find_mesh_connectivity_2d_tri(
            points, cells
        )
        edge_lengths, edge_deltas = calculate_edge_geometry_2d(
            points, centroids, edge_index, face_verts
        )

        if edge_index.numel() == 0:
            warnings.warn(
                "Generated mesh has no adjacencies; skipping sample.", RuntimeWarning
            )
            continue

        temperatures = torch.zeros(len(cells), 1)
        distances = torch.norm(centroids - torch.tensor([[0.0, 0.0]]), dim=1)
        hot_mask = distances < 0.2
        temperatures[hot_mask] = 15.0
        temperatures[boundary_mask] = 0.0

        S_n = sample_random_source(
            boundary_mask, len(cells), source_value_min, source_value_max
        )
        source_feat = dt * S_n
        x = assemble_node_features(
            temperatures, boundary_mask, source_feat, inv_volumes
        )
        edge_attr = calculate_masked_flux_term_edge_attr(
            x, edge_index, alpha, edge_lengths, edge_deltas
        )

        T_np1 = fvm_1step(
            x[:, 0:1],
            S_n,
            edge_index,
            edge_lengths,
            edge_deltas,
            volumes,
            boundary_mask,
            alpha,
            dt,
        )

        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=T_np1[:, 0:1],
                pos=centroids,
                volumes=volumes,
                edge_lengths=edge_lengths,
                edge_deltas=edge_deltas,
                boundary_mask=boundary_mask,
                graph_type="ood_unstructured",
            )
        )

    return data_list


class MinimalFVMDataset(Dataset):
    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        num_samples: int = 150,
        alpha: float = 1.0,
        dt: float = 0.1,
        source_value_min: float = 5.0,
        source_value_max: float = 15.0,
    ):
        self.data_list = generate_minimal_graphs(
            num_samples, alpha, dt, source_value_min, source_value_max
        )
        super().__init__(root, transform, pre_transform)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


class OODFVMDataset(Dataset):
    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        num_samples: int = 50,
        alpha: float = 1.0,
        dt: float = 0.1,
        source_value_min: float = 5.0,
        source_value_max: float = 15.0,
        mesh_max_edge_size: float = 0.1,
    ):
        self.data_list = generate_ood_unstructured_graphs(
            num_samples,
            alpha,
            dt,
            source_value_min,
            source_value_max,
            mesh_max_edge_size,
        )
        super().__init__(root, transform, pre_transform)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]
