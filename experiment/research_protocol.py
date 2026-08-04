"""Research-grade protocol definitions and shared environment construction.

This module is deliberately separate from ``run_experiment.py``. The original
A--J pipeline remains available for reproducing the first completed benchmark,
while this protocol uses independent environment replication and time-indexed
dynamic exposure costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np

from dynamic_nfz import build_default_dynamic_nfzs, render_dynamic_nfzs
from environment_3d import Environment3D
from graph_builder import build_graph
from grid_gen import CELL_NFZ, CELL_SIZE_M, DEPOT_POS, generate_grid
from run_experiment import _apply_3d_altitude_penalties, _check_graph_connectivity


FINAL_ENVIRONMENT_SEEDS = (42, 88, 99, 123, 219, 220, 221, 226, 229, 236)
FINAL_OPTIMIZER_SEEDS = (2000, 2001, 2002, 2003, 2004)
CALIBRATION_ENVIRONMENT_SEEDS = (314160, 271830)
CALIBRATION_OPTIMIZER_SEEDS = (9100, 9101, 9102, 9103, 9104)

TIME_BIN_WIDTH_STEPS = 50
TIME_BIN_COUNT = 32
EXPOSURE_DETOUR_M_PER_CELL = 2.0 * CELL_SIZE_M


SCENARIO_FAMILIES: Dict[str, Dict[str, Any]] = {
    "S1": {
        "label": "2D control",
        "n_deliveries": 20,
        "nfz_count": 0,
        "has_nfz": False,
        "dynamic": False,
        "use_3d": False,
        "n_linear": 0,
        "n_osc": 0,
    },
    "S2": {
        "label": "2D static",
        "n_deliveries": 20,
        "nfz_count": 18,
        "has_nfz": True,
        "dynamic": False,
        "use_3d": False,
        "n_linear": 0,
        "n_osc": 0,
    },
    "S3": {
        "label": "2D dynamic",
        "n_deliveries": 20,
        "nfz_count": 10,
        "has_nfz": True,
        "dynamic": True,
        "use_3d": False,
        "n_linear": 4,
        "n_osc": 4,
    },
    "S4": {
        "label": "3D-aware static",
        "n_deliveries": 20,
        "nfz_count": 18,
        "has_nfz": True,
        "dynamic": False,
        "use_3d": True,
        "n_linear": 0,
        "n_osc": 0,
    },
    "S5": {
        "label": "3D-aware dynamic",
        "n_deliveries": 20,
        "nfz_count": 18,
        "has_nfz": True,
        "dynamic": True,
        "use_3d": True,
        "n_linear": 3,
        "n_osc": 3,
    },
    "S6": {
        "label": "3D-aware static scalability",
        "n_deliveries": 50,
        "nfz_count": 18,
        "has_nfz": True,
        "dynamic": False,
        "use_3d": True,
        "n_linear": 0,
        "n_osc": 0,
    },
    "S7": {
        "label": "3D-aware dynamic scalability",
        "n_deliveries": 50,
        "nfz_count": 18,
        "has_nfz": True,
        "dynamic": True,
        "use_3d": True,
        "n_linear": 4,
        "n_osc": 4,
    },
}


CALIBRATION_FAMILIES: Dict[str, Dict[str, Any]] = {
    "C20": {**SCENARIO_FAMILIES["S3"], "label": "Calibration dynamic 20"},
    "C50": {**SCENARIO_FAMILIES["S7"], "label": "Calibration 3D-aware dynamic 50"},
}


class InvalidEnvironmentSeed(RuntimeError):
    """The predeclared environment is not initially feasible."""


@dataclass(frozen=True)
class TimeIndexedCost:
    """Shared generalized cost used by both MGA and ACO."""

    node_to_index: Mapping[Tuple[int, int], int]
    base_cost_matrix: np.ndarray
    physical_distance_matrix: np.ndarray
    duration_matrix: np.ndarray
    cost_tensor: np.ndarray
    exposure_tensor: np.ndarray
    bin_width_steps: int = TIME_BIN_WIDTH_STEPS

    def time_bin(self, elapsed_steps: int) -> int:
        return min(
            max(int(elapsed_steps) // int(self.bin_width_steps), 0),
            self.cost_tensor.shape[0] - 1,
        )

    def leg_by_index(self, src: int, dst: int, elapsed_steps: int) -> Tuple[float, int]:
        time_bin = self.time_bin(elapsed_steps)
        return (
            float(self.cost_tensor[time_bin, src, dst]),
            int(self.duration_matrix[src, dst]),
        )

    def leg_by_point(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        elapsed_steps: int,
    ) -> Tuple[float, int]:
        return self.leg_by_index(
            int(self.node_to_index[src]),
            int(self.node_to_index[dst]),
            elapsed_steps,
        )

    def route_metrics(self, chromosome: Sequence[int]) -> Dict[str, float]:
        order = [0] + [int(gene) + 1 for gene in chromosome] + [0]
        elapsed = 0
        generalized_cost = 0.0
        base_cost = 0.0
        physical_distance = 0.0
        exposure_cells = 0.0
        for src, dst in zip(order, order[1:]):
            time_bin = self.time_bin(elapsed)
            generalized_cost += float(self.cost_tensor[time_bin, src, dst])
            base_cost += float(self.base_cost_matrix[src, dst])
            physical_distance += float(self.physical_distance_matrix[src, dst])
            exposure_cells += float(self.exposure_tensor[time_bin, src, dst])
            elapsed += int(self.duration_matrix[src, dst])
        return {
            "generalized_cost": generalized_cost,
            "base_cost": base_cost,
            "physical_distance_m": physical_distance,
            "exposure_cells": exposure_cells,
            "estimated_duration_steps": float(elapsed),
        }


def _all_pairs_paths(
    graph: nx.Graph,
    nodes: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    n_nodes = len(nodes)
    matrix = np.full((n_nodes, n_nodes), np.inf, dtype=float)
    paths: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for src_idx, source in enumerate(nodes):
        lengths, source_paths = nx.single_source_dijkstra(graph, source, weight="weight")
        for dst_idx, target in enumerate(nodes):
            if target not in lengths:
                continue
            matrix[src_idx, dst_idx] = float(lengths[target])
            paths[(src_idx, dst_idx)] = list(source_paths[target])
    return matrix, paths


def _physical_path_matrix(
    nodes: Sequence[Tuple[int, int]],
    paths: Mapping[Tuple[int, int], Sequence[Tuple[int, int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    n_nodes = len(nodes)
    distance = np.full((n_nodes, n_nodes), np.inf, dtype=float)
    duration = np.zeros((n_nodes, n_nodes), dtype=int)
    for src_idx in range(n_nodes):
        for dst_idx in range(n_nodes):
            path = paths.get((src_idx, dst_idx), ())
            if not path:
                continue
            duration[src_idx, dst_idx] = max(len(path) - 1, 0)
            distance[src_idx, dst_idx] = duration[src_idx, dst_idx] * CELL_SIZE_M
    return distance, duration


def _initial_dynamic_snapshot_is_connected(
    static_grid: np.ndarray,
    dynamic_nfzs: Sequence[Any],
    delivery_points: Sequence[Tuple[int, int]],
) -> bool:
    initial_grid = render_dynamic_nfzs(static_grid, list(dynamic_nfzs), t=0)
    _, connected = _check_graph_connectivity(
        initial_grid, DEPOT_POS, list(delivery_points)
    )
    return bool(connected)


def _dynamic_exposure_tensor(
    static_grid: np.ndarray,
    dynamic_nfzs: Sequence[Any],
    paths: Mapping[Tuple[int, int], Sequence[Tuple[int, int]]],
    n_nodes: int,
    bin_count: int,
    bin_width_steps: int,
) -> np.ndarray:
    exposure = np.zeros((bin_count, n_nodes, n_nodes), dtype=np.float32)
    if not dynamic_nfzs:
        return exposure

    max_path_steps = max((len(path) - 1 for path in paths.values()), default=0)
    max_time = (bin_count - 1) * bin_width_steps + max_path_steps
    occupancy = np.empty(
        (max_time + 1, static_grid.shape[0], static_grid.shape[1]),
        dtype=np.bool_,
    )
    for time_step in range(max_time + 1):
        occupancy[time_step] = (
            render_dynamic_nfzs(
                static_grid, list(dynamic_nfzs), t=int(time_step)
            )
            == CELL_NFZ
        )

    for time_bin in range(bin_count):
        departure = time_bin * bin_width_steps
        for src_idx in range(n_nodes):
            for dst_idx in range(n_nodes):
                if src_idx == dst_idx:
                    continue
                path = paths.get((src_idx, dst_idx), ())
                if not path:
                    exposure[time_bin, src_idx, dst_idx] = np.inf
                    continue
                path_array = np.asarray(path[1:], dtype=int)
                steps = np.arange(1, len(path), dtype=int)
                blocked = occupancy[
                    departure + steps,
                    path_array[:, 0],
                    path_array[:, 1],
                ]
                exposure[time_bin, src_idx, dst_idx] = float(np.count_nonzero(blocked))
    return exposure


def build_research_environment(
    family_id: str,
    family: Mapping[str, Any],
    environment_seed: int,
    *,
    bin_count: int = TIME_BIN_COUNT,
    bin_width_steps: int = TIME_BIN_WIDTH_STEPS,
) -> Dict[str, Any]:
    """Build one independent map instance and all shared optimizer inputs."""
    grid, delivery_points, nfz_blocks = generate_grid(
        seed=int(environment_seed),
        has_nfz=bool(family.get("has_nfz", True)),
        num_deliveries=int(family["n_deliveries"]),
        nfz_count=int(family["nfz_count"]),
    )
    static_grid = grid.copy()
    static_graph = build_graph(static_grid)
    if DEPOT_POS not in static_graph or any(
        point not in static_graph or not nx.has_path(static_graph, DEPOT_POS, point)
        for point in delivery_points
    ):
        raise InvalidEnvironmentSeed(
            f"{family_id}/seed={environment_seed}: static graph is disconnected"
        )

    dynamic_nfzs: List[Any] = []
    if family.get("dynamic"):
        dynamic_nfzs = build_default_dynamic_nfzs(
            seed=int(environment_seed),
            n_linear=int(family.get("n_linear", 4)),
            n_oscillating=int(family.get("n_osc", 4)),
        )
        if not _initial_dynamic_snapshot_is_connected(
            static_grid, dynamic_nfzs, delivery_points
        ):
            raise InvalidEnvironmentSeed(
                f"{family_id}/seed={environment_seed}: t=0 dynamic graph is disconnected"
            )

    env3d = None
    cost_graph = build_graph(static_grid)
    if family.get("use_3d"):
        env3d = Environment3D(
            seed=int(environment_seed),
            nfz_blocks_2d=nfz_blocks,
        )
        _apply_3d_altitude_penalties(cost_graph, env3d)

    nodes = [DEPOT_POS] + list(delivery_points)
    base_cost_matrix, paths = _all_pairs_paths(cost_graph, nodes)
    if not np.isfinite(base_cost_matrix).all():
        raise InvalidEnvironmentSeed(
            f"{family_id}/seed={environment_seed}: waypoint matrix is incomplete"
        )
    physical_matrix, duration_matrix = _physical_path_matrix(nodes, paths)
    exposure_tensor = _dynamic_exposure_tensor(
        static_grid,
        dynamic_nfzs,
        paths,
        len(nodes),
        bin_count,
        bin_width_steps,
    )
    cost_tensor = (
        base_cost_matrix[np.newaxis, :, :]
        + exposure_tensor * EXPOSURE_DETOUR_M_PER_CELL
    )
    if not family.get("dynamic"):
        cost_tensor = cost_tensor[:1]
        exposure_tensor = exposure_tensor[:1]

    time_cost = TimeIndexedCost(
        node_to_index={node: idx for idx, node in enumerate(nodes)},
        base_cost_matrix=base_cost_matrix,
        physical_distance_matrix=physical_matrix,
        duration_matrix=duration_matrix,
        cost_tensor=cost_tensor,
        exposure_tensor=exposure_tensor,
        bin_width_steps=bin_width_steps,
    )

    physical_distances = {
        (nodes[src], nodes[dst]): float(physical_matrix[src, dst])
        for src in range(len(nodes))
        for dst in range(len(nodes))
        if src != dst
    }
    static_cost_distances = {
        (nodes[src], nodes[dst]): float(base_cost_matrix[src, dst])
        for src in range(len(nodes))
        for dst in range(len(nodes))
        if src != dst
    }

    return {
        "family_id": family_id,
        "family": dict(family),
        "environment_seed": int(environment_seed),
        "grid": static_grid,
        "delivery_points": list(delivery_points),
        "nfz_blocks": nfz_blocks,
        "dynamic_nfzs": dynamic_nfzs,
        "env3d": env3d,
        "nodes": nodes,
        "paths": paths,
        "physical_distances": physical_distances,
        "static_cost_distances": static_cost_distances,
        "time_cost": time_cost,
        "shared_precompute": {
            "time_bins": int(cost_tensor.shape[0]),
            "time_bin_width_steps": int(bin_width_steps),
            "exposure_detour_m_per_cell": float(EXPOSURE_DETOUR_M_PER_CELL),
        },
    }


def select_families(
    requested: Iterable[str],
    definitions: Mapping[str, Dict[str, Any]] = SCENARIO_FAMILIES,
) -> Dict[str, Dict[str, Any]]:
    requested = list(requested)
    if not requested or requested == ["all"]:
        return dict(definitions)
    unknown = sorted(set(requested) - set(definitions))
    if unknown:
        raise ValueError(f"Unknown scenario families: {unknown}")
    return {family_id: dict(definitions[family_id]) for family_id in requested}
