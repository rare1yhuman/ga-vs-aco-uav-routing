"""
Experiment Orchestrator — GA vs ACO Paper  (Parallelized)

Runs all paper trials (A-J) for both MGA-Enhanced and ACO.
50 independent runs per algorithm per trial, parallelized across CPU cores.

Runtime estimate on 12-core machine: ~1.5-2 hours total.

Usage:
    python run_experiment.py --trial all          # all paper trials
    python run_experiment.py --trial A B C        # subset
    python run_experiment.py --trial all --runs 5 # quick test
    python run_experiment.py --sanity             # 2-run smoke-test
    python run_experiment.py --workers 8          # limit parallelism
    python run_experiment.py --output results     # paper-ready output folder
    python run_experiment.py --clean-output       # replace previous results
    python run_experiment.py --params config/paper_params.json
    python run_experiment.py --resume             # continue a stopped run
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from grid_gen      import generate_grid, DEPOT_POS, CELL_NFZ, CELL_SIZE_M
from graph_builder import build_graph
from dijkstra      import build_dijkstra_tour
from aco           import run_aco, compute_distance_matrix
from memetic_ga    import run_memetic_ga, build_distance_matrix
from energy_model  import tour_energy_2d, battery_summary, leg_energy_3d
from dynamic_nfz   import build_default_dynamic_nfzs, render_dynamic_nfzs
from environment_3d import Environment3D
from statistical_tests import run_all_tests, print_latex_table
import networkx as nx


# ── Parameters ───────────────────────────────────────────────────────────────

N_RUNS             = 50
N_WORKERS          = max(1, cpu_count() - 1)   # leave 1 core free
RANDOM_SEED_OFFSET = 2000

# Optimised defaults — fast enough for real runs, still statistically valid
DEFAULT_GA_PARAMS = {
    'pop_size':           200,   # restore original
    'generations':        500,   # restore original
    'crossover_rate':     0.85,
    'mutation_rate':      0.15,  # restore original
    'tournament_k':       7,
    'elitism_frac':       0.05,
    'two_opt_max_iter':   150,   # compromise (was 100, original 200)
    'two_opt_every_n':    3,
    'or_opt_every_n':     5,
    'local_search_top_frac': 0.10,
    'initial_local_search_frac': 0.15,
    'local_search_candidate_k': 8,
    'final_refine_candidate_k': 20,
    'exhaustive_final_max_n': 25,
    'final_refine_best':  True,
    'early_stop_patience': None,
    'stagnation_window':  25,
    'adaptive_mut_boost': 3.0,
    'inject_frac':        0.20,
    'inject_after':       50,
    'crossover_ops':      ('ox1', 'pmx', 'erx'),
    'crossover_probs':    (0.45, 0.25, 0.30),
    'dynamic_risk_weight': 1.0,
}

DEFAULT_ACO_PARAMS = {
    'num_ants':   200,
    'max_iter':   500,
    'alpha':      1.0,
    'beta':       2.0,
    'rho_global': 0.1,
    'rho_local':  0.01,
    'q0':         0.9,
    'candidate_list_size': 15,
}

# ── Trial Definitions ────────────────────────────────────────────────────────

TRIALS = {
    'A': dict(seed=42, n_deliveries=20, nfz_count=18, dynamic=False, use_3d=False,
              label='2D Baseline — seed 42'),
    'B': dict(seed=7,  n_deliveries=20, nfz_count=18, dynamic=False, use_3d=False,
              label='2D Baseline — seed 7'),
    'C': dict(seed=99, n_deliveries=20, nfz_count=18, dynamic=False, use_3d=False,
              label='2D Baseline — seed 99'),
    'D': dict(seed=42, n_deliveries=20, nfz_count=0, has_nfz=False, dynamic=False, use_3d=False,
              label='2D Control — no NFZ'),
    'E': dict(seed=42, n_deliveries=20, nfz_count=10, n_linear=4, n_osc=4,
              dynamic=True, use_3d=False, label='2D Dynamic NFZ — seed 42'),
    'F': dict(seed=7, n_deliveries=20, nfz_count=6, n_linear=6, n_osc=6,
              dynamic=True, use_3d=False, label='2D Heavy Dynamic NFZ — seed 7'),
    'G': dict(seed=42, n_deliveries=20, nfz_count=18, dynamic=False, use_3d=True,
              label='3D Environment — seed 42'),
    'H': dict(seed=7,  n_deliveries=20, nfz_count=18, n_linear=3, n_osc=3,
              dynamic=True, use_3d=True, label='3D + Dynamic Air Corridors — seed 7'),
    'I': dict(seed=42, n_deliveries=50, nfz_count=18, dynamic=False, use_3d=True,
              label='3D Scalability 50pts — seed 42'),
    'J': dict(seed=99, n_deliveries=50, nfz_count=18, n_linear=4, n_osc=4,
              dynamic=True, use_3d=True, label='3D + Dynamic + 50pts ULTIMATE — seed 99'),
}

MODEL_NOTES = {
    "dynamic_nfz": (
        "Dynamic NFZ trials optimise on the first connected NFZ snapshot, then "
        "evaluate every returned route with a deterministic time-stepped flight "
        "simulation. If a moving NFZ blocks a leg, the evaluator replans that "
        "leg from the current cell and records collisions/replans."
    ),
    "3d": (
        "3D trials use a voxel environment, altitude-restricted columns, air "
        "corridors, and altitude-dependent energy. Route realisation lifts each "
        "2D path cell to the minimum safe altitude with climb/descent accounting."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Environment Builder  (runs once per trial — shared across all 50 runs)
# ═══════════════════════════════════════════════════════════════════════════

def _check_graph_connectivity(grid, depot, delivery_points):
    """Build graph and verify all delivery points are reachable from depot."""
    G = build_graph(grid)
    if depot not in G:
        return G, False
    for dp in delivery_points:
        if dp not in G or not nx.has_path(G, depot, dp):
            return G, False
    return G, True


def _apply_3d_altitude_penalties(G, env3d):
    """
    Modify edge weights to reflect 3D altitude costs.
    
    For each free cell, compute the required flight altitude based on
    the tallest building within a proximity radius. Drones flying near
    tall buildings must ascend to clear them, adding energy cost.
    
    This creates a 'terrain difficulty map' that makes 3D trials
    meaningfully harder than their 2D equivalents.
    
    Penalty model:
      base_weight × (1 + 0.20 × required_altitude)
    
    z=0 → ×1.0 (open area, no buildings nearby)
    z=2 → ×1.4 (near short building)
    z=5 → ×2.0 (near tall building — doubles effective distance)
    """
    ALTITUDE_PENALTY_FACTOR = 0.20   # per altitude layer
    required_alt = env3d.required_altitude_map()
    
    # Step 3: Apply altitude penalty to each edge
    modified = 0
    for u, v, data in G.edges(data=True):
        z_u = required_alt[u[0], u[1]]
        z_v = required_alt[v[0], v[1]]
        z_max = max(z_u, z_v)
        if z_max > 0:
            data['weight'] *= (1.0 + ALTITUDE_PENALTY_FACTOR * z_max)
            modified += 1
    return modified


def build_trial_environment(trial_id: str, cfg: dict) -> dict:
    print(f"\n[{trial_id}] Building environment: {cfg['label']}")

    grid, dps, nfzs = generate_grid(
        seed=cfg['seed'],
        has_nfz=cfg.get('has_nfz', True),
        num_deliveries=cfg['n_deliveries'],
        nfz_count=cfg['nfz_count'],
    )
    static_grid = grid.copy()

    # ── Dynamic NFZs with connectivity validation ────────────────────────
    dyn_nfzs = []
    dyn_start_t = 0
    if cfg.get('dynamic'):
        dyn_nfzs = build_default_dynamic_nfzs(
            seed=cfg['seed'],
            n_linear=cfg.get('n_linear', 4),
            n_oscillating=cfg.get('n_osc', 4)
        )
        
        # Try rendering at t=0, then t=1..10 if graph becomes disconnected
        grid_with_dyn = None
        for t_init in range(11):
            candidate = render_dynamic_nfzs(grid, dyn_nfzs, t=t_init)
            _, connected = _check_graph_connectivity(candidate, DEPOT_POS, dps)
            if connected:
                grid_with_dyn = candidate
                dyn_start_t = t_init
                if t_init > 0:
                    print(f"[{trial_id}]   Dynamic NFZs at t=0 disconnected graph; "
                          f"using t={t_init} instead")
                break
        
        if grid_with_dyn is None:
            # Last resort: regenerate dynamic NFZs with perturbed seed
            for seed_offset in range(1, 20):
                dyn_nfzs = build_default_dynamic_nfzs(
                    seed=cfg['seed'] + seed_offset,
                    n_linear=cfg.get('n_linear', 4),
                    n_oscillating=cfg.get('n_osc', 4)
                )
                candidate = render_dynamic_nfzs(grid, dyn_nfzs, t=0)
                _, connected = _check_graph_connectivity(candidate, DEPOT_POS, dps)
                if connected:
                    grid_with_dyn = candidate
                    dyn_start_t = 0
                    print(f"[{trial_id}]   Regenerated dynamic NFZs with "
                          f"seed={cfg['seed']+seed_offset} for connectivity")
                    break
        
        if grid_with_dyn is None:
            raise RuntimeError(
                f"Trial {trial_id}: Could not find connected graph with dynamic NFZs"
            )
        grid = grid_with_dyn
    
    # ── Build graph ──────────────────────────────────────────────────────
    env3d = None
    if cfg.get('use_3d'):
        env3d = Environment3D(seed=cfg['seed'], nfz_blocks_2d=nfzs)

    G = build_graph(grid)
    
    # ── Apply 3D altitude penalties to edge weights ──────────────────────
    if env3d is not None:
        n_modified = _apply_3d_altitude_penalties(G, env3d)
        print(f"[{trial_id}]   3D altitude penalties applied to {n_modified} edges")
    
    all_nodes = [DEPOT_POS] + list(dps)

    print(f"[{trial_id}]   Precomputing {len(all_nodes)}×{len(all_nodes)} shortest paths...")
    t0        = time.perf_counter()
    dist_dict = compute_distance_matrix(G, all_nodes)
    dist_mat  = build_distance_matrix(dps, DEPOT_POS, dist_dict)
    print(f"[{trial_id}]   Precompute done in {time.perf_counter()-t0:.1f}s")

    dijk          = build_dijkstra_tour(G, DEPOT_POS, dps)
    dijkstra_dist = dijk['total_distance_m']
    print(f"[{trial_id}]   Dijkstra baseline: {dijkstra_dist:.1f} m")

    return {
        'grid': grid, 'G': G, 'dps': dps, 'nfzs': nfzs,
        'static_grid': static_grid, 'dyn_nfzs': dyn_nfzs,
        'dyn_start_t': dyn_start_t, 'env3d': env3d,
        'dist_dict': dist_dict, 'dist_matrix': dist_mat,
        'dijkstra_dist': dijkstra_dist, 'cfg': cfg,
    }


def _winner_label(mga_mean: float, aco_mean: float, tol: float = 1e-9) -> str:
    if abs(mga_mean - aco_mean) <= tol:
        return 'TIE'
    return 'MGA' if mga_mean < aco_mean else 'ACO'


def _route_waypoints(chrom: List[int], dps: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not chrom:
        return []
    return [DEPOT_POS] + [dps[int(i)] for i in chrom] + [DEPOT_POS]


def _path_distance_m(path: List[Tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(
        np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) * CELL_SIZE_M
        for i in range(len(path) - 1)
    ))


def _line_cells(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Integer cells on a straight segment, used only for cheap dynamic-risk scoring."""
    r1, c1 = a
    r2, c2 = b
    steps = int(max(abs(r2 - r1), abs(c2 - c1)))
    if steps <= 0:
        return [a]
    rows = np.rint(np.linspace(r1, r2, steps + 1)).astype(int)
    cols = np.rint(np.linspace(c1, c2, steps + 1)).astype(int)
    return list(dict.fromkeys(zip(rows.tolist(), cols.tolist())))


def _build_dynamic_risk_matrix(env: Dict[str, Any],
                               sample_times: Tuple[int, ...] = tuple(range(0, 241, 12))
                               ) -> np.ndarray:
    """
    Build a cheap pairwise penalty matrix for dynamic trials.

    This is not the final evaluator. It is a lightweight fitness tie-breaker that
    nudges MGA away from legs whose straight corridor is frequently occupied by
    moving NFZs. Raw planned distance remains the reported primary metric.
    """
    nodes = [DEPOT_POS] + list(env['dps'])
    n = len(nodes)
    risk = np.zeros((n, n), dtype=float)
    if not env['cfg'].get('dynamic') or not env.get('dyn_nfzs'):
        return risk

    grids = [
        render_dynamic_nfzs(env['static_grid'], env['dyn_nfzs'], t=env['dyn_start_t'] + t)
        for t in sample_times
    ]

    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i == j:
                continue
            cells = _line_cells(src, dst)
            if not cells:
                continue
            hits = 0
            checks = 0
            for grid_t in grids:
                for cell in cells:
                    checks += 1
                    if grid_t[cell] == CELL_NFZ:
                        hits += 1
            blocked_fraction = hits / max(checks, 1)
            base_dist = env['dist_matrix'][i, j]
            if np.isfinite(base_dist):
                risk[i, j] = blocked_fraction * min(base_dist * 0.15, 3000.0)
    return risk


def _objective_matrix(dist_matrix: np.ndarray,
                      risk_matrix: np.ndarray = None,
                      risk_weight: float = 0.0) -> np.ndarray:
    """Shared optimizer objective for MGA and ACO."""
    obj = np.array(dist_matrix, dtype=float, copy=True)
    if risk_matrix is not None and risk_weight > 0:
        finite = np.isfinite(obj)
        obj[finite] = obj[finite] + risk_weight * risk_matrix[finite]
    return obj


def _matrix_to_dist_dict(nodes: List[Tuple[int, int]], matrix: np.ndarray) -> Dict:
    """Convert an all-points matrix back to the tuple-key dict used by ACO."""
    out = {}
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i != j:
                out[(src, dst)] = float(matrix[i, j])
    return out


def _route_cost_matrix(chrom: List[int], matrix: np.ndarray) -> float:
    if not chrom:
        return 0.0
    order = [0] + [int(g) + 1 for g in chrom] + [0]
    return float(sum(matrix[order[i], order[i + 1]] for i in range(len(order) - 1)))


def _shortest_path_or_none(grid: np.ndarray,
                           src: Tuple[int, int],
                           dst: Tuple[int, int]) -> List[Tuple[int, int]]:
    G = build_graph(grid)
    if src not in G or dst not in G:
        return []
    try:
        return nx.shortest_path(G, src, dst, weight='weight')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _realize_static_route(chrom: List[int], dps: List[Tuple[int, int]],
                          G: nx.Graph) -> Tuple[List[Tuple[int, int]], str]:
    waypoints = _route_waypoints(chrom, dps)
    if len(waypoints) < 2:
        return [], "empty_route"
    full_path: List[Tuple[int, int]] = []
    for i in range(len(waypoints) - 1):
        try:
            seg = nx.shortest_path(G, waypoints[i], waypoints[i + 1], weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            return full_path, str(exc)
        full_path.extend(seg if not full_path else seg[1:])
    return full_path, ""


def _realize_dynamic_route(chrom: List[int], dps: List[Tuple[int, int]],
                           static_grid: np.ndarray, dyn_nfzs: list,
                           start_t: int) -> Dict[str, Any]:
    waypoints = _route_waypoints(chrom, dps)
    if len(waypoints) < 2:
        return {
            'path': [], 'collision_count': 0, 'replan_count': 0,
            'dynamic_feasible': False, 'time_to_complete_s': 0.0,
            'error': 'empty_route',
        }

    full_path: List[Tuple[int, int]] = []
    current_time = int(start_t)
    collision_count = 0
    replan_count = 0

    for leg_idx in range(len(waypoints) - 1):
        current = waypoints[leg_idx]
        target = waypoints[leg_idx + 1]
        attempts = 0
        wait_steps = 0
        while current != target:
            attempts += 1
            if attempts > 40:
                return {
                    'path': full_path, 'collision_count': collision_count,
                    'replan_count': replan_count, 'dynamic_feasible': False,
                    'time_to_complete_s': float(current_time - start_t),
                    'error': f'leg_{leg_idx}_replan_limit',
                }

            grid_t = render_dynamic_nfzs(static_grid, dyn_nfzs, t=current_time)
            seg = _shortest_path_or_none(grid_t, current, target)
            if not seg:
                wait_steps += 1
                if wait_steps <= 30:
                    replan_count += 1
                    current_time += 1
                    continue
                return {
                    'path': full_path, 'collision_count': collision_count,
                    'replan_count': replan_count, 'dynamic_feasible': False,
                    'time_to_complete_s': float(current_time - start_t),
                    'error': f'leg_{leg_idx}_no_path_at_t_{current_time}',
                }

            blocked_at = None
            for step, cell in enumerate(seg[1:], start=1):
                future_grid = render_dynamic_nfzs(static_grid, dyn_nfzs, t=current_time + step)
                if future_grid[cell] == CELL_NFZ:
                    blocked_at = step
                    break

            if blocked_at is None:
                full_path.extend(seg if not full_path else seg[1:])
                current_time += max(0, len(seg) - 1)
                current = target
            else:
                collision_count += 1
                replan_count += 1
                safe_prefix = seg[:blocked_at]
                if safe_prefix:
                    full_path.extend(safe_prefix if not full_path else safe_prefix[1:])
                    current = safe_prefix[-1]
                    current_time += max(1, len(safe_prefix) - 1)
                else:
                    current_time += 1

    return {
        'path': full_path,
        'collision_count': collision_count,
        'replan_count': replan_count,
        'dynamic_feasible': True,
        'time_to_complete_s': float(current_time - start_t),
        'error': "",
    }


def _lift_path_to_3d(path2d: List[Tuple[int, int]], env3d: Environment3D) -> Dict[str, Any]:
    if not path2d:
        return {
            'trajectory_3d': [], 'max_altitude_m': 0.0,
            'mean_altitude_m': 0.0, 'climb_m': 0.0,
            'energy_3d_j': float('inf'), 'path_dimension': '3d',
            'error': 'empty_3d_path',
        }

    required_altitude = env3d.required_altitude_map()
    trajectory = []
    for r, c in path2d:
        z = int(required_altitude[r, c])
        if z < 0:
            return {
                'trajectory_3d': trajectory, 'max_altitude_m': 0.0,
                'mean_altitude_m': 0.0, 'climb_m': 0.0,
                'energy_3d_j': float('inf'), 'path_dimension': '3d',
                'error': f'blocked_column_{r}_{c}',
            }
        trajectory.append((int(r), int(c), int(z)))

    climb_m = 0.0
    energy = 0.0
    charge_left = None
    for i in range(len(trajectory) - 1):
        r1, c1, z1 = trajectory[i]
        r2, c2, z2 = trajectory[i + 1]
        climb_m += max(0, z2 - z1) * 10.0
        energy += leg_energy_3d(r1, c1, z1, r2, c2, z2)

    alts = [p[2] * 10.0 for p in trajectory]
    return {
        'trajectory_3d': trajectory,
        'max_altitude_m': float(max(alts)),
        'mean_altitude_m': float(np.mean(alts)),
        'climb_m': float(climb_m),
        'energy_3d_j': float(energy),
        'path_dimension': '3d',
        'error': "",
    }


def _evaluate_route(chrom: List[int], env: Dict[str, Any]) -> Dict[str, Any]:
    cfg = env['cfg']
    if cfg.get('dynamic'):
        dyn = _realize_dynamic_route(
            chrom, env['dps'], env['static_grid'], env['dyn_nfzs'], env['dyn_start_t']
        )
        path2d = dyn['path']
        error = dyn['error']
    else:
        path2d, error = _realize_static_route(chrom, env['dps'], env['G'])
        dyn = {
            'collision_count': 0,
            'replan_count': 0,
            'dynamic_feasible': not bool(error),
            'time_to_complete_s': float(max(0, len(path2d) - 1)),
        }

    distance_realized = _path_distance_m(path2d)
    metrics = {
        'realized_distance_m': distance_realized,
        'collision_count': dyn['collision_count'],
        'replan_count': dyn['replan_count'],
        'dynamic_feasible': dyn['dynamic_feasible'],
        'time_to_complete_s': dyn['time_to_complete_s'],
        'path_dimension': '2d',
        'max_altitude_m': 0.0,
        'mean_altitude_m': 0.0,
        'climb_m': 0.0,
        'trajectory_3d': [],
        'evaluation_error': error,
    }

    if env.get('env3d') is not None:
        lift = _lift_path_to_3d(path2d, env['env3d'])
        metrics.update({
            'path_dimension': lift['path_dimension'],
            'max_altitude_m': lift['max_altitude_m'],
            'mean_altitude_m': lift['mean_altitude_m'],
            'climb_m': lift['climb_m'],
            'trajectory_3d': lift['trajectory_3d'],
        })
        if np.isfinite(lift['energy_3d_j']):
            metrics['energy_j'] = lift['energy_3d_j']
        if lift['error']:
            metrics['evaluation_error'] = lift['error']
    return metrics


def _annotate_evaluations(results: List[Dict[str, Any]], env: Dict[str, Any]) -> None:
    for result in results:
        if result.get('error') or not result.get('best_path'):
            result.update({
                'realized_distance_m': result.get('distance_m', float('inf')),
                'collision_count': 0,
                'replan_count': 0,
                'dynamic_feasible': False,
                'time_to_complete_s': 0.0,
                'path_dimension': '3d' if env.get('env3d') is not None else '2d',
                'max_altitude_m': 0.0,
                'mean_altitude_m': 0.0,
                'climb_m': 0.0,
                'trajectory_3d': [],
                'evaluation_error': result.get('error') or 'missing_best_path',
            })
            continue

        metrics = _evaluate_route(result['best_path'], env)
        if 'energy_j' in metrics:
            result['energy_j'] = metrics.pop('energy_j')
            bat = battery_summary(result['energy_j'])
            result['battery_used_%'] = bat['battery_used_%']
            result['feasible'] = bat['feasible']
        result.update(metrics)
        if result['evaluation_error']:
            result['feasible'] = False


# ═══════════════════════════════════════════════════════════════════════════
# Worker functions  (called by multiprocessing.Pool — must be top-level)
# ═══════════════════════════════════════════════════════════════════════════

def _mga_worker(args):
    """Top-level worker for MGA — required for multiprocessing pickling."""
    if len(args) == 7:
        dist_matrix, obj_matrix, dist_dict, dps, ga_params, run_seed, risk_matrix = args
    else:
        dist_matrix, dist_dict, dps, ga_params, run_seed = args
        risk_matrix = None
        obj_matrix = dist_matrix
    import random as _random
    n_genes = len(dps)

    params = {**ga_params, 'seed': run_seed}
    dynamic_risk_weight = float(params.get('dynamic_risk_weight', 0.0))

    def planned_distance(chrom):
        return _route_cost_matrix(chrom, dist_matrix)

    def dynamic_risk(chrom):
        if risk_matrix is None or dynamic_risk_weight <= 0:
            return 0.0
        return dynamic_risk_weight * _route_cost_matrix(chrom, risk_matrix)

    def fitness(chrom):
        return planned_distance(chrom) + dynamic_risk(chrom)

    try:
        dp_matrix = dist_matrix[1:, 1:]
        params['objective_matrix'] = obj_matrix
        
        result = run_memetic_ga(
            n_genes=n_genes, fitness_fn=fitness,
            dist_matrix=dp_matrix, params=params, verbose=False,
        )
        best_chrom = result['best_chromosome']
        best_dist  = planned_distance(best_chrom)   # report raw planned distance
        best_score = result['best_fitness']
        risk_penalty = dynamic_risk(best_chrom)

        pl = {(k[0], k[1]): v for k, v in dist_dict.items()}
        energy = tour_energy_2d(dps, best_chrom, pl, DEPOT_POS)
        bat    = battery_summary(energy)

        return {
            'seed':                 run_seed,
            'distance_m':           best_dist,
            'optimization_score_m':  best_score,
            'dynamic_risk_penalty_m': risk_penalty,
            'energy_j':             energy,
            'battery_used_%':       bat['battery_used_%'],
            'feasible':             bat['feasible'],
            'convergence_gen':      result['convergence_gen'],
            'two_opt_improvements': result.get('two_opt_improvements', 0),
            'or_opt_improvements':  result.get('or_opt_improvements', 0),
            'diversity_injections': result.get('diversity_injections', 0),
            'crossover_counts':      json.dumps(result.get('crossover_counts', {})),
            'fitness_evaluations':    result.get('fitness_evaluations', 0),
            'runtime_s':            result['runtime_s'],
            'best_path':            list(best_chrom),
            'convergence_curve':    json.dumps([float(v) for v in result['history']]),
            'error': None,
        }
    except Exception as e:
        return {'seed': run_seed, 'distance_m': float('inf'), 'energy_j': float('inf'),
                'optimization_score_m': float('inf'), 'dynamic_risk_penalty_m': 0.0,
                'battery_used_%': 999, 'feasible': False,
                'convergence_gen': -1, 'two_opt_improvements': 0,
                'or_opt_improvements': 0, 'diversity_injections': 0,
                'crossover_counts': '{}',
                'runtime_s': 0, 'best_path': [], 'convergence_curve': '[]',
                'error': str(e)}


def _aco_worker(args):
    """Top-level worker for ACO — required for multiprocessing pickling."""
    dps, dist_dict, obj_dist_dict, dijkstra_dist, aco_params, run_seed, risk_matrix = args
    import random as _random
    import numpy as _np

    _np.random.seed(run_seed)
    _random.seed(run_seed)

    try:
        result = run_aco(
            G=None, grid=None, depot=DEPOT_POS, delivery_points=dps,
            dijkstra_baseline=dijkstra_dist, verbose=False, params=aco_params,
            precomputed_distances=obj_dist_dict,
            build_full_path=False,
            compute_heatmap=False,
        )
        best_score = result['best_tour_distance']
        tour_order = result['best_tour_order']

        dp_set = {dp: i for i, dp in enumerate(dps)}
        chrom  = [dp_set[p] for p in tour_order if p in dp_set]
        pl     = {(k[0], k[1]): v for k, v in dist_dict.items()}
        nodes = [DEPOT_POS] + list(dps)
        dist_matrix = build_distance_matrix(dps, DEPOT_POS, pl)
        best_dist = _route_cost_matrix(chrom, dist_matrix) if chrom else float('inf')
        risk_penalty = _route_cost_matrix(chrom, risk_matrix) if chrom and risk_matrix is not None else 0.0
        energy = tour_energy_2d(dps, chrom, pl, DEPOT_POS) if chrom else 0.0
        bat    = battery_summary(energy)

        return {
            'seed':             run_seed,
            'distance_m':      best_dist,
            'optimization_score_m': best_score,
            'dynamic_risk_penalty_m': risk_penalty,
            'energy_j':        energy,
            'battery_used_%':  bat['battery_used_%'],
            'feasible':        bat['feasible'],
            'convergence_gen': result['convergence_iteration'],
            'runtime_s':       result['compute_time_s'],
            'best_path':       chrom,
            'convergence_curve': json.dumps([float(v) for v in result['convergence_curve']]),
            'error': None,
        }
    except Exception as e:
        return {'seed': run_seed, 'distance_m': float('inf'), 'energy_j': float('inf'),
                'optimization_score_m': float('inf'), 'dynamic_risk_penalty_m': 0.0,
                'battery_used_%': 999, 'feasible': False,
                'convergence_gen': -1, 'runtime_s': 0,
                'best_path': [], 'convergence_curve': '[]', 'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# Trial Runner  (parallelised)
# ═══════════════════════════════════════════════════════════════════════════

def run_trial(trial_id: str, cfg: dict, ga_params: dict,
              n_runs: int, output_dir: Path, n_workers: int,
              aco_params: dict = None,
              strict_errors: bool = True) -> dict:

    print(f"\n{'='*60}")
    print(f"TRIAL {trial_id}: {cfg['label']}")
    print(f"  {n_runs} runs × 2 algos, {n_workers} workers")
    print(f"{'='*60}")

    env = build_trial_environment(trial_id, cfg)
    dps = env['dps']
    aco_params = aco_params or DEFAULT_ACO_PARAMS
    dynamic_risk_weight = float(ga_params.get('dynamic_risk_weight', 0.0))
    risk_matrix = _build_dynamic_risk_matrix(env) if cfg.get('dynamic') else None
    obj_matrix = _objective_matrix(env['dist_matrix'], risk_matrix, dynamic_risk_weight)
    all_nodes = [DEPOT_POS] + list(dps)
    obj_dist_dict = _matrix_to_dist_dict(all_nodes, obj_matrix)

    seeds = [RANDOM_SEED_OFFSET + r for r in range(n_runs)]

    # ── MGA parallel runs ────────────────────────────────────────────────────
    mga_args = [
        (env['dist_matrix'], obj_matrix, env['dist_dict'], dps, ga_params, s, risk_matrix)
        for s in seeds
    ]
    print(f"  Running {n_runs} MGA runs on {n_workers} cores...")
    t0 = time.perf_counter()
    with Pool(n_workers) as pool:
        mga_results = list(tqdm(
            pool.imap(_mga_worker, mga_args), total=n_runs, desc=f"  MGA-{trial_id}"
        ))
    print(f"  MGA done in {time.perf_counter()-t0:.1f}s")
    _annotate_evaluations(mga_results, env)

    # ── ACO parallel runs ────────────────────────────────────────────────────
    aco_args = [
        (dps, env['dist_dict'], obj_dist_dict, env['dijkstra_dist'], aco_params, s, risk_matrix)
        for s in seeds
    ]
    print(f"  Running {n_runs} ACO runs on {n_workers} cores...")
    t0 = time.perf_counter()
    with Pool(n_workers) as pool:
        aco_results = list(tqdm(
            pool.imap(_aco_worker, aco_args), total=n_runs, desc=f"  ACO-{trial_id}"
        ))
    print(f"  ACO done in {time.perf_counter()-t0:.1f}s")
    _annotate_evaluations(aco_results, env)

    # ── Save results ─────────────────────────────────────────────────────────
    trial_dir = output_dir / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    mga_records = [
        {'trial': trial_id, 'run': i, 'algo': 'MGA', 'algorithm': 'MGA', **r}
        for i, r in enumerate(mga_results)
    ]
    aco_records = [
        {'trial': trial_id, 'run': i, 'algo': 'ACO', 'algorithm': 'ACO', **r}
        for i, r in enumerate(aco_results)
    ]
    df_all = pd.concat([pd.DataFrame(mga_records), pd.DataFrame(aco_records)])
    _atomic_csv(df_all, trial_dir / 'raw_results.csv')

    def summary(recs):
        dists = [r['distance_m'] for r in recs if np.isfinite(r['distance_m'])]
        realized = [r['realized_distance_m'] for r in recs if np.isfinite(r.get('realized_distance_m', float('inf')))]
        scores = [r.get('optimization_score_m', r['distance_m']) for r in recs
                  if np.isfinite(r.get('optimization_score_m', r['distance_m']))]
        risk_penalties = [r.get('dynamic_risk_penalty_m', 0.0) for r in recs]
        energies = [r['energy_j'] for r in recs if np.isfinite(r['energy_j'])]
        runtimes = [r['runtime_s'] for r in recs if np.isfinite(r['runtime_s'])]
        if not dists:
            dists = [float('inf')]
        return {
            'mean':          float(np.mean(dists)),
            'std':           float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
            'best':          float(np.min(dists)),
            'worst':         float(np.max(dists)),
            'valid_runs':     len(dists) if np.isfinite(dists[0]) else 0,
            'mean_realized_distance_m': float(np.mean(realized)) if realized else float('inf'),
            'mean_optimization_score_m': float(np.mean(scores)) if scores else float('inf'),
            'mean_dynamic_risk_penalty_m': float(np.mean(risk_penalties)) if risk_penalties else 0.0,
            'mean_energy_j': float(np.mean(energies)) if energies else float('inf'),
            'mean_runtime_s':float(np.mean(runtimes)) if runtimes else 0.0,
            'feasible_pct':  float(100 * np.mean([r['feasible'] for r in recs])),
            'dynamic_feasible_pct': float(100 * np.mean([r.get('dynamic_feasible', False) for r in recs])),
            'mean_collisions': float(np.mean([r.get('collision_count', 0) for r in recs])),
            'mean_replans': float(np.mean([r.get('replan_count', 0) for r in recs])),
            'mean_max_altitude_m': float(np.mean([r.get('max_altitude_m', 0.0) for r in recs])),
            'mean_climb_m': float(np.mean([r.get('climb_m', 0.0) for r in recs])),
            'errors':        sum(1 for r in recs if r.get('error')),
            'evaluation_errors': sum(1 for r in recs if r.get('evaluation_error')),
        }

    mga_summary = summary(mga_results)
    aco_summary = summary(aco_results)
    complete = (
        mga_summary['valid_runs'] == n_runs and
        aco_summary['valid_runs'] == n_runs and
        mga_summary['errors'] == 0 and
        aco_summary['errors'] == 0
    )

    result = {
        'trial':              trial_id,
        'label':              cfg['label'],
        'n_runs':             n_runs,
        'status':             'complete' if complete else 'failed',
        'dijkstra_baseline_m':env['dijkstra_dist'],
        'MGA':                mga_summary,
        'ACO':                aco_summary,
        'mga_distances':      [r['distance_m'] for r in mga_results if np.isfinite(r['distance_m'])],
        'aco_distances':      [r['distance_m'] for r in aco_results if np.isfinite(r['distance_m'])],
        'model_notes':        MODEL_NOTES,
    }

    _atomic_json(trial_dir / 'summary.json', result)

    m, a = result['MGA'], result['ACO']
    winner = _winner_label(m['mean'], a['mean'])
    pct    = 100 * (a['mean'] - m['mean']) / max(a['mean'], 1e-6)
    print(f"\n  MGA: {m['mean']:.0f} ± {m['std']:.0f} m  (best={m['best']:.0f})")
    print(f"  ACO: {a['mean']:.0f} ± {a['std']:.0f} m  (best={a['best']:.0f})")
    print(f"  Winner: {winner}  ({pct:+.1f}%)")

    if strict_errors and not complete:
        raise RuntimeError(
            f"Trial {trial_id} incomplete: "
            f"MGA valid/errors={mga_summary['valid_runs']}/{mga_summary['errors']}, "
            f"ACO valid/errors={aco_summary['valid_runs']}/{aco_summary['errors']}. "
            f"Raw details saved to {trial_dir / 'raw_results.csv'}."
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Master Runner
# ═══════════════════════════════════════════════════════════════════════════

def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return 'not-a-git-repo'


def _write_run_metadata(output_dir: Path, trial_ids: List[str], n_runs: int,
                        n_workers: int, ga_params: Dict[str, Any],
                        aco_params: Dict[str, Any],
                        status: str,
                        params_source: str = 'built-in defaults') -> None:
    metadata = {
        'status': status,
        'command': ' '.join(sys.argv),
        'trial_ids': trial_ids,
        'n_runs_per_algorithm_per_trial': n_runs,
        'n_workers': n_workers,
        'random_seed_offset': RANDOM_SEED_OFFSET,
        'ga_params': ga_params,
        'aco_params': aco_params,
        'params_source': params_source,
        'params_hash': _params_hash(ga_params, aco_params),
        'trial_definitions': {tid: TRIALS[tid] for tid in trial_ids},
        'model_notes': MODEL_NOTES,
        'python': sys.version,
        'platform': platform.platform(),
        'pid': os.getpid(),
        'git_revision': _git_revision(),
        'timestamp_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    _atomic_json(output_dir / 'run_metadata.json', metadata)


def _normalise_aco_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = {**DEFAULT_ACO_PARAMS, **(params or {})}
    if 'n_ants' in p:
        p['num_ants'] = p['n_ants']
    if 'n_iterations' in p:
        p['max_iter'] = p['n_iterations']
    if 'rho' in p:
        p['rho_global'] = p['rho']
    return {
        'num_ants': int(p['num_ants']),
        'max_iter': int(p['max_iter']),
        'alpha': float(p['alpha']),
        'beta': float(p['beta']),
        'rho_global': float(p['rho_global']),
        'rho_local': float(p.get('rho_local', 0.01)),
        'q0': float(p['q0']),
        'candidate_list_size': (
            None if p.get('candidate_list_size') in (None, 0, "none")
            else int(p.get('candidate_list_size'))
        ),
    }


def _normalise_ga_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = {**DEFAULT_GA_PARAMS, **(params or {})}
    for key in ('pop_size', 'generations', 'tournament_k', 'two_opt_max_iter',
                'two_opt_every_n', 'or_opt_every_n', 'stagnation_window',
                'inject_after', 'local_search_candidate_k',
                'final_refine_candidate_k', 'exhaustive_final_max_n'):
        if p.get(key) is not None:
            p[key] = int(p[key])
    for key in ('crossover_rate', 'mutation_rate', 'elitism_frac',
                'adaptive_mut_boost', 'inject_frac', 'local_search_top_frac',
                'initial_local_search_frac', 'dynamic_risk_weight'):
        p[key] = float(p[key])
    p['final_refine_best'] = bool(p.get('final_refine_best', True))
    return p


def _params_hash(ga_params: Dict[str, Any], aco_params: Dict[str, Any]) -> str:
    payload = json.dumps({'ga': ga_params, 'aco': aco_params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_protocol_params(params_path: str = None, output_dir: Path = None) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    candidates = []
    if params_path:
        candidates.append(Path(params_path))
    if output_dir is not None:
        candidates.append(Path(output_dir) / 'best_params.json')

    for path in candidates:
        if path and path.exists():
            with open(path) as f:
                payload = json.load(f)
            return (
                _normalise_ga_params(payload.get('ga', DEFAULT_GA_PARAMS)),
                _normalise_aco_params(payload.get('aco', DEFAULT_ACO_PARAMS)),
                str(path),
            )

    return _normalise_ga_params(DEFAULT_GA_PARAMS), _normalise_aco_params(DEFAULT_ACO_PARAMS), 'built-in defaults'


def _load_completed_trial(output_dir: Path, tid: str, n_runs: int) -> Dict[str, Any]:
    summary_path = output_dir / f"trial_{tid}" / "summary.json"
    raw_path = output_dir / f"trial_{tid}" / "raw_results.csv"
    if not summary_path.exists() or not raw_path.exists():
        return {}
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        raw = pd.read_csv(raw_path)
        counts = raw.groupby('algo')['run'].nunique().to_dict()
        if (
            summary.get('status') == 'complete'
            and int(summary.get('n_runs', 0)) == n_runs
            and counts.get('MGA', 0) == n_runs
            and counts.get('ACO', 0) == n_runs
        ):
            return summary
    except Exception:
        return {}
    return {}


def run_all_trials(trial_ids: List[str], n_runs: int, output_dir: Path,
                   n_workers: int, strict_errors: bool = True,
                   clean_output: bool = False,
                   params_path: str = None,
                   resume: bool = False) -> None:
    if clean_output and resume:
        raise ValueError("--resume cannot be used with --clean-output")
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    ga_params, aco_params, params_source = _load_protocol_params(params_path, output_dir)
    if params_source == 'built-in defaults':
        print("[run_experiment] Using default params (run parameter_tuning.py first for best results)")
    else:
        print(f"[run_experiment] Loaded params from {params_source}")

    print(f"[run_experiment] {n_workers} workers, {n_runs} runs/algo/trial")
    print(f"[run_experiment] GA: pop={ga_params['pop_size']}, "
          f"gens={ga_params['generations']}, 2opt_every={ga_params['two_opt_every_n']}")
    print(f"[run_experiment] ACO: ants={aco_params['num_ants']}, "
          f"iters={aco_params['max_iter']}, alpha={aco_params['alpha']}, beta={aco_params['beta']}")
    _write_run_metadata(output_dir, trial_ids, n_runs, n_workers, ga_params, aco_params, 'running', params_source)

    all_results = {}
    mga_all, aco_all = {}, {}
    failures = {}

    master_pbar = tqdm(trial_ids, desc="Overall Experiment Progress", unit="trial", position=0)
    
    for tid in master_pbar:
        master_pbar.set_description(f"Overall Progress (Running Trial {tid})")
        try:
            result = _load_completed_trial(output_dir, tid, n_runs) if resume else {}
            if result:
                print(f"\n[resume] Trial {tid} already complete; using existing results.")
            else:
                result = run_trial(tid, TRIALS[tid], ga_params, n_runs, output_dir,
                                   n_workers, aco_params=aco_params,
                                   strict_errors=strict_errors)
            all_results[tid] = result
            mga_all[tid]     = result['mga_distances']
            aco_all[tid]     = result['aco_distances']
        except Exception as exc:
            print(f"\n[ERROR] Trial {tid} failed:")
            traceback.print_exc()
            failures[tid] = str(exc)

    # Statistical tests
    if len(mga_all) >= 2:
        print("\n" + "="*60)
        print("STATISTICAL TESTS")
        print("="*60)
        stat_df = run_all_tests(mga_all, aco_all, output_dir)
        print(stat_df[['trial','winner','pct_improvement',
                        'p_wilcoxon','p_wilcoxon_holm',
                        'cohens_d','cliffs_delta','effect_size']].to_string(index=False))
        print_latex_table(stat_df)

    # Master CSV
    rows = []
    for tid, res in all_results.items():
        m, a = res['MGA'], res['ACO']
        rows.append({
            'trial': tid, 'label': res['label'], 'n_runs': res['n_runs'],
            'status': res['status'],
            'dijkstra_m':     res['dijkstra_baseline_m'],
            'mga_mean_m':     round(m['mean'], 1),
            'mga_std_m':      round(m['std'], 1),
            'mga_best_m':     round(m['best'], 1),
            'mga_realized_m': round(m['mean_realized_distance_m'], 1),
            'mga_score_m':    round(m['mean_optimization_score_m'], 1),
            'mga_risk_penalty_m': round(m['mean_dynamic_risk_penalty_m'], 1),
            'mga_energy_j':   round(m['mean_energy_j'], 0),
            'mga_runtime_s':  round(m['mean_runtime_s'], 2),
            'mga_feasible_%': round(m['feasible_pct'], 1),
            'mga_dynamic_feasible_%': round(m['dynamic_feasible_pct'], 1),
            'mga_collisions': round(m['mean_collisions'], 2),
            'mga_replans': round(m['mean_replans'], 2),
            'mga_max_altitude_m': round(m['mean_max_altitude_m'], 1),
            'mga_climb_m': round(m['mean_climb_m'], 1),
            'aco_mean_m':     round(a['mean'], 1),
            'aco_std_m':      round(a['std'], 1),
            'aco_best_m':     round(a['best'], 1),
            'aco_realized_m': round(a['mean_realized_distance_m'], 1),
            'aco_score_m':    round(a['mean_optimization_score_m'], 1),
            'aco_risk_penalty_m': round(a['mean_dynamic_risk_penalty_m'], 1),
            'aco_energy_j':   round(a['mean_energy_j'], 0),
            'aco_runtime_s':  round(a['mean_runtime_s'], 2),
            'aco_dynamic_feasible_%': round(a['dynamic_feasible_pct'], 1),
            'aco_collisions': round(a['mean_collisions'], 2),
            'aco_replans': round(a['mean_replans'], 2),
            'aco_max_altitude_m': round(a['mean_max_altitude_m'], 1),
            'aco_climb_m': round(a['mean_climb_m'], 1),
            'winner':         _winner_label(m['mean'], a['mean']),
            'pct_diff':       round(100*(a['mean']-m['mean'])/max(a['mean'],1e-6), 2),
        })

    _atomic_csv(pd.DataFrame(rows), output_dir / 'master_results.csv')
    if failures:
        _atomic_json(output_dir / 'failed_trials.json', failures)
    else:
        stale_failed = output_dir / 'failed_trials.json'
        if stale_failed.exists():
            stale_failed.unlink()
    elapsed = time.perf_counter() - t_total
    final_status = 'complete' if not failures and len(all_results) == len(trial_ids) else 'failed'
    _write_run_metadata(output_dir, trial_ids, n_runs, n_workers, ga_params, aco_params, final_status, params_source)
    print(f"\n[run_experiment] Total: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print(f"[run_experiment] Results in: {output_dir.resolve()}")
    if failures:
        print(f"[run_experiment] FAILED trials: {sorted(failures)}")
        if strict_errors:
            raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial',   nargs='+', default=['all'])
    parser.add_argument('--runs',    type=int, default=N_RUNS)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    parser.add_argument('--output',  default='results')
    parser.add_argument('--params', default=None,
                        help='Parameter JSON outside results/. Use config/paper_params.json for paper runs.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip complete trial folders and rerun only missing/failed trials. Incompatible with --clean-output.')
    parser.add_argument('--clean-output', action='store_true',
                        help='Delete the selected output directory before running. Use for intentional fresh paper runs.')
    parser.add_argument('--allow-errors', action='store_true',
                        help='Write summaries even when some runs fail. Not recommended for paper results.')
    parser.add_argument('--sanity',  action='store_true',
                        help='Quick smoke-test: Trial A, 2 runs')
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.sanity:
        print("=== SANITY CHECK (Trial A, 2 runs) ===")
        run_all_trials(['A'], n_runs=2, output_dir=output_dir, n_workers=2,
                       strict_errors=not args.allow_errors,
                       clean_output=args.clean_output,
                       params_path=args.params,
                       resume=args.resume)
        return

    trial_ids = list(TRIALS.keys()) if 'all' in args.trial else args.trial
    invalid   = [t for t in trial_ids if t not in TRIALS]
    if invalid:
        print(f"[ERROR] Unknown trials: {invalid}. Valid: {list(TRIALS.keys())}")
        return

    run_all_trials(trial_ids, n_runs=args.runs,
                   output_dir=output_dir, n_workers=args.workers,
                   strict_errors=not args.allow_errors,
                   clean_output=args.clean_output,
                   params_path=args.params,
                   resume=args.resume)


if __name__ == '__main__':
    main()
