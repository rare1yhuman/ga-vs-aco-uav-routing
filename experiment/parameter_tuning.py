"""
Parameter tuning for the GA vs ACO experiment.

This script uses the same environment and current algorithm APIs as
run_experiment.py. It is intentionally slower than a smoke test because its
purpose is fairness: freeze tuned parameters before paper-grade experiments.
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from aco import compute_distance_matrix, run_aco
from dijkstra import build_dijkstra_tour
from graph_builder import build_graph
from grid_gen import DEPOT_POS, generate_grid
from memetic_ga import build_distance_matrix, run_memetic_ga


GA_PARAM_GRID = {
    'pop_size':       [100, 150, 200, 250, 300],
    'crossover_rate': [0.6, 0.7, 0.8, 0.9, 0.95],
    'mutation_rate':  [0.01, 0.05, 0.10, 0.15, 0.20],
    'tournament_k':   [3, 5, 7, 10],
}

ACO_PARAM_GRID = {
    'num_ants': [100, 150, 200],
    'alpha':  [0.5, 1.0, 1.5, 2.0],
    'beta':   [1.0, 2.0, 3.0, 5.0],
    'rho_global': [0.05, 0.1, 0.2],
    'q0':     [0.7, 0.8, 0.9],
}

TUNING_RUNS_PER_CONFIG = 10
TUNING_GENERATIONS = 150
TUNING_SEED_BASE = 1000
TUNING_GRID_SEED = 42
TUNING_N_DELIVERIES = 20


def build_tuning_environment(seed: int = TUNING_GRID_SEED) -> Dict[str, Any]:
    print(f"[tuning] Building tuning environment (seed={seed})...")
    grid, delivery_points, nfz_blocks = generate_grid(
        seed=seed,
        num_deliveries=TUNING_N_DELIVERIES,
    )
    graph = build_graph(grid)
    all_nodes = [DEPOT_POS] + list(delivery_points)
    path_lengths = compute_distance_matrix(graph, all_nodes)
    dist_matrix = build_distance_matrix(delivery_points, DEPOT_POS, path_lengths)
    dijkstra = build_dijkstra_tour(graph, DEPOT_POS, delivery_points)

    print(
        f"[tuning] Environment ready: {len(delivery_points)} deliveries, "
        f"{len(nfz_blocks)} NFZs, Dijkstra={dijkstra['total_distance_m']:.1f} m"
    )
    return {
        'grid': grid,
        'graph': graph,
        'delivery_points': delivery_points,
        'path_lengths': path_lengths,
        'dist_matrix': dist_matrix,
        'dijkstra_dist': dijkstra['total_distance_m'],
    }


def _tour_fitness(chrom, dist_matrix: np.ndarray) -> float:
    total = dist_matrix[0, chrom[0] + 1]
    total += float(sum(
        dist_matrix[chrom[i] + 1, chrom[i + 1] + 1]
        for i in range(len(chrom) - 1)
    ))
    total += dist_matrix[chrom[-1] + 1, 0]
    return float(total)


def tune_ga(env: Dict[str, Any], output_dir: Path,
            n_runs: int = TUNING_RUNS_PER_CONFIG) -> Dict[str, Any]:
    keys = list(GA_PARAM_GRID.keys())
    configs = list(itertools.product(*GA_PARAM_GRID.values()))
    dist_matrix = env['dist_matrix']
    dp_matrix = dist_matrix[1:, 1:]
    n_genes = len(env['delivery_points'])
    fitness_fn = lambda c: _tour_fitness(c, dist_matrix)

    print(f"\n[GA Tuning] {len(configs)} configs x {n_runs} runs")
    records = []
    for combo in tqdm(configs, desc="GA Tuning"):
        params = dict(zip(keys, combo))
        params['generations'] = TUNING_GENERATIONS
        run_fits = []

        for run in range(n_runs):
            run_params = {**params, 'seed': TUNING_SEED_BASE + run}
            result = run_memetic_ga(
                n_genes=n_genes,
                fitness_fn=fitness_fn,
                dist_matrix=dp_matrix,
                params=run_params,
                verbose=False,
            )
            run_fits.append(result['best_fitness'])

        records.append({
            **params,
            'mean_fitness': float(np.mean(run_fits)),
            'std_fitness': float(np.std(run_fits, ddof=1)) if len(run_fits) > 1 else 0.0,
            'min_fitness': float(np.min(run_fits)),
        })

    df = pd.DataFrame(records).sort_values('mean_fitness')
    outfile = output_dir / 'tuning_ga.csv'
    df.to_csv(outfile, index=False)
    best = df.iloc[0].to_dict()
    print(f"[GA Tuning] Results saved to {outfile}")
    print(f"[GA Tuning] Best config: {best}")
    return {k: best[k] for k in keys}


def _run_aco_with_params(env: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    run_params = {
        'num_ants': int(params['num_ants']),
        'max_iter': int(params['max_iter']),
        'alpha': float(params['alpha']),
        'beta': float(params['beta']),
        'rho_global': float(params['rho_global']),
        'rho_local': 0.01,
        'q0': float(params['q0']),
    }
    return run_aco(
        G=env['graph'],
        grid=env['grid'],
        depot=DEPOT_POS,
        delivery_points=env['delivery_points'],
        dijkstra_baseline=env['dijkstra_dist'],
        verbose=False,
        params=run_params,
    )


def tune_aco(env: Dict[str, Any], output_dir: Path,
             n_runs: int = TUNING_RUNS_PER_CONFIG) -> Dict[str, Any]:
    keys = list(ACO_PARAM_GRID.keys())
    configs = list(itertools.product(*ACO_PARAM_GRID.values()))

    print(f"\n[ACO Tuning] {len(configs)} configs x {n_runs} runs")
    records = []
    for combo in tqdm(configs, desc="ACO Tuning"):
        params = dict(zip(keys, combo))
        params['max_iter'] = TUNING_GENERATIONS
        run_fits = []

        for run in range(n_runs):
            seed = TUNING_SEED_BASE + run
            np.random.seed(seed)
            random.seed(seed)
            result = _run_aco_with_params(env, params)
            run_fits.append(result['best_tour_distance'])

        records.append({
            **params,
            'mean_fitness': float(np.mean(run_fits)),
            'std_fitness': float(np.std(run_fits, ddof=1)) if len(run_fits) > 1 else 0.0,
            'min_fitness': float(np.min(run_fits)),
        })

    df = pd.DataFrame(records).sort_values('mean_fitness')
    outfile = output_dir / 'tuning_aco.csv'
    df.to_csv(outfile, index=False)
    best = df.iloc[0].to_dict()
    print(f"[ACO Tuning] Results saved to {outfile}")
    print(f"[ACO Tuning] Best config: {best}")
    return {k: best[k] for k in keys}


def save_best_params(best_ga: Dict[str, Any], best_aco: Dict[str, Any],
                     output_dir: Path) -> Dict[str, Any]:
    best_ga_full = {
        **best_ga,
        'pop_size': int(best_ga.get('pop_size', 200)),
        'tournament_k': int(best_ga.get('tournament_k', 7)),
        'crossover_rate': float(best_ga.get('crossover_rate', 0.85)),
        'mutation_rate': float(best_ga.get('mutation_rate', 0.15)),
        'generations': 500,
        'elitism_frac': 0.05,
        'two_opt_max_iter': 150,
        'two_opt_every_n': 2,
        'or_opt_every_n': 3,
        'stagnation_window': 25,
        'adaptive_mut_boost': 3.0,
        'inject_frac': 0.20,
        'inject_after': 50,
    }
    best_aco_full = {
        **best_aco,
        'num_ants': int(best_aco.get('num_ants', 200)),
        'max_iter': 500,
        'alpha': float(best_aco.get('alpha', 1.0)),
        'beta': float(best_aco.get('beta', 2.0)),
        'rho_global': float(best_aco.get('rho_global', 0.1)),
        'rho_local': 0.01,
        'q0': float(best_aco.get('q0', 0.9)),
    }
    out = {
        'ga': best_ga_full,
        'aco': best_aco_full,
        'tuned_on_seed': TUNING_GRID_SEED,
        'tuning_runs_per_config': TUNING_RUNS_PER_CONFIG,
        'tuning_generations': TUNING_GENERATIONS,
    }
    outfile = output_dir / 'best_params.json'
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[tuning] Best params saved to {outfile}")
    return out


def quick_sanity_check(env: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    print("\n[sanity check] Running mini tuning smoke test...")
    dist_matrix = env['dist_matrix']
    dp_matrix = dist_matrix[1:, 1:]
    n_genes = len(env['delivery_points'])
    fitness_fn = lambda c: _tour_fitness(c, dist_matrix)

    mini_ga_grid = {
        'pop_size': [30, 50],
        'crossover_rate': [0.8],
        'mutation_rate': [0.1],
        'tournament_k': [3],
    }
    keys = list(mini_ga_grid.keys())
    records = []
    for combo in itertools.product(*mini_ga_grid.values()):
        params = dict(zip(keys, combo))
        params['generations'] = 10
        run_fits = []
        for run in range(2):
            result = run_memetic_ga(
                n_genes,
                fitness_fn,
                dp_matrix,
                {**params, 'seed': run},
            )
            run_fits.append(result['best_fitness'])
        records.append({**params, 'mean_fitness': float(np.mean(run_fits))})

    df = pd.DataFrame(records).sort_values('mean_fitness')
    df.to_csv(output_dir / 'tuning_sanity_ga.csv', index=False)
    best_ga = df.iloc[0][keys].to_dict()
    best_aco = {k: values[0] for k, values in ACO_PARAM_GRID.items()}
    params_out = save_best_params(best_ga, best_aco, output_dir)
    print("[sanity check] Done.")
    return params_out


def main():
    parser = argparse.ArgumentParser(description='Parameter tuning for GA and ACO.')
    parser.add_argument('--algo', choices=['ga', 'aco', 'both', 'sanity'], default='both')
    parser.add_argument('--seed', type=int, default=TUNING_GRID_SEED)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    env = build_tuning_environment(args.seed)

    if args.algo == 'sanity':
        quick_sanity_check(env, output_dir)
    else:
        best_ga, best_aco = {}, {}
        if args.algo in ('ga', 'both'):
            best_ga = tune_ga(env, output_dir)
        if args.algo in ('aco', 'both'):
            best_aco = tune_aco(env, output_dir)
        if args.algo == 'both':
            save_best_params(best_ga, best_aco, output_dir)

    elapsed = time.perf_counter() - t_start
    print(f"\n[tuning] Total wall-clock time: {elapsed/3600:.2f} hours ({elapsed:.0f}s)")


if __name__ == '__main__':
    main()
