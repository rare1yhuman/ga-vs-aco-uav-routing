"""
NSGA-II — Multi-Objective Genetic Algorithm — GA vs ACO Paper

Objectives:
    1. Minimize total tour distance (metres)
    2. Minimize total energy consumption (Joules)

Operators:
    - Non-dominated sorting + Crowding distance
    - Binary tournament selection
    - OX1 crossover + swap/inversion mutation (reuses ga.py)
"""

import numpy as np
import time
from typing import List, Dict, Any, Callable, Tuple

from ga import (
    Chromosome, Population,
    _init_population, _ox1_crossover, _mutate,
)

ObjectiveVec = Tuple[float, float]
ObjectiveFn  = Callable[[Chromosome], ObjectiveVec]

NSGA2_PARAMS = {
    'pop_size':       200,
    'generations':    500,
    'crossover_rate': 0.85,
    'mutation_rate':  0.15,
    'seed':           None,
}


def run_nsga2(n_genes: int, objective_fn: ObjectiveFn,
              params: Dict[str, Any] = None, verbose: bool = False) -> Dict[str, Any]:
    p   = {**NSGA2_PARAMS, **(params or {})}
    rng = np.random.default_rng(p['seed'])

    pop        = _init_population(n_genes, p['pop_size'], rng)
    objectives = [objective_fn(c) for c in pop]
    history    = []
    t0         = time.perf_counter()

    for gen in range(p['generations']):
        fronts   = _non_dominated_sort(objectives)
        crowding = _crowding_distances(objectives, fronts)

        offspring, offspring_obj = [], []
        while len(offspring) < p['pop_size']:
            p1 = _tournament(pop, objectives, fronts, crowding, rng)
            p2 = _tournament(pop, objectives, fronts, crowding, rng)

            c1, c2 = (_ox1_crossover(p1, p2, rng) if rng.random() < p['crossover_rate']
                      else (p1.copy(), p2.copy()))

            if rng.random() < p['mutation_rate']: c1 = _mutate(c1, rng)
            if rng.random() < p['mutation_rate']: c2 = _mutate(c2, rng)

            offspring.append(c1); offspring_obj.append(objective_fn(c1))
            offspring.append(c2); offspring_obj.append(objective_fn(c2))

        pop, objectives = _env_select(
            pop + offspring[:p['pop_size']],
            objectives + offspring_obj[:p['pop_size']],
            p['pop_size']
        )

        fronts0 = _non_dominated_sort(objectives)
        f0 = fronts0[0] if fronts0 else []
        bd = min((objectives[i][0] for i in f0), default=float('inf'))
        be = min((objectives[i][1] for i in f0), default=float('inf'))
        history.append((gen, len(f0), bd, be))

        if verbose and gen % 50 == 0:
            print(f"  [NSGA-II] Gen {gen:>4d}  front_size={len(f0)}  "
                  f"best_dist={bd:.1f}  best_energy={be:.1f}")

    fronts = _non_dominated_sort(objectives)
    pareto = sorted(
        [{'chromosome': pop[i], 'distance': objectives[i][0], 'energy': objectives[i][1]}
         for i in (fronts[0] if fronts else [])],
        key=lambda x: x['distance']
    )
    knee = _knee_point(pareto)

    return {
        'pareto_front':   pareto,
        'knee_point':     knee,
        'history':        history,
        'all_objectives': objectives,
        'runtime_s':      time.perf_counter() - t0,
    }


def _dominates(a: ObjectiveVec, b: ObjectiveVec) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def _non_dominated_sort(objectives: List[ObjectiveVec]) -> List[List[int]]:
    n          = len(objectives)
    dom_count  = [0] * n
    dom_set    = [[] for _ in range(n)]
    fronts     = [[]]

    for i in range(n):
        for j in range(n):
            if i == j: continue
            if _dominates(objectives[i], objectives[j]): dom_set[i].append(j)
            elif _dominates(objectives[j], objectives[i]): dom_count[i] += 1
        if dom_count[i] == 0: fronts[0].append(i)

    k = 0
    while fronts[k]:
        nf = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0: nf.append(j)
        k += 1
        fronts.append(nf)
    return [f for f in fronts if f]


def _crowding_distances(objectives: List[ObjectiveVec],
                         fronts: List[List[int]]) -> List[float]:
    n = len(objectives)
    d = [0.0] * n
    for front in fronts:
        m = len(front)
        if m <= 2:
            for i in front: d[i] = float('inf')
            continue
        for dim in range(2):
            sf  = sorted(front, key=lambda i: objectives[i][dim])
            d[sf[0]] = d[sf[-1]] = float('inf')
            span = objectives[sf[-1]][dim] - objectives[sf[0]][dim] or 1e-9
            for k in range(1, m - 1):
                d[sf[k]] += (objectives[sf[k+1]][dim] - objectives[sf[k-1]][dim]) / span
    return d


def _rank_of(idx, fronts):
    for r, f in enumerate(fronts):
        if idx in f: return r
    return len(fronts)


def _tournament(pop, objectives, fronts, crowding, rng):
    i, j = rng.integers(0, len(pop), size=2)
    ri, rj = _rank_of(i, fronts), _rank_of(j, fronts)
    if ri < rj or (ri == rj and crowding[i] >= crowding[j]):
        return pop[i].copy()
    return pop[j].copy()


def _env_select(pop, objectives, size):
    fronts   = _non_dominated_sort(objectives)
    crowding = _crowding_distances(objectives, fronts)
    idx = []
    for front in fronts:
        if len(idx) + len(front) <= size:
            idx.extend(front)
        else:
            remaining = size - len(idx)
            idx.extend(sorted(front, key=lambda i: -crowding[i])[:remaining])
            break
    return [pop[i] for i in idx], [objectives[i] for i in idx]


def _knee_point(pareto: List[Dict]) -> Dict:
    if not pareto: return {}
    if len(pareto) == 1: return pareto[0]
    ds = np.array([s['distance'] for s in pareto])
    es = np.array([s['energy']   for s in pareto])
    nd = (ds - ds.min()) / (ds.max() - ds.min() or 1)
    ne = (es - es.min()) / (es.max() - es.min() or 1)
    return pareto[int(np.argmax(np.abs(nd + ne - 1)))]


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n   = 12
    pts = rng.integers(0, 100, size=(n, 2)).astype(float)
    D   = np.array([[np.linalg.norm(pts[i]-pts[j]) for j in range(n)] for i in range(n)])

    def obj(c):
        d = sum(D[c[i], c[(i+1)%n]] for i in range(n))
        e = d * (1 + 0.01 * sum(abs(c[i]-c[(i+1)%n]) for i in range(n)))
        return (d, e)

    r = run_nsga2(n, obj, params={'pop_size': 60, 'generations': 80, 'seed': 42}, verbose=True)
    print(f"\nPareto front: {len(r['pareto_front'])} solutions")
    knee = r['knee_point']
    print(f"Knee point  : dist={knee['distance']:.2f}, energy={knee['energy']:.2f}")
    print(f"Runtime     : {r['runtime_s']:.2f}s")
