"""
Memetic GA (MGA) — Enhanced with Or-opt, Adaptive Mutation & Diversity Injection

Improvements over baseline MGA-2opt:
  1. Or-opt   — relocates segments of 1, 2, or 3 consecutive cities to better
                positions in the tour. Finds improvements that 2-opt misses.
  2. Adaptive mutation — automatically increases mutation rate when the
                population stagnates (no improvement for `stagnation_window` gens).
  3. Diversity injection — if still stagnant after adaptive mutation, replaces
                the bottom 20% of the population with fresh random chromosomes,
                preventing premature convergence.
  4. Multi-operator crossover — mixes OX1, PMX, and edge recombination so the
                GA does not depend on a single permutation operator.

Labelled "MGA-Enhanced" in the paper.
"""

import time
from typing import List, Callable, Dict, Any, Tuple, Optional

import numpy as np

from ga_reference import (
    Population, Chromosome, FitnessFunc,
    DEFAULT_PARAMS,
    _init_population, _tournament_select, _ox1_crossover, _mutate,
)


# ── Additional hyperparameters ───────────────────────────────────────────────
MGA_EXTRA_PARAMS = {
    'two_opt_max_iter':   200,   # max 2-opt passes per chromosome
    'two_opt_every_n':    1,     # apply 2-opt every N gens (1 = always)
    'or_opt_every_n':     2,     # apply Or-opt every N gens
    'local_search_top_frac': 0.20,  # refine only the best unique offspring
    'initial_local_search_frac': 0.25,
    'local_search_candidate_k': 10,
    'final_refine_candidate_k': 20,
    'exhaustive_final_max_n': 25,
    'final_refine_best': True,
    'stagnation_window':  30,    # gens without improvement → trigger adaptive
    'adaptive_mut_boost': 3.0,   # multiply mutation_rate by this on stagnation
    'inject_frac':        0.20,  # fraction of population to replace on stagnation
    'inject_after':       60,    # inject diversity after this many stagnant gens
    'crossover_ops':      ('ox1', 'pmx', 'erx'),
    'crossover_probs':    (0.45, 0.25, 0.30),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Nearest-Neighbour Seeding
# ═══════════════════════════════════════════════════════════════════════════════

def _nearest_neighbor_tour(start: int, dist_matrix: np.ndarray) -> List[int]:
    """Build a greedy nearest-neighbour tour starting from node `start`."""
    n = dist_matrix.shape[0]
    visited = {start}
    tour = [start]
    current = start
    while len(tour) < n:
        best_next, best_dist = -1, float('inf')
        for j in range(n):
            if j not in visited and dist_matrix[current, j] < best_dist:
                best_dist = dist_matrix[current, j]
                best_next = j
        if best_next < 0:
            break
        tour.append(best_next)
        visited.add(best_next)
        current = best_next
    return tour


def _seeded_population(n: int, pop_size: int, dist_matrix: np.ndarray,
                       rng: np.random.Generator, seed_frac: float = 0.15
                       ) -> Population:
    """
    Create initial population with nearest-neighbour seeded tours.

    Seeds ~15% of the population with NN heuristic tours starting from
    different delivery points, giving the GA a strong starting point.
    The remaining 85% are random permutations for genetic diversity.
    """
    pop: Population = []
    n_seeds = max(1, int(seed_frac * pop_size))

    # NN tours starting from each delivery point index
    starts = list(range(min(n_seeds, n)))
    # If we need more seeds than nodes, add random starting points
    while len(starts) < n_seeds:
        starts.append(int(rng.integers(0, n)))

    for start_node in starts[:n_seeds]:
        tour = _nearest_neighbor_tour(start_node, dist_matrix)
        if len(tour) == n:
            pop.append(tour)

    # Fill remaining slots with random permutations
    while len(pop) < pop_size:
        pop.append(list(rng.permutation(n)))

    return pop[:pop_size]


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_memetic_ga(
    n_genes:      int,
    fitness_fn:   FitnessFunc,
    dist_matrix:  np.ndarray,          # precomputed pairwise distances [n×n]
    params:       Dict[str, Any] = None,
    verbose:      bool = False,
) -> Dict[str, Any]:
    """
    Run Memetic GA with 2-opt local search.

    Args:
        n_genes:     Number of delivery points (chromosome length).
        fitness_fn:  Lower is better. Used for selection.
        dist_matrix: n×n array of pairwise inter-point distances.
                     Index matches gene (delivery point) index.
        params:      Hyperparameter dict (merged with DEFAULT_PARAMS + MGA extras).
        verbose:     Print progress every 50 generations.

    Returns:
        Same structure as ga.run_ga(), plus:
            'two_opt_improvements': total number of 2-opt improvements applied
    """
    p = {**DEFAULT_PARAMS, **MGA_EXTRA_PARAMS, **(params or {})}
    rng = np.random.default_rng(p['seed'])
    objective_matrix = p.get('objective_matrix', None)
    if objective_matrix is None:
        objective_matrix = p.get('_objective_matrix', None)
    if objective_matrix is not None:
        objective_matrix = np.asarray(objective_matrix, dtype=float)
    authoritative_fitness = bool(p.get('fitness_fn_authoritative', False))
    candidate_k = int(p.get('local_search_candidate_k', 10))

    eval_count = 0
    fitness_cache: Dict[Tuple[int, ...], float] = {}

    def eval_fitness(chrom: Chromosome) -> float:
        nonlocal eval_count
        key = tuple(int(g) for g in chrom)
        if key not in fitness_cache:
            if objective_matrix is not None and not authoritative_fitness:
                fitness_cache[key] = _route_cost_with_depot(chrom, objective_matrix)
            else:
                fitness_cache[key] = float(fitness_fn(chrom))
            eval_count += 1
        return fitness_cache[key]

    def refine(chrom: Chromosome, exhaustive: bool = False) -> Tuple[Chromosome, int, int, int]:
        if objective_matrix is None:
            before = _tour_distance(chrom, dist_matrix)
            after = _two_opt(chrom, dist_matrix, p['two_opt_max_iter'])
            two = int(_tour_distance(after, dist_matrix) < before)
            before = _tour_distance(after, dist_matrix)
            after = _or_opt(after, dist_matrix, seg_sizes=(1, 2, 3))
            ore = int(_tour_distance(after, dist_matrix) < before)
            before_fit = eval_fitness(after)
            after = _depot_rotation(after, eval_fitness)
            dep = int(eval_fitness(after) < before_fit)
            return after, two, ore, dep

        if exhaustive and n_genes <= int(p.get('exhaustive_final_max_n', 25)):
            k = None
        elif exhaustive:
            k = int(p.get('final_refine_candidate_k', candidate_k))
        else:
            k = candidate_k
        before = eval_fitness(chrom)
        candidate = _two_opt_objective(
            chrom, objective_matrix, p['two_opt_max_iter'], candidate_k=k
        )
        candidate_cost = eval_fitness(candidate)
        after = candidate if candidate_cost < before else chrom.copy()
        two = int(candidate_cost < before)
        before = eval_fitness(after)
        candidate = _or_opt_objective(
            after, objective_matrix, seg_sizes=(1, 2, 3), candidate_k=k
        )
        candidate_cost = eval_fitness(candidate)
        after = candidate if candidate_cost < before else after
        ore = int(candidate_cost < before)
        before = eval_fitness(after)
        candidate = _depot_rotation_matrix(after, objective_matrix)
        candidate_cost = eval_fitness(candidate)
        after = candidate if candidate_cost < before else after
        dep = int(candidate_cost < before)
        return after, two, ore, dep

    pop        = _seeded_population(n_genes, p['pop_size'], dist_matrix, rng,
                                     seed_frac=p.get('nn_seed_frac', 0.15))
    t0                  = time.perf_counter()
    cpu_t0              = time.process_time()
    elite_size          = max(1, int(p['elitism_frac'] * p['pop_size']))
    inject_size         = max(1, int(p['inject_frac']  * p['pop_size']))
    conv_gen            = p['generations']
    two_opt_improvements = 0
    or_opt_improvements  = 0
    depot_rot_improvements = 0
    stagnant_gens        = 0
    current_mut_rate     = p['mutation_rate']
    diversity_injections = 0
    crossover_counts     = {'ox1': 0, 'pmx': 0, 'erx': 0, 'clone': 0}

    initial_refine_n = max(1, int(float(p.get('initial_local_search_frac', 0.25)) * len(pop)))
    initial_idx = np.argsort([eval_fitness(c) for c in pop])[:initial_refine_n]
    for idx in initial_idx:
        pop[idx], t_imp, o_imp, d_imp = refine(pop[idx])
        two_opt_improvements += t_imp
        or_opt_improvements += o_imp
        depot_rot_improvements += d_imp

    fitnesses  = [eval_fitness(c) for c in pop]
    history             = []
    history_wall_s      = []
    history_cpu_s       = []
    history_evaluations = []

    best_ever  = min(fitnesses)
    best_chrom = pop[int(np.argmin(fitnesses))].copy()

    total_no_improve     = 0
    EARLY_STOP_PATIENCE  = p.get('early_stop_patience', None)

    generations_completed = 0
    for gen in range(p['generations']):
        generations_completed = gen + 1
        sorted_idx = np.argsort(fitnesses)
        elites     = [pop[i].copy() for i in sorted_idx[:elite_size]]

        # ── Stagnation detection ─────────────────────────────────────────────
        if history and min(fitnesses) >= best_ever - 1e-9:
            stagnant_gens += 1
        else:
            stagnant_gens  = 0
            current_mut_rate = p['mutation_rate']   # reset on improvement

        # ── Adaptive mutation ────────────────────────────────────────────────
        if stagnant_gens == p['stagnation_window']:
            current_mut_rate = min(0.5, p['mutation_rate'] * p['adaptive_mut_boost'])
            if verbose:
                print(f"  [MGA] Gen {gen}: stagnation → mutation boosted to {current_mut_rate:.3f}")

        # ── Diversity injection ──────────────────────────────────────────────
        if stagnant_gens >= p['inject_after']:
            worst_idx = sorted_idx[-inject_size:]
            for idx in worst_idx:
                best_perturbed = _double_bridge(best_chrom, rng)
                best_perturbed, _, _, _ = refine(best_perturbed)
                best_cost = eval_fitness(best_perturbed)
                for _ in range(5):  # try 5 perturbations
                    candidate = _double_bridge(best_chrom, rng)
                    candidate, _, _, _ = refine(candidate)
                    cost = eval_fitness(candidate)
                    if cost < best_cost:
                        best_cost = cost
                        best_perturbed = candidate
                pop[idx]       = best_perturbed
                fitnesses[idx] = best_cost
            stagnant_gens      = 0
            current_mut_rate   = p['mutation_rate']
            diversity_injections += 1
            if verbose:
                print(f"  [MGA] Gen {gen}: diversity injected ({inject_size} chromosomes)")

        # ── Build offspring ──────────────────────────────────────────────────
        new_pop = elites.copy()
        apply_2opt   = (gen % p['two_opt_every_n'] == 0)
        apply_or_opt = (gen % p['or_opt_every_n']  == 0)

        while len(new_pop) < p['pop_size']:
            p1 = _tournament_select(pop, fitnesses, p['tournament_k'], rng)
            p2 = _tournament_select(pop, fitnesses, p['tournament_k'], rng)

            if rng.random() < p['crossover_rate']:
                c1, c2, op = _multi_crossover(p1, p2, rng, p)
                crossover_counts[op] = crossover_counts.get(op, 0) + 1
            else:
                c1, c2 = p1.copy(), p2.copy()
                crossover_counts['clone'] += 1

            if rng.random() < current_mut_rate: c1 = _mutate(c1, rng)
            if rng.random() < current_mut_rate: c2 = _mutate(c2, rng)

            new_pop.extend([c1, c2])

        new_pop = new_pop[:p['pop_size']]
        if apply_2opt or apply_or_opt:
            seen = set()
            candidate_idx = []
            for idx in range(elite_size, len(new_pop)):
                key = tuple(new_pop[idx])
                if key not in seen:
                    seen.add(key)
                    candidate_idx.append(idx)
            candidate_idx = sorted(candidate_idx, key=lambda i: eval_fitness(new_pop[i]))
            refine_n = max(1, int(float(p.get('local_search_top_frac', 0.20)) * p['pop_size']))
            for idx in candidate_idx[:refine_n]:
                before_key = tuple(new_pop[idx])
                new_pop[idx], t_imp, o_imp, d_imp = refine(new_pop[idx])
                if tuple(new_pop[idx]) != before_key:
                    two_opt_improvements += t_imp if apply_2opt else 0
                    or_opt_improvements += o_imp if apply_or_opt else 0
                    depot_rot_improvements += d_imp

        pop       = new_pop
        fitnesses = [eval_fitness(c) for c in pop]

        gen_best = min(fitnesses)
        history.append(gen_best)
        history_wall_s.append(time.perf_counter() - t0)
        history_cpu_s.append(time.process_time() - cpu_t0)
        history_evaluations.append(eval_count)

        if gen_best < best_ever:
            best_ever  = gen_best
            best_chrom = pop[int(np.argmin(fitnesses))].copy()
            total_no_improve = 0
        else:
            total_no_improve += 1
            
        # Early stop if no improvement for patience gens
        if EARLY_STOP_PATIENCE and total_no_improve >= EARLY_STOP_PATIENCE:
            if conv_gen == p['generations']:
                conv_gen = gen
            if verbose:
                print(f"  [MGA] Gen {gen}: Early stopping triggered.")
            break

        if conv_gen == p['generations'] and gen > 10:
            if best_ever <= history[0] * 0.99:
                conv_gen = gen

        if verbose and gen % 50 == 0:
            print(f"  [MGA] Gen {gen:>4d}  best={gen_best:.1f}  "
                  f"mut={current_mut_rate:.3f}  2opt={two_opt_improvements}  "
                  f"or_opt={or_opt_improvements}  inject={diversity_injections}")

    if p.get('final_refine_best', True):
        before_final_refine = best_ever
        best_chrom, t_imp, o_imp, d_imp = refine(best_chrom, exhaustive=True)
        best_ever = eval_fitness(best_chrom)
        two_opt_improvements += t_imp
        or_opt_improvements += o_imp
        depot_rot_improvements += d_imp
        if best_ever < before_final_refine:
            history.append(best_ever)
            history_wall_s.append(time.perf_counter() - t0)
            history_cpu_s.append(time.process_time() - cpu_t0)
            history_evaluations.append(eval_count)

    return {
        'best_chromosome':    best_chrom,
        'best_fitness':       best_ever,
        'history':            history,
        'history_wall_s':     history_wall_s,
        'history_cpu_s':      history_cpu_s,
        'history_evaluations': history_evaluations,
        'convergence_gen':    conv_gen,
        'runtime_s':          time.perf_counter() - t0,
        'cpu_time_s':         time.process_time() - cpu_t0,
        'generations_completed': generations_completed,
        'candidate_constructions': int(p['pop_size']) * (generations_completed + 1),
        'two_opt_improvements':    two_opt_improvements,
        'or_opt_improvements':     or_opt_improvements,
        'depot_rot_improvements':  depot_rot_improvements,
        'diversity_injections':    diversity_injections,
        'crossover_counts':        crossover_counts,
        'fitness_evaluations':      eval_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Crossover And Local Search Operators
# ═══════════════════════════════════════════════════════════════════════════════

def _multi_crossover(p1: Chromosome, p2: Chromosome,
                     rng: np.random.Generator,
                     params: Dict[str, Any]) -> Tuple[Chromosome, Chromosome, str]:
    """Choose one permutation crossover operator according to configured weights."""
    ops = tuple(params.get('crossover_ops', ('ox1', 'pmx', 'erx')))
    probs = np.array(params.get('crossover_probs', (0.45, 0.25, 0.30)), dtype=float)
    if len(ops) != len(probs) or probs.sum() <= 0:
        ops = ('ox1', 'pmx', 'erx')
        probs = np.array((0.45, 0.25, 0.30), dtype=float)
    probs = probs / probs.sum()
    op = str(rng.choice(ops, p=probs))

    if op == 'pmx':
        return (*_pmx_crossover(p1, p2, rng), 'pmx')
    if op == 'erx':
        return (_edge_recombination(p1, p2, rng),
                _edge_recombination(p2, p1, rng),
                'erx')
    return (*_ox1_crossover(p1, p2, rng), 'ox1')


def _pmx_crossover(p1: Chromosome, p2: Chromosome,
                   rng: np.random.Generator) -> Tuple[Chromosome, Chromosome]:
    """
    Partially Mapped Crossover (PMX).

    PMX preserves absolute positions from a mapped segment. It complements OX1,
    which mainly preserves relative order.
    """
    n = len(p1)
    if n < 2:
        return p1.copy(), p2.copy()
    a, b = sorted(rng.choice(n, size=2, replace=False))
    if a > b:
        a, b = b, a
    b += 1

    def _child(parent_a, parent_b):
        child = [-1] * n
        child[a:b] = parent_a[a:b]

        for i in range(a, b):
            gene = parent_b[i]
            if gene in child:
                continue
            pos = i
            while True:
                mapped = parent_a[pos]
                pos = parent_b.index(mapped)
                if child[pos] == -1:
                    child[pos] = gene
                    break

        for i in range(n):
            if child[i] == -1:
                child[i] = parent_b[i]
        return child

    return _child(p1, p2), _child(p2, p1)


def _edge_recombination(p1: Chromosome, p2: Chromosome,
                        rng: np.random.Generator) -> Chromosome:
    """
    Edge Recombination Crossover (ERX).

    ERX preserves adjacency information, which is useful for routing because
    good tours are mostly good edges arranged in a valid order.
    """
    n = len(p1)
    if n < 2:
        return p1.copy()

    edge_map = {g: set() for g in p1}
    for parent in (p1, p2):
        for i, gene in enumerate(parent):
            edge_map[gene].add(parent[(i - 1) % n])
            edge_map[gene].add(parent[(i + 1) % n])

    current = p1[0] if rng.random() < 0.5 else p2[0]
    child = [current]
    unused = set(p1)
    unused.remove(current)

    while unused:
        for neighbours in edge_map.values():
            neighbours.discard(current)

        candidates = [g for g in edge_map[current] if g in unused]
        if candidates:
            min_degree = min(len(edge_map[g]) for g in candidates)
            best = [g for g in candidates if len(edge_map[g]) == min_degree]
            current = int(rng.choice(best))
        else:
            current = int(rng.choice(list(unused)))

        child.append(current)
        unused.remove(current)

    return child

def _double_bridge(chrom: Chromosome, rng: np.random.Generator) -> Chromosome:
    """Double-bridge (4-opt) perturbation — breaks 4 edges to escape local optima."""
    n = len(chrom)
    if n < 4:
        return chrom.copy()
    cuts = sorted(rng.choice(range(1, n), size=3, replace=False))
    a, b, c = cuts
    # Reconnect as: seg1 + seg3 + seg2 + seg4
    return chrom[:a] + chrom[b:c] + chrom[a:b] + chrom[c:]

def _two_opt(chrom: Chromosome, dist_matrix: np.ndarray,
             max_iter: int = 200) -> Chromosome:
    """2-opt: try reversing every sub-segment. Accept if shorter."""
    tour     = chrom.copy()
    n        = len(tour)
    improved = True
    itr      = 0
    while improved and itr < max_iter:
        improved = False
        itr     += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]
                       - dist_matrix[a, b] - dist_matrix[c, d])
                if delta < -1e-9:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
    return tour


def _or_opt(chrom: Chromosome, dist_matrix: np.ndarray,
            seg_sizes: Tuple[int, ...] = (1, 2, 3, 4, 5)) -> Chromosome:
    """
    Or-opt: try relocating segments of 1, 2, or 3 consecutive nodes
    to every other position in the tour. Accept the best improvement found.
    Complements 2-opt by exploring a different neighbourhood.
    """
    tour     = chrom.copy()
    n        = len(tour)
    improved = True

    while improved:
        improved = False
        for seg in seg_sizes:
            if seg >= n:
                continue
            for i in range(n):
                # Extract segment
                seg_idx  = [(i + k) % n for k in range(seg)]
                seg_nodes= [tour[k] for k in seg_idx]

                # Remaining nodes in order
                rest_idx = [k for k in range(n) if k not in seg_idx]
                rest     = [tour[k] for k in rest_idx]

                # Cost of removing segment from current position
                prev_i = tour[(i - 1) % n]
                next_i = tour[(i + seg) % n]
                removal_gain = (dist_matrix[prev_i, seg_nodes[0]]
                              + dist_matrix[seg_nodes[-1], next_i]
                              - dist_matrix[prev_i, next_i])

                best_gain    = 1e-9   # must beat this threshold
                best_pos     = -1

                # Try inserting segment after each remaining node
                for j in range(len(rest)):
                    before = rest[j]
                    after  = rest[(j + 1) % len(rest)]
                    insert_cost = (dist_matrix[before, seg_nodes[0]]
                                 + dist_matrix[seg_nodes[-1], after]
                                 - dist_matrix[before, after])
                    gain = removal_gain - insert_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_pos  = j

                if best_pos >= 0:
                    # Rebuild tour with segment inserted at best_pos
                    new_tour = rest[:best_pos+1] + seg_nodes + rest[best_pos+1:]
                    tour     = new_tour
                    improved = True
                    break
            if improved:
                break
    return tour


def _depot_rotation(chrom: Chromosome, fitness_fn: FitnessFunc) -> Chromosome:
    """
    Rotate the chromosome so that the delivery point with the lowest
    combined depot→first + last→depot cost comes first.

    Equivalent to trying all n starting points and picking the best.
    O(n) — negligible overhead.

    This mirrors ACO's natural behaviour of always building routes that
    start and end at the depot. Uses fitness_fn which correctly handles depot legs.
    """
    n = len(chrom)
    if n <= 1:
        return chrom

    best_cost = float('inf')
    best_tour = chrom

    for rot in range(n):
        rotated = chrom[rot:] + chrom[:rot]
        cost = fitness_fn(rotated)
        if cost < best_cost:
            best_cost = cost
            best_tour = rotated

    return best_tour


def _tour_distance(chrom: Chromosome, dist_matrix: np.ndarray) -> float:
    """Total round-trip distance of a chromosome using dist_matrix."""
    n    = len(chrom)
    dist = sum(dist_matrix[chrom[i], chrom[(i+1) % n]] for i in range(n))
    return float(dist)


def _route_cost_with_depot(chrom: Chromosome, full_matrix: np.ndarray) -> float:
    """Depot-to-depot route cost. Matrix index 0 is depot, genes map to index+1."""
    if not chrom:
        return 0.0
    order = [0] + [int(g) + 1 for g in chrom] + [0]
    return float(sum(full_matrix[order[i], order[i + 1]] for i in range(len(order) - 1)))


def _nearest_candidates(full_matrix: np.ndarray, n_genes: int,
                        candidate_k: Optional[int]) -> Dict[int, set]:
    """Delivery-node candidate list keyed by chromosome gene value."""
    if candidate_k is None or candidate_k <= 0 or candidate_k >= n_genes - 1:
        return {g: set(range(n_genes)) - {g} for g in range(n_genes)}
    out: Dict[int, set] = {}
    for gene in range(n_genes):
        row = full_matrix[gene + 1, 1:]
        order = [int(i) for i in np.argsort(row) if int(i) != gene]
        out[gene] = set(order[:candidate_k])
    return out


def _two_opt_objective(chrom: Chromosome, full_matrix: np.ndarray,
                       max_iter: int = 200,
                       candidate_k: Optional[int] = 10) -> Chromosome:
    """
    Depot-aware 2-opt using the same objective matrix as the GA fitness.

    The move evaluation recomputes the full route cost. That is deliberate:
    dynamic-risk matrices may be asymmetric, so the common symmetric 2-opt delta
    shortcut can accept the wrong reversal.
    """
    tour = chrom.copy()
    n = len(tour)
    if n < 3:
        return _depot_rotation_matrix(tour, full_matrix)
    candidates = _nearest_candidates(full_matrix, n, candidate_k)
    symmetric = np.allclose(full_matrix, full_matrix.T, equal_nan=True)
    best_cost = _route_cost_with_depot(tour, full_matrix)
    improved = True
    itr = 0
    while improved and itr < max_iter:
        improved = False
        itr += 1
        best_move = None
        best_delta = 0.0
        for i in range(n - 1):
            a_gene = tour[i]
            for j in range(i + 1, n):
                c_gene = tour[j]
                if c_gene not in candidates.get(a_gene, set()) and a_gene not in candidates.get(c_gene, set()):
                    continue
                if symmetric:
                    route = [0] + [g + 1 for g in tour] + [0]
                    a, b = route[i], route[i + 1]
                    c, d = route[j + 1], route[j + 2]
                    delta = full_matrix[a, c] + full_matrix[b, d] - full_matrix[a, b] - full_matrix[c, d]
                    if delta < best_delta - 1e-9:
                        best_delta = float(delta)
                        best_move = tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]
                else:
                    candidate = tour[:i] + list(reversed(tour[i:j + 1])) + tour[j + 1:]
                    cost = _route_cost_with_depot(candidate, full_matrix)
                    if cost < best_cost - 1e-9:
                        best_cost = cost
                        best_move = candidate
        if best_move is not None:
            tour = best_move
            if symmetric:
                best_cost += best_delta
            improved = True
    return tour


def _or_opt_objective(chrom: Chromosome, full_matrix: np.ndarray,
                      seg_sizes: Tuple[int, ...] = (1, 2, 3),
                      candidate_k: Optional[int] = 10) -> Chromosome:
    """Depot-aware Or-opt relocation with candidate-list restriction."""
    tour = chrom.copy()
    n = len(tour)
    if n < 3:
        return _depot_rotation_matrix(tour, full_matrix)
    candidates = _nearest_candidates(full_matrix, n, candidate_k)
    symmetric = np.allclose(full_matrix, full_matrix.T, equal_nan=True)
    best_cost = _route_cost_with_depot(tour, full_matrix)
    improved = True
    while improved:
        improved = False
        best_tour = tour
        best_delta = 0.0
        for seg in seg_sizes:
            if seg >= n:
                continue
            for i in range(0, n - seg + 1):
                segment = tour[i:i + seg]
                rest = tour[:i] + tour[i + seg:]
                route = [0] + [g + 1 for g in tour] + [0]
                prev_i = route[i]
                first = route[i + 1]
                last = route[i + seg]
                next_i = route[i + seg + 1]
                removal_delta = full_matrix[prev_i, next_i] - full_matrix[prev_i, first] - full_matrix[last, next_i]
                for pos in range(len(rest) + 1):
                    if pos == i:
                        continue
                    before_gene = rest[pos - 1] if pos > 0 else None
                    after_gene = rest[pos] if pos < len(rest) else None
                    anchor_ok = True
                    if before_gene is not None:
                        anchor_ok = segment[0] in candidates.get(before_gene, set())
                    if after_gene is not None:
                        anchor_ok = anchor_ok or after_gene in candidates.get(segment[-1], set())
                    if not anchor_ok and candidate_k is not None:
                        continue
                    candidate = rest[:pos] + segment + rest[pos:]
                    if symmetric:
                        rest_route = [0] + [g + 1 for g in rest] + [0]
                        before = rest_route[pos]
                        after = rest_route[pos + 1]
                        insert_delta = (
                            full_matrix[before, first]
                            + full_matrix[last, after]
                            - full_matrix[before, after]
                        )
                        delta = removal_delta + insert_delta
                        if delta < best_delta - 1e-9:
                            best_delta = float(delta)
                            best_tour = candidate
                            improved = True
                    else:
                        cost = _route_cost_with_depot(candidate, full_matrix)
                        if cost < best_cost - 1e-9:
                            best_cost = cost
                            best_tour = candidate
                            improved = True
        tour = best_tour
        if symmetric:
            best_cost += best_delta
    return tour


def _depot_rotation_matrix(chrom: Chromosome, full_matrix: np.ndarray) -> Chromosome:
    """Rotate route in O(n), accounting for depot start/end legs exactly."""
    n = len(chrom)
    if n <= 1:
        return chrom.copy()
    cycle_cost = 0.0
    for i in range(n):
        cycle_cost += full_matrix[chrom[i] + 1, chrom[(i + 1) % n] + 1]

    best_cost = float('inf')
    best_rot = chrom.copy()
    for rot in range(n):
        first = chrom[rot]
        last = chrom[(rot - 1) % n]
        broken_cycle_edge = full_matrix[last + 1, first + 1]
        cost = cycle_cost - broken_cycle_edge + full_matrix[0, first + 1] + full_matrix[last + 1, 0]
        if cost < best_cost:
            best_cost = float(cost)
            best_rot = chrom[rot:] + chrom[:rot]
    return best_rot



def build_distance_matrix(delivery_points, depot, path_lengths: dict) -> np.ndarray:
    """
    Build an (n+1)×(n+1) distance matrix where index 0 = depot,
    indices 1..n = delivery points.

    path_lengths: dict mapping (src, dst) → shortest path length in metres.
    """
    nodes = [depot] + list(delivery_points)
    n     = len(nodes)
    D     = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)

    for i in range(n):
        for j in range(n):
            if i != j:
                key = (nodes[i], nodes[j])
                if key in path_lengths:
                    D[i, j] = path_lengths[key]

    return D


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 8-city random distance matrix
    rng = np.random.default_rng(0)
    n   = 8
    pts = rng.integers(0, 100, size=(n, 2)).astype(float)
    D   = np.array([[np.linalg.norm(pts[i] - pts[j]) for j in range(n)]
                    for i in range(n)])

    def fitness(chrom):
        return sum(D[chrom[i], chrom[(i+1) % n]] for i in range(n))

    result = run_memetic_ga(
        n_genes=n, fitness_fn=fitness, dist_matrix=D,
        params={'pop_size': 30, 'generations': 50, 'seed': 42},
        verbose=True
    )
    print(f"\nBest tour     : {result['best_chromosome']}")
    print(f"Best distance : {result['best_fitness']:.2f}")
    print(f"2-opt applied : {result['two_opt_improvements']} improvements")
    print(f"Runtime       : {result['runtime_s']:.3f}s")
