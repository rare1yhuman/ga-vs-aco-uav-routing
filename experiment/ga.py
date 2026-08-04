"""
Genetic Algorithm (GA) — Core Implementation
GA vs ACO Paper

Chromosome encoding: permutation of delivery-point indices
    [d3, d7, d1, d5, ...]  = visit order (depot is implicit start/end)

Selection  : Tournament (k=7)
Crossover  : Order Crossover OX1 (preserves relative order, no repeats)
Mutation   : Swap mutation + Inversion mutation (randomly chosen)
Elitism    : Top 5% carried forward unchanged
"""

import numpy as np
import time
from typing import List, Tuple, Callable, Dict, Any

# ── Type aliases ─────────────────────────────────────────────────────────────
Chromosome  = List[int]            # permutation of delivery-point indices
Population  = List[Chromosome]
FitnessFunc = Callable[[Chromosome], float]   # lower = better


# ── Default hyperparameters ──────────────────────────────────────────────────
DEFAULT_PARAMS = {
    'pop_size':       200,
    'generations':    500,
    'crossover_rate': 0.85,
    'mutation_rate':  0.15,
    'tournament_k':   7,
    'elitism_frac':   0.05,
    'seed':           None,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_ga(
    n_genes:      int,
    fitness_fn:   FitnessFunc,
    params:       Dict[str, Any] = None,
    verbose:      bool = False,
) -> Dict[str, Any]:
    """
    Run a standard Genetic Algorithm.

    Args:
        n_genes:    Number of delivery points (chromosome length).
        fitness_fn: Function mapping Chromosome → float (lower is better).
        params:     Hyperparameter dict (merged with DEFAULT_PARAMS).
        verbose:    Print progress every 50 generations.

    Returns:
        Dict with keys:
            'best_chromosome'  : best Chromosome found
            'best_fitness'     : best fitness value
            'history'          : List of best-fitness-per-generation
            'convergence_gen'  : generation at which 99% of best was achieved
            'runtime_s'        : total wall-clock time
    """
    p   = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(p['seed'])

    pop         = _init_population(n_genes, p['pop_size'], rng)
    fitnesses   = [fitness_fn(c) for c in pop]
    history     = []
    elite_size  = max(1, int(p['elitism_frac'] * p['pop_size']))
    t0          = time.perf_counter()
    conv_gen    = p['generations']

    best_ever   = min(fitnesses)
    best_chrom  = pop[int(np.argmin(fitnesses))].copy()

    for gen in range(p['generations']):
        # ── Elitism: keep top solutions ──────────────────────────────────────
        sorted_idx = np.argsort(fitnesses)
        elites     = [pop[i].copy() for i in sorted_idx[:elite_size]]

        # ── Build new population ─────────────────────────────────────────────
        new_pop = elites.copy()
        while len(new_pop) < p['pop_size']:
            # Selection
            p1 = _tournament_select(pop, fitnesses, p['tournament_k'], rng)
            p2 = _tournament_select(pop, fitnesses, p['tournament_k'], rng)

            # Crossover
            if rng.random() < p['crossover_rate']:
                c1, c2 = _ox1_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            if rng.random() < p['mutation_rate']:
                c1 = _mutate(c1, rng)
            if rng.random() < p['mutation_rate']:
                c2 = _mutate(c2, rng)

            new_pop.extend([c1, c2])

        pop       = new_pop[:p['pop_size']]
        fitnesses = [fitness_fn(c) for c in pop]

        gen_best = min(fitnesses)
        history.append(gen_best)

        if gen_best < best_ever:
            best_ever  = gen_best
            best_chrom = pop[int(np.argmin(fitnesses))].copy()

        # Convergence detection (within 1% of best ever)
        if conv_gen == p['generations'] and gen > 10:
            threshold = history[0] * 0.99
            if best_ever <= threshold:
                conv_gen = gen

        if verbose and gen % 50 == 0:
            print(f"  [GA] Gen {gen:>4d}  best={gen_best:.1f}")

    return {
        'best_chromosome': best_chrom,
        'best_fitness':    best_ever,
        'history':         history,
        'convergence_gen': conv_gen,
        'runtime_s':       time.perf_counter() - t0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GA Operators
# ═══════════════════════════════════════════════════════════════════════════════

def _init_population(n: int, size: int, rng: np.random.Generator) -> Population:
    """Create a population of random permutations."""
    return [list(rng.permutation(n)) for _ in range(size)]


def _tournament_select(pop: Population, fits: List[float],
                        k: int, rng: np.random.Generator) -> Chromosome:
    """Tournament selection: pick k random individuals, return best."""
    idx = rng.integers(0, len(pop), size=k)
    winner = idx[np.argmin([fits[i] for i in idx])]
    return pop[winner].copy()


def _ox1_crossover(p1: Chromosome, p2: Chromosome,
                    rng: np.random.Generator) -> Tuple[Chromosome, Chromosome]:
    """
    Order Crossover (OX1) — standard for permutation problems.
    Preserves relative order of genes from one parent and fills
    missing genes in the order they appear in the other parent.
    """
    n    = len(p1)
    a, b = sorted(rng.integers(0, n, size=2))

    def _child(primary, secondary):
        segment = primary[a:b]
        child   = [-1] * n
        child[a:b] = segment
        fill_vals = [g for g in secondary if g not in segment]
        fill_idx  = [i for i in range(n) if child[i] == -1]
        for i, v in zip(fill_idx, fill_vals):
            child[i] = v
        return child

    return _child(p1, p2), _child(p2, p1)


def _mutate(chrom: Chromosome, rng: np.random.Generator) -> Chromosome:
    """
    Apply either swap mutation or inversion mutation (50/50).

    Swap     : exchange two random genes
    Inversion: reverse a random sub-segment
    """
    c = chrom.copy()
    n = len(c)
    if n < 2:
        return c

    if rng.random() < 0.5:
        # Swap mutation
        i, j = rng.integers(0, n, size=2)
        c[i], c[j] = c[j], c[i]
    else:
        # Inversion mutation
        i, j = sorted(rng.integers(0, n, size=2))
        c[i:j+1] = c[i:j+1][::-1]

    return c


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Toy problem: minimise sum of positions (trivial)
    def toy_fitness(chrom):
        return sum(i * v for i, v in enumerate(chrom))

    result = run_ga(n_genes=10, fitness_fn=toy_fitness,
                    params={'pop_size': 50, 'generations': 100, 'seed': 42},
                    verbose=True)
    print(f"\nBest chromosome : {result['best_chromosome']}")
    print(f"Best fitness    : {result['best_fitness']:.4f}")
    print(f"Runtime         : {result['runtime_s']:.3f}s")
    print(f"Converged at gen: {result['convergence_gen']}")
