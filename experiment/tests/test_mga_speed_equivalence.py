"""Exact-output checks for the optimized MGA implementation."""

from __future__ import annotations

import unittest

import numpy as np

import ga
import ga_reference
import memetic_ga
import memetic_ga_reference


class MGASpeedEquivalenceTests(unittest.TestCase):
    def test_local_search_matches_reference(self) -> None:
        for n_genes in (8, 12):
            for seed in range(8):
                rng = np.random.default_rng(seed + n_genes * 100)
                chromosome = list(rng.permutation(n_genes))
                raw = rng.uniform(
                    1.0, 500.0, size=(n_genes + 1, n_genes + 1)
                )
                np.fill_diagonal(raw, 0.0)
                matrices = (raw, (raw + raw.T) / 2.0)
                candidate_k = min(5, n_genes - 2)
                for matrix in matrices:
                    expected = memetic_ga_reference._two_opt_objective(
                        chromosome,
                        matrix,
                        max_iter=20,
                        candidate_k=candidate_k,
                    )
                    actual = memetic_ga._two_opt_objective(
                        chromosome,
                        matrix,
                        max_iter=20,
                        candidate_k=candidate_k,
                    )
                    self.assertEqual(expected, actual)

                    expected = memetic_ga_reference._or_opt_objective(
                        chromosome,
                        matrix,
                        candidate_k=candidate_k,
                    )
                    actual = memetic_ga._or_opt_objective(
                        chromosome,
                        matrix,
                        candidate_k=candidate_k,
                    )
                    self.assertEqual(expected, actual)

    def test_crossovers_match_reference_rng_sequence(self) -> None:
        for n_genes in (8, 12, 20):
            rng = np.random.default_rng(4420 + n_genes)
            parent_a = list(rng.permutation(n_genes))
            parent_b = list(rng.permutation(n_genes))
            operators = (
                (
                    ga_reference._ox1_crossover,
                    ga._ox1_crossover,
                ),
                (
                    memetic_ga_reference._pmx_crossover,
                    memetic_ga._pmx_crossover,
                ),
                (
                    memetic_ga_reference._edge_recombination,
                    memetic_ga._edge_recombination,
                ),
            )
            for seed, (reference, optimized) in enumerate(operators):
                reference_rng = np.random.default_rng(seed + 991)
                optimized_rng = np.random.default_rng(seed + 991)
                self.assertEqual(
                    reference(parent_a, parent_b, reference_rng),
                    optimized(parent_a, parent_b, optimized_rng),
                )

    def test_authoritative_full_run_is_exact(self) -> None:
        rng = np.random.default_rng(24680)
        n_genes = 16
        objective = rng.uniform(
            100.0, 2000.0, size=(n_genes + 1, n_genes + 1)
        )
        np.fill_diagonal(objective, 0.0)
        tensors = np.stack([
            objective * (1.0 + 0.02 * time_bin)
            for time_bin in range(4)
        ])
        duration = rng.integers(
            1, 30, size=(n_genes + 1, n_genes + 1)
        )

        def fitness(chromosome):
            route = [0] + [int(gene) + 1 for gene in chromosome] + [0]
            elapsed = 0
            total = 0.0
            for source, target in zip(route, route[1:]):
                time_bin = min(elapsed // 100, 3)
                total += tensors[time_bin, source, target]
                elapsed += int(duration[source, target])
            return float(total)

        params = {
            "seed": 13579,
            "pop_size": 48,
            "generations": 24,
            "crossover_rate": 0.95,
            "mutation_rate": 0.2,
            "tournament_k": 3,
            "elitism_frac": 0.02,
            "objective_matrix": objective,
            "fitness_fn_authoritative": True,
            "two_opt_max_iter": 30,
            "two_opt_every_n": 5,
            "or_opt_every_n": 5,
            "local_search_top_frac": 0.1,
            "initial_local_search_frac": 0.15,
            "local_search_candidate_k": 8,
            "final_refine_candidate_k": 12,
            "exhaustive_final_max_n": 25,
            "final_refine_best": True,
            "stagnation_window": 30,
            "adaptive_mut_boost": 2.0,
            "inject_frac": 0.2,
            "inject_after": 60,
            "crossover_ops": ["ox1", "pmx", "erx"],
            "crossover_probs": [0.45, 0.25, 0.3],
        }
        reference = memetic_ga_reference.run_memetic_ga(
            n_genes,
            fitness,
            objective[1:, 1:],
            params,
        )
        optimized = memetic_ga.run_memetic_ga(
            n_genes,
            fitness,
            objective[1:, 1:],
            params,
        )
        timing_keys = {
            "runtime_s",
            "cpu_time_s",
            "history_wall_s",
            "history_cpu_s",
        }
        for key in reference:
            if key not in timing_keys:
                self.assertEqual(
                    reference[key],
                    optimized[key],
                    msg=f"Mismatch for {key}",
                )

        expected_budget = params["pop_size"] * (
            params["generations"] + 1
        )
        self.assertEqual(
            optimized["candidate_constructions"], expected_budget
        )


if __name__ == "__main__":
    unittest.main()
