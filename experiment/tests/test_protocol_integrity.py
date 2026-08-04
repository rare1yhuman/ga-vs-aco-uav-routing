import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aco import run_aco
from memetic_ga import run_memetic_ga
from run_experiment import _load_protocol_params
from validate_results import _failures


class ProtocolIntegrityTests(unittest.TestCase):
    def test_memetic_ga_uses_depot_aware_objective_matrix(self):
        matrix = np.array([
            [0, 1, 100, 100],
            [1, 0, 1, 10],
            [100, 1, 0, 1],
            [100, 10, 1, 0],
        ], dtype=float)
        result = run_memetic_ga(
            3,
            lambda chrom: 999999.0,
            matrix[1:, 1:],
            params={
                "pop_size": 20,
                "generations": 30,
                "seed": 7,
                "objective_matrix": matrix,
                "two_opt_every_n": 2,
                "or_opt_every_n": 3,
                "local_search_top_frac": 0.25,
                "initial_local_search_frac": 0.25,
                "early_stop_patience": None,
            },
        )
        self.assertEqual(sorted(result["best_chromosome"]), [0, 1, 2])
        self.assertLess(result["best_fitness"], 120)

    def test_aco_accepts_precomputed_distances_without_graph(self):
        depot = (0, 0)
        dps = [(0, 1), (0, 2), (0, 3)]
        points = [depot] + dps
        distances = {}
        for a in points:
            for b in points:
                if a != b:
                    distances[(a, b)] = abs(a[1] - b[1]) + 1
        result = run_aco(
            G=None,
            grid=None,
            depot=depot,
            delivery_points=dps,
            dijkstra_baseline=10,
            params={"num_ants": 5, "max_iter": 5, "candidate_list_size": 2},
            precomputed_distances=distances,
            build_full_path=False,
            compute_heatmap=False,
        )
        self.assertTrue(np.isfinite(result["best_tour_distance"]))
        self.assertEqual(result["precompute_time_s"], 0.0)

    def test_load_params_from_explicit_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "params.json"
            path.write_text(json.dumps({
                "ga": {"pop_size": 12, "generations": 3},
                "aco": {"num_ants": 8, "max_iter": 4},
            }))
            ga, aco, source = _load_protocol_params(str(path), Path(td) / "results")
        self.assertEqual(ga["pop_size"], 12)
        self.assertEqual(aco["num_ants"], 8)
        self.assertTrue(source.endswith("params.json"))

    def test_validator_accepts_selected_complete_trial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "trial_A").mkdir()
            metadata = {"status": "complete", "trial_ids": ["A"]}
            (root / "run_metadata.json").write_text(json.dumps(metadata))
            pd.DataFrame([{"trial": "A"}]).to_csv(root / "master_results.csv", index=False)
            rows = []
            for algo in ("MGA", "ACO"):
                rows.append({"trial": "A", "algo": algo, "run": 0, "distance_m": 1.0, "error": ""})
            pd.DataFrame(rows).to_csv(root / "trial_A" / "raw_results.csv", index=False)
            (root / "trial_A" / "summary.json").write_text(json.dumps({
                "status": "complete",
                "n_runs": 1,
            }))
            self.assertEqual(_failures(root, expected_runs=1), [])


if __name__ == "__main__":
    unittest.main()
