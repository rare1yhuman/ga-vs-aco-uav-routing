import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aco import get_path_score, initialize_delivery_path_score
from calibrate_protocol_v2 import generate_candidates
from research_protocol import TimeIndexedCost
from run_research_experiment import load_protocol_config


class ResearchProtocolV2Tests(unittest.TestCase):
    def test_time_indexed_route_uses_elapsed_leg_time(self):
        nodes = {(0, 0): 0, (0, 1): 1, (0, 2): 2}
        base = np.array([
            [0, 10, 10],
            [10, 0, 10],
            [10, 10, 0],
        ], dtype=float)
        duration = np.array([
            [0, 6, 6],
            [6, 0, 6],
            [6, 6, 0],
        ], dtype=int)
        tensor = np.stack([base, base + 100])
        exposure = np.zeros_like(tensor)
        cost = TimeIndexedCost(
            node_to_index=nodes,
            base_cost_matrix=base,
            physical_distance_matrix=base,
            duration_matrix=duration,
            cost_tensor=tensor,
            exposure_tensor=exposure,
            bin_width_steps=5,
        )
        metrics = cost.route_metrics([0, 1])
        self.assertEqual(metrics["generalized_cost"], 230.0)
        self.assertEqual(metrics["estimated_duration_steps"], 18.0)

    def test_calibration_candidates_have_equal_100k_budget(self):
        candidates = generate_candidates(24)
        self.assertEqual(len(candidates["MGA"]), 24)
        self.assertEqual(len(candidates["ACO"]), 24)
        self.assertTrue(all(
            config["pop_size"] * (config["generations"] + 1) == 100000
            for config in candidates["MGA"]
        ))
        self.assertTrue(all(
            config["num_ants"] * config["max_iter"] == 100000
            for config in candidates["ACO"]
        ))

    def test_final_runner_rejects_provisional_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps({
                "protocol": {"status": "provisional"},
                "ga": {"pop_size": 10, "generations": 9},
                "aco": {"num_ants": 10, "max_iter": 10},
            }))
            with self.assertRaises(RuntimeError):
                load_protocol_config(path)

    def test_pilot_can_explicitly_load_provisional_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps({
                "protocol": {"status": "provisional"},
                "ga": {"pop_size": 10, "generations": 9},
                "aco": {"num_ants": 10, "max_iter": 10},
            }))
            _, ga, aco = load_protocol_config(path, allow_provisional=True)
            self.assertEqual(ga["pop_size"] * (ga["generations"] + 1), 100)
            self.assertEqual(aco["num_ants"] * aco["max_iter"], 100)

    def test_dynamic_aco_pheromone_can_be_directional(self):
        a, b = (0, 0), (0, 1)
        scores = initialize_delivery_path_score([a, b], 1.0, directed=True)
        scores[(a, b)] = 2.0
        self.assertEqual(get_path_score(scores, a, b, directed=True), 2.0)
        self.assertEqual(get_path_score(scores, b, a, directed=True), 1.0)


if __name__ == "__main__":
    unittest.main()
