"""Profile the frozen ACO path without changing its transition semantics."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import random
import time

import numpy as np

from aco import run_aco
from grid_gen import DEPOT_POS
from research_protocol import (
    CALIBRATION_ENVIRONMENT_SEEDS,
    CALIBRATION_FAMILIES,
    CALIBRATION_OPTIMIZER_SEEDS,
    build_research_environment,
)
from run_research_experiment import (
    _nearest_neighbour_reference,
    load_protocol_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/paper_protocol_v2_optimized.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evidence/aco_implementation_audit"),
    )
    parser.add_argument("--sample-ants", type=int, default=25)
    parser.add_argument("--sample-iterations", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, _, frozen_aco_params = load_protocol_config(args.config)
    aco_params = {
        **frozen_aco_params,
        "num_ants": int(args.sample_ants),
        "max_iter": int(args.sample_iterations),
    }
    family_id = "C20"
    environment_seed = int(CALIBRATION_ENVIRONMENT_SEEDS[0])
    optimizer_seed = int(CALIBRATION_OPTIMIZER_SEEDS[0])
    environment = build_research_environment(
        family_id,
        CALIBRATION_FAMILIES[family_id],
        environment_seed,
    )
    time_cost = environment["time_cost"]
    reference_cost = _nearest_neighbour_reference(
        time_cost.base_cost_matrix
    )
    random.seed(optimizer_seed)
    np.random.seed(optimizer_seed)

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    result = run_aco(
        G=None,
        grid=None,
        depot=DEPOT_POS,
        delivery_points=environment["delivery_points"],
        dijkstra_baseline=reference_cost,
        params=aco_params,
        precomputed_distances=environment["static_cost_distances"],
        build_full_path=False,
        compute_heatmap=False,
        leg_cost_provider=time_cost.leg_by_point,
    )
    profiler.disable()
    wall_time = time.perf_counter() - wall_start

    profile_path = args.output_dir / "aco_c20.prof"
    profiler.dump_stats(profile_path)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
    stats.sort_stats("cumulative").print_stats(30)
    profile_text = stream.getvalue()
    (args.output_dir / "aco_c20_top30.txt").write_text(profile_text)

    expected_sample_budget = int(args.sample_ants) * int(args.sample_iterations)
    if int(result["candidate_constructions"]) != expected_sample_budget:
        raise RuntimeError("ACO audit sample did not complete its declared budget")
    report = {
        "status": "complete",
        "scope": (
            "reduced-budget source-level profile; not inferential timing and "
            "not part of the final experiment"
        ),
        "family": family_id,
        "environment_seed": environment_seed,
        "optimizer_seed": optimizer_seed,
        "config_sha256": _sha256(args.config),
        "sample_candidate_constructions": int(
            result["candidate_constructions"]
        ),
        "frozen_candidate_constructions": int(
            frozen_aco_params["num_ants"]
            * frozen_aco_params["max_iter"]
        ),
        "sample_ants": int(args.sample_ants),
        "sample_iterations": int(args.sample_iterations),
        "best_cost": float(result["best_tour_distance"]),
        "iterations": int(result["iterations_completed"]),
        "profiled_wall_s": wall_time,
        "full_path_building": False,
        "pheromone_heatmap": False,
        "shared_environment_precompute_included": False,
        "transition_order_changed": False,
        "random_draw_sequence_changed": False,
        "decision": (
            "Retain frozen ACO. Post-hoc vectorization or candidate-order "
            "rewrites were not adopted because they could change floating-"
            "point accumulation, tie order, or random transition decisions."
        ),
    }
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
