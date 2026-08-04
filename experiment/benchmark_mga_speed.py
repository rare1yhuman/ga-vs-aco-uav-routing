"""Paired equivalence and CPU benchmark for reference versus optimized MGA."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from memetic_ga import run_memetic_ga as run_optimized
from memetic_ga_reference import run_memetic_ga as run_reference
from research_protocol import (
    CALIBRATION_ENVIRONMENT_SEEDS,
    CALIBRATION_FAMILIES,
    CALIBRATION_OPTIMIZER_SEEDS,
    TimeIndexedCost,
    build_research_environment,
)


IMPLEMENTATIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "reference": run_reference,
    "optimized": run_optimized,
}

EXACT_RESULT_KEYS = (
    "best_chromosome",
    "best_fitness",
    "history",
    "history_evaluations",
    "convergence_gen",
    "generations_completed",
    "candidate_constructions",
    "two_opt_improvements",
    "or_opt_improvements",
    "depot_rot_improvements",
    "diversity_injections",
    "crossover_counts",
    "fitness_evaluations",
)


def _load_ga_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    protocol = payload.get("protocol", {})
    if protocol.get("status") != "calibrated":
        raise RuntimeError(
            f"{path} must contain a calibrated protocol configuration"
        )
    return dict(payload["ga"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_equivalent(
    reference: Mapping[str, Any],
    optimized: Mapping[str, Any],
    label: str,
) -> None:
    mismatches = [
        key for key in EXACT_RESULT_KEYS
        if reference.get(key) != optimized.get(key)
    ]
    if mismatches:
        details = {
            key: {
                "reference": reference.get(key),
                "optimized": optimized.get(key),
            }
            for key in mismatches
        }
        raise RuntimeError(
            f"Output equivalence failed for {label}: "
            f"{json.dumps(details, default=str)[:4000]}"
        )


def _run_once(
    implementation: str,
    runner: Callable[..., Dict[str, Any]],
    time_cost: TimeIndexedCost,
    n_deliveries: int,
    ga_config: Mapping[str, Any],
    optimizer_seed: int,
    profile_path: Path | None = None,
) -> Dict[str, Any]:
    mean_objective = np.mean(time_cost.cost_tensor, axis=0)
    params = {
        **ga_config,
        "seed": int(optimizer_seed),
        "objective_matrix": mean_objective,
        "fitness_fn_authoritative": True,
    }

    def fitness(chromosome: Sequence[int]) -> float:
        return float(
            time_cost.route_metrics(chromosome)["generalized_cost"]
        )

    args = (
        n_deliveries,
        fitness,
        mean_objective[1:, 1:],
        params,
        False,
    )
    if profile_path is None:
        return runner(*args)
    profiler = cProfile.Profile()
    result = profiler.runcall(runner, *args)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(profile_path)
    return result


def _family_seed(family_id: str) -> int:
    family_ids = list(CALIBRATION_FAMILIES)
    return int(
        CALIBRATION_ENVIRONMENT_SEEDS[family_ids.index(family_id)]
    )


def run_benchmark(
    families: Iterable[str],
    optimizer_seeds: Sequence[int],
    repeats: int,
    config_path: Path,
    output_dir: Path,
    profile: bool,
    enforce_gate: bool,
    min_c50_speedup: float,
    min_c20_speedup: float,
) -> pd.DataFrame:
    ga_config = _load_ga_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for family_id in families:
        family = CALIBRATION_FAMILIES[family_id]
        environment_seed = _family_seed(family_id)
        print(
            f"[speed] Building {family_id}/seed={environment_seed}",
            flush=True,
        )
        environment = build_research_environment(
            family_id, family, environment_seed
        )
        time_cost: TimeIndexedCost = environment["time_cost"]
        n_deliveries = len(environment["delivery_points"])

        for repeat in range(repeats):
            for optimizer_seed in optimizer_seeds:
                order = list(IMPLEMENTATIONS)
                random.Random(
                    730001 * environment_seed
                    + 1009 * int(optimizer_seed)
                    + repeat
                ).shuffle(order)
                results: Dict[str, Dict[str, Any]] = {}
                for implementation in order:
                    profile_path = None
                    if profile:
                        profile_path = (
                            output_dir
                            / "profiles"
                            / (
                                f"{family_id}_{optimizer_seed}_r{repeat}_"
                                f"{implementation}.prof"
                            )
                        )
                    print(
                        f"[speed] {family_id} seed={optimizer_seed} "
                        f"repeat={repeat} {implementation}",
                        flush=True,
                    )
                    result = _run_once(
                        implementation,
                        IMPLEMENTATIONS[implementation],
                        time_cost,
                        n_deliveries,
                        ga_config,
                        int(optimizer_seed),
                        profile_path,
                    )
                    results[implementation] = result
                    rows.append({
                        "family": family_id,
                        "environment_seed": environment_seed,
                        "optimizer_seed": int(optimizer_seed),
                        "repeat": repeat,
                        "implementation": implementation,
                        "cpu_time_s": float(result["cpu_time_s"]),
                        "wall_time_s": float(result["runtime_s"]),
                        "best_fitness": float(result["best_fitness"]),
                        "candidate_constructions": int(
                            result["candidate_constructions"]
                        ),
                        "fitness_evaluations": int(
                            result["fitness_evaluations"]
                        ),
                        "profiled": bool(profile),
                    })
                _assert_equivalent(
                    results["reference"],
                    results["optimized"],
                    (
                        f"{family_id}/seed={optimizer_seed}/"
                        f"repeat={repeat}"
                    ),
                )
                print("[speed] exact equivalence: PASS", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "benchmark_runs.csv", index=False)
    pivot = frame.pivot_table(
        index=[
            "family", "environment_seed", "optimizer_seed", "repeat"
        ],
        columns="implementation",
        values="cpu_time_s",
        aggfunc="first",
    ).reset_index()
    pivot["cpu_speedup"] = pivot["reference"] / pivot["optimized"]
    pivot.to_csv(output_dir / "paired_speedups.csv", index=False)
    summary = (
        pivot.groupby("family")
        .agg(
            pairs=("cpu_speedup", "size"),
            median_reference_cpu_s=("reference", "median"),
            median_optimized_cpu_s=("optimized", "median"),
            median_speedup=("cpu_speedup", "median"),
            min_speedup=("cpu_speedup", "min"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    (output_dir / "equivalence_report.json").write_text(
        json.dumps({
            "status": "pass",
            "exact_keys": list(EXACT_RESULT_KEYS),
            "families": list(families),
            "optimizer_seeds": [int(seed) for seed in optimizer_seeds],
            "repeats": int(repeats),
            "profiled": bool(profile),
            "implementation_sha256": {
                "reference": _sha256(
                    Path(__file__).with_name("memetic_ga_reference.py")
                ),
                "optimized": _sha256(
                    Path(__file__).with_name("memetic_ga.py")
                ),
            },
        }, indent=2)
    )
    print("\n" + summary.to_string(index=False))
    if enforce_gate:
        if profile:
            raise RuntimeError(
                "Do not enforce the speed gate on profiled executions"
            )
        speedups = dict(
            zip(summary["family"], summary["median_speedup"])
        )
        failures = []
        if "C20" in speedups and speedups["C20"] < min_c20_speedup:
            failures.append(
                f"C20 speedup {speedups['C20']:.3f} "
                f"< {min_c20_speedup:.3f}"
            )
        if "C50" in speedups and speedups["C50"] < min_c50_speedup:
            failures.append(
                f"C50 speedup {speedups['C50']:.3f} "
                f"< {min_c50_speedup:.3f}"
            )
        if failures:
            raise RuntimeError(
                "Speed acceptance gate failed: " + "; ".join(failures)
            )
        print("[speed] acceptance gate: PASS")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(CALIBRATION_FAMILIES),
        default=list(CALIBRATION_FAMILIES),
    )
    parser.add_argument(
        "--optimizer-seeds",
        nargs="+",
        type=int,
        default=list(CALIBRATION_OPTIMIZER_SEEDS),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/paper_protocol_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("speed_benchmark"),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Write cProfile data; use a single family/seed for profiling.",
    )
    parser.add_argument("--enforce-gate", action="store_true")
    parser.add_argument("--min-c50-speedup", type=float, default=1.20)
    parser.add_argument("--min-c20-speedup", type=float, default=0.95)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least one")
    run_benchmark(
        args.families,
        args.optimizer_seeds,
        args.repeats,
        args.config,
        args.output_dir,
        args.profile,
        args.enforce_gate,
        args.min_c50_speedup,
        args.min_c20_speedup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
