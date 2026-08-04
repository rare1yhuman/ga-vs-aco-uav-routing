"""Paired, independently replicated MGA-versus-ACO paper experiment.

The legacy A--J runner remains available in ``run_experiment.py``. This runner
implements the paper protocol with seven scenario families, ten independently
generated maps per family, five paired optimizer seeds, randomized execution
order, and resumable per-map checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from aco import run_aco
from grid_gen import DEPOT_POS
from memetic_ga import run_memetic_ga
from research_protocol import (
    FINAL_ENVIRONMENT_SEEDS,
    FINAL_OPTIMIZER_SEEDS,
    SCENARIO_FAMILIES,
    TimeIndexedCost,
    build_research_environment,
    select_families,
)
from run_experiment import DEFAULT_ACO_PARAMS, DEFAULT_GA_PARAMS


ALGORITHMS = ("MGA", "ACO")
PROTOCOL_VERSION = "independent-map-paired-v2"
_WORKER_ENV: Dict[str, Any] = {}
_WORKER_GA: Dict[str, Any] = {}
_WORKER_ACO: Dict[str, Any] = {}


def _json_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=float)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _normalise_params(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ga = {**DEFAULT_GA_PARAMS, **dict(payload.get("ga", {}))}
    aco = {**DEFAULT_ACO_PARAMS, **dict(payload.get("aco", {}))}
    ga.pop("dynamic_risk_weight", None)
    ga["fitness_fn_authoritative"] = True
    for key in ("pop_size", "generations", "tournament_k", "two_opt_max_iter",
                "two_opt_every_n", "or_opt_every_n", "stagnation_window",
                "inject_after", "local_search_candidate_k",
                "final_refine_candidate_k", "exhaustive_final_max_n"):
        if ga.get(key) is not None:
            ga[key] = int(ga[key])
    for key in ("num_ants", "max_iter"):
        aco[key] = int(aco[key])
    return ga, aco


def load_protocol_config(
    path: Path,
    *,
    allow_provisional: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(path.read_text())
    protocol = dict(payload.get("protocol", {}))
    if (
        protocol.get("status") != "calibrated"
        and not allow_provisional
    ):
        raise RuntimeError(
            f"{path} is {protocol.get('status', 'unlabelled')!r}. "
            "Run calibrate_protocol_v2.py, or use --allow-provisional only "
            "for smoke tests and pilot runs."
        )
    ga, aco = _normalise_params(payload)
    ga_budget = int(ga["pop_size"]) * (int(ga["generations"]) + 1)
    aco_budget = int(aco["num_ants"]) * int(aco["max_iter"])
    if ga_budget != aco_budget:
        raise ValueError(
            f"Unequal construction budgets: MGA={ga_budget}, ACO={aco_budget}"
        )
    return protocol, ga, aco


def _nearest_neighbour_reference(matrix: np.ndarray) -> float:
    unvisited = set(range(1, matrix.shape[0]))
    current = 0
    total = 0.0
    while unvisited:
        next_node = min(unvisited, key=lambda node: matrix[current, node])
        total += float(matrix[current, next_node])
        current = next_node
        unvisited.remove(next_node)
    total += float(matrix[current, 0])
    return max(total, 1.0)


def _init_pair_worker(
    worker_env: Dict[str, Any],
    ga_params: Dict[str, Any],
    aco_params: Dict[str, Any],
) -> None:
    global _WORKER_ENV, _WORKER_GA, _WORKER_ACO
    _WORKER_ENV = worker_env
    _WORKER_GA = ga_params
    _WORKER_ACO = aco_params


def _chromosome_from_aco(
    route: Sequence[Tuple[int, int]],
    time_cost: TimeIndexedCost,
) -> List[int]:
    return [
        int(time_cost.node_to_index[point]) - 1
        for point in route[1:-1]
    ]


def _base_record(
    algorithm: str,
    optimizer_seed: int,
    pair_order: str,
) -> Dict[str, Any]:
    family = _WORKER_ENV["family"]
    return {
        "protocol": PROTOCOL_VERSION,
        "family": _WORKER_ENV["family_id"],
        "family_label": family["label"],
        "environment_seed": int(_WORKER_ENV["environment_seed"]),
        "optimizer_seed": int(optimizer_seed),
        "pair_order": pair_order,
        "algo": algorithm,
        "n_deliveries": int(family["n_deliveries"]),
        "nfz_count": int(family["nfz_count"]),
        "dynamic": bool(family["dynamic"]),
        "use_3d": bool(family["use_3d"]),
        "error": "",
    }


def _run_mga(optimizer_seed: int, pair_order: str) -> Dict[str, Any]:
    time_cost: TimeIndexedCost = _WORKER_ENV["time_cost"]
    n_deliveries = len(_WORKER_ENV["delivery_points"])
    mean_objective = np.mean(time_cost.cost_tensor, axis=0)
    params = {
        **_WORKER_GA,
        "seed": int(optimizer_seed),
        "objective_matrix": mean_objective,
        "fitness_fn_authoritative": True,
    }

    def fitness(chromosome: Sequence[int]) -> float:
        return float(time_cost.route_metrics(chromosome)["generalized_cost"])

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = run_memetic_ga(
        n_genes=n_deliveries,
        fitness_fn=fitness,
        dist_matrix=mean_objective[1:, 1:],
        params=params,
        verbose=False,
    )
    runner_wall = time.perf_counter() - wall_start
    runner_cpu = time.process_time() - cpu_start
    chromosome = [int(gene) for gene in result["best_chromosome"]]
    metrics = time_cost.route_metrics(chromosome)
    record = _base_record("MGA", optimizer_seed, pair_order)
    record.update({
        **metrics,
        "dynamic_safe": bool(metrics["exposure_cells"] == 0),
        "best_path": _json_value(chromosome),
        "wall_time_s": float(runner_wall),
        "cpu_time_s": float(runner_cpu),
        "native_wall_time_s": float(result["runtime_s"]),
        "native_cpu_time_s": float(result["cpu_time_s"]),
        "candidate_constructions": int(result["candidate_constructions"]),
        "fitness_evaluations": int(result["fitness_evaluations"]),
        "iterations_completed": int(result["generations_completed"]),
        "history_cost": _json_value(result["history"]),
        "history_wall_s": _json_value(result["history_wall_s"]),
        "history_cpu_s": _json_value(result["history_cpu_s"]),
        "history_evaluations": _json_value(result["history_evaluations"]),
    })
    return record


def _run_aco(optimizer_seed: int, pair_order: str) -> Dict[str, Any]:
    time_cost: TimeIndexedCost = _WORKER_ENV["time_cost"]
    random.seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed))
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = run_aco(
        G=None,
        grid=None,
        depot=DEPOT_POS,
        delivery_points=_WORKER_ENV["delivery_points"],
        dijkstra_baseline=_WORKER_ENV["reference_cost"],
        params={**_WORKER_ACO},
        precomputed_distances=_WORKER_ENV["static_cost_distances"],
        build_full_path=False,
        compute_heatmap=False,
        leg_cost_provider=time_cost.leg_by_point,
    )
    runner_wall = time.perf_counter() - wall_start
    runner_cpu = time.process_time() - cpu_start
    chromosome = _chromosome_from_aco(result["best_tour_order"], time_cost)
    metrics = time_cost.route_metrics(chromosome)
    record = _base_record("ACO", optimizer_seed, pair_order)
    record.update({
        **metrics,
        "dynamic_safe": bool(metrics["exposure_cells"] == 0),
        "best_path": _json_value(chromosome),
        "wall_time_s": float(runner_wall),
        "cpu_time_s": float(runner_cpu),
        "native_wall_time_s": float(result["compute_time_s"]),
        "native_cpu_time_s": float(result["cpu_time_s"]),
        "candidate_constructions": int(result["candidate_constructions"]),
        "fitness_evaluations": int(result["candidate_constructions"]),
        "iterations_completed": int(result["iterations_completed"]),
        "history_cost": _json_value(result["convergence_curve"]),
        "history_wall_s": _json_value(result["convergence_wall_s"]),
        "history_cpu_s": _json_value(result["convergence_cpu_s"]),
        "history_evaluations": _json_value(result["convergence_evaluations"]),
    })
    return record


def _pair_worker(optimizer_seed: int) -> List[Dict[str, Any]]:
    order = list(ALGORITHMS)
    order_rng = random.Random(
        int(optimizer_seed) * 1000003
        + int(_WORKER_ENV["environment_seed"]) * 9176
        + sum(ord(char) for char in _WORKER_ENV["family_id"])
    )
    order_rng.shuffle(order)
    order_label = "->".join(order)
    records: List[Dict[str, Any]] = []
    for algorithm in order:
        try:
            records.append(
                _run_mga(optimizer_seed, order_label)
                if algorithm == "MGA"
                else _run_aco(optimizer_seed, order_label)
            )
        except Exception as exc:
            record = _base_record(algorithm, optimizer_seed, order_label)
            record.update({
                "generalized_cost": float("inf"),
                "base_cost": float("inf"),
                "physical_distance_m": float("inf"),
                "exposure_cells": float("inf"),
                "estimated_duration_steps": float("inf"),
                "dynamic_safe": False,
                "best_path": "[]",
                "wall_time_s": float("nan"),
                "cpu_time_s": float("nan"),
                "native_wall_time_s": float("nan"),
                "native_cpu_time_s": float("nan"),
                "candidate_constructions": 0,
                "fitness_evaluations": 0,
                "iterations_completed": 0,
                "history_cost": "[]",
                "history_wall_s": "[]",
                "history_cpu_s": "[]",
                "history_evaluations": "[]",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            records.append(record)
    return records


def _worker_environment(env: Mapping[str, Any]) -> Dict[str, Any]:
    time_cost: TimeIndexedCost = env["time_cost"]
    return {
        "family_id": env["family_id"],
        "family": env["family"],
        "environment_seed": env["environment_seed"],
        "delivery_points": env["delivery_points"],
        "static_cost_distances": env["static_cost_distances"],
        "time_cost": time_cost,
        "reference_cost": _nearest_neighbour_reference(
            time_cost.base_cost_matrix
        ),
    }


def _environment_is_complete(
    directory: Path,
    optimizer_seeds: Sequence[int],
    params_hash: str,
) -> bool:
    raw_path = directory / "raw_results.csv"
    summary_path = directory / "summary.json"
    if not raw_path.exists() or not summary_path.exists():
        return False
    try:
        rows = pd.read_csv(raw_path)
        summary = json.loads(summary_path.read_text())
    except Exception:
        return False
    expected = {
        (algorithm, int(seed))
        for algorithm in ALGORITHMS
        for seed in optimizer_seeds
    }
    actual = {
        (str(row.algo), int(row.optimizer_seed))
        for row in rows.itertuples()
    }
    return (
        summary.get("status") == "complete"
        and summary.get("params_hash") == params_hash
        and actual == expected
        and not rows["error"].fillna("").astype(str).str.strip().any()
    )


def _environment_summary(
    family_id: str,
    environment_seed: int,
    rows: Sequence[Mapping[str, Any]],
    params_hash: str,
) -> Dict[str, Any]:
    frame = pd.DataFrame(rows)
    algorithms: Dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        subset = frame[frame["algo"] == algorithm]
        algorithms[algorithm] = {
            "median_generalized_cost": float(subset["generalized_cost"].median()),
            "mean_generalized_cost": float(subset["generalized_cost"].mean()),
            "median_cpu_time_s": float(subset["cpu_time_s"].median()),
            "dynamic_safety_rate": float(subset["dynamic_safe"].mean()),
            "n": int(len(subset)),
        }
    errors = frame["error"].fillna("").astype(str).str.strip()
    return {
        "status": "complete" if not errors.any() else "failed",
        "protocol": PROTOCOL_VERSION,
        "family": family_id,
        "environment_seed": int(environment_seed),
        "params_hash": params_hash,
        "algorithms": algorithms,
        "error_count": int(errors.astype(bool).sum()),
    }


def _combine_results(output_dir: Path) -> pd.DataFrame:
    paths = sorted(output_dir.glob("S*/env_*/raw_results.csv"))
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        temporary = output_dir / "raw_results.csv.tmp"
        combined.to_csv(temporary, index=False)
        temporary.replace(output_dir / "raw_results.csv")
    return combined


def _metadata(
    *,
    status: str,
    family_ids: Sequence[str],
    environment_seeds: Sequence[int],
    optimizer_seeds: Sequence[int],
    workers: int,
    config_path: Path,
    protocol: Mapping[str, Any],
    ga_params: Mapping[str, Any],
    aco_params: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "status": status,
        "protocol": PROTOCOL_VERSION,
        "families": list(family_ids),
        "scenario_definitions": {
            family_id: SCENARIO_FAMILIES[family_id]
            for family_id in family_ids
        },
        "environment_seeds": list(map(int, environment_seeds)),
        "optimizer_seeds": list(map(int, optimizer_seeds)),
        "workers": int(workers),
        "expected_rows": (
            len(family_ids) * len(environment_seeds)
            * len(optimizer_seeds) * len(ALGORITHMS)
        ),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "params_hash": _hash_payload({"ga": ga_params, "aco": aco_params}),
        "protocol_config": dict(protocol),
        "ga": dict(ga_params),
        "aco": dict(aco_params),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_protocol(
    family_ids: Sequence[str],
    environment_seeds: Sequence[int],
    optimizer_seeds: Sequence[int],
    workers: int,
    output_dir: Path,
    config_path: Path,
    *,
    clean_output: bool,
    resume: bool,
    allow_provisional: bool,
) -> None:
    if clean_output and resume:
        raise ValueError("--clean-output and --resume cannot be combined")
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol, ga_params, aco_params = load_protocol_config(
        config_path, allow_provisional=allow_provisional
    )
    params_hash = _hash_payload({"ga": ga_params, "aco": aco_params})
    metadata_args = dict(
        family_ids=family_ids,
        environment_seeds=environment_seeds,
        optimizer_seeds=optimizer_seeds,
        workers=workers,
        config_path=config_path,
        protocol=protocol,
        ga_params=ga_params,
        aco_params=aco_params,
    )
    _atomic_json(
        output_dir / "run_metadata.json",
        _metadata(status="running", **metadata_args),
    )

    print(f"[research] Families: {', '.join(family_ids)}")
    print(
        f"[research] {len(environment_seeds)} maps/family, "
        f"{len(optimizer_seeds)} paired seeds/map, {workers} workers"
    )
    print(
        f"[research] Equal construction budget: "
            f"{ga_params['pop_size'] * (ga_params['generations'] + 1):,} "
            "per algorithm/run"
    )
    instances = [
        (family_id, int(environment_seed))
        for family_id in family_ids
        for environment_seed in environment_seeds
    ]
    failures: List[str] = []
    started = time.perf_counter()
    progress = tqdm(instances, desc="Research protocol", unit="map")
    for family_id, environment_seed in progress:
        progress.set_postfix_str(f"{family_id}/seed={environment_seed}")
        directory = output_dir / family_id / f"env_{environment_seed}"
        if resume and _environment_is_complete(
            directory, optimizer_seeds, params_hash
        ):
            continue
        try:
            env = build_research_environment(
                family_id,
                SCENARIO_FAMILIES[family_id],
                environment_seed,
            )
            worker_env = _worker_environment(env)
            if workers == 1:
                _init_pair_worker(worker_env, ga_params, aco_params)
                batches = [
                    _pair_worker(seed)
                    for seed in tqdm(
                        optimizer_seeds,
                        desc=f"{family_id}-{environment_seed}",
                        leave=False,
                    )
                ]
            else:
                with Pool(
                    processes=min(workers, len(optimizer_seeds)),
                    initializer=_init_pair_worker,
                    initargs=(worker_env, ga_params, aco_params),
                ) as pool:
                    batches = list(tqdm(
                        pool.imap_unordered(_pair_worker, optimizer_seeds),
                        total=len(optimizer_seeds),
                        desc=f"{family_id}-{environment_seed}",
                        leave=False,
                    ))
            rows = sorted(
                [record for batch in batches for record in batch],
                key=lambda row: (int(row["optimizer_seed"]), str(row["algo"])),
            )
            summary = _environment_summary(
                family_id, environment_seed, rows, params_hash
            )
            _atomic_csv(directory / "raw_results.csv", rows)
            _atomic_json(directory / "summary.json", summary)
            if summary["status"] != "complete":
                failures.append(
                    f"{family_id}/seed={environment_seed}: "
                    f"{summary['error_count']} algorithm errors"
                )
        except Exception as exc:
            failures.append(
                f"{family_id}/seed={environment_seed}: "
                f"{type(exc).__name__}: {exc}"
            )
            _atomic_json(directory / "failure.json", {
                "family": family_id,
                "environment_seed": environment_seed,
                "error": failures[-1],
                "traceback": traceback.format_exc(),
            })
        _combine_results(output_dir)

    combined = _combine_results(output_dir)
    final_status = "complete" if not failures else "failed"
    final_metadata = _metadata(status=final_status, **metadata_args)
    final_metadata.update({
        "elapsed_wall_s": time.perf_counter() - started,
        "actual_rows": int(len(combined)),
        "failures": failures,
    })
    _atomic_json(output_dir / "run_metadata.json", final_metadata)
    if failures:
        raise RuntimeError(
            "Research run has failed map instances:\n  - "
            + "\n  - ".join(failures)
        )


def _parse_ints(values: Iterable[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        parsed.extend(int(item) for item in value.split(",") if item.strip())
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", default=["all"],
        help="Scenario families S1-S7, or all.",
    )
    parser.add_argument(
        "--environment-seeds", nargs="+",
        default=[str(seed) for seed in FINAL_ENVIRONMENT_SEEDS],
    )
    parser.add_argument(
        "--optimizer-seeds", nargs="+",
        default=[str(seed) for seed in FINAL_OPTIMIZER_SEEDS],
    )
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/paper_protocol_v2.json"),
    )
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-provisional", action="store_true",
        help="Permit an uncalibrated config for smoke tests only.",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run S3 on one map and one optimizer seed.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.pilot:
        family_ids = ["S3"]
        environment_seeds = [42]
        optimizer_seeds = [2000]
        args.allow_provisional = True
    else:
        family_ids = list(select_families(args.families))
        environment_seeds = _parse_ints(args.environment_seeds)
        optimizer_seeds = _parse_ints(args.optimizer_seeds)
    run_protocol(
        family_ids,
        environment_seeds,
        optimizer_seeds,
        args.workers,
        args.output,
        args.config,
        clean_output=args.clean_output,
        resume=args.resume,
        allow_provisional=args.allow_provisional,
    )
    print(f"[research] Complete: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
