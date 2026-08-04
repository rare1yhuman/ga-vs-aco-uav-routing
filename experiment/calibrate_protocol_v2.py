"""Held-out, equal-budget calibration for the v2 paper protocol."""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from research_protocol import (
    CALIBRATION_ENVIRONMENT_SEEDS,
    CALIBRATION_FAMILIES,
    CALIBRATION_OPTIMIZER_SEEDS,
    build_research_environment,
)
from run_research_experiment import (
    PROTOCOL_VERSION,
    _hash_payload,
    _init_pair_worker,
    _run_aco,
    _run_mga,
    _worker_environment,
)


GA_BUDGET_SHAPES = ((100, 999), (200, 499), (250, 399))
ACO_BUDGET_SHAPES = ((100, 1000), (200, 500), (250, 400))
RNG_SEED = 20260726
_CALIBRATION_BASE_GA: Dict[str, Any] = {}
_CALIBRATION_BASE_ACO: Dict[str, Any] = {}


def _sample_ga_configs(count: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    configs: List[Dict[str, Any]] = []
    seen = set()
    while len(configs) < count:
        pop_size, generations = GA_BUDGET_SHAPES[
            int(rng.integers(0, len(GA_BUDGET_SHAPES)))
        ]
        config = {
            "pop_size": pop_size,
            "generations": generations,
            "crossover_rate": float(rng.choice([0.75, 0.85, 0.95])),
            "mutation_rate": float(rng.choice([0.05, 0.10, 0.15, 0.20])),
            "tournament_k": int(rng.choice([3, 5, 7])),
            "elitism_frac": float(rng.choice([0.02, 0.05, 0.10])),
            "two_opt_max_iter": int(rng.choice([75, 125, 175])),
            "two_opt_every_n": int(rng.choice([2, 3, 5])),
            "or_opt_every_n": int(rng.choice([3, 5, 7])),
            "local_search_top_frac": float(rng.choice([0.05, 0.10, 0.20])),
            "initial_local_search_frac": float(rng.choice([0.10, 0.15, 0.25])),
            "local_search_candidate_k": int(rng.choice([6, 8, 12])),
            "final_refine_candidate_k": 20,
            "exhaustive_final_max_n": 25,
            "final_refine_best": True,
            "early_stop_patience": None,
            "stagnation_window": int(rng.choice([20, 30, 40])),
            "adaptive_mut_boost": float(rng.choice([2.0, 3.0, 4.0])),
            "inject_frac": float(rng.choice([0.10, 0.20])),
            "inject_after": int(rng.choice([40, 60, 80])),
            "crossover_ops": ["ox1", "pmx", "erx"],
            "crossover_probs": [0.45, 0.25, 0.30],
            "fitness_fn_authoritative": True,
        }
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def _sample_aco_configs(count: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED + 1)
    configs: List[Dict[str, Any]] = []
    seen = set()
    while len(configs) < count:
        num_ants, max_iter = ACO_BUDGET_SHAPES[
            int(rng.integers(0, len(ACO_BUDGET_SHAPES)))
        ]
        config = {
            "num_ants": num_ants,
            "max_iter": max_iter,
            "alpha": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            "beta": float(rng.choice([1.0, 2.0, 3.0, 5.0])),
            "rho_global": float(rng.choice([0.05, 0.10, 0.20])),
            "rho_local": float(rng.choice([0.005, 0.01, 0.02])),
            "q0": float(rng.choice([0.70, 0.80, 0.90, 0.95])),
            "candidate_list_size": int(rng.choice([10, 15, 20])),
        }
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def generate_candidates(count: int = 24) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "MGA": _sample_ga_configs(count),
        "ACO": _sample_aco_configs(count),
    }


def _calibration_worker(
    task: Tuple[str, int, Dict[str, Any], int],
) -> Dict[str, Any]:
    algorithm, config_id, params, optimizer_seed = task
    global _CALIBRATION_BASE_GA, _CALIBRATION_BASE_ACO
    order = f"calibration-{algorithm}"
    import run_research_experiment as runner
    try:
        if algorithm == "MGA":
            _CALIBRATION_BASE_GA = params
            runner._WORKER_GA = params
            result = _run_mga(optimizer_seed, order)
        else:
            _CALIBRATION_BASE_ACO = params
            runner._WORKER_ACO = params
            result = _run_aco(optimizer_seed, order)
    except Exception as exc:
        result = runner._base_record(algorithm, optimizer_seed, order)
        result.update({
            "generalized_cost": float("inf"),
            "dynamic_safe": False,
            "cpu_time_s": float("inf"),
            "wall_time_s": float("inf"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    result["config_id"] = int(config_id)
    result["config_json"] = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return result


def _rank_configs(frame: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    subset = frame[frame["algo"] == algorithm].copy()
    medians = subset.groupby(["family", "config_id"])["generalized_cost"].transform("median")
    family_best = medians.groupby(subset["family"]).transform("min")
    subset["normalised_cost"] = subset["generalized_cost"] / family_best
    rows = []
    for config_id, group in subset.groupby("config_id"):
        rows.append({
            "algo": algorithm,
            "config_id": int(config_id),
            "safety_rate": float(group["dynamic_safe"].mean()),
            "median_normalised_cost": float(group["normalised_cost"].median()),
            "std_normalised_cost": float(group["normalised_cost"].std(ddof=1)),
            "median_cpu_time_s": float(group["cpu_time_s"].median()),
            "error_count": int(group["error"].fillna("").astype(str).str.strip().astype(bool).sum()),
            "config_json": str(group["config_json"].iloc[0]),
        })
    ranked = pd.DataFrame(rows).sort_values(
        [
            "error_count",
            "safety_rate",
            "median_normalised_cost",
            "std_normalised_cost",
            "median_cpu_time_s",
            "config_id",
        ],
        ascending=[True, False, True, True, True, True],
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _write_calibrated_config(
    output_config: Path,
    ga: Mapping[str, Any],
    aco: Mapping[str, Any],
    calibration_csv: Path,
    ranking_csv: Path,
) -> None:
    payload = {
        "protocol": {
            "name": PROTOCOL_VERSION,
            "status": "calibrated",
            "candidate_constructions_per_run": 100000,
            "calibration_environment_seeds": list(CALIBRATION_ENVIRONMENT_SEEDS),
            "calibration_optimizer_seeds": list(CALIBRATION_OPTIMIZER_SEEDS),
            "final_environment_seeds": [42, 88, 99, 123, 219, 220, 221, 226, 229, 236],
            "final_optimizer_seeds": [2000, 2001, 2002, 2003, 2004],
            "selection_rule": "feasibility, median normalized cost, variance, CPU time",
            "calibration_results": str(calibration_csv),
            "calibration_ranking": str(ranking_csv),
            "calibration_results_sha256": _hash_payload(
                pd.read_csv(calibration_csv).to_dict(orient="records")
            ),
        },
        "ga": dict(ga),
        "aco": dict(aco),
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_config.with_suffix(output_config.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(output_config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--configs", type=int, default=24)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("calibration")
    )
    parser.add_argument(
        "--output-config", type=Path,
        default=Path("config/paper_protocol_v2.json"),
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Two configurations, one optimizer seed; does not replace the paper config.",
    )
    parser.add_argument(
        "--families", nargs="+", default=["all"],
        choices=["all", *CALIBRATION_FAMILIES.keys()],
        help="Run all calibration families or one independent shard.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.resume and args.clean_output:
        parser.error("--resume and --clean-output cannot be combined")
    config_count = 2 if args.quick else args.configs
    optimizer_seeds: Sequence[int] = (
        CALIBRATION_OPTIMIZER_SEEDS[:1]
        if args.quick else CALIBRATION_OPTIMIZER_SEEDS
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.quick and args.output_config == Path("config/paper_protocol_v2.json"):
        args.output_config = output_dir / "quick_protocol.json"

    candidates = generate_candidates(config_count)
    selected_families = (
        list(CALIBRATION_FAMILIES)
        if args.families == ["all"]
        else list(dict.fromkeys(args.families))
    )
    calibration_path = output_dir / "calibration_results.csv"
    manifest_path = output_dir / "calibration_manifest.json"
    candidate_manifest = {
        "rng_seed": RNG_SEED,
        "config_count": config_count,
        "optimizer_seeds": list(optimizer_seeds),
        "environment_seeds": list(CALIBRATION_ENVIRONMENT_SEEDS),
        "candidate_hash": _hash_payload(candidates),
    }
    if args.clean_output and calibration_path.exists():
        calibration_path.unlink()
    if args.clean_output and manifest_path.exists():
        manifest_path.unlink()
    if calibration_path.exists() and not args.resume:
        raise RuntimeError(
            f"{calibration_path} already exists. Use --resume to continue it "
            "or --clean-output to replace it."
        )
    if args.resume:
        if not manifest_path.exists():
            raise RuntimeError("Cannot resume without calibration_manifest.json")
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != candidate_manifest:
            raise RuntimeError(
                "Calibration candidates or seeds differ from the checkpoint manifest."
            )
    else:
        manifest_path.write_text(json.dumps(candidate_manifest, indent=2))
    all_rows: List[Dict[str, Any]] = (
        pd.read_csv(calibration_path).to_dict(orient="records")
        if calibration_path.exists() else []
    )
    started = time.perf_counter()
    family_seed_pairs = [
        pair for pair in zip(
        CALIBRATION_FAMILIES.items(),
        CALIBRATION_ENVIRONMENT_SEEDS,
        )
        if pair[0][0] in selected_families
    ]
    for (family_id, family), environment_seed in family_seed_pairs:
        print(f"[calibration] Building {family_id}/seed={environment_seed}")
        env = build_research_environment(
            family_id, family, int(environment_seed)
        )
        worker_env = _worker_environment(env)
        tasks = [
            (algorithm, config_id, params, int(seed))
            for algorithm in ("MGA", "ACO")
            for config_id, params in enumerate(candidates[algorithm])
            for seed in optimizer_seeds
        ]
        completed = {
            (
                str(row["family"]),
                str(row["algo"]),
                int(row["config_id"]),
                int(row["optimizer_seed"]),
            )
            for row in all_rows
        }
        tasks = [
            task for task in tasks
            if (family_id, task[0], int(task[1]), int(task[3])) not in completed
        ]
        random.Random(RNG_SEED + int(environment_seed)).shuffle(tasks)
        if not tasks:
            print(f"[calibration] {family_id} already complete; skipping")
            continue
        worker_count = min(args.workers, len(tasks))
        with Pool(
            processes=worker_count,
            initializer=_init_pair_worker,
            initargs=(worker_env, candidates["MGA"][0], candidates["ACO"][0]),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_calibration_worker, tasks),
                total=len(tasks),
                desc=family_id,
            ):
                all_rows.append(result)
                temporary = calibration_path.with_suffix(".csv.tmp")
                pd.DataFrame(all_rows).to_csv(temporary, index=False)
                temporary.replace(calibration_path)

    selected_rows = [
        row for row in all_rows if str(row["family"]) in selected_families
    ]
    expected_rows = (
        len(selected_families) * 2 * config_count * len(optimizer_seeds)
    )
    if len(selected_rows) != expected_rows:
        raise RuntimeError(
            f"Selected calibration shard has {len(selected_rows)} rows, "
            f"expected {expected_rows}; "
            "resume before freezing a config."
        )
    if set(selected_families) != set(CALIBRATION_FAMILIES):
        shard_summary = {
            "status": "complete",
            "families": selected_families,
            "rows": len(selected_rows),
            "configs_per_algorithm": config_count,
            "optimizer_seeds": list(optimizer_seeds),
            "candidate_hash": _hash_payload(candidates),
        }
        (output_dir / "shard_summary.json").write_text(
            json.dumps(shard_summary, indent=2)
        )
        print(
            f"[calibration] Shard complete: {', '.join(selected_families)}. "
            "Use merge_calibration_shards.py after all shards finish."
        )
        print(f"[calibration] Elapsed: {(time.perf_counter() - started) / 60:.1f} min")
        return 0
    ranking = pd.concat(
        [_rank_configs(pd.DataFrame(all_rows), algorithm) for algorithm in ("MGA", "ACO")],
        ignore_index=True,
    )
    ranking_path = output_dir / "calibration_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    winners = {
        algorithm: json.loads(
            ranking[
                (ranking["algo"] == algorithm) & (ranking["rank"] == 1)
            ]["config_json"].iloc[0]
        )
        for algorithm in ("MGA", "ACO")
    }
    _write_calibrated_config(
        args.output_config,
        winners["MGA"],
        winners["ACO"],
        calibration_path,
        ranking_path,
    )
    print(f"[calibration] Results: {calibration_path}")
    print(f"[calibration] Ranking: {ranking_path}")
    print(f"[calibration] Frozen config: {args.output_config}")
    print(f"[calibration] Elapsed: {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
