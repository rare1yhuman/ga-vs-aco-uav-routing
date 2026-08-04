"""Strict validation for the seven-family v2 research result set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from research_protocol import (
    CALIBRATION_ENVIRONMENT_SEEDS,
    FINAL_ENVIRONMENT_SEEDS,
    FINAL_OPTIMIZER_SEEDS,
    SCENARIO_FAMILIES,
)
from run_research_experiment import ALGORITHMS, PROTOCOL_VERSION


REQUIRED_COLUMNS = {
    "protocol", "family", "environment_seed", "optimizer_seed", "algo",
    "generalized_cost", "base_cost", "physical_distance_m",
    "exposure_cells", "dynamic_safe", "wall_time_s", "cpu_time_s",
    "candidate_constructions", "best_path", "history_cost",
    "history_wall_s", "history_cpu_s", "history_evaluations", "error",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _failures(results_dir: Path, require_final: bool = False) -> List[str]:
    failures: List[str] = []
    metadata_path = results_dir / "run_metadata.json"
    raw_path = results_dir / "raw_results.csv"
    if not metadata_path.exists():
        return ["missing run_metadata.json"]
    if not raw_path.exists():
        return ["missing raw_results.csv"]
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        failures.append(
            f"metadata status is {metadata.get('status')!r}, expected 'complete'"
        )
    if metadata.get("protocol") != PROTOCOL_VERSION:
        failures.append(f"unexpected protocol {metadata.get('protocol')!r}")
    frame = pd.read_csv(raw_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        failures.append(f"missing columns: {missing_columns}")
        return failures

    family_ids = [str(item) for item in metadata.get("families", [])]
    environment_seeds = [int(item) for item in metadata.get("environment_seeds", [])]
    optimizer_seeds = [int(item) for item in metadata.get("optimizer_seeds", [])]
    if require_final:
        if family_ids != list(SCENARIO_FAMILIES):
            failures.append(f"families are {family_ids}, expected S1-S7")
        if environment_seeds != list(FINAL_ENVIRONMENT_SEEDS):
            failures.append("environment seeds do not match frozen final seeds")
        if optimizer_seeds != list(FINAL_OPTIMIZER_SEEDS):
            failures.append("optimizer seeds do not match frozen final seeds")

    expected = {
        (family, environment_seed, optimizer_seed, algorithm)
        for family in family_ids
        for environment_seed in environment_seeds
        for optimizer_seed in optimizer_seeds
        for algorithm in ALGORITHMS
    }
    actual = {
        (
            str(row.family),
            int(row.environment_seed),
            int(row.optimizer_seed),
            str(row.algo),
        )
        for row in frame.itertuples()
    }
    if expected != actual:
        failures.append(
            f"record-key mismatch: missing={len(expected - actual)}, "
            f"unexpected={len(actual - expected)}"
        )
    if len(frame) != len(expected):
        failures.append(f"row count {len(frame)}, expected {len(expected)}")
    duplicate_count = int(frame.duplicated(
        ["family", "environment_seed", "optimizer_seed", "algo"]
    ).sum())
    if duplicate_count:
        failures.append(f"duplicate paired records: {duplicate_count}")

    errors = frame["error"].fillna("").astype(str).str.strip()
    if errors.astype(bool).any():
        failures.append(f"software error rows: {int(errors.astype(bool).sum())}")
    for column in (
        "generalized_cost", "base_cost", "physical_distance_m",
        "wall_time_s", "cpu_time_s",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            failures.append(f"{column} contains non-finite values")
    if (frame["candidate_constructions"].astype(int) <= 0).any():
        failures.append("non-positive candidate-construction counts")

    paired_budget = frame.pivot_table(
        index=["family", "environment_seed", "optimizer_seed"],
        columns="algo",
        values="candidate_constructions",
        aggfunc="first",
    )
    if set(ALGORITHMS).issubset(paired_budget.columns):
        unequal = paired_budget["MGA"].astype(int) != paired_budget["ACO"].astype(int)
        if unequal.any():
            failures.append(f"unequal paired construction budgets: {int(unequal.sum())}")

    for row in frame.itertuples():
        try:
            chromosome = json.loads(row.best_path)
            history = json.loads(row.history_cost)
            wall = json.loads(row.history_wall_s)
            cpu = json.loads(row.history_cpu_s)
            evaluations = json.loads(row.history_evaluations)
        except (TypeError, json.JSONDecodeError) as exc:
            failures.append(
                f"{row.family}/{row.environment_seed}/{row.optimizer_seed}/"
                f"{row.algo}: invalid JSON ({exc})"
            )
            continue
        if sorted(map(int, chromosome)) != list(range(int(row.n_deliveries))):
            failures.append(
                f"{row.family}/{row.environment_seed}/{row.optimizer_seed}/"
                f"{row.algo}: invalid chromosome"
            )
        if not (len(history) == len(wall) == len(cpu) == len(evaluations)):
            failures.append(
                f"{row.family}/{row.environment_seed}/{row.optimizer_seed}/"
                f"{row.algo}: convergence histories differ in length"
            )
        if any(float(b) < float(a) for a, b in zip(wall, wall[1:])):
            failures.append(
                f"{row.family}/{row.environment_seed}/{row.optimizer_seed}/"
                f"{row.algo}: wall-time history is not monotonic"
            )

    calibration_seeds = set(
        metadata.get("protocol_config", {}).get(
            "calibration_environment_seeds",
            CALIBRATION_ENVIRONMENT_SEEDS,
        )
    )
    overlap = calibration_seeds.intersection(environment_seeds)
    if overlap:
        failures.append(f"calibration/evaluation seed leakage: {sorted(overlap)}")

    for family in family_ids:
        for seed in environment_seeds:
            directory = results_dir / family / f"env_{seed}"
            if not (directory / "raw_results.csv").exists():
                failures.append(f"missing checkpoint CSV: {family}/env_{seed}")
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                failures.append(f"missing checkpoint summary: {family}/env_{seed}")
            elif json.loads(summary_path.read_text()).get("status") != "complete":
                failures.append(f"incomplete checkpoint: {family}/env_{seed}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--require-final", action="store_true",
        help="Require all 700 rows from the frozen paper matrix.",
    )
    args = parser.parse_args()
    failures = _failures(args.results, require_final=args.require_final)
    if failures:
        print("[validate-research] FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    frame = pd.read_csv(args.results / "raw_results.csv")
    print(
        f"[validate-research] OK: {len(frame)} rows, "
        f"{frame['family'].nunique()} families, "
        f"{frame['environment_seed'].nunique()} environment seeds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
