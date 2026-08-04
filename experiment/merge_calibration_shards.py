"""Validate and merge C20/C50 calibration shards, then freeze paper parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from calibrate_protocol_v2 import (
    _rank_configs,
    _write_calibrated_config,
    generate_candidates,
)
from research_protocol import (
    CALIBRATION_ENVIRONMENT_SEEDS,
    CALIBRATION_FAMILIES,
    CALIBRATION_OPTIMIZER_SEEDS,
)


KEY_COLUMNS = ["family", "algo", "config_id", "optimizer_seed"]


def merge(
    inputs: List[Path],
    output_dir: Path,
    output_config: Path,
    config_count: int,
) -> None:
    frames = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_shard"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    duplicate_groups = combined[combined.duplicated(KEY_COLUMNS, keep=False)]
    if not duplicate_groups.empty:
        for _, group in duplicate_groups.groupby(KEY_COLUMNS):
            comparable_columns = [
                column for column in (
                    "family", "algo", "config_id", "optimizer_seed",
                    "config_json", "generalized_cost", "dynamic_safe",
                    "best_path", "error",
                )
                if column in group.columns
            ]
            comparable = group[comparable_columns].fillna("").astype(str)
            if len(comparable.drop_duplicates()) != 1:
                raise RuntimeError(
                    "Conflicting duplicate calibration records for "
                    f"{tuple(group.iloc[0][column] for column in KEY_COLUMNS)}"
                )
        # Keep the later input so a dedicated family shard supersedes an
        # accidentally started record from the other droplet.
        combined = combined.drop_duplicates(KEY_COLUMNS, keep="last")

    expected = {
        (family, algorithm, config_id, optimizer_seed)
        for family in CALIBRATION_FAMILIES
        for algorithm in ("MGA", "ACO")
        for config_id in range(config_count)
        for optimizer_seed in CALIBRATION_OPTIMIZER_SEEDS
    }
    actual = {
        (
            str(row.family),
            str(row.algo),
            int(row.config_id),
            int(row.optimizer_seed),
        )
        for row in combined.itertuples()
    }
    if actual != expected:
        raise RuntimeError(
            f"Calibration shard keys are incomplete: missing={len(expected - actual)}, "
            f"unexpected={len(actual - expected)}"
        )
    errors = combined["error"].fillna("").astype(str).str.strip()
    if errors.astype(bool).any():
        print(
            f"[merge] Warning: {int(errors.astype(bool).sum())} failed parameter "
            "runs remain and will rank behind successful configurations."
        )

    candidates = generate_candidates(config_count)
    expected_configs = {
        (algorithm, config_id): json.dumps(
            params, sort_keys=True, separators=(",", ":")
        )
        for algorithm, configs in candidates.items()
        for config_id, params in enumerate(configs)
    }
    for row in combined.itertuples():
        expected_json = expected_configs[(str(row.algo), int(row.config_id))]
        if str(row.config_json) != expected_json:
            raise RuntimeError(
                f"Configuration mismatch for {row.algo}/{row.config_id}"
            )

    combined = combined.drop(columns=["source_shard"]).sort_values(KEY_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "calibration_results.csv"
    combined.to_csv(calibration_path, index=False)
    ranking = pd.concat(
        [_rank_configs(combined, algorithm) for algorithm in ("MGA", "ACO")],
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
        output_config,
        winners["MGA"],
        winners["ACO"],
        calibration_path,
        ranking_path,
    )
    print(f"[merge] Validated {len(combined)} calibration records.")
    print(f"[merge] Frozen config: {output_config}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("config/paper_protocol_v2.json"),
    )
    parser.add_argument("--configs", type=int, default=24)
    args = parser.parse_args()
    merge(args.inputs, args.output_dir, args.output_config, args.configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
