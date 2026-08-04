"""Verify exact non-timing equivalence between two final research CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


KEY_COLUMNS = ["family", "environment_seed", "optimizer_seed", "algo"]
TIMING_COLUMNS = {
    "wall_time_s",
    "cpu_time_s",
    "native_wall_time_s",
    "native_cpu_time_s",
    "history_wall_s",
    "history_cpu_s",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(reference: Path, candidate: Path) -> Dict:
    left = pd.read_csv(reference).sort_values(KEY_COLUMNS).reset_index(drop=True)
    right = pd.read_csv(candidate).sort_values(KEY_COLUMNS).reset_index(drop=True)
    if list(left.columns) != list(right.columns):
        raise ValueError("CSV schemas differ")
    if len(left) != len(right):
        raise ValueError(
            f"Row counts differ: reference={len(left)}, candidate={len(right)}"
        )
    differing: Dict[str, int] = {}
    exact_columns: List[str] = []
    for column in left.columns:
        left_values = left[column].fillna("<NA>").astype(str)
        right_values = right[column].fillna("<NA>").astype(str)
        count = int((left_values != right_values).sum())
        if column not in TIMING_COLUMNS:
            if count:
                differing[column] = count
            else:
                exact_columns.append(column)
    return {
        "status": "pass" if not differing else "fail",
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_sha256": _sha256(reference),
        "candidate_sha256": _sha256(candidate),
        "rows": len(left),
        "key_columns": KEY_COLUMNS,
        "excluded_timing_columns": sorted(TIMING_COLUMNS),
        "exact_non_timing_columns": exact_columns,
        "differing_non_timing_columns": differing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
