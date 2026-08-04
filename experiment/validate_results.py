"""
Validate a completed GA vs ACO experiment result directory.

This checks run completeness, duplicate records, metadata, and software errors.
Route infeasibility is not treated as a software error; it is a measured outcome.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


EXPECTED_TRIALS = list("ABCDEFGHIJ")
EXPECTED_ALGOS = ("MGA", "ACO")


def _failures(results_dir: Path, expected_runs: int, require_all: bool = False) -> List[str]:
    failures: List[str] = []
    metadata = results_dir / "run_metadata.json"
    master = results_dir / "master_results.csv"
    expected_trials = EXPECTED_TRIALS

    if not metadata.exists():
        failures.append("missing run_metadata.json")
    else:
        with open(metadata) as f:
            meta = json.load(f)
        if meta.get("status") != "complete":
            failures.append(f"metadata status is {meta.get('status')!r}, expected 'complete'")
        if not require_all:
            expected_trials = [str(t) for t in meta.get("trial_ids", expected_trials)]
        elif sorted(meta.get("trial_ids", [])) != EXPECTED_TRIALS:
            failures.append(f"metadata trial_ids is {meta.get('trial_ids')}, expected {EXPECTED_TRIALS}")

    if not master.exists():
        failures.append("missing master_results.csv")
    else:
        df_master = pd.read_csv(master)
        if len(df_master) != len(expected_trials):
            failures.append(f"master_results.csv has {len(df_master)} rows, expected {len(expected_trials)}")
        missing_trials = sorted(set(expected_trials) - set(df_master.get("trial", [])))
        if missing_trials:
            failures.append(f"master_results.csv missing trials: {missing_trials}")

    all_records = []
    for tid in expected_trials:
        trial_dir = results_dir / f"trial_{tid}"
        raw = trial_dir / "raw_results.csv"
        summary = trial_dir / "summary.json"
        if not raw.exists():
            failures.append(f"trial {tid}: missing raw_results.csv")
            continue
        if not summary.exists():
            failures.append(f"trial {tid}: missing summary.json")
        else:
            with open(summary) as f:
                payload = json.load(f)
            if payload.get("status") != "complete":
                failures.append(f"trial {tid}: summary status is {payload.get('status')!r}")

        df = pd.read_csv(raw)
        all_records.append(df)
        for algo in EXPECTED_ALGOS:
            sub = df[df["algo"] == algo]
            if len(sub) != expected_runs:
                failures.append(f"trial {tid} {algo}: {len(sub)} rows, expected {expected_runs}")
            if sub["run"].nunique() != expected_runs:
                failures.append(f"trial {tid} {algo}: duplicate or missing run ids")
            finite = np.isfinite(sub["distance_m"].replace([np.inf, -np.inf], np.nan).astype(float))
            if int(finite.sum()) != expected_runs:
                failures.append(f"trial {tid} {algo}: non-finite distance rows present")
            errors = sub["error"].fillna("").astype(str)
            software_errors = [e for e in errors if e and e.lower() != "none" and e.lower() != "nan"]
            if software_errors:
                failures.append(f"trial {tid} {algo}: {len(software_errors)} software error rows")

    if all_records:
        full = pd.concat(all_records, ignore_index=True)
        expected_total = len(expected_trials) * len(EXPECTED_ALGOS) * expected_runs
        if len(full) != expected_total:
            failures.append(f"total raw rows {len(full)}, expected {expected_total}")
        dup = full.duplicated(subset=["trial", "algo", "run"]).sum()
        if int(dup):
            failures.append(f"duplicate trial/algo/run rows: {int(dup)}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--require-all", action="store_true",
                        help="Require the full A-J paper trial matrix.")
    args = parser.parse_args()

    results_dir = Path(args.results)
    failures = _failures(results_dir, args.runs, require_all=args.require_all)
    if failures:
        print("[validate_results] FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1
    label = "complete full paper run" if args.require_all else "complete selected-trial run"
    print(f"[validate_results] OK: {results_dir.resolve()} is a {label}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
