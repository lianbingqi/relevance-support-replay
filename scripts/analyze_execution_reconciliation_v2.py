from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import save_json


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_sorted(path, cols):
    return (
        pd.read_csv(path)
        .sort_values(cols)
        .reset_index(drop=True)
    )


def compare_numeric(a, b, keys, name_a, name_b):
    joined = a.merge(
        b,
        on=keys,
        suffixes=("_a", "_b"),
        how="outer",
        indicator=True,
    )

    if not bool(
        (joined["_merge"] == "both").all()
    ):
        raise RuntimeError(
            f"Key mismatch: {name_a} vs {name_b}"
        )

    rows = []

    for col in a.columns:
        if col in keys:
            continue

        if (
            col in b.columns
            and pd.api.types.is_numeric_dtype(a[col])
            and pd.api.types.is_numeric_dtype(b[col])
        ):
            x = joined[f"{col}_a"].to_numpy(dtype=float)
            y = joined[f"{col}_b"].to_numpy(dtype=float)

            diff = np.abs(x - y)
            finite = np.isfinite(diff)

            rows.append({
                "comparison": f"{name_a}__vs__{name_b}",
                "column": col,
                "exact_equal": bool(
                    np.array_equal(
                        x,
                        y,
                        equal_nan=True,
                    )
                ),
                "max_abs_difference": (
                    float(np.max(diff[finite]))
                    if finite.any()
                    else 0.0
                ),
            })

    return pd.DataFrame(rows)


def metric_drift(reference, candidate, ref_name, cand_name):
    joined = reference.merge(
        candidate,
        on=["seed", "method"],
        suffixes=("_ref", "_cand"),
        how="inner",
    )

    rows = []

    for _, r in joined.iterrows():
        for metric in [
            "zero_shot_score",
            "recovery_step",
            "AUC_0_2000",
            "post_switch_mean_score_0_2000",
            "final_score",
        ]:
            ref = float(r[f"{metric}_ref"])
            cur = float(r[f"{metric}_cand"])

            rows.append({
                "reference": ref_name,
                "candidate": cand_name,
                "seed": int(r["seed"]),
                "method": r["method"],
                "metric": metric,
                "reference_value": ref,
                "candidate_value": cur,
                "absolute_difference": cur - ref,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic_v2/execution_reconciliation.yaml",
    )
    args = parser.parse_args()

    rec = load_yaml(
        ROOT / args.config
    )["reconciliation"]

    root = (
        ROOT
        / "results"
        / "synthetic_v2"
        / "execution_reconciliation"
    )

    labels = [
        "sequential_a",
        "sequential_b",
        "parallel_a",
        "parallel_b",
    ]

    metrics = {
        x: read_sorted(
            root / x / "metrics.csv",
            ["seed", "method"],
        )
        for x in labels
    }

    curves = {
        x: read_sorted(
            root / x / "A2_recovery_curve.csv",
            ["seed", "method", "phase_step"],
        )
        for x in labels
    }

    comparisons = []

    for a, b in [
        ("sequential_a", "sequential_b"),
        ("parallel_a", "parallel_b"),
        ("sequential_a", "parallel_a"),
    ]:
        comparisons.append(
            compare_numeric(
                metrics[a],
                metrics[b],
                ["seed", "method"],
                a,
                b,
            )
        )

        comparisons.append(
            compare_numeric(
                curves[a],
                curves[b],
                ["seed", "method", "phase_step"],
                a + "_curve",
                b + "_curve",
            )
        )

    exact = pd.concat(
        comparisons,
        ignore_index=True,
    )

    exact.to_csv(
        root / "within_runtime_exact_comparisons.csv",
        index=False,
    )

    historical = read_sorted(
        ROOT / rec["historical_oracle_results"],
        ["seed", "method"],
    )

    current07 = read_sorted(
        ROOT / rec["current_07_results"],
        ["seed", "method"],
    )

    current07 = current07[
        current07["method"].isin(
            ["recent", "oracle"]
        )
    ].copy()

    drift = []

    for label in labels:
        drift.append(
            metric_drift(
                historical,
                metrics[label],
                "historical_oracle_parallel",
                label,
            )
        )
        drift.append(
            metric_drift(
                current07,
                metrics[label],
                "current_07_controls",
                label,
            )
        )

    drift = pd.concat(
        drift,
        ignore_index=True,
    )

    drift.to_csv(
        root / "reference_metric_drift.csv",
        index=False,
    )

    atol = float(
        rec["comparison_atol"]
    )

    def is_close(metric_name, curve_name):
        sub = exact[
            exact["comparison"].isin(
                [metric_name, curve_name]
            )
        ]
        return bool(
            (
                sub["max_abs_difference"]
                <= atol
            ).all()
        )

    sequential_reproducible = is_close(
        "sequential_a__vs__sequential_b",
        "sequential_a_curve__vs__sequential_b_curve",
    )

    parallel_reproducible = is_close(
        "parallel_a__vs__parallel_b",
        "parallel_a_curve__vs__parallel_b_curve",
    )

    sequential_parallel_same = is_close(
        "sequential_a__vs__parallel_a",
        "sequential_a_curve__vs__parallel_a_curve",
    )

    if not sequential_reproducible:
        diagnosis = (
            "CURRENT_SEQUENTIAL_NOT_REPRODUCIBLE"
        )
        recommendation = (
            "LOCK_DETERMINISM_AND_RERUN_FORMAL_RQ2"
        )

    elif parallel_reproducible and not sequential_parallel_same:
        diagnosis = (
            "EXECUTION_PATH_EFFECT"
        )
        recommendation = (
            "USE_SEQUENTIAL_AS_CANONICAL_FOR_FUTURE_FORMAL_RUNS"
        )

    elif sequential_parallel_same:
        diagnosis = (
            "NO_CURRENT_EXECUTION_PATH_EFFECT"
        )
        recommendation = (
            "USE_CURRENT_SEQUENTIAL_RUNTIME_AS_CANONICAL_AND_RECORD_VERSION_MANIFEST"
        )

    else:
        diagnosis = (
            "PARALLEL_NOT_REPRODUCIBLE_OR_MIXED"
        )
        recommendation = (
            "USE_SEQUENTIAL_AS_CANONICAL_FOR_FUTURE_FORMAL_RUNS"
        )

    summary = {
        "comparison_atol": atol,
        "sequential_reproducible": sequential_reproducible,
        "parallel_reproducible": parallel_reproducible,
        "sequential_parallel_same": sequential_parallel_same,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "rq2_results_modified": False,
    }

    save_json(
        summary,
        root / "reconciliation_summary.json",
    )

    print("\n=== Exact/tolerance comparisons ===")
    print(exact.to_string(index=False))

    print("\n=== Reference drift ===")
    print(drift.to_string(index=False))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
