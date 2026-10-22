from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/synthetic/stationary/uniform",
    )
    parser.add_argument(
        "--figure",
        default="figures/synthetic/stationary_multiseed_learning_curve.png",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    curves = []
    code_commits = []

    for seed in args.seeds:
        run_dir = root / f"seed_{seed}"
        summary = json.loads((run_dir / "summary.json").read_text())
        metrics = pd.read_csv(run_dir / "metrics.csv")

        code_commits.append(summary["code_commit"])
        rows.append(
            {
                "seed": seed,
                "code_commit": summary["code_commit"],
                "late_train_return_mean": summary["late_train_return_mean"],
                "late_train_return_std": summary["late_train_return_std"],
                "eval_return_mean": summary["eval_return_mean"],
                "eval_return_std": summary["eval_return_std"],
                "return_improvement": summary["return_improvement"],
                "late_full_length_fraction": summary["late_full_length_fraction"],
                "late_termination_fraction": summary["late_termination_fraction"],
                "late_truncation_fraction": summary["late_truncation_fraction"],
                "sanity_pass": summary["sanity_pass"],
            }
        )

        curves.append(
            metrics[["episode", "return"]]
            .rename(columns={"return": f"seed_{seed}"})
            .set_index("episode")
        )

    if len(set(code_commits)) != 1:
        raise RuntimeError(
            "Formal seed runs do not share one code commit SHA: "
            + ", ".join(sorted(set(code_commits)))
        )

    table = pd.DataFrame(rows)
    table.to_csv(root / "stationary_summary.csv", index=False)

    numeric_cols = [
        "late_train_return_mean",
        "eval_return_mean",
        "return_improvement",
        "late_full_length_fraction",
        "late_termination_fraction",
        "late_truncation_fraction",
    ]

    aggregate = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "code_commit": code_commits[0],
        "all_sanity_pass": bool(table["sanity_pass"].all()),
    }

    for col in numeric_cols:
        aggregate[f"{col}_mean"] = float(table[col].mean())
        aggregate[f"{col}_sd_across_seeds"] = float(
            table[col].std(ddof=1)
        )

    (root / "stationary_aggregate.json").write_text(
        json.dumps(aggregate, indent=2)
    )

    curve = pd.concat(curves, axis=1)
    mean_curve = curve.mean(axis=1)
    sd_curve = curve.std(axis=1, ddof=1)

    smooth_window = 20
    smooth_mean = mean_curve.rolling(
        smooth_window, min_periods=1
    ).mean()
    smooth_sd = sd_curve.rolling(
        smooth_window, min_periods=1
    ).mean()

    fig_path = Path(args.figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.8))
    plt.plot(
        smooth_mean.index,
        smooth_mean.values,
        linewidth=2,
        label=f"Mean return ({smooth_window}-episode moving average)",
    )
    plt.fill_between(
        smooth_mean.index,
        smooth_mean.values - smooth_sd.values,
        smooth_mean.values + smooth_sd.values,
        alpha=0.2,
        label="±1 SD across seeds",
    )
    plt.xlabel("Episode")
    plt.ylabel("Return (higher / less negative is better)")
    plt.title("Stationary DDQN + Uniform Replay, 5 Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print("\\n=== Per-seed stationary summary ===")
    print(table.to_string(index=False))
    print("\\n=== Across-seed aggregate ===")
    print(json.dumps(aggregate, indent=2))
    print("\\nSaved:")
    print(" ", root / "stationary_summary.csv")
    print(" ", root / "stationary_aggregate.json")
    print(" ", fig_path)


if __name__ == "__main__":
    main()
