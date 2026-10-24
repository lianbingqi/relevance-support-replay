from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium
import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_oracle_recovery_3seed_v2 import run_seed
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def runtime_manifest():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "code_commit": git_sha(),
    }


def normalize(result):
    seed = int(result["seed"])

    metrics = result["metrics"].copy()
    metrics["seed"] = seed
    metrics = metrics.sort_values(
        ["seed", "method"]
    ).reset_index(drop=True)

    curve = result["curve"].copy()
    curve["seed"] = seed
    curve = curve.sort_values(
        ["seed", "method", "phase_step"]
    ).reset_index(drop=True)

    a1 = result["a1"].copy()
    a1["seed"] = seed
    a1 = a1.sort_values(
        ["seed", "phase_step"]
    ).reset_index(drop=True)

    diagnostics = result["diagnostics"].copy()
    diagnostics["seed"] = seed
    diagnostics = diagnostics.sort_values(
        ["seed", "method"]
    ).reset_index(drop=True)

    return {
        "seed": seed,
        "metrics": metrics,
        "curve": curve,
        "a1": a1,
        "diagnostics": diagnostics,
        "metadata": dict(result["metadata"]),
    }


def execute_sequential(seeds, cfg):
    return [
        normalize(run_seed(int(seed), cfg))
        for seed in seeds
    ]


def execute_parallel(seeds, cfg, max_workers):
    results = []

    with ProcessPoolExecutor(
        max_workers=int(max_workers)
    ) as executor:
        futures = {
            executor.submit(
                run_seed,
                int(seed),
                cfg,
            ): int(seed)
            for seed in seeds
        }

        for future in as_completed(futures):
            expected = futures[future]
            result = normalize(
                future.result()
            )

            if int(result["seed"]) != expected:
                raise RuntimeError(
                    f"Seed mismatch: expected {expected}, got {result['seed']}"
                )

            results.append(result)

    results.sort(
        key=lambda x: x["seed"]
    )
    return results


def save_results(results, out):
    metrics = pd.concat(
        [x["metrics"] for x in results],
        ignore_index=True,
    ).sort_values(
        ["seed", "method"]
    ).reset_index(drop=True)

    curve = pd.concat(
        [x["curve"] for x in results],
        ignore_index=True,
    ).sort_values(
        ["seed", "method", "phase_step"]
    ).reset_index(drop=True)

    a1 = pd.concat(
        [x["a1"] for x in results],
        ignore_index=True,
    ).sort_values(
        ["seed", "phase_step"]
    ).reset_index(drop=True)

    diagnostics = pd.concat(
        [x["diagnostics"] for x in results],
        ignore_index=True,
    ).sort_values(
        ["seed", "method"]
    ).reset_index(drop=True)

    metrics.to_csv(out / "metrics.csv", index=False)
    curve.to_csv(out / "A2_recovery_curve.csv", index=False)
    a1.to_csv(out / "A1_reference_evaluations.csv", index=False)
    diagnostics.to_csv(out / "sampler_diagnostics.csv", index=False)

    save_json(
        {"per_seed": [x["metadata"] for x in results]},
        out / "run_metadata.json",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic_v2/execution_reconciliation.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        required=True,
    )
    parser.add_argument(
        "--label",
        required=True,
    )
    args = parser.parse_args()

    rec = load_yaml(
        ROOT / args.config
    )["reconciliation"]

    seeds = [
        int(x)
        for x in rec["seeds"]
    ]

    oracle_cfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / rec["oracle_config"],
    )

    out = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "execution_reconciliation"
        / args.label
    )

    save_json(
        {
            **runtime_manifest(),
            "mode": args.mode,
            "label": args.label,
            "seeds": seeds,
            "oracle_config": rec["oracle_config"],
        },
        out / "runtime_manifest.json",
    )

    if args.mode == "sequential":
        results = execute_sequential(
            seeds,
            oracle_cfg,
        )
    else:
        results = execute_parallel(
            seeds,
            oracle_cfg,
            rec["max_workers"],
        )

    save_results(
        results,
        out,
    )

    print(pd.read_csv(out / "metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
