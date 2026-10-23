from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.run_oracle_recovery_isolation_v2 import (
    auc_0_2000,
    evaluate,
    first_sustained_recovery,
    make_oracle_from_recent,
    random_A,
    train_A2_branch,
    train_common_phase,
)
from scripts.train_stationary_v2 import make_agent
from src.replay.recent import RecentReplayBuffer
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run_seed(seed: int, cfg: dict):
    torch.set_num_threads(1)
    set_global_seed(seed)

    iso = cfg["isolation"]

    agent = make_agent(cfg, seed)
    replay = RecentReplayBuffer(
        capacity=int(iso["replay_capacity"]),
        recent_window=int(iso["recent_window"]),
        seed=seed,
    )

    global_step = 0

    global_step, a1_rows = train_common_phase(
        agent,
        replay,
        "A",
        int(iso["A1_steps"]),
        global_step,
        seed,
        0,
        cfg,
        eval_steps=[int(x) for x in iso["A1_reference_steps"]],
    )

    a1_df = pd.DataFrame(a1_rows)
    a1_reference = float(a1_df["eval_return_mean"].mean())

    global_step, _ = train_common_phase(
        agent,
        replay,
        "B",
        int(iso["B_steps"]),
        global_step,
        seed,
        1,
        cfg,
    )

    B_final = float(evaluate(agent, "B", seed, cfg)["return_mean"])
    rand_A = float(random_A(seed, cfg))

    recent_agent = copy.deepcopy(agent)
    oracle_agent = copy.deepcopy(agent)

    recent_replay = copy.deepcopy(replay)
    oracle_replay = make_oracle_from_recent(replay, seed, cfg)

    recent_zero = float(evaluate(recent_agent, "A", seed, cfg)["return_mean"])
    oracle_zero = float(evaluate(oracle_agent, "A", seed, cfg)["return_mean"])

    if not np.isclose(recent_zero, oracle_zero, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"seed={seed}: forked agents are not identical")

    recent_rows, recent_diag = train_A2_branch(
        "recent", recent_agent, recent_replay, global_step, seed, cfg
    )

    oracle_rows, oracle_diag = train_A2_branch(
        "oracle", oracle_agent, oracle_replay, global_step, seed, cfg
    )

    curve = pd.DataFrame(recent_rows + oracle_rows)

    denom = a1_reference - rand_A
    if denom <= 0:
        raise RuntimeError(f"seed={seed}: A1 reference does not beat random A")

    curve["normalized_A_score"] = (
        curve["eval_return_mean"] - rand_A
    ) / denom

    threshold = float(iso["recovery_threshold"])
    consecutive = int(iso["recovery_consecutive_checkpoints"])

    metric_rows = []

    for method in ["recent", "oracle"]:
        d = (
            curve[curve["method"] == method]
            .sort_values("phase_step")
            .copy()
        )

        recovery = first_sustained_recovery(d, threshold, consecutive)

        post = d[
            (d["phase_step"] > 0)
            & (d["phase_step"] <= 2000)
        ]

        metric_rows.append({
            "seed": seed,
            "method": method,
            "zero_shot_score": float(d.iloc[0]["normalized_A_score"]),
            "recovery_step": (
                recovery if recovery is not None else np.nan
            ),
            "AUC_0_2000": float(auc_0_2000(d)),
            "post_switch_mean_score_0_2000": float(
                post["normalized_A_score"].mean()
            ),
            "final_score": float(d.iloc[-1]["normalized_A_score"]),
        })

    metrics = pd.DataFrame(metric_rows)

    diagnostics = pd.DataFrame([
        {"seed": seed, **recent_diag},
        {"seed": seed, **oracle_diag},
    ])

    a1_df["seed"] = seed
    curve["seed"] = seed

    return {
        "seed": seed,
        "a1": a1_df,
        "curve": curve,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "metadata": {
            "seed": seed,
            "A1_reference_return": a1_reference,
            "B_final_return": B_final,
            "random_A": rand_A,
            "fork_zero_shot_return": recent_zero,
        },
    }


def load_seed1():
    base = (
        ROOT
        / "results"
        / "synthetic_v2"
        / "oracle_recovery_isolation"
    )

    metrics = pd.read_csv(base / "isolation_metrics.csv")
    metrics.insert(0, "seed", 1)

    curve = pd.read_csv(base / "A2_recovery_curve.csv")
    curve["seed"] = 1

    diagnostics = pd.read_csv(base / "sampler_diagnostics.csv")
    diagnostics.insert(0, "seed", 1)

    return metrics, curve, diagnostics


def make_paired(metrics):
    rows = []

    for seed in sorted(metrics["seed"].unique()):
        sub = metrics[metrics["seed"] == seed].set_index("method")

        recent_recovery = float(sub.loc["recent", "recovery_step"])
        oracle_recovery = float(sub.loc["oracle", "recovery_step"])

        rows.append({
            "seed": int(seed),
            "recent_recovery_step": recent_recovery,
            "oracle_recovery_step": oracle_recovery,
            "recovery_improvement_steps": (
                recent_recovery - oracle_recovery
            ),
            "recent_AUC_0_2000": float(
                sub.loc["recent", "AUC_0_2000"]
            ),
            "oracle_AUC_0_2000": float(
                sub.loc["oracle", "AUC_0_2000"]
            ),
            "AUC_0_2000_advantage": float(
                sub.loc["oracle", "AUC_0_2000"]
                - sub.loc["recent", "AUC_0_2000"]
            ),
        })

    return pd.DataFrame(rows)


def plot_curve(curve, path, max_step):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for method in ["recent", "oracle"]:
        subset = curve[
            (curve["method"] == method)
            & (curve["phase_step"] <= max_step)
        ]

        grouped = (
            subset
            .groupby("phase_step")["normalized_A_score"]
            .agg(["mean", "std"])
            .reset_index()
        )

        line = ax.plot(
            grouped["phase_step"],
            grouped["mean"],
            marker="o",
            label=method,
        )[0]

        sd = grouped["std"].fillna(0.0)

        ax.fill_between(
            grouped["phase_step"],
            grouped["mean"] - sd,
            grouped["mean"] + sd,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.axhline(
        0.90,
        linestyle="--",
        label="recovery threshold",
    )

    ax.set_xlabel("A2 training transitions")
    ax.set_ylabel("Normalized A performance")
    ax.set_title("Oracle Recovery Replication (3 Seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic_v2/oracle_recovery_3seed.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs/base.yaml", ROOT / args.config)
    iso = cfg["isolation"]

    seeds = [int(x) for x in iso["seeds"]]

    out = ensure_dir(
        ROOT / "results/synthetic_v2/oracle_recovery_3seed"
    )
    figs = ensure_dir(
        ROOT / "figures/synthetic_v2/oracle_recovery_3seed"
    )

    jobs = []

    with ProcessPoolExecutor(
        max_workers=min(int(iso["max_workers"]), len(seeds))
    ) as executor:
        futures = {
            executor.submit(run_seed, seed, cfg): seed
            for seed in seeds
        }

        for k, future in enumerate(as_completed(futures), start=1):
            seed = futures[future]
            result = future.result()
            jobs.append(result)
            print(f"[{k}/{len(seeds)}] finished seed={seed}")

    new_metrics = pd.concat(
        [x["metrics"] for x in jobs],
        ignore_index=True,
    )
    new_curve = pd.concat(
        [x["curve"] for x in jobs],
        ignore_index=True,
    )
    new_diag = pd.concat(
        [x["diagnostics"] for x in jobs],
        ignore_index=True,
    )
    new_a1 = pd.concat(
        [x["a1"] for x in jobs],
        ignore_index=True,
    )

    new_metrics.to_csv(
        out / "heldout_metrics_seeds23.csv",
        index=False,
    )
    new_curve.to_csv(
        out / "heldout_A2_curves_seeds23.csv",
        index=False,
    )
    new_diag.to_csv(
        out / "heldout_sampler_diagnostics_seeds23.csv",
        index=False,
    )
    new_a1.to_csv(
        out / "heldout_A1_reference_seeds23.csv",
        index=False,
    )

    seed1_metrics, seed1_curve, seed1_diag = load_seed1()

    all_metrics = pd.concat(
        [seed1_metrics, new_metrics],
        ignore_index=True,
    )
    all_curve = pd.concat(
        [seed1_curve, new_curve],
        ignore_index=True,
    )
    all_diag = pd.concat(
        [seed1_diag, new_diag],
        ignore_index=True,
    )

    paired = make_paired(all_metrics)

    summary = pd.DataFrame([{
        "n_seeds": int(len(paired)),
        "mean_recovery_improvement_steps": float(
            paired["recovery_improvement_steps"].mean()
        ),
        "sd_recovery_improvement_steps": float(
            paired["recovery_improvement_steps"].std(ddof=1)
        ),
        "positive_recovery_seeds": int(
            (paired["recovery_improvement_steps"] > 0).sum()
        ),
        "mean_AUC_0_2000_advantage": float(
            paired["AUC_0_2000_advantage"].mean()
        ),
        "sd_AUC_0_2000_advantage": float(
            paired["AUC_0_2000_advantage"].std(ddof=1)
        ),
        "positive_AUC_seeds": int(
            (paired["AUC_0_2000_advantage"] > 0).sum()
        ),
    }])

    all_metrics.to_csv(
        out / "metrics_all_3seeds.csv",
        index=False,
    )
    all_curve.to_csv(
        out / "A2_curves_all_3seeds.csv",
        index=False,
    )
    all_diag.to_csv(
        out / "sampler_diagnostics_all_3seeds.csv",
        index=False,
    )
    paired.to_csv(
        out / "paired_advantages_all_3seeds.csv",
        index=False,
    )
    summary.to_csv(
        out / "summary_all_3seeds.csv",
        index=False,
    )

    plot_curve(
        all_curve,
        figs / "A2_recovery_first_2000_3seeds.png",
        max_step=2000,
    )
    plot_curve(
        all_curve,
        figs / "A2_recovery_full_3seeds.png",
        max_step=int(iso["A2_steps"]),
    )

    heldout = paired[paired["seed"].isin(seeds)]

    save_json(
        {
            "code_commit": git_sha(),
            "pilot_seed": 1,
            "heldout_replication_seeds": seeds,
            "protocol_frozen_from_seed1": True,
            "heldout_positive_recovery_seeds": int(
                (heldout["recovery_improvement_steps"] > 0).sum()
            ),
            "heldout_positive_AUC_seeds": int(
                (heldout["AUC_0_2000_advantage"] > 0).sum()
            ),
            "all_3seed_mean_recovery_improvement_steps": float(
                paired["recovery_improvement_steps"].mean()
            ),
            "all_3seed_mean_AUC_0_2000_advantage": float(
                paired["AUC_0_2000_advantage"].mean()
            ),
        },
        out / "replication_metadata.json",
    )

    save_yaml(
        cfg,
        out / "resolved_config.yaml",
    )

    print("\n=== Held-out seeds 2 and 3 ===")
    print(heldout.to_string(index=False))

    print("\n=== All three seeds ===")
    print(paired.to_string(index=False))

    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
