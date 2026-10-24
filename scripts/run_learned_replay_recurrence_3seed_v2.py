from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copytree_exact(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_replication_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_heldout_seed(
    seed,
    pilot_cfg_path,
    pilot_script_path,
    pilot_result_dir,
    pilot_figure_dir,
    formal_result_root,
    formal_figure_root,
):
    with open(pilot_cfg_path, "r", encoding="utf-8") as f:
        seed_cfg = yaml.safe_load(f)

    original_seed = int(seed_cfg["isolation"]["seed"])
    if original_seed != 1:
        raise RuntimeError(
            f"Expected frozen development seed 1, found {original_seed}"
        )

    # Only allowed held-out change.
    seed_cfg["isolation"]["seed"] = int(seed)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_seed{seed}.yaml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(
            seed_cfg,
            tmp,
            sort_keys=False,
        )
        tmp_path = Path(tmp.name)

    try:
        if pilot_result_dir.exists():
            shutil.rmtree(pilot_result_dir)
        if pilot_figure_dir.exists():
            shutil.rmtree(pilot_figure_dir)

        proc = subprocess.run(
            [
                sys.executable,
                str(pilot_script_path),
                "--config",
                str(tmp_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"seed={seed} pilot subprocess failed "
                f"with exit code {proc.returncode}"
            )

        seed_result_dir = formal_result_root / f"seed_{seed}"
        seed_figure_dir = formal_figure_root / f"seed_{seed}"

        copytree_exact(
            pilot_result_dir,
            seed_result_dir,
        )
        copytree_exact(
            pilot_figure_dir,
            seed_figure_dir,
        )

        metrics = pd.read_csv(
            seed_result_dir / "metrics.csv"
        )
        metrics.insert(0, "seed", int(seed))

        curve = pd.read_csv(
            seed_result_dir / "A2_recovery_curve.csv"
        )
        curve["seed"] = int(seed)

        diag = pd.read_csv(
            seed_result_dir / "sampler_diagnostics.csv"
        )
        diag.insert(0, "seed", int(seed))

        bank = pd.read_csv(
            seed_result_dir / "historical_bank_summary.csv"
        )
        bank.insert(0, "seed", int(seed))

        encoder = pd.read_csv(
            seed_result_dir / "encoder_offline_summary.csv"
        )
        encoder.insert(0, "seed", int(seed))

        meta = json.loads(
            (seed_result_dir / "pilot_metadata.json").read_text(
                encoding="utf-8"
            )
        )

        if int(meta["seed"]) != int(seed):
            raise RuntimeError(
                f"seed metadata mismatch for held-out seed {seed}"
            )

        if bool(meta["learned_selector_uses_true_context_labels"]):
            raise RuntimeError(
                f"seed={seed}: learned selector used true labels"
            )

        return {
            "metrics": metrics,
            "curve": curve,
            "diag": diag,
            "bank": bank,
            "encoder": encoder,
            "meta": meta,
        }

    finally:
        tmp_path.unlink(missing_ok=True)


def load_seed1(pilot_result_dir):
    metrics = pd.read_csv(
        pilot_result_dir / "metrics.csv"
    )
    metrics.insert(0, "seed", 1)

    curve = pd.read_csv(
        pilot_result_dir / "A2_recovery_curve.csv"
    )
    curve["seed"] = 1

    diag = pd.read_csv(
        pilot_result_dir / "sampler_diagnostics.csv"
    )
    diag.insert(0, "seed", 1)

    bank = pd.read_csv(
        pilot_result_dir / "historical_bank_summary.csv"
    )
    bank.insert(0, "seed", 1)

    encoder = pd.read_csv(
        pilot_result_dir / "encoder_offline_summary.csv"
    )
    encoder.insert(0, "seed", 1)

    return {
        "metrics": metrics,
        "curve": curve,
        "diag": diag,
        "bank": bank,
        "encoder": encoder,
    }


def paired_table(metrics):
    rows = []

    for seed in sorted(metrics["seed"].unique()):
        sub = (
            metrics[
                metrics["seed"] == seed
            ]
            .set_index("method")
        )

        rr = float(sub.loc["recent", "recovery_step"])
        lr = float(sub.loc["learned", "recovery_step"])
        orr = float(sub.loc["oracle", "recovery_step"])

        ra = float(sub.loc["recent", "AUC_0_2000"])
        la = float(sub.loc["learned", "AUC_0_2000"])
        oa = float(sub.loc["oracle", "AUC_0_2000"])

        rows.append({
            "seed": int(seed),
            "recent_recovery_step": rr,
            "learned_recovery_step": lr,
            "oracle_recovery_step": orr,
            "learned_recovery_improvement_vs_recent": rr - lr,
            "oracle_recovery_improvement_vs_recent": rr - orr,
            "learned_recovery_gap_to_oracle": lr - orr,
            "recent_AUC_0_2000": ra,
            "learned_AUC_0_2000": la,
            "oracle_AUC_0_2000": oa,
            "learned_AUC_advantage_vs_recent": la - ra,
            "oracle_AUC_advantage_vs_recent": oa - ra,
            "learned_minus_oracle_AUC": la - oa,
        })

    return pd.DataFrame(rows)


def control_drift(metrics):
    prior = pd.read_csv(
        ROOT
        / "results"
        / "synthetic_v2"
        / "oracle_recovery_3seed"
        / "metrics_all_3seeds.csv"
    )

    rows = []

    for seed in [2, 3]:
        cur = (
            metrics[
                metrics["seed"] == seed
            ]
            .set_index("method")
        )

        old = (
            prior[
                prior["seed"] == seed
            ]
            .set_index("method")
        )

        for method in ["recent", "oracle"]:
            for metric in [
                "zero_shot_score",
                "recovery_step",
                "AUC_0_2000",
                "post_switch_mean_score_0_2000",
                "final_score",
            ]:
                now = float(cur.loc[method, metric])
                hist = float(old.loc[method, metric])

                rows.append({
                    "seed": seed,
                    "method": method,
                    "metric": metric,
                    "historical_value": hist,
                    "current_value": now,
                    "absolute_difference": now - hist,
                })

    return pd.DataFrame(rows)


def plot_three_seed(curve, path, max_step):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for method in ["recent", "learned", "oracle"]:
        d = curve[
            (curve["method"] == method)
            & (curve["phase_step"] <= max_step)
        ]

        g = (
            d.groupby("phase_step")["normalized_A_score"]
            .agg(["mean", "std"])
            .reset_index()
        )

        line = ax.plot(
            g["phase_step"],
            g["mean"],
            marker="o",
            label=method,
        )[0]

        sd = g["std"].fillna(0.0)

        ax.fill_between(
            g["phase_step"],
            g["mean"] - sd,
            g["mean"] + sd,
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
    ax.set_title(
        "Learned Relevance Recurrence Replication (3 Seeds)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "learned_replay_recurrence_3seed.yaml"
        ),
    )
    args = parser.parse_args()

    rep_cfg = load_replication_config(
        ROOT / args.config
    )["replication"]

    heldout_seeds = [
        int(x)
        for x in rep_cfg["heldout_seeds"]
    ]

    if heldout_seeds != [2, 3]:
        raise RuntimeError(
            f"Expected held-out seeds [2, 3], got {heldout_seeds}"
        )

    pilot_cfg_path = ROOT / rep_cfg["pilot_config"]
    pilot_script_path = ROOT / rep_cfg["pilot_script"]

    pilot_result_dir = (
        ROOT
        / "results"
        / "synthetic_v2"
        / "learned_replay_recurrence_pilot"
    )

    pilot_figure_dir = (
        ROOT
        / "figures"
        / "synthetic_v2"
        / "learned_replay_recurrence_pilot"
    )

    formal_result_root = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "learned_replay_recurrence_3seed"
    )

    formal_figure_root = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "learned_replay_recurrence_3seed"
    )

    pilot_cfg_sha = sha256_file(pilot_cfg_path)
    pilot_script_sha = sha256_file(pilot_script_path)

    with tempfile.TemporaryDirectory() as td:
        backup = Path(td)
        backup_results = backup / "seed1_results"
        backup_figures = backup / "seed1_figures"

        copytree_exact(
            pilot_result_dir,
            backup_results,
        )
        copytree_exact(
            pilot_figure_dir,
            backup_figures,
        )

        heldout = []

        try:
            for seed in heldout_seeds:
                print(
                    f"\n=== Held-out seed {seed} ==="
                )

                heldout.append(
                    run_heldout_seed(
                        seed=seed,
                        pilot_cfg_path=pilot_cfg_path,
                        pilot_script_path=pilot_script_path,
                        pilot_result_dir=pilot_result_dir,
                        pilot_figure_dir=pilot_figure_dir,
                        formal_result_root=formal_result_root,
                        formal_figure_root=formal_figure_root,
                    )
                )

        finally:
            copytree_exact(
                backup_results,
                pilot_result_dir,
            )
            copytree_exact(
                backup_figures,
                pilot_figure_dir,
            )

    # Verify seed 1 was restored exactly enough for tracked evidence.
    seed1_meta = json.loads(
        (pilot_result_dir / "pilot_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    if int(seed1_meta["seed"]) != 1:
        raise RuntimeError(
            "Seed-1 pilot results were not restored correctly"
        )

    seed1 = load_seed1(
        pilot_result_dir
    )

    heldout_metrics = pd.concat(
        [x["metrics"] for x in heldout],
        ignore_index=True,
    )

    heldout_curve = pd.concat(
        [x["curve"] for x in heldout],
        ignore_index=True,
    )

    heldout_diag = pd.concat(
        [x["diag"] for x in heldout],
        ignore_index=True,
    )

    heldout_bank = pd.concat(
        [x["bank"] for x in heldout],
        ignore_index=True,
    )

    heldout_encoder = pd.concat(
        [x["encoder"] for x in heldout],
        ignore_index=True,
    )

    heldout_metrics.to_csv(
        formal_result_root / "heldout_metrics_seeds23.csv",
        index=False,
    )

    heldout_curve.to_csv(
        formal_result_root / "heldout_A2_curves_seeds23.csv",
        index=False,
    )

    heldout_diag.to_csv(
        formal_result_root / "heldout_sampler_diagnostics_seeds23.csv",
        index=False,
    )

    heldout_bank.to_csv(
        formal_result_root / "heldout_historical_bank_seeds23.csv",
        index=False,
    )

    heldout_encoder.to_csv(
        formal_result_root / "heldout_encoder_summary_seeds23.csv",
        index=False,
    )

    all_metrics = pd.concat(
        [
            seed1["metrics"],
            heldout_metrics,
        ],
        ignore_index=True,
    )

    all_curve = pd.concat(
        [
            seed1["curve"],
            heldout_curve,
        ],
        ignore_index=True,
    )

    all_diag = pd.concat(
        [
            seed1["diag"],
            heldout_diag,
        ],
        ignore_index=True,
    )

    all_bank = pd.concat(
        [
            seed1["bank"],
            heldout_bank,
        ],
        ignore_index=True,
    )

    all_encoder = pd.concat(
        [
            seed1["encoder"],
            heldout_encoder,
        ],
        ignore_index=True,
        sort=False,
    )

    paired = paired_table(
        all_metrics
    )

    heldout_paired = paired[
        paired["seed"].isin([2, 3])
    ].copy()

    drift = control_drift(
        all_metrics
    )

    summary = pd.DataFrame([{
        "n_seeds": 3,
        "mean_learned_recovery_improvement_vs_recent": float(
            paired[
                "learned_recovery_improvement_vs_recent"
            ].mean()
        ),
        "sd_learned_recovery_improvement_vs_recent": float(
            paired[
                "learned_recovery_improvement_vs_recent"
            ].std(ddof=1)
        ),
        "positive_learned_recovery_seeds": int(
            (
                paired[
                    "learned_recovery_improvement_vs_recent"
                ] > 0
            ).sum()
        ),
        "mean_learned_AUC_advantage_vs_recent": float(
            paired[
                "learned_AUC_advantage_vs_recent"
            ].mean()
        ),
        "sd_learned_AUC_advantage_vs_recent": float(
            paired[
                "learned_AUC_advantage_vs_recent"
            ].std(ddof=1)
        ),
        "positive_learned_AUC_seeds": int(
            (
                paired[
                    "learned_AUC_advantage_vs_recent"
                ] > 0
            ).sum()
        ),
        "mean_learned_recovery_gap_to_oracle": float(
            paired[
                "learned_recovery_gap_to_oracle"
            ].mean()
        ),
        "mean_learned_minus_oracle_AUC": float(
            paired[
                "learned_minus_oracle_AUC"
            ].mean()
        ),
    }])

    all_metrics.to_csv(
        formal_result_root / "metrics_all_3seeds.csv",
        index=False,
    )

    all_curve.to_csv(
        formal_result_root / "A2_curves_all_3seeds.csv",
        index=False,
    )

    all_diag.to_csv(
        formal_result_root / "sampler_diagnostics_all_3seeds.csv",
        index=False,
    )

    all_bank.to_csv(
        formal_result_root / "historical_bank_all_3seeds.csv",
        index=False,
    )

    all_encoder.to_csv(
        formal_result_root / "encoder_summary_all_3seeds.csv",
        index=False,
    )

    paired.to_csv(
        formal_result_root / "paired_advantages_all_3seeds.csv",
        index=False,
    )

    drift.to_csv(
        formal_result_root / "heldout_oracle_control_drift_seeds23.csv",
        index=False,
    )

    summary.to_csv(
        formal_result_root / "summary_all_3seeds.csv",
        index=False,
    )

    plot_three_seed(
        all_curve,
        formal_figure_root / "A2_recovery_first_2000_3seeds.png",
        max_step=2000,
    )

    pilot_full_cfg = load_config(
        ROOT / "configs/base.yaml",
        pilot_cfg_path,
    )

    plot_three_seed(
        all_curve,
        formal_figure_root / "A2_recovery_full_3seeds.png",
        max_step=int(
            pilot_full_cfg["isolation"]["A2_steps"]
        ),
    )

    save_json(
        {
            "code_commit": git_sha(),
            "development_seed": 1,
            "heldout_replication_seeds": [2, 3],
            "seed1_result_reused_without_rerun": True,
            "heldout_runs_call_exact_seed1_pilot_script": True,
            "only_config_change_per_heldout_run": "isolation.seed",
            "pilot_config_sha256": pilot_cfg_sha,
            "pilot_script_sha256": pilot_script_sha,
            "no_hyperparameter_tuning_between_seeds": True,
            "learned_selector_uses_true_context_labels": False,
            "heldout_positive_learned_recovery_seeds": int(
                (
                    heldout_paired[
                        "learned_recovery_improvement_vs_recent"
                    ] > 0
                ).sum()
            ),
            "heldout_positive_learned_AUC_seeds": int(
                (
                    heldout_paired[
                        "learned_AUC_advantage_vs_recent"
                    ] > 0
                ).sum()
            ),
            "all3_positive_learned_recovery_seeds": int(
                (
                    paired[
                        "learned_recovery_improvement_vs_recent"
                    ] > 0
                ).sum()
            ),
            "all3_positive_learned_AUC_seeds": int(
                (
                    paired[
                        "learned_AUC_advantage_vs_recent"
                    ] > 0
                ).sum()
            ),
            "n_significance_tests": 0,
        },
        formal_result_root / "replication_metadata.json",
    )

    save_yaml(
        rep_cfg,
        formal_result_root / "resolved_replication_config.yaml",
    )

    print("\n=== Held-out paired results ===")
    print(heldout_paired.to_string(index=False))

    print("\n=== All three seeds ===")
    print(paired.to_string(index=False))

    print("\n=== Three-seed summary ===")
    print(summary.to_string(index=False))

    print(
        "\nHeld-out positive recovery seeds:",
        int(
            (
                heldout_paired[
                    "learned_recovery_improvement_vs_recent"
                ] > 0
            ).sum()
        ),
        "/ 2",
    )

    print(
        "Held-out positive AUC seeds:",
        int(
            (
                heldout_paired[
                    "learned_AUC_advantage_vs_recent"
                ] > 0
            ).sum()
        ),
        "/ 2",
    )


if __name__ == "__main__":
    main()
