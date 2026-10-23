from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import scripts.run_behavior_diverse_retrieval_pilot_v2 as behavior_module
from scripts.run_dynamics_isolated_retrieval_pilot_v2 import (
    online_windows_dyn,
    train_rep,
)
from scripts.run_learned_relevance_retrieval_pilot_v2 import (
    encode_histories,
    run_phase,
)
from scripts.train_stationary_v2 import make_agent
from src.replay.recent import RecentReplayBuffer
from src.representation.similarity import topk_label_precision
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


PHASES = [
    ("A1", "A", 0, "A1_steps"),
    ("B", "B", 1, "B_steps"),
    ("A2", "A", 2, "A2_steps"),
]


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def bundle_hash(bundle):
    h = hashlib.sha256()

    for key in sorted(bundle):
        arr = np.asarray(bundle[key])
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(arr).tobytes())

    return h.hexdigest()


def array_hash(arr):
    arr = np.asarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def flatten_phase(episodes, phase_code):
    states = []
    actions = []
    next_states = []
    phase_codes = []
    episode_ids = []
    phase_steps = []
    local_steps = []

    for episode in episodes:
        S = episode["states"]
        A = episode["actions"]
        start = int(episode["start_phase_step"])

        for i, action in enumerate(A):
            states.append(S[i])
            actions.append(int(action))
            next_states.append(S[i + 1])
            phase_codes.append(int(phase_code))
            episode_ids.append(int(episode["episode_id"]))
            phase_steps.append(int(start + i + 1))
            local_steps.append(int(i + 1))

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "phase_code": np.asarray(phase_codes, dtype=np.int8),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "phase_step": np.asarray(phase_steps, dtype=np.int32),
        "episode_local_step": np.asarray(local_steps, dtype=np.int32),
    }


def generate_once(cfg):
    tcfg = cfg["trajectory"]
    seed = int(tcfg["seed"])

    torch.set_num_threads(1)
    set_global_seed(seed)

    agent = make_agent(cfg, seed)

    replay = RecentReplayBuffer(
        capacity=int(tcfg["replay_capacity"]),
        recent_window=int(tcfg["recent_window"]),
        seed=seed,
    )

    global_step = 0
    pieces = []
    episode_rows = []

    for phase_name, context, phase_code, step_key in PHASES:
        expected_steps = int(tcfg[step_key])

        global_step, episodes = run_phase(
            agent=agent,
            replay=replay,
            context=context,
            phase_steps=expected_steps,
            global_step=global_step,
            seed=seed,
            phase_index=phase_code,
            cfg=cfg,
        )

        logged_steps = int(
            sum(
                len(e["actions"])
                for e in episodes
            )
        )

        if logged_steps != expected_steps:
            raise RuntimeError(
                f"{phase_name} logged transition count mismatch: "
                f"actual={logged_steps}, expected={expected_steps}, "
                f"n_episodes={len(episodes)}"
            )

        piece = flatten_phase(
            episodes,
            phase_code,
        )

        if len(piece["actions"]) != expected_steps:
            raise RuntimeError(
                f"{phase_name} flattened transition count mismatch: "
                f"actual={len(piece['actions'])}, expected={expected_steps}"
            )

        pieces.append(piece)

        for episode in episodes:
            episode_rows.append({
                "phase": phase_name,
                "context": context,
                "episode_id": int(episode["episode_id"]),
                "start_phase_step": int(episode["start_phase_step"]),
                "n_transitions": int(len(episode["actions"])),
            })

    bundle = {
        key: np.concatenate(
            [piece[key] for piece in pieces],
            axis=0,
        )
        for key in pieces[0]
    }

    return bundle, pd.DataFrame(episode_rows)


def assert_identical(first, second):
    if first.keys() != second.keys():
        raise RuntimeError(
            "Repeat trajectory bundles have different keys"
        )

    for key in first:
        if not np.array_equal(first[key], second[key]):
            raise RuntimeError(
                f"Repeat generation mismatch in array: {key}"
            )


def freeze(cfg):
    out = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "frozen_retrieval_trajectory"
    )

    first, first_eps = generate_once(cfg)
    second, second_eps = generate_once(cfg)

    assert_identical(
        first,
        second,
    )

    if not first_eps.equals(second_eps):
        raise RuntimeError(
            "Repeat generation episode boundaries differ"
        )

    expected_by_phase = {
        "A1": int(cfg["trajectory"]["A1_steps"]),
        "B": int(cfg["trajectory"]["B_steps"]),
        "A2": int(cfg["trajectory"]["A2_steps"]),
    }

    actual_by_phase = {
        phase_name: int(
            np.sum(
                first["phase_code"] == phase_code
            )
        )
        for phase_name, _, phase_code, _ in PHASES
    }

    if actual_by_phase != expected_by_phase:
        raise RuntimeError(
            f"Phase transition counts mismatch: "
            f"actual={actual_by_phase}, expected={expected_by_phase}"
        )

    expected_total = int(
        sum(expected_by_phase.values())
    )

    if len(first["actions"]) != expected_total:
        raise RuntimeError(
            f"Total transition count mismatch: "
            f"actual={len(first['actions'])}, expected={expected_total}"
        )

    npz_path = (
        out
        / "canonical_seed1_recent_ABA.npz"
    )

    np.savez_compressed(
        npz_path,
        **first,
    )

    first_eps.to_csv(
        out
        / "canonical_episode_boundaries.csv",
        index=False,
    )

    episode_counts = {
        phase_name: int(
            first_eps[
                first_eps["phase"] == phase_name
            ].shape[0]
        )
        for phase_name in expected_by_phase
    }

    manifest = {
        "code_commit": git_sha(),
        "seed": int(cfg["trajectory"]["seed"]),
        "n_transitions": int(len(first["actions"])),
        "phase_transition_counts": actual_by_phase,
        "phase_episode_counts": episode_counts,
        "repeat_generation_verified_exact": True,
        "array_bundle_sha256": bundle_hash(first),
        "npz_sha256": file_hash(npz_path),
        "purpose": (
            "immutable corrected evaluation trajectory "
            "for frozen 05b vs 05c comparison"
        ),
    }

    with (
        out
        / "trajectory_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        "\n=== Frozen trajectory manifest ==="
    )
    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


def load_bundle(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def reconstruct(bundle, phase_name, context):
    code = {
        "A1": 0,
        "B": 1,
        "A2": 2,
    }[phase_name]

    phase_idx = np.flatnonzero(
        bundle["phase_code"] == code
    )

    episodes = []

    for episode_id in np.unique(
        bundle["episode_id"][phase_idx]
    ):
        idx = phase_idx[
            bundle["episode_id"][phase_idx]
            == episode_id
        ]

        idx = idx[
            np.argsort(
                bundle["phase_step"][idx]
            )
        ]

        states = np.vstack([
            bundle["states"][idx[0]][None, :],
            bundle["next_states"][idx],
        ]).astype(np.float32)

        episodes.append({
            "episode_id": int(episode_id),
            "context": context,
            "states": states,
            "actions": bundle["actions"][idx].astype(np.int64),
            "start_phase_step": int(
                bundle["phase_step"][idx[0]] - 1
            ),
        })

    return episodes


def balanced_idx(labels, max_per_context, seed):
    rng = np.random.default_rng(seed)
    keep = []

    for label in ["A", "B"]:
        idx = np.flatnonzero(
            labels == label
        )

        if len(idx) > int(max_per_context):
            idx = rng.choice(
                idx,
                size=int(max_per_context),
                replace=False,
            )

        keep.extend(
            idx.tolist()
        )

    return np.asarray(
        sorted(keep),
        dtype=np.int64,
    )


def eval_encoder(
    name,
    model,
    history_std,
    bank_history,
    bank_labels,
    query_history,
    query_meta,
    keep,
    ccfg,
):
    bank_z = encode_histories(
        model,
        bank_history[keep],
        history_std,
    )

    query_z = encode_histories(
        model,
        query_history,
        history_std,
    )

    kept_labels = bank_labels[keep]

    result10 = topk_label_precision(
        query_z,
        bank_z,
        kept_labels,
        target_label="A",
        k=int(ccfg["top_k_primary"]),
    )

    result50 = topk_label_precision(
        query_z,
        bank_z,
        kept_labels,
        target_label="A",
        k=int(ccfg["top_k_secondary"]),
    )

    df = pd.DataFrame({
        "encoder": name,
        "phase_step": query_meta["phase_step"],
        "episode_id": query_meta["episode_id"],
        "episode_local_step": query_meta["episode_local_step"],
        "precision_at_10": result10["precision_per_query"],
        "precision_at_50": result50["precision_per_query"],
        "best_similarity": result10["best_similarity"],
        "same_vs_other_margin": result10["same_vs_other_margin"],
    })

    early = (
        df["phase_step"]
        <= int(ccfg["early_query_max_step"])
    )

    metrics = {
        "encoder": name,
        "n_bank_windows": int(len(keep)),
        "n_A2_queries": int(len(df)),
        "bank_A_fraction": float(
            np.mean(
                kept_labels == "A"
            )
        ),
        "early_A2_precision_at_10": float(
            df.loc[
                early,
                "precision_at_10",
            ].mean()
        ),
        "first2k_precision_at_10": float(
            df["precision_at_10"].mean()
        ),
        "first2k_precision_at_50": float(
            df["precision_at_50"].mean()
        ),
        "early_same_vs_other_margin": float(
            df.loc[
                early,
                "same_vs_other_margin",
            ].mean()
        ),
    }

    return metrics, df


def compare(cfg):
    ccfg = cfg["comparison"]

    traj_dir = (
        ROOT
        / "results"
        / "synthetic_v2"
        / "frozen_retrieval_trajectory"
    )

    manifest_path = (
        traj_dir
        / "trajectory_manifest.json"
    )

    npz_path = (
        traj_dir
        / "canonical_seed1_recent_ABA.npz"
    )

    if not manifest_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            "Frozen trajectory artifacts are missing. Run --mode freeze first."
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    if not manifest[
        "repeat_generation_verified_exact"
    ]:
        raise RuntimeError(
            "Frozen trajectory is not repeat-generation verified"
        )

    bundle = load_bundle(
        npz_path
    )

    A1 = reconstruct(
        bundle,
        "A1",
        "A",
    )

    B = reconstruct(
        bundle,
        "B",
        "B",
    )

    A2 = reconstruct(
        bundle,
        "A2",
        "A",
    )

    L = int(
        ccfg["history_length"]
    )

    a1 = online_windows_dyn(
        A1,
        L,
        int(ccfg["bank_stride"]),
    )

    b = online_windows_dyn(
        B,
        L,
        int(ccfg["bank_stride"]),
    )

    a2 = online_windows_dyn(
        A2,
        L,
        int(ccfg["query_stride"]),
    )

    bank_history = np.concatenate(
        [
            a1["history"],
            b["history"],
        ],
        axis=0,
    )

    bank_labels = np.concatenate(
        [
            a1["label"],
            b["label"],
        ],
        axis=0,
    )

    query_history = a2["history"]

    keep = balanced_idx(
        bank_labels,
        int(
            ccfg[
                "max_bank_windows_per_context"
            ]
        ),
        int(
            cfg["trajectory"]["seed"]
        ),
    )

    bank_hash = array_hash(
        bank_history[keep]
    )

    query_hash = array_hash(
        query_history
    )

    cfg_05b = load_config(
        ROOT / "configs/base.yaml",
        ROOT
        / ccfg[
            "encoder_05b_config"
        ],
    )

    cfg_05c = load_config(
        ROOT / "configs/base.yaml",
        ROOT
        / ccfg[
            "encoder_05c_config"
        ],
    )

    torch.set_num_threads(1)

    set_global_seed(
        int(
            cfg_05b[
                "representation"
            ][
                "seed"
            ]
        )
    )

    rep_05b = train_rep(
        cfg_05b
    )

    metrics_05b, queries_05b = eval_encoder(
        name="05b_random_only",
        model=rep_05b["model"],
        history_std=rep_05b["history_std"],
        bank_history=bank_history,
        bank_labels=bank_labels,
        query_history=query_history,
        query_meta=a2,
        keep=keep,
        ccfg=ccfg,
    )

    behavior_module.GLOBAL_CFG = cfg_05c

    set_global_seed(
        int(
            cfg_05c[
                "representation"
            ][
                "seed"
            ]
        )
    )

    rep_05c = (
        behavior_module
        .train_behavior_diverse_encoder(
            cfg_05c
        )
    )

    metrics_05c, queries_05c = eval_encoder(
        name="05c_behavior_diverse",
        model=rep_05c["model"],
        history_std=rep_05c["history_std"],
        bank_history=bank_history,
        bank_labels=bank_labels,
        query_history=query_history,
        query_meta=a2,
        keep=keep,
        ccfg=ccfg,
    )

    query_key_cols = [
        "phase_step",
        "episode_id",
        "episode_local_step",
    ]

    if not np.array_equal(
        queries_05b[
            query_key_cols
        ].to_numpy(),
        queries_05c[
            query_key_cols
        ].to_numpy(),
    ):
        raise RuntimeError(
            "05b and 05c query metadata differ under frozen evaluation"
        )

    comparison = pd.DataFrame([
        metrics_05b,
        metrics_05c,
    ])

    criteria = cfg["criteria"]

    comparison["early_pass"] = (
        comparison[
            "early_A2_precision_at_10"
        ]
        >= float(
            criteria[
                "min_early_precision_at_10"
            ]
        )
    )

    comparison["first2k_pass"] = (
        comparison[
            "first2k_precision_at_10"
        ]
        >= float(
            criteria[
                "min_first2k_precision_at_10"
            ]
        )
    )

    comparison["formal_pass"] = (
        comparison["early_pass"]
        & comparison["first2k_pass"]
    )

    out = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "frozen_trajectory_encoder_comparison"
    )

    figs = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "frozen_trajectory_encoder_comparison"
    )

    comparison.to_csv(
        out
        / "frozen_comparison.csv",
        index=False,
    )

    queries_05b.to_csv(
        out
        / "A2_query_metrics_05b.csv",
        index=False,
    )

    queries_05c.to_csv(
        out
        / "A2_query_metrics_05c.csv",
        index=False,
    )

    combined = pd.concat(
        [
            queries_05b,
            queries_05c,
        ],
        ignore_index=True,
    )

    bins = [
        0,
        64,
        128,
        256,
        512,
        1024,
        2000,
    ]

    labels = [
        "8-64",
        "65-128",
        "129-256",
        "257-512",
        "513-1024",
        "1025-2000",
    ]

    combined["step_bin"] = pd.cut(
        combined["phase_step"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    by_bin = (
        combined
        .groupby(
            [
                "encoder",
                "step_bin",
            ],
            observed=False,
        )
        .agg(
            mean_precision_at_10=(
                "precision_at_10",
                "mean",
            ),
            mean_precision_at_50=(
                "precision_at_50",
                "mean",
            ),
            mean_margin=(
                "same_vs_other_margin",
                "mean",
            ),
            n_queries=(
                "phase_step",
                "size",
            ),
        )
        .reset_index()
    )

    by_bin.to_csv(
        out
        / "frozen_retrieval_by_step_bin.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.0)
    )

    for name, group in combined.groupby(
        "encoder"
    ):
        ax.plot(
            group["phase_step"],
            group[
                "precision_at_10"
            ].rolling(
                15,
                min_periods=1,
            ).mean(),
            label=name,
        )

    ax.axhline(
        0.90,
        linestyle="--",
        label="go/no-go threshold",
    )

    ax.set_xlabel(
        "Frozen A2 phase step"
    )

    ax.set_ylabel(
        "Precision@10"
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.set_title(
        "05d: 05b vs 05c on Identical Frozen Trajectory"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        figs
        / "frozen_A2_precision_comparison.png",
        dpi=180,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    x = np.arange(
        len(
            comparison
        )
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        comparison[
            "early_A2_precision_at_10"
        ],
        width,
        label="early A2",
    )

    ax.bar(
        x + width / 2,
        comparison[
            "first2k_precision_at_10"
        ],
        width,
        label="first 2k",
    )

    ax.axhline(
        0.90,
        linestyle="--",
    )

    ax.set_xticks(
        x,
        comparison["encoder"],
        rotation=10,
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_ylabel(
        "Precision@10"
    )

    ax.set_title(
        "Frozen-Trajectory Retrieval Summary"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        figs
        / "frozen_summary_comparison.png",
        dpi=180,
    )

    plt.close(fig)

    metadata = {
        "code_commit": git_sha(),
        "trajectory_array_bundle_sha256": manifest[
            "array_bundle_sha256"
        ],
        "trajectory_npz_sha256": manifest[
            "npz_sha256"
        ],
        "trajectory_repeat_generation_verified_exact": True,
        "same_frozen_bank_for_both_encoders": True,
        "same_frozen_queries_for_both_encoders": True,
        "balanced_bank_history_sha256": bank_hash,
        "A2_query_history_sha256": query_hash,
        "n_balanced_bank_windows": int(len(keep)),
        "n_A2_queries": int(len(query_history)),
        "criteria": criteria,
        "05b_formal_pass": bool(
            comparison.loc[
                comparison["encoder"]
                == "05b_random_only",
                "formal_pass",
            ].iloc[0]
        ),
        "05c_formal_pass": bool(
            comparison.loc[
                comparison["encoder"]
                == "05c_behavior_diverse",
                "formal_pass",
            ].iloc[0]
        ),
        "purpose": (
            "evaluation-control cleanup on bitwise-identical "
            "corrected RL experience"
        ),
    }

    save_json(
        metadata,
        out
        / "frozen_comparison_metadata.json",
    )

    save_yaml(
        cfg,
        out
        / "resolved_control_config.yaml",
    )

    print(
        "\n=== Frozen 05b vs 05c comparison ==="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\n=== Frozen retrieval by A2 step bin ==="
    )

    print(
        by_bin.to_string(
            index=False
        )
    )

    print(
        "\nBalanced bank hash:",
        bank_hash,
    )

    print(
        "A2 query hash:",
        query_hash,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "freeze",
            "compare",
        ],
        required=True,
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "frozen_trajectory_retrieval_comparison.yaml"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / args.config,
    )

    if args.mode == "freeze":
        freeze(cfg)
    else:
        compare(cfg)


if __name__ == "__main__":
    main()
