from __future__ import annotations

import argparse
import copy
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

from scripts.run_context_representation_pilot_v2 import (
    Std,
    cat,
    collect,
    fit_models,
    pred_base,
    pred_hist,
    retrieval,
)
from scripts.run_dynamics_isolated_retrieval_pilot_v2 import (
    online_windows_dyn,
    windows_dyn,
)
from scripts.run_learned_relevance_retrieval_pilot_v2 import (
    balance_bank,
    encode_histories,
    run_phase,
)
from scripts.train_stationary_v2 import (
    epsilon_at,
    make_agent,
    make_env,
)
from src.replay.recent import RecentReplayBuffer
from src.replay.uniform import UniformReplayBuffer
from src.representation.similarity import topk_label_precision
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


@torch.no_grad()
def greedy_action(agent, state):
    x = torch.as_tensor(
        np.asarray(state, dtype=np.float32),
        device=agent.device,
    ).unsqueeze(0)

    return int(
        agent.online(x).argmax(dim=1).item()
    )


def train_stationary_policy(
    context,
    seed,
    steps,
    cfg,
):
    set_global_seed(seed)

    env = make_env(
        cfg,
        context,
    )

    agent = make_agent(
        cfg,
        seed,
    )

    replay = UniformReplayBuffer(
        capacity=50000,
        seed=seed,
    )

    learning_starts = int(
        cfg["experiment"]["learning_starts"]
    )

    train_frequency = int(
        cfg["experiment"]["train_frequency"]
    )

    batch_size = int(
        cfg["agent"]["batch_size"]
    )

    state, info = env.reset(
        seed=150_000_000 + seed * 1000
    )

    global_step = 0
    episode_id = 0

    while global_step < int(steps):
        action = agent.act(
            state,
            epsilon=epsilon_at(
                global_step,
                cfg,
            ),
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(action)

        done = bool(
            terminated
            or truncated
        )

        replay.add(
            state,
            action,
            reward,
            next_state,
            terminated,
            context_id=context,
            context_vector=info[
                "context_vector"
            ],
        )

        if (
            global_step >= learning_starts
            and global_step % train_frequency == 0
            and len(replay) >= batch_size
        ):
            agent.update(
                replay.sample(
                    batch_size
                )
            )

        state = next_state
        info = next_info
        global_step += 1

        if done and global_step < int(steps):
            episode_id += 1

            state, info = env.reset(
                seed=(
                    150_000_000
                    + seed * 1000
                    + episode_id
                )
            )

    env.close()
    agent.online.eval()
    agent.target.eval()

    return agent


def collect_policy_episodes(
    policy,
    behavior_name,
    context,
    n_episodes,
    steps_per_episode,
    seed_base,
):
    episodes = []

    for ep in range(int(n_episodes)):
        env = make_env(
            GLOBAL_CFG,
            context,
        )

        state, _ = env.reset(
            seed=int(seed_base) + ep
        )

        states = [
            np.asarray(
                state,
                dtype=np.float32,
            )
        ]

        actions = []

        for _ in range(
            int(steps_per_episode)
        ):
            action = greedy_action(
                policy,
                state,
            )

            (
                next_state,
                _,
                terminated,
                truncated,
                _,
            ) = env.step(
                action
            )

            actions.append(
                int(action)
            )

            states.append(
                np.asarray(
                    next_state,
                    dtype=np.float32,
                )
            )

            state = next_state

            if (
                terminated
                or truncated
            ):
                break

        env.close()

        episodes.append({
            "episode_id": ep,
            "context": context,
            "behavior": behavior_name,
            "states": np.asarray(
                states,
                dtype=np.float32,
            ),
            "actions": np.asarray(
                actions,
                dtype=np.int64,
            ),
        })

    return episodes


def add_behavior_field(
    data,
    behavior_name,
):
    out = {
        key: value.copy()
        for key, value in data.items()
    }

    out["behavior"] = np.asarray(
        [behavior_name] * len(
            out["label"]
        )
    )

    return out


def concat_datasets(
    datasets,
):
    keys = datasets[0].keys()

    return {
        key: np.concatenate(
            [
                data[key]
                for data in datasets
            ],
            axis=0,
        )
        for key in keys
    }


def make_random_splits(
    context,
    cfg,
):
    r = cfg["representation"]

    n_train = int(
        r["train_episodes_per_context"]
    )
    n_val = int(
        r["val_episodes_per_context"]
    )
    n_test = int(
        r["test_episodes_per_context"]
    )

    total = (
        n_train
        + n_val
        + n_test
    )

    episodes = collect(
        context,
        total,
        cfg,
    )

    train = windows_dyn(
        episodes[
            :n_train
        ],
        cfg,
    )

    val = windows_dyn(
        episodes[
            n_train:
            n_train + n_val
        ],
        cfg,
    )

    test = windows_dyn(
        episodes[
            n_train + n_val:
        ],
        cfg,
    )

    return (
        add_behavior_field(
            train,
            "random",
        ),
        add_behavior_field(
            val,
            "random",
        ),
        add_behavior_field(
            test,
            "random",
        ),
    )


def make_policy_splits(
    policy,
    behavior_name,
    context,
    cfg,
    seed_offset,
):
    b = cfg["behavior_diverse"]

    n_train = int(
        b[
            "ddqn_train_episodes_per_behavior_context"
        ]
    )
    n_val = int(
        b[
            "ddqn_val_episodes_per_behavior_context"
        ]
    )
    n_test = int(
        b[
            "ddqn_test_episodes_per_behavior_context"
        ]
    )

    total = (
        n_train
        + n_val
        + n_test
    )

    episodes = collect_policy_episodes(
        policy=policy,
        behavior_name=behavior_name,
        context=context,
        n_episodes=total,
        steps_per_episode=int(
            b[
                "rollout_steps_per_episode"
            ]
        ),
        seed_base=(
            160_000_000
            + int(seed_offset)
        ),
    )

    train = windows_dyn(
        episodes[
            :n_train
        ],
        cfg,
    )

    val = windows_dyn(
        episodes[
            n_train:
            n_train + n_val
        ],
        cfg,
    )

    test = windows_dyn(
        episodes[
            n_train + n_val:
        ],
        cfg,
    )

    return (
        add_behavior_field(
            train,
            behavior_name,
        ),
        add_behavior_field(
            val,
            behavior_name,
        ),
        add_behavior_field(
            test,
            behavior_name,
        ),
    )


def standardize_dataset(
    data,
    history_std,
    current_std,
    target_std,
):
    out = {
        key: value.copy()
        for key, value in data.items()
    }

    out["history"] = history_std.t(
        data["history"]
    )

    out["current"] = current_std.t(
        data["current"]
    )

    out["target"] = target_std.t(
        data["target"]
    )

    return out


def strip_aux_fields(
    data,
):
    return {
        key: value
        for key, value in data.items()
        if key in {
            "history",
            "current",
            "target",
            "label",
            "episode_id",
        }
    }


def train_behavior_diverse_encoder(
    cfg,
):
    b = cfg["behavior_diverse"]
    r = cfg["representation"]

    policy_A = train_stationary_policy(
        context="A",
        seed=int(
            b["policy_A_seed"]
        ),
        steps=int(
            b["policy_train_steps"]
        ),
        cfg=cfg,
    )

    policy_B = train_stationary_policy(
        context="B",
        seed=int(
            b["policy_B_seed"]
        ),
        steps=int(
            b["policy_train_steps"]
        ),
        cfg=cfg,
    )

    train_parts = []
    val_parts = []
    test_parts = []

    for context_index, context in enumerate(
        ["A", "B"]
    ):
        random_train, random_val, random_test = (
            make_random_splits(
                context,
                cfg,
            )
        )

        train_parts.append(
            random_train
        )
        val_parts.append(
            random_val
        )
        test_parts.append(
            random_test
        )

        for behavior_index, (
            behavior_name,
            policy,
        ) in enumerate([
            ("policy_A", policy_A),
            ("policy_B", policy_B),
        ]):
            (
                p_train,
                p_val,
                p_test,
            ) = make_policy_splits(
                policy=policy,
                behavior_name=behavior_name,
                context=context,
                cfg=cfg,
                seed_offset=(
                    context_index * 100_000
                    + behavior_index * 10_000
                ),
            )

            train_parts.append(
                p_train
            )
            val_parts.append(
                p_val
            )
            test_parts.append(
                p_test
            )

    train = concat_datasets(
        train_parts
    )

    val = concat_datasets(
        val_parts
    )

    test = concat_datasets(
        test_parts
    )

    history_std = Std.fit(
        train["history"].reshape(
            -1,
            train["history"].shape[-1],
        )
    )

    current_std = Std.fit(
        train["current"]
    )

    target_std = Std.fit(
        train["target"]
    )

    train_s = standardize_dataset(
        train,
        history_std,
        current_std,
        target_std,
    )

    val_s = standardize_dataset(
        val,
        history_std,
        current_std,
        target_std,
    )

    test_s = standardize_dataset(
        test,
        history_std,
        current_std,
        target_std,
    )

    fit_train = strip_aux_fields(
        train_s
    )

    fit_val = strip_aux_fields(
        val_s
    )

    fit_test = strip_aux_fields(
        test_s
    )

    (
        model,
        baseline,
        history_curve,
        baseline_curve,
        model_meta,
        baseline_meta,
    ) = fit_models(
        fit_train,
        fit_val,
        cfg,
    )

    hist_pred_norm, test_z = pred_hist(
        model,
        fit_test,
        r["batch_size"],
    )

    base_pred_norm = pred_base(
        baseline,
        fit_test,
        r["batch_size"],
    )

    hist_pred = target_std.inv(
        hist_pred_norm
    )

    base_pred = target_std.inv(
        base_pred_norm
    )

    target = test["target"]

    history_mse = float(
        np.mean(
            (
                hist_pred
                - target
            ) ** 2
        )
    )

    baseline_mse = float(
        np.mean(
            (
                base_pred
                - target
            ) ** 2
        )
    )

    relative_mse_improvement = float(
        1.0
        - history_mse
        / baseline_mse
    )

    _, train_z = pred_hist(
        model,
        fit_train,
        r["batch_size"],
    )

    (
        overall_precision,
        overall_centroid_accuracy,
        _,
        _,
    ) = retrieval(
        train_z,
        train["label"],
        test_z,
        test["label"],
        10,
        8000,
        int(
            r["seed"]
        ),
    )

    behavior_rows = []

    for behavior in [
        "random",
        "policy_A",
        "policy_B",
    ]:
        mask = (
            test["behavior"]
            == behavior
        )

        if not np.any(mask):
            continue

        (
            precision,
            centroid_accuracy,
            _,
            _,
        ) = retrieval(
            train_z,
            train["label"],
            test_z[
                mask
            ],
            test["label"][
                mask
            ],
            10,
            8000,
            int(
                r["seed"]
            ),
        )

        behavior_rows.append({
            "behavior": behavior,
            "n_test_windows": int(
                np.sum(mask)
            ),
            "precision_at_10": float(
                precision
            ),
            "centroid_accuracy": float(
                centroid_accuracy
            ),
        })

    model.eval()

    return {
        "model": model,
        "history_std": history_std,
        "history_curve": history_curve,
        "model_meta": model_meta,
        "history_mse": history_mse,
        "baseline_mse": baseline_mse,
        "relative_mse_improvement": relative_mse_improvement,
        "offline_precision_at_10": float(
            overall_precision
        ),
        "offline_centroid_accuracy": float(
            overall_centroid_accuracy
        ),
        "behavior_diagnostics": pd.DataFrame(
            behavior_rows
        ),
        "train_source_counts": pd.DataFrame([
            {
                "split": "train",
                "behavior": behavior,
                "context": context,
                "n_windows": int(
                    np.sum(
                        (
                            train["behavior"]
                            == behavior
                        )
                        & (
                            train["label"]
                            == context
                        )
                    )
                ),
            }
            for behavior in [
                "random",
                "policy_A",
                "policy_B",
            ]
            for context in [
                "A",
                "B",
            ]
        ]),
    }


def main():
    global GLOBAL_CFG

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "behavior_diverse_retrieval_pilot.yaml"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / args.config,
    )

    GLOBAL_CFG = cfg

    r = cfg["representation"]
    q = cfg["retrieval"]
    seed = int(
        q["seed"]
    )

    torch.set_num_threads(
        1
    )

    set_global_seed(
        seed
    )

    out = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "behavior_diverse_retrieval_pilot"
    )

    figs = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "behavior_diverse_retrieval_pilot"
    )

    rep = train_behavior_diverse_encoder(
        cfg
    )

    model = rep["model"]
    history_std = rep["history_std"]

    # Unchanged seed-1 nonstationary Recent-DDQN evaluation trajectory.
    set_global_seed(seed)

    agent = make_agent(
        cfg,
        seed,
    )

    replay = RecentReplayBuffer(
        capacity=int(
            q["replay_capacity"]
        ),
        recent_window=int(
            q["recent_window"]
        ),
        seed=seed,
    )

    global_step = 0

    global_step, A1 = run_phase(
        agent,
        replay,
        "A",
        int(
            q["A1_steps"]
        ),
        global_step,
        seed,
        0,
        cfg,
    )

    global_step, B = run_phase(
        agent,
        replay,
        "B",
        int(
            q["B_steps"]
        ),
        global_step,
        seed,
        1,
        cfg,
    )

    global_step, A2 = run_phase(
        agent,
        replay,
        "A",
        int(
            q["A2_steps"]
        ),
        global_step,
        seed,
        2,
        cfg,
    )

    L = int(
        r["history_length"]
    )

    a1 = online_windows_dyn(
        A1,
        L,
        int(
            q["bank_stride"]
        ),
    )

    b = online_windows_dyn(
        B,
        L,
        int(
            q["bank_stride"]
        ),
    )

    a2 = online_windows_dyn(
        A2,
        L,
        int(
            q["query_stride"]
        ),
    )

    bank_h = np.concatenate(
        [
            a1["history"],
            b["history"],
        ],
        axis=0,
    )

    bank_y = np.concatenate(
        [
            a1["label"],
            b["label"],
        ],
        axis=0,
    )

    bank_steps = np.concatenate(
        [
            a1["phase_step"],
            b["phase_step"],
        ],
        axis=0,
    )

    bank_z = encode_histories(
        model,
        bank_h,
        history_std,
    )

    query_z = encode_histories(
        model,
        a2["history"],
        history_std,
    )

    (
        bank_z,
        bank_y,
        bank_steps,
    ) = balance_bank(
        bank_z,
        bank_y,
        bank_steps,
        int(
            q[
                "max_bank_windows_per_context"
            ]
        ),
        seed,
    )

    result10 = topk_label_precision(
        query_z,
        bank_z,
        bank_y,
        target_label="A",
        k=int(
            q["top_k_primary"]
        ),
    )

    result50 = topk_label_precision(
        query_z,
        bank_z,
        bank_y,
        target_label="A",
        k=int(
            q["top_k_secondary"]
        ),
    )

    df = pd.DataFrame({
        "phase_step": a2["phase_step"],
        "episode_id": a2["episode_id"],
        "episode_local_step": a2[
            "episode_local_step"
        ],
        "precision_at_10": result10[
            "precision_per_query"
        ],
        "precision_at_50": result50[
            "precision_per_query"
        ],
        "best_similarity": result10[
            "best_similarity"
        ],
        "same_vs_other_margin": result10[
            "same_vs_other_margin"
        ],
    })

    early_mask = (
        df["phase_step"]
        <= int(
            q[
                "early_query_max_step"
            ]
        )
    )

    early_precision = float(
        df.loc[
            early_mask,
            "precision_at_10",
        ].mean()
    )

    full_precision = float(
        df[
            "precision_at_10"
        ].mean()
    )

    full_precision50 = float(
        df[
            "precision_at_50"
        ].mean()
    )

    early_margin = float(
        df.loc[
            early_mask,
            "same_vs_other_margin",
        ].mean()
    )

    bank_A_fraction = float(
        np.mean(
            bank_y == "A"
        )
    )

    criteria = cfg["criteria"]

    early_pass = bool(
        early_precision
        >= float(
            criteria[
                "min_early_precision_at_10"
            ]
        )
    )

    full_pass = bool(
        full_precision
        >= float(
            criteria[
                "min_first2k_precision_at_10"
            ]
        )
    )

    pilot_pass = bool(
        early_pass
        and full_pass
    )

    prior = pd.read_csv(
        ROOT
        / "results"
        / "synthetic_v2"
        / "dynamics_isolated_retrieval_pilot"
        / "retrieval_summary.csv"
    ).iloc[0]

    comparison = pd.DataFrame([
        {
            "encoder_training_distribution": "random_only",
            "history_feature": "action_delta_only",
            "early_A2_precision_at_10": float(
                prior[
                    "early_A2_precision_at_10"
                ]
            ),
            "first2k_precision_at_10": float(
                prior[
                    "first2k_precision_at_10"
                ]
            ),
            "first2k_precision_at_50": float(
                prior[
                    "first2k_precision_at_50"
                ]
            ),
        },
        {
            "encoder_training_distribution": (
                "random_plus_policyA_plus_policyB"
            ),
            "history_feature": "action_delta_only",
            "early_A2_precision_at_10": early_precision,
            "first2k_precision_at_10": full_precision,
            "first2k_precision_at_50": full_precision50,
        },
    ])

    comparison.to_csv(
        out
        / "comparison_to_05b.csv",
        index=False,
    )

    summary = pd.DataFrame([
        {
            "history_feature": "action_delta_only",
            "encoder_training_distribution": (
                "random_plus_policyA_plus_policyB"
            ),
            "n_bank_windows": int(
                len(
                    bank_z
                )
            ),
            "bank_A_fraction": bank_A_fraction,
            "n_A2_queries": int(
                len(
                    df
                )
            ),
            "offline_history_test_mse": rep[
                "history_mse"
            ],
            "offline_no_history_test_mse": rep[
                "baseline_mse"
            ],
            "offline_relative_mse_improvement": rep[
                "relative_mse_improvement"
            ],
            "offline_AB_precision_at_10": rep[
                "offline_precision_at_10"
            ],
            "offline_AB_centroid_accuracy": rep[
                "offline_centroid_accuracy"
            ],
            "early_A2_precision_at_10": early_precision,
            "first2k_precision_at_10": full_precision,
            "first2k_precision_at_50": full_precision50,
            "early_same_vs_other_margin": early_margin,
            "early_pass": early_pass,
            "first2k_pass": full_pass,
            "pilot_pass": pilot_pass,
        }
    ])

    summary.to_csv(
        out
        / "retrieval_summary.csv",
        index=False,
    )

    df.to_csv(
        out
        / "A2_query_metrics.csv",
        index=False,
    )

    rep[
        "behavior_diagnostics"
    ].to_csv(
        out
        / "offline_diagnostics_by_behavior.csv",
        index=False,
    )

    rep[
        "train_source_counts"
    ].to_csv(
        out
        / "encoder_train_source_counts.csv",
        index=False,
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

    df["step_bin"] = pd.cut(
        df["phase_step"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    binned = (
        df
        .groupby(
            "step_bin",
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

    binned.to_csv(
        out
        / "retrieval_by_A2_step_bin.csv",
        index=False,
    )

    episode_rows = []

    for episode_id, group in df.groupby(
        "episode_id"
    ):
        cut = int(
            group[
                "episode_local_step"
            ].min()
        ) + 40

        early_episode = group[
            group[
                "episode_local_step"
            ]
            <= cut
        ]

        episode_rows.append({
            "episode_id": int(
                episode_id
            ),
            "phase_start": int(
                group[
                    "phase_step"
                ].min()
            ),
            "phase_end": int(
                group[
                    "phase_step"
                ].max()
            ),
            "mean_precision_at_10": float(
                group[
                    "precision_at_10"
                ].mean()
            ),
            "first_40ish_precision_at_10": float(
                early_episode[
                    "precision_at_10"
                ].mean()
            ),
            "mean_margin": float(
                group[
                    "same_vs_other_margin"
                ].mean()
            ),
        })

    pd.DataFrame(
        episode_rows
    ).to_csv(
        out
        / "retrieval_by_episode.csv",
        index=False,
    )

    rep[
        "history_curve"
    ].to_csv(
        out
        / "encoder_training_curve.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.0)
    )

    ax.plot(
        df[
            "phase_step"
        ],
        df[
            "precision_at_10"
        ].rolling(
            15,
            min_periods=1,
        ).mean(),
        label="precision@10",
    )

    ax.plot(
        df[
            "phase_step"
        ],
        df[
            "precision_at_50"
        ].rolling(
            15,
            min_periods=1,
        ).mean(),
        label="precision@50",
    )

    ax.axhline(
        bank_A_fraction,
        linestyle="--",
        label="bank A prevalence",
    )

    ax.axhline(
        0.90,
        linestyle=":",
        label="go/no-go threshold",
    )

    ax.set_xlabel(
        "A2 phase step"
    )

    ax.set_ylabel(
        "Retrieval precision"
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.set_title(
        "05c Behavior-Diverse Historical Retrieval"
    )

    ax.legend()
    fig.tight_layout()

    fig.savefig(
        figs
        / "A2_retrieval_precision.png",
        dpi=180,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    ax.bar(
        comparison[
            "encoder_training_distribution"
        ],
        comparison[
            "early_A2_precision_at_10"
        ],
    )

    ax.axhline(
        0.90,
        linestyle="--",
    )

    ax.set_ylabel(
        "Early A2 precision@10"
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_title(
        "05b vs 05c: Behavior-Diverse Encoder Training"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()

    fig.savefig(
        figs
        / "behavior_training_comparison.png",
        dpi=180,
    )

    plt.close(fig)

    save_json(
        {
            "code_commit": git_sha(),
            "seed": seed,
            "history_feature": "action_delta_only",
            "only_conceptual_change_from_05b": (
                "encoder training distribution broadened from random-only "
                "to random + independent policy_A + independent policy_B, "
                "with each policy rolled out under both A and B dynamics"
            ),
            "encoder_training_labels_used": False,
            "evaluation_trajectory_used_for_encoder_training": False,
            "policy_A_seed": int(
                cfg[
                    "behavior_diverse"
                ][
                    "policy_A_seed"
                ]
            ),
            "policy_B_seed": int(
                cfg[
                    "behavior_diverse"
                ][
                    "policy_B_seed"
                ]
            ),
            "offline_relative_mse_improvement": rep[
                "relative_mse_improvement"
            ],
            "offline_AB_precision_at_10": rep[
                "offline_precision_at_10"
            ],
            "offline_AB_centroid_accuracy": rep[
                "offline_centroid_accuracy"
            ],
            "old_05b_early_A2_precision_at_10": float(
                prior[
                    "early_A2_precision_at_10"
                ]
            ),
            "new_05c_early_A2_precision_at_10": early_precision,
            "old_05b_first2k_precision_at_10": float(
                prior[
                    "first2k_precision_at_10"
                ]
            ),
            "new_05c_first2k_precision_at_10": full_precision,
            "criteria": criteria,
            "early_pass": early_pass,
            "first2k_pass": full_pass,
            "behavior_diverse_retrieval_pilot_pass": pilot_pass,
            "next_if_pass": (
                "fork-at-return end-to-end Recent vs Learned Relevant vs Oracle"
            ),
            "next_if_fail": (
                "stop representation micro-tuning; reassess learned-relevance "
                "scope or use prediction-error/support-aware matching"
            ),
        },
        out
        / "retrieval_metadata.json",
    )

    save_yaml(
        cfg,
        out
        / "resolved_config.yaml",
    )

    print(
        "\n=== 05c behavior-diverse retrieval summary ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== Direct comparison to 05b ==="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\n=== Offline diagnostics by behavior ==="
    )

    print(
        rep[
            "behavior_diagnostics"
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Retrieval by A2 step bin ==="
    )

    print(
        binned.to_string(
            index=False
        )
    )

    print(
        "\n05C BEHAVIOR-DIVERSE RETRIEVAL PILOT PASS:",
        pilot_pass,
    )


GLOBAL_CFG = None

if __name__ == "__main__":
    main()
