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
    pred_hist,
    sub,
    windows,
)
from scripts.train_stationary_v2 import (
    epsilon_at,
    make_agent,
    make_env,
)
from src.replay.recent import RecentReplayBuffer
from src.representation.history import (
    current_feature,
    transition_feature,
)
from src.representation.similarity import (
    topk_label_precision,
)
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def train_representation(cfg):
    rep = cfg[
        "representation"
    ]

    nt = int(
        rep[
            "train_episodes_per_context"
        ]
    )

    nv = int(
        rep[
            "val_episodes_per_context"
        ]
    )

    nte = int(
        rep[
            "test_episodes_per_context"
        ]
    )

    total = (
        nt
        + nv
        + nte
    )

    combined = cat([
        windows(
            collect(
                "A",
                total,
                cfg,
            ),
            cfg,
        ),
        windows(
            collect(
                "B",
                total,
                cfg,
            ),
            cfg,
        ),
    ])

    ids = combined[
        "episode_id"
    ]

    train = sub(
        combined,
        ids < nt,
    )

    val = sub(
        combined,
        (
            ids >= nt
        )
        & (
            ids < nt + nv
        ),
    )

    hs = Std.fit(
        train[
            "history"
        ].reshape(
            -1,
            train[
                "history"
            ].shape[-1],
        )
    )

    cs = Std.fit(
        train[
            "current"
        ]
    )

    ys = Std.fit(
        train[
            "target"
        ]
    )

    def standardize(
        data
    ):
        out = {
            key: value.copy()
            for key, value
            in data.items()
        }

        out[
            "history"
        ] = hs.t(
            data[
                "history"
            ]
        )

        out[
            "current"
        ] = cs.t(
            data[
                "current"
            ]
        )

        out[
            "target"
        ] = ys.t(
            data[
                "target"
            ]
        )

        return out

    train_s = standardize(
        train
    )

    val_s = standardize(
        val
    )

    (
        history_model,
        baseline,
        history_curve,
        baseline_curve,
        history_meta,
        baseline_meta,
    ) = fit_models(
        train_s,
        val_s,
        cfg,
    )

    del baseline
    del baseline_curve
    del baseline_meta

    history_model.eval()

    return (
        history_model,
        hs,
        cs,
        ys,
        history_meta,
        history_curve,
    )


def finish_episode(
    episodes,
    states,
    actions,
    context,
    episode_id,
    start_phase_step,
):
    if len(
        actions
    ) == 0:
        return

    episodes.append({
        "episode_id": int(
            episode_id
        ),
        "context": context,
        "states": np.asarray(
            states,
            dtype=np.float32,
        ),
        "actions": np.asarray(
            actions,
            dtype=np.int64,
        ),
        "start_phase_step": int(
            start_phase_step
        ),
    })


def run_phase(
    agent,
    replay,
    context,
    phase_steps,
    global_step,
    seed,
    phase_index,
    cfg,
):
    env = make_env(
        cfg,
        context,
    )

    batch_size = int(
        cfg[
            "agent"
        ][
            "batch_size"
        ]
    )

    learning_starts = int(
        cfg[
            "experiment"
        ][
            "learning_starts"
        ]
    )

    train_frequency = int(
        cfg[
            "experiment"
        ][
            "train_frequency"
        ]
    )

    state, info = env.reset(
        seed=(
            130_000_000
            + seed
            * 1_000_000
            + phase_index
            * 100_000
        )
    )

    phase_step = 0
    episode_id = 0
    episodes = []

    episode_states = [
        np.asarray(
            state,
            dtype=np.float32,
        )
    ]

    episode_actions = []

    episode_start = 0

    while phase_step < phase_steps:
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
        ) = env.step(
            action
        )

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
            global_step
            >= learning_starts
            and global_step
            % train_frequency
            == 0
            and len(
                replay
            )
            >= batch_size
        ):
            agent.update(
                replay.sample(
                    batch_size
                )
            )

        episode_actions.append(
            int(
                action
            )
        )

        episode_states.append(
            np.asarray(
                next_state,
                dtype=np.float32,
            )
        )

        state = next_state
        info = next_info

        phase_step += 1
        global_step += 1

        if done:
            finish_episode(
                episodes,
                episode_states,
                episode_actions,
                context,
                episode_id,
                episode_start,
            )

            episode_id += 1

            if phase_step < phase_steps:
                state, info = env.reset(
                    seed=(
                        130_000_000
                        + seed
                        * 1_000_000
                        + phase_index
                        * 100_000
                        + episode_id
                    )
                )

                episode_states = [
                    np.asarray(
                        state,
                        dtype=np.float32,
                    )
                ]

                episode_actions = []

                episode_start = phase_step

    if len(
        episode_actions
    ) > 0:
        finish_episode(
            episodes,
            episode_states,
            episode_actions,
            context,
            episode_id,
            episode_start,
        )

    env.close()

    return (
        global_step,
        episodes,
    )


def history_windows_from_online(
    episodes,
    history_length,
    stride,
):
    histories = []
    labels = []
    phase_steps = []
    episode_ids = []

    for episode in episodes:
        states = episode[
            "states"
        ]

        actions = episode[
            "actions"
        ]

        T = len(
            actions
        )

        if T < history_length:
            continue

        features = [
            transition_feature(
                states[t],
                int(
                    actions[t]
                ),
                states[
                    t + 1
                ],
                5,
            )
            for t in range(
                T
            )
        ]

        for end in range(
            history_length,
            T + 1,
            stride,
        ):
            histories.append(
                np.stack(
                    features[
                        end
                        - history_length:
                        end
                    ]
                ).astype(
                    np.float32
                )
            )

            labels.append(
                episode[
                    "context"
                ]
            )

            # Number of completed transitions within the phase.
            phase_steps.append(
                int(
                    episode[
                        "start_phase_step"
                    ]
                    + end
                )
            )

            episode_ids.append(
                int(
                    episode[
                        "episode_id"
                    ]
                )
            )

    return {
        "history": np.stack(
            histories
        ).astype(
            np.float32
        ),
        "label": np.asarray(
            labels
        ),
        "phase_step": np.asarray(
            phase_steps,
            dtype=np.int64,
        ),
        "episode_id": np.asarray(
            episode_ids,
            dtype=np.int64,
        ),
    }


def encode_histories(
    model,
    history,
    standardizer,
    batch_size=512,
):
    history_s = standardizer.t(
        history
    )

    zs = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(
                history_s
            ),
            batch_size,
        ):
            batch = torch.from_numpy(
                history_s[
                    start:
                    start
                    + batch_size
                ]
            )

            z = model.encoder(
                batch
            )

            zs.append(
                z.numpy()
            )

    return np.concatenate(
        zs,
        axis=0,
    )


def balance_bank(
    z,
    labels,
    phase_steps,
    max_per_context,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    keep = []

    for label in [
        "A",
        "B",
    ]:
        idx = np.flatnonzero(
            labels
            == label
        )

        if len(
            idx
        ) > max_per_context:
            idx = rng.choice(
                idx,
                size=max_per_context,
                replace=False,
            )

        keep.extend(
            idx.tolist()
        )

    keep = np.asarray(
        sorted(
            keep
        ),
        dtype=np.int64,
    )

    return (
        z[
            keep
        ],
        labels[
            keep
        ],
        phase_steps[
            keep
        ],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "learned_relevance_retrieval_pilot.yaml"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        ROOT
        / "configs/base.yaml",
        ROOT
        / args.config,
    )

    rep = cfg[
        "representation"
    ]

    ret = cfg[
        "retrieval"
    ]

    seed = int(
        ret[
            "seed"
        ]
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
        / "learned_relevance_retrieval_pilot"
    )

    figs = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "learned_relevance_retrieval_pilot"
    )

    (
        encoder_model,
        history_std,
        current_std,
        target_std,
        encoder_meta,
        encoder_curve,
    ) = train_representation(
        cfg
    )

    del current_std
    del target_std

    # Independent nonstationary RL trajectory.
    agent = make_agent(
        cfg,
        seed,
    )

    replay = RecentReplayBuffer(
        capacity=int(
            ret[
                "replay_capacity"
            ]
        ),
        recent_window=int(
            ret[
                "recent_window"
            ]
        ),
        seed=seed,
    )

    global_step = 0

    global_step, A1_episodes = run_phase(
        agent,
        replay,
        "A",
        int(
            ret[
                "A1_steps"
            ]
        ),
        global_step,
        seed,
        0,
        cfg,
    )

    global_step, B_episodes = run_phase(
        agent,
        replay,
        "B",
        int(
            ret[
                "B_steps"
            ]
        ),
        global_step,
        seed,
        1,
        cfg,
    )

    global_step, A2_episodes = run_phase(
        agent,
        replay,
        "A",
        int(
            ret[
                "A2_steps"
            ]
        ),
        global_step,
        seed,
        2,
        cfg,
    )

    L = int(
        rep[
            "history_length"
        ]
    )

    bank_stride = int(
        ret[
            "bank_stride"
        ]
    )

    query_stride = int(
        ret[
            "query_stride"
        ]
    )

    A1_windows = history_windows_from_online(
        A1_episodes,
        L,
        bank_stride,
    )

    B_windows = history_windows_from_online(
        B_episodes,
        L,
        bank_stride,
    )

    A2_windows = history_windows_from_online(
        A2_episodes,
        L,
        query_stride,
    )

    bank_history = np.concatenate(
        [
            A1_windows[
                "history"
            ],
            B_windows[
                "history"
            ],
        ],
        axis=0,
    )

    bank_labels = np.concatenate(
        [
            A1_windows[
                "label"
            ],
            B_windows[
                "label"
            ],
        ],
        axis=0,
    )

    bank_phase_steps = np.concatenate(
        [
            A1_windows[
                "phase_step"
            ],
            B_windows[
                "phase_step"
            ],
        ],
        axis=0,
    )

    bank_z = encode_histories(
        encoder_model,
        bank_history,
        history_std,
    )

    query_z = encode_histories(
        encoder_model,
        A2_windows[
            "history"
        ],
        history_std,
    )

    (
        bank_z,
        bank_labels,
        bank_phase_steps,
    ) = balance_bank(
        bank_z,
        bank_labels,
        bank_phase_steps,
        int(
            ret[
                "max_bank_windows_per_context"
            ]
        ),
        seed,
    )

    k10 = int(
        ret[
            "top_k_primary"
        ]
    )

    k50 = int(
        ret[
            "top_k_secondary"
        ]
    )

    result10 = topk_label_precision(
        query_z,
        bank_z,
        bank_labels,
        target_label="A",
        k=k10,
    )

    result50 = topk_label_precision(
        query_z,
        bank_z,
        bank_labels,
        target_label="A",
        k=k50,
    )

    query_df = pd.DataFrame({
        "phase_step": A2_windows[
            "phase_step"
        ],
        "episode_id": A2_windows[
            "episode_id"
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
        query_df[
            "phase_step"
        ]
        <= int(
            ret[
                "early_query_max_step"
            ]
        )
    )

    early_precision = float(
        query_df.loc[
            early_mask,
            "precision_at_10",
        ].mean()
    )

    full_precision = float(
        query_df[
            "precision_at_10"
        ].mean()
    )

    full_precision50 = float(
        query_df[
            "precision_at_50"
        ].mean()
    )

    early_margin = float(
        query_df.loc[
            early_mask,
            "same_vs_other_margin",
        ].mean()
    )

    bank_A_fraction = float(
        np.mean(
            bank_labels
            == "A"
        )
    )

    criteria = cfg[
        "criteria"
    ]

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

    summary = pd.DataFrame([
        {
            "n_bank_windows": int(
                len(
                    bank_z
                )
            ),
            "bank_A_fraction": bank_A_fraction,
            "n_A2_queries": int(
                len(
                    query_df
                )
            ),
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

    query_df.to_csv(
        out
        / "A2_query_metrics.csv",
        index=False,
    )

    bank_counts = pd.DataFrame([
        {
            "context": label,
            "n_windows": int(
                np.sum(
                    bank_labels
                    == label
                )
            ),
        }
        for label in [
            "A",
            "B",
        ]
    ])

    bank_counts.to_csv(
        out
        / "historical_bank_counts.csv",
        index=False,
    )

    encoder_curve.to_csv(
        out
        / "encoder_training_curve.csv",
        index=False,
    )

    # Aggregate retrieval precision by phase-step bins.
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

    query_df[
        "step_bin"
    ] = pd.cut(
        query_df[
            "phase_step"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    binned = (
        query_df
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

    fig, ax = plt.subplots(
        figsize=(8.5, 5.0)
    )

    ax.plot(
        query_df[
            "phase_step"
        ],
        query_df[
            "precision_at_10"
        ].rolling(
            15,
            min_periods=1,
        ).mean(),
        label="precision@10",
    )

    ax.plot(
        query_df[
            "phase_step"
        ],
        query_df[
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
        "Learned Historical Retrieval During A Recurrence"
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
        figsize=(8.5, 5.0)
    )

    ax.plot(
        query_df[
            "phase_step"
        ],
        query_df[
            "same_vs_other_margin"
        ].rolling(
            15,
            min_periods=1,
        ).mean(),
    )

    ax.axhline(
        0.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "A2 phase step"
    )

    ax.set_ylabel(
        "Best-A minus best-B cosine similarity"
    )

    ax.set_title(
        "Latent Retrieval Similarity Margin"
    )

    fig.tight_layout()
    fig.savefig(
        figs
        / "A2_similarity_margin.png",
        dpi=180,
    )
    plt.close(fig)

    save_json(
        {
            "code_commit": git_sha(),
            "seed": seed,
            "encoder_training_labels_used": False,
            "encoder_best_epoch": int(
                encoder_meta[
                    "best_epoch"
                ]
            ),
            "retrieval_trajectory_used_for_encoder_training": False,
            "history_length": L,
            "bank_A_fraction": bank_A_fraction,
            "n_bank_windows": int(
                len(
                    bank_z
                )
            ),
            "n_A2_queries": int(
                len(
                    query_df
                )
            ),
            "early_A2_precision_at_10": early_precision,
            "first2k_precision_at_10": full_precision,
            "first2k_precision_at_50": full_precision50,
            "early_same_vs_other_margin": early_margin,
            "criteria": criteria,
            "early_pass": early_pass,
            "first2k_pass": full_pass,
            "learned_relevance_retrieval_pilot_pass": pilot_pass,
            "next_if_pass": (
                "fork-at-return end-to-end Recent vs Learned Relevant vs Oracle"
            ),
            "next_if_fail": (
                "diagnose behavior-distribution shift before changing replay"
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
        "\n=== Learned relevance retrieval summary ==="
    )

    print(
        summary.to_string(
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
        "\nLEARNED RELEVANCE RETRIEVAL PILOT PASS:",
        pilot_pass,
    )


if __name__ == "__main__":
    main()
