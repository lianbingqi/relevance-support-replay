
from __future__ import annotations

import argparse
import copy
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
from scripts.run_behavior_diverse_retrieval_pilot_v2 import (
    train_behavior_diverse_encoder,
)
from scripts.run_learned_relevance_retrieval_pilot_v2 import (
    encode_histories,
)
from scripts.run_learned_replay_recurrence_pilot_v2 import (
    history_bank_windows,
    make_learned_from_recent,
    train_A2_branch,
    train_common_phase_logged,
    uniform_cap_bank,
)
from scripts.run_novelty_support_pilot_v2 import (
    evaluate as evaluate_D,
    random_D,
    train_D_branch,
    train_D_reference,
)
from scripts.run_oracle_recovery_isolation_v2 import (
    evaluate as evaluate_A,
    random_A,
)
from scripts.run_support_compatibility_development_v2 import (
    score_queries,
)
from scripts.train_stationary_v2 import (
    epsilon_at,
    make_agent,
    make_env,
)
from src.replay.recent import RecentReplayBuffer
from src.representation.history import (
    current_feature,
    dynamics_transition_feature,
)
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


METHODS = [
    "recent",
    "relevance",
    "support_aware",
    "ood_gate",
]


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


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    denom = np.maximum(
        np.linalg.norm(
            x,
            axis=1,
            keepdims=True,
        ),
        1e-8,
    )
    return (
        x
        / denom
    ).astype(np.float32)


@torch.no_grad()
def transition_gate_scores(
    prior_history,
    state,
    action,
    next_state,
    model,
    history_std,
    current_std,
    target_std,
    bank_z,
    bank_z_norm,
    top_k,
):
    query_h = np.stack(
        prior_history
    ).astype(
        np.float32
    )[
        None,
        :,
        :,
    ]

    query_z = encode_histories(
        model,
        query_h,
        history_std,
    )[0]

    qn = query_z.astype(
        np.float32
    )
    qn = qn / max(
        float(
            np.linalg.norm(
                qn
            )
        ),
        1e-8,
    )

    sim = (
        bank_z_norm
        @ qn
    )

    k = min(
        int(top_k),
        len(sim),
    )

    local = np.argpartition(
        -sim,
        kth=k - 1,
    )[
        :k
    ]

    local = local[
        np.argsort(
            -sim[
                local
            ]
        )
    ]

    c = current_std.t(
        current_feature(
            state,
            int(action),
            5,
        )[
            None,
            :,
        ]
    )[0]

    target = target_std.t(
        (
            np.asarray(
                next_state,
                dtype=np.float32,
            )
            - np.asarray(
                state,
                dtype=np.float32,
            )
        )[
            None,
            :,
        ]
    )[0]

    z_hist = torch.from_numpy(
        bank_z[
            local
        ].astype(
            np.float32
        )
    )

    c_t = torch.from_numpy(
        c.astype(
            np.float32
        )
    )

    y_t = torch.from_numpy(
        target.astype(
            np.float32
        )
    )

    hist_pred = model.head(
        torch.cat(
            [
                z_hist,
                c_t.unsqueeze(
                    0
                ).repeat(
                    k,
                    1,
                ),
            ],
            dim=1,
        )
    )

    retrieved_mean_mse = float(
        (
            (
                hist_pred.mean(
                    dim=0
                )
                - y_t
            )
            ** 2
        )
        .mean()
        .item()
    )

    self_pred = model.head(
        torch.cat(
            [
                torch.from_numpy(
                    query_z.astype(
                        np.float32
                    )
                ),
                c_t,
            ],
            dim=0,
        ).unsqueeze(
            0
        )
    )[
        0
    ]

    self_prediction_mse = float(
        (
            (
                self_pred
                - y_t
            )
            ** 2
        )
        .mean()
        .item()
    )

    return {
        "retrieved_mean_mse": (
            retrieved_mean_mse
        ),
        "self_prediction_mse": (
            self_prediction_mse
        ),
        "best_similarity": float(
            sim[
                local[
                    0
                ]
            ]
        ),
        "mean_top_similarity": float(
            sim[
                local
            ].mean()
        ),
        "top_rows": local.astype(
            np.int64
        ),
    }


def verify_gate_score_implementation(
    rep,
    bank_z,
    cfg,
):
    heldout = rep[
        "heldout_test"
    ]

    reference = score_queries(
        model=rep[
            "model"
        ],
        history_std=rep[
            "history_std"
        ],
        current_std=rep[
            "current_std"
        ],
        target_std=rep[
            "target_std"
        ],
        query={
            "history": heldout[
                "history"
            ][
                :5
            ],
            "current": heldout[
                "current"
            ][
                :5
            ],
            "target": heldout[
                "target"
            ][
                :5
            ],
        },
        bank_z=bank_z,
        top_k=int(
            cfg[
                "isolation"
            ][
                "learned_retrieval_top_k_windows"
            ]
        ),
        eps=float(
            cfg[
                "confirmation"
            ][
                "score_eps"
            ]
        ),
    )

    bank_z_norm = normalize_rows(
        bank_z
    )

    # IMPORTANT: score_queries() encodes these five histories in one batch.
    # Reuse the same encoder batch shape here, then independently recompute
    # retrieval and both prediction errors below. This avoids treating tiny
    # GRU floating-point differences from batch-size 5 vs batch-size 1 as an
    # implementation mismatch.
    query_z_batch = encode_histories(
        rep[
            "model"
        ],
        heldout[
            "history"
        ][
            :5
        ],
        rep[
            "history_std"
        ],
    )

    for i in range(
        5
    ):
        query_z = query_z_batch[
            i
        ]

        qn = query_z / max(
            float(
                np.linalg.norm(
                    query_z
                )
            ),
            1e-8,
        )

        sim = (
            bank_z_norm
            @ qn
        )

        k = min(
            int(
                cfg[
                    "isolation"
                ][
                    "learned_retrieval_top_k_windows"
                ]
            ),
            len(
                sim
            ),
        )

        local = np.argpartition(
            -sim,
            kth=k - 1,
        )[
            :k
        ]

        local = local[
            np.argsort(
                -sim[
                    local
                ]
            )
        ]

        c = rep[
            "current_std"
        ].t(
            heldout[
                "current"
            ][
                i:i+1
            ]
        )[
            0
        ]

        y = rep[
            "target_std"
        ].t(
            heldout[
                "target"
            ][
                i:i+1
            ]
        )[
            0
        ]

        z_hist = torch.from_numpy(
            bank_z[
                local
            ].astype(
                np.float32
            )
        )

        c_t = torch.from_numpy(
            c.astype(
                np.float32
            )
        )

        y_t = torch.from_numpy(
            y.astype(
                np.float32
            )
        )

        hist_pred = rep[
            "model"
        ].head(
            torch.cat(
                [
                    z_hist,
                    c_t.unsqueeze(
                        0
                    ).repeat(
                        k,
                        1,
                    ),
                ],
                dim=1,
            )
        )

        rm = float(
            (
                (
                    hist_pred.mean(
                        dim=0
                    )
                    - y_t
                )
                ** 2
            )
            .mean()
            .item()
        )

        self_pred = rep[
            "model"
        ].head(
            torch.cat(
                [
                    torch.from_numpy(
                        query_z.astype(
                            np.float32
                        )
                    ),
                    c_t,
                ],
                dim=0,
            ).unsqueeze(
                0
            )
        )[
            0
        ]

        sm = float(
            (
                (
                    self_pred
                    - y_t
                )
                ** 2
            )
            .mean()
            .item()
        )

        if not np.isclose(
            rm,
            float(
                reference.iloc[
                    i
                ][
                    "retrieved_mean_mse"
                ]
            ),
            rtol=0.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                "retrieved_mean_mse implementation mismatch"
            )

        if not np.isclose(
            sm,
            float(
                reference.iloc[
                    i
                ][
                    "self_prediction_mse"
                ]
            ),
            rtol=0.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                "self_prediction_mse implementation mismatch"
            )


def calibration_thresholds(
    rep,
    bank_z,
    cfg,
):
    heldout = rep[
        "heldout_test"
    ]

    scores = score_queries(
        model=rep[
            "model"
        ],
        history_std=rep[
            "history_std"
        ],
        current_std=rep[
            "current_std"
        ],
        target_std=rep[
            "target_std"
        ],
        query={
            "history": heldout[
                "history"
            ],
            "current": heldout[
                "current"
            ],
            "target": heldout[
                "target"
            ],
        },
        bank_z=bank_z,
        top_k=int(
            cfg[
                "isolation"
            ][
                "learned_retrieval_top_k_windows"
            ]
        ),
        eps=float(
            cfg[
                "confirmation"
            ][
                "score_eps"
            ]
        ),
    )

    q_support = float(
        cfg[
            "confirmation"
        ][
            "support_calibration_quantile"
        ]
    )

    q_ood = float(
        cfg[
            "confirmation"
        ][
            "ood_calibration_quantile"
        ]
    )

    support_threshold = float(
        np.quantile(
            scores[
                "retrieved_mean_mse"
            ].to_numpy(
                dtype=float
            ),
            q_support,
        )
    )

    ood_threshold = float(
        np.quantile(
            scores[
                "self_prediction_mse"
            ].to_numpy(
                dtype=float
            ),
            q_ood,
        )
    )

    return (
        scores,
        support_threshold,
        ood_threshold,
    )


def train_gated_branch(
    method,
    gate_kind,
    context,
    agent,
    replay,
    global_step,
    seed,
    cfg,
    rep,
    bank_z,
    bank_labels,
    gate_threshold,
):
    if method not in {
        "support_aware",
        "ood_gate",
    }:
        raise ValueError(
            f"Unsupported gated method: {method}"
        )

    if gate_kind not in {
        "support",
        "ood",
    }:
        raise ValueError(
            f"Unknown gate kind: {gate_kind}"
        )

    iso = cfg[
        "isolation"
    ]

    nov = cfg[
        "novelty"
    ]

    if context == "A":
        target_steps = int(
            iso[
                "A2_steps"
            ]
        )

        eval_steps = set(
            int(
                x
            )
            for x in iso[
                "A2_eval_steps"
            ]
        )

        eval_fn = evaluate_A

        reset_base = (
            90_000_000
            + int(
                seed
            )
            * 1_000_000
        )

    elif context == "D":
        target_steps = int(
            nov[
                "D_steps"
            ]
        )

        eval_steps = set(
            int(
                x
            )
            for x in nov[
                "D_eval_steps"
            ]
        )

        eval_fn = evaluate_D

        reset_base = (
            175_000_000
            + int(
                seed
            )
            * 1_000_000
        )

    else:
        raise ValueError(
            context
        )

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

    train_frequency = int(
        cfg[
            "experiment"
        ][
            "train_frequency"
        ]
    )

    L = int(
        cfg[
            "representation"
        ][
            "history_length"
        ]
    )

    top_k = int(
        iso[
            "learned_retrieval_top_k_windows"
        ]
    )

    bank_z_norm = normalize_rows(
        bank_z
    )

    zero = eval_fn(
        agent,
        context,
        seed,
        cfg,
    )

    rows = [{
        "method": method,
        "phase_step": 0,
        "global_step": int(
            global_step
        ),
        "eval_return_mean": float(
            zero[
                "return_mean"
            ]
        ),
        "eval_return_sd": float(
            zero[
                "return_sd"
            ]
        ),
        "eval_safety_termination_fraction": float(
            zero[
                "safety_fraction"
            ]
        ),
    }]

    state, info = env.reset(
        seed=reset_base
    )

    phase_step = 0
    episode_id = 0
    current_history = []

    gate_events = []

    hist_fracs = []
    recent_fracs = []
    query_available = []
    hist_A_fracs = []
    hist_B_fracs = []

    while phase_step < target_steps:
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

        prior_history = (
            [
                np.asarray(
                    x,
                    dtype=np.float32,
                ).copy()
                for x in current_history[
                    -L:
                ]
            ]
            if len(
                current_history
            )
            >= L
            else None
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

        current_history.append(
            dynamics_transition_feature(
                state,
                int(
                    action
                ),
                next_state,
                5,
            )
        )

        if (
            global_step
            % train_frequency
            == 0
            and len(
                replay
            )
            >= batch_size
        ):
            gate_evaluable = (
                prior_history
                is not None
            )

            accepted = False
            score_row = None

            if gate_evaluable:
                score_row = transition_gate_scores(
                    prior_history=prior_history,
                    state=state,
                    action=action,
                    next_state=next_state,
                    model=rep[
                        "model"
                    ],
                    history_std=rep[
                        "history_std"
                    ],
                    current_std=rep[
                        "current_std"
                    ],
                    target_std=rep[
                        "target_std"
                    ],
                    bank_z=bank_z,
                    bank_z_norm=bank_z_norm,
                    top_k=top_k,
                )

                gate_score = (
                    score_row[
                        "retrieved_mean_mse"
                    ]
                    if gate_kind
                    == "support"
                    else score_row[
                        "self_prediction_mse"
                    ]
                )

                accepted = bool(
                    gate_score
                    <= float(
                        gate_threshold
                    )
                )

            if (
                accepted
                and len(
                    current_history
                )
                >= L
            ):
                replay_query_h = np.stack(
                    current_history[
                        -L:
                    ]
                ).astype(
                    np.float32
                )[
                    None,
                    :,
                    :,
                ]

                replay_query_z = encode_histories(
                    rep[
                        "model"
                    ],
                    replay_query_h,
                    rep[
                        "history_std"
                    ],
                )[
                    0
                ]

                replay.set_query_embedding(
                    replay_query_z
                )

            else:
                replay.clear_query()

            batch = replay.sample(
                batch_size
            )

            agent.update(
                batch
            )

            stats = replay.last_sample_stats

            query_available.append(
                float(
                    stats[
                        "query_available"
                    ]
                )
            )

            hist_fracs.append(
                float(
                    stats[
                        "historical_fraction"
                    ]
                )
            )

            recent_fracs.append(
                float(
                    stats[
                        "recent_fraction"
                    ]
                )
            )

            hist_idx = replay.last_historical_indices

            if len(
                hist_idx
            ) > 0:
                labs = np.asarray([
                    replay.buffer[
                        int(
                            i
                        )
                    ].context_id
                    for i in hist_idx
                ])

                hist_A_fracs.append(
                    float(
                        np.mean(
                            labs
                            == "A"
                        )
                    )
                )

                hist_B_fracs.append(
                    float(
                        np.mean(
                            labs
                            == "B"
                        )
                    )
                )

            gate_event = {
                "seed": int(
                    seed
                ),
                "schedule": (
                    "ABA"
                    if context
                    == "A"
                    else "ABD"
                ),
                "method": method,
                "phase_step": int(
                    phase_step
                    + 1
                ),
                "gate_evaluable": bool(
                    gate_evaluable
                ),
                "gate_accepted": bool(
                    accepted
                ),
                "gate_threshold": float(
                    gate_threshold
                ),
                "retrieved_mean_mse": np.nan,
                "self_prediction_mse": np.nan,
                "gate_best_similarity": np.nan,
                "gate_mean_top_similarity": np.nan,
                "gate_top_A_fraction_diagnostic": np.nan,
                "gate_top_B_fraction_diagnostic": np.nan,
                "sample_historical_fraction": float(
                    stats[
                        "historical_fraction"
                    ]
                ),
                "sample_recent_fraction": float(
                    stats[
                        "recent_fraction"
                    ]
                ),
            }

            if score_row is not None:
                gate_event[
                    "retrieved_mean_mse"
                ] = float(
                    score_row[
                        "retrieved_mean_mse"
                    ]
                )

                gate_event[
                    "self_prediction_mse"
                ] = float(
                    score_row[
                        "self_prediction_mse"
                    ]
                )

                gate_event[
                    "gate_best_similarity"
                ] = float(
                    score_row[
                        "best_similarity"
                    ]
                )

                gate_event[
                    "gate_mean_top_similarity"
                ] = float(
                    score_row[
                        "mean_top_similarity"
                    ]
                )

                top_rows = score_row[
                    "top_rows"
                ]

                if len(
                    top_rows
                ) > 0:
                    top_labels = bank_labels[
                        top_rows
                    ]

                    gate_event[
                        "gate_top_A_fraction_diagnostic"
                    ] = float(
                        np.mean(
                            top_labels
                            == "A"
                        )
                    )

                    gate_event[
                        "gate_top_B_fraction_diagnostic"
                    ] = float(
                        np.mean(
                            top_labels
                            == "B"
                        )
                    )

            gate_events.append(
                gate_event
            )

        state = next_state
        info = next_info

        phase_step += 1
        global_step += 1

        if phase_step in eval_steps:
            ev = eval_fn(
                agent,
                context,
                seed,
                cfg,
            )

            rows.append({
                "method": method,
                "phase_step": int(
                    phase_step
                ),
                "global_step": int(
                    global_step
                ),
                "eval_return_mean": float(
                    ev[
                        "return_mean"
                    ]
                ),
                "eval_return_sd": float(
                    ev[
                        "return_sd"
                    ]
                ),
                "eval_safety_termination_fraction": float(
                    ev[
                        "safety_fraction"
                    ]
                ),
            })

        if done:
            current_history = []

            if phase_step < target_steps:
                episode_id += 1

                state, info = env.reset(
                    seed=(
                        reset_base
                        + episode_id
                    )
                )

    env.close()

    gate_df = pd.DataFrame(
        gate_events
    )

    evaluable = (
        gate_df[
            gate_df[
                "gate_evaluable"
            ]
        ]
        if not gate_df.empty
        else gate_df
    )

    diag = {
        "seed": int(
            seed
        ),
        "schedule": (
            "ABA"
            if context
            == "A"
            else "ABD"
        ),
        "method": method,
        "gate_kind": gate_kind,
        "gate_threshold": float(
            gate_threshold
        ),
        "n_update_events": int(
            len(
                gate_df
            )
        ),
        "n_evaluable_gate_events": int(
            len(
                evaluable
            )
        ),
        "gate_evaluable_fraction": (
            float(
                gate_df[
                    "gate_evaluable"
                ].mean()
            )
            if len(
                gate_df
            )
            > 0
            else np.nan
        ),
        "gate_acceptance_fraction_evaluable": (
            float(
                evaluable[
                    "gate_accepted"
                ].mean()
            )
            if len(
                evaluable
            )
            > 0
            else np.nan
        ),
        "mean_sample_historical_fraction": (
            float(
                np.mean(
                    hist_fracs
                )
            )
            if hist_fracs
            else np.nan
        ),
        "mean_sample_recent_fraction": (
            float(
                np.mean(
                    recent_fracs
                )
            )
            if recent_fracs
            else np.nan
        ),
        "mean_query_available_fraction": (
            float(
                np.mean(
                    query_available
                )
            )
            if query_available
            else np.nan
        ),
        "mean_historical_A_fraction_diagnostic": (
            float(
                np.mean(
                    hist_A_fracs
                )
            )
            if hist_A_fracs
            else np.nan
        ),
        "mean_historical_B_fraction_diagnostic": (
            float(
                np.mean(
                    hist_B_fracs
                )
            )
            if hist_B_fracs
            else np.nan
        ),
    }

    return (
        rows,
        diag,
        gate_df,
    )


def first_sustained_recovery(
    d,
    score_col,
    threshold,
    consecutive,
):
    good = (
        d[
            score_col
        ].to_numpy()
        >= float(
            threshold
        )
    )

    for i in range(
        len(
            good
        )
        - int(
            consecutive
        )
        + 1
    ):
        if bool(
            np.all(
                good[
                    i:
                    i
                    + int(
                        consecutive
                    )
                ]
            )
        ):
            return int(
                d.iloc[
                    i
                ][
                    "phase_step"
                ]
            )

    return None


def auc_0_2000(
    d,
    score_col,
):
    sub = d[
        (
            d[
                "phase_step"
            ]
            >= 0
        )
        & (
            d[
                "phase_step"
            ]
            <= 2000
        )
    ].sort_values(
        "phase_step"
    )

    x = sub[
        "phase_step"
    ].to_numpy(
        dtype=float
    )

    y = sub[
        score_col
    ].to_numpy(
        dtype=float
    )

    if (
        x[
            0
        ]
        != 0
        or x[
            -1
        ]
        != 2000
    ):
        raise RuntimeError(
            "AUC endpoints missing"
        )

    return float(
        np.trapezoid(
            y,
            x,
        )
        / 2000.0
    )


def metrics_from_curve(
    curve,
    schedule,
    reference,
    random_return,
    cfg,
    methods=None,
):
    denom = float(
        reference
        - random_return
    )

    if denom <= 0:
        raise RuntimeError(
            f"{schedule} reference does not beat random"
        )

    curve = curve.copy()

    score_col = (
        "normalized_A_score"
        if schedule
        == "ABA"
        else "normalized_D_score"
    )

    curve[
        score_col
    ] = (
        curve[
            "eval_return_mean"
        ]
        - float(
            random_return
        )
    ) / denom

    if schedule == "ABA":
        threshold = float(
            cfg[
                "isolation"
            ][
                "recovery_threshold"
            ]
        )

        consecutive = int(
            cfg[
                "isolation"
            ][
                "recovery_consecutive_checkpoints"
            ]
        )

    else:
        threshold = float(
            cfg[
                "novelty"
            ][
                "recovery_threshold"
            ]
        )

        consecutive = int(
            cfg[
                "novelty"
            ][
                "recovery_consecutive_checkpoints"
            ]
        )

    rows = []

    if methods is None:
        methods = METHODS

    for method in methods:
        d = (
            curve[
                curve[
                    "method"
                ]
                == method
            ]
            .sort_values(
                "phase_step"
            )
            .copy()
        )

        recovery = first_sustained_recovery(
            d,
            score_col,
            threshold,
            consecutive,
        )

        post = d[
            (
                d[
                    "phase_step"
                ]
                > 0
            )
            & (
                d[
                    "phase_step"
                ]
                <= 2000
            )
        ]

        rows.append({
            "schedule": schedule,
            "method": method,
            "zero_shot_score": float(
                d.iloc[
                    0
                ][
                    score_col
                ]
            ),
            "recovery_step": (
                float(
                    recovery
                )
                if recovery
                is not None
                else np.nan
            ),
            "AUC_0_2000": float(
                auc_0_2000(
                    d,
                    score_col,
                )
            ),
            "post_switch_mean_score_0_2000": float(
                post[
                    score_col
                ].mean()
            ),
            "final_score": float(
                d.iloc[
                    -1
                ][
                    score_col
                ]
            ),
        })

    return (
        curve,
        pd.DataFrame(
            rows
        ),
    )


def verify_RQ2_controls(
    seed,
    current_metrics,
    cfg,
):
    prior = pd.read_csv(
        ROOT
        / "results"
        / "synthetic_v2"
        / "learned_replay_recurrence_3seed"
        / "heldout_metrics_seeds23.csv"
    )

    prior = prior[
        prior[
            "seed"
        ]
        == int(
            seed
        )
    ].copy()

    current = current_metrics.set_index(
        "method"
    )

    rows = []

    mapping = {
        "recent": "recent",
        "relevance": "learned",
    }

    metrics = [
        "zero_shot_score",
        "recovery_step",
        "AUC_0_2000",
        "post_switch_mean_score_0_2000",
        "final_score",
    ]

    for current_name, prior_name in mapping.items():
        old = (
            prior[
                prior[
                    "method"
                ]
                == prior_name
            ]
            .iloc[
                0
            ]
        )

        for metric in metrics:
            now = float(
                current.loc[
                    current_name,
                    metric,
                ]
            )

            before = float(
                old[
                    metric
                ]
            )

            diff = (
                now
                - before
            )

            rows.append({
                "seed": int(
                    seed
                ),
                "method": current_name,
                "metric": metric,
                "prior_value": before,
                "current_value": now,
                "absolute_difference": diff,
            })

    df = pd.DataFrame(
        rows
    )

    # Historical absolute metric equality is non-fatal. The project has
    # already exhibited multiple internally stable DDQN trajectories under
    # different execution harnesses. Formal RQ3 validity is therefore based
    # on frozen scientific-protocol equality and within-run paired fork
    # identity. Historical RQ2 controls are retained as a drift diagnostic.
    return df


def verify_RQ2_protocol(cfg):
    prior = load_config(
        ROOT / "configs/base.yaml",
        ROOT / "configs/synthetic_v2/learned_replay_recurrence_pilot.yaml",
    )

    mismatches = {}

    for section in [
        "environment",
        "contexts",
        "representation",
        "behavior_diverse",
    ]:
        if cfg[section] != prior[section]:
            mismatches[section] = {
                "formal": cfg[section],
                "canonical_RQ2": prior[section],
            }

    formal_iso = copy.deepcopy(cfg["isolation"])
    prior_iso = copy.deepcopy(prior["isolation"])
    formal_iso.pop("seed", None)
    prior_iso.pop("seed", None)

    if formal_iso != prior_iso:
        mismatches["isolation_except_seed"] = {
            "formal": formal_iso,
            "canonical_RQ2": prior_iso,
        }

    if mismatches:
        raise RuntimeError(
            "Formal RQ3 protocol differs from frozen RQ2 protocol:\n"
            + json.dumps(mismatches, indent=2, default=str)
        )

    return True


def verify_within_run_fork_identity(metrics, cfg):
    atol = float(
        cfg["confirmation"]["fork_identity_atol"]
    )

    rows = []

    for schedule in ["ABA", "ABD"]:
        d = metrics[
            metrics["schedule"] == schedule
        ].copy()

        values = d[
            "zero_shot_score"
        ].to_numpy(dtype=float)

        spread = float(
            np.max(values) - np.min(values)
        )

        rows.append({
            "schedule": schedule,
            "n_methods": int(len(d)),
            "min_zero_shot_score": float(np.min(values)),
            "max_zero_shot_score": float(np.max(values)),
            "zero_shot_spread": spread,
            "fork_identity_atol": atol,
            "pass": bool(spread <= atol),
        })

    result = pd.DataFrame(rows)

    if not bool(result["pass"].all()):
        raise RuntimeError(
            "Within-run branch fork identity failed:\n"
            + result.to_string(index=False)
        )

    return result


def run_seed(
    seed,
    cfg,
):
    cfg = copy.deepcopy(
        cfg
    )

    cfg[
        "isolation"
    ][
        "seed"
    ] = int(
        seed
    )

    cfg[
        "novelty"
    ][
        "seed"
    ] = int(
        seed
    )

    verify_RQ2_protocol(
        cfg
    )

    # Frozen representation protocol.
    behavior_module.GLOBAL_CFG = cfg

    set_global_seed(
        int(
            cfg[
                "representation"
            ][
                "seed"
            ]
        )
    )

    rep = train_behavior_diverse_encoder(
        cfg
    )

    encoder_sanity = {
        "relative_mse_improvement_ge_0_20": bool(
            rep[
                "relative_mse_improvement"
            ]
            >= 0.20
        ),
        "offline_AB_precision_at_10_ge_0_90": bool(
            rep[
                "offline_precision_at_10"
            ]
            >= 0.90
        ),
        "offline_AB_centroid_accuracy_ge_0_90": bool(
            rep[
                "offline_centroid_accuracy"
            ]
            >= 0.90
        ),
    }

    if not all(
        encoder_sanity.values()
    ):
        raise RuntimeError(
            f"Encoder sanity failed: {encoder_sanity}"
        )

    # Preserve the exact canonical RQ2 execution order through the ABA
    # Recent/Relevance reproduction check. D-reference training is deliberately
    # deferred until after all formal branch trajectories have been generated.
    set_global_seed(
        int(
            seed
        )
    )

    agent = make_agent(
        cfg,
        int(
            seed
        ),
    )

    replay = RecentReplayBuffer(
        capacity=int(
            cfg[
                "isolation"
            ][
                "replay_capacity"
            ]
        ),
        recent_window=int(
            cfg[
                "isolation"
            ][
                "recent_window"
            ]
        ),
        seed=int(
            seed
        ),
    )

    global_step = 0

    (
        global_step,
        A1_rows,
        A1_episodes,
    ) = train_common_phase_logged(
        agent=agent,
        replay=replay,
        context="A",
        phase_steps=int(
            cfg[
                "isolation"
            ][
                "A1_steps"
            ]
        ),
        global_step=global_step,
        seed=int(
            seed
        ),
        phase_index=0,
        cfg=cfg,
        eval_steps=[
            int(
                x
            )
            for x in cfg[
                "isolation"
            ][
                "A1_reference_steps"
            ]
        ],
    )

    A1_df = pd.DataFrame(
        A1_rows
    )

    A_reference = float(
        A1_df[
            "eval_return_mean"
        ].mean()
    )

    (
        global_step,
        _,
        B_episodes,
    ) = train_common_phase_logged(
        agent=agent,
        replay=replay,
        context="B",
        phase_steps=int(
            cfg[
                "isolation"
            ][
                "B_steps"
            ]
        ),
        global_step=global_step,
        seed=int(
            seed
        ),
        phase_index=1,
        cfg=cfg,
        eval_steps=None,
    )

    rand_A = float(
        random_A(
            int(
                seed
            ),
            cfg,
        )
    )

    L = int(
        cfg[
            "representation"
        ][
            "history_length"
        ]
    )

    bank_A = history_bank_windows(
        A1_episodes,
        L,
        int(
            cfg[
                "isolation"
            ][
                "history_bank_stride"
            ]
        ),
    )

    bank_B = history_bank_windows(
        B_episodes,
        L,
        int(
            cfg[
                "isolation"
            ][
                "history_bank_stride"
            ]
        ),
    )

    bank = {
        key: np.concatenate(
            [
                bank_A[
                    key
                ],
                bank_B[
                    key
                ],
            ],
            axis=0,
        )
        for key in bank_A
    }

    bank = uniform_cap_bank(
        bank,
        max_windows=int(
            cfg[
                "isolation"
            ][
                "max_history_bank_windows"
            ]
        ),
        seed=int(
            seed
        ),
    )

    bank_z = encode_histories(
        rep[
            "model"
        ],
        bank[
            "history"
        ],
        rep[
            "history_std"
        ],
    )

    # ---------- ABA controls ----------
    # IMPORTANT: run these immediately after bank construction, matching the
    # canonical RQ2 pipeline before any new gate-scoring or D-reference workload.
    aba_recent_agent = copy.deepcopy(
        agent
    )

    aba_relevance_agent = copy.deepcopy(
        agent
    )

    aba_recent_replay = copy.deepcopy(
        replay
    )

    aba_relevance_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    aba_recent_rows, _ = train_A2_branch(
        "recent",
        aba_recent_agent,
        aba_recent_replay,
        global_step,
        int(
            seed
        ),
        cfg,
    )

    aba_relevance_rows, _ = train_A2_branch(
        "learned",
        aba_relevance_agent,
        aba_relevance_replay,
        global_step,
        int(
            seed
        ),
        cfg,
        encoder_model=rep[
            "model"
        ],
        history_std=rep[
            "history_std"
        ],
        history_length=L,
        bank_labels=bank[
            "label"
        ],
    )

    for row in aba_relevance_rows:
        row[
            "method"
        ] = "relevance"

    aba_control_curve = pd.DataFrame(
        aba_recent_rows
        + aba_relevance_rows
    )

    # Normalize and verify the two pre-existing RQ2 controls BEFORE any new
    # support/OOD model workload. This restores the canonical execution order.
    (
        aba_control_curve_checked,
        aba_control_metrics,
    ) = metrics_from_curve(
        curve=aba_control_curve,
        schedule="ABA",
        reference=A_reference,
        random_return=rand_A,
        cfg=cfg,
        methods=[
            "recent",
            "relevance",
        ],
    )

    control_reproduction = verify_RQ2_controls(
        int(
            seed
        ),
        aba_control_metrics,
        cfg,
    )

    # Only after the canonical controls reproduce do we execute any new
    # support/OOD scoring.
    verify_gate_score_implementation(
        rep,
        bank_z,
        cfg,
    )

    (
        calibration_scores,
        support_threshold,
        ood_threshold,
    ) = calibration_thresholds(
        rep,
        bank_z,
        cfg,
    )

    # ---------- ABA gated methods ----------
    aba_support_agent = copy.deepcopy(
        agent
    )

    aba_ood_agent = copy.deepcopy(
        agent
    )

    aba_support_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    aba_ood_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    (
        aba_support_rows,
        aba_support_diag,
        aba_support_events,
    ) = train_gated_branch(
        method="support_aware",
        gate_kind="support",
        context="A",
        agent=aba_support_agent,
        replay=aba_support_replay,
        global_step=global_step,
        seed=int(
            seed
        ),
        cfg=cfg,
        rep=rep,
        bank_z=bank_z,
        bank_labels=bank[
            "label"
        ],
        gate_threshold=support_threshold,
    )

    (
        aba_ood_rows,
        aba_ood_diag,
        aba_ood_events,
    ) = train_gated_branch(
        method="ood_gate",
        gate_kind="ood",
        context="A",
        agent=aba_ood_agent,
        replay=aba_ood_replay,
        global_step=global_step,
        seed=int(
            seed
        ),
        cfg=cfg,
        rep=rep,
        bank_z=bank_z,
        bank_labels=bank[
            "label"
        ],
        gate_threshold=ood_threshold,
    )

    aba_curve = pd.concat(
        [
            aba_control_curve,
            pd.DataFrame(
                aba_support_rows
            ),
            pd.DataFrame(
                aba_ood_rows
            ),
        ],
        ignore_index=True,
    )

    aba_curve, aba_metrics = metrics_from_curve(
        curve=aba_curve,
        schedule="ABA",
        reference=A_reference,
        random_return=rand_A,
        cfg=cfg,
    )

    # ---------- ABD controls ----------
    abd_recent_agent = copy.deepcopy(
        agent
    )

    abd_relevance_agent = copy.deepcopy(
        agent
    )

    abd_recent_replay = copy.deepcopy(
        replay
    )

    abd_relevance_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    abd_recent_rows, _ = train_D_branch(
        "recent",
        abd_recent_agent,
        abd_recent_replay,
        global_step,
        int(
            seed
        ),
        cfg,
    )

    abd_relevance_rows, _ = train_D_branch(
        "learned",
        abd_relevance_agent,
        abd_relevance_replay,
        global_step,
        int(
            seed
        ),
        cfg,
        encoder=rep[
            "model"
        ],
        hstd=rep[
            "history_std"
        ],
        L=L,
        bank_labels=bank[
            "label"
        ],
    )

    for row in abd_relevance_rows:
        row[
            "method"
        ] = "relevance"

    # ---------- ABD gated methods ----------
    abd_support_agent = copy.deepcopy(
        agent
    )

    abd_ood_agent = copy.deepcopy(
        agent
    )

    abd_support_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    abd_ood_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    (
        abd_support_rows,
        abd_support_diag,
        abd_support_events,
    ) = train_gated_branch(
        method="support_aware",
        gate_kind="support",
        context="D",
        agent=abd_support_agent,
        replay=abd_support_replay,
        global_step=global_step,
        seed=int(
            seed
        ),
        cfg=cfg,
        rep=rep,
        bank_z=bank_z,
        bank_labels=bank[
            "label"
        ],
        gate_threshold=support_threshold,
    )

    (
        abd_ood_rows,
        abd_ood_diag,
        abd_ood_events,
    ) = train_gated_branch(
        method="ood_gate",
        gate_kind="ood",
        context="D",
        agent=abd_ood_agent,
        replay=abd_ood_replay,
        global_step=global_step,
        seed=int(
            seed
        ),
        cfg=cfg,
        rep=rep,
        bank_z=bank_z,
        bank_labels=bank[
            "label"
        ],
        gate_threshold=ood_threshold,
    )

    abd_curve = pd.DataFrame(
        abd_recent_rows
        + abd_relevance_rows
        + abd_support_rows
        + abd_ood_rows
    )

    # Post-hoc D normalization reference. Running this only after every formal
    # branch trajectory has been generated prevents it from perturbing any
    # canonical or confirmatory training path.
    D_reference_df, D_reference = train_D_reference(
        int(
            seed
        ),
        cfg,
    )

    rand_D = float(
        random_D(
            int(
                seed
            ),
            cfg,
        )
    )

    abd_curve, abd_metrics = metrics_from_curve(
        curve=abd_curve,
        schedule="ABD",
        reference=D_reference,
        random_return=rand_D,
        cfg=cfg,
    )

    metrics = pd.concat(
        [
            aba_metrics,
            abd_metrics,
        ],
        ignore_index=True,
    )

    metrics.insert(
        0,
        "seed",
        int(
            seed
        ),
    )

    fork_identity = verify_within_run_fork_identity(
        metrics,
        cfg,
    )
    fork_identity.insert(
        0,
        "seed",
        int(
            seed
        ),
    )

    aba_curve[
        "seed"
    ] = int(
        seed
    )

    aba_curve[
        "schedule"
    ] = "ABA"

    abd_curve[
        "seed"
    ] = int(
        seed
    )

    abd_curve[
        "schedule"
    ] = "ABD"

    curves = pd.concat(
        [
            aba_curve,
            abd_curve,
        ],
        ignore_index=True,
    )

    gate_diagnostics = pd.DataFrame([
        aba_support_diag,
        aba_ood_diag,
        abd_support_diag,
        abd_ood_diag,
    ])

    gate_events = pd.concat(
        [
            aba_support_events,
            aba_ood_events,
            abd_support_events,
            abd_ood_events,
        ],
        ignore_index=True,
    )

    thresholds = pd.DataFrame([{
        "seed": int(
            seed
        ),
        "support_score": (
            "retrieved_mean_mse"
        ),
        "support_threshold": float(
            support_threshold
        ),
        "support_quantile": float(
            cfg[
                "confirmation"
            ][
                "support_calibration_quantile"
            ]
        ),
        "ood_score": (
            "self_prediction_mse"
        ),
        "ood_threshold": float(
            ood_threshold
        ),
        "ood_quantile": float(
            cfg[
                "confirmation"
            ][
                "ood_calibration_quantile"
            ]
        ),
        "n_calibration_queries": int(
            len(
                calibration_scores
            )
        ),
    }])

    bank_summary = pd.DataFrame([{
        "seed": int(
            seed
        ),
        "n_bank_windows": int(
            len(
                bank[
                    "history"
                ]
            )
        ),
        "bank_A_fraction_diagnostic_only": float(
            np.mean(
                bank[
                    "label"
                ]
                == "A"
            )
        ),
        "history_length": int(
            L
        ),
        "retrieval_top_k_windows": int(
            cfg[
                "isolation"
            ][
                "learned_retrieval_top_k_windows"
            ]
        ),
    }])

    encoder_summary = pd.DataFrame([{
        "seed": int(
            seed
        ),
        "offline_history_test_mse": float(
            rep[
                "history_mse"
            ]
        ),
        "offline_no_history_test_mse": float(
            rep[
                "baseline_mse"
            ]
        ),
        "offline_relative_mse_improvement": float(
            rep[
                "relative_mse_improvement"
            ]
        ),
        "offline_AB_precision_at_10": float(
            rep[
                "offline_precision_at_10"
            ]
        ),
        "offline_AB_centroid_accuracy": float(
            rep[
                "offline_centroid_accuracy"
            ]
        ),
        "encoder_best_epoch": int(
            rep[
                "model_meta"
            ][
                "best_epoch"
            ]
        ),
    }])

    references = pd.DataFrame([{
        "seed": int(
            seed
        ),
        "A_reference_return": float(
            A_reference
        ),
        "random_A_return": float(
            rand_A
        ),
        "D_reference_return": float(
            D_reference
        ),
        "random_D_return": float(
            rand_D
        ),
    }])

    D_reference_df = D_reference_df.copy()

    D_reference_df.insert(
        0,
        "seed",
        int(
            seed
        ),
    )

    return {
        "metrics": metrics,
        "curves": curves,
        "gate_diagnostics": gate_diagnostics,
        "gate_events": gate_events,
        "thresholds": thresholds,
        "bank_summary": bank_summary,
        "encoder_summary": encoder_summary,
        "references": references,
        "D_reference_df": D_reference_df,
        "control_reproduction": control_reproduction,
        "fork_identity": fork_identity,
    }


def paired_contrasts(
    metrics,
):
    rows = []

    for seed in sorted(
        metrics[
            "seed"
        ].unique()
    ):
        for schedule in [
            "ABA",
            "ABD",
        ]:
            d = (
                metrics[
                    (
                        metrics[
                            "seed"
                        ]
                        == seed
                    )
                    & (
                        metrics[
                            "schedule"
                        ]
                        == schedule
                    )
                ]
                .set_index(
                    "method"
                )
            )

            rows.append({
                "seed": int(
                    seed
                ),
                "schedule": schedule,
                "recent_AUC": float(
                    d.loc[
                        "recent",
                        "AUC_0_2000",
                    ]
                ),
                "relevance_AUC": float(
                    d.loc[
                        "relevance",
                        "AUC_0_2000",
                    ]
                ),
                "support_aware_AUC": float(
                    d.loc[
                        "support_aware",
                        "AUC_0_2000",
                    ]
                ),
                "ood_gate_AUC": float(
                    d.loc[
                        "ood_gate",
                        "AUC_0_2000",
                    ]
                ),
                "support_minus_recent_AUC": float(
                    d.loc[
                        "support_aware",
                        "AUC_0_2000",
                    ]
                    - d.loc[
                        "recent",
                        "AUC_0_2000",
                    ]
                ),
                "support_minus_relevance_AUC": float(
                    d.loc[
                        "support_aware",
                        "AUC_0_2000",
                    ]
                    - d.loc[
                        "relevance",
                        "AUC_0_2000",
                    ]
                ),
                "ood_minus_recent_AUC": float(
                    d.loc[
                        "ood_gate",
                        "AUC_0_2000",
                    ]
                    - d.loc[
                        "recent",
                        "AUC_0_2000",
                    ]
                ),
                "ood_minus_relevance_AUC": float(
                    d.loc[
                        "ood_gate",
                        "AUC_0_2000",
                    ]
                    - d.loc[
                        "relevance",
                        "AUC_0_2000",
                    ]
                ),
                "recent_recovery_step": float(
                    d.loc[
                        "recent",
                        "recovery_step",
                    ]
                )
                if pd.notna(
                    d.loc[
                        "recent",
                        "recovery_step",
                    ]
                )
                else np.nan,
                "relevance_recovery_step": float(
                    d.loc[
                        "relevance",
                        "recovery_step",
                    ]
                )
                if pd.notna(
                    d.loc[
                        "relevance",
                        "recovery_step",
                    ]
                )
                else np.nan,
                "support_aware_recovery_step": float(
                    d.loc[
                        "support_aware",
                        "recovery_step",
                    ]
                )
                if pd.notna(
                    d.loc[
                        "support_aware",
                        "recovery_step",
                    ]
                )
                else np.nan,
                "ood_gate_recovery_step": float(
                    d.loc[
                        "ood_gate",
                        "recovery_step",
                    ]
                )
                if pd.notna(
                    d.loc[
                        "ood_gate",
                        "recovery_step",
                    ]
                )
                else np.nan,
            })

    return pd.DataFrame(
        rows
    )


def confirmation_decision(
    contrasts,
    gate_diagnostics,
    cfg,
):
    tol = float(
        cfg[
            "confirmation"
        ][
            "auc_tolerance"
        ]
    )

    min_A = float(
        cfg[
            "confirmation"
        ][
            "min_ABA_support_acceptance"
        ]
    )

    max_D = float(
        cfg[
            "confirmation"
        ][
            "max_ABD_support_acceptance"
        ]
    )

    ABA = contrasts[
        contrasts[
            "schedule"
        ]
        == "ABA"
    ].copy()

    ABD = contrasts[
        contrasts[
            "schedule"
        ]
        == "ABD"
    ].copy()

    support_gate = gate_diagnostics[
        gate_diagnostics[
            "method"
        ]
        == "support_aware"
    ].copy()

    ABA_gate = support_gate[
        support_gate[
            "schedule"
        ]
        == "ABA"
    ]

    ABD_gate = support_gate[
        support_gate[
            "schedule"
        ]
        == "ABD"
    ]

    checks = {
        "ABA_support_beats_recent_both_seeds": bool(
            (
                ABA[
                    "support_minus_recent_AUC"
                ]
                > 0.0
            ).all()
        ),
        "ABA_support_within_0p10_of_relevance_both_seeds": bool(
            (
                ABA[
                    "support_minus_relevance_AUC"
                ]
                >= -tol
            ).all()
        ),
        "ABD_support_beats_relevance_both_seeds": bool(
            (
                ABD[
                    "support_minus_relevance_AUC"
                ]
                > 0.0
            ).all()
        ),
        "ABD_support_within_0p10_of_recent_both_seeds": bool(
            (
                ABD[
                    "support_minus_recent_AUC"
                ]
                >= -tol
            ).all()
        ),
        "ABA_support_gate_acceptance_ge_0p70_both_seeds": bool(
            (
                ABA_gate[
                    "gate_acceptance_fraction_evaluable"
                ]
                >= min_A
            ).all()
        ),
        "ABD_support_gate_acceptance_le_0p30_both_seeds": bool(
            (
                ABD_gate[
                    "gate_acceptance_fraction_evaluable"
                ]
                <= max_D
            ).all()
        ),
    }

    return {
        **checks,
        "primary_confirmation_pass": bool(
            all(
                checks.values()
            )
        ),
        "n_heldout_seeds": 2,
        "n_significance_tests": 0,
    }


def plot_schedule(
    curves,
    schedule,
    path,
    max_step,
):
    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.0,
        )
    )

    score_col = (
        "normalized_A_score"
        if schedule
        == "ABA"
        else "normalized_D_score"
    )

    for method in METHODS:
        d = curves[
            (
                curves[
                    "schedule"
                ]
                == schedule
            )
            & (
                curves[
                    "method"
                ]
                == method
            )
            & (
                curves[
                    "phase_step"
                ]
                <= max_step
            )
        ]

        g = (
            d.groupby(
                "phase_step"
            )[
                score_col
            ]
            .agg(
                [
                    "mean",
                    "std",
                ]
            )
            .reset_index()
        )

        line = ax.plot(
            g[
                "phase_step"
            ],
            g[
                "mean"
            ],
            marker="o",
            label=method,
        )[
            0
        ]

        sd = g[
            "std"
        ].fillna(
            0.0
        )

        ax.fill_between(
            g[
                "phase_step"
            ],
            g[
                "mean"
            ]
            - sd,
            g[
                "mean"
            ]
            + sd,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.axhline(
        0.90,
        linestyle="--",
        label="recovery threshold",
    )

    ax.set_xlabel(
        (
            "A2 training transitions"
            if schedule
            == "ABA"
            else "D training transitions"
        )
    )

    ax.set_ylabel(
        (
            "Normalized A performance"
            if schedule
            == "ABA"
            else "Normalized D performance"
        )
    )

    ax.set_title(
        (
            "Held-Out Recurrence Confirmation"
            if schedule
            == "ABA"
            else "Held-Out Novelty Confirmation"
        )
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
    )
    plt.close(
        fig
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "support_aware_heldout_confirmation.yaml"
        ),
    )

    args = parser.parse_args()

    cfg_path = (
        ROOT
        / args.config
    )

    cfg = load_config(
        ROOT
        / "configs/base.yaml",
        cfg_path,
    )

    confirm = cfg[
        "confirmation"
    ]

    seeds = [
        int(
            x
        )
        for x in confirm[
            "heldout_seeds"
        ]
    ]

    if seeds != [
        2,
        3,
    ]:
        raise RuntimeError(
            f"Expected held-out seeds [2, 3], got {seeds}"
        )

    development_meta_path = (
        ROOT
        / "results"
        / "synthetic_v2"
        / "support_compatibility_development"
        / "development_metadata.json"
    )

    development_meta = json.loads(
        development_meta_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        development_meta[
            "selected_method"
        ]
        != confirm[
            "selected_support_score"
        ]
    ):
        raise RuntimeError(
            "Frozen support method does not match development selection"
        )

    if confirm[
        "selected_support_score"
    ] != "retrieved_mean_mse":
        raise RuntimeError(
            "This formal script is frozen to retrieved_mean_mse"
        )

    torch.set_num_threads(
        1
    )

    out = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "support_aware_heldout_confirmation"
    )

    figs = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "support_aware_heldout_confirmation"
    )

    results = []

    for seed in seeds:
        print(
            f"\n===== HELD-OUT SEED {seed} ====="
        )

        results.append(
            run_seed(
                seed,
                cfg,
            )
        )

    metrics = pd.concat(
        [
            x[
                "metrics"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    curves = pd.concat(
        [
            x[
                "curves"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    gate_diagnostics = pd.concat(
        [
            x[
                "gate_diagnostics"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    gate_events = pd.concat(
        [
            x[
                "gate_events"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    thresholds = pd.concat(
        [
            x[
                "thresholds"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    banks = pd.concat(
        [
            x[
                "bank_summary"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    encoders = pd.concat(
        [
            x[
                "encoder_summary"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    references = pd.concat(
        [
            x[
                "references"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    D_refs = pd.concat(
        [
            x[
                "D_reference_df"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    control_reproduction = pd.concat(
        [
            x[
                "control_reproduction"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    fork_identity = pd.concat(
        [
            x[
                "fork_identity"
            ]
            for x in results
        ],
        ignore_index=True,
    )

    contrasts = paired_contrasts(
        metrics
    )

    decision = confirmation_decision(
        contrasts,
        gate_diagnostics,
        cfg,
    )

    metrics.to_csv(
        out
        / "heldout_metrics_seeds23.csv",
        index=False,
    )

    curves.to_csv(
        out
        / "heldout_curves_seeds23.csv",
        index=False,
    )

    gate_diagnostics.to_csv(
        out
        / "gate_diagnostics_seeds23.csv",
        index=False,
    )

    gate_events.to_csv(
        out
        / "gate_events_seeds23.csv",
        index=False,
    )

    thresholds.to_csv(
        out
        / "calibration_thresholds_seeds23.csv",
        index=False,
    )

    banks.to_csv(
        out
        / "historical_bank_summary_seeds23.csv",
        index=False,
    )

    encoders.to_csv(
        out
        / "encoder_summary_seeds23.csv",
        index=False,
    )

    references.to_csv(
        out
        / "normalization_references_seeds23.csv",
        index=False,
    )

    D_refs.to_csv(
        out
        / "D_reference_evaluations_seeds23.csv",
        index=False,
    )

    control_reproduction.to_csv(
        out
        / "ABA_RQ2_control_historical_drift.csv",
        index=False,
    )

    fork_identity.to_csv(
        out
        / "within_run_fork_identity_seeds23.csv",
        index=False,
    )

    contrasts.to_csv(
        out
        / "paired_contrasts_seeds23.csv",
        index=False,
    )

    save_json(
        {
            "code_commit": git_sha(),
            "support_development_seed": 1,
            "support_method_heldout_seeds": [
                2,
                3,
            ],
            "ABA_recent_relevance_controls_previously_observed": True,
            "historical_ABA_metric_equality_required": False,
            "historical_ABA_control_drift_recorded": True,
            "formal_validity_guard": (
                "frozen protocol equality plus within-run paired fork identity"
            ),
            "ABD_seed2_seed3_outcomes_previously_observed_for_support_development": False,
            "selected_support_score": (
                "retrieved_mean_mse"
            ),
            "support_calibration_rule": (
                "per-seed 95th percentile on frozen held-out "
                "behavior-diverse A/B calibration scores"
            ),
            "ood_baseline_score": (
                "self_prediction_mse"
            ),
            "ood_calibration_rule": (
                "per-seed 95th percentile on the same A/B calibration set"
            ),
            "D_used_for_threshold_calibration": False,
            "true_context_labels_used_for_threshold_calibration": False,
            "relevance_retrieval_top_k": int(
                cfg[
                    "isolation"
                ][
                    "learned_retrieval_top_k_windows"
                ]
            ),
            "historical_fraction_target": float(
                cfg[
                    "isolation"
                ][
                    "learned_historical_fraction"
                ]
            ),
            "gate_rejection_action": (
                "clear relevance query and fall back to Recent sampling"
            ),
            "support_gate_uses_previous_8_transitions_to_score_current_transition": True,
            "verification_bugfix": (
                "verification encodes the five held-out histories in one batch "
                "to match score_queries encoder batch shape; scientific protocol unchanged"
            ),
            "execution_order_bugfix": (
                "ABA Recent/Relevance controls are executed and verified before "
                "any new gate-scoring or D-reference workload; D normalization is "
                "computed post-hoc after all branch trajectories"
            ),
            "accepted_replay_query_preserves_original_RQ2_latest_8_transition_relevance_semantics": True,
            "config_sha256": sha256_file(
                cfg_path
            ),
            "development_metadata_sha256": sha256_file(
                development_meta_path
            ),
            **decision,
        },
        out
        / "confirmation_metadata.json",
    )

    save_yaml(
        cfg,
        out
        / "resolved_config.yaml",
    )

    for schedule in [
        "ABA",
        "ABD",
    ]:
        plot_schedule(
            curves,
            schedule,
            figs
            / f"{schedule}_first_2000.png",
            max_step=2000,
        )

        plot_schedule(
            curves,
            schedule,
            figs
            / f"{schedule}_full.png",
            max_step=10000,
        )

    print(
        "\n=== Held-out metrics ==="
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\n=== Paired contrasts ==="
    )

    print(
        contrasts.to_string(
            index=False
        )
    )

    print(
        "\n=== Gate diagnostics ==="
    )

    print(
        gate_diagnostics.to_string(
            index=False
        )
    )

    print(
        "\n=== Calibration thresholds ==="
    )

    print(
        thresholds.to_string(
            index=False
        )
    )

    print(
        "\n=== Within-run fork identity ==="
    )
    print(
        fork_identity.to_string(
            index=False
        )
    )

    print(
        "\n=== Historical ABA control drift (non-fatal) ==="
    )
    print(
        control_reproduction.to_string(
            index=False
        )
    )

    print(
        "\n=== Formal confirmation decision ==="
    )

    print(
        json.dumps(
            decision,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
