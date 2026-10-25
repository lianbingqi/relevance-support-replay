
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import deque
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
    random_D,
    train_D_branch,
    train_D_reference,
)
from scripts.run_oracle_recovery_isolation_v2 import (
    random_A,
)
from scripts.run_support_aware_heldout_confirmation_v2 import (
    metrics_from_curve,
    normalize_rows,
    train_gated_branch,
    transition_gate_scores,
    verify_RQ2_protocol,
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
    dynamics_transition_feature,
)
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


METHODS = [
    "recent",
    "relevance",
    "single_support",
    "window_support",
]


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def calibration_segment_ids(
    heldout,
):
    behavior = np.asarray(
        heldout[
            "behavior"
        ]
    )

    episode = np.asarray(
        heldout[
            "episode_id"
        ]
    )

    seg = np.zeros(
        len(
            episode
        ),
        dtype=np.int64,
    )

    current = 0

    for i in range(
        1,
        len(
            episode
        ),
    ):
        if (
            behavior[
                i
            ]
            != behavior[
                i - 1
            ]
            or episode[
                i
            ]
            < episode[
                i - 1
            ]
        ):
            current += 1

        seg[
            i
        ] = current

    return seg


def temporal_calibration(
    rep,
    bank_z,
    cfg,
):
    heldout = rep[
        "heldout_test"
    ]

    one_step = score_queries(
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
                "development"
            ][
                "score_eps"
            ]
        ),
    )

    single_threshold = float(
        np.quantile(
            one_step[
                "retrieved_mean_mse"
            ].to_numpy(
                dtype=float
            ),
            float(
                cfg[
                    "development"
                ][
                    "single_support_calibration_quantile"
                ]
            ),
        )
    )

    sample_stride = int(
        cfg[
            "representation"
        ][
            "sample_stride"
        ]
    )

    train_frequency = int(
        cfg[
            "experiment"
        ][
            "train_frequency"
        ]
    )

    if (
        train_frequency
        % sample_stride
        != 0
    ):
        raise RuntimeError(
            "Calibration sample stride must divide train frequency"
        )

    keep_every = (
        train_frequency
        // sample_stride
    )

    W = int(
        cfg[
            "development"
        ][
            "temporal_score_window_updates"
        ]
    )

    if W != int(
        cfg[
            "representation"
        ][
            "history_length"
        ]
    ):
        raise RuntimeError(
            "RQ3d freezes temporal score window to history_length"
        )

    cal = one_step.copy()

    cal[
        "behavior"
    ] = heldout[
        "behavior"
    ]

    cal[
        "episode_id"
    ] = heldout[
        "episode_id"
    ]

    cal[
        "segment_id"
    ] = calibration_segment_ids(
        heldout
    )

    rolling_rows = []

    for (
        segment_id,
        episode_id,
    ), g in cal.groupby(
        [
            "segment_id",
            "episode_id",
        ],
        sort=False,
    ):
        g = g.reset_index(
            drop=True
        )

        g = g.iloc[
            ::keep_every
        ].reset_index(
            drop=True
        )

        values = g[
            "retrieved_mean_mse"
        ].to_numpy(
            dtype=float
        )

        if len(
            values
        ) < W:
            continue

        for end in range(
            W,
            len(
                values
            )
            + 1,
        ):
            rolling_rows.append({
                "segment_id": int(
                    segment_id
                ),
                "episode_id": int(
                    episode_id
                ),
                "window_end_index": int(
                    end - 1
                ),
                "temporal_support_score": float(
                    np.mean(
                        values[
                            end - W:
                            end
                        ]
                    )
                ),
            })

    rolling = pd.DataFrame(
        rolling_rows
    )

    if rolling.empty:
        raise RuntimeError(
            "No temporal calibration windows were created"
        )

    temporal_threshold = float(
        np.quantile(
            rolling[
                "temporal_support_score"
            ].to_numpy(
                dtype=float
            ),
            float(
                cfg[
                    "development"
                ][
                    "temporal_support_calibration_quantile"
                ]
            ),
        )
    )

    return (
        one_step,
        rolling,
        single_threshold,
        temporal_threshold,
    )


def evaluate_context(
    agent,
    context,
    seed,
    cfg,
):
    if context == "A":
        from scripts.run_oracle_recovery_isolation_v2 import (
            evaluate,
        )

        return evaluate(
            agent,
            "A",
            seed,
            cfg,
        )

    if context == "D":
        from scripts.run_novelty_support_pilot_v2 import (
            evaluate,
        )

        return evaluate(
            agent,
            "D",
            seed,
            cfg,
        )

    raise ValueError(
        context
    )


def train_window_support_branch(
    context,
    agent,
    replay,
    global_step,
    seed,
    cfg,
    rep,
    bank_z,
    bank_labels,
    temporal_threshold,
):
    iso = cfg[
        "isolation"
    ]

    nov = cfg[
        "novelty"
    ]

    dev = cfg[
        "development"
    ]

    if context == "A":
        schedule = "ABA"

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

        reset_base = (
            90_000_000
            + int(
                seed
            )
            * 1_000_000
        )

    elif context == "D":
        schedule = "ABD"

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

    W = int(
        dev[
            "temporal_score_window_updates"
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

    zero = evaluate_context(
        agent,
        context,
        seed,
        cfg,
    )

    rows = [{
        "method": "window_support",
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
    score_history = deque(
        maxlen=W
    )

    gate_rows = []

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
            score_row = None

            if prior_history is not None:
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

                score_history.append(
                    float(
                        score_row[
                            "retrieved_mean_mse"
                        ]
                    )
                )

            gate_evaluable = (
                len(
                    score_history
                )
                == W
            )

            temporal_score = (
                float(
                    np.mean(
                        list(
                            score_history
                        )
                    )
                )
                if gate_evaluable
                else np.nan
            )

            accepted = bool(
                gate_evaluable
                and temporal_score
                <= float(
                    temporal_threshold
                )
            )

            if (
                accepted
                and len(
                    current_history
                )
                >= L
            ):
                query_h = np.stack(
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

                query_z = encode_histories(
                    rep[
                        "model"
                    ],
                    query_h,
                    rep[
                        "history_std"
                    ],
                )[
                    0
                ]

                replay.set_query_embedding(
                    query_z
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

            hist_idx = replay.last_historical_indices

            hist_A = np.nan
            hist_B = np.nan

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

                hist_A = float(
                    np.mean(
                        labs
                        == "A"
                    )
                )

                hist_B = float(
                    np.mean(
                        labs
                        == "B"
                    )
                )

            top_A = np.nan
            top_B = np.nan

            if (
                score_row
                is not None
            ):
                top_rows = score_row[
                    "top_rows"
                ]

                if len(
                    top_rows
                ) > 0:
                    top_labels = bank_labels[
                        top_rows
                    ]

                    top_A = float(
                        np.mean(
                            top_labels
                            == "A"
                        )
                    )

                    top_B = float(
                        np.mean(
                            top_labels
                            == "B"
                        )
                    )

            gate_rows.append({
                "seed": int(
                    seed
                ),
                "schedule": schedule,
                "phase_step": int(
                    phase_step
                    + 1
                ),
                "score_history_length": int(
                    len(
                        score_history
                    )
                ),
                "gate_evaluable": bool(
                    gate_evaluable
                ),
                "gate_accepted": bool(
                    accepted
                ),
                "one_step_score": (
                    float(
                        score_row[
                            "retrieved_mean_mse"
                        ]
                    )
                    if score_row
                    is not None
                    else np.nan
                ),
                "temporal_support_score": float(
                    temporal_score
                ),
                "temporal_threshold": float(
                    temporal_threshold
                ),
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
                "historical_A_fraction_diagnostic": hist_A,
                "historical_B_fraction_diagnostic": hist_B,
                "top50_A_fraction_diagnostic": top_A,
                "top50_B_fraction_diagnostic": top_B,
            })

        state = next_state
        info = next_info

        phase_step += 1
        global_step += 1

        if phase_step in eval_steps:
            ev = evaluate_context(
                agent,
                context,
                seed,
                cfg,
            )

            rows.append({
                "method": "window_support",
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
            score_history.clear()

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
        gate_rows
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

    diagnostic = {
        "seed": int(
            seed
        ),
        "schedule": schedule,
        "method": "window_support",
        "temporal_threshold": float(
            temporal_threshold
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
                gate_df[
                    "sample_historical_fraction"
                ].mean()
            )
            if len(
                gate_df
            )
            > 0
            else np.nan
        ),
        "mean_sample_recent_fraction": (
            float(
                gate_df[
                    "sample_recent_fraction"
                ].mean()
            )
            if len(
                gate_df
            )
            > 0
            else np.nan
        ),
        "mean_temporal_support_score_evaluable": (
            float(
                evaluable[
                    "temporal_support_score"
                ].mean()
            )
            if len(
                evaluable
            )
            > 0
            else np.nan
        ),
    }

    return (
        rows,
        diagnostic,
        gate_df,
    )


def fork_identity_table(
    metrics,
    seed,
    cfg,
):
    atol = float(
        cfg[
            "development"
        ][
            "fork_identity_atol"
        ]
    )

    rows = []

    for schedule in [
        "ABA",
        "ABD",
    ]:
        d = metrics[
            metrics[
                "schedule"
            ]
            == schedule
        ]

        z = d[
            "zero_shot_score"
        ].to_numpy(
            dtype=float
        )

        spread = float(
            np.max(
                z
            )
            - np.min(
                z
            )
        )

        rows.append({
            "seed": int(
                seed
            ),
            "schedule": schedule,
            "n_methods": int(
                len(
                    z
                )
            ),
            "zero_shot_spread": spread,
            "atol": atol,
            "pass": bool(
                spread
                <= atol
            ),
        })

    out = pd.DataFrame(
        rows
    )

    if not bool(
        out[
            "pass"
        ].all()
    ):
        raise RuntimeError(
            "Within-run fork identity failed:\n"
            + out.to_string(
                index=False
            )
        )

    return out


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

    sanity = {
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
        sanity.values()
    ):
        raise RuntimeError(
            f"Encoder sanity failed: {sanity}"
        )

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

    # Create all branch objects from the exact same post-B fork.
    aba_recent_agent = copy.deepcopy(
        agent
    )

    aba_relevance_agent = copy.deepcopy(
        agent
    )

    aba_single_agent = copy.deepcopy(
        agent
    )

    aba_window_agent = copy.deepcopy(
        agent
    )

    abd_recent_agent = copy.deepcopy(
        agent
    )

    abd_relevance_agent = copy.deepcopy(
        agent
    )

    abd_single_agent = copy.deepcopy(
        agent
    )

    abd_window_agent = copy.deepcopy(
        agent
    )

    aba_recent_replay = copy.deepcopy(
        replay
    )

    abd_recent_replay = copy.deepcopy(
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

    aba_single_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    aba_window_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
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

    abd_single_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    abd_window_replay = make_learned_from_recent(
        replay,
        bank,
        bank_z,
        int(
            seed
        ),
        cfg,
    )

    # Run parent strategies before any new calibration workload.
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

    (
        one_step_cal,
        temporal_cal,
        single_threshold,
        temporal_threshold,
    ) = temporal_calibration(
        rep,
        bank_z,
        cfg,
    )

    # First-generation single-step Support, now only a development comparator.
    (
        aba_single_rows,
        aba_single_diag,
        aba_single_events,
    ) = train_gated_branch(
        method="support_aware",
        gate_kind="support",
        context="A",
        agent=aba_single_agent,
        replay=aba_single_replay,
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
        gate_threshold=single_threshold,
    )

    for row in aba_single_rows:
        row[
            "method"
        ] = "single_support"

    aba_single_diag[
        "method"
    ] = "single_support"

    if not aba_single_events.empty:
        aba_single_events = (
            aba_single_events.copy()
        )

        aba_single_events[
            "method"
        ] = "single_support"

    (
        abd_single_rows,
        abd_single_diag,
        abd_single_events,
    ) = train_gated_branch(
        method="support_aware",
        gate_kind="support",
        context="D",
        agent=abd_single_agent,
        replay=abd_single_replay,
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
        gate_threshold=single_threshold,
    )

    for row in abd_single_rows:
        row[
            "method"
        ] = "single_support"

    abd_single_diag[
        "method"
    ] = "single_support"

    if not abd_single_events.empty:
        abd_single_events = (
            abd_single_events.copy()
        )

        abd_single_events[
            "method"
        ] = "single_support"

    (
        aba_window_rows,
        aba_window_diag,
        aba_window_events,
    ) = train_window_support_branch(
        context="A",
        agent=aba_window_agent,
        replay=aba_window_replay,
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
        temporal_threshold=temporal_threshold,
    )

    (
        abd_window_rows,
        abd_window_diag,
        abd_window_events,
    ) = train_window_support_branch(
        context="D",
        agent=abd_window_agent,
        replay=abd_window_replay,
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
        temporal_threshold=temporal_threshold,
    )

    # Post-hoc D normalization only after every branch trajectory exists.
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

    aba_curve = pd.DataFrame(
        aba_recent_rows
        + aba_relevance_rows
        + aba_single_rows
        + aba_window_rows
    )

    abd_curve = pd.DataFrame(
        abd_recent_rows
        + abd_relevance_rows
        + abd_single_rows
        + abd_window_rows
    )

    (
        aba_curve,
        aba_metrics,
    ) = metrics_from_curve(
        curve=aba_curve,
        schedule="ABA",
        reference=A_reference,
        random_return=rand_A,
        cfg=cfg,
        methods=METHODS,
    )

    (
        abd_curve,
        abd_metrics,
    ) = metrics_from_curve(
        curve=abd_curve,
        schedule="ABD",
        reference=D_reference,
        random_return=rand_D,
        cfg=cfg,
        methods=METHODS,
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

    fork_identity = fork_identity_table(
        metrics,
        int(
            seed
        ),
        cfg,
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

    diagnostics = pd.DataFrame([
        aba_single_diag,
        abd_single_diag,
        aba_window_diag,
        abd_window_diag,
    ])

    gate_events = pd.concat(
        [
            aba_single_events,
            abd_single_events,
            aba_window_events,
            abd_window_events,
        ],
        ignore_index=True,
        sort=False,
    )

    thresholds = pd.DataFrame([{
        "seed": int(
            seed
        ),
        "single_support_threshold": float(
            single_threshold
        ),
        "single_support_quantile": float(
            cfg[
                "development"
            ][
                "single_support_calibration_quantile"
            ]
        ),
        "temporal_support_threshold": float(
            temporal_threshold
        ),
        "temporal_support_quantile": float(
            cfg[
                "development"
            ][
                "temporal_support_calibration_quantile"
            ]
        ),
        "temporal_score_window_updates": int(
            cfg[
                "development"
            ][
                "temporal_score_window_updates"
            ]
        ),
        "n_one_step_calibration_queries": int(
            len(
                one_step_cal
            )
        ),
        "n_temporal_calibration_windows": int(
            len(
                temporal_cal
            )
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

    D_reference_df = (
        D_reference_df.copy()
    )

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
        "diagnostics": diagnostics,
        "gate_events": gate_events,
        "thresholds": thresholds,
        "references": references,
        "D_reference_df": D_reference_df,
        "fork_identity": fork_identity,
    }


def regret_table(
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

            recent = float(
                d.loc[
                    "recent",
                    "AUC_0_2000",
                ]
            )

            relevance = float(
                d.loc[
                    "relevance",
                    "AUC_0_2000",
                ]
            )

            single = float(
                d.loc[
                    "single_support",
                    "AUC_0_2000",
                ]
            )

            window = float(
                d.loc[
                    "window_support",
                    "AUC_0_2000",
                ]
            )

            best_parent = max(
                recent,
                relevance,
            )

            rows.append({
                "seed": int(
                    seed
                ),
                "schedule": schedule,
                "recent_AUC": recent,
                "relevance_AUC": relevance,
                "best_parent_AUC": best_parent,
                "single_support_AUC": single,
                "window_support_AUC": window,
                "single_support_regret": float(
                    best_parent
                    - single
                ),
                "window_support_regret": float(
                    best_parent
                    - window
                ),
                "window_minus_single_AUC": float(
                    window
                    - single
                ),
            })

    return pd.DataFrame(
        rows
    )


def development_decision(
    regrets,
    cfg,
):
    dev = cfg[
        "development"
    ]

    tol = float(
        dev[
            "max_regret_to_best_parent"
        ]
    )

    mean_window = float(
        regrets[
            "window_support_regret"
        ].mean()
    )

    mean_single = float(
        regrets[
            "single_support_regret"
        ].mean()
    )

    worst_window = float(
        regrets[
            "window_support_regret"
        ].max()
    )

    worst_single = float(
        regrets[
            "single_support_regret"
        ].max()
    )

    checks = {
        "all_six_window_regret_le_0p10": bool(
            (
                regrets[
                    "window_support_regret"
                ]
                <= tol
            ).all()
        ),
        "mean_window_regret_lower_than_single": bool(
            mean_window
            < mean_single
        ),
        "worst_window_regret_lower_than_single": bool(
            worst_window
            < worst_single
        ),
    }

    return {
        **checks,
        "mean_window_regret": mean_window,
        "mean_single_regret": mean_single,
        "worst_window_regret": worst_window,
        "worst_single_regret": worst_single,
        "eligible_to_freeze_for_seeds_4_5_6": bool(
            all(
                checks.values()
            )
        ),
    }


def plot_development(
    curves,
    schedule,
    path,
):
    score_col = (
        "normalized_A_score"
        if schedule
        == "ABA"
        else "normalized_D_score"
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.0,
        )
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
                <= 2000
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
        f"Temporal Support Development — {schedule}"
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
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "temporal_support_development.yaml"
        ),
    )

    args = ap.parse_args()

    cfg = load_config(
        ROOT
        / "configs/base.yaml",
        ROOT
        / args.config,
    )

    seeds = [
        int(
            x
        )
        for x in cfg[
            "development"
        ][
            "development_seeds"
        ]
    ]

    reserved = [
        int(
            x
        )
        for x in cfg[
            "development"
        ][
            "reserved_confirmation_seeds"
        ]
    ]

    if seeds != [
        1,
        2,
        3,
    ]:
        raise RuntimeError(
            f"Development seeds must be [1,2,3], got {seeds}"
        )

    if reserved != [
        4,
        5,
        6,
    ]:
        raise RuntimeError(
            f"Reserved seeds must be [4,5,6], got {reserved}"
        )

    if set(
        seeds
    ) & set(
        reserved
    ):
        raise RuntimeError(
            "Development and confirmation seeds overlap"
        )

    torch.set_num_threads(
        1
    )

    out = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "temporal_support_development"
    )

    figs = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "temporal_support_development"
    )

    runs = []

    for seed in seeds:
        print(
            f"\n===== DEVELOPMENT SEED {seed} ====="
        )

        runs.append(
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
            for x in runs
        ],
        ignore_index=True,
    )

    curves = pd.concat(
        [
            x[
                "curves"
            ]
            for x in runs
        ],
        ignore_index=True,
    )

    diagnostics = pd.concat(
        [
            x[
                "diagnostics"
            ]
            for x in runs
        ],
        ignore_index=True,
    )

    gate_events = pd.concat(
        [
            x[
                "gate_events"
            ]
            for x in runs
        ],
        ignore_index=True,
        sort=False,
    )

    thresholds = pd.concat(
        [
            x[
                "thresholds"
            ]
            for x in runs
        ],
        ignore_index=True,
    )

    references = pd.concat(
        [
            x[
                "references"
            ]
            for x in runs
        ],
        ignore_index=True,
    )

    D_refs = pd.concat(
        [
            x[
                "D_reference_df"
            ]
            for x in runs
        ],
        ignore_index=True,
    )

    fork_identity = pd.concat(
        [
            x[
                "fork_identity"
            ]
            for x in runs
        ],
        ignore_index=True,
    )

    regrets = regret_table(
        metrics
    )

    decision = development_decision(
        regrets,
        cfg,
    )

    metrics.to_csv(
        out
        / "development_metrics_seeds123.csv",
        index=False,
    )

    curves.to_csv(
        out
        / "development_curves_seeds123.csv",
        index=False,
    )

    diagnostics.to_csv(
        out
        / "gate_diagnostics_seeds123.csv",
        index=False,
    )

    gate_events.to_csv(
        out
        / "gate_events_seeds123.csv",
        index=False,
    )

    thresholds.to_csv(
        out
        / "calibration_thresholds_seeds123.csv",
        index=False,
    )

    references.to_csv(
        out
        / "normalization_references_seeds123.csv",
        index=False,
    )

    D_refs.to_csv(
        out
        / "D_reference_evaluations_seeds123.csv",
        index=False,
    )

    fork_identity.to_csv(
        out
        / "within_run_fork_identity_seeds123.csv",
        index=False,
    )

    regrets.to_csv(
        out
        / "regret_to_best_parent_seeds123.csv",
        index=False,
    )

    save_json(
        {
            "code_commit": git_sha(),
            "development_seeds": [
                1,
                2,
                3,
            ],
            "reserved_confirmation_seeds_not_run": [
                4,
                5,
                6,
            ],
            "rq3c_primary_confirmation_passed": False,
            "rq3d_target": (
                "historical reuse support, not novelty detection"
            ),
            "new_method": (
                "mean of last 8 one-step-ahead retrieved_mean_mse scores"
            ),
            "temporal_window_tuned": False,
            "temporal_window_equals_history_length": True,
            "D_used_for_threshold_calibration": False,
            "true_context_labels_used_for_threshold_calibration": False,
            "threshold_quantile": float(
                cfg[
                    "development"
                ][
                    "temporal_support_calibration_quantile"
                ]
            ),
            "development_success_rule": (
                "all six seed-by-schedule window regrets <=0.10 and both "
                "mean and worst-case regret improve over single-step support"
            ),
            **decision,
        },
        out
        / "development_metadata.json",
    )

    save_yaml(
        cfg,
        out
        / "resolved_config.yaml",
    )

    plot_development(
        curves,
        "ABA",
        figs
        / "ABA_first_2000.png",
    )

    plot_development(
        curves,
        "ABD",
        figs
        / "ABD_first_2000.png",
    )

    print(
        "\n=== Development metrics ==="
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print(
        "\n=== Regret to better parent ==="
    )

    print(
        regrets.to_string(
            index=False
        )
    )

    print(
        "\n=== Gate diagnostics ==="
    )

    print(
        diagnostics.to_string(
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
        "\n=== Development decision ==="
    )

    print(
        json.dumps(
            decision,
            indent=2,
        )
    )

    if decision[
        "eligible_to_freeze_for_seeds_4_5_6"
    ]:
        print(
            "\nELIGIBLE TO FREEZE. Do not run seeds 4/5/6 until this "
            "development result has been reviewed."
        )
    else:
        print(
            "\nNOT ELIGIBLE TO FREEZE. Do not run seeds 4/5/6."
        )


if __name__ == "__main__":
    main()
