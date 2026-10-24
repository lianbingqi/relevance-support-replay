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

import scripts.run_behavior_diverse_retrieval_pilot_v2 as behavior_module
from scripts.run_behavior_diverse_retrieval_pilot_v2 import (
    train_behavior_diverse_encoder,
)
from scripts.run_learned_relevance_retrieval_pilot_v2 import (
    encode_histories,
)
from scripts.run_oracle_recovery_isolation_v2 import (
    auc_0_2000,
    evaluate,
    first_sustained_recovery,
    make_oracle_from_recent,
    random_A,
)
from scripts.train_stationary_v2 import (
    epsilon_at,
    make_agent,
    make_env,
)
from src.replay.recent import RecentReplayBuffer
from src.replay.relevance import LearnedRelevantReplayBuffer
from src.representation.history import dynamics_transition_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def finish_logged_episode(
    episodes,
    states,
    actions,
    replay_indices,
    context,
    episode_id,
    start_phase_step,
):
    if len(actions) == 0:
        return

    episodes.append({
        "episode_id": int(episode_id),
        "context": context,
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "replay_indices": np.asarray(replay_indices, dtype=np.int64),
        "start_phase_step": int(start_phase_step),
    })


def train_common_phase_logged(
    agent,
    replay,
    context,
    phase_steps,
    global_step,
    seed,
    phase_index,
    cfg,
    eval_steps=None,
):
    """Match the frozen Oracle common-phase semantics, adding logging only."""
    env = make_env(cfg, context)

    learning_starts = int(cfg["experiment"]["learning_starts"])
    train_frequency = int(cfg["experiment"]["train_frequency"])
    batch_size = int(cfg["agent"]["batch_size"])

    state, info = env.reset(
        seed=80_000_000 + seed * 1_000_000 + phase_index * 100_000
    )

    eval_steps = set(eval_steps or [])
    eval_rows = []

    phase_step = 0
    episode_id = 0
    episodes = []

    episode_states = [np.asarray(state, dtype=np.float32)]
    episode_actions = []
    episode_replay_indices = []
    episode_start = 0

    while phase_step < phase_steps:
        action = agent.act(
            state,
            epsilon=epsilon_at(global_step, cfg),
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(action)

        done = bool(terminated or truncated)

        if len(replay) >= replay.capacity:
            raise RuntimeError(
                "Pilot assumes no replay-capacity wrap before/during A2"
            )

        replay_index = len(replay)

        replay.add(
            state,
            action,
            reward,
            next_state,
            terminated,
            context_id=context,
            context_vector=info["context_vector"],
        )

        if (
            global_step >= learning_starts
            and global_step % train_frequency == 0
            and len(replay) >= batch_size
        ):
            agent.update(replay.sample(batch_size))

        episode_actions.append(int(action))
        episode_replay_indices.append(int(replay_index))
        episode_states.append(
            np.asarray(next_state, dtype=np.float32)
        )

        state = next_state
        info = next_info

        phase_step += 1
        global_step += 1

        if phase_step in eval_steps:
            stats = evaluate(agent, "A", seed, cfg)
            eval_rows.append({
                "phase_step": phase_step,
                "global_step": global_step,
                "eval_return_mean": stats["return_mean"],
                "eval_return_sd": stats["return_sd"],
            })

        if done:
            finish_logged_episode(
                episodes,
                episode_states,
                episode_actions,
                episode_replay_indices,
                context,
                episode_id,
                episode_start,
            )

            episode_id += 1
            episode_states = []
            episode_actions = []
            episode_replay_indices = []

            if phase_step < phase_steps:
                state, info = env.reset(
                    seed=(
                        80_000_000
                        + seed * 1_000_000
                        + phase_index * 100_000
                        + episode_id
                    )
                )

                episode_states = [
                    np.asarray(state, dtype=np.float32)
                ]
                episode_start = phase_step

    if len(episode_actions) > 0:
        finish_logged_episode(
            episodes,
            episode_states,
            episode_actions,
            episode_replay_indices,
            context,
            episode_id,
            episode_start,
        )

    env.close()

    logged = int(sum(len(e["actions"]) for e in episodes))
    if logged != int(phase_steps):
        raise RuntimeError(
            f"{context} logged {logged} transitions; expected {phase_steps}"
        )

    return global_step, eval_rows, episodes


def history_bank_windows(
    episodes,
    history_length,
    stride,
):
    histories = []
    replay_windows = []
    labels = []
    phase_steps = []

    for episode in episodes:
        states = episode["states"]
        actions = episode["actions"]
        replay_indices = episode["replay_indices"]
        T = len(actions)

        if T < history_length:
            continue

        features = [
            dynamics_transition_feature(
                states[t],
                int(actions[t]),
                states[t + 1],
                5,
            )
            for t in range(T)
        ]

        for end in range(history_length, T + 1, stride):
            histories.append(
                np.stack(
                    features[end-history_length:end]
                ).astype(np.float32)
            )

            replay_windows.append(
                replay_indices[
                    end-history_length:end
                ].astype(np.int64)
            )

            labels.append(episode["context"])

            phase_steps.append(
                int(
                    episode["start_phase_step"]
                    + end
                )
            )

    if not histories:
        raise RuntimeError("No historical bank windows were created")

    return {
        "history": np.stack(histories).astype(np.float32),
        "replay_indices": np.stack(replay_windows).astype(np.int64),
        "label": np.asarray(labels),
        "phase_step": np.asarray(phase_steps, dtype=np.int64),
    }


def uniform_cap_bank(bank, max_windows, seed):
    n = len(bank["history"])

    if n <= int(max_windows):
        keep = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        keep = np.sort(
            rng.choice(
                n,
                size=int(max_windows),
                replace=False,
            )
        )

    return {
        key: value[keep]
        for key, value in bank.items()
    }


def make_learned_from_recent(
    recent_replay,
    bank,
    bank_z,
    seed,
    cfg,
):
    iso = cfg["isolation"]

    learned = LearnedRelevantReplayBuffer(
        capacity=int(iso["replay_capacity"]),
        recent_window=int(iso["recent_window"]),
        historical_fraction=float(
            iso["learned_historical_fraction"]
        ),
        retrieval_top_k_windows=int(
            iso["learned_retrieval_top_k_windows"]
        ),
        seed=seed,
    )

    learned.buffer = copy.deepcopy(recent_replay.buffer)
    learned.rng.bit_generator.state = copy.deepcopy(
        recent_replay.rng.bit_generator.state
    )

    learned.set_historical_bank(
        bank["replay_indices"],
        bank_z,
    )

    return learned


def train_A2_branch(
    method,
    agent,
    replay,
    global_step,
    seed,
    cfg,
    encoder_model=None,
    history_std=None,
    history_length=None,
    bank_labels=None,
):
    env = make_env(cfg, "A")

    batch_size = int(cfg["agent"]["batch_size"])
    train_frequency = int(cfg["experiment"]["train_frequency"])
    target_steps = int(cfg["isolation"]["A2_steps"])
    eval_steps = set(
        int(x)
        for x in cfg["isolation"]["A2_eval_steps"]
    )

    if method == "oracle":
        replay.set_current_context("A")

    zero = evaluate(agent, "A", seed, cfg)

    rows = [{
        "method": method,
        "phase_step": 0,
        "global_step": global_step,
        "eval_return_mean": zero["return_mean"],
        "eval_return_sd": zero["return_sd"],
        "eval_safety_termination_fraction": zero["safety_fraction"],
    }]

    state, info = env.reset(
        seed=90_000_000 + seed * 1_000_000
    )

    phase_step = 0
    episode_id = 0
    current_history = []

    oracle_hist_fracs = []
    oracle_recent_fracs = []

    learned_query_available = []
    learned_hist_fracs = []
    learned_recent_fracs = []
    learned_hist_A_fracs = []
    learned_top_A_fracs = []
    learned_best_sims = []
    learned_candidate_sizes = []

    while phase_step < target_steps:
        action = agent.act(
            state,
            epsilon=epsilon_at(global_step, cfg),
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(action)

        done = bool(terminated or truncated)

        replay.add(
            state,
            action,
            reward,
            next_state,
            terminated,
            context_id="A",
            context_vector=info["context_vector"],
        )

        if method == "learned":
            current_history.append(
                dynamics_transition_feature(
                    state,
                    int(action),
                    next_state,
                    5,
                )
            )

        if (
            global_step % train_frequency == 0
            and len(replay) >= batch_size
        ):
            if method == "oracle":
                replay.set_current_context("A")

            if method == "learned":
                if len(current_history) >= int(history_length):
                    query_h = np.stack(
                        current_history[-int(history_length):]
                    ).astype(np.float32)[None, :, :]

                    query_z = encode_histories(
                        encoder_model,
                        query_h,
                        history_std,
                    )[0]

                    replay.set_query_embedding(query_z)
                else:
                    replay.clear_query()

            batch = replay.sample(batch_size)
            agent.update(batch)

            if method == "oracle":
                oracle_hist_fracs.append(
                    replay.last_sample_stats[
                        "historical_relevant_fraction"
                    ]
                )
                oracle_recent_fracs.append(
                    replay.last_sample_stats[
                        "recent_fraction"
                    ]
                )

            if method == "learned":
                stats = replay.last_sample_stats

                learned_query_available.append(
                    float(stats["query_available"])
                )
                learned_hist_fracs.append(
                    float(stats["historical_fraction"])
                )
                learned_recent_fracs.append(
                    float(stats["recent_fraction"])
                )
                learned_best_sims.append(
                    float(stats["best_similarity"])
                    if np.isfinite(stats["best_similarity"])
                    else np.nan
                )
                learned_candidate_sizes.append(
                    float(
                        stats["candidate_historical_transitions"]
                    )
                )

                hist_idx = replay.last_historical_indices
                if len(hist_idx) > 0:
                    learned_hist_A_fracs.append(
                        float(
                            np.mean([
                                replay.buffer[int(i)].context_id == "A"
                                for i in hist_idx
                            ])
                        )
                    )

                top_rows = replay.last_top_bank_rows
                if bank_labels is not None and len(top_rows) > 0:
                    # Diagnostic only. These labels never enter sampling.
                    learned_top_A_fracs.append(
                        float(
                            np.mean(
                                bank_labels[top_rows] == "A"
                            )
                        )
                    )

        state = next_state
        info = next_info
        phase_step += 1
        global_step += 1

        if phase_step in eval_steps:
            stats = evaluate(agent, "A", seed, cfg)
            rows.append({
                "method": method,
                "phase_step": phase_step,
                "global_step": global_step,
                "eval_return_mean": stats["return_mean"],
                "eval_return_sd": stats["return_sd"],
                "eval_safety_termination_fraction": stats["safety_fraction"],
            })

        if done:
            current_history = []

            if phase_step < target_steps:
                episode_id += 1
                state, info = env.reset(
                    seed=(
                        90_000_000
                        + seed * 1_000_000
                        + episode_id
                    )
                )

    env.close()

    diag = {
        "method": method,
        "mean_oracle_historical_fraction": (
            float(np.mean(oracle_hist_fracs))
            if oracle_hist_fracs else np.nan
        ),
        "mean_oracle_recent_fraction": (
            float(np.mean(oracle_recent_fracs))
            if oracle_recent_fracs else np.nan
        ),
        "learned_query_available_update_fraction": (
            float(np.mean(learned_query_available))
            if learned_query_available else np.nan
        ),
        "mean_learned_historical_fraction": (
            float(np.mean(learned_hist_fracs))
            if learned_hist_fracs else np.nan
        ),
        "mean_learned_recent_fraction": (
            float(np.mean(learned_recent_fracs))
            if learned_recent_fracs else np.nan
        ),
        "mean_learned_historical_A_fraction_diagnostic": (
            float(np.mean(learned_hist_A_fracs))
            if learned_hist_A_fracs else np.nan
        ),
        "mean_learned_top50_A_window_fraction_diagnostic": (
            float(np.mean(learned_top_A_fracs))
            if learned_top_A_fracs else np.nan
        ),
        "mean_learned_best_similarity": (
            float(np.nanmean(learned_best_sims))
            if learned_best_sims else np.nan
        ),
        "mean_learned_candidate_historical_transitions": (
            float(np.mean(learned_candidate_sizes))
            if learned_candidate_sizes else np.nan
        ),
    }

    return rows, diag


def metrics_from_curve(
    curve,
    a1_reference,
    random_A_return,
    cfg,
):
    denom = a1_reference - random_A_return
    if denom <= 0:
        raise RuntimeError("A1 reference does not beat random A")

    curve = curve.copy()
    curve["normalized_A_score"] = (
        curve["eval_return_mean"] - random_A_return
    ) / denom

    threshold = float(
        cfg["isolation"]["recovery_threshold"]
    )
    consecutive = int(
        cfg["isolation"]["recovery_consecutive_checkpoints"]
    )

    rows = []

    for method in ["recent", "learned", "oracle"]:
        d = (
            curve[curve["method"] == method]
            .sort_values("phase_step")
            .copy()
        )

        recovery = first_sustained_recovery(
            d,
            threshold,
            consecutive,
        )

        post = d[
            (d["phase_step"] > 0)
            & (d["phase_step"] <= 2000)
        ]

        rows.append({
            "method": method,
            "zero_shot_score": float(
                d.iloc[0]["normalized_A_score"]
            ),
            "recovery_step": (
                recovery if recovery is not None else np.nan
            ),
            "AUC_0_2000": float(auc_0_2000(d)),
            "post_switch_mean_score_0_2000": float(
                post["normalized_A_score"].mean()
            ),
            "final_score": float(
                d.iloc[-1]["normalized_A_score"]
            ),
        })

    return curve, pd.DataFrame(rows)


def validate_oracle_protocol_and_measure_drift(cfg, metrics):
    prior_cfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / "configs/synthetic_v2/oracle_recovery_isolation.yaml",
    )

    keys = [
        "A1_steps",
        "B_steps",
        "A2_steps",
        "replay_capacity",
        "recent_window",
        "oracle_historical_fraction",
        "eval_episodes",
        "A1_reference_steps",
        "A2_eval_steps",
        "recovery_threshold",
        "recovery_consecutive_checkpoints",
        "auc_window_steps",
        "random_baseline_episodes",
    ]

    mismatches = {}

    for key in keys:
        current_value = cfg["isolation"][key]
        prior_value = prior_cfg["isolation"][key]

        if current_value != prior_value:
            mismatches[key] = {
                "current": current_value,
                "prior": prior_value,
            }

    if mismatches:
        raise RuntimeError(
            "Current Oracle control protocol differs from the prior frozen "
            f"seed-1 protocol: {mismatches}"
        )

    prior = pd.read_csv(
        ROOT
        / "results"
        / "synthetic_v2"
        / "oracle_recovery_isolation"
        / "isolation_metrics.csv"
    ).set_index("method")

    current = metrics.set_index("method")

    rows = []
    cols = [
        "zero_shot_score",
        "recovery_step",
        "AUC_0_2000",
        "post_switch_mean_score_0_2000",
        "final_score",
    ]

    for method in ["recent", "oracle"]:
        for col in cols:
            a = float(current.loc[method, col])
            b = float(prior.loc[method, col])

            rows.append({
                "method": method,
                "metric": col,
                "historical_value": b,
                "current_value": a,
                "absolute_difference": a - b,
            })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "learned_replay_recurrence_pilot.yaml"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / args.config,
    )

    iso = cfg["isolation"]
    seed = int(iso["seed"])

    torch.set_num_threads(1)

    # 1. Independently retrain the unchanged 05c representation protocol.
    behavior_module.GLOBAL_CFG = cfg
    set_global_seed(
        int(cfg["representation"]["seed"])
    )
    rep = train_behavior_diverse_encoder(cfg)

    encoder_model = rep["model"]
    history_std = rep["history_std"]
    history_length = int(
        cfg["representation"]["history_length"]
    )

    prior_offline = pd.read_csv(
        ROOT
        / "results"
        / "synthetic_v2"
        / "behavior_diverse_retrieval_pilot"
        / "offline_summary.csv"
    ).iloc[0]

    offline_checks = {
        "offline_history_test_mse": rep["history_mse"],
        "offline_no_history_test_mse": rep["baseline_mse"],
        "offline_relative_mse_improvement": rep[
            "relative_mse_improvement"
        ],
        "offline_AB_precision_at_10": rep[
            "offline_precision_at_10"
        ],
        "offline_AB_centroid_accuracy": rep[
            "offline_centroid_accuracy"
        ],
    }

    offline_drift_rows = []

    for key, current in offline_checks.items():
        prior = float(prior_offline[key])
        current = float(current)

        offline_drift_rows.append({
            "metric": key,
            "historical_value": prior,
            "current_value": current,
            "absolute_difference": current - prior,
            "relative_difference": (
                (current - prior) / abs(prior)
                if prior != 0.0
                else np.nan
            ),
        })

    # Cross-runtime neural-network training need not reproduce historical
    # floating-point metrics exactly. Guard the established scientific
    # capability while recording historical drift.
    encoder_sanity = {
        "relative_mse_improvement_ge_0_20": bool(
            rep["relative_mse_improvement"] >= 0.20
        ),
        "offline_AB_precision_at_10_ge_0_90": bool(
            rep["offline_precision_at_10"] >= 0.90
        ),
        "offline_AB_centroid_accuracy_ge_0_90": bool(
            rep["offline_centroid_accuracy"] >= 0.90
        ),
    }

    if not all(encoder_sanity.values()):
        raise RuntimeError(
            "05c encoder no longer satisfies established representation "
            f"sanity levels: {encoder_sanity}"
        )

    # 2. Common A1 -> B trajectory with the frozen Oracle protocol semantics.
    set_global_seed(seed)

    agent = make_agent(cfg, seed)

    replay = RecentReplayBuffer(
        capacity=int(iso["replay_capacity"]),
        recent_window=int(iso["recent_window"]),
        seed=seed,
    )

    global_step = 0

    global_step, a1_rows, A1_episodes = train_common_phase_logged(
        agent=agent,
        replay=replay,
        context="A",
        phase_steps=int(iso["A1_steps"]),
        global_step=global_step,
        seed=seed,
        phase_index=0,
        cfg=cfg,
        eval_steps=[
            int(x) for x in iso["A1_reference_steps"]
        ],
    )

    a1_df = pd.DataFrame(a1_rows)
    a1_reference = float(
        a1_df["eval_return_mean"].mean()
    )

    global_step, _, B_episodes = train_common_phase_logged(
        agent=agent,
        replay=replay,
        context="B",
        phase_steps=int(iso["B_steps"]),
        global_step=global_step,
        seed=seed,
        phase_index=1,
        cfg=cfg,
    )

    B_final = float(
        evaluate(agent, "B", seed, cfg)["return_mean"]
    )
    rand_A = float(random_A(seed, cfg))

    # 3. Build a label-free historical bank from shared A1+B history.
    bank_A1 = history_bank_windows(
        A1_episodes,
        history_length,
        int(iso["history_bank_stride"]),
    )
    bank_B = history_bank_windows(
        B_episodes,
        history_length,
        int(iso["history_bank_stride"]),
    )

    bank = {
        key: np.concatenate(
            [bank_A1[key], bank_B[key]],
            axis=0,
        )
        for key in bank_A1
    }

    bank = uniform_cap_bank(
        bank,
        max_windows=int(
            iso["max_history_bank_windows"]
        ),
        seed=seed,
    )

    bank_z = encode_histories(
        encoder_model,
        bank["history"],
        history_std,
    )

    bank_A_fraction = float(
        np.mean(bank["label"] == "A")
    )

    # 4. Exact fork at B -> A2.
    recent_agent = copy.deepcopy(agent)
    oracle_agent = copy.deepcopy(agent)
    learned_agent = copy.deepcopy(agent)

    recent_replay = copy.deepcopy(replay)
    oracle_replay = make_oracle_from_recent(
        replay,
        seed,
        cfg,
    )
    learned_replay = make_learned_from_recent(
        recent_replay=replay,
        bank=bank,
        bank_z=bank_z,
        seed=seed,
        cfg=cfg,
    )

    zero_returns = {
        "recent": float(
            evaluate(recent_agent, "A", seed, cfg)["return_mean"]
        ),
        "oracle": float(
            evaluate(oracle_agent, "A", seed, cfg)["return_mean"]
        ),
        "learned": float(
            evaluate(learned_agent, "A", seed, cfg)["return_mean"]
        ),
    }

    if not (
        np.isclose(
            zero_returns["recent"],
            zero_returns["oracle"],
            rtol=0.0,
            atol=1e-10,
        )
        and np.isclose(
            zero_returns["recent"],
            zero_returns["learned"],
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise RuntimeError(
            f"Forked agents are not identical: {zero_returns}"
        )

    recent_rows, recent_diag = train_A2_branch(
        "recent",
        recent_agent,
        recent_replay,
        global_step,
        seed,
        cfg,
    )

    oracle_rows, oracle_diag = train_A2_branch(
        "oracle",
        oracle_agent,
        oracle_replay,
        global_step,
        seed,
        cfg,
    )

    learned_rows, learned_diag = train_A2_branch(
        "learned",
        learned_agent,
        learned_replay,
        global_step,
        seed,
        cfg,
        encoder_model=encoder_model,
        history_std=history_std,
        history_length=history_length,
        bank_labels=bank["label"],
    )

    curve = pd.DataFrame(
        recent_rows + learned_rows + oracle_rows
    )

    curve, metrics = metrics_from_curve(
        curve=curve,
        a1_reference=a1_reference,
        random_A_return=rand_A,
        cfg=cfg,
    )

    # Keep the prior Oracle protocol exactly fixed. Historical numerical
    # drift across runtimes is measured, not treated as bitwise pass/fail.
    oracle_metric_drift = validate_oracle_protocol_and_measure_drift(
        cfg,
        metrics,
    )

    m = metrics.set_index("method")

    def censored_recovery(method):
        x = m.loc[method, "recovery_step"]
        if pd.notna(x):
            return float(x)
        return float(iso["A2_steps"]) + 100.0

    learned_recovery_improvement = (
        censored_recovery("recent")
        - censored_recovery("learned")
    )

    learned_auc_advantage = float(
        m.loc["learned", "AUC_0_2000"]
        - m.loc["recent", "AUC_0_2000"]
    )

    learned_vs_oracle_auc_gap = float(
        m.loc["oracle", "AUC_0_2000"]
        - m.loc["learned", "AUC_0_2000"]
    )

    recovery_screen = bool(
        learned_recovery_improvement
        >= float(
            cfg["screen"][
                "min_learned_recovery_improvement_steps"
            ]
        )
    )

    auc_screen = bool(
        learned_auc_advantage
        >= float(
            cfg["screen"][
                "min_learned_auc_advantage"
            ]
        )
    )

    supports_expansion = bool(
        recovery_screen or auc_screen
    )

    out = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "learned_replay_recurrence_pilot"
    )
    figs = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "learned_replay_recurrence_pilot"
    )

    curve.to_csv(
        out / "A2_recovery_curve.csv",
        index=False,
    )
    metrics.to_csv(
        out / "metrics.csv",
        index=False,
    )
    a1_df.to_csv(
        out / "A1_reference_evaluations.csv",
        index=False,
    )

    diagnostics = pd.DataFrame([
        recent_diag,
        learned_diag,
        oracle_diag,
    ])
    diagnostics.to_csv(
        out / "sampler_diagnostics.csv",
        index=False,
    )

    rep["behavior_diagnostics"].to_csv(
        out / "encoder_offline_diagnostics_by_behavior.csv",
        index=False,
    )

    pd.DataFrame(
        offline_drift_rows
    ).to_csv(
        out / "encoder_historical_metric_drift.csv",
        index=False,
    )

    oracle_metric_drift.to_csv(
        out / "oracle_control_historical_metric_drift.csv",
        index=False,
    )

    pd.DataFrame([{
        **offline_checks,
        "encoder_best_epoch": int(
            rep["model_meta"]["best_epoch"]
        ),
    }]).to_csv(
        out / "encoder_offline_summary.csv",
        index=False,
    )

    pd.DataFrame([{
        "n_bank_windows": int(len(bank["history"])),
        "bank_A_fraction_diagnostic_only": bank_A_fraction,
        "history_length": history_length,
        "bank_stride": int(iso["history_bank_stride"]),
        "retrieval_top_k_windows": int(
            iso["learned_retrieval_top_k_windows"]
        ),
    }]).to_csv(
        out / "historical_bank_summary.csv",
        index=False,
    )

    for zoom, filename, title in [
        (
            2000,
            "A2_recovery_first_2000.png",
            "Learned Relevance Recovery (First 2k)",
        ),
        (
            int(iso["A2_steps"]),
            "A2_recovery_full.png",
            "Learned Relevance Recovery",
        ),
    ]:
        fig, ax = plt.subplots(
            figsize=(8.5, 5.0)
        )

        for method in ["recent", "learned", "oracle"]:
            d = curve[
                (curve["method"] == method)
                & (curve["phase_step"] <= zoom)
            ].sort_values("phase_step")

            ax.plot(
                d["phase_step"],
                d["normalized_A_score"],
                marker="o",
                label=method,
            )

        ax.axhline(
            float(iso["recovery_threshold"]),
            linestyle="--",
            label="recovery threshold",
        )

        ax.set_xlabel("A2 training transitions")
        ax.set_ylabel("Normalized A performance")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / filename, dpi=180)
        plt.close(fig)

    save_json(
        {
            "code_commit": git_sha(),
            "seed": seed,
            "A1_reference_return": a1_reference,
            "B_final_return": B_final,
            "random_A": rand_A,
            "fork_zero_shot_returns": zero_returns,
            "oracle_control_protocol_matches_prior_config": True,
            "historical_metric_equality_required": False,
            "encoder_sanity": encoder_sanity,
            "05d_early_retrieval_formal_pass": False,
            "05d_early_retrieval_precision_at_10": 0.8064516186714172,
            "05d_first2k_retrieval_precision_at_10": 0.9504132270812988,
            "learned_selector_uses_true_context_labels": False,
            "bank_subsampling_uses_true_context_labels": False,
            "historical_fraction_target": float(
                iso["learned_historical_fraction"]
            ),
            "retrieval_top_k_windows": int(
                iso["learned_retrieval_top_k_windows"]
            ),
            "n_history_bank_windows": int(
                len(bank["history"])
            ),
            "bank_A_fraction_diagnostic_only": bank_A_fraction,
            "learned_recovery_improvement_steps": float(
                learned_recovery_improvement
            ),
            "learned_AUC_0_2000_advantage": float(
                learned_auc_advantage
            ),
            "learned_vs_oracle_AUC_gap": float(
                learned_vs_oracle_auc_gap
            ),
            "recovery_screen_pass": recovery_screen,
            "auc_screen_pass": auc_screen,
            "pilot_supports_3seed_expansion": supports_expansion,
            "next_if_supported": (
                "run paired seeds 2 and 3 with this frozen "
                "Recent/Learned/Oracle fork-at-return protocol"
            ),
            "next_if_not_supported": (
                "do not tune the encoder to force a pass; record that "
                "05c relevance is informative but insufficient for "
                "end-to-end recurrence utility and diagnose reuse mechanics"
            ),
        },
        out / "pilot_metadata.json",
    )

    save_yaml(
        cfg,
        out / "resolved_config.yaml",
    )

    print("\n=== 05c encoder regression check ===")
    print(
        pd.DataFrame([offline_checks]).to_string(
            index=False
        )
    )

    print("\n=== Historical bank ===")
    print(
        "n_windows=",
        len(bank["history"]),
        "diagnostic_A_fraction=",
        bank_A_fraction,
    )

    print("\n=== A2 metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== Sampler diagnostics ===")
    print(diagnostics.to_string(index=False))

    print(
        "\nLearned recovery improvement steps:",
        learned_recovery_improvement,
    )
    print(
        "Learned AUC(0-2000) advantage:",
        learned_auc_advantage,
    )
    print(
        "Learned vs Oracle AUC gap:",
        learned_vs_oracle_auc_gap,
    )
    print(
        "\nPILOT SUPPORTS 3-SEED EXPANSION:",
        supports_expansion,
    )


if __name__ == "__main__":
    main()
