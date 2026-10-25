
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import scripts.run_behavior_diverse_retrieval_pilot_v2 as behavior_module
from scripts.run_behavior_diverse_retrieval_pilot_v2 import train_behavior_diverse_encoder
from scripts.run_learned_relevance_retrieval_pilot_v2 import encode_histories
from scripts.run_learned_replay_recurrence_pilot_v2 import (
    history_bank_windows, train_common_phase_logged, uniform_cap_bank,
)
from scripts.train_stationary_v2 import epsilon_at, make_agent, make_env
from src.replay.recent import RecentReplayBuffer
from src.representation.history import current_feature, dynamics_transition_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed

CANDIDATES = [
    "retrieved_member_mse",
    "retrieved_mean_mse",
    "excess_mse",
    "log_ratio_mse",
]

def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def collect_probe(agent, context, global_step, seed, cfg):
    import copy

    dev = cfg["development"]
    agent = copy.deepcopy(agent)
    env = make_env(cfg, context)

    state, _ = env.reset(
        seed=173_000_000 + int(seed) * 1_000_000
    )

    episodes = []
    states = [np.asarray(state, dtype=np.float32)]
    actions = []
    episode_start = 0
    episode_id = 0
    phase_step = 0

    while phase_step < int(dev["probe_steps"]):
        action = agent.act(
            state,
            epsilon=epsilon_at(
                global_step + phase_step,
                cfg,
            ),
        )

        next_state, _, terminated, truncated, _ = env.step(action)

        actions.append(int(action))
        states.append(np.asarray(next_state, dtype=np.float32))
        state = next_state
        phase_step += 1

        if terminated or truncated:
            episodes.append({
                "episode_id": int(episode_id),
                "context": context,
                "states": np.asarray(states, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64),
                "start_phase_step": int(episode_start),
            })

            episode_id += 1
            states = []
            actions = []

            if phase_step < int(dev["probe_steps"]):
                state, _ = env.reset(
                    seed=(
                        173_000_000
                        + int(seed) * 1_000_000
                        + episode_id
                    )
                )
                states = [np.asarray(state, dtype=np.float32)]
                episode_start = phase_step

    if actions:
        episodes.append({
            "episode_id": int(episode_id),
            "context": context,
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "start_phase_step": int(episode_start),
        })

    env.close()
    return episodes


def compatibility_samples_from_probe(episodes, history_length, stride):
    histories, currents, targets = [], [], []
    labels, episode_ids, phase_steps = [], [], []
    L = int(history_length)

    for episode in episodes:
        states = episode["states"]
        actions = episode["actions"]
        T = len(actions)
        if T <= L:
            continue

        features = [
            dynamics_transition_feature(states[t], int(actions[t]), states[t+1], 5)
            for t in range(T)
        ]

        for t in range(L, T, int(stride)):
            histories.append(np.stack(features[t-L:t]).astype(np.float32))
            currents.append(current_feature(states[t], int(actions[t]), 5))
            targets.append((states[t+1] - states[t]).astype(np.float32))
            labels.append(episode["context"])
            episode_ids.append(int(episode["episode_id"]))
            phase_steps.append(int(episode["start_phase_step"] + t + 1))

    if not histories:
        raise RuntimeError("No compatibility samples created")

    return {
        "history": np.stack(histories).astype(np.float32),
        "current": np.stack(currents).astype(np.float32),
        "target": np.stack(targets).astype(np.float32),
        "label": np.asarray(labels),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "phase_step": np.asarray(phase_steps, dtype=np.int64),
    }

def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

@torch.no_grad()
def score_queries(model, history_std, current_std, target_std, query, bank_z, top_k, eps):
    query_z = encode_histories(model, query["history"], history_std)
    bank_norm = normalize_rows(bank_z)
    query_norm = normalize_rows(query_z)
    current_s = current_std.t(query["current"])
    target_s = target_std.t(query["target"])

    rows = []
    for i in range(len(query_z)):
        sim = bank_norm @ query_norm[i]
        k = min(int(top_k), len(sim))
        local = np.argpartition(-sim, kth=k-1)[:k]
        local = local[np.argsort(-sim[local])]

        z_hist = torch.from_numpy(bank_z[local].astype(np.float32))
        c = torch.from_numpy(current_s[i].astype(np.float32))
        y = torch.from_numpy(target_s[i].astype(np.float32))
        c_repeat = c.unsqueeze(0).repeat(k, 1)

        hist_pred = model.head(torch.cat([z_hist, c_repeat], dim=1))
        q_pred = model.head(
            torch.cat(
                [torch.from_numpy(query_z[i].astype(np.float32)), c],
                dim=0,
            ).unsqueeze(0)
        )[0]

        member_error = ((hist_pred - y.unsqueeze(0)) ** 2).mean(dim=1)
        retrieved_member_mse = float(member_error.mean().item())
        retrieved_mean_mse = float(
            ((hist_pred.mean(dim=0) - y) ** 2).mean().item()
        )
        self_mse = float(((q_pred - y) ** 2).mean().item())
        excess_mse = float(retrieved_member_mse - self_mse)
        log_ratio_mse = float(
            np.log(
                (retrieved_member_mse + float(eps))
                / (self_mse + float(eps))
            )
        )

        rows.append({
            "query_index": int(i),
            "best_similarity": float(sim[local[0]]),
            "mean_top_similarity": float(sim[local].mean()),
            "retrieved_member_mse": retrieved_member_mse,
            "retrieved_mean_mse": retrieved_mean_mse,
            "self_prediction_mse": self_mse,
            "excess_mse": excess_mse,
            "log_ratio_mse": log_ratio_mse,
        })

    return pd.DataFrame(rows)

def threshold_and_evaluate(calibration_scores, A_scores, D_scores, calibration_data, cfg):
    q = float(cfg["development"]["calibration_quantile"])
    criteria = cfg["criteria"]
    summaries, grouped_rows = [], []

    for score_name in CANDIDATES:
        threshold = float(
            np.quantile(
                calibration_scores[score_name].to_numpy(float),
                q,
            )
        )

        A = A_scores[score_name].to_numpy(float)
        D = D_scores[score_name].to_numpy(float)
        A_support = A <= threshold
        D_support = D <= threshold

        y = np.r_[np.zeros(len(A), dtype=int), np.ones(len(D), dtype=int)]
        score = np.r_[A, D]
        auroc = float(roc_auc_score(y, score))

        A_rate = float(A_support.mean())
        D_rate = float(D_support.mean())

        eligible = bool(
            A_rate >= float(criteria["min_recurrence_A_support_rate"])
            and D_rate <= float(criteria["max_novel_D_support_rate"])
            and auroc >= float(criteria["min_A_vs_D_anomaly_auroc"])
        )

        summaries.append({
            "score": score_name,
            "threshold": threshold,
            "calibration_quantile": q,
            "calibration_support_rate": float(
                (calibration_scores[score_name].to_numpy(float) <= threshold).mean()
            ),
            "A_support_rate": A_rate,
            "D_support_rate": D_rate,
            "A_score_mean": float(A.mean()),
            "D_score_mean": float(D.mean()),
            "A_vs_D_anomaly_AUROC": auroc,
            "eligible": eligible,
        })

        diag = calibration_data[["behavior", "label"]].copy()
        diag["supported"] = (
            calibration_scores[score_name].to_numpy(float) <= threshold
        )
        grouped = (
            diag.groupby(["behavior", "label"], observed=False)
            .agg(n=("supported", "size"), support_rate=("supported", "mean"))
            .reset_index()
        )
        grouped["score"] = score_name
        grouped_rows.append(grouped)

    summary = pd.DataFrame(summaries)
    eligible = (
        summary[summary["eligible"]]
        .sort_values(
            ["A_vs_D_anomaly_AUROC", "D_support_rate", "A_support_rate"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )
    selected = None if eligible.empty else str(eligible.iloc[0]["score"])
    return summary, pd.concat(grouped_rows, ignore_index=True), selected

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/synthetic_v2/support_compatibility_development.yaml",
    )
    args = ap.parse_args()

    cfg = load_config(ROOT/"configs/base.yaml", ROOT/args.config)
    dev = cfg["development"]
    seed = int(dev["seed"])
    torch.set_num_threads(1)

    out = ensure_dir(ROOT/"results/synthetic_v2/support_compatibility_development")
    figs = ensure_dir(ROOT/"figures/synthetic_v2/support_compatibility_development")

    behavior_module.GLOBAL_CFG = cfg
    set_global_seed(int(cfg["representation"]["seed"]))
    rep = train_behavior_diverse_encoder(cfg)

    model = rep["model"]
    history_std = rep["history_std"]
    current_std = rep["current_std"]
    target_std = rep["target_std"]
    heldout = rep["heldout_test"]
    L = int(cfg["representation"]["history_length"])

    sanity = {
        "relative_mse_improvement_ge_0_20": rep["relative_mse_improvement"] >= 0.20,
        "offline_AB_precision_at_10_ge_0_90": rep["offline_precision_at_10"] >= 0.90,
        "offline_AB_centroid_accuracy_ge_0_90": rep["offline_centroid_accuracy"] >= 0.90,
    }
    if not all(sanity.values()):
        raise RuntimeError(f"Frozen 05c capability check failed: {sanity}")

    set_global_seed(seed)
    agent = make_agent(cfg, seed)
    replay = RecentReplayBuffer(
        capacity=int(dev["replay_capacity"]),
        recent_window=int(dev["recent_window"]),
        seed=seed,
    )

    global_step = 0
    global_step, _, A1_episodes = train_common_phase_logged(
        agent, replay, "A", int(dev["A1_steps"]),
        global_step, seed, 0, cfg, eval_steps=None,
    )
    global_step, _, B_episodes = train_common_phase_logged(
        agent, replay, "B", int(dev["B_steps"]),
        global_step, seed, 1, cfg, eval_steps=None,
    )

    bank_A = history_bank_windows(A1_episodes, L, int(dev["history_bank_stride"]))
    bank_B = history_bank_windows(B_episodes, L, int(dev["history_bank_stride"]))
    bank = {
        key: np.concatenate([bank_A[key], bank_B[key]], axis=0)
        for key in bank_A
    }
    bank = uniform_cap_bank(
        bank,
        int(dev["max_history_bank_windows"]),
        seed,
    )
    bank_z = encode_histories(model, bank["history"], history_std)

    calibration_query = {
        "history": heldout["history"],
        "current": heldout["current"],
        "target": heldout["target"],
    }
    calibration_scores = score_queries(
        model, history_std, current_std, target_std,
        calibration_query, bank_z,
        int(dev["retrieval_top_k_windows"]),
        float(dev["score_eps"]),
    )
    calibration_data = pd.DataFrame({
        "behavior": heldout["behavior"],
        "label": heldout["label"],
    })

    A_probe = compatibility_samples_from_probe(
        collect_probe(agent, "A", global_step, seed, cfg),
        L,
        int(dev["probe_stride"]),
    )
    D_probe = compatibility_samples_from_probe(
        collect_probe(agent, "D", global_step, seed, cfg),
        L,
        int(dev["probe_stride"]),
    )

    A_scores = score_queries(
        model, history_std, current_std, target_std,
        A_probe, bank_z,
        int(dev["retrieval_top_k_windows"]),
        float(dev["score_eps"]),
    )
    D_scores = score_queries(
        model, history_std, current_std, target_std,
        D_probe, bank_z,
        int(dev["retrieval_top_k_windows"]),
        float(dev["score_eps"]),
    )

    summary, calibration_by_group, selected = threshold_and_evaluate(
        calibration_scores, A_scores, D_scores, calibration_data, cfg
    )

    y = np.r_[np.zeros(len(A_scores), dtype=int), np.ones(len(D_scores), dtype=int)]
    self_auroc = float(
        roc_auc_score(
            y,
            np.r_[
                A_scores["self_prediction_mse"].to_numpy(float),
                D_scores["self_prediction_mse"].to_numpy(float),
            ],
        )
    )

    prior = pd.read_csv(
        ROOT/"results/synthetic_v2/novelty_support_pilot/support_probe_queries.csv"
    )
    prior_y = (prior["context"] == "D_novel").astype(int).to_numpy()

    baselines = pd.DataFrame([
        {
            "baseline": "09_whitened_knn_density",
            "A_vs_D_anomaly_AUROC": float(
                roc_auc_score(prior_y, prior["support_distance"].to_numpy(float))
            ),
        },
        {
            "baseline": "absolute_best_cosine",
            "A_vs_D_anomaly_AUROC": float(
                roc_auc_score(prior_y, -prior["best_similarity"].to_numpy(float))
            ),
        },
        {
            "baseline": "self_prediction_mse",
            "A_vs_D_anomaly_AUROC": self_auroc,
        },
    ])

    A_out = A_scores.copy()
    A_out["context"] = "A_recurrence"
    A_out["phase_step"] = A_probe["phase_step"]
    A_out["episode_id"] = A_probe["episode_id"]

    D_out = D_scores.copy()
    D_out["context"] = "D_novel"
    D_out["phase_step"] = D_probe["phase_step"]
    D_out["episode_id"] = D_probe["episode_id"]

    summary.to_csv(out/"candidate_summary.csv", index=False)
    calibration_by_group.to_csv(
        out/"calibration_support_by_behavior_context.csv", index=False
    )
    calibration_scores.to_csv(out/"calibration_query_scores.csv", index=False)
    pd.concat([A_out, D_out], ignore_index=True).to_csv(
        out/"development_query_scores.csv", index=False
    )
    baselines.to_csv(out/"diagnostic_baselines.csv", index=False)

    pd.DataFrame([{
        "n_bank_windows": int(len(bank_z)),
        "bank_A_fraction_diagnostic_only": float(np.mean(bank["label"] == "A")),
        "n_calibration_queries": int(len(calibration_scores)),
        "n_A_development_queries": int(len(A_scores)),
        "n_D_development_queries": int(len(D_scores)),
    }]).to_csv(out/"data_summary.csv", index=False)

    save_json(
        {
            "code_commit": git_sha(),
            "seed": seed,
            "seed_role": "development_only",
            "heldout_confirmation_seeds": [2, 3],
            "true_context_labels_used_for_threshold_calibration": False,
            "D_used_for_threshold_calibration": False,
            "D_used_for_candidate_selection": True,
            "candidate_family_predeclared": CANDIDATES,
            "calibration_source": "frozen 05c held-out behavior-diverse A/B test queries",
            "calibration_quantile": float(dev["calibration_quantile"]),
            "retrieval_top_k_windows": int(dev["retrieval_top_k_windows"]),
            "selected_method": selected,
            "selection_success": bool(selected is not None),
            "selection_rule": (
                "eligible iff A support >=0.70, D support <=0.30, "
                "A-vs-D AUROC >=0.80; highest AUROC then lower D support "
                "then higher A support"
            ),
            "next_if_selected": (
                "freeze selected score and threshold rule, implement support-aware "
                "replay, then evaluate only held-out seeds 2/3 under ABA and ABD"
            ),
            "next_if_none_selected": (
                "do not touch held-out seeds 2/3; reconsider support formulation"
            ),
        },
        out/"development_metadata.json",
    )
    save_yaml(cfg, out/"resolved_config.yaml")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for _, row in summary.iterrows():
        ax.scatter(
            row["D_support_rate"],
            row["A_support_rate"],
            label=row["score"],
        )
    ax.axvline(float(cfg["criteria"]["max_novel_D_support_rate"]), linestyle="--")
    ax.axhline(float(cfg["criteria"]["min_recurrence_A_support_rate"]), linestyle="--")
    ax.set_xlabel("Novel-D support rate (lower is better)")
    ax.set_ylabel("Recurrence-A support rate (higher is better)")
    ax.set_title("Support Compatibility Development")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs/"candidate_support_rates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ordered = summary.sort_values("A_vs_D_anomaly_AUROC", ascending=False)
    ax.bar(ordered["score"], ordered["A_vs_D_anomaly_AUROC"])
    ax.axhline(float(cfg["criteria"]["min_A_vs_D_anomaly_auroc"]), linestyle="--")
    ax.set_ylabel("A-vs-D anomaly AUROC")
    ax.set_title("Compatibility Score Separation")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figs/"candidate_auroc.png", dpi=180)
    plt.close(fig)

    print("\n=== Candidate summary ===")
    print(summary.to_string(index=False))
    print("\n=== Diagnostic baselines ===")
    print(baselines.to_string(index=False))
    print("\n=== Calibration support by behavior/context ===")
    print(calibration_by_group.to_string(index=False))
    print("\nSELECTED SUPPORT METHOD:", selected)

    if selected is None:
        print("NO CANDIDATE PASSED. Do not touch held-out seeds 2/3.")
    else:
        print("Candidate selected on development seed 1. Freeze before held-out evaluation.")

if __name__ == "__main__":
    main()
