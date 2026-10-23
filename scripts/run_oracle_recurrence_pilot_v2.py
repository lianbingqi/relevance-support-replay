from __future__ import annotations

import argparse
import os
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

from scripts.train_stationary_v2 import epsilon_at, make_agent, make_env
from src.replay.oracle import OracleRelevantReplayBuffer
from src.replay.recent import RecentReplayBuffer
from src.replay.uniform import UniformReplayBuffer
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


@torch.no_grad()
def greedy_action(agent, state):
    x = torch.as_tensor(
        np.asarray(state, dtype=np.float32),
        device=agent.device,
    ).unsqueeze(0)
    return int(agent.online(x).argmax(dim=1).item())


def evaluate(agent, context_name, seed, cfg):
    env = make_env(cfg, context_name)
    n = int(cfg["recurrence"]["eval_episodes"])

    context_offset = 0 if context_name == "A" else 5000
    base = 20_000_000 + seed * 10_000 + context_offset

    returns = []
    safety = []

    for ep in range(n):
        state, _ = env.reset(seed=base + ep)
        done = False
        ret = 0.0
        final_term = False

        while not done:
            action = greedy_action(agent, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
            final_term = bool(terminated)

        returns.append(ret)
        safety.append(float(final_term))

    env.close()

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_sd": float(np.std(returns)),
        "eval_safety_termination_fraction": float(np.mean(safety)),
    }


def random_A_baseline(seed, cfg):
    env = make_env(cfg, "A")
    n = int(cfg["recurrence"]["random_baseline_episodes"])
    rng = np.random.default_rng(31_000_000 + seed)
    returns = []

    for ep in range(n):
        state, _ = env.reset(seed=31_000_000 + seed * 1000 + ep)
        done = False
        ret = 0.0

        while not done:
            action = int(rng.integers(env.action_space.n))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward

        returns.append(ret)

    env.close()
    return float(np.mean(returns))


def make_replay(method, seed, cfg):
    r = cfg["recurrence"]
    capacity = int(r["replay_capacity"])

    if method == "uniform":
        return UniformReplayBuffer(capacity=capacity, seed=seed)

    if method == "recent":
        return RecentReplayBuffer(
            capacity=capacity,
            recent_window=int(r["recent_window"]),
            seed=seed,
        )

    if method == "oracle":
        return OracleRelevantReplayBuffer(
            capacity=capacity,
            recent_window=int(r["recent_window"]),
            historical_fraction=float(r["oracle_historical_fraction"]),
            seed=seed,
        )

    raise ValueError(method)


def run_one(method, seed, cfg):
    torch.set_num_threads(1)
    set_global_seed(seed)

    agent = make_agent(cfg, seed)
    replay = make_replay(method, seed, cfg)

    eval_interval = int(cfg["recurrence"]["eval_interval_steps"])
    learning_starts = int(cfg["experiment"]["learning_starts"])
    train_frequency = int(cfg["experiment"]["train_frequency"])
    batch_size = int(cfg["agent"]["batch_size"])

    global_step = 0
    global_episode = 0
    eval_rows = []
    phase_rows = []

    for phase_index, phase in enumerate(cfg["recurrence"]["phases"]):
        phase_name = str(phase["name"])
        context = str(phase["context"])
        target_steps = int(phase["steps"])

        env = make_env(cfg, context)

        if method == "oracle":
            replay.set_current_context(context)

        eval_rows.append({
            "method": method,
            "seed": seed,
            "phase": phase_name,
            "context": context,
            "phase_step": 0,
            "global_step": global_step,
            **evaluate(agent, context, seed, cfg),
        })

        state, info = env.reset(
            seed=40_000_000 + seed * 1_000_000 + phase_index * 100_000
        )

        phase_step = 0
        updates = 0
        current_context_fractions = []
        oracle_hist_fractions = []

        while phase_step < target_steps:
            epsilon = epsilon_at(global_step, cfg)
            action = agent.act(state, epsilon=epsilon)

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

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
                if method == "oracle":
                    replay.set_current_context(context)

                batch = replay.sample(batch_size)
                agent.update(batch)
                updates += 1

                current_context_fractions.append(
                    float(np.mean([c == context for c in batch.context_ids]))
                )

                if method == "oracle":
                    oracle_hist_fractions.append(
                        float(
                            replay.last_sample_stats[
                                "historical_relevant_fraction"
                            ]
                        )
                    )

            state = next_state
            info = next_info
            phase_step += 1
            global_step += 1

            if (
                phase_step % eval_interval == 0
                or phase_step == target_steps
            ):
                eval_rows.append({
                    "method": method,
                    "seed": seed,
                    "phase": phase_name,
                    "context": context,
                    "phase_step": phase_step,
                    "global_step": global_step,
                    **evaluate(agent, context, seed, cfg),
                })

            if done and phase_step < target_steps:
                global_episode += 1
                state, info = env.reset(
                    seed=(
                        40_000_000
                        + seed * 1_000_000
                        + phase_index * 100_000
                        + global_episode
                    )
                )

        counts = {}
        for item in replay.buffer:
            counts[item.context_id] = counts.get(item.context_id, 0) + 1

        phase_rows.append({
            "method": method,
            "seed": seed,
            "phase": phase_name,
            "context": context,
            "global_step_end": global_step,
            "updates": updates,
            "replay_size": len(replay),
            "replay_A_count": counts.get("A", 0),
            "replay_B_count": counts.get("B", 0),
            "mean_sample_current_context_fraction": (
                float(np.mean(current_context_fractions))
                if current_context_fractions else np.nan
            ),
            "mean_oracle_historical_relevant_fraction": (
                float(np.mean(oracle_hist_fractions))
                if oracle_hist_fractions else np.nan
            ),
        })

        env.close()

    return {
        "method": method,
        "seed": seed,
        "eval_rows": eval_rows,
        "phase_rows": phase_rows,
        "random_A": random_A_baseline(seed, cfg),
    }


def compute_metrics(eval_df, random_df, cfg):
    r = cfg["recurrence"]
    threshold = float(r["recovery_threshold"])
    consecutive = int(r["recovery_consecutive_checkpoints"])
    ref_n = int(r["a1_reference_checkpoints"])
    early_window = int(r["early_a2_window_steps"])
    a2_total = int(
        [p["steps"] for p in r["phases"] if p["name"] == "A2"][0]
    )
    eval_interval = int(r["eval_interval_steps"])

    rows = []

    for method in r["methods"]:
        seed = int(r["seeds"][0])
        run = eval_df[
            (eval_df.method == method) & (eval_df.seed == seed)
        ].copy()

        a1 = run[run.phase == "A1"].sort_values("phase_step")
        a2 = run[run.phase == "A2"].sort_values("phase_step").copy()

        a1_ref = float(a1.tail(ref_n).eval_return_mean.mean())
        random_A = float(
            random_df[
                (random_df.method == method) & (random_df.seed == seed)
            ].random_A.iloc[0]
        )

        denom = a1_ref - random_A
        if denom <= 0:
            raise RuntimeError(f"{method}: A1 reference does not beat random")

        a2["score"] = (a2.eval_return_mean - random_A) / denom
        good = a2.score.to_numpy() >= threshold

        recovery_step = None
        for i in range(len(good) - consecutive + 1):
            if bool(np.all(good[i:i+consecutive])):
                recovery_step = int(a2.iloc[i].phase_step)
                break

        censored = (
            recovery_step
            if recovery_step is not None
            else a2_total + eval_interval
        )

        early = a2[a2.phase_step <= early_window]

        rows.append({
            "method": method,
            "seed": seed,
            "random_A": random_A,
            "a1_reference_return": a1_ref,
            "recovered": recovery_step is not None,
            "recovery_step": (
                recovery_step if recovery_step is not None else np.nan
            ),
            "censored_recovery_step": censored,
            "A2_start_score": float(a2.iloc[0].score),
            "early_A2_score_mean": float(early.score.mean()),
            "A2_final_score": float(a2.iloc[-1].score),
        })

    return pd.DataFrame(rows)


def plot_A2(eval_df, metrics_df, cfg, path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in cfg["recurrence"]["methods"]:
        metric = metrics_df[metrics_df.method == method].iloc[0]
        run = (
            eval_df[
                (eval_df.method == method)
                & (eval_df.phase == "A2")
            ]
            .copy()
            .sort_values("phase_step")
        )

        denom = metric.a1_reference_return - metric.random_A
        score = (run.eval_return_mean - metric.random_A) / denom

        ax.plot(run.phase_step, score, marker="o", label=method)

    ax.axhline(
        float(cfg["recurrence"]["recovery_threshold"]),
        linestyle="--",
        label="recovery threshold",
    )
    ax.set_xlabel("A2 training transitions")
    ax.set_ylabel("Normalized A performance")
    ax.set_title("Oracle Recurrence Pilot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic_v2/oracle_recurrence_pilot.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs/base.yaml", ROOT / args.config)

    out = ensure_dir(ROOT / "results/synthetic_v2/oracle_recurrence_pilot")
    figs = ensure_dir(ROOT / "figures/synthetic_v2/oracle_recurrence_pilot")

    jobs = [
        (method, int(cfg["recurrence"]["seeds"][0]))
        for method in cfg["recurrence"]["methods"]
    ]

    max_workers = min(
        int(cfg["recurrence"]["max_workers"]),
        len(jobs),
    )

    eval_rows = []
    phase_rows = []
    random_rows = []

    print(
        f"jobs={len(jobs)}, workers={max_workers}, "
        f"steps/run={sum(int(p['steps']) for p in cfg['recurrence']['phases'])}"
    )

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_one, method, seed, cfg): (method, seed)
            for method, seed in jobs
        }

        for k, future in enumerate(as_completed(futures), start=1):
            method, seed = futures[future]
            result = future.result()

            eval_rows.extend(result["eval_rows"])
            phase_rows.extend(result["phase_rows"])
            random_rows.append({
                "method": method,
                "seed": seed,
                "random_A": result["random_A"],
            })

            print(f"[{k}/{len(jobs)}] finished {method}, seed={seed}")

    eval_df = pd.DataFrame(eval_rows)
    phase_df = pd.DataFrame(phase_rows)
    random_df = pd.DataFrame(random_rows)
    metrics_df = compute_metrics(eval_df, random_df, cfg)

    eval_df.to_csv(out / "evaluation_curve.csv", index=False)
    phase_df.to_csv(out / "phase_diagnostics.csv", index=False)
    random_df.to_csv(out / "random_baselines.csv", index=False)
    metrics_df.to_csv(out / "pilot_metrics.csv", index=False)

    recent = metrics_df[metrics_df.method == "recent"].iloc[0]
    oracle = metrics_df[metrics_df.method == "oracle"].iloc[0]

    recovery_improvement = (
        float(recent.censored_recovery_step)
        - float(oracle.censored_recovery_step)
    )

    early_advantage = (
        float(oracle.early_A2_score_mean)
        - float(recent.early_A2_score_mean)
    )

    screen = cfg["pilot_screen"]

    recovery_pass = bool(
        recovery_improvement
        >= float(screen["min_recovery_improvement_steps"])
    )
    early_pass = bool(
        early_advantage
        >= float(screen["min_early_a2_score_advantage"])
    )

    pilot_pass = bool(recovery_pass and early_pass)

    plot_A2(
        eval_df,
        metrics_df,
        cfg,
        figs / "A2_recovery_pilot.png",
    )

    save_json({
        "code_commit": git_sha(),
        "pilot_seed": int(cfg["recurrence"]["seeds"][0]),
        "recovery_improvement_steps": recovery_improvement,
        "early_A2_score_advantage": early_advantage,
        "recovery_screen_pass": recovery_pass,
        "early_score_screen_pass": early_pass,
        "pilot_pass": pilot_pass,
        "decision_if_pass": "run seeds 2 and 3 using the same frozen protocol",
        "decision_if_fail": "diagnose mechanism before running more seeds",
    }, out / "pilot_metadata.json")

    save_yaml(cfg, out / "resolved_config.yaml")

    print("\n=== Pilot metrics ===")
    print(metrics_df.to_string(index=False))
    print("\nRecovery improvement (Recent - Oracle):", recovery_improvement)
    print("Early A2 score advantage (Oracle - Recent):", early_advantage)
    print("\nPILOT PASS:", pilot_pass)


if __name__ == "__main__":
    main()
