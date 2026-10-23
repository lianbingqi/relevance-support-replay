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

from scripts.train_stationary_v2 import epsilon_at, make_agent, make_env
from src.replay.oracle import OracleRelevantReplayBuffer
from src.replay.recent import RecentReplayBuffer
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


def evaluate(agent, context, seed, cfg):
    env = make_env(cfg, context)
    n = int(cfg["isolation"]["eval_episodes"])
    offset = 0 if context == "A" else 5000
    base = 70_000_000 + seed * 10_000 + offset

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
        "return_mean": float(np.mean(returns)),
        "return_sd": float(np.std(returns)),
        "safety_fraction": float(np.mean(safety)),
    }


def random_A(seed, cfg):
    env = make_env(cfg, "A")
    n = int(cfg["isolation"]["random_baseline_episodes"])
    rng = np.random.default_rng(72_000_000 + seed)
    returns = []

    for ep in range(n):
        state, _ = env.reset(seed=72_000_000 + seed * 1000 + ep)
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


def train_common_phase(
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

    while phase_step < phase_steps:
        action = agent.act(
            state,
            epsilon=epsilon_at(global_step, cfg),
        )

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
            agent.update(replay.sample(batch_size))

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

        if done and phase_step < phase_steps:
            episode_id += 1
            state, info = env.reset(
                seed=(
                    80_000_000
                    + seed * 1_000_000
                    + phase_index * 100_000
                    + episode_id
                )
            )

    env.close()
    return global_step, eval_rows


def make_oracle_from_recent(recent_replay, seed, cfg):
    iso = cfg["isolation"]

    oracle = OracleRelevantReplayBuffer(
        capacity=int(iso["replay_capacity"]),
        recent_window=int(iso["recent_window"]),
        historical_fraction=float(iso["oracle_historical_fraction"]),
        seed=seed,
    )

    oracle.buffer = copy.deepcopy(recent_replay.buffer)
    oracle.rng.bit_generator.state = copy.deepcopy(
        recent_replay.rng.bit_generator.state
    )

    return oracle


def train_A2_branch(method, agent, replay, global_step, seed, cfg):
    env = make_env(cfg, "A")
    batch_size = int(cfg["agent"]["batch_size"])
    train_frequency = int(cfg["experiment"]["train_frequency"])
    target_steps = int(cfg["isolation"]["A2_steps"])
    eval_steps = set(int(x) for x in cfg["isolation"]["A2_eval_steps"])

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

    state, info = env.reset(seed=90_000_000 + seed * 1_000_000)

    phase_step = 0
    episode_id = 0
    hist_fracs = []
    recent_fracs = []

    while phase_step < target_steps:
        action = agent.act(
            state,
            epsilon=epsilon_at(global_step, cfg),
        )

        next_state, reward, terminated, truncated, next_info = env.step(action)
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

        if global_step % train_frequency == 0 and len(replay) >= batch_size:
            if method == "oracle":
                replay.set_current_context("A")

            batch = replay.sample(batch_size)
            agent.update(batch)

            if method == "oracle":
                hist_fracs.append(
                    replay.last_sample_stats["historical_relevant_fraction"]
                )
                recent_fracs.append(
                    replay.last_sample_stats["recent_fraction"]
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

        if done and phase_step < target_steps:
            episode_id += 1
            state, info = env.reset(
                seed=90_000_000 + seed * 1_000_000 + episode_id
            )

    env.close()

    return rows, {
        "method": method,
        "mean_oracle_historical_fraction": (
            float(np.mean(hist_fracs)) if hist_fracs else np.nan
        ),
        "mean_oracle_recent_fraction": (
            float(np.mean(recent_fracs)) if recent_fracs else np.nan
        ),
    }


def first_sustained_recovery(df, threshold, consecutive):
    good = df["normalized_A_score"].to_numpy() >= threshold

    for i in range(len(good) - consecutive + 1):
        if bool(np.all(good[i:i+consecutive])):
            return int(df.iloc[i]["phase_step"])

    return None


def auc_0_2000(df):
    d = df[
        (df["phase_step"] >= 0)
        & (df["phase_step"] <= 2000)
    ].sort_values("phase_step")

    x = d["phase_step"].to_numpy(dtype=float)
    y = d["normalized_A_score"].to_numpy(dtype=float)

    if x[0] != 0 or x[-1] != 2000:
        raise RuntimeError("AUC endpoints are missing")

    return float(np.trapz(y, x) / 2000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic_v2/oracle_recovery_isolation.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs/base.yaml", ROOT / args.config)
    iso = cfg["isolation"]
    seed = int(iso["seed"])

    torch.set_num_threads(1)
    set_global_seed(seed)

    out = ensure_dir(
        ROOT / "results/synthetic_v2/oracle_recovery_isolation"
    )
    figs = ensure_dir(
        ROOT / "figures/synthetic_v2/oracle_recovery_isolation"
    )

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

    B_final = evaluate(agent, "B", seed, cfg)["return_mean"]
    rand_A = random_A(seed, cfg)

    recent_agent = copy.deepcopy(agent)
    oracle_agent = copy.deepcopy(agent)

    recent_replay = copy.deepcopy(replay)
    oracle_replay = make_oracle_from_recent(replay, seed, cfg)

    recent_zero = evaluate(recent_agent, "A", seed, cfg)["return_mean"]
    oracle_zero = evaluate(oracle_agent, "A", seed, cfg)["return_mean"]

    if not np.isclose(recent_zero, oracle_zero, rtol=0.0, atol=1e-10):
        raise RuntimeError("Forked agents are not identical")

    recent_rows, recent_diag = train_A2_branch(
        "recent", recent_agent, recent_replay, global_step, seed, cfg
    )
    oracle_rows, oracle_diag = train_A2_branch(
        "oracle", oracle_agent, oracle_replay, global_step, seed, cfg
    )

    curve = pd.DataFrame(recent_rows + oracle_rows)

    denom = a1_reference - rand_A
    if denom <= 0:
        raise RuntimeError("A1 reference does not beat random A")

    curve["normalized_A_score"] = (
        curve["eval_return_mean"] - rand_A
    ) / denom

    curve.to_csv(out / "A2_recovery_curve.csv", index=False)
    a1_df.to_csv(out / "A1_reference_evaluations.csv", index=False)
    pd.DataFrame([recent_diag, oracle_diag]).to_csv(
        out / "sampler_diagnostics.csv",
        index=False,
    )

    threshold = float(iso["recovery_threshold"])
    consecutive = int(iso["recovery_consecutive_checkpoints"])

    metric_rows = []

    for method in ["recent", "oracle"]:
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

        metric_rows.append({
            "method": method,
            "zero_shot_score": float(d.iloc[0]["normalized_A_score"]),
            "recovery_step": recovery if recovery is not None else np.nan,
            "AUC_0_2000": auc_0_2000(d),
            "post_switch_mean_score_0_2000": float(
                post["normalized_A_score"].mean()
            ),
            "final_score": float(d.iloc[-1]["normalized_A_score"]),
        })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "isolation_metrics.csv", index=False)

    recent_m = metrics[metrics["method"] == "recent"].iloc[0]
    oracle_m = metrics[metrics["method"] == "oracle"].iloc[0]

    recent_recovery = (
        float(recent_m["recovery_step"])
        if pd.notna(recent_m["recovery_step"])
        else float(iso["A2_steps"]) + 100
    )

    oracle_recovery = (
        float(oracle_m["recovery_step"])
        if pd.notna(oracle_m["recovery_step"])
        else float(iso["A2_steps"]) + 100
    )

    recovery_improvement = recent_recovery - oracle_recovery
    auc_advantage = float(
        oracle_m["AUC_0_2000"] - recent_m["AUC_0_2000"]
    )

    recovery_screen = bool(
        recovery_improvement
        >= float(cfg["screen"]["min_recovery_improvement_steps"])
    )

    auc_screen = bool(
        auc_advantage
        >= float(cfg["screen"]["min_auc_advantage"])
    )

    supports_expansion = bool(recovery_screen or auc_screen)

    for zoom, filename, title in [
        (2000, "A2_recovery_first_2000.png", "High-Resolution Recovery (First 2k)"),
        (int(iso["A2_steps"]), "A2_recovery_isolated.png", "Isolated A2 Recovery"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for method in ["recent", "oracle"]:
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
            threshold,
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
            "fork_zero_shot_return": recent_zero,
            "recovery_improvement_steps": recovery_improvement,
            "AUC_0_2000_advantage": auc_advantage,
            "recovery_screen_pass": recovery_screen,
            "auc_screen_pass": auc_screen,
            "diagnostic_supports_expansion": supports_expansion,
            "next_if_supported": (
                "run paired seeds 2 and 3 with this frozen fork-at-return protocol"
            ),
            "next_if_not_supported": (
                "do not add GRU yet; reconsider replay hypothesis or objective"
            ),
        },
        out / "isolation_metadata.json",
    )

    save_yaml(cfg, out / "resolved_config.yaml")

    print("\\n=== A1 reference ===")
    print(a1_df.to_string(index=False))
    print("\\nA1 reference return:", a1_reference)
    print("B final return:", B_final)
    print("A2 fork zero-shot return:", recent_zero)

    print("\\n=== Isolated recovery metrics ===")
    print(metrics.to_string(index=False))

    print("\\nRecovery improvement steps:", recovery_improvement)
    print("AUC(0-2000) advantage:", auc_advantage)
    print("\\nDIAGNOSTIC SUPPORTS EXPANSION:", supports_expansion)


if __name__ == "__main__":
    main()
