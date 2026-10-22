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

from src.agents.ddqn import DDQNAgent
from src.environments.rotated_actuator import HiddenRotatedActuatorEnv
from src.replay.uniform import UniformReplayBuffer
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def make_env(cfg, context_name):
    env_cfg = cfg["environment"]

    return HiddenRotatedActuatorEnv(
        contexts=cfg["contexts"],
        context_name=context_name,
        rho=env_cfg["rho"],
        beta=env_cfg["beta"],
        noise_std=env_cfg["noise_std"],
        init_position_range=env_cfg["init_position_range"],
        init_velocity_range=env_cfg["init_velocity_range"],
        max_episode_steps=env_cfg["max_episode_steps"],
        safety_radius=env_cfg["safety_radius"],
        safety_speed=env_cfg["safety_speed"],
    )


def epsilon_at(step, cfg):
    exp = cfg["experiment"]
    start = float(exp["epsilon_start"])
    end = float(exp["epsilon_end"])
    decay = float(exp["epsilon_decay_steps"])

    if decay <= 0:
        return end

    frac = min(max(step / decay, 0.0), 1.0)
    return start + frac * (end - start)


def make_agent(cfg, seed):
    a = cfg["agent"]

    return DDQNAgent(
        observation_dim=4,
        action_dim=5,
        gamma=a["gamma"],
        learning_rate=a["learning_rate"],
        target_update_steps=a["target_update_steps"],
        hidden_sizes=a["hidden_sizes"],
        grad_clip_norm=a["grad_clip_norm"],
        seed=seed,
        device="cpu",
    )


def train_one(context_name, seed, cfg):
    torch.set_num_threads(1)
    set_global_seed(seed)

    env = make_env(cfg, context_name)
    agent = make_agent(cfg, seed)

    replay = UniformReplayBuffer(
        capacity=int(cfg["replay"]["capacity"]),
        seed=seed,
    )

    train_cfg = cfg["stationary"]
    exp = cfg["experiment"]
    agent_cfg = cfg["agent"]

    global_step = 0
    rows = []

    for episode in range(int(train_cfg["training_episodes"])):
        state, info = env.reset(
            seed=seed * 100_000 + episode
        )

        done = False
        episode_return = 0.0
        length = 0
        final_terminated = False
        final_truncated = False

        while not done:
            epsilon = epsilon_at(
                global_step,
                cfg,
            )

            action = agent.act(
                state,
                epsilon=epsilon,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                next_info,
            ) = env.step(action)

            done = terminated or truncated

            # Time-limit truncation ends the rollout but is not
            # an MDP terminal for DDQN bootstrapping.
            replay.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,
                context_id=info["true_context"],
                context_vector=info["context_vector"],
            )

            if (
                global_step >= int(exp["learning_starts"])
                and global_step % int(exp["train_frequency"]) == 0
                and len(replay) >= int(agent_cfg["batch_size"])
            ):
                agent.update(
                    replay.sample(
                        int(agent_cfg["batch_size"])
                    )
                )

            state = next_state
            info = next_info
            episode_return += reward
            length += 1
            global_step += 1

            final_terminated = bool(terminated)
            final_truncated = bool(truncated)

        rows.append({
            "context": context_name,
            "seed": seed,
            "episode": episode,
            "global_step": global_step,
            "return": episode_return,
            "length": length,
            "safety_terminated": final_terminated,
            "time_limit_truncated": final_truncated,
            "epsilon": epsilon,
        })

    env.close()

    return agent, rows


def evaluate(agent, context_name, seed, cfg):
    env = make_env(cfg, context_name)
    n_episodes = int(
        cfg["stationary"]["eval_episodes"]
    )

    returns = []
    lengths = []
    safety_terminations = []

    # Matched evaluation initial-state streams across contexts.
    eval_seed_base = (
        5_000_000
        + seed * 10_000
    )

    for episode in range(n_episodes):
        state, _ = env.reset(
            seed=eval_seed_base + episode
        )

        done = False
        episode_return = 0.0
        length = 0
        final_terminated = False

        while not done:
            action = agent.act(
                state,
                epsilon=0.0,
            )

            (
                state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            done = terminated or truncated
            episode_return += reward
            length += 1
            final_terminated = bool(terminated)

        returns.append(episode_return)
        lengths.append(length)
        safety_terminations.append(
            float(final_terminated)
        )

    env.close()

    return {
        "eval_return_mean": float(
            np.mean(returns)
        ),
        "eval_return_sd": float(
            np.std(returns)
        ),
        "eval_full_length_fraction": float(
            np.mean(
                np.asarray(lengths)
                == int(
                    cfg["environment"][
                        "max_episode_steps"
                    ]
                )
            )
        ),
        "eval_safety_termination_fraction": float(
            np.mean(safety_terminations)
        ),
    }


def run_job(context_name, seed, cfg):
    agent, rows = train_one(
        context_name,
        seed,
        cfg,
    )

    eval_stats = evaluate(
        agent,
        context_name,
        seed,
        cfg,
    )

    return {
        "context": context_name,
        "seed": seed,
        "training_rows": rows,
        "eval_stats": eval_stats,
    }


def random_baseline(context_name, cfg):
    env = make_env(cfg, context_name)

    episodes = int(
        cfg["stationary"][
            "random_baseline_episodes"
        ]
    )

    rng = np.random.default_rng(
        9_000_000
    )

    returns = []
    safety_terminations = []

    for episode in range(episodes):
        state, _ = env.reset(
            seed=9_000_000 + episode
        )

        done = False
        episode_return = 0.0
        final_terminated = False

        while not done:
            action = int(
                rng.integers(
                    env.action_space.n
                )
            )

            (
                state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            done = terminated or truncated
            episode_return += reward
            final_terminated = bool(terminated)

        returns.append(
            episode_return
        )
        safety_terminations.append(
            float(final_terminated)
        )

    env.close()

    return {
        "context": context_name,
        "random_return_mean": float(
            np.mean(returns)
        ),
        "random_return_sd": float(
            np.std(returns)
        ),
        "random_safety_termination_fraction": float(
            np.mean(safety_terminations)
        ),
    }


def plot_learning_curves(training_df, cfg, figure_path):
    smoothing = int(
        cfg["stationary"][
            "smoothing_window"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.0)
    )

    for context_name in cfg["contexts"]:
        subset = training_df[
            training_df["context"]
            == context_name
        ]

        per_seed = []

        for seed in sorted(
            subset["seed"].unique()
        ):
            curve = (
                subset[
                    subset["seed"] == seed
                ]
                .sort_values("episode")
                .set_index("episode")[
                    "return"
                ]
                .rolling(
                    smoothing,
                    min_periods=1,
                )
                .mean()
            )

            per_seed.append(
                curve.rename(seed)
            )

        curves = pd.concat(
            per_seed,
            axis=1,
        )

        mean_curve = curves.mean(
            axis=1
        )
        sd_curve = curves.std(
            axis=1,
            ddof=1,
        )

        line = ax.plot(
            mean_curve.index,
            mean_curve.values,
            label=context_name,
        )[0]

        ax.fill_between(
            mean_curve.index,
            mean_curve.values
            - sd_curve.values,
            mean_curve.values
            + sd_curve.values,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Return")
    ax.set_title(
        "Synthetic v2 Stationary DDQN Learning"
    )
    ax.legend()
    fig.tight_layout()

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        figure_path,
        dpi=180,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/synthetic_v2/"
            "stationary.yaml"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / args.config,
    )

    output_dir = ensure_dir(
        ROOT
        / "results"
        / "synthetic_v2"
        / "stationary"
    )

    figure_dir = ensure_dir(
        ROOT
        / "figures"
        / "synthetic_v2"
        / "stationary"
    )

    commit = git_sha()

    context_names = list(
        cfg["contexts"].keys()
    )
    seeds = [
        int(seed)
        for seed in cfg[
            "stationary"
        ]["train_seeds"]
    ]

    random_rows = [
        random_baseline(
            context_name,
            cfg,
        )
        for context_name
        in context_names
    ]

    random_df = pd.DataFrame(
        random_rows
    )

    random_df.to_csv(
        output_dir
        / "random_baselines.csv",
        index=False,
    )

    jobs = [
        (context_name, seed)
        for context_name
        in context_names
        for seed in seeds
    ]

    max_workers = min(
        int(
            cfg["stationary"][
                "max_workers"
            ]
        ),
        len(jobs),
    )

    print(
        f"CPU count={os.cpu_count()}, "
        f"jobs={len(jobs)}, "
        f"workers={max_workers}"
    )

    training_rows = []
    eval_rows = []

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {
            executor.submit(
                run_job,
                context_name,
                seed,
                cfg,
            ): (
                context_name,
                seed,
            )
            for context_name, seed
            in jobs
        }

        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            context_name, seed = (
                futures[future]
            )

            result = future.result()

            training_rows.extend(
                result[
                    "training_rows"
                ]
            )

            eval_rows.append({
                "code_commit": commit,
                "context": context_name,
                "seed": seed,
                **result["eval_stats"],
            })

            print(
                f"[{completed}/{len(jobs)}] "
                f"finished "
                f"{context_name}, seed={seed}"
            )

    training_df = pd.DataFrame(
        training_rows
    )

    eval_df = pd.DataFrame(
        eval_rows
    )

    training_df.to_csv(
        output_dir
        / "training_metrics.csv",
        index=False,
    )

    eval_df.to_csv(
        output_dir
        / "evaluation_by_seed.csv",
        index=False,
    )

    random_map = (
        random_df
        .set_index("context")
        .to_dict("index")
    )

    early_window = int(
        cfg["stationary"][
            "early_window"
        ]
    )
    late_window = int(
        cfg["stationary"][
            "late_window"
        ]
    )

    summary_rows = []

    for context_name in context_names:
        context_train = training_df[
            training_df["context"]
            == context_name
        ]

        early = (
            context_train
            .groupby(
                "seed",
                group_keys=False,
            )
            .head(early_window)
        )

        late = (
            context_train
            .groupby(
                "seed",
                group_keys=False,
            )
            .tail(late_window)
        )

        context_eval = eval_df[
            eval_df["context"]
            == context_name
        ]

        eval_mean = float(
            context_eval[
                "eval_return_mean"
            ].mean()
        )

        random_mean = float(
            random_map[
                context_name
            ][
                "random_return_mean"
            ]
        )

        late_mean = float(
            late["return"].mean()
        )

        early_mean = float(
            early["return"].mean()
        )

        zero_eval_safety = bool(
            np.isclose(
                context_eval[
                    "eval_safety_termination_fraction"
                ].max(),
                0.0,
            )
        )

        beats_random = bool(
            eval_mean > random_mean
        )

        training_improves = bool(
            late_mean > early_mean
        )

        passes = bool(
            zero_eval_safety
            and beats_random
            and training_improves
        )

        summary_rows.append({
            "context": context_name,
            "rotation_deg": float(
                cfg["contexts"][
                    context_name
                ][
                    "rotation_deg"
                ]
            ),
            "early_train_return_mean": early_mean,
            "late_train_return_mean": late_mean,
            "late_train_return_sd": float(
                late["return"].std(
                    ddof=1
                )
            ),
            "eval_return_mean": eval_mean,
            "eval_return_sd_across_seeds": float(
                context_eval[
                    "eval_return_mean"
                ].std(
                    ddof=1
                )
            ),
            "eval_full_length_fraction": float(
                context_eval[
                    "eval_full_length_fraction"
                ].mean()
            ),
            "eval_safety_termination_fraction": float(
                context_eval[
                    "eval_safety_termination_fraction"
                ].mean()
            ),
            "random_return_mean": random_mean,
            "random_safety_termination_fraction": float(
                random_map[
                    context_name
                ][
                    "random_safety_termination_fraction"
                ]
            ),
            "beats_random": beats_random,
            "training_improves": training_improves,
            "zero_eval_safety_terminations": zero_eval_safety,
            "sanity_pass": passes,
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        output_dir
        / "stationary_summary.csv",
        index=False,
    )

    all_pass = bool(
        summary_df[
            "sanity_pass"
        ].all()
    )

    save_json(
        {
            "code_commit": commit,
            "contexts": cfg["contexts"],
            "train_seeds": seeds,
            "all_contexts_pass": all_pass,
            "next_step_if_pass": (
                "v2_02_context_manipulation_check"
            ),
        },
        output_dir
        / "stationary_metadata.json",
    )

    save_yaml(
        cfg,
        output_dir
        / "resolved_config.yaml",
    )

    plot_learning_curves(
        training_df,
        cfg,
        figure_dir
        / "stationary_learning_curves.png",
    )

    print(
        "\n=== Synthetic v2 stationary sanity ==="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nALL CONTEXTS PASS:",
        all_pass,
    )


if __name__ == "__main__":
    main()
