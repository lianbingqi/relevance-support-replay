from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.agents.ddqn import DDQNAgent
from src.environments.synthetic_control import HiddenContextControlEnv
from src.replay.uniform import UniformReplayBuffer
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def linear_epsilon(step, start, end, decay_steps):
    if decay_steps <= 0:
        return float(end)
    frac = min(max(step / float(decay_steps), 0.0), 1.0)
    return float(start + frac * (end - start))


def current_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_env(cfg):
    e = cfg["environment"]
    return HiddenContextControlEnv(
        contexts=cfg["contexts"],
        context_name=e["context_name"],
        max_episode_steps=e["max_episode_steps"],
        noise_std=e["noise_std"],
        init_x_range=e["init_x_range"],
        init_v_range=e["init_v_range"],
        x_limit=e["x_limit"],
        v_limit=e["v_limit"],
    )


def evaluate(agent, cfg, seed, episodes):
    env = build_env(cfg)
    returns = []

    for ep in range(int(episodes)):
        state, _ = env.reset(seed=seed + 100_000 + ep)
        episode_done = False
        episode_return = 0.0

        while not episode_done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_done = terminated or truncated
            episode_return += reward
            state = next_state

        returns.append(episode_return)

    env.close()
    return np.asarray(returns, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic/stationary.yaml",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / args.config,
    )

    seed = int(args.seed)
    set_global_seed(seed)

    env = build_env(cfg)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    agent_cfg = cfg["agent"]
    exp_cfg = cfg["experiment"]
    replay_cfg = cfg["replay"]

    agent = DDQNAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        gamma=agent_cfg["gamma"],
        learning_rate=agent_cfg["learning_rate"],
        target_update_steps=agent_cfg["target_update_steps"],
        hidden_sizes=agent_cfg["hidden_sizes"],
        grad_clip_norm=agent_cfg["grad_clip_norm"],
        seed=seed,
    )

    replay = UniformReplayBuffer(
        capacity=replay_cfg["capacity"],
        seed=seed,
    )

    episodes = (
        int(args.episodes)
        if args.episodes is not None
        else int(exp_cfg["episodes"])
    )
    eval_episodes = (
        int(args.eval_episodes)
        if args.eval_episodes is not None
        else int(exp_cfg["eval_episodes"])
    )

    if args.output_dir is None:
        output_dir = (
            ROOT
            / "results"
            / "synthetic"
            / "stationary"
            / "uniform"
            / f"seed_{seed}"
        )
    else:
        output_dir = Path(args.output_dir)

    output_dir = ensure_dir(output_dir)

    global_step = 0
    episode_rows = []
    update_losses = []

    for episode in range(episodes):
        state, info = env.reset(seed=seed * 10_000 + episode)
        episode_return = 0.0
        episode_steps = 0
        episode_done = False
        final_terminated = False
        final_truncated = False
        epsilon = exp_cfg["epsilon_start"]

        while not episode_done:
            epsilon = linear_epsilon(
                global_step,
                exp_cfg["epsilon_start"],
                exp_cfg["epsilon_end"],
                exp_cfg["epsilon_decay_steps"],
            )

            action = agent.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, next_info = env.step(action)

            # Episode control and Bellman terminal masking are deliberately distinct.
            # A time-limit truncation ends the rollout but is not an MDP terminal.
            episode_done = terminated or truncated
            replay_terminal = terminated

            replay.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=replay_terminal,
                context_id=info["true_context"],
                context_vector=info["context_vector"],
            )

            if (
                global_step >= int(exp_cfg["learning_starts"])
                and global_step % int(exp_cfg["train_frequency"]) == 0
                and len(replay) >= int(agent_cfg["batch_size"])
            ):
                batch = replay.sample(int(agent_cfg["batch_size"]))
                stats = agent.update(batch)
                update_losses.append(stats.loss)

            state = next_state
            info = next_info
            episode_return += reward
            episode_steps += 1
            global_step += 1

            final_terminated = bool(terminated)
            final_truncated = bool(truncated)

        episode_rows.append(
            {
                "episode": episode,
                "global_step": global_step,
                "context": env.context_name,
                "return": episode_return,
                "length": episode_steps,
                "terminated": final_terminated,
                "truncated": final_truncated,
                "epsilon": epsilon,
                "buffer_size": len(replay),
                "mean_recent_loss": (
                    float(np.mean(update_losses[-100:]))
                    if update_losses
                    else np.nan
                ),
            }
        )

        if (
            episode == 0
            or (episode + 1) % 25 == 0
            or episode + 1 == episodes
        ):
            recent = [r["return"] for r in episode_rows[-25:]]
            print(
                f"episode={episode + 1:4d} "
                f"steps={global_step:7d} "
                f"epsilon={epsilon:.3f} "
                f"mean_return_25={np.mean(recent):.3f} "
                f"device={agent.device}"
            )

    env.close()

    metrics = pd.DataFrame(episode_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    eval_returns = evaluate(
        agent=agent,
        cfg=cfg,
        seed=seed,
        episodes=eval_episodes,
    )

    window = min(50, max(10, episodes // 5))
    early = metrics["return"].head(window)
    late = metrics["return"].tail(window)

    early_mean = float(early.mean())
    late_mean = float(late.mean())
    late_std = float(late.std(ddof=0))
    eval_mean = float(np.mean(eval_returns))
    eval_std = float(np.std(eval_returns))
    late_full_length_fraction = float(
        (metrics["length"].tail(window) == int(cfg["environment"]["max_episode_steps"])).mean()
    )
    late_termination_fraction = float(
        metrics["terminated"].tail(window).mean()
    )
    late_truncation_fraction = float(
        metrics["truncated"].tail(window).mean()
    )

    summary = {
        "seed": seed,
        "device": str(agent.device),
        "code_commit": current_git_sha(),
        "episodes": episodes,
        "total_steps": int(global_step),
        "early_train_return_mean": early_mean,
        "late_train_return_mean": late_mean,
        "late_train_return_std": late_std,
        "return_improvement": late_mean - early_mean,
        "late_full_length_fraction": late_full_length_fraction,
        "late_termination_fraction": late_termination_fraction,
        "late_truncation_fraction": late_truncation_fraction,
        "eval_return_mean": eval_mean,
        "eval_return_std": eval_std,
        "eval_episodes": eval_episodes,
        "sanity_pass": bool(
            late_mean > early_mean
            and eval_mean > early_mean
            and late_full_length_fraction >= 0.90
        ),
    }

    save_json(summary, output_dir / "summary.json")
    save_yaml(cfg, output_dir / "resolved_config.yaml")

    print("\\n=== Stationary sanity summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
