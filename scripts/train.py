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
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
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
        state, _ = env.reset(seed=seed + 100000 + ep)
        done = False
        ret = 0.0
        while not done:
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            done = terminated or truncated
        returns.append(ret)
    env.close()
    return np.asarray(returns, dtype=np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic/stationary.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs/base.yaml", ROOT / args.config)
    seed = int(args.seed)
    set_global_seed(seed)

    env = build_env(cfg)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    a = cfg["agent"]
    e = cfg["experiment"]
    r = cfg["replay"]

    agent = DDQNAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        gamma=a["gamma"],
        learning_rate=a["learning_rate"],
        target_update_steps=a["target_update_steps"],
        hidden_sizes=a["hidden_sizes"],
        grad_clip_norm=a["grad_clip_norm"],
        seed=seed,
    )
    replay = UniformReplayBuffer(capacity=r["capacity"], seed=seed)

    episodes = int(args.episodes or e["episodes"])
    eval_episodes = int(args.eval_episodes or e["eval_episodes"])

    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results/synthetic/stationary/uniform" / f"seed_{seed}"
    )
    output_dir = ensure_dir(output_dir)

    global_step = 0
    rows = []
    losses = []

    for episode in range(episodes):
        state, info = env.reset(seed=seed * 10000 + episode)
        done = False
        ret = 0.0
        ep_steps = 0
        epsilon = e["epsilon_start"]

        while not done:
            epsilon = linear_epsilon(
                global_step, e["epsilon_start"], e["epsilon_end"],
                e["epsilon_decay_steps"]
            )
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            replay.add(
                state, action, reward, next_state, done,
                context_id=info["true_context"],
                context_vector=info["context_vector"],
            )

            if (
                global_step >= int(e["learning_starts"])
                and global_step % int(e["train_frequency"]) == 0
                and len(replay) >= int(a["batch_size"])
            ):
                stats = agent.update(replay.sample(int(a["batch_size"])))
                losses.append(stats.loss)

            state = next_state
            info = next_info
            ret += reward
            ep_steps += 1
            global_step += 1

        rows.append({
            "episode": episode,
            "global_step": global_step,
            "context": env.context_name,
            "return": ret,
            "length": ep_steps,
            "epsilon": epsilon,
            "buffer_size": len(replay),
            "mean_recent_loss": float(np.mean(losses[-100:])) if losses else np.nan,
        })

        if episode == 0 or (episode + 1) % 25 == 0 or episode + 1 == episodes:
            recent = [x["return"] for x in rows[-25:]]
            print(
                f"episode={episode+1:4d} steps={global_step:7d} "
                f"epsilon={epsilon:.3f} mean_return_25={np.mean(recent):.3f} "
                f"device={agent.device}"
            )

    env.close()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    eval_returns = evaluate(agent, cfg, seed, eval_episodes)
    window = min(50, max(10, episodes // 5))
    early = float(metrics["return"].head(window).mean())
    late = float(metrics["return"].tail(window).mean())
    eval_mean = float(eval_returns.mean())

    summary = {
        "seed": seed,
        "device": str(agent.device),
        "git_commit": current_git_sha(),
        "episodes": episodes,
        "total_steps": int(global_step),
        "early_train_return_mean": early,
        "late_train_return_mean": late,
        "return_improvement": late - early,
        "eval_return_mean": eval_mean,
        "eval_return_std": float(eval_returns.std()),
        "eval_episodes": eval_episodes,
        "sanity_pass": bool(late > early and eval_mean > early),
    }

    save_json(summary, output_dir / "summary.json")
    save_yaml(cfg, output_dir / "resolved_config.yaml")

    print("\n=== Stationary sanity summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
