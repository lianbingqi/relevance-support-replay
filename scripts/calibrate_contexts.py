from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.ddqn import DDQNAgent
from src.environments.synthetic_control import HiddenContextControlEnv
from src.replay.uniform import UniformReplayBuffer
from src.utils.config import load_yaml
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def make_env(contexts, name, cfg):
    return HiddenContextControlEnv(
        contexts=contexts,
        context_name=name,
        max_episode_steps=cfg["max_episode_steps"],
        noise_std=cfg["noise_std"],
        init_x_range=cfg["init_x_range"],
        init_v_range=cfg["init_v_range"],
        x_limit=cfg["x_limit"],
        v_limit=cfg["v_limit"],
    )


def epsilon_at(step, cfg):
    start = cfg["epsilon_start"]
    end = cfg["epsilon_end"]
    decay = cfg["epsilon_decay_steps"]
    frac = min(max(step / float(decay), 0.0), 1.0)
    return float(start + frac * (end - start))


def train_policy(context_name, seed, contexts, env_cfg, base_cfg, episodes):
    set_global_seed(seed)
    env = make_env(contexts, context_name, env_cfg)

    agent_cfg = base_cfg["agent"]
    exp_cfg = base_cfg["experiment"]
    replay_cfg = base_cfg["replay"]

    agent = DDQNAgent(
        observation_dim=2,
        action_dim=3,
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

    global_step = 0
    rows = []

    for episode in range(int(episodes)):
        state, info = env.reset(seed=seed * 100000 + episode)
        episode_done = False
        ret = 0.0
        length = 0
        final_terminated = False
        final_truncated = False

        while not episode_done:
            eps = epsilon_at(global_step, exp_cfg)
            action = agent.act(state, epsilon=eps)
            next_state, reward, terminated, truncated, next_info = env.step(action)

            episode_done = terminated or truncated

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
                global_step >= int(exp_cfg["learning_starts"])
                and global_step % int(exp_cfg["train_frequency"]) == 0
                and len(replay) >= int(agent_cfg["batch_size"])
            ):
                agent.update(
                    replay.sample(int(agent_cfg["batch_size"]))
                )

            state = next_state
            info = next_info
            ret += reward
            length += 1
            global_step += 1
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)

        rows.append({
            "train_context": context_name,
            "seed": seed,
            "episode": episode,
            "global_step": global_step,
            "return": ret,
            "length": length,
            "terminated": final_terminated,
            "truncated": final_truncated,
        })

    env.close()
    return agent, rows


def evaluate_agent(agent, context_name, contexts, env_cfg, seed, episodes):
    env = make_env(contexts, context_name, env_cfg)
    returns, lengths, terminals = [], [], []
    base_seed = 1_000_000 + seed * 10_000

    for ep in range(int(episodes)):
        state, _ = env.reset(seed=base_seed + ep)
        done = False
        ret = 0.0
        length = 0
        final_terminal = False

        while not done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ret += reward
            length += 1
            final_terminal = bool(terminated)

        returns.append(ret)
        lengths.append(length)
        terminals.append(float(final_terminal))

    env.close()

    return {
        "return_mean": float(np.mean(returns)),
        "return_sd": float(np.std(returns)),
        "full_length_fraction": float(np.mean(
            np.asarray(lengths) == int(env_cfg["max_episode_steps"])
        )),
        "termination_fraction": float(np.mean(terminals)),
    }


def random_baseline(context_name, contexts, env_cfg, episodes):
    seed = 314159
    rng = np.random.default_rng(seed)
    env = make_env(contexts, context_name, env_cfg)

    returns, lengths = [], []

    for ep in range(int(episodes)):
        state, _ = env.reset(seed=seed + ep)
        done = False
        ret = 0.0
        length = 0

        while not done:
            action = int(rng.integers(env.action_space.n))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ret += reward
            length += 1

        returns.append(ret)
        lengths.append(length)

    env.close()

    return {
        "random_return_mean": float(np.mean(returns)),
        "random_return_sd": float(np.std(returns)),
        "random_full_length_fraction": float(np.mean(
            np.asarray(lengths) == int(env_cfg["max_episode_steps"])
        )),
    }


def heatmap(df, title, path, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(df.values, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Test context")
    ax.set_ylabel("Train context")
    ax.set_title(title)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(
                j, i, format(float(df.iloc[i, j]), fmt),
                ha="center", va="center", fontsize=7
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic/context_calibration.yaml",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(ROOT / "configs/base.yaml")
    cfg = load_yaml(ROOT / args.config)

    contexts = cfg["contexts"]
    names = list(contexts.keys())
    reference = cfg["reference_context"]
    cal = cfg["calibration"]
    env_cfg = cfg["environment"]

    out = ensure_dir(ROOT / "results/synthetic/calibration")
    figs = ensure_dir(ROOT / "figures/synthetic/calibration")
    commit = git_sha()

    random_rows = []
    for name in names:
        random_rows.append({
            "context": name,
            "rho": float(contexts[name]["rho"]),
            "beta": float(contexts[name]["beta"]),
            **random_baseline(
                name, contexts, env_cfg,
                cal["random_baseline_episodes"]
            ),
        })
    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(out / "random_baselines.csv", index=False)

    training_rows = []
    transfer_rows = []

    for train_context in names:
        for seed in cal["train_seeds"]:
            print("=" * 72)
            print(f"Training context={train_context}, seed={seed}")

            agent, rows = train_policy(
                train_context,
                int(seed),
                contexts,
                env_cfg,
                base_cfg,
                cal["training_episodes"],
            )
            training_rows.extend(rows)

            for test_context in names:
                stats = evaluate_agent(
                    agent,
                    test_context,
                    contexts,
                    env_cfg,
                    int(seed),
                    cal["eval_episodes_per_policy"],
                )

                transfer_rows.append({
                    "code_commit": commit,
                    "train_context": train_context,
                    "train_seed": int(seed),
                    "test_context": test_context,
                    **stats,
                })

    training_df = pd.DataFrame(training_rows)
    transfer_raw = pd.DataFrame(transfer_rows)

    training_df.to_csv(out / "training_metrics.csv", index=False)
    transfer_raw.to_csv(out / "transfer_raw.csv", index=False)

    mean_matrix = transfer_raw.pivot_table(
        index="train_context",
        columns="test_context",
        values="return_mean",
        aggfunc="mean",
    ).loc[names, names]

    sd_matrix = transfer_raw.pivot_table(
        index="train_context",
        columns="test_context",
        values="return_mean",
        aggfunc="std",
    ).loc[names, names]

    full_matrix = transfer_raw.pivot_table(
        index="train_context",
        columns="test_context",
        values="full_length_fraction",
        aggfunc="mean",
    ).loc[names, names]

    random_map = (
        random_df.set_index("context")["random_return_mean"].to_dict()
    )

    normalized = mean_matrix.copy().astype(float)

    for test_context in names:
        self_return = float(
            mean_matrix.loc[test_context, test_context]
        )
        random_return = float(random_map[test_context])
        denominator = self_return - random_return

        if denominator <= 0:
            raise RuntimeError(
                f"{test_context}: self policy does not beat random baseline."
            )

        normalized.loc[:, test_context] = (
            mean_matrix.loc[:, test_context] - random_return
        ) / denominator

    dissimilarity = pd.DataFrame(
        0.0, index=names, columns=names
    )

    for i in names:
        for j in names:
            dissimilarity.loc[i, j] = (
                1.0 - 0.5 * (
                    float(normalized.loc[i, j])
                    + float(normalized.loc[j, i])
                )
            )

    mean_matrix.to_csv(out / "transfer_return_mean.csv")
    sd_matrix.to_csv(out / "transfer_return_sd_across_seeds.csv")
    full_matrix.to_csv(out / "transfer_full_length_fraction.csv")
    normalized.to_csv(out / "transfer_normalized_retention.csv")
    dissimilarity.to_csv(out / "pairwise_dissimilarity.csv")

    summary_rows = []

    for name in names:
        self_rows = transfer_raw[
            (transfer_raw["train_context"] == name)
            & (transfer_raw["test_context"] == name)
        ]

        tail = (
            training_df[
                training_df["train_context"] == name
            ]
            .groupby("seed", group_keys=False)
            .tail(50)
        )

        summary_rows.append({
            "context": name,
            "rho": float(contexts[name]["rho"]),
            "beta": float(contexts[name]["beta"]),
            "self_eval_return_mean": float(
                self_rows["return_mean"].mean()
            ),
            "self_eval_return_sd_across_seeds": float(
                self_rows["return_mean"].std(ddof=1)
            ),
            "self_eval_full_length_fraction": float(
                self_rows["full_length_fraction"].mean()
            ),
            "late_train_return_mean": float(
                tail["return"].mean()
            ),
            "late_train_full_length_fraction": float(np.mean(
                tail["length"] == int(env_cfg["max_episode_steps"])
            )),
            "random_return_mean": float(random_map[name]),
        })

    context_summary = pd.DataFrame(summary_rows)
    context_summary.to_csv(out / "context_summary.csv", index=False)

    ranking_rows = []

    for name in names:
        if name == reference:
            continue

        ranking_rows.append({
            "candidate": name,
            "rho": float(contexts[name]["rho"]),
            "beta": float(contexts[name]["beta"]),
            "candidate_self_return": float(
                mean_matrix.loc[name, name]
            ),
            "candidate_self_full_length": float(
                full_matrix.loc[name, name]
            ),
            "A_policy_on_candidate_return": float(
                mean_matrix.loc[reference, name]
            ),
            "A_to_candidate_retention": float(
                normalized.loc[reference, name]
            ),
            "candidate_policy_on_A_return": float(
                mean_matrix.loc[name, reference]
            ),
            "candidate_to_A_retention": float(
                normalized.loc[name, reference]
            ),
            "symmetric_dissimilarity_from_A": float(
                dissimilarity.loc[reference, name]
            ),
        })

    ranking = pd.DataFrame(ranking_rows).sort_values(
        "symmetric_dissimilarity_from_A",
        ascending=False,
    )
    ranking.to_csv(
        out / "reference_A_candidate_ranking.csv",
        index=False,
    )

    save_json({
        "code_commit": commit,
        "reference_context": reference,
        "train_seeds": cal["train_seeds"],
        "training_episodes": cal["training_episodes"],
        "eval_episodes_per_policy": cal["eval_episodes_per_policy"],
        "random_baseline_episodes": cal["random_baseline_episodes"],
        "candidate_contexts": contexts,
        "selection_status": "diagnostics_only_not_frozen",
    }, out / "calibration_metadata.json")

    save_yaml(cfg, out / "resolved_calibration_config.yaml")

    heatmap(
        mean_matrix,
        "Cross-Context Transfer: Mean Return",
        figs / "transfer_return_mean.png",
        ".1f",
    )
    heatmap(
        normalized,
        "Cross-Context Transfer: Normalized Retention",
        figs / "transfer_normalized_retention.png",
        ".2f",
    )
    heatmap(
        dissimilarity,
        "Behavioral Dissimilarity Between Contexts",
        figs / "pairwise_dissimilarity.png",
        ".2f",
    )

    print("\n=== Context summary ===")
    print(context_summary.to_string(index=False))
    print("\n=== Ranking relative to A ===")
    print(ranking.to_string(index=False))
    print("\nDo not freeze B or D until these results are reviewed.")


if __name__ == "__main__":
    main()
