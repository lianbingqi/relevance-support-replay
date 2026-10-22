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
from src.environments.synthetic_control import HiddenContextControlEnv
from src.replay.uniform import UniformReplayBuffer
from src.utils.config import load_yaml
from src.utils.logger import ensure_dir, save_json, save_yaml
from src.utils.seed import set_global_seed


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
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
    start = float(cfg["epsilon_start"])
    end = float(cfg["epsilon_end"])
    decay = float(cfg["epsilon_decay_steps"])
    if decay <= 0:
        return end
    frac = min(max(step / decay, 0.0), 1.0)
    return start + frac * (end - start)


def make_agent(seed, base_cfg):
    a = base_cfg["agent"]
    return DDQNAgent(
        observation_dim=2,
        action_dim=3,
        gamma=a["gamma"],
        learning_rate=a["learning_rate"],
        target_update_steps=a["target_update_steps"],
        hidden_sizes=a["hidden_sizes"],
        grad_clip_norm=a["grad_clip_norm"],
        seed=seed,
        device="cpu",
    )


def train_policy(train_context, seed, contexts, env_cfg, base_cfg, episodes):
    torch.set_num_threads(1)
    set_global_seed(seed)

    env = make_env(contexts, train_context, env_cfg)
    exp = base_cfg["experiment"]
    rep = base_cfg["replay"]
    agent_cfg = base_cfg["agent"]

    agent = make_agent(seed, base_cfg)
    replay = UniformReplayBuffer(rep["capacity"], seed=seed)

    global_step = 0
    rows = []

    for episode in range(int(episodes)):
        state, info = env.reset(seed=seed * 100000 + episode)
        done = False
        ret = 0.0
        length = 0
        final_terminated = False
        final_truncated = False

        while not done:
            eps = epsilon_at(global_step, exp)
            action = agent.act(state, epsilon=eps)
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            replay.add(
                state, action, reward, next_state,
                done=terminated,
                context_id=info["true_context"],
                context_vector=info["context_vector"],
            )

            if (
                global_step >= int(exp["learning_starts"])
                and global_step % int(exp["train_frequency"]) == 0
                and len(replay) >= int(agent_cfg["batch_size"])
            ):
                agent.update(replay.sample(int(agent_cfg["batch_size"])))

            state, info = next_state, next_info
            ret += reward
            length += 1
            global_step += 1
            final_terminated = bool(terminated)
            final_truncated = bool(truncated)

        rows.append({
            "train_context": train_context,
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


def evaluate_agent(agent, test_context, contexts, env_cfg, seed, episodes):
    env = make_env(contexts, test_context, env_cfg)
    returns, lengths, terminals = [], [], []
    base_seed = 1_000_000 + seed * 10_000

    for episode in range(int(episodes)):
        state, _ = env.reset(seed=base_seed + episode)
        done = False
        ret = 0.0
        length = 0
        final_terminal = False

        while not done:
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
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


def policy_grid(agent, train_context, seed, grid_cfg):
    xs = np.linspace(grid_cfg["x_min"], grid_cfg["x_max"], int(grid_cfg["x_points"]))
    vs = np.linspace(grid_cfg["v_min"], grid_cfg["v_max"], int(grid_cfg["v_points"]))
    rows = []
    grid_id = 0

    for x in xs:
        for v in vs:
            state = np.asarray([x, v], dtype=np.float32)
            rows.append({
                "train_context": train_context,
                "seed": seed,
                "grid_id": grid_id,
                "x": float(x),
                "v": float(v),
                "action": int(agent.act(state, epsilon=0.0)),
            })
            grid_id += 1
    return rows


def train_evaluate_job(
    train_context, seed, contexts, env_cfg, base_cfg, cal_cfg, grid_cfg
):
    agent, training_rows = train_policy(
        train_context, seed, contexts, env_cfg, base_cfg, cal_cfg["training_episodes"]
    )

    transfer_rows = []
    for test_context in contexts:
        transfer_rows.append({
            "train_context": train_context,
            "train_seed": seed,
            "test_context": test_context,
            **evaluate_agent(
                agent, test_context, contexts, env_cfg, seed,
                cal_cfg["eval_episodes_per_policy"]
            ),
        })

    return {
        "training_rows": training_rows,
        "transfer_rows": transfer_rows,
        "grid_rows": policy_grid(agent, train_context, seed, grid_cfg),
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
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
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
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(df.values, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Test context")
    ax.set_ylabel("Train context")
    ax.set_title(title)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, format(float(df.iloc[i, j]), fmt),
                    ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/synthetic/drift_calibration.yaml"
    )
    args = parser.parse_args()

    base_cfg = load_yaml(ROOT / "configs/base.yaml")
    cfg = load_yaml(ROOT / args.config)
    reference = cfg["reference_context"]
    cal = cfg["calibration"]
    env_cfg = cfg["environment"]
    grid_cfg = cfg["policy_grid"]
    contexts = cfg["contexts"]
    names = list(contexts.keys())
    seeds = [int(s) for s in cal["train_seeds"]]

    out = ensure_dir(ROOT / "results/synthetic/drift_calibration")
    figs = ensure_dir(ROOT / "figures/synthetic/drift_calibration")
    commit = git_sha()

    random_rows = []
    for name in names:
        random_rows.append({
            "context": name,
            "drift": float(contexts[name]["drift"]),
            **random_baseline(name, contexts, env_cfg, cal["random_baseline_episodes"]),
        })
    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(out / "random_baselines.csv", index=False)

    jobs = [(name, seed) for name in names for seed in seeds]
    max_workers = min(int(cal["max_workers"]), len(jobs))
    print(f"CPU count={os.cpu_count()}, jobs={len(jobs)}, workers={max_workers}")

    training_rows, transfer_rows, grid_rows = [], [], []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                train_evaluate_job,
                name, seed, contexts, env_cfg, base_cfg, cal, grid_cfg
            ): (name, seed)
            for name, seed in jobs
        }

        for completed, future in enumerate(as_completed(futures), start=1):
            name, seed = futures[future]
            result = future.result()
            training_rows.extend(result["training_rows"])
            transfer_rows.extend(result["transfer_rows"])
            grid_rows.extend(result["grid_rows"])
            print(f"[{completed}/{len(jobs)}] finished {name}, seed={seed}")

    training_df = pd.DataFrame(training_rows)
    raw = pd.DataFrame(transfer_rows)
    grid = pd.DataFrame(grid_rows)

    training_df.to_csv(out / "training_metrics.csv", index=False)
    raw.to_csv(out / "transfer_raw.csv", index=False)
    grid.to_csv(out / "policy_grid_actions.csv", index=False)

    mean_matrix = raw.pivot_table(
        index="train_context", columns="test_context",
        values="return_mean", aggfunc="mean"
    ).loc[names, names]

    full_matrix = raw.pivot_table(
        index="train_context", columns="test_context",
        values="full_length_fraction", aggfunc="mean"
    ).loc[names, names]

    random_map = random_df.set_index("context")["random_return_mean"].to_dict()
    normalized = mean_matrix.copy().astype(float)

    for test_context in names:
        self_ret = float(mean_matrix.loc[test_context, test_context])
        rand_ret = float(random_map[test_context])
        denom = self_ret - rand_ret
        if denom <= 0:
            raise RuntimeError(
                f"{test_context}: self-trained policy does not beat random."
            )
        normalized.loc[:, test_context] = (
            mean_matrix.loc[:, test_context] - rand_ret
        ) / denom

    dissimilarity = pd.DataFrame(0.0, index=names, columns=names)
    for i in names:
        for j in names:
            dissimilarity.loc[i, j] = 1.0 - 0.5 * (
                float(normalized.loc[i, j]) + float(normalized.loc[j, i])
            )

    mean_matrix.to_csv(out / "transfer_return_mean.csv")
    full_matrix.to_csv(out / "transfer_full_length_fraction.csv")
    normalized.to_csv(out / "transfer_normalized_retention.csv")
    dissimilarity.to_csv(out / "pairwise_dissimilarity.csv")

    disagreement_rows = []
    for candidate in names:
        if candidate == reference:
            continue
        for seed in seeds:
            ref = grid[
                (grid.train_context == reference) & (grid.seed == seed)
            ][["grid_id", "action"]].rename(columns={"action": "ref_action"})
            cand = grid[
                (grid.train_context == candidate) & (grid.seed == seed)
            ][["grid_id", "action"]].rename(columns={"action": "cand_action"})
            merged = ref.merge(cand, on="grid_id")
            disagreement_rows.append({
                "candidate": candidate,
                "seed": seed,
                "action_disagreement": float(
                    np.mean(merged.ref_action != merged.cand_action)
                ),
            })

    disagreement_by_seed = pd.DataFrame(disagreement_rows)
    disagreement_summary = (
        disagreement_by_seed
        .groupby("candidate")["action_disagreement"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "action_disagreement_mean",
            "std": "action_disagreement_sd",
        })
    )

    disagreement_by_seed.to_csv(
        out / "action_disagreement_by_seed.csv", index=False
    )
    disagreement_summary.to_csv(
        out / "action_disagreement_summary.csv", index=False
    )

    context_rows = []
    for name in names:
        self_rows = raw[
            (raw.train_context == name) & (raw.test_context == name)
        ]
        tail = (
            training_df[training_df.train_context == name]
            .groupby("seed", group_keys=False)
            .tail(50)
        )
        context_rows.append({
            "context": name,
            "drift": float(contexts[name]["drift"]),
            "self_eval_return_mean": float(self_rows.return_mean.mean()),
            "self_eval_return_sd_across_seeds": float(
                self_rows.return_mean.std(ddof=1)
            ),
            "self_eval_full_length_fraction": float(
                self_rows.full_length_fraction.mean()
            ),
            "late_train_return_mean": float(tail["return"].mean()),
            "late_train_full_length_fraction": float(np.mean(
                tail["length"] == int(env_cfg["max_episode_steps"])
            )),
            "random_return_mean": float(random_map[name]),
        })

    context_summary = pd.DataFrame(context_rows)
    context_summary.to_csv(out / "context_summary.csv", index=False)

    dlookup = disagreement_summary.set_index("candidate").to_dict("index")
    ranking_rows = []

    for candidate in names:
        if candidate == reference:
            continue
        ranking_rows.append({
            "candidate": candidate,
            "drift": float(contexts[candidate]["drift"]),
            "candidate_self_return": float(mean_matrix.loc[candidate, candidate]),
            "candidate_self_full_length": float(full_matrix.loc[candidate, candidate]),
            "A_policy_on_candidate_return": float(mean_matrix.loc[reference, candidate]),
            "A_to_candidate_retention": float(normalized.loc[reference, candidate]),
            "candidate_policy_on_A_return": float(mean_matrix.loc[candidate, reference]),
            "candidate_to_A_retention": float(normalized.loc[candidate, reference]),
            "symmetric_dissimilarity_from_A": float(
                dissimilarity.loc[reference, candidate]
            ),
            "action_disagreement_from_A": float(
                dlookup[candidate]["action_disagreement_mean"]
            ),
        })

    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["symmetric_dissimilarity_from_A", "action_disagreement_from_A"],
        ascending=False,
    )
    ranking.to_csv(out / "reference_A_candidate_ranking.csv", index=False)

    save_json({
        "code_commit": commit,
        "reference_context": reference,
        "train_seeds": seeds,
        "training_episodes": int(cal["training_episodes"]),
        "max_workers": max_workers,
        "candidate_contexts": contexts,
        "selection_status": "diagnostics_only_not_frozen",
    }, out / "calibration_metadata.json")

    save_yaml(cfg, out / "resolved_calibration_config.yaml")

    heatmap(
        mean_matrix, "Drift Calibration: Mean Return",
        figs / "transfer_return_mean.png", ".1f"
    )
    heatmap(
        normalized, "Drift Calibration: Normalized Transfer Retention",
        figs / "transfer_normalized_retention.png", ".2f"
    )
    heatmap(
        dissimilarity, "Drift Calibration: Behavioral Dissimilarity",
        figs / "pairwise_dissimilarity.png", ".2f"
    )

    plot_df = ranking.sort_values("drift")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(plot_df["candidate"], plot_df["action_disagreement_from_A"])
    ax.set_ylabel("Greedy action disagreement vs A")
    ax.set_xlabel("Candidate context")
    ax.set_title("Policy Disagreement on Shared State Grid")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figs / "action_disagreement_from_A.png", dpi=180)
    plt.close(fig)

    print("\n=== Drift context learnability ===")
    print(context_summary.to_string(index=False))
    print("\n=== Ranking relative to A ===")
    print(ranking.to_string(index=False))
    print("\nDo not freeze B or D until these diagnostics are reviewed.")


if __name__ == "__main__":
    main()
