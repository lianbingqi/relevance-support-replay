from __future__ import annotations

import argparse, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.train_stationary_v2 import make_env, train_one
from src.utils.config import load_config
from src.utils.logger import ensure_dir, save_json, save_yaml


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def evaluate_cross(agent, test_context, seed, cfg):
    env = make_env(cfg, test_context)
    n = int(cfg["stationary"]["eval_episodes"])
    returns, lengths, safety = [], [], []
    base_seed = 7_000_000 + seed * 10_000

    for ep in range(n):
        state, _ = env.reset(seed=base_seed + ep)
        done = False
        ret = 0.0
        length = 0
        final_terminated = False

        while not done:
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
            length += 1
            final_terminated = bool(terminated)

        returns.append(ret)
        lengths.append(length)
        safety.append(float(final_terminated))

    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_sd": float(np.std(returns)),
        "full_length_fraction": float(np.mean(
            np.asarray(lengths) == int(cfg["environment"]["max_episode_steps"])
        )),
        "safety_termination_fraction": float(np.mean(safety)),
    }


def probe_states(cfg):
    m = cfg["manipulation"]
    rng = np.random.default_rng(int(m["probe_seed"]))
    n = int(m["probe_states"])
    p = float(m["probe_position_range"])
    v = float(m["probe_velocity_range"])
    positions = rng.uniform(-p, p, size=(n, 2))
    velocities = rng.uniform(-v, v, size=(n, 2))
    return np.concatenate([positions, velocities], axis=1).astype(np.float32)


def run_job(train_context, seed, cfg):
    torch.set_num_threads(1)
    agent, _ = train_one(train_context, seed, cfg)

    transfer_rows = []
    for test_context in cfg["contexts"]:
        transfer_rows.append({
            "train_context": train_context,
            "train_seed": seed,
            "test_context": test_context,
            **evaluate_cross(agent, test_context, seed, cfg),
        })

    states = probe_states(cfg)
    actions = np.asarray(
        [agent.act(s, epsilon=0.0) for s in states],
        dtype=np.int64,
    )

    probe_rows = [
        {
            "train_context": train_context,
            "seed": seed,
            "probe_id": i,
            "action": int(a),
        }
        for i, a in enumerate(actions)
    ]

    return transfer_rows, probe_rows


def random_baseline(context_name, cfg):
    env = make_env(cfg, context_name)
    n = int(cfg["stationary"]["random_baseline_episodes"])
    rng = np.random.default_rng(9_000_000)
    returns = []

    for ep in range(n):
        state, _ = env.reset(seed=9_000_000 + ep)
        done = False
        ret = 0.0
        while not done:
            action = int(rng.integers(env.action_space.n))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
        returns.append(ret)

    env.close()
    return {
        "context": context_name,
        "random_return_mean": float(np.mean(returns)),
        "random_return_sd": float(np.std(returns)),
    }


def save_heatmap(df, title, path, fmt):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(df.values, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Test context")
    ax.set_ylabel("Policy trained in")
    ax.set_title(title)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, format(float(df.iloc[i, j]), fmt),
                    ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/synthetic_v2/manipulation.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs/base.yaml", ROOT / args.config)
    out = ensure_dir(ROOT / "results/synthetic_v2/manipulation")
    figs = ensure_dir(ROOT / "figures/synthetic_v2/manipulation")

    commit = git_sha()
    contexts = list(cfg["contexts"])
    seeds = [int(x) for x in cfg["stationary"]["train_seeds"]]

    random_df = pd.DataFrame([
        random_baseline(c, cfg) for c in contexts
    ])
    random_df.to_csv(out / "random_baselines.csv", index=False)

    jobs = [(c, s) for c in contexts for s in seeds]
    max_workers = min(int(cfg["stationary"]["max_workers"]), len(jobs))
    transfer_rows, probe_rows = [], []

    print(f"CPU count={os.cpu_count()}, jobs={len(jobs)}, workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_job, c, s, cfg): (c, s)
            for c, s in jobs
        }
        for k, f in enumerate(as_completed(futures), start=1):
            c, s = futures[f]
            trows, prows = f.result()
            transfer_rows.extend(trows)
            probe_rows.extend(prows)
            print(f"[{k}/{len(jobs)}] finished {c}, seed={s}")

    raw = pd.DataFrame(transfer_rows)
    probes = pd.DataFrame(probe_rows)
    raw.to_csv(out / "transfer_by_seed.csv", index=False)
    probes.to_csv(out / "probe_actions.csv", index=False)

    mean_matrix = raw.pivot_table(
        index="train_context", columns="test_context",
        values="return_mean", aggfunc="mean"
    ).loc[contexts, contexts]

    sd_matrix = raw.pivot_table(
        index="train_context", columns="test_context",
        values="return_mean", aggfunc="std"
    ).loc[contexts, contexts]

    safety_matrix = raw.pivot_table(
        index="train_context", columns="test_context",
        values="safety_termination_fraction", aggfunc="mean"
    ).loc[contexts, contexts]

    random_map = random_df.set_index("context")["random_return_mean"].to_dict()
    normalized = mean_matrix.copy().astype(float)

    for test_context in contexts:
        self_ret = float(mean_matrix.loc[test_context, test_context])
        rand_ret = float(random_map[test_context])
        denom = self_ret - rand_ret
        if denom <= 0:
            raise RuntimeError(f"{test_context}: self policy does not beat random")
        normalized.loc[:, test_context] = (
            mean_matrix.loc[:, test_context] - rand_ret
        ) / denom

    mean_matrix.to_csv(out / "transfer_return_mean.csv")
    sd_matrix.to_csv(out / "transfer_return_sd_across_seeds.csv")
    safety_matrix.to_csv(out / "transfer_safety_termination.csv")
    normalized.to_csv(out / "transfer_normalized_retention.csv")

    pairs = [("A", "B"), ("A", "D"), ("B", "D")]
    pair_rows = []

    for left, right in pairs:
        for seed in seeds:
            a = probes[
                (probes.train_context == left) & (probes.seed == seed)
            ][["probe_id", "action"]].rename(columns={"action": "a"})
            b = probes[
                (probes.train_context == right) & (probes.seed == seed)
            ][["probe_id", "action"]].rename(columns={"action": "b"})
            joined = a.merge(b, on="probe_id")
            pair_rows.append({
                "left_context": left,
                "right_context": right,
                "seed": seed,
                "action_disagreement": float(np.mean(joined.a != joined.b)),
            })

    pair_by_seed = pd.DataFrame(pair_rows)
    pair_summary = (
        pair_by_seed
        .groupby(["left_context", "right_context"])["action_disagreement"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "action_disagreement_mean",
            "std": "action_disagreement_sd",
        })
    )

    pair_by_seed.to_csv(out / "action_disagreement_by_seed.csv", index=False)
    pair_summary.to_csv(out / "action_disagreement_summary.csv", index=False)

    checks = []
    criteria = cfg["criteria"]

    for test_context in contexts:
        self_ret = float(mean_matrix.loc[test_context, test_context])
        off = {
            train_context: float(mean_matrix.loc[train_context, test_context])
            for train_context in contexts
            if train_context != test_context
        }
        best_off_context = max(off, key=off.get)
        best_off_return = float(off[best_off_context])
        best_off_retention = float(
            normalized.loc[best_off_context, test_context]
        )
        checks.append({
            "test_context": test_context,
            "self_return": self_ret,
            "best_off_context": best_off_context,
            "best_off_return": best_off_return,
            "diagonal_advantage": self_ret - best_off_return,
            "best_off_normalized_retention": best_off_retention,
            "diagonal_dominance_pass": bool(self_ret > best_off_return),
            "retention_pass": bool(
                best_off_retention <= float(criteria["max_off_context_retention"])
            ),
        })

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out / "context_manipulation_checks.csv", index=False)

    all_diag = bool(checks_df["diagonal_dominance_pass"].all())
    all_ret = bool(checks_df["retention_pass"].all())
    all_dis = bool(
        (
            pair_summary["action_disagreement_mean"]
            >= float(criteria["min_pairwise_action_disagreement"])
        ).all()
    )
    manipulation_pass = bool(all_diag and all_ret and all_dis)

    save_json({
        "code_commit": commit,
        "contexts": cfg["contexts"],
        "train_seeds": seeds,
        "criteria": criteria,
        "all_diagonal_dominance_pass": all_diag,
        "all_retention_pass": all_ret,
        "all_pairwise_action_disagreement_pass": all_dis,
        "manipulation_pass": manipulation_pass,
        "context_angles_frozen": True,
        "next_step_if_pass": "v2_03_oracle_recurrence",
    }, out / "manipulation_metadata.json")

    save_yaml(cfg, out / "resolved_config.yaml")

    save_heatmap(
        mean_matrix,
        "Synthetic v2 Cross-Context Return",
        figs / "transfer_return_mean.png",
        ".1f",
    )
    save_heatmap(
        normalized,
        "Synthetic v2 Normalized Transfer Retention",
        figs / "transfer_normalized_retention.png",
        ".2f",
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = pair_summary.left_context + " vs " + pair_summary.right_context
    ax.bar(
        labels,
        pair_summary.action_disagreement_mean,
        yerr=pair_summary.action_disagreement_sd,
        capsize=4,
    )
    ax.axhline(
        float(criteria["min_pairwise_action_disagreement"]),
        linestyle="--",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Greedy action disagreement")
    ax.set_title("Policy Separation on Shared Probe States")
    fig.tight_layout()
    fig.savefig(figs / "action_disagreement.png", dpi=180)
    plt.close(fig)

    print("\n=== Cross-context mean return ===")
    print(mean_matrix.to_string())
    print("\n=== Normalized retention ===")
    print(normalized.to_string())
    print("\n=== Context checks ===")
    print(checks_df.to_string(index=False))
    print("\n=== Pairwise action disagreement ===")
    print(pair_summary.to_string(index=False))
    print("\nMANIPULATION PASS:", manipulation_pass)


if __name__ == "__main__":
    main()
