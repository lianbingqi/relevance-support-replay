"""Development-only shared-checkpoint Recent vs Uniform vs proportional PER.

Starts with the identical Recent-trained A1(20k)->B(20k) trajectory as the
already merged Uniform baseline, then forks the DDQN and 40k-item buffer at
B->A (recurrence) and B->D (novelty). All three post-switch branches run
10k environment steps with 2500 DDQN updates, using the same reset families.

PER uses alpha=0.6, epsilon=1e-6, beta 0.4->1.0 across the post-switch phase,
initial priorities calculated from the frozen pre-switch learner, and new
items at current maximum priority. Samples are proportional *with replacement*,
and IS weights correct the minibatch Huber loss. This isolates replay sampling
under a common pre-switch checkpoint; it does NOT test PER from step zero.

Development seeds 1/2/3 only. Never overwrite earlier experimental results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys

# Make repository modules importable when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import run_uniform_nonstationary_postswitch_v2 as uniform
from scripts.run_oracle_recovery_isolation_v2 import random_A, train_common_phase
from scripts.run_novelty_support_pilot_v2 import random_D, train_D_reference
from scripts.train_stationary_v2 import epsilon_at, make_agent, make_env
from src.replay.recent import RecentReplayBuffer
from src.replay.prioritized import PrioritizedReplayBuffer
from src.utils.config import load_yaml
from src.utils.seed import set_global_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/synthetic_v2/per_nonstationary_postswitch_v2.yaml"
SCHEDULES = ("ABA", "ABD")
METHODS = ("recent", "uniform", "per")


def load_protocol(path: str):
    spec = load_yaml(ROOT / path)
    p = spec["per_postswitch"]
    assert spec["references"]["uniform_config"] == uniform.DEFAULT_CFG
    cfg, _ = uniform.load_protocol(uniform.DEFAULT_CFG)
    assert list(p["development_seeds"]) == [1, 2, 3]
    assert list(p["reserved_confirmation_seeds"]) == [4, 5, 6]
    assert list(p["methods"]) == list(METHODS)
    assert list(p["schedules"]) == list(SCHEDULES)
    assert p["pre_switch_replay"] == "recent"
    assert p["post_switch_only"] is True
    assert p["initial_priority"] == "abs_td_error_at_fork"
    assert p["priority_sampling"] == "proportional_with_replacement"
    assert p["importance_sampling"] == "weighted_smooth_l1_loss"
    assert p["new_transition_priority"] == "current_max"
    assert p["beta_annealing"] == "linear_post_switch"
    assert float(p["alpha"]) == 0.6
    assert float(p["beta_start"]) == 0.4
    assert float(p["beta_end"]) == 1.0
    assert float(p["priority_epsilon"]) == 1e-6
    assert int(p["td_init_chunk_size"]) == 2048
    return cfg, spec


@torch.no_grad()
def td_abs_on_transitions(agent, transitions, chunk_size: int = 2048):
    """Compute the DDQN pre-update |TD| on a fixed list of transitions."""
    collected = []
    for start in range(0, len(transitions), chunk_size):
        rows = transitions[start:start + chunk_size]
        states = torch.as_tensor(np.stack([t.state for t in rows]),
                                 dtype=torch.float32, device=agent.device)
        nxt = torch.as_tensor(np.stack([t.next_state for t in rows]),
                              dtype=torch.float32, device=agent.device)
        actions = torch.as_tensor([t.action for t in rows],
                                  dtype=torch.long, device=agent.device)
        rewards = torch.as_tensor([t.reward for t in rows],
                                  dtype=torch.float32, device=agent.device)
        dones = torch.as_tensor([t.done for t in rows],
                                dtype=torch.float32, device=agent.device)
        q = agent.online(states).gather(1, actions[:, None]).squeeze(1)
        next_actions = agent.online(nxt).argmax(dim=1)
        nxt_q = agent.target(nxt).gather(1, next_actions[:, None]).squeeze(1)
        target = rewards + agent.gamma * (1.0 - dones) * nxt_q
        collected.append((target - q).abs().cpu().numpy())
    return np.concatenate(collected, axis=0)


def clone_per(recent_buffer, pre_switch_agent, seed: int, cfg, spec):
    p = spec["per_postswitch"]
    buf = PrioritizedReplayBuffer(
        capacity=recent_buffer.capacity, alpha=float(p["alpha"]),
        priority_epsilon=float(p["priority_epsilon"]), seed=seed)
    buf.buffer = copy.deepcopy(recent_buffer.buffer)
    buf.rng.bit_generator.state = copy.deepcopy(recent_buffer.rng.bit_generator.state)
    uniform.assert_buffer_copy(recent_buffer, buf)
    # Initialize priorities only using the frozen checkpoint and data already in
    # the replay buffer. Neither new-context observations nor context labels leak.
    values = td_abs_on_transitions(
        pre_switch_agent, list(buf.buffer), int(p["td_init_chunk_size"]))
    buf.initialize_priorities(values)
    assert len(values) == len(buf) == 40000
    assert np.isclose(buf.tree[1], np.sum(buf.raw_priorities[:len(buf)] ** buf.alpha))
    return buf


def weighted_per_ddqn_update(agent, batch, weights):
    """Identical DDQN target and optimizer, with per-example IS-weighted loss.

    Return absolute TD errors calculated before the gradient update, for PER
    priority refresh. Unit-tested against unweighted DDQN with unit weights.
    """
    dev = agent.device
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=dev)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=dev)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=dev)
    next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=dev)
    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=dev)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=dev)
    if weights_t.shape != rewards.shape or not torch.isfinite(weights_t).all():
        raise ValueError("Malformed importance weights")
    if (weights_t <= 0).any() or (weights_t > 1.0 + 1e-6).any():
        raise ValueError("IS weights must be in (0, 1]")

    q = agent.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = agent.online(next_states).argmax(dim=1)
        next_q = agent.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + agent.gamma * (1.0 - dones) * next_q
    per_sample_loss = F.smooth_l1_loss(q, target, reduction="none")
    loss = (weights_t * per_sample_loss).mean()
    agent.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.online.parameters(), agent.grad_clip_norm)
    agent.optimizer.step()
    agent.update_count += 1
    if agent.update_count % agent.target_update_steps == 0:
        agent.target.load_state_dict(agent.online.state_dict())
    return (target - q.detach()).abs().detach().cpu().numpy()


def run_per_branch(agent, replay, schedule, seed, cfg, global_step, spec):
    context = "A" if schedule == "ABA" else "D"
    iso, nov, p = cfg["isolation"], cfg["novelty"], spec["per_postswitch"]
    steps = int(iso["A2_steps"] if schedule == "ABA" else nov["D_steps"])
    eval_steps = set(map(int, iso["A2_eval_steps"] if schedule == "ABA"
                         else nov["D_eval_steps"]))
    batch_size = int(cfg["agent"]["batch_size"])
    freq = int(cfg["experiment"]["train_frequency"])
    curves = [{"seed": seed, "schedule": schedule, "method": "per",
               "phase_step": 0, "global_step": global_step,
               **uniform.evaluate(agent, context, seed, cfg)}]
    env = make_env(cfg, context)
    base = (90_000_000 if context == "A" else 175_000_000) + seed * 1_000_000
    state, info = env.reset(seed=base)
    episode_id = 0
    sampled_a = sampled_b = sampled_current = update_count = 0
    beta_log, is_log = [], []
    try:
        for t in range(1, steps + 1):
            action = agent.act(state, epsilon=epsilon_at(global_step, cfg))
            nxt, reward, terminated, truncated, ninfo = env.step(action)
            replay.add(state, action, reward, nxt, terminated,
                       context_id=context, context_vector=info["context_vector"])
            if global_step % freq == 0 and len(replay) >= batch_size:
                beta = float(p["beta_start"]) + (
                    float(p["beta_end"]) - float(p["beta_start"])) * t / steps
                batch = replay.sample(batch_size, beta=beta)
                abs_td = weighted_per_ddqn_update(agent, batch, replay.last_is_weights)
                replay.update_priorities(replay.last_sample_indices, abs_td)
                update_count += 1
                labels = np.asarray(batch.context_ids)
                sampled_a += int(np.sum(labels == "A"))
                sampled_b += int(np.sum(labels == "B"))
                sampled_current += int(np.sum(labels == context))
                beta_log.append(beta)
                is_log.append(float(np.mean(replay.last_is_weights)))
            state, info = nxt, ninfo
            global_step += 1
            if t in eval_steps:
                curves.append({"seed": seed, "schedule": schedule,
                               "method": "per", "phase_step": t,
                               "global_step": global_step,
                               **uniform.evaluate(agent, context, seed, cfg)})
            if (terminated or truncated) and t < steps:
                episode_id += 1
                state, info = env.reset(seed=base + episode_id)
    finally:
        env.close()
    assert len(curves) == len(eval_steps)
    assert update_count == steps // freq
    denom = float(max(update_count * batch_size, 1))
    summary = {
        "seed": seed, "schedule": schedule, "method": "per",
        "updates": update_count,
        "sampled_A_fraction": sampled_a / denom,
        "sampled_B_fraction": sampled_b / denom,
        "sampled_current_fraction": sampled_current / denom,
        "sampling_pool": "full_buffer_priority_proportional_with_replacement",
        "beta_first_update": float(beta_log[0]),
        "beta_last_update": float(beta_log[-1]),
        "mean_importance_weight": float(np.mean(is_log)),
        **replay.priority_summary(),
    }
    return pd.DataFrame(curves), pd.DataFrame([summary])


def summarize(curve, seed, schedule, baseline_return, reference_return,
              no_op_return, cfg):
    denom = reference_return - baseline_return
    if denom <= 0:
        raise RuntimeError("Nonpositive normalization denominator")
    curves = curve.copy()
    curves["normalized_score"] = (curves.eval_return_mean - baseline_return) / denom
    curves["advantage_vs_noop"] = curves.eval_return_mean - no_op_return
    metrics = []
    for method in METHODS:
        d = curves.loc[curves.method == method].sort_values("phase_step")
        assert len(d) * len(METHODS) == len(curves)
        metrics.append({
            "seed": seed, "schedule": schedule, "method": method,
            "zero_shot_score": float(d.normalized_score.iloc[0]),
            "AUC_0_2000": uniform.auc_0_2000(
                d.phase_step.to_numpy(float), d.normalized_score.to_numpy(float)),
            "raw_return_auc_0_2000": uniform.auc_0_2000(
                d.phase_step.to_numpy(float), d.eval_return_mean.to_numpy(float)),
            "recovery_step": uniform.recovery_at(
                d, float(cfg["isolation"]["recovery_threshold"]),
                int(cfg["isolation"]["recovery_consecutive_checkpoints"])),
            "final_score": float(d.normalized_score.iloc[-1]),
            "final_raw_return": float(d.eval_return_mean.iloc[-1]),
            "final_advantage_vs_noop": float(d.advantage_vs_noop.iloc[-1]),
            "final_noop_fraction": float(d.eval_noop_fraction.iloc[-1]),
            "random_baseline_return": float(baseline_return),
            "stationary_reference_return": float(reference_return),
            "no_op_return": float(no_op_return),
        })
    return curves, pd.DataFrame(metrics)


def run_seed(seed, cfg, spec, out_dir, config_path):
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if list(seed_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {seed_dir}")
    torch.set_num_threads(1)
    set_global_seed(seed)
    iso = cfg["isolation"]
    agent = make_agent(cfg, seed)
    recent = RecentReplayBuffer(capacity=int(iso["replay_capacity"]),
                                recent_window=int(iso["recent_window"]), seed=seed)
    step, a1_evals = train_common_phase(
        agent, recent, "A", int(iso["A1_steps"]), 0, seed, 0, cfg,
        eval_steps=iso["A1_reference_steps"])
    assert step == 20000 and len(a1_evals) == 3
    a1_ref = float(np.mean([row["eval_return_mean"] for row in a1_evals]))
    step, _ = train_common_phase(
        agent, recent, "B", int(iso["B_steps"]), step, seed, 1, cfg)
    assert step == 40000 and len(recent) == 40000

    pre_A = uniform.evaluate(agent, "A", seed, cfg)
    pre_D = uniform.evaluate(agent, "D", seed, cfg)
    random_A_ref, random_D_ref = random_A(seed, cfg), random_D(seed, cfg)
    _, D_ref = train_D_reference(seed, cfg)
    if not a1_ref > random_A_ref or not D_ref > random_D_ref:
        raise RuntimeError("Stationary reference failed")

    curves, metrics, sampling, forks = [], [], [], []
    for schedule in SCHEDULES:
        ctx = "A" if schedule == "ABA" else "D"
        agents = {m: copy.deepcopy(agent) for m in METHODS}
        replays = {
            "recent": copy.deepcopy(recent),
            "uniform": uniform.clone_uniform(recent, seed),
            "per": clone_per(recent, agent, seed, cfg, spec),
        }
        zero = {m: uniform.evaluate(agents[m], ctx, seed, cfg) for m in METHODS}
        assert max(x["eval_return_mean"] for x in zero.values()) - min(
            x["eval_return_mean"] for x in zero.values()) < 1e-9
        ref_parameters = list(agents["recent"].online.parameters())
        ref_target = list(agents["recent"].target.parameters())
        ref_optimizer = agents["recent"].optimizer.state_dict()
        for m in METHODS:
            assert all(torch.equal(p, q) for p, q in zip(
                ref_parameters, agents[m].online.parameters()))
            assert all(torch.equal(p, q) for p, q in zip(
                ref_target, agents[m].target.parameters()))
            assert agents[m].update_count == agents["recent"].update_count
            other_optimizer = agents[m].optimizer.state_dict()
            assert ref_optimizer["param_groups"] == other_optimizer["param_groups"]
            for state_idx, state in ref_optimizer["state"].items():
                other_state = other_optimizer["state"][state_idx]
                for key, value in state.items():
                    assert (torch.equal(value, other_state[key]) if torch.is_tensor(value)
                            else value == other_state[key]), f"optimizer mismatch: {m}/{key}"
            assert agents[m].rng.bit_generator.state == agents["recent"].rng.bit_generator.state
            assert replays[m].rng.bit_generator.state == replays["recent"].rng.bit_generator.state
            assert len(replays[m]) == len(replays["recent"]) == 40000
        forks.append({"seed": seed, "schedule": schedule,
                      "fork_zero_return_recent": zero["recent"]["eval_return_mean"],
                      "fork_zero_return_uniform": zero["uniform"]["eval_return_mean"],
                      "fork_zero_return_per": zero["per"]["eval_return_mean"],
                      "fork_initial_parameters_equal": True,
                      "fork_agent_rng_equal": True,
                      "fork_sampler_rng_equal": True})
        branches = []
        for method in METHODS:
            if method == "per":
                c, s = run_per_branch(agents[method], replays[method],
                                      schedule, seed, cfg, step, spec)
            else:
                c, s = uniform.run_post_switch(method, agents[method],
                                               replays[method], schedule,
                                               seed, cfg, step)
            branches.append(c)
            sampling.append(s)
        paired = pd.concat(branches, ignore_index=True)
        assert paired.loc[paired.phase_step == 0, "eval_return_mean"].max() - \
               paired.loc[paired.phase_step == 0, "eval_return_mean"].min() < 1e-9
        baseline = random_A_ref if ctx == "A" else random_D_ref
        reference = a1_ref if ctx == "A" else D_ref
        no_op = uniform.evaluate(agent, ctx, seed, cfg, passive=True)["eval_return_mean"]
        c, m = summarize(paired, seed, schedule, baseline, reference, no_op, cfg)
        curves.append(c)
        metrics.append(m)
        print(f"[seed={seed} {schedule}] " + " | ".join(
            f"{row.method} AUC={row.AUC_0_2000:.4f}, recovery={row.recovery_step}"
            for row in m.itertuples()), flush=True)

    manifest = {
        "experiment": "synthetic-v2-shared-checkpoint-postswitch-recent-uniform-per",
        "seed": seed, "development_only": True,
        "reserved_confirmation_seeds_not_run": [4, 5, 6],
        "schedules": list(SCHEDULES), "methods": list(METHODS),
        "common_pre_switch_steps": 40000,
        "post_switch_steps_per_branch": 10000,
        "not_full_PER_from_step_zero": True,
        "initial_priority": "absolute TD error from pre-switch frozen DDQN",
        "per_settings": spec["per_postswitch"],
        "source_commit": uniform.git_sha(),
        "source_config": config_path,
        "script_sha256": uniform.sha256(Path(__file__)),
        "config_sha256": uniform.sha256(ROOT / config_path),
        "per_module_sha256": uniform.sha256(ROOT / "src/replay/prioritized.py"),
        "uniform_script_sha256": uniform.sha256(Path(uniform.__file__)),
        "runtime": dict(python=platform.python_version(), numpy=np.__version__,
                        torch=torch.__version__, gymnasium=gymnasium.__version__,
                        cuda_available=bool(torch.cuda.is_available()),
                        torch_num_threads=torch.get_num_threads(), device="cpu"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_values": dict(A1_reference=a1_ref, random_A=random_A_ref,
                                 stationary_D=D_ref, random_D=random_D_ref),
        "before_switch_returns": dict(A=pre_A["eval_return_mean"],
                                      D=pre_D["eval_return_mean"]),
    }
    uniform.save_atomic_csv(pd.concat(curves, ignore_index=True), seed_dir / "curves.csv")
    uniform.save_atomic_csv(pd.concat(metrics, ignore_index=True), seed_dir / "metrics.csv")
    uniform.save_atomic_csv(pd.concat(sampling, ignore_index=True), seed_dir / "sampling.csv")
    uniform.save_atomic_csv(pd.DataFrame(forks), seed_dir / "fork_checks.csv")
    uniform.save_atomic_json(manifest, seed_dir / "manifest.json")
    print("Saved", seed_dir, flush=True)


def aggregate(out_dir):
    assert all((out_dir / f"seed_{s}" / "manifest.json").exists()
               for s in (1, 2, 3)), "All three seeds required"
    curves = pd.concat([pd.read_csv(out_dir / f"seed_{s}" / "curves.csv")
                        for s in (1, 2, 3)], ignore_index=True)
    metrics = pd.concat([pd.read_csv(out_dir / f"seed_{s}" / "metrics.csv")
                         for s in (1, 2, 3)], ignore_index=True)
    samples = pd.concat([pd.read_csv(out_dir / f"seed_{s}" / "sampling.csv")
                         for s in (1, 2, 3)], ignore_index=True)
    forks = pd.concat([pd.read_csv(out_dir / f"seed_{s}" / "fork_checks.csv")
                       for s in (1, 2, 3)], ignore_index=True)
    assert len(metrics) == 3 * 2 * 3 and len(curves) == 3 * 2 * 3 * 20
    assert forks.fork_initial_parameters_equal.all()
    piv = metrics.pivot(index=["seed", "schedule"], columns="method",
                        values="AUC_0_2000").reset_index()
    for name in ("uniform", "per"):
        piv[f"{name}_minus_recent_AUC"] = piv[name] - piv["recent"]
    piv["per_minus_uniform_AUC"] = piv["per"] - piv["uniform"]
    summary = piv.groupby("schedule").agg(
        per_minus_recent_mean=("per_minus_recent_AUC", "mean"),
        per_minus_recent_sd=("per_minus_recent_AUC", "std"),
        per_minus_uniform_mean=("per_minus_uniform_AUC", "mean"),
        per_minus_uniform_sd=("per_minus_uniform_AUC", "std"),
        n_seeds=("seed", "size"),
    ).reset_index()
    for name, df in (("curves_all_development.csv", curves),
                     ("metrics_all_development.csv", metrics),
                     ("sampling_all_development.csv", samples),
                     ("fork_checks_all_development.csv", forks),
                     ("paired_auc_development.csv", piv),
                     ("descriptive_summary.csv", summary)):
        uniform.save_atomic_csv(df, out_dir / name)

    # New, independent execution compared against last merged Uniform/Recent
    # results. A cross-run drift is recorded, never tuned away post-outcome.
    hist = ROOT / "results/synthetic_v2/uniform_nonstationary_postswitch_v2/metrics_all_development.csv"
    if hist.exists():
        old = pd.read_csv(hist)
        old = old.loc[old.method.isin(("recent", "uniform")),
                      ["seed", "schedule", "method", "AUC_0_2000", "zero_shot_score"]]
        joined = metrics.loc[metrics.method.isin(("recent", "uniform")),
                             ["seed", "schedule", "method", "AUC_0_2000", "zero_shot_score"]].merge(
            old, on=["seed", "schedule", "method"], suffixes=("_this_run", "_prior_uniform_run"),
            validate="one_to_one")
        joined["absolute_AUC_difference"] = (
            joined["AUC_0_2000_this_run"] - joined["AUC_0_2000_prior_uniform_run"]).abs()
        joined["absolute_zero_shot_difference"] = (
            joined["zero_shot_score_this_run"] - joined["zero_shot_score_prior_uniform_run"]).abs()
        joined["historical_metric_match_atol_1e_minus_6"] = (
            (joined.absolute_AUC_difference <= 1e-6) &
            (joined.absolute_zero_shot_difference <= 1e-6))
        uniform.save_atomic_csv(joined, out_dir / "prior_uniform_cross_run_drift_audit.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)
    for ax, schedule in zip(axes, SCHEDULES):
        for method in METHODS:
            sub = curves.loc[(curves.schedule == schedule) & (curves.method == method)]
            d = sub.groupby("phase_step").normalized_score.agg(["mean", "std"]).reset_index()
            x, y, std = d.phase_step.to_numpy(float), d["mean"].to_numpy(float), d["std"].to_numpy(float)
            ax.plot(x, y, label=method)
            ax.fill_between(x, y - std, y + std, alpha=0.12)
        ax.set(title=schedule, xlabel="Post-switch environment steps",
               ylabel="Normalized evaluation score")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Post-switch Recent / Uniform / PER (development seeds 1–3)")
    fig.tight_layout()
    fig.savefig(out_dir / "per_vs_uniform_recent_curves.png", dpi=200)
    plt.close(fig)
    print("\nPaired AUC by seed and schedule:\n", piv.to_string(index=False))
    print("\nDevelopment summary:\n", summary.to_string(index=False))
    print("Saved aggregate results to", out_dir)


def smoke_test(cfg, spec):
    uniform.smoke_test(cfg)
    torch.set_num_threads(1)
    set_global_seed(1)
    a = make_agent(cfg, 1)
    b = RecentReplayBuffer(capacity=100000, recent_window=5000, seed=1)
    for ctx in ("A", "B"):
        env = make_env(cfg, ctx)
        s, info = env.reset(seed=200)
        for t in range(64):
            action = t % 5
            ns, reward, term, trunc, ninfo = env.step(action)
            b.add(s, action, reward, ns, term, context_id=ctx,
                  context_vector=info["context_vector"])
            s, info = ns, ninfo
            if term or trunc:
                s, info = env.reset(seed=201 + t)
        env.close()
    # Small separate test of priority initialization and weighted optimization.
    p = PrioritizedReplayBuffer(capacity=256, alpha=0.6, seed=1)
    p.buffer = copy.deepcopy(b.buffer)
    p.initialize_priorities(td_abs_on_transitions(a, list(p.buffer)))
    batch = p.sample(64, beta=0.4)
    before = a.update_count
    abs_td = weighted_per_ddqn_update(a, batch, p.last_is_weights)
    p.update_priorities(p.last_sample_indices, abs_td)
    assert a.update_count == before + 1
    assert len(batch.actions) == 64 and np.isfinite(p.tree[1])
    print("PASS: PER priorities, sampling, weighted DDQN updates, environment")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default=DEFAULT_CONFIG)
    pa.add_argument("--seeds", type=int, nargs="+", default=[1])
    pa.add_argument("--smoke", action="store_true")
    pa.add_argument("--aggregate", action="store_true")
    args = pa.parse_args()
    cfg, spec = load_protocol(args.config)
    if torch.cuda.is_available():
        raise RuntimeError("Use CPU Colab runtime to match previous experiments")
    torch.set_num_threads(1)
    if args.smoke:
        smoke_test(cfg, spec)
        return
    out_dir = ROOT / spec["per_postswitch"]["output_directory"]
    if args.aggregate:
        aggregate(out_dir)
        return
    seeds = list(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in (1, 2, 3) for s in seeds):
        raise ValueError("Only development seeds 1,2,3 are allowed; reserve 4,5,6")
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        run_seed(seed, cfg, spec, out_dir, args.config)


if __name__ == "__main__":
    main()
