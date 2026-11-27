"""Shared-checkpoint post-switch Uniform vs Recent replay, synthetic v2.

Research scope:
  - pre-switch common trajectory uses the EXISTING Recent implementation
  - at B->A or B->D, fork identical learner and buffer
  - after switch, compare Uniform(full-buffer) vs Recent(5000-window)
  - seeds 1/2/3 are DEVELOPMENT, NOT confirmation

The experiment deliberately does not claim an end-to-end Uniform-from-step-zero
baseline. It isolates post-switch replay policy while matching canonical RQ2/RQ3
training/evaluation conventions. Outputs never overwrite historical experiments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.run_oracle_recovery_isolation_v2 import (
    random_A,
    train_common_phase,
)
from scripts.run_novelty_support_pilot_v2 import random_D, train_D_reference
from scripts.train_stationary_v2 import epsilon_at, make_agent, make_env
from src.replay.recent import RecentReplayBuffer
from src.replay.uniform import UniformReplayBuffer
from src.utils.config import deep_merge, load_config, load_yaml
from src.utils.seed import set_global_seed

DEFAULT_CFG = "configs/synthetic_v2/uniform_nonstationary_postswitch_v2.yaml"
METHODS = ("recent", "uniform")
SCHEDULES = ("ABA", "ABD")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_protocol(config_path: str):
    spec_file = ROOT / config_path
    spec = load_yaml(spec_file)
    baseline = spec["uniform_postswitch"]
    assert list(baseline["development_seeds"]) == [1, 2, 3]
    assert list(baseline["reserved_confirmation_seeds"]) == [4, 5, 6]
    assert list(baseline["schedules"]) == list(SCHEDULES)
    assert list(baseline["methods"]) == list(METHODS)
    assert baseline["pre_switch_replay"] == "recent"
    assert baseline["uniform_candidate_pool"] == "all_buffer_transitions"
    assert baseline["recent_candidate_pool"] == "last_5000_transitions"
    assert baseline["full_uniform_from_step_zero"] is False

    rootcfg = load_config(
        ROOT / "configs/base.yaml",
        ROOT / spec["references"]["recurrence_config"],
    )
    cfg = deep_merge(
        rootcfg,
        load_yaml(ROOT / spec["references"]["novelty_config"]),
    )
    # Validate scientific protocol -- do not tune these values here.
    iso, nov, exp = cfg["isolation"], cfg["novelty"], cfg["experiment"]
    assert int(iso["A1_steps"]) == int(nov["A1_steps"]) == 20000
    assert int(iso["B_steps"]) == int(nov["B_steps"]) == 20000
    assert int(iso["A2_steps"]) == int(nov["D_steps"]) == 10000
    assert int(iso["replay_capacity"]) == int(nov["replay_capacity"]) == 100000
    assert int(iso["recent_window"]) == int(nov["recent_window"]) == 5000
    assert int(exp["train_frequency"]) == 4
    assert int(cfg["agent"]["batch_size"]) == 64
    assert int(iso["eval_episodes"]) == int(nov["eval_episodes"]) == 5
    assert iso["A2_eval_steps"] == nov["D_eval_steps"]
    assert int(iso["A2_eval_steps"][0]) == 0
    assert int(iso["A2_eval_steps"][-1]) == 10000
    assert (int(cfg["contexts"]["A"]["rotation_deg"]),
            int(cfg["contexts"]["B"]["rotation_deg"]),
            int(cfg["contexts"]["D"]["rotation_deg"])) == (0, 90, 180)
    return cfg, spec


@torch.no_grad()
def greedy_action(agent, state) -> int:
    # Avoid agent.act(epsilon=0): it consumes its exploration RNG.
    x = torch.as_tensor(np.asarray(state, dtype=np.float32),
                        device=agent.device).unsqueeze(0)
    return int(agent.online(x).argmax(dim=1).item())


def evaluation_base(context: str, seed: int) -> int:
    # Exact reset families from the historical RQ2 (A) and RQ3 (D) scripts.
    if context == "A":
        return 70_000_000 + seed * 10_000
    if context == "D":
        return 170_000_000 + seed * 20_000 + 10_000
    raise ValueError(context)


def evaluate(agent, context: str, seed: int, cfg, passive: bool = False):
    env = make_env(cfg, context)
    n = int(cfg["isolation"]["eval_episodes"] if context == "A"
            else cfg["novelty"]["eval_episodes"])
    returns, safety, noop, distance = [], [], [], []
    try:
        for ep in range(n):
            state, _ = env.reset(seed=evaluation_base(context, seed) + ep)
            ret = 0.0
            count = 0
            noop_count = 0
            distances = []
            while True:
                action = 0 if passive else greedy_action(agent, state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                ret += float(reward)
                count += 1
                noop_count += (action == 0)
                distances.append(float(np.linalg.norm(next_state[:2])))
                state = next_state
                if terminated or truncated:
                    returns.append(ret)
                    safety.append(float(terminated))
                    noop.append(noop_count / count)
                    distance.append(float(np.mean(distances)))
                    break
    finally:
        env.close()
    return dict(eval_return_mean=float(np.mean(returns)),
                eval_return_sd=float(np.std(returns)),
                eval_safety_termination_fraction=float(np.mean(safety)),
                eval_noop_fraction=float(np.mean(noop)),
                eval_mean_distance=float(np.mean(distance)))


def assert_buffer_copy(source, target):
    assert len(source) == len(target)
    assert source.capacity == target.capacity
    assert source.rng.bit_generator.state == target.rng.bit_generator.state
    if len(source):
        assert all(np.array_equal(a.state, b.state) and a.action == b.action
                   for a, b in ((source.buffer[0], target.buffer[0]),
                                (source.buffer[-1], target.buffer[-1])))


def clone_uniform(recent_replay, seed: int):
    uniform = UniformReplayBuffer(capacity=recent_replay.capacity, seed=seed)
    uniform.buffer = copy.deepcopy(recent_replay.buffer)
    uniform.rng.bit_generator.state = copy.deepcopy(
        recent_replay.rng.bit_generator.state)
    assert_buffer_copy(recent_replay, uniform)
    return uniform


def run_post_switch(method, agent, replay, schedule, seed, cfg, global_step):
    context = "A" if schedule == "ABA" else "D"
    iso, nov = cfg["isolation"], cfg["novelty"]
    total_steps = int(iso["A2_steps"] if schedule == "ABA" else nov["D_steps"])
    eval_steps = set(map(int, iso["A2_eval_steps"] if schedule == "ABA"
                             else nov["D_eval_steps"]))
    batch_size = int(cfg["agent"]["batch_size"])
    frequency = int(cfg["experiment"]["train_frequency"])
    eval_zero = evaluate(agent, context, seed, cfg)
    curves = [{"seed": seed, "schedule": schedule, "method": method,
               "phase_step": 0, "global_step": global_step, **eval_zero}]
    sampled_contexts = []
    env = make_env(cfg, context)
    base = (90_000_000 if context == "A" else 175_000_000) + seed * 1_000_000
    state, info = env.reset(seed=base)
    episode_id = 0
    sampled_a = sampled_b = sampled_current = update_count = 0
    try:
        for t in range(1, total_steps + 1):
            action = agent.act(state, epsilon=epsilon_at(global_step, cfg))
            nxt, reward, terminated, truncated, next_info = env.step(action)
            replay.add(state, action, reward, nxt, terminated,
                       context_id=context,
                       context_vector=info["context_vector"])
            if (global_step % frequency == 0 and len(replay) >= batch_size):
                batch = replay.sample(batch_size)
                agent.update(batch)
                update_count += 1
                labels = np.asarray(batch.context_ids)
                sampled_a += int(np.sum(labels == "A"))
                sampled_b += int(np.sum(labels == "B"))
                sampled_current += int(np.sum(labels == context))
            state, info = nxt, next_info
            global_step += 1
            if t in eval_steps:
                curves.append({"seed": seed, "schedule": schedule,
                               "method": method, "phase_step": t,
                               "global_step": global_step,
                               **evaluate(agent, context, seed, cfg)})
            if (terminated or truncated) and t < total_steps:
                episode_id += 1
                state, info = env.reset(seed=base + episode_id)
    finally:
        env.close()
    assert len(curves) == len(eval_steps)
    assert update_count == total_steps // frequency
    denom = max(batch_size * update_count, 1)
    sampled_contexts.append({
        "seed": seed, "schedule": schedule, "method": method,
        "updates": update_count, "sampled_A_fraction": sampled_a / denom,
        "sampled_B_fraction": sampled_b / denom,
        "sampled_current_fraction": sampled_current / denom,
        "sampling_pool": ("full_buffer" if method == "uniform" else "last_5000"),
    })
    return pd.DataFrame(curves), pd.DataFrame(sampled_contexts)


def auc_0_2000(x, y):
    mask = (x >= 0) & (x <= 2000)
    xx, yy = x[mask], y[mask]
    assert len(xx) >= 2 and xx[0] == 0 and xx[-1] == 2000
    # NumPy 1.x / 2.x compatibility.
    integral = np.trapezoid(yy, xx) if hasattr(np, "trapezoid") else np.trapz(yy, xx)
    return float(integral / 2000.0)


def recovery_at(df, threshold: float, consecutive: int):
    scores = df["normalized_score"].to_numpy()
    steps = df["phase_step"].to_numpy()
    ok = scores >= threshold
    for i in range(len(scores) - consecutive + 1):
        if bool(np.all(ok[i:i+consecutive])):
            return int(steps[i])
    return float("nan")


def summarize(curve, seed: int, schedule: str, baseline_return: float,
              reference_return: float, no_op_return: float, cfg):
    denom = reference_return - baseline_return
    if denom <= 0:
        raise RuntimeError(f"Nonpositive normalization denominator {schedule}/{seed}: {denom}")
    curves = curve.copy()
    curves["normalized_score"] = (
        curves["eval_return_mean"] - baseline_return) / denom
    curves["advantage_vs_noop"] = curves["eval_return_mean"] - no_op_return
    metrics = []
    for method in METHODS:
        d = curves[curves.method == method].sort_values("phase_step")
        assert len(d) == len(curves) // 2
        assert d.phase_step.tolist() == sorted(d.phase_step.tolist())
        metrics.append({
            "seed": seed, "schedule": schedule, "method": method,
            "zero_shot_score": float(d.normalized_score.iloc[0]),
            "AUC_0_2000": auc_0_2000(
                d.phase_step.to_numpy(float), d.normalized_score.to_numpy(float)),
            "raw_return_auc_0_2000": auc_0_2000(
                d.phase_step.to_numpy(float), d.eval_return_mean.to_numpy(float)),
            "recovery_step": recovery_at(
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


def save_atomic_csv(df, path: Path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".partial")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def save_atomic_json(payload, path: Path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run_seed(seed: int, cfg, spec, out_dir: Path, config_path: str):
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if list(seed_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing seed outputs in {seed_dir}; "
            "move them aside manually before rerunning.")
    torch.set_num_threads(1)
    set_global_seed(seed)
    iso = cfg["isolation"]
    agent = make_agent(cfg, seed)
    recent = RecentReplayBuffer(capacity=int(iso["replay_capacity"]),
                                recent_window=int(iso["recent_window"]), seed=seed)
    step, A1_evals = train_common_phase(
        agent, recent, "A", int(iso["A1_steps"]), 0, seed, 0, cfg,
        eval_steps=iso["A1_reference_steps"])
    assert step == 20000 and len(A1_evals) == 3
    a1_ref = float(np.mean([x["eval_return_mean"] for x in A1_evals]))
    step, _ = train_common_phase(
        agent, recent, "B", int(iso["B_steps"]), step, seed, 1, cfg)
    assert step == 40000 and len(recent) == 40000

    # Historical Q network and sampler are frozen here; all schedule/method arms fork.
    pre_A = evaluate(agent, "A", seed, cfg)
    pre_D = evaluate(agent, "D", seed, cfg)
    random_A_ref = random_A(seed, cfg)
    random_D_ref = random_D(seed, cfg)
    _, D_ref = train_D_reference(seed, cfg)  # Historical novelty normalization protocol.
    if not a1_ref > random_A_ref:
        raise RuntimeError("Pre-switch A1 reference does not outperform Random A")
    if not D_ref > random_D_ref:
        raise RuntimeError("Stationary D reference does not outperform Random D")

    all_curves, all_metrics, all_samples = [], [], []
    fork_checks = []
    for schedule in SCHEDULES:
        ctx = "A" if schedule == "ABA" else "D"
        base_agents = {m: copy.deepcopy(agent) for m in METHODS}
        base_replays = {"recent": copy.deepcopy(recent),
                        "uniform": clone_uniform(recent, seed)}
        zrs = [evaluate(base_agents[m], ctx, seed, cfg) for m in METHODS]
        if abs(zrs[0]["eval_return_mean"] - zrs[1]["eval_return_mean"]) > 1e-9:
            raise AssertionError("Forked agents have different zero-shot policy returns")
        assert all(torch.equal(p, q) for p, q in zip(
            base_agents["recent"].online.parameters(),
            base_agents["uniform"].online.parameters()))
        assert base_agents["recent"].rng.bit_generator.state == base_agents["uniform"].rng.bit_generator.state
        fork_checks.append({"seed": seed, "schedule": schedule,
                            "fork_zero_return_recent": zrs[0]["eval_return_mean"],
                            "fork_zero_return_uniform": zrs[1]["eval_return_mean"],
                            "fork_initial_parameters_equal": True,
                            "fork_sampler_rng_equal": True})
        branches = []
        for method in METHODS:
            curve, samples = run_post_switch(method, base_agents[method],
                                             base_replays[method], schedule,
                                             seed, cfg, step)
            branches.append(curve)
            all_samples.append(samples)
        paired = pd.concat(branches, ignore_index=True)
        for t, grp in paired.groupby("phase_step"):
            if t == 0 and grp["eval_return_mean"].max() - grp["eval_return_mean"].min() > 1e-9:
                raise AssertionError("Fork zero-step values differ")
        baseline = random_A_ref if ctx == "A" else random_D_ref
        reference = a1_ref if ctx == "A" else D_ref
        passive_return = evaluate(agent, ctx, seed, cfg, passive=True)["eval_return_mean"]
        c, m = summarize(paired, seed, schedule, baseline, reference,
                         passive_return, cfg)
        all_curves.append(c)
        all_metrics.append(m)
        print(f"[seed={seed} {schedule}] " + " | ".join(
            f"{row.method} AUC={row.AUC_0_2000:.4f} recovery={row.recovery_step}"
            for row in m.itertuples()), flush=True)
    manifest = {
        "experiment": "synthetic-v2-shared-checkpoint-postswitch-uniform-vs-recent",
        "methodology": "identical Recent-trained A1/B checkpoint, fork Uniform and Recent at switch",
        "not_full_uniform_from_step_zero": True,
        "seed": seed,
        "schedules": list(SCHEDULES),
        "development_only": True,
        "reserved_seeds_not_run": [4, 5, 6],
        "common_pre_switch_steps": 40000,
        "post_switch_steps_per_branch": 10000,
        "source_commit": git_sha(),
        "source_config": config_path,
        "script_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(ROOT / config_path),
        "runtime": dict(python=platform.python_version(), torch=torch.__version__,
                        numpy=np.__version__, gymnasium=gymnasium.__version__,
                        cuda_available=bool(torch.cuda.is_available()),
                        torch_num_threads=torch.get_num_threads(),
                        device="cpu"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_values": dict(A1_reference=a1_ref, random_A=random_A_ref,
                                 stationary_D=D_ref, random_D=random_D_ref),
        "before_switch_returns": dict(A=pre_A["eval_return_mean"],
                                      D=pre_D["eval_return_mean"]),
    }
    save_atomic_csv(pd.concat(all_curves, ignore_index=True), seed_dir / "curves.csv")
    save_atomic_csv(pd.concat(all_metrics, ignore_index=True), seed_dir / "metrics.csv")
    save_atomic_csv(pd.concat(all_samples, ignore_index=True), seed_dir / "sampling.csv")
    save_atomic_csv(pd.DataFrame(fork_checks), seed_dir / "fork_checks.csv")
    save_atomic_json(manifest, seed_dir / "manifest.json")
    print(f"Saved seed {seed} to {seed_dir}", flush=True)


def aggregate(out_dir: Path):
    missing = [s for s in (1, 2, 3) if not (out_dir / f"seed_{s}" / "manifest.json").exists()]
    if missing:
        raise RuntimeError(f"Need seeds 1, 2, 3 before aggregation. Missing: {missing}")
    curves = pd.concat([pd.read_csv(out_dir / f"seed_{s}" / "curves.csv")
                        for s in (1, 2, 3)], ignore_index=True)
    metrics = pd.concat([pd.read_csv(out_dir / f"seed_{s}" / "metrics.csv")
                         for s in (1, 2, 3)], ignore_index=True)
    sampling = pd.concat([pd.read_csv(out_dir / f"seed_{s}" / "sampling.csv")
                          for s in (1, 2, 3)], ignore_index=True)
    paired = metrics.pivot(index=["seed", "schedule"], columns="method",
                           values="AUC_0_2000").reset_index()
    paired["uniform_minus_recent_AUC"] = paired["uniform"] - paired["recent"]
    descriptive = paired.groupby("schedule").agg(
        mean_uniform_minus_recent=("uniform_minus_recent_AUC", "mean"),
        sd_uniform_minus_recent=("uniform_minus_recent_AUC", "std"),
        n_seeds=("seed", "size")
    ).reset_index()
    out_dir.mkdir(exist_ok=True, parents=True)
    for name, df in (("curves_all_development.csv", curves),
                     ("metrics_all_development.csv", metrics),
                     ("sampling_all_development.csv", sampling),
                     ("paired_auc_development.csv", paired),
                     ("descriptive_summary.csv", descriptive)):
        save_atomic_csv(df, out_dir / name)
    # Cross-protocol audit: previous RQ2 Recent metrics may exhibit execution drift.
    hist_path = ROOT / "results/synthetic_v2/learned_replay_recurrence_3seed/metrics_all_3seeds.csv"
    if hist_path.exists():
        old = pd.read_csv(hist_path)
        old = old.loc[old.method == "recent", ["seed", "AUC_0_2000"]].rename(
            columns={"AUC_0_2000": "historic_RQ2_recent_AUC"})
        now = metrics.loc[(metrics.method == "recent") & (metrics.schedule == "ABA"),
                          ["seed", "AUC_0_2000"]].rename(
            columns={"AUC_0_2000": "this_run_recent_AUC"})
        drift = now.merge(old, on="seed", validate="one_to_one")
        drift["drift_AUC"] = drift.this_run_recent_AUC - drift.historic_RQ2_recent_AUC
        drift["exact_protocol_match_atol_1e_minus_6"] = drift.drift_AUC.abs() <= 1e-6
        save_atomic_csv(drift, out_dir / "historical_recent_drift_audit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharex=True)
    for ax, schedule in zip(axes, SCHEDULES):
        for method in METHODS:
            subs = curves.loc[(curves.schedule == schedule)&(curves.method == method)]
            grp = subs.groupby("phase_step").normalized_score.agg(["mean", "std"]).reset_index()
            xs = grp["phase_step"].to_numpy(dtype=float)
            means = grp["mean"].to_numpy(dtype=float)
            sds = grp["std"].to_numpy(dtype=float)
            ax.plot(xs, means, label=method)
            ax.fill_between(xs, means - sds, means + sds, alpha=.12)
        ax.set(title=schedule, xlabel="Post-switch environment steps",
               ylabel="Normalized evaluation score")
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle("Uniform vs Recent: shared A1/B learner checkpoint (development seeds 1–3)")
    fig.tight_layout()
    fig.savefig(out_dir / "uniform_vs_recent_curves.png", dpi=200)
    plt.close(fig)
    print("\nPer-seed AUC:\n", paired.to_string(index=False))
    print("\nDevelopment summary (descriptive):\n", descriptive.to_string(index=False))
    print("Saved aggregate files to", out_dir)


def smoke_test(cfg):
    torch.set_num_threads(1)
    set_global_seed(1)
    agent = make_agent(cfg, 1)
    r = RecentReplayBuffer(capacity=100000, recent_window=64, seed=1)
    for ctx in ("A", "B"):
        env = make_env(cfg, ctx)
        state, info = env.reset(seed=333)
        for i in range(64):
            a = i % 5
            nxt, reward, terminated, truncated, next_info = env.step(a)
            r.add(state, a, reward, nxt, terminated, context_id=ctx,
                  context_vector=info["context_vector"])
            state, info = nxt, next_info
            if terminated or truncated:
                state, info = env.reset(seed=334 + i)
        env.close()
    u = clone_uniform(r, seed=1)
    recent_batch = r.sample(64)
    uniform_batch = u.sample(64)
    assert len(recent_batch.actions) == len(uniform_batch.actions) == 64
    assert set(recent_batch.context_ids) == {"B"}, "Recent must use the last 64 B samples"
    assert "A" in set(uniform_batch.context_ids), "Uniform must sample from A+B buffer"
    aa, bb = copy.deepcopy(agent), copy.deepcopy(agent)
    assert evaluate(aa, "A", 1, cfg)["eval_return_mean"] == evaluate(bb, "A", 1, cfg)["eval_return_mean"]
    assert aa.rng.bit_generator.state == bb.rng.bit_generator.state
    # Verify that our extended policy diagnostics preserve historical return evaluation.
    from scripts.run_oracle_recovery_isolation_v2 import evaluate as rq2_evaluate
    from scripts.run_novelty_support_pilot_v2 import evaluate as rq3_evaluate
    new_A = evaluate(aa, "A", 1, cfg)["eval_return_mean"]
    new_D = evaluate(aa, "D", 1, cfg)["eval_return_mean"]
    ref_A = rq2_evaluate(aa, "A", 1, cfg)["return_mean"]
    ref_D = rq3_evaluate(aa, "D", 1, cfg)["return_mean"]
    assert abs(new_A - ref_A) < 1e-7, (new_A, ref_A)
    assert abs(new_D - ref_D) < 1e-7, (new_D, ref_D)
    print("PASS: environment, Uniform/Recent support, fork identity, replay RNG, RQ2/RQ3 eval alignment")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CFG)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()
    cfg, spec = load_protocol(args.config)
    # Use CPU even on A100; DDQN network and single environment are tiny, and
    # canonical v2 comparisons use CPU single-thread.
    if torch.cuda.is_available():
        raise RuntimeError("Choose Colab CPU runtime. This experiment intentionally forbids CUDA.")
    torch.set_num_threads(1)
    if args.smoke:
        smoke_test(cfg)
        return
    out_dir = ROOT / spec["uniform_postswitch"]["output_directory"]
    if args.aggregate:
        aggregate(out_dir)
        return
    seeds = args.seeds
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset({1, 2, 3}):
        raise ValueError("Development seeds 1, 2, 3 only. Confirmation seeds 4–6 are reserved.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        run_seed(seed, cfg, spec, out_dir, args.config)


if __name__ == "__main__":
    main()
