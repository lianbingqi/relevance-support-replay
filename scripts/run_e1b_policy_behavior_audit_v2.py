"""E1b: policy behavior audit of the UNCHANGED E1 experiment.

Instrument existing E1 evaluation calls rather than rewriting its training or
replay algorithms. Development seeds 1,2,3 only. Reserved 4,5,6 never run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium
import numpy as np
import pandas as pd
import torch

import scripts.run_e1_history_conditioning_development_v2 as e1
from src.representation.history import dynamics_transition_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir


ORIGINAL_EVAL = e1.eval_greedy
ORIGINAL_COLLECT = e1.collect_common
ORIGINAL_ADAPT = e1.adapt_branch

ACTIONS = ("noop", "left", "right", "down", "up")
AUDIT_EVALS: list[dict] = []
AUDIT_EPISODES: list[dict] = []
ACTIVE_CONTEXT: dict | None = None
NOOP_CACHE: dict[tuple, dict] = {}


def _sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _getarg(func, *args, **kwargs):
    """Bind positional/keyword parameters without changing original functions."""
    return inspect.signature(func).bind(*args, **kwargs).arguments


def policy_episode(agent, context, seed, cfg, architecture, rep, L, ep):
    """Exactly one original E1 greedy evaluation episode, with diagnostics."""
    env = e1.make_env(cfg, context)
    base = (70_000_000 + int(seed) * 10_000) if context == "A" else (
        170_000_000 + int(seed) * 20_000 + 10_000
    )
    counts = np.zeros(len(ACTIONS), dtype=int)
    distances, speeds = [], []
    ret = 0.0
    history = []
    try:
        state, _ = env.reset(seed=base + int(ep))
        while True:
            z = e1.frozen_context(history, rep, L) if architecture == "history_conditioned" else None
            action = e1.get_action(agent, state, z, architecture, epsilon=0.0)
            ns, reward, terminated, truncated, _ = env.step(action)
            history.append(dynamics_transition_feature(state, int(action), ns, 5))
            counts[int(action)] += 1
            ret += float(reward)
            distances.append(float(np.linalg.norm(ns[:2])))
            speeds.append(float(np.linalg.norm(ns[2:])))
            state = ns
            if terminated or truncated:
                return {
                    "return": float(ret),
                    "terminated": bool(terminated),
                    "episode_length": int(sum(counts)),
                    "mean_distance": float(np.mean(distances)),
                    "final_distance": float(np.linalg.norm(state[:2])),
                    "mean_speed": float(np.mean(speeds)),
                    "final_speed": float(np.linalg.norm(state[2:])),
                    **{f"n_{a}": int(k) for a, k in zip(ACTIONS, counts)},
                }
    finally:
        env.close()


def noop_episode(context, seed, cfg, ep):
    """Independent, no-learning No-op baseline with identical evaluation reset."""
    key = (context, int(seed), int(ep))
    if key in NOOP_CACHE:
        return NOOP_CACHE[key]
    base = (70_000_000 + int(seed) * 10_000) if context == "A" else (
        170_000_000 + int(seed) * 20_000 + 10_000
    )
    env = e1.make_env(cfg, context)
    ret = 0.0
    distances = []
    try:
        state, _ = env.reset(seed=base + int(ep))
        while True:
            state, r, terminated, truncated, _ = env.step(0)
            ret += float(r)
            distances.append(float(np.linalg.norm(state[:2])))
            if terminated or truncated:
                ans = {
                    "noop_return": float(ret),
                    "noop_terminated": bool(terminated),
                    "noop_final_distance": float(np.linalg.norm(state[:2])),
                    "noop_mean_distance": float(np.mean(distances)),
                }
                NOOP_CACHE[key] = ans
                return ans
    finally:
        env.close()


def audited_eval_greedy(agent, context, seed, cfg, architecture, rep, L):
    """Drop-in replacement: same episode count, agent calls, and RNG usage."""
    global ACTIVE_CONTEXT
    if ACTIVE_CONTEXT is None:
        raise RuntimeError("E1b evaluation executed outside an instrumented phase")
    tag = ACTIVE_CONTEXT
    idx = tag["index"]
    if idx >= len(tag["steps"]):
        raise AssertionError("Unexpected additional E1 evaluation")
    step = int(tag["steps"][idx])
    tag["index"] += 1
    n = int(cfg["isolation"]["eval_episodes"] if context == "A" else cfg["novelty"]["eval_episodes"])
    raw = []
    for ep in range(n):
        obs = policy_episode(agent, context, seed, cfg, architecture, rep, L, ep)
        noop = noop_episode(context, seed, cfg, ep)
        row = {
            "seed": int(seed), "architecture": architecture, "phase": tag["phase"],
            "schedule": tag["schedule"], "method": tag["method"],
            "eval_context": context, "phase_step": step, "episode": ep,
            **obs, **noop,
        }
        row["advantage_vs_noop"] = row["return"] - row["noop_return"]
        row["noop_fraction"] = row["n_noop"] / row["episode_length"]
        AUDIT_EPISODES.append(row)
        raw.append(row)
    mean = lambda key: float(np.mean([x[key] for x in raw]))
    aggregate = {
        "seed": int(seed), "architecture": architecture, "phase": tag["phase"],
        "schedule": tag["schedule"], "method": tag["method"],
        "eval_context": context, "phase_step": step,
        "eval_return_mean": mean("return"), "eval_return_sd": float(np.std([x["return"] for x in raw])),
        "eval_safety_termination_fraction": mean("terminated"),
        "noop_return_mean": mean("noop_return"),
        "advantage_vs_noop": mean("advantage_vs_noop"),
        "mean_distance": mean("mean_distance"),
        "final_distance": mean("final_distance"),
        "mean_speed": mean("mean_speed"),
        "final_speed": mean("final_speed"),
        "noop_fraction": float(sum(x["n_noop"] for x in raw) / sum(x["episode_length"] for x in raw)),
        "mean_episode_length": mean("episode_length"),
    }
    total = sum(int(x["episode_length"]) for x in raw)
    for action in ACTIONS:
        aggregate[f"action_{action}_fraction"] = sum(x[f"n_{action}"] for x in raw) / total
    AUDIT_EVALS.append(aggregate)
    # Keep E1's original return contract *exactly*.
    return {"return_mean": aggregate["eval_return_mean"],
            "return_sd": aggregate["eval_return_sd"],
            "safety_fraction": aggregate["eval_safety_termination_fraction"]}


def _audit_phase(phase, schedule, method, steps):
    return {"phase": phase, "schedule": schedule, "method": method,
            "steps": [int(s) for s in steps], "index": 0}


def audited_collect(*args, **kwargs):
    global ACTIVE_CONTEXT
    bound = _getarg(ORIGINAL_COLLECT, *args, **kwargs)
    prior = ACTIVE_CONTEXT
    ctx = bound["context"]
    steps = sorted(bound.get("checkpoint_evals", ())) if ctx == "A" else []
    ACTIVE_CONTEXT = _audit_phase("A1", "pre_switch", "common", steps)
    try:
        result = ORIGINAL_COLLECT(*args, **kwargs)
        if ACTIVE_CONTEXT["index"] != len(steps):
            raise AssertionError("A1 evaluation count diverged")
        if ctx == "B":
            # Diagnostics at the shared post-B checkpoint, without consuming the
            # live agent RNG or modifying the original optimizer/replay fork.
            agent = bound["agent"]
            probe_agent = copy.deepcopy(agent)
            ACTIVE_CONTEXT = _audit_phase("post_B", "pre_switch", "common", [int(result[0])])
            audited_eval_greedy(probe_agent, "A", bound["seed"], bound["cfg"],
                                bound["arch"], bound["rep"], bound["L"])
            ACTIVE_CONTEXT = _audit_phase("post_B", "pre_switch", "common", [int(result[0])])
            audited_eval_greedy(probe_agent, "B", bound["seed"], bound["cfg"],
                                bound["arch"], bound["rep"], bound["L"])
        return result
    finally:
        ACTIVE_CONTEXT = prior


def audited_adapt(*args, **kwargs):
    global ACTIVE_CONTEXT
    bound = _getarg(ORIGINAL_ADAPT, *args, **kwargs)
    schedule = bound["schedule"]
    cfg = bound["cfg"]
    part = cfg["isolation"] if schedule == "ABA" else cfg["novelty"]
    keys = part["A2_eval_steps"] if schedule == "ABA" else part["D_eval_steps"]
    prior = ACTIVE_CONTEXT
    ACTIVE_CONTEXT = _audit_phase("adaptation", schedule, bound["method"], [0, *keys])
    try:
        result = ORIGINAL_ADAPT(*args, **kwargs)
        if ACTIVE_CONTEXT["index"] != len(ACTIVE_CONTEXT["steps"]):
            raise AssertionError("Adaptation evaluation count diverged")
        return result
    finally:
        ACTIVE_CONTEXT = prior


def install_instrumentation():
    if e1.eval_greedy is not ORIGINAL_EVAL:
        raise RuntimeError("E1 is already instrumented; start a fresh process")
    e1.eval_greedy = audited_eval_greedy
    e1.collect_common = audited_collect
    e1.adapt_branch = audited_adapt


def compare_e1_with_current_audit(curves, evals):
    """Reject labelling mistakes; match every adaptation evaluation row."""
    columns = ["seed", "architecture", "schedule", "method", "phase_step"]
    curve = curves[columns + ["eval_return_mean", "eval_return_sd"]]
    audit = evals[evals.phase == "adaptation"]
    joined = curve.merge(audit[columns + ["eval_return_mean", "eval_return_sd"]],
                         on=columns, validate="one_to_one", suffixes=("_e1", "_audit"))
    if len(joined) != len(curve) or len(joined) != len(audit):
        raise AssertionError("Evaluation labels do not match E1 curves")
    if (np.abs(joined.eval_return_mean_e1 - joined.eval_return_mean_audit).max() > 1e-8 or
            np.abs(joined.eval_return_sd_e1 - joined.eval_return_sd_audit).max() > 1e-8):
        raise AssertionError("Evaluation instrumentation changed E1 return semantics")
    return joined


def add_control_quality(evals, refs):
    """No-op anchored progress: 0=no-op, 1=state-only stationary reference."""
    res = evals.copy()
    for seed in sorted(res.seed.unique()):
        rr = refs[refs.seed == seed].iloc[0]
        A = float(rr["A_reference_state_only"])
        D = float(rr["D_reference_state_only"])
        for ctx, ref in (("A", A), ("D", D)):
            mask = (res.seed == seed) & (res.eval_context == ctx)
            denom = ref - res.loc[mask, "noop_return_mean"]
            if (denom <= 0).any():
                raise AssertionError("No-op must be worse than stationary state-only reference")
            res.loc[mask, "control_progress_from_noop"] = res.loc[mask, "advantage_vs_noop"] / denom
    return res


def original_e1_drift(metrics):
    historical = ROOT / "results/synthetic_v2/e1_history_conditioning_development/development_metrics_seeds123.csv"
    if not historical.exists():
        return pd.DataFrame()
    old = pd.read_csv(historical)
    key = ["seed", "architecture", "schedule", "method"]
    columns = key + ["AUC_0_2000", "zero_shot_score", "final_score"]
    merged = metrics[columns].merge(old[columns], on=key, validate="one_to_one",
                                   suffixes=("_e1b", "_original_e1"))
    for m in ("AUC_0_2000", "zero_shot_score", "final_score"):
        merged[m + "_drift"] = merged[m + "_e1b"] - merged[m + "_original_e1"]
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic_v2/e1_history_conditioning_development.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs/base.yaml", ROOT / args.config)
    seeds = list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= {1, 2, 3}:
        raise RuntimeError("E1b may run development seeds 1/2/3 ONLY")
    if cfg["e1"]["development_seeds"] != [1, 2, 3] or cfg["e1"]["reserved_confirmation_seeds"] != [4, 5, 6]:
        raise RuntimeError("Original E1 seed discipline changed")
    if torch.cuda.is_available():
        raise RuntimeError("Canonical E1b must run with CUDA invisible")
    torch.set_num_threads(1)
    install_instrumentation()
    target = ensure_dir(ROOT / "results/synthetic_v2/e1b_policy_behavior_audit")
    parts = []
    for seed in seeds:
        print(f"\nE1b audit development seed {seed} ...", flush=True)
        before = len(AUDIT_EVALS)
        part = e1.run_seed(seed, cfg)
        ev = pd.DataFrame(AUDIT_EVALS[before:])
        compare_e1_with_current_audit(part["curves"], ev)
        parts.append(part)
        # Per-seed checkpoints for failure recovery; E1b does not rerun any other seed.
        pd.DataFrame(AUDIT_EVALS[before:]).to_csv(target / f"policy_evals_seed{seed}.csv", index=False)
        print(f"Completed seed {seed}; audited {len(ev)} evaluation checkpoints", flush=True)
    all_evals = pd.DataFrame(AUDIT_EVALS)
    all_eps = pd.DataFrame(AUDIT_EPISODES)
    curves = pd.concat([p["curves"] for p in parts], ignore_index=True)
    metrics = pd.concat([p["metrics"] for p in parts], ignore_index=True)
    refs = pd.concat([p["references"] for p in parts], ignore_index=True)
    all_evals = add_control_quality(all_evals, refs)
    all_eps.to_csv(target / "policy_evaluation_episodes.csv", index=False)
    all_evals.to_csv(target / "policy_evaluation_summaries.csv", index=False)
    metrics.to_csv(target / "reproduced_e1_metrics.csv", index=False)
    refs.to_csv(target / "reproduced_normalization_references.csv", index=False)
    drift = original_e1_drift(metrics)
    drift.to_csv(target / "historical_e1_reproduction_drift.csv", index=False)
    filtered = all_evals[(all_evals.phase == "adaptation") & (all_evals.phase_step.isin([0, 200, 500, 1000, 2000, 10000]))]
    filtered.to_csv(target / "key_adaptation_policy_checkpoints.csv", index=False)
    (target / "e1b_audit_metadata.json").write_text(json.dumps({
        "experiment": "E1b no-training-change policy behavior audit",
        "code_commit": _sha(), "seeds": seeds,
        "reserved_seeds_not_run": [4, 5, 6],
        "legacy_e1_script_unchanged": True,
        "e1b_intervention": "instrument existing greedy evaluations only; post-B probe uses deep-copied agent",
        "primary_diagnostics": ["action frequencies", "exact-seed noop baseline", "distance to target", "raw return", "control progress from noop"],
        "cross_runtime_drift_diagnostic_only": True,
        "evaluation_return_alignment_verified": True,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "gymnasium": gymnasium.__version__, "cuda": torch.cuda.is_available()},
    }, indent=2) + "\n", encoding="utf-8")
    print("\n=== Key E1b policy checkpoints ===")
    print(filtered[["seed", "architecture", "schedule", "method", "phase_step", "eval_return_mean", "noop_return_mean", "action_noop_fraction", "mean_distance", "control_progress_from_noop"]].to_string(index=False))
    print("\nE1b complete. Diagnostic only; no seeds 4,5,6; original E1 files unchanged.")


if __name__ == "__main__":
    main()
