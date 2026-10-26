"""E1c v2.15: diagnose history-conditioned learner collapse during A1->B.

Instrument the UNMODIFIED E1 common-phase training implementation only.
No replay strategy, train budget, model hyperparameter, evaluation seed, or
training action call is modified. Development seeds 1/2/3 only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
from pathlib import Path

# Respect the historical CPU execution path.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium
import numpy as np
import pandas as pd
import torch

import scripts.run_behavior_diverse_retrieval_pilot_v2 as behavior_module
import scripts.run_e1_history_conditioning_development_v2 as e1
import scripts.run_e1b_policy_behavior_audit_v2 as e1b
from scripts.run_behavior_diverse_retrieval_pilot_v2 import train_behavior_diverse_encoder
from src.representation.history import dynamics_transition_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir
from src.utils.seed import set_global_seed

ORIGINAL_UPDATE = e1.update_from_indexed_batch
ARCHES = ("state_only", "history_conditioned")
CONTEXTS = ("A", "B")
ACTION_NAMES = ("noop", "left", "right", "down", "up")
CURRENT: dict | None = None
UPDATES: list[dict] = []
SNAPSHOTS: list[dict] = []
Q_PROBES: list[dict] = []
FIXED_PROBES: dict | None = None
EVAL_EPISODES = 5  # unchanged E1 evaluation episode count


def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _rng_fingerprint(agent, replay):
    state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().numpy().copy(),
        agent.rng.bit_generator.state,
        replay.rng.bit_generator.state,
    )
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def _agent_parameter_fingerprint(agent):
    h = hashlib.sha256()
    for model in (agent.online, agent.target):
        for t in model.parameters():
            h.update(t.detach().cpu().numpy().tobytes())
    opt = agent.optimizer.state_dict()
    for group in opt["param_groups"]:
        h.update(repr(sorted(group.items(), key=lambda kv: kv[0])).encode())
    for idx in sorted(opt["state"]):
        h.update(str(idx).encode())
        for k, value in sorted(opt["state"][idx].items()):
            h.update(str(k).encode())
            if torch.is_tensor(value):
                h.update(value.detach().cpu().numpy().tobytes())
            else:
                h.update(repr(value).encode())
    h.update(str(agent.update_count).encode())
    return h.hexdigest()


def build_fixed_probes(rep, cfg, seed, L):
    """Independent researcher-only A/B probe trajectories; not training input."""
    probes = {}
    settings = cfg["e1c"]
    n_episodes = int(settings["fixed_probe_episodes_per_context"])
    n_steps = int(settings["fixed_probe_steps_per_episode"])
    seed_base = int(settings["fixed_probe_seed_base"])
    for cnum, context in enumerate(CONTEXTS):
        states, histories = [], []
        rng = np.random.default_rng(seed_base + 10_000 * seed + cnum)
        for ep in range(n_episodes):
            env = e1.make_env(cfg, context)
            try:
                s, _ = env.reset(seed=seed_base + 1000 + 10_000 * seed + 100 * cnum + ep)
                hist = []
                for _ in range(n_steps):
                    states.append(np.asarray(s, dtype=np.float32).copy())
                    histories.append(np.stack(hist[-L:]).astype(np.float32) if len(hist) >= L else None)
                    a = int(rng.integers(5))
                    ns, _, term, trunc, _ = env.step(a)
                    hist.append(dynamics_transition_feature(s, a, ns, 5))
                    s = ns
                    if term or trunc:
                        break
            finally:
                env.close()
        s_arr = np.stack(states).astype(np.float32)
        dim = int(rep["model"].encoder.proj[0].out_features)
        latent = np.zeros((len(s_arr), dim + 1), dtype=np.float32)
        valid = [i for i, h in enumerate(histories) if h is not None]
        if valid:
            enc = e1.encode_histories(rep["model"], np.stack([histories[i] for i in valid]), rep["history_std"])
            latent[np.array(valid), :-1] = enc
            latent[np.array(valid), -1] = 1.0
        probes[context] = {"state": s_arr, "latent": latent, "n": len(s_arr)}
    return probes


def fixed_probe_q(agent, arch, phase, phase_step, seed):
    """Raw Q diagnostics on identical external states, not Q accuracy claims."""
    if FIXED_PROBES is None:
        raise RuntimeError("Missing frozen probe bank")
    for context, data in FIXED_PROBES.items():
        inputs = data["state"] if arch == "state_only" else np.concatenate(
            (data["state"], data["latent"]), axis=1)
        with torch.no_grad():
            q = agent.online(torch.as_tensor(inputs, dtype=torch.float32, device=agent.device)).cpu().numpy()
        if q.shape != (data["n"], 5):
            raise AssertionError("Unexpected Q-probe output dimension")
        best_active = np.max(q[:, 1:], axis=1)
        margin = q[:, 0] - best_active
        actions = np.argmax(q, axis=1)
        z_valid = float(np.mean(data["latent"][:, -1]))
        row = {
            "seed": seed, "architecture": arch, "phase": phase,
            "phase_step": int(phase_step), "probe_context": context,
            "probe_n": len(actions), "probe_latent_valid_fraction": z_valid,
            "probe_noop_greedy_fraction": float(np.mean(actions == 0)),
            "probe_noop_margin_mean": float(np.mean(margin)),
            "probe_noop_margin_median": float(np.median(margin)),
            "probe_noop_margin_positive_fraction": float(np.mean(margin >= 0)),
            "probe_q_noop_mean": float(np.mean(q[:, 0])),
            "probe_q_best_active_mean": float(np.mean(best_active)),
            "probe_q_abs_mean": float(np.mean(np.abs(q))),
            "probe_q_max_mean": float(np.mean(np.max(q, axis=1))),
        }
        for i, name in enumerate(ACTION_NAMES):
            row[f"probe_greedy_{name}_fraction"] = float(np.mean(actions == i))
        Q_PROBES.append(row)


def _snapshot(agent, replay, rep, arch, phase, phase_step, seed, cfg, L):
    """Deep-copy probes and verify they cannot perturb training RNG/weights."""
    before_rng = _rng_fingerprint(agent, replay)
    before_params = _agent_parameter_fingerprint(agent)
    shadow = copy.deepcopy(agent)
    fixed_probe_q(shadow, arch, phase, phase_step, seed)
    for context in CONTEXTS:
        episodes = [e1b.policy_episode(shadow, context, seed, cfg, arch, rep, L, ep)
                    for ep in range(EVAL_EPISODES)]
        passive = [e1b.noop_episode(context, seed, cfg, ep)
                   for ep in range(EVAL_EPISODES)]
        n = sum(x["episode_length"] for x in episodes)
        count = sum(x["n_noop"] for x in episodes)
        raw = float(np.mean([x["return"] for x in episodes]))
        noop = float(np.mean([x["noop_return"] for x in passive]))
        SNAPSHOTS.append({
            "seed": int(seed), "architecture": arch, "phase": phase,
            "phase_step": int(phase_step), "eval_context": context,
            "raw_return": raw, "noop_return": noop, "advantage_vs_noop": raw - noop,
            "eval_noop_fraction": float(count / n),
            "mean_distance": float(np.mean([x["mean_distance"] for x in episodes])),
            "final_distance": float(np.mean([x["final_distance"] for x in episodes])),
            "mean_speed": float(np.mean([x["mean_speed"] for x in episodes])),
            "safety_fraction": float(np.mean([x["terminated"] for x in episodes])),
        })
    del shadow
    if _rng_fingerprint(agent, replay) != before_rng:
        raise AssertionError("E1c probe changed live training RNG state")
    if _agent_parameter_fingerprint(agent) != before_params:
        raise AssertionError("E1c probe changed live learner parameters")


def instrument_update(agent, replay, store, architecture, batch_size):
    """Call E1 unchanged; append detached observables after the update."""
    if CURRENT is None:
        raise AssertionError("E1c instrumented update outside common training")
    p = CURRENT
    event = int(p["update_index"])
    phase_global = int(p["first_update_global"] + event * p["freq"])
    phase_step = int(phase_global - p["start_global"] + 1)
    result = ORIGINAL_UPDATE(agent, replay, store, architecture, batch_size)
    selected = np.asarray(replay.last_sample_indices, dtype=np.int64)
    if len(selected) != batch_size:
        raise AssertionError("E1 sampling semantics unexpectedly changed")
    if architecture == "history_conditioned":
        before, after = store.take(selected)
        valid_before = float(np.mean(before[:, -1]))
        valid_after = float(np.mean(after[:, -1]))
        valid_recent = (
            float(np.mean([z[-1] for z in store.before[-min(len(store.before), 5000):]]))
            if event % 50 == 0 else np.nan
        )
    else:
        valid_before = valid_after = valid_recent = np.nan
    sampled_A = float(np.mean([replay.buffer[int(i)].context_id == "A" for i in selected]))
    UPDATES.append({
        "seed": p["seed"], "architecture": architecture,
        "phase": p["phase"], "phase_step": phase_step,
        "global_step_before_update": phase_global,
        "update_index_in_phase": event,
        "loss": float(result.loss), "q_mean": float(result.q_mean),
        "target_mean": float(result.target_mean),
        "td_abs_mean": float(result.td_abs_mean),
        "sampled_A_fraction_diagnostic": sampled_A,
        "batch_history_valid_before": valid_before,
        "batch_history_valid_after": valid_after,
        "recent_history_valid_fraction": valid_recent,
    })
    p["update_index"] += 1
    # In the original implementation B updates occur at phase steps 1,5,9,...
    # Hence snapshots at 201,1001,... are exact update timestamps, not 200,1000.
    if phase_step in p["update_snapshots"]:
        _snapshot(agent, replay, p["rep"], architecture, p["phase"], phase_step,
                  p["seed"], p["cfg"], p["L"])
    return result


def _begin_phase(phase, seed, arch, cfg, rep, replay, agent, L, start_global):
    freq = int(cfg["experiment"]["train_frequency"])
    learning_starts = int(cfg["experiment"]["learning_starts"])
    first = max(int(start_global), learning_starts)
    first += (-first) % freq
    return {
        "phase": phase, "seed": seed, "arch": arch, "cfg": cfg,
        "rep": rep, "L": L, "freq": freq, "start_global": start_global,
        "first_update_global": first, "update_index": 0,
        "update_snapshots": set(
            int(x) for x in cfg["e1c"]["q_snapshot_target_steps_A1" if phase == "A1"
                                      else "q_snapshot_target_steps_B"]
            if 0 < int(x) < (20000 if phase == "B" else int(cfg["isolation"]["A1_steps"]))
        ),
    }


def summarize_update_bins(df, bin_width=500):
    x = df.copy()
    if x.empty:
        return x
    x["phase_bin_start"] = ((x.phase_step - 1) // bin_width) * bin_width
    cols = ["loss", "q_mean", "target_mean", "td_abs_mean",
            "sampled_A_fraction_diagnostic", "batch_history_valid_before",
            "batch_history_valid_after", "recent_history_valid_fraction"]
    g = x.groupby(["seed", "architecture", "phase", "phase_bin_start"], dropna=False)
    return g[cols].mean().reset_index().merge(
        g.size().rename("n_updates").reset_index(),
        on=["seed", "architecture", "phase", "phase_bin_start"], validate="one_to_one")


def compare_e1b(snapshots, repo):
    path = repo / "results/synthetic_v2/e1b_policy_behavior_audit/policy_evaluation_summaries.csv"
    if not path.exists():
        raise FileNotFoundError("E1b reference results missing; run from GitHub main")
    old = pd.read_csv(path)
    audit = snapshots[((snapshots.phase == "A1") & (snapshots.phase_step == 20000) &
                       (snapshots.eval_context == "A")) |
                      ((snapshots.phase == "B") & (snapshots.phase_step == 20000))].copy()
    lookup = []
    for _, row in audit.iterrows():
        sub = old[(old.seed == row.seed) & (old.architecture == row.architecture)]
        if row.phase == "A1":
            sub = sub[(sub.phase == "A1") & (sub.phase_step == 20000) & (sub.eval_context == "A")]
        else:
            sub = sub[(sub.phase == "post_B") & (sub.eval_context == row.eval_context)]
        if len(sub) != 1:
            raise AssertionError("Missing/ambiguous E1b checkpoint reference")
        ref = sub.iloc[0]
        lookup.append({
            "seed": row.seed, "architecture": row.architecture,
            "phase": row.phase, "context": row.eval_context,
            "e1c_return": row.raw_return, "e1b_return": ref.eval_return_mean,
            "return_drift": float(row.raw_return - ref.eval_return_mean),
            "e1c_noop_fraction": row.eval_noop_fraction,
            "e1b_noop_fraction": ref.noop_fraction,
            "noop_fraction_drift": float(row.eval_noop_fraction - ref.noop_fraction),
        })
    return pd.DataFrame(lookup)


def run_seed(seed, cfg, out):
    global CURRENT, FIXED_PROBES
    set_global_seed(int(cfg["representation"]["seed"]))
    behavior_module.GLOBAL_CFG = cfg
    rep = train_behavior_diverse_encoder(cfg)
    if not (rep["relative_mse_improvement"] >= 0.20 and
            rep["offline_precision_at_10"] >= 0.90 and
            rep["offline_centroid_accuracy"] >= 0.90):
        raise RuntimeError("Existing frozen 05c representation sanity failed")
    L = int(cfg["representation"]["history_length"])
    FIXED_PROBES = build_fixed_probes(rep, cfg, seed, L)
    base_counts = (len(UPDATES), len(SNAPSHOTS), len(Q_PROBES))
    for arch in ARCHES:
        set_global_seed(seed)
        agent = (e1.make_agent(cfg, seed) if arch == "state_only"
                 else e1.HistoryConditionedDDQN(cfg, seed, int(rep["model"].encoder.proj[0].out_features)))
        replay = e1.IndexedRecentReplayBuffer(
            capacity=int(cfg["isolation"]["replay_capacity"]),
            recent_window=int(cfg["isolation"]["recent_window"]), seed=seed)
        store = e1.LatentStore.empty()
        CURRENT = _begin_phase("A1", seed, arch, cfg, rep, replay, agent, L, 0)
        _snapshot(agent, replay, rep, arch, "A1", 0, seed, cfg, L)
        step, A_ep, A_eval = e1.collect_common(
            agent, replay, store, arch, rep, "A", cfg, seed, phase_index=0,
            start_global=0, steps=int(cfg["isolation"]["A1_steps"]), L=L,
            checkpoint_evals=set(int(x) for x in cfg["isolation"]["A1_reference_steps"]))
        assert step == 20000 and len(A_eval) == 3
        _snapshot(agent, replay, rep, arch, "A1", int(cfg["isolation"]["A1_steps"]), seed, cfg, L)
        CURRENT = _begin_phase("B", seed, arch, cfg, rep, replay, agent, L, step)
        _snapshot(agent, replay, rep, arch, "B", 0, seed, cfg, L)
        step, B_ep, _ = e1.collect_common(
            agent, replay, store, arch, rep, "B", cfg, seed, phase_index=1,
            start_global=step, steps=20000, L=L)
        assert step == 40000 and len(replay) == 40000
        _snapshot(agent, replay, rep, arch, "B", 20000, seed, cfg, L)
        CURRENT = None
    u, s, q = base_counts
    updates = pd.DataFrame(UPDATES[u:])
    snapshots = pd.DataFrame(SNAPSHOTS[s:])
    probes = pd.DataFrame(Q_PROBES[q:])
    compared = compare_e1b(snapshots, ROOT)
    tolerance = float(cfg["e1c"]["reproduction_atol"])
    maxdrift = max(compared.return_drift.abs().max(), compared.noop_fraction_drift.abs().max())
    if maxdrift > tolerance:
        raise RuntimeError(f"E1c common-phase reproduction drift {maxdrift:.6g} > {tolerance:.2g}")
    updates.to_csv(out / f"updates_seed{seed}.csv", index=False)
    summarize_update_bins(updates).to_csv(out / f"update_bins_seed{seed}.csv", index=False)
    snapshots.to_csv(out / f"policy_snapshots_seed{seed}.csv", index=False)
    probes.to_csv(out / f"fixed_q_probes_seed{seed}.csv", index=False)
    compared.to_csv(out / f"reproduction_checks_seed{seed}.csv", index=False)
    print(f"E1c seed {seed} complete: {len(updates)} updates, {len(snapshots)} rollouts, "
          f"{len(probes)} fixed Q probes. Max E1b drift {maxdrift:.2g}", flush=True)


def aggregate(out, requested_seeds):
    files = {
        "updates": "updates_seed{seed}.csv",
        "update_bins": "update_bins_seed{seed}.csv",
        "policy_snapshots": "policy_snapshots_seed{seed}.csv",
        "fixed_q_probes": "fixed_q_probes_seed{seed}.csv",
        "reproduction_checks": "reproduction_checks_seed{seed}.csv",
    }
    for name, pattern in files.items():
        ps = [out / pattern.format(seed=seed) for seed in requested_seeds]
        if not all(p.exists() for p in ps):
            raise FileNotFoundError(f"Cannot aggregate {name}; missing some requested seed files")
        pd.concat([pd.read_csv(p) for p in ps], ignore_index=True).to_csv(
            out / f"{name}_all_development.csv", index=False)
    snaps = pd.read_csv(out / "policy_snapshots_all_development.csv")
    q = pd.read_csv(out / "fixed_q_probes_all_development.csv")
    critical = snaps[(snaps.phase == "B") & (snaps.phase_step.isin([0, 1001, 5001, 10001, 20000]))]
    critical.to_csv(out / "B_collapse_key_checkpoints.csv", index=False)
    manifest = {
        "experiment": "E1c learner collapse diagnosis, descriptive development only",
        "source_commit": git_sha(), "development_seeds": requested_seeds,
        "reserved_confirmation_seeds_not_run": [4, 5, 6],
        "training_implementation": "unchanged E1 collect_common/update_from_indexed_batch",
        "audit_method": "post-update stats; deepcopy policy probe and fixed held-out Q probes",
        "no_algorithm_or_training_hyperparameter_changed": True,
        "E1b_reference_trajectory_reproduced": True,
        "checkpoint_phase_step": "1-indexed when captured post-update; 0/20000 captured at boundaries",
        "q_probes": "researcher-only independent random-action trajectories from A and B; no labels supplied to learner",
        "q_probe_qvalues": "raw uncalibrated DDQN estimates, not true action values",
        "n_policy_snapshots": len(snaps), "n_fixed_q_probe_rows": len(q),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "gymnasium": gymnasium.__version__,
                    "threads": torch.get_num_threads(), "cuda": torch.cuda.is_available()},
    }
    (out / "e1c_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return critical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_v2/e1c_learner_collapse_diagnosis.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds]
    if (not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= {1, 2, 3}):
        raise RuntimeError("E1c restricted to development seeds 1/2/3")
    cfg = load_config(ROOT / "configs/base.yaml", ROOT / args.config)
    if cfg["e1"]["development_seeds"] != [1, 2, 3] or cfg["e1"]["reserved_confirmation_seeds"] != [4, 5, 6]:
        raise RuntimeError("E1 seed guard changed; abort")
    if cfg["e1c"]["development_seeds"] != [1, 2, 3] or cfg["e1c"]["reserved_confirmation_seeds"] != [4, 5, 6]:
        raise RuntimeError("E1c seed guard changed; abort")
    if torch.cuda.is_available():
        raise RuntimeError("Use CPU runtime for E1c")
    if int(cfg["isolation"]["eval_episodes"]) != EVAL_EPISODES or int(cfg["novelty"]["eval_episodes"]) != EVAL_EPISODES:
        raise RuntimeError("Frozen E1 evaluation episodes changed")
    for phase, arr in (("A1", cfg["e1c"]["q_snapshot_target_steps_A1"]),
                       ("B", cfg["e1c"]["q_snapshot_target_steps_B"])):
        if list(arr) != sorted(set(arr)) or arr[0] != 0 or arr[-1] != 20000:
            raise RuntimeError(f"Malformed/future-altered fixed E1c {phase} checkpoint schedule")
    torch.set_num_threads(1)
    out = ensure_dir(ROOT / "results/synthetic_v2/e1c_learner_collapse_diagnosis")
    if not args.aggregate_only:
        if e1.update_from_indexed_batch is not ORIGINAL_UPDATE:
            raise RuntimeError("E1 update function already instrumented: start fresh process")
        e1.update_from_indexed_batch = instrument_update
        try:
            for seed in seeds:
                print(f"\n=== E1c seed {seed} (A1->B common training only) ===", flush=True)
                run_seed(seed, cfg, out)
        finally:
            e1.update_from_indexed_batch = ORIGINAL_UPDATE
    if seeds == [1, 2, 3]:
        crit = aggregate(out, seeds)
        print("\nE1c completed. B phase key checkpoints:\n",
              crit[["seed", "architecture", "phase_step", "eval_context",
                    "eval_noop_fraction", "advantage_vs_noop", "mean_distance"]].to_string(index=False))
    else:
        print("Completed selected development seeds; use --aggregate-only --seeds 1 2 3 when all are present.")


if __name__ == "__main__":
    main()
