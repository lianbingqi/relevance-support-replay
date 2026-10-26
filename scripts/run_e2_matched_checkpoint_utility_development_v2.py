"""E2 v2.17: matched-checkpoint replay-utility mechanism study.

Development seeds 1/2/3 only. No replay-gate method is proposed or fitted.
The canonical state-only DDQN follows A1(20k)->B(20k)->A2/D(2k).
At post-switch Recent-only checkpoints, use OFFLINE 16-gradient-update forks,
no new environment experience, identical network/optimizer/target initial states.

Contrasts compare current-PHASE-only replay to an 80% historical + 20%
current-phase batch. They are NOT the legacy Recent rolling-window protocol.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import gymnasium

import scripts.run_behavior_diverse_retrieval_pilot_v2 as behavior_module
import scripts.run_e1_history_conditioning_development_v2 as e1
import scripts.run_e1b_policy_behavior_audit_v2 as e1b
from scripts.run_behavior_diverse_retrieval_pilot_v2 import train_behavior_diverse_encoder
from scripts.run_learned_replay_recurrence_pilot_v2 import history_bank_windows, uniform_cap_bank
from scripts.run_learned_relevance_retrieval_pilot_v2 import encode_histories
from scripts.run_support_aware_heldout_confirmation_v2 import transition_gate_scores
from src.replay.recent import _to_batch
from src.representation.history import dynamics_transition_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir
from src.utils.seed import set_global_seed

CANDIDATES = ("retrieved_top50", "uniform50", "least_relevant50")
METHODS = ("current_only",) + CANDIDATES


def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def stable_seed(*parts):
    """Avoid Python's salted hash and prevent schedule/arm RNG collisions."""
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**32 - 1)


def batch_at(replay, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or len(set(indices.tolist())) != len(indices):
        raise AssertionError("Within-batch indices must be nonempty and unique")
    if indices.min() < 0 or indices.max() >= len(replay):
        raise AssertionError("Future/unavailable replay entry in E2 batch")
    return _to_batch([replay.buffer[int(i)] for i in indices])


def choose_rows(query, bank_z, k, seed):
    """No labels used; top, uniform, bottom candidates in frozen embedding bank."""
    if len(bank_z) < k * 2:
        raise AssertionError("Too few bank windows")
    q = np.asarray(query, dtype=np.float32)
    norm = np.linalg.norm(bank_z, axis=1)
    sim = (bank_z / np.maximum(norm[:, None], 1e-8)) @ (q / max(np.linalg.norm(q), 1e-8))
    order = np.argsort(-sim, kind="stable")
    unif = np.random.default_rng(seed).choice(len(bank_z), size=k, replace=False)
    return {
        "retrieved_top50": order[:k].astype(np.int64),
        "uniform50": np.sort(unif).astype(np.int64),
        "least_relevant50": order[-k:].astype(np.int64),
    }, {"top_similarity": float(sim[order[0]]), "bottom_similarity": float(sim[order[-1]])}


def candidate_pool(bank, rows, old_cutoff):
    ids = np.unique(bank["replay_indices"][np.asarray(rows, dtype=np.int64)].reshape(-1))
    ids = ids[ids < old_cutoff]
    if len(ids) < 64:
        raise AssertionError("Candidate historical pool < 64; cannot test frozen 80% replay")
    return ids


def paired_indices(current_pool, hist_pool, updates, batch_size, hist_fraction, seed):
    """Share the 20% current subset between current-only and mixed updates."""
    rng_recent = np.random.default_rng(seed)
    rng_historical = np.random.default_rng((int(seed) + 0x9E3779B9) % (2**32 - 1))
    nr_hist = int(round(batch_size * hist_fraction))
    nr_current = batch_size - nr_hist
    assert nr_hist + nr_current == batch_size and nr_current > 0
    if len(current_pool) < batch_size or (hist_pool is not None and len(hist_pool) < nr_hist):
        raise AssertionError("Insufficient unique sampling candidates")
    current, mixed = [], []
    for _ in range(updates):
        r = rng_recent.choice(current_pool, size=batch_size, replace=False).astype(np.int64)
        current.append(r)
        if hist_pool is not None:
            h = rng_historical.choice(hist_pool, size=nr_hist, replace=False).astype(np.int64)
            m = np.concatenate([r[:nr_current], h])
            if len(set(m.tolist())) != batch_size:
                raise AssertionError("Overlapping old/current pools")
            mixed.append(rng_historical.permutation(m))
    return current, mixed


def td_loss_value(agent, batch):
    """Held-out Double-DQN TD loss, NOT an on-policy utility label."""
    with torch.no_grad():
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=agent.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=agent.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=agent.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=agent.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=agent.device)
        q = agent.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        a_next = agent.online(next_states).argmax(1)
        q_next = agent.target(next_states).gather(1, a_next.unsqueeze(1)).squeeze(1)
        targets = rewards + agent.gamma * (1 - dones) * q_next
        return float(agent.loss_fn(q, targets).item())


def gradient_vector(agent, batch):
    """Frozen-checkpoint TD-loss gradient; never calls optimizer.step."""
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=agent.device)
    next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=agent.device)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=agent.device)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=agent.device)
    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=agent.device)
    q = agent.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        a_next = agent.online(next_states).argmax(1)
        q_next = agent.target(next_states).gather(1, a_next.unsqueeze(1)).squeeze(1)
        y = rewards + agent.gamma * (1 - dones) * q_next
    loss = agent.loss_fn(q, y)
    grads = torch.autograd.grad(loss, tuple(agent.online.parameters()), allow_unused=False)
    return torch.cat([g.detach().reshape(-1) for g in grads]).cpu().numpy()


def cosine(x, y):
    return float(np.dot(x, y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-12))


def evaluate(agent, context, seed, cfg):
    """Canonical 5 greedy eval episodes with actual action/control diagnostics."""
    shadow = copy.deepcopy(agent)
    L = int(cfg["representation"]["history_length"])
    values = [e1b.policy_episode(shadow, context, seed, cfg, "state_only", None, L, ep)
              for ep in range(5)]
    passive = [e1b.noop_episode(context, seed, cfg, ep) for ep in range(5)]
    return {
        "raw_return": float(np.mean([x["return"] for x in values])),
        "advantage_vs_noop": float(np.mean([x["return"]-y["noop_return"] for x,y in zip(values,passive)])),
        "noop_fraction": float(sum(x["n_noop"] for x in values) / sum(x["episode_length"] for x in values)),
        "mean_distance": float(np.mean([x["mean_distance"] for x in values])),
        "safety_fraction": float(np.mean([x["terminated"] for x in values])),
    }


def compatibility(rep, bank_z, rows, evidence):
    """Mean retrieved-dynamics one-step MSE on observed pre-checkpoint transitions.

    Candidate set selected by last query; each evidence score uses its OWN
    preceding 8 transitions and its observed transition. No future observation.
    """
    if not evidence:
        raise AssertionError("No valid post-switch prior-history evidence")
    subset = np.asarray(bank_z[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
    subset_norm = subset / np.maximum(np.linalg.norm(subset, axis=1, keepdims=True), 1e-8)
    vals = []
    for prior, state, action, next_state in evidence:
        obj = transition_gate_scores(
            prior_history=prior, state=state, action=int(action), next_state=next_state,
            model=rep["model"], history_std=rep["history_std"],
            current_std=rep["current_std"], target_std=rep["target_std"],
            bank_z=subset, bank_z_norm=subset_norm, top_k=len(subset),
        )
        vals.append(float(obj["retrieved_mean_mse"]))
    return float(np.mean(vals)), float(np.median(vals))


def run_update_fork(base_agent, replay, batch_indices, test_batch):
    learner = copy.deepcopy(base_agent)
    before = td_loss_value(learner, test_batch)
    losses = []
    for idx in batch_indices:
        stats = learner.update(batch_at(replay, idx))
        losses.append(stats.loss)
    after = td_loss_value(learner, test_batch)
    return learner, before, after, float(np.mean(losses))


def matched_checkpoint(seed, schedule, phase_step, agent, replay, rep, bank, bank_z,
                       query_history, evidence, current_indices, cfg):
    """All model updates happen only on forked copies; no new env data."""
    e2 = cfg["e2"]
    ctx = "A" if schedule == "ABA" else "D"
    hist_frac = float(e2["historical_fraction"])
    bs = int(cfg["agent"]["batch_size"])
    if len(current_indices) != phase_step or min(current_indices) < 40000:
        raise AssertionError("Current-only pool must contain EXACT post-switch transitions")
    if len(query_history) != int(cfg["representation"]["history_length"]):
        raise AssertionError("No valid recent query history")
    z = encode_histories(rep["model"], np.stack(query_history)[None,:,:].astype(np.float32), rep["history_std"])[0]
    rows, sim_stats = choose_rows(z, bank_z, int(e2["candidate_bank_windows"]),
                                  stable_seed(seed,schedule,phase_step,"uniform_bank"))
    pools = {key:candidate_pool(bank, ids, 40000) for key,ids in rows.items()}
    rng = np.random.default_rng(stable_seed(seed,schedule,phase_step,"holdout"))
    held = np.sort(rng.choice(current_indices, size=int(e2["recent_holdout_n"]),replace=False))
    trainpool = np.setdiff1d(np.asarray(current_indices,dtype=np.int64),held)
    if len(trainpool)<bs or np.any(trainpool < 40000):
        raise AssertionError("Current-phase training pool malformed")
    test_batch=batch_at(replay,held)
    prior_parameters = [(x.detach().cpu().clone()) for x in agent.online.parameters()]
    prior_count = agent.update_count
    prior_agent_rng = copy.deepcopy(agent.rng.bit_generator.state)
    before_eval = evaluate(agent, ctx, seed, cfg)
    result_rows=[]; info_rows=[]
    for method in CANDIDATES:
        mse, med = compatibility(rep,bank_z,rows[method],evidence)
        allids = pools[method]
        context_counts = [replay.buffer[int(i)].context_id for i in allids]
        grad_seed = stable_seed(seed,schedule,phase_step,method,"gradient")
        grng=np.random.default_rng(grad_seed)
        grad_r=batch_at(replay,grng.choice(trainpool,size=int(e2["q_grad_n"]),replace=False))
        grad_h=batch_at(replay,grng.choice(allids,size=int(e2["q_grad_n"]),replace=False))
        cos=cosine(gradient_vector(agent,grad_r),gradient_vector(agent,grad_h))
        info_rows.append({"seed":seed,"schedule":schedule,"checkpoint_step":phase_step,
                          "candidate":method,"compatibility_mse":mse,
                          "compatibility_mse_median":med,"gradient_cosine":cos,
                          "historical_pool_n":len(allids),
                          "historical_A_fraction_diagnostic":float(np.mean(np.asarray(context_counts)=="A")),
                          "best_bank_similarity":sim_stats["top_similarity"],
                          "worst_bank_similarity":sim_stats["bottom_similarity"],
                          "compatibility_evidence_n":len(evidence),
                          "holdout_n":len(held)})
    for repeat in range(int(e2["batch_replicates"])):
        # All methods use the identical Recent-only update batches for a given
        # seed/schedule/checkpoint/replicate. Repeated candidate branches share
        # their Recent 20% with those full batches.
        for method in METHODS:
            sid = stable_seed(seed,schedule,phase_step,repeat,"paired_batches")
            current_batches, mixed_batches=paired_indices(trainpool,
                None if method=="current_only" else pools[method],
                int(e2["branch_updates"]),bs,hist_frac,sid)
            use_batches = current_batches if method=="current_only" else mixed_batches
            updated,before_td,after_td,train_loss = run_update_fork(agent,replay,use_batches,test_batch)
            after_eval=evaluate(updated,ctx,seed,cfg)
            result_rows.append({"seed":seed,"schedule":schedule,"checkpoint_step":phase_step,
                                "replicate":repeat,"method":method,
                                "updates":int(e2["branch_updates"]),"batch_size":bs,
                                "historical_fraction":0.0 if method=="current_only" else hist_frac,
                                "before_raw_return":before_eval["raw_return"],
                                "after_raw_return":after_eval["raw_return"],
                                "gain_from_checkpoint":after_eval["raw_return"]-before_eval["raw_return"],
                                "after_advantage_vs_noop":after_eval["advantage_vs_noop"],
                                "after_noop_fraction":after_eval["noop_fraction"],
                                "after_mean_distance":after_eval["mean_distance"],
                                "after_safety_fraction":after_eval["safety_fraction"],
                                "td_heldout_before":before_td,"td_heldout_after":after_td,
                                "td_heldout_change":after_td-before_td,"train_loss":train_loss})
    if agent.update_count != prior_count or agent.rng.bit_generator.state != prior_agent_rng:
        raise AssertionError("Matched fork modified live learner update count/action RNG")
    if not all(torch.equal(a, b) for a, b in zip(prior_parameters, agent.online.parameters())):
        raise AssertionError("Matched fork changed live online Q-network")
    return result_rows, info_rows


def reproduction_check(seed, agent, cfg, phase, context):
    """Guard canonical state-only A1/B trajectories before mechanism work."""
    historical = pd.read_csv(ROOT/"results/synthetic_v2/e1b_policy_behavior_audit/policy_evaluation_summaries.csv")
    mask=(historical.seed==seed)&(historical.architecture=="state_only")&(historical.eval_context==context)
    mask &= (historical.phase=="A1")&(historical.phase_step==20000) if phase=="A1" else (historical.phase=="post_B")
    reference=historical[mask]
    if len(reference)!=1:
        raise AssertionError("E1b control reference missing or non-unique")
    actual=evaluate(agent,context,seed,cfg)
    r=reference.iloc[0]
    dr=actual["raw_return"]-float(r.eval_return_mean)
    df=actual["noop_fraction"]-float(r.noop_fraction)
    if max(abs(dr),abs(df))>float(cfg["e2"]["reference_reproduction_tolerance"]):
        raise AssertionError(f"Canonical E1b trajectory drift seed={seed} {phase}/{context}: {dr}, {df}")
    return {"seed":seed,"phase":phase,"context":context,
            "return_drift":dr,"noop_fraction_drift":df}


def run_seed(seed,cfg,out):
    """Train common A1/B once, then Recent-only A2/D checkpoint trajectory."""
    set_global_seed(int(cfg["representation"]["seed"]))
    behavior_module.GLOBAL_CFG=cfg
    rep=train_behavior_diverse_encoder(cfg)
    if not (rep["relative_mse_improvement"]>=.20 and
            rep["offline_precision_at_10"]>=.90 and
            rep["offline_centroid_accuracy"]>=.90):
        raise RuntimeError("Frozen 05c representation sanity failed")
    L=int(cfg["representation"]["history_length"])
    set_global_seed(seed)
    agent=e1.make_agent(cfg,seed)
    replay=e1.IndexedRecentReplayBuffer(capacity=int(cfg["isolation"]["replay_capacity"]),
                                          recent_window=int(cfg["isolation"]["recent_window"]),seed=seed)
    store=e1.LatentStore.empty()
    step,A_ep,A_eval=e1.collect_common(agent,replay,store,"state_only",rep,"A",cfg,seed,
                phase_index=0,start_global=0,steps=20000,L=L,
                checkpoint_evals=set(cfg["isolation"]["A1_reference_steps"]))
    assert step==20000 and len(A_eval)==3
    checks=[reproduction_check(seed,agent,cfg,"A1","A")]
    step,B_ep,_=e1.collect_common(agent,replay,store,"state_only",rep,"B",cfg,seed,
                phase_index=1,start_global=step,steps=20000,L=L)
    assert step==40000 and len(replay)==40000
    checks.extend([reproduction_check(seed,agent,cfg,"post_B",ctx) for ctx in ("A","B")])
    bankA=history_bank_windows(A_ep,L,int(cfg["isolation"]["history_bank_stride"]))
    bankB=history_bank_windows(B_ep,L,int(cfg["isolation"]["history_bank_stride"]))
    bank={key:np.concatenate([bankA[key],bankB[key]]) for key in bankA}
    bank=uniform_cap_bank(bank,int(cfg["isolation"]["max_history_bank_windows"]),seed)
    bank_z=encode_histories(rep["model"],bank["history"],rep["history_std"])
    outrows=[]; diagnostics=[]; controls=[]
    for schedule in cfg["e2"]["schedules"]:
        context="A" if schedule=="ABA" else "D"
        current_agent=copy.deepcopy(agent)
        current_replay=copy.deepcopy(replay)
        env=e1.make_env(cfg,context)
        reset_base=(90_000_000 if context=="A" else 175_000_000)+seed*1_000_000
        state,info=env.reset(seed=reset_base)
        hist=[]; episode_id=0; global_step=40000
        current_ids=[]; last_query=None
        evid=deque(maxlen=int(cfg["e2"]["compatibility_evidence_transitions"]))
        eval_steps=set(int(x) for x in cfg["e2"]["checkpoint_steps"])
        try:
            for t in range(1,max(eval_steps)+1):
                a=current_agent.act(state,e1.epsilon_at(global_step,cfg))
                ns,reward,terminated,truncated,next_info=env.step(a)
                if len(current_replay)==current_replay.capacity:
                    raise AssertionError("Replay wrapped")
                if len(hist)>=L:
                    evid.append((np.stack(hist[-L:]).copy(),
                                 np.asarray(state,dtype=np.float32).copy(),
                                 int(a),np.asarray(ns,dtype=np.float32).copy()))
                index=len(current_replay)
                current_replay.add(state,a,reward,ns,terminated,context_id=context,
                                   context_vector=info["context_vector"])
                current_ids.append(index)
                hist.append(dynamics_transition_feature(state,int(a),ns,5))
                if len(hist)>=L:
                    last_query=np.stack(hist[-L:]).copy()
                if global_step % int(cfg["experiment"]["train_frequency"])==0:
                    e1.update_from_indexed_batch(current_agent,current_replay,store,"state_only",
                                                  int(cfg["agent"]["batch_size"]))
                global_step+=1
                state,info=ns,next_info
                if t in eval_steps:
                    if last_query is None or len(evid)<int(cfg["e2"]["compatibility_evidence_transitions"]):
                        raise AssertionError("Insufficient current-context evidence at checkpoint")
                    obs,meta=matched_checkpoint(seed,schedule,t,current_agent,current_replay,rep,bank,bank_z,
                                                     last_query,list(evid),current_ids,cfg)
                    outrows.extend(obs); diagnostics.extend(meta)
                    print(f"E2 seed={seed}, {schedule}, t={t}: {len(obs)} matched forks",flush=True)
                if terminated or truncated:
                    hist=[]
                    if t < max(eval_steps):
                        episode_id+=1
                        state,info=env.reset(seed=reset_base+episode_id)
        finally:
            env.close()
        controls.append({"seed":seed,"schedule":schedule,"current_trajectory_steps":len(current_ids),
                         "bank_windows":len(bank_z),"A_bank_fraction_diagnostic":float(np.mean(bank["label"]=="A")),
                         "common_B_return":float(evaluate(agent,"B",seed,cfg)["raw_return"])})
    data={"fork_outcomes":pd.DataFrame(outrows),"candidate_diagnostics":pd.DataFrame(diagnostics),
          "reproduction_checks":pd.DataFrame(checks),"trajectory_controls":pd.DataFrame(controls)}
    for label,df in data.items():
        tmp=out/f"{label}_seed{seed}.csv.partial"
        df.to_csv(tmp,index=False); tmp.replace(out/f"{label}_seed{seed}.csv")
    print(f"E2 seed {seed} saved: {len(outrows)} fork outcomes, {len(diagnostics)} candidate comparisons",flush=True)


def aggregate(out,seeds,cfg):
    names=("fork_outcomes","candidate_diagnostics","reproduction_checks","trajectory_controls")
    for name in names:
        paths=[out/f"{name}_seed{s}.csv" for s in seeds]
        if not all(p.exists() for p in paths):
            raise FileNotFoundError(f"Incomplete E2: missing {name} per-seed file")
        pd.concat([pd.read_csv(p) for p in paths],ignore_index=True).to_csv(out/f"{name}_all_development.csv",index=False)
    o=pd.read_csv(out/"fork_outcomes_all_development.csv")
    expected_cells = len(seeds) * 2 * 4 * 3 * len(METHODS)
    if len(o) != expected_cells or not np.all(np.isfinite(o["after_raw_return"])):
        raise AssertionError(f"Incomplete/nonfinite E2 outcomes: {len(o)} / {expected_cells}")
    if not (o.groupby(["seed","schedule","checkpoint_step"])["before_raw_return"].nunique()==1).all():
        raise AssertionError("Within-checkpoint before-policy identity failed")
    parent=o[o.method=="current_only"].drop(columns=["method"]).copy()
    deltas=o[o.method!="current_only"].merge(
        parent[["seed","schedule","checkpoint_step","replicate","after_raw_return","after_advantage_vs_noop","td_heldout_after"]],
        on=["seed","schedule","checkpoint_step","replicate"],suffixes=("_historical","_current"),validate="many_to_one")
    deltas["utility_vs_current_raw"]=(deltas.after_raw_return_historical-deltas.after_raw_return_current)
    deltas["utility_vs_current_advantage"]=(deltas.after_advantage_vs_noop_historical-deltas.after_advantage_vs_noop_current)
    deltas["heldout_td_difference"]=(deltas.td_heldout_after_historical-deltas.td_heldout_after_current)
    deltas=deltas.merge(pd.read_csv(out/"candidate_diagnostics_all_development.csv"),
                        left_on=["seed","schedule","checkpoint_step","method"],
                        right_on=["seed","schedule","checkpoint_step","candidate"],validate="many_to_one")
    deltas.to_csv(out/"matched_utility_vs_current_all_development.csv",index=False)
    cells=deltas.groupby(["seed","schedule","checkpoint_step","method"],as_index=False).agg(
        utility_mean=("utility_vs_current_raw","mean"),utility_min=("utility_vs_current_raw","min"),
        utility_max=("utility_vs_current_raw","max"),mean_compatibility_mse=("compatibility_mse","mean"),
        gradient_cosine=("gradient_cosine","mean"),
        n_replicates=("replicate","nunique"))
    cells.to_csv(out/"checkpoint_utility_summary.csv",index=False)
    manifest={"experiment":"E2 v2.17 matched-checkpoint historical replay utility development",
              "source_commit":git_sha(),"development_seeds":seeds,"reserved_seeds_not_run":[4,5,6],
              "not_replay_gate":True,"causal_estimand":"controlled effect of substituting 80% old samples for current-phase-only samples in 16 offline DDQN updates",
              "not_legacy_recent":"Current-only excludes all pre-switch B transitions, unlike rolling Recent from RQ2/RQ3",
              "design":cfg["e2"],"runtime":{"python":platform.python_version(),"numpy":np.__version__,
                         "torch":torch.__version__,"gymnasium":gymnasium.__version__,
                         "cuda":torch.cuda.is_available(),"threads":torch.get_num_threads()}}
    (out/"e2_metadata.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return cells


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/synthetic_v2/e2_matched_checkpoint_utility_development.yaml")
    p.add_argument("--seeds",type=int,nargs="+",default=[1,2,3])
    p.add_argument("--aggregate-only",action="store_true")
    args=p.parse_args()
    seeds=list(args.seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(x not in [1,2,3] for x in seeds):
        raise RuntimeError("E2 restricted to development seeds 1/2/3")
    cfg=load_config(ROOT/"configs/base.yaml",ROOT/args.config)
    e2=cfg["e2"]
    if e2["development_seeds"]!=[1,2,3] or e2["reserved_confirmation_seeds"]!=[4,5,6]:
        raise RuntimeError("E2 development/confirmation seed guard changed")
    if list(e2["schedules"])!=["ABA","ABD"] or list(e2["checkpoint_steps"])!=[200,500,1000,2000]:
        raise RuntimeError("E2 locked schedule/checkpoints changed")
    if (int(e2["branch_updates"])!=16 or int(e2["batch_replicates"])!=3 or
        float(e2["historical_fraction"])!=.8 or int(e2["candidate_bank_windows"])!=50):
        raise RuntimeError("E2 frozen sampling design changed")
    if int(cfg["isolation"]["A1_steps"])!=20000 or int(cfg["isolation"]["B_steps"])!=20000:
        raise RuntimeError("Common training length changed")
    if torch.cuda.is_available():
        raise RuntimeError("Use canonical CPU runtime for E2")
    torch.set_num_threads(1)
    out=ensure_dir(ROOT/"results/synthetic_v2/e2_matched_checkpoint_utility_development")
    if not args.aggregate_only:
        for seed in seeds:
            print(f"=== E2 seed {seed} ===",flush=True)
            run_seed(seed,cfg,out)
    if seeds==[1,2,3]:
        cells=aggregate(out,seeds,cfg)
        print("\nE2 checkpoint-level utility comparison:\n",cells.to_string(index=False),flush=True)
    else:
        print("E2 per-seed output saved. Aggregate after seeds 1 2 3 finish.",flush=True)


if __name__=="__main__":
    main()
