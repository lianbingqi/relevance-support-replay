"""E1d v2.16: on-policy Q diagnosis and matched-shape latent ablation.

Development seeds 1/2/3 only. A1->B common phases; no historical replay branches.
Primary scientific caution: policy-conditioned probes are descriptive; arms follow
separate on-policy trajectories, so this is not an isolated gradient-level cause.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

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
import scripts.run_e1c_learner_collapse_diagnosis_v2 as e1c
from scripts.run_behavior_diverse_retrieval_pilot_v2 import train_behavior_diverse_encoder
from src.representation.history import dynamics_transition_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir
from src.utils.seed import set_global_seed

ARMS = ("state_only", "hc_full_05c", "hc_zero_latent")
CHECKPOINTS_A1 = (20000,)
CHECKPOINTS_B = (0, 201, 1001, 2001, 5001, 10001, 15001, 20000)
ORIGINAL_CONTEXT = e1.frozen_context
ORIGINAL_UPDATE = e1.update_from_indexed_batch
ACTIVE = None
PROBE_OUTPUTS = []
FIXED_OUTPUTS = []


def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def latent_transform(z, mode):
    """Keep the valid-history flag; remove only 05c coordinates in control."""
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 1 or len(z) < 2 or z[-1] not in (0.0, 1.0):
        raise AssertionError("Expected 05c latent plus availability bit")
    if mode == "full":
        return z.copy()
    if mode == "zero":
        ans = np.zeros_like(z)
        ans[-1] = z[-1]
        return ans
    raise ValueError(mode)


def patched_context(mode):
    def f(history, rep, L):
        return latent_transform(ORIGINAL_CONTEXT(history, rep, L), mode)
    return f


def _eval_seed(context, seed, episode):
    base = (70_000_000 + int(seed) * 10_000 if context == "A"
            else 170_000_000 + int(seed) * 20_000 + 10_000)
    return base + int(episode)


@torch.no_grad()
def eval_onpolicy_q(agent, architecture, arm, rep, cfg, seed, context, L, phase, step, n_episodes=5):
    """Greedy rollouts on actual visited state-history inputs; no training update."""
    rows = []
    all_qm = []
    all_greedy = []
    all_valid = []
    all_dist = []
    all_early = []
    for ep in range(n_episodes):
        env = e1.make_env(cfg, context)
        try:
            state, _ = env.reset(seed=_eval_seed(context, seed, ep))
            hist = []
            ret = 0.0
            n_active = 0
            n_noop = 0
            terminated_safety = False
            for t in range(200):
                z = e1.frozen_context(hist, rep, L) if architecture == "history_conditioned" else None
                inp = (np.concatenate([state, z]).astype(np.float32)
                       if z is not None else np.asarray(state, dtype=np.float32))
                q = agent.online(torch.from_numpy(inp).unsqueeze(0)).cpu().numpy()[0]
                a = e1.get_action(agent, state, z, architecture, epsilon=0.0)
                if int(a) != int(np.argmax(q)):
                    raise AssertionError("Evaluated greedy action differs from Q argmax")
                margin = float(q[0] - np.max(q[1:]))
                all_qm.append(margin)
                all_greedy.append(a)
                all_early.append(t < L)
                all_valid.append(float(z[-1]) if z is not None else np.nan)
                all_dist.append(float(np.linalg.norm(state[:2])))
                n_noop += int(a == 0)
                n_active += int(a != 0)
                ns, r, term, trunc, _ = env.step(a)
                ret += float(r)
                hist.append(dynamics_transition_feature(state, int(a), ns, 5))
                state = ns
                if term or trunc:
                    terminated_safety = bool(term)
                    break
            passive = e1b.noop_episode(context, seed, cfg, ep)
            rows.append({"seed":seed,"arm":arm,"phase":phase,"phase_step":step,
                         "eval_context":context,"episode":ep,"raw_return":ret,
                         "noop_return":float(passive["noop_return"]),
                         "advantage_vs_noop":ret - float(passive["noop_return"]),
                         "episode_length":n_noop+n_active,"noop_fraction":n_noop/(n_noop+n_active),
                         "final_distance":float(np.linalg.norm(state[:2])),
                         "safety_terminated":float(terminated_safety)})
        finally:
            env.close()
    margins = np.asarray(all_qm, dtype=float)
    greedy = np.asarray(all_greedy, dtype=int)
    early = np.asarray(all_early, dtype=bool)
    dist = np.asarray(all_dist, dtype=float)
    valid = np.asarray(all_valid, dtype=float)
    summary = {"seed":seed,"arm":arm,"architecture":architecture,
               "phase":phase,"phase_step":int(step),"eval_context":context,
               "n_steps":len(greedy),"n_episodes":len(rows),
               "eval_return_mean":float(np.mean([x["raw_return"] for x in rows])),
               "noop_return_mean":float(np.mean([x["noop_return"] for x in rows])),
               "advantage_vs_noop":float(np.mean([x["advantage_vs_noop"] for x in rows])),
               "eval_noop_fraction":float(np.mean(greedy == 0)),
               "onpolicy_noop_margin_mean":float(np.mean(margins)),
               "onpolicy_noop_margin_median":float(np.median(margins)),
               "onpolicy_noop_margin_positive_fraction":float(np.mean(margins >= 0)),
               "onpolicy_early_noop_fraction":float(np.mean(greedy[early] == 0)) if np.any(early) else np.nan,
               "onpolicy_late_noop_fraction":float(np.mean(greedy[~early] == 0)) if np.any(~early) else np.nan,
               "onpolicy_early_noop_margin_mean":float(np.mean(margins[early])) if np.any(early) else np.nan,
               "onpolicy_late_noop_margin_mean":float(np.mean(margins[~early])) if np.any(~early) else np.nan,
               "onpolicy_latent_valid_fraction":float(np.mean(valid)) if architecture == "history_conditioned" else np.nan,
               "onpolicy_mean_state_distance":float(np.mean(dist)),
               "final_distance":float(np.mean([x["final_distance"] for x in rows])),
               "safety_fraction":float(np.mean([x["safety_terminated"] for x in rows]))}
    for flag, label in [(dist < 1.5,"near"),(dist >= 1.5,"far")]:
        summary[f"onpolicy_{label}_n"] = int(np.sum(flag))
        summary[f"onpolicy_{label}_margin_mean"] = float(np.mean(margins[flag])) if np.any(flag) else np.nan
        summary[f"onpolicy_{label}_noop_fraction"] = float(np.mean(greedy[flag] == 0)) if np.any(flag) else np.nan
    return summary, rows


@torch.no_grad()
def eval_fixed_q(agent, architecture, arm, probes, seed, phase, step):
    out = []
    for ctx, v in probes.items():
        z = v["latent"].copy()
        if arm == "hc_zero_latent":
            z[:, :-1] = 0.0
        x = v["state"] if architecture == "state_only" else np.concatenate([v["state"], z], axis=1)
        q = agent.online(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        margin = q[:,0] - np.max(q[:,1:], axis=1)
        out.append({"seed":seed,"arm":arm,"phase":phase,"phase_step":step,
                    "probe_context":ctx,"probe_n":len(x),
                    "probe_latent_valid_fraction":float(np.mean(z[:,-1])),
                    "probe_noop_margin_mean":float(np.mean(margin)),
                    "probe_noop_greedy_fraction":float(np.mean(np.argmax(q,axis=1)==0)),
                    "probe_noop_margin_median":float(np.median(margin)),
                    "probe_q_abs_mean":float(np.mean(np.abs(q)))})
    return out


def snapshot(agent, replay, rep, cfg, probes, seed, arm, architecture, phase, step, L):
    """Exact live-parameter/RNG noninterference check; only clone is evaluated."""
    before_rng = e1c._rng_fingerprint(agent, replay)
    before_model = e1c._agent_parameter_fingerprint(agent)
    shadow = copy.deepcopy(agent)
    for context in ("A", "B"):
        summary, eps = eval_onpolicy_q(shadow, architecture, arm, rep, cfg, seed, context, L, phase, step)
        PROBE_OUTPUTS.append(summary)
        # Keep per-episode paired raw returns as a separate output.
        EPISODES.extend(eps)
    FIXED_OUTPUTS.extend(eval_fixed_q(shadow, architecture, arm, probes, seed, phase, step))
    del shadow
    if before_rng != e1c._rng_fingerprint(agent, replay):
        raise AssertionError("Read-only probes perturbed learner or replay RNG")
    if before_model != e1c._agent_parameter_fingerprint(agent):
        raise AssertionError("Read-only probes perturbed model or optimizer")


EPISODES = []


def audit_update(agent, replay, store, architecture, batch_size):
    """Original E1 update, followed by read-only snapshots at frozen timestamps."""
    if ACTIVE is None:
        raise RuntimeError("E1d update outside audited common phase")
    a = ACTIVE
    event = int(a["update_index"])
    global_step = int(a["first_update_global"] + a["freq"]*event)
    phase_step = global_step - int(a["start_global"]) + 1
    result = ORIGINAL_UPDATE(agent, replay, store, architecture, batch_size)
    a["update_index"] += 1
    if phase_step in a["snapshot_steps"]:
        snapshot(agent,replay,a["rep"],a["cfg"],a["probes"],a["seed"],
                 a["arm"],architecture,a["phase"],phase_step,a["L"])
    return result


def begin_phase(phase, seed, arm, cfg, rep, probes, L, start_global):
    freq = int(cfg["experiment"]["train_frequency"])
    first = max(int(start_global), int(cfg["experiment"]["learning_starts"]))
    first += (-first) % freq
    return {"phase":phase,"seed":seed,"arm":arm,"cfg":cfg,"rep":rep,"probes":probes,
            "L":L,"freq":freq,"start_global":start_global,
            "first_update_global":first,"update_index":0,
            "snapshot_steps":set(x for x in CHECKPOINTS_B if 0<x<20000) if phase=="B" else set()}


def validate_frozen_protocol(cfg):
    if cfg["e1"]["development_seeds"] != [1,2,3] or cfg["e1"]["reserved_confirmation_seeds"] != [4,5,6]:
        raise RuntimeError("E1 development/confirmation seeds unexpectedly changed")
    if cfg["e1d"]["development_seeds"] != [1,2,3] or cfg["e1d"]["reserved_confirmation_seeds"] != [4,5,6]:
        raise RuntimeError("E1d seed guard modified")
    if int(cfg["isolation"]["A1_steps"])!=20000 or int(cfg["isolation"]["B_steps"])!=20000:
        raise RuntimeError("Common training phase length changed")
    if int(cfg["representation"]["history_length"])!=8:
        raise RuntimeError("05c history_length changed")
    if int(cfg["isolation"]["eval_episodes"])!=5:
        raise RuntimeError("Evaluation episode count changed")
    if int(cfg["experiment"]["train_frequency"])!=4:
        raise RuntimeError("Update cadence changed")
    if int(cfg["e1c"]["fixed_probe_seed_base"]) != 931000 or int(cfg["e1c"]["fixed_probe_steps_per_episode"]) != 64 or int(cfg["e1c"]["fixed_probe_episodes_per_context"]) != 6:
        raise RuntimeError("E1c fixed-probe protocol changed")
    if list(cfg["e1d"]["arms"]) != list(ARMS):
        raise RuntimeError("E1d ablation arms changed")
    if list(cfg["e1d"]["checkpoints_B"]) != list(CHECKPOINTS_B):
        raise RuntimeError("E1d checkpoint schedule changed")


def compare_baselines(current, qfixed, repo):
    """Old E1b/E1c raw returns, action fractions, and fixed Q must reproduce."""
    old = pd.read_csv(repo/"results/synthetic_v2/e1b_policy_behavior_audit/policy_evaluation_summaries.csv")
    old_q = pd.read_csv(repo/"results/synthetic_v2/e1c_learner_collapse_diagnosis/fixed_q_probes_all_development.csv")
    rows=[]
    for _, r in current.iterrows():
        if r.arm not in ("state_only","hc_full_05c"):
            continue
        arch = "state_only" if r.arm == "state_only" else "history_conditioned"
        if r.phase == "A1" and r.phase_step == 20000 and r.eval_context == "A":
            ref=old[(old.seed==r.seed)&(old.architecture==arch)&(old.phase=="A1")&
                    (old.phase_step==20000)&(old.eval_context=="A")]
        elif r.phase == "B" and r.phase_step == 20000:
            ref=old[(old.seed==r.seed)&(old.architecture==arch)&(old.phase=="post_B")&
                    (old.eval_context==r.eval_context)]
        else:
            continue
        if len(ref)!=1:
            raise AssertionError("Missing/ambiguous E1b reference")
        ref=ref.iloc[0]
        rows.append({"seed":r.seed,"arm":r.arm,"phase":r.phase,"phase_step":r.phase_step,
                     "eval_context":r.eval_context,"return_drift":float(r.eval_return_mean-ref.eval_return_mean),
                     "noop_fraction_drift":float(r.eval_noop_fraction-ref.noop_fraction)})
    assert len(rows)==6, f"Expected 6 per-seed E1b reproduction checks, got {len(rows)}"
    result = pd.DataFrame(rows)
    tolerance = 1e-6
    if result[["return_drift","noop_fraction_drift"]].abs().max().max() > tolerance:
        raise RuntimeError("E1d changed original full/state-only training trajectory")
    checks=[]
    for _,r in qfixed[(qfixed.phase=="B")&(qfixed.phase_step==20000)&
                      (qfixed.arm.isin(["state_only","hc_full_05c"]))].iterrows():
        arch = "state_only" if r.arm=="state_only" else "history_conditioned"
        ref=old_q[(old_q.seed==r.seed)&(old_q.architecture==arch)&(old_q.phase=="B")&
                  (old_q.phase_step==20000)&(old_q.probe_context==r.probe_context)]
        if len(ref)!=1: raise AssertionError("Missing fixed-Q reference")
        ref=ref.iloc[0]
        checks.append({"seed":r.seed,"arm":r.arm,"context":r.probe_context,
                       "margin_drift":float(r.probe_noop_margin_mean-ref.probe_noop_margin_mean),
                       "noop_fraction_drift":float(r.probe_noop_greedy_fraction-ref.probe_noop_greedy_fraction)})
    assert len(checks)==4, f"Expected 4 per-seed Q check rows, got {len(checks)}"
    qc = pd.DataFrame(checks)
    if qc[["margin_drift","noop_fraction_drift"]].abs().max().max() > 1e-4:
        raise RuntimeError("E1d fixed-Q probe does not reproduce E1c")
    return result,qc


def run_seed(seed, cfg, out):
    global ACTIVE
    set_global_seed(int(cfg["representation"]["seed"]))
    behavior_module.GLOBAL_CFG=cfg
    rep = train_behavior_diverse_encoder(cfg)
    if not (rep["relative_mse_improvement"]>=0.20 and
            rep["offline_precision_at_10"]>=0.90 and
            rep["offline_centroid_accuracy"]>=0.90):
        raise RuntimeError("Frozen 05c sanity criteria failed")
    L=int(cfg["representation"]["history_length"])
    probes=e1c.build_fixed_probes(rep,cfg,seed,L)
    before=(len(PROBE_OUTPUTS),len(FIXED_OUTPUTS),len(EPISODES))
    for arm in ARMS:
        set_global_seed(seed)
        architecture = "state_only" if arm=="state_only" else "history_conditioned"
        if architecture=="state_only":
            agent=e1.make_agent(cfg,seed)
        else:
            agent=e1.HistoryConditionedDDQN(cfg,seed,int(rep["model"].encoder.proj[0].out_features))
        replay=e1.IndexedRecentReplayBuffer(capacity=int(cfg["isolation"]["replay_capacity"]),
                                            recent_window=int(cfg["isolation"]["recent_window"]),seed=seed)
        store=e1.LatentStore.empty()
        mode="zero" if arm=="hc_zero_latent" else "full"
        if e1.frozen_context is not ORIGINAL_CONTEXT:
            raise AssertionError("Context transform not restored from previous arm")
        e1.frozen_context=patched_context(mode)
        try:
            ACTIVE=begin_phase("A1",seed,arm,cfg,rep,probes,L,0)
            step, A_ep, A_eval=e1.collect_common(
                agent,replay,store,architecture,rep,"A",cfg,seed,phase_index=0,
                start_global=0,steps=20000,L=L,checkpoint_evals={16000,18000,20000})
            assert step==20000 and len(A_eval)==3
            snapshot(agent,replay,rep,cfg,probes,seed,arm,architecture,"A1",20000,L)
            ACTIVE=begin_phase("B",seed,arm,cfg,rep,probes,L,step)
            snapshot(agent,replay,rep,cfg,probes,seed,arm,architecture,"B",0,L)
            step,B_ep,_=e1.collect_common(
                agent,replay,store,architecture,rep,"B",cfg,seed,phase_index=1,
                start_global=step,steps=20000,L=L)
            assert step==40000 and len(replay)==40000
            snapshot(agent,replay,rep,cfg,probes,seed,arm,architecture,"B",20000,L)
        finally:
            ACTIVE=None
            e1.frozen_context=ORIGINAL_CONTEXT
    a,b,c=before
    current=pd.DataFrame(PROBE_OUTPUTS[a:]); qfixed=pd.DataFrame(FIXED_OUTPUTS[b:]); episodes=pd.DataFrame(EPISODES[c:])
    # Compare to unchanged historical training before writing seed's finalized files.
    reproductions,qchecks=compare_baselines(current,qfixed,ROOT)
    data={"onpolicy_q":current,"fixed_q":qfixed,"evaluation_episodes":episodes,
          "baseline_reproduction":reproductions,"fixed_probe_reproduction":qchecks}
    for name,frame in data.items():
        path=out/f"{name}_seed{seed}.csv"; tmp=path.with_suffix(".csv.partial")
        frame.to_csv(tmp,index=False); tmp.replace(path)
    print(f"E1d seed {seed} complete: {len(current)} on-policy evaluations, "
          f"max E1b return drift {reproductions.return_drift.abs().max():.2g}",flush=True)


def aggregate(seeds, out):
    names=["onpolicy_q","fixed_q","evaluation_episodes","baseline_reproduction","fixed_probe_reproduction"]
    for name in names:
        parts=[out/f"{name}_seed{s}.csv" for s in seeds]
        if not all(p.is_file() for p in parts):
            raise FileNotFoundError(f"Missing {name} result files; first finish all seeds")
        pd.concat([pd.read_csv(p) for p in parts],ignore_index=True).to_csv(
            out/f"{name}_all_development.csv",index=False)
    d=pd.read_csv(out/"onpolicy_q_all_development.csv")
    f=pd.read_csv(out/"fixed_q_all_development.csv")
    b=d[(d.phase=="B")&(d.phase_step.isin([0,1001,5001,10001,20000]))].copy()
    b.to_csv(out/"B_onpolicy_key_checkpoints.csv",index=False)
    merged=d.merge(f,left_on=["seed","arm","phase","phase_step","eval_context"],
                   right_on=["seed","arm","phase","phase_step","probe_context"],validate="one_to_one")
    merged["noop_fraction_on_minus_fixed"] = (merged["eval_noop_fraction"]-
                                              merged["probe_noop_greedy_fraction"])
    merged["noop_margin_on_minus_fixed"] = (merged["onpolicy_noop_margin_mean"]-
                                            merged["probe_noop_margin_mean"])
    merged.to_csv(out/"on_vs_fixed_q_comparison.csv",index=False)
    records=[]
    for seed in seeds:
        sel=d[(d.seed==seed)&(d.phase=="B")&(d.phase_step==20000)&(d.eval_context=="B")]
        sel=sel.set_index("arm")
        h=sel.loc["hc_full_05c"]; z=sel.loc["hc_zero_latent"]
        records.append({"seed":seed,"context":"B","full_advantage_vs_noop":h.advantage_vs_noop,
                        "zero_advantage_vs_noop":z.advantage_vs_noop,
                        "zero_minus_full_advantage":z.advantage_vs_noop-h.advantage_vs_noop,
                        "full_noop_fraction":h.eval_noop_fraction,"zero_noop_fraction":z.eval_noop_fraction,
                        "zero_minus_full_noop_fraction":z.eval_noop_fraction-h.eval_noop_fraction})
    pd.DataFrame(records).to_csv(out/"latent_ablation_B_endpoint.csv",index=False)
    manifest={"experiment":"E1d v2.16 on-policy Q / frozen-latent ablation development",
              "source_commit":git_sha(),"development_seeds":list(seeds),
              "reserved_seeds_not_run":[4,5,6],
              "arms":list(ARMS),"latent_ablation":"zero 05c coordinates, preserve availability bit and 21-dimensional DDQN architecture",
              "frozen_representation":"05c; trained only on A/B behavior-diverse data",
              "common_phases":"A1 20000 then B 20000, unchanged E1 train code and recent replay",
              "primary_diagnostics":["policy-state Q margin vs same-seed fixed Q", "B-end raw advantage over No-op", "greedy No-op fraction"],
              "mechanism_interpretation":"descriptive, no isolated proof of causality; zero-latent arms change learning trajectories",
              "E1b_reference_reproduction":True,"E1c_fixed_Q_reference_reproduction":True,
              "runtime":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,
                         "gymnasium":gymnasium.__version__,"cuda":torch.cuda.is_available(),"threads":torch.get_num_threads()}}
    (out/"e1d_metadata.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return b


def main():
    global ACTIVE
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/synthetic_v2/e1d_onpolicy_q_representation_ablation.yaml")
    parser.add_argument("--seeds",nargs="+",type=int,default=[1,2,3])
    parser.add_argument("--aggregate-only",action="store_true")
    args=parser.parse_args()
    seeds=list(args.seeds)
    if len(seeds)!=len(set(seeds)) or not seeds or any(s not in (1,2,3) for s in seeds):
        raise RuntimeError("Only development seeds 1/2/3 may run")
    cfg=load_config(ROOT/"configs/base.yaml",ROOT/args.config)
    validate_frozen_protocol(cfg)
    if torch.cuda.is_available():
        raise RuntimeError("E1d uses canonical CPU runtime")
    torch.set_num_threads(1)
    out=ensure_dir(ROOT/"results/synthetic_v2/e1d_onpolicy_q_representation_ablation")
    if not args.aggregate_only:
        if e1.update_from_indexed_batch is not ORIGINAL_UPDATE or e1.frozen_context is not ORIGINAL_CONTEXT:
            raise RuntimeError("Begin from clean E1 process; function already patched")
        e1.update_from_indexed_batch=audit_update
        try:
            for seed in seeds:
                print(f"=== E1d development seed {seed} ===",flush=True)
                run_seed(seed,cfg,out)
        finally:
            e1.update_from_indexed_batch=ORIGINAL_UPDATE
            e1.frozen_context=ORIGINAL_CONTEXT
            ACTIVE=None
    if seeds==[1,2,3]:
        b=aggregate(seeds,out)
        print("\nE1d B-stage on-policy key checkpoints:\n",b[["seed","arm","phase_step",
              "eval_context","eval_noop_fraction","onpolicy_noop_margin_mean",
              "advantage_vs_noop"]].to_string(index=False))
    else:
        print("Seed results saved. Aggregate after all seeds 1/2/3 finish.")


if __name__=="__main__":
    main()
