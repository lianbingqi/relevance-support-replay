from __future__ import annotations
import argparse, copy, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import scripts.run_behavior_diverse_retrieval_pilot_v2 as behavior_module
from scripts.run_behavior_diverse_retrieval_pilot_v2 import train_behavior_diverse_encoder
from scripts.run_learned_relevance_retrieval_pilot_v2 import encode_histories
from scripts.run_learned_replay_recurrence_pilot_v2 import history_bank_windows,make_learned_from_recent,train_common_phase_logged,uniform_cap_bank
from scripts.train_stationary_v2 import epsilon_at,make_agent,make_env
from src.replay.recent import RecentReplayBuffer
from src.replay.uniform import UniformReplayBuffer
from src.representation.history import dynamics_transition_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir,save_json,save_yaml
from src.utils.seed import set_global_seed

def git_sha(): return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

@torch.no_grad()
def greedy(agent,state):
    x=torch.as_tensor(np.asarray(state,dtype=np.float32),device=agent.device).unsqueeze(0)
    return int(agent.online(x).argmax(1).item())

def evaluate(agent,ctx,seed,cfg):
    env=make_env(cfg,ctx); n=int(cfg["novelty"]["eval_episodes"]); off={"A":0,"B":5000,"D":10000}[ctx]
    rs=[]; saf=[]
    for ep in range(n):
        s,_=env.reset(seed=170_000_000+seed*20_000+off+ep); done=False; ret=0.; term0=False
        while not done:
            a=greedy(agent,s); s,r,term,trunc,_=env.step(a); done=term or trunc; ret+=r; term0=bool(term)
        rs.append(ret); saf.append(float(term0))
    env.close()
    return {"return_mean":float(np.mean(rs)),"return_sd":float(np.std(rs)),"safety_fraction":float(np.mean(saf))}

def random_D(seed,cfg):
    env=make_env(cfg,"D"); n=int(cfg["novelty"]["random_baseline_episodes"]); rng=np.random.default_rng(171_000_000+seed); rs=[]
    for ep in range(n):
        s,_=env.reset(seed=171_500_000+seed*1000+ep); done=False; ret=0.
        while not done:
            a=int(rng.integers(env.action_space.n)); s,r,term,trunc,_=env.step(a); done=term or trunc; ret+=r
        rs.append(ret)
    env.close(); return float(np.mean(rs))

def train_D_reference(seed,cfg):
    nov=cfg["novelty"]; set_global_seed(seed); agent=make_agent(cfg,seed); replay=UniformReplayBuffer(50000,seed=seed); env=make_env(cfg,"D")
    s,info=env.reset(seed=172_000_000+seed*1_000_000); step=0; ep=0; evals=set(map(int,nov["D_reference_eval_steps"])); rows=[]
    while step<int(nov["D_reference_steps"]):
        a=agent.act(s,epsilon=epsilon_at(step,cfg)); ns,r,term,trunc,ninfo=env.step(a); done=term or trunc
        replay.add(s,a,r,ns,term,context_id="D",context_vector=info["context_vector"])
        if step>=int(cfg["experiment"]["learning_starts"]) and step%int(cfg["experiment"]["train_frequency"])==0 and len(replay)>=int(cfg["agent"]["batch_size"]):
            agent.update(replay.sample(int(cfg["agent"]["batch_size"])))
        s,info=ns,ninfo; step+=1
        if step in evals: rows.append({"step":step,**evaluate(agent,"D",seed,cfg)})
        if done and step<int(nov["D_reference_steps"]):
            ep+=1; s,info=env.reset(seed=172_000_000+seed*1_000_000+ep)
    env.close(); df=pd.DataFrame(rows); return df,float(df["return_mean"].mean())

def collect_probe(agent,ctx,global_step,seed,cfg):
    nov=cfg["novelty"]; agent=copy.deepcopy(agent); env=make_env(cfg,ctx); s,_=env.reset(seed=173_000_000+seed*1_000_000)
    eps=[]; states=[np.asarray(s,dtype=np.float32)]; actions=[]; start=0; eid=0; p=0
    while p<int(nov["probe_steps"]):
        a=agent.act(s,epsilon=epsilon_at(global_step+p,cfg)); ns,_,term,trunc,_=env.step(a); actions.append(int(a)); states.append(np.asarray(ns,dtype=np.float32)); s=ns; p+=1
        if term or trunc:
            eps.append({"episode_id":eid,"context":ctx,"states":np.asarray(states,dtype=np.float32),"actions":np.asarray(actions,dtype=np.int64),"start_phase_step":start})
            eid+=1; states=[]; actions=[]
            if p<int(nov["probe_steps"]):
                s,_=env.reset(seed=173_000_000+seed*1_000_000+eid); states=[np.asarray(s,dtype=np.float32)]; start=p
    if actions: eps.append({"episode_id":eid,"context":ctx,"states":np.asarray(states,dtype=np.float32),"actions":np.asarray(actions,dtype=np.int64),"start_phase_step":start})
    env.close(); return eps

def probe_windows(eps,L,stride):
    H=[]; steps=[]; ids=[]; labs=[]
    for e in eps:
        S,A=e["states"],e["actions"]; T=len(A)
        if T<L: continue
        F=[dynamics_transition_feature(S[t],int(A[t]),S[t+1],5) for t in range(T)]
        for end in range(L,T+1,stride):
            H.append(np.stack(F[end-L:end]).astype(np.float32)); steps.append(e["start_phase_step"]+end); ids.append(e["episode_id"]); labs.append(e["context"])
    return {"history":np.stack(H).astype(np.float32),"phase_step":np.asarray(steps),"episode_id":np.asarray(ids),"label":np.asarray(labs)}

def fit_support(bank_z,seed,cfg):
    sup=cfg["support"]; rng=np.random.default_rng(174_000_000+seed); perm=rng.permutation(len(bank_z)); nref=int(round(len(perm)*float(sup["reference_fraction"])))
    ref=bank_z[perm[:nref]].astype(np.float64); cal=bank_z[perm[nref:]].astype(np.float64)
    mean=ref.mean(0); std=ref.std(0); std=np.where(std<1e-6,1.,std); refs=(ref-mean)/std; cals=(cal-mean)/std
    knn=NearestNeighbors(n_neighbors=int(sup["knn_k"]),metric="euclidean").fit(refs)
    cd=knn.kneighbors(cals,return_distance=True)[0].mean(1); thr=float(np.quantile(cd,float(sup["calibration_quantile"])))
    return {"knn":knn,"mean":mean,"std":std,"threshold":thr,"nref":len(ref),"ncal":len(cal)}

def support_dist(sm,z):
    zs=(z.astype(np.float64)-sm["mean"])/sm["std"]; return sm["knn"].kneighbors(zs,return_distance=True)[0].mean(1)

def cosine_diag(qz,bz,bl,k):
    bn=bz/np.maximum(np.linalg.norm(bz,axis=1,keepdims=True),1e-8); qn=qz/np.maximum(np.linalg.norm(qz,axis=1,keepdims=True),1e-8); rows=[]
    for q in qn:
        sim=bn@q; kk=min(k,len(sim)); idx=np.argpartition(-sim,kk-1)[:kk]; idx=idx[np.argsort(-sim[idx])]; labs=bl[idx]
        rows.append({"best_similarity":float(sim[idx[0]]),"mean_top50_similarity":float(sim[idx].mean()),"top50_A_fraction_diagnostic":float(np.mean(labs=="A")),"top50_B_fraction_diagnostic":float(np.mean(labs=="B"))})
    return pd.DataFrame(rows)

def train_D_branch(method,agent,replay,global_step,seed,cfg,encoder=None,hstd=None,L=None,bank_labels=None):
    nov=cfg["novelty"]; env=make_env(cfg,"D"); bs=int(cfg["agent"]["batch_size"]); freq=int(cfg["experiment"]["train_frequency"]); evals=set(map(int,nov["D_eval_steps"]))
    zero=evaluate(agent,"D",seed,cfg); rows=[{"method":method,"phase_step":0,"global_step":global_step,"eval_return_mean":zero["return_mean"],"eval_return_sd":zero["return_sd"],"eval_safety_termination_fraction":zero["safety_fraction"]}]
    s,info=env.reset(seed=175_000_000+seed*1_000_000); p=0; eid=0; hist=[]; qa=[]; hf=[]; rf=[]; ha=[]; hb=[]; ta=[]; tb=[]; sims=[]
    while p<int(nov["D_steps"]):
        a=agent.act(s,epsilon=epsilon_at(global_step,cfg)); ns,r,term,trunc,ninfo=env.step(a); done=term or trunc
        replay.add(s,a,r,ns,term,context_id="D",context_vector=info["context_vector"])
        if method=="learned": hist.append(dynamics_transition_feature(s,int(a),ns,5))
        if global_step%freq==0 and len(replay)>=bs:
            if method=="learned":
                if len(hist)>=L:
                    q=encode_histories(encoder,np.stack(hist[-L:]).astype(np.float32)[None,:,:],hstd)[0]; replay.set_query_embedding(q)
                else: replay.clear_query()
            batch=replay.sample(bs); agent.update(batch)
            if method=="learned":
                st=replay.last_sample_stats; qa.append(float(st["query_available"])); hf.append(float(st["historical_fraction"])); rf.append(float(st["recent_fraction"])); sims.append(float(st["best_similarity"]) if np.isfinite(st["best_similarity"]) else np.nan)
                hi=replay.last_historical_indices
                if len(hi):
                    labs=np.asarray([replay.buffer[int(i)].context_id for i in hi]); ha.append(float(np.mean(labs=="A"))); hb.append(float(np.mean(labs=="B")))
                tr=replay.last_top_bank_rows
                if bank_labels is not None and len(tr):
                    labs=bank_labels[tr]; ta.append(float(np.mean(labs=="A"))); tb.append(float(np.mean(labs=="B")))
        s,info=ns,ninfo; p+=1; global_step+=1
        if p in evals:
            ev=evaluate(agent,"D",seed,cfg); rows.append({"method":method,"phase_step":p,"global_step":global_step,"eval_return_mean":ev["return_mean"],"eval_return_sd":ev["return_sd"],"eval_safety_termination_fraction":ev["safety_fraction"]})
        if done:
            hist=[]
            if p<int(nov["D_steps"]):
                eid+=1; s,info=env.reset(seed=175_000_000+seed*1_000_000+eid)
    env.close()
    mean=lambda x: float(np.nanmean(x)) if x else np.nan
    return rows,{"method":method,"learned_query_available_update_fraction":mean(qa),"mean_learned_historical_fraction":mean(hf),"mean_learned_recent_fraction":mean(rf),"mean_learned_historical_A_fraction_diagnostic":mean(ha),"mean_learned_historical_B_fraction_diagnostic":mean(hb),"mean_learned_top50_A_window_fraction_diagnostic":mean(ta),"mean_learned_top50_B_window_fraction_diagnostic":mean(tb),"mean_learned_best_similarity":mean(sims)}

def first_recovery(df,col,thr,consecutive):
    good=df[col].to_numpy()>=thr
    for i in range(len(good)-consecutive+1):
        if np.all(good[i:i+consecutive]): return int(df.iloc[i]["phase_step"])
    return None

def auc2k(df,col):
    d=df[(df.phase_step>=0)&(df.phase_step<=2000)].sort_values("phase_step"); x=d.phase_step.to_numpy(float); y=d[col].to_numpy(float)
    if x[0]!=0 or x[-1]!=2000: raise RuntimeError("AUC endpoints missing")
    return float(np.trapz(y,x)/2000.)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/synthetic_v2/novelty_support_pilot.yaml"); args=ap.parse_args()
    cfg=load_config(ROOT/"configs/base.yaml",ROOT/args.config); nov=cfg["novelty"]; seed=int(nov["seed"]); torch.set_num_threads(1)
    out=ensure_dir(ROOT/"results/synthetic_v2/novelty_support_pilot"); figs=ensure_dir(ROOT/"figures/synthetic_v2/novelty_support_pilot")
    behavior_module.GLOBAL_CFG=cfg; set_global_seed(int(cfg["representation"]["seed"])); rep=train_behavior_diverse_encoder(cfg); encoder=rep["model"]; hstd=rep["history_std"]; L=int(cfg["representation"]["history_length"])
    sanity={"relative_mse_improvement_ge_0_20":rep["relative_mse_improvement"]>=.20,"offline_AB_precision_at_10_ge_0_90":rep["offline_precision_at_10"]>=.90,"offline_AB_centroid_accuracy_ge_0_90":rep["offline_centroid_accuracy"]>=.90}
    if not all(sanity.values()): raise RuntimeError(f"05c capability check failed: {sanity}")
    Dref_df,Dref=train_D_reference(seed,cfg); rD=random_D(seed,cfg)
    if Dref<=rD: raise RuntimeError("D reference does not beat random D")
    set_global_seed(seed); agent=make_agent(cfg,seed); replay=RecentReplayBuffer(int(nov["replay_capacity"]),int(nov["recent_window"]),seed=seed); g=0
    g,_,A1=train_common_phase_logged(agent,replay,"A",int(nov["A1_steps"]),g,seed,0,cfg,eval_steps=None)
    g,_,B=train_common_phase_logged(agent,replay,"B",int(nov["B_steps"]),g,seed,1,cfg,eval_steps=None)
    b1=history_bank_windows(A1,L,int(nov["history_bank_stride"])); b2=history_bank_windows(B,L,int(nov["history_bank_stride"]))
    bank={k:np.concatenate([b1[k],b2[k]],axis=0) for k in b1}; bank=uniform_cap_bank(bank,int(nov["max_history_bank_windows"]),seed); bz=encode_histories(encoder,bank["history"],hstd)
    sm=fit_support(bz,seed,cfg)
    Ap=probe_windows(collect_probe(agent,"A",g,seed,cfg),L,int(nov["probe_stride"])); Dp=probe_windows(collect_probe(agent,"D",g,seed,cfg),L,int(nov["probe_stride"]))
    Az=encode_histories(encoder,Ap["history"],hstd); Dz=encode_histories(encoder,Dp["history"],hstd); Ad=support_dist(sm,Az); Dd=support_dist(sm,Dz); thr=sm["threshold"]; As=Ad<=thr; Ds=Dd<=thr
    auroc=float(roc_auc_score(np.r_[np.zeros(len(Ad),int),np.ones(len(Dd),int)],np.r_[Ad,Dd]))
    Ac=cosine_diag(Az,bz,bank["label"],int(nov["learned_retrieval_top_k_windows"])); Dc=cosine_diag(Dz,bz,bank["label"],int(nov["learned_retrieval_top_k_windows"]))
    Aq=pd.concat([pd.DataFrame({"context":"A_recurrence","phase_step":Ap["phase_step"],"episode_id":Ap["episode_id"],"support_distance":Ad,"supported":As}),Ac],axis=1)
    Dq=pd.concat([pd.DataFrame({"context":"D_novel","phase_step":Dp["phase_step"],"episode_id":Dp["episode_id"],"support_distance":Dd,"supported":Ds}),Dc],axis=1)
    queries=pd.concat([Aq,Dq],ignore_index=True)
    ss=pd.DataFrame([{"support_threshold":thr,"support_knn_k":int(cfg["support"]["knn_k"]),"support_calibration_quantile":float(cfg["support"]["calibration_quantile"]),"n_A_queries":len(Aq),"n_D_queries":len(Dq),"A_support_rate":float(As.mean()),"D_support_rate":float(Ds.mean()),"A_support_distance_mean":float(Ad.mean()),"D_support_distance_mean":float(Dd.mean()),"A_vs_D_anomaly_AUROC":auroc,"A_best_cosine_mean":float(Aq.best_similarity.mean()),"D_best_cosine_mean":float(Dq.best_similarity.mean()),"A_mean_top50_cosine":float(Aq.mean_top50_similarity.mean()),"D_mean_top50_cosine":float(Dq.mean_top50_similarity.mean()),"D_top50_A_fraction_diagnostic":float(Dq.top50_A_fraction_diagnostic.mean()),"D_top50_B_fraction_diagnostic":float(Dq.top50_B_fraction_diagnostic.mean())}])
    crit=cfg["criteria"]; Asp=bool(As.mean()>=float(crit["min_recurrence_A_support_rate"])); Dsp=bool(Ds.mean()<=float(crit["max_novel_D_support_rate"])); AUp=bool(auroc>=float(crit["min_A_vs_D_anomaly_auroc"])); support_pass=Asp and Dsp and AUp
    ra,la=copy.deepcopy(agent),copy.deepcopy(agent); rr=copy.deepcopy(replay)
    lcfg={**cfg,"isolation":{"replay_capacity":int(nov["replay_capacity"]),"recent_window":int(nov["recent_window"]),"learned_historical_fraction":float(nov["learned_historical_fraction"]),"learned_retrieval_top_k_windows":int(nov["learned_retrieval_top_k_windows"])}}
    lr=make_learned_from_recent(replay,bank,bz,seed,lcfg)
    rz=evaluate(ra,"D",seed,cfg)["return_mean"]; lz=evaluate(la,"D",seed,cfg)["return_mean"]
    if not np.isclose(rz,lz,rtol=0,atol=1e-10): raise RuntimeError("Fork mismatch")
    rrows,rdiag=train_D_branch("recent",ra,rr,g,seed,cfg); lrows,ldiag=train_D_branch("learned",la,lr,g,seed,cfg,encoder,hstd,L,bank["label"])
    curve=pd.DataFrame(rrows+lrows); curve["normalized_D_score"]=(curve.eval_return_mean-rD)/(Dref-rD)
    mrows=[]
    for method in ["recent","learned"]:
        d=curve[curve.method==method].sort_values("phase_step"); rec=first_recovery(d,"normalized_D_score",float(nov["recovery_threshold"]),int(nov["recovery_consecutive_checkpoints"])); post=d[(d.phase_step>0)&(d.phase_step<=2000)]
        mrows.append({"method":method,"zero_shot_score":float(d.iloc[0].normalized_D_score),"recovery_step":rec if rec is not None else np.nan,"AUC_0_2000":auc2k(d,"normalized_D_score"),"post_switch_mean_score_0_2000":float(post.normalized_D_score.mean()),"final_score":float(d.iloc[-1].normalized_D_score)})
    metrics=pd.DataFrame(mrows).set_index("method")
    censor=lambda method: float(metrics.loc[method,"recovery_step"]) if pd.notna(metrics.loc[method,"recovery_step"]) else float(nov["D_steps"])+100.
    delay=censor("learned")-censor("recent"); auc_dis=float(metrics.loc["recent","AUC_0_2000"]-metrics.loc["learned","AUC_0_2000"])
    rec_fail=delay>=float(crit["min_negative_transfer_recovery_delay"]); auc_fail=auc_dis>=float(crit["min_negative_transfer_auc_disadvantage"]); neg=bool(rec_fail or auc_fail)
    metrics.reset_index().to_csv(out/"adaptation_metrics.csv",index=False); curve.to_csv(out/"D_adaptation_curve.csv",index=False); pd.DataFrame([rdiag,ldiag]).to_csv(out/"sampler_diagnostics.csv",index=False); queries.to_csv(out/"support_probe_queries.csv",index=False); ss.to_csv(out/"support_probe_summary.csv",index=False); Dref_df.to_csv(out/"D_reference_evaluations.csv",index=False)
    pd.DataFrame([{"D_reference_return":Dref,"random_D_return":rD}]).to_csv(out/"D_normalization_reference.csv",index=False)
    pd.DataFrame([{"n_bank_windows":len(bank["history"]),"bank_A_fraction_diagnostic_only":float(np.mean(bank["label"]=="A")),"history_length":L,"bank_stride":int(nov["history_bank_stride"]),"retrieval_top_k_windows":int(nov["learned_retrieval_top_k_windows"]),"support_reference_windows":sm["nref"],"support_calibration_windows":sm["ncal"]}]).to_csv(out/"historical_bank_summary.csv",index=False)
    pd.DataFrame([{"offline_history_test_mse":rep["history_mse"],"offline_no_history_test_mse":rep["baseline_mse"],"offline_relative_mse_improvement":rep["relative_mse_improvement"],"offline_AB_precision_at_10":rep["offline_precision_at_10"],"offline_AB_centroid_accuracy":rep["offline_centroid_accuracy"],"encoder_best_epoch":int(rep["model_meta"]["best_epoch"])}]).to_csv(out/"encoder_offline_summary.csv",index=False)
    save_json({"code_commit":git_sha(),"seed":seed,"protocol":"A1_20k_to_B_20k_to_D_10k","D_reference_return":Dref,"random_D_return":rD,"fork_zero_shot_return":float(rz),"learned_selector_uses_true_context_labels":False,"support_threshold_uses_context_labels":False,"support_metric":"mean whitened Euclidean distance to 20 nearest reference bank embeddings","support_threshold_rule":"95th percentile of label-free bank calibration distances","support_A_rate":float(As.mean()),"support_D_rate":float(Ds.mean()),"support_A_vs_D_AUROC":auroc,"A_support_rate_pass":Asp,"D_support_rate_pass":Dsp,"support_AUROC_pass":AUp,"support_diagnostic_pass":support_pass,"learned_recovery_delay_vs_recent":delay,"learned_AUC_disadvantage_vs_recent":auc_dis,"negative_transfer_recovery_screen":bool(rec_fail),"negative_transfer_AUC_screen":bool(auc_fail),"relevance_negative_transfer_observed":neg,"next_if_both_pass":"freeze this support rule and implement support-aware replay without tuning on D","next_if_no_negative_transfer":"do not tune D or relevance to force a failure","next_if_support_fails":"do not tune this threshold on D; diagnose support estimator separately"},out/"pilot_metadata.json"); save_yaml(cfg,out/"resolved_config.yaml")
    fig,ax=plt.subplots(figsize=(8.5,5)); 
    for ctx,grp in queries.groupby("context"): ax.hist(grp.support_distance,bins=35,alpha=.45,label=ctx)
    ax.axvline(thr,ls="--",label="support threshold"); ax.set_xlabel("Whitened latent kNN support distance"); ax.set_ylabel("Query count"); ax.set_title("Recurrence vs Novelty Support Diagnostic"); ax.legend(); fig.tight_layout(); fig.savefig(figs/"support_distance_A_vs_D.png",dpi=180); plt.close(fig)
    for mx,fn in [(2000,"D_adaptation_first_2000.png"),(int(nov["D_steps"]),"D_adaptation_full.png")]:
        fig,ax=plt.subplots(figsize=(8.5,5))
        for method in ["recent","learned"]:
            d=curve[(curve.method==method)&(curve.phase_step<=mx)]; ax.plot(d.phase_step,d.normalized_D_score,marker="o",label=method)
        ax.axhline(float(nov["recovery_threshold"]),ls="--",label="recovery threshold"); ax.set_xlabel("D training transitions"); ax.set_ylabel("Normalized D performance"); ax.set_title("Novel-D Adaptation"); ax.legend(); fig.tight_layout(); fig.savefig(figs/fn,dpi=180); plt.close(fig)
    print("\n=== Support diagnostic ==="); print(ss.to_string(index=False)); print("\nSupport diagnostic pass:",support_pass)
    print("\n=== D adaptation metrics ==="); print(metrics.reset_index().to_string(index=False)); print("\nLearned recovery delay vs Recent:",delay); print("Learned AUC disadvantage vs Recent:",auc_dis); print("Relevance negative transfer observed:",neg)
if __name__=="__main__": main()
