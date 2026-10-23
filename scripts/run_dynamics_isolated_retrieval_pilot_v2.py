from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from scripts.run_context_representation_pilot_v2 import Std,cat,collect,fit_models,pred_hist,pred_base,retrieval,sub
from scripts.run_learned_relevance_retrieval_pilot_v2 import balance_bank,encode_histories,run_phase
from scripts.train_stationary_v2 import make_agent
from src.replay.recent import RecentReplayBuffer
from src.representation.history import current_feature,dynamics_transition_feature
from src.representation.similarity import topk_label_precision
from src.utils.config import load_config
from src.utils.logger import ensure_dir,save_json,save_yaml
from src.utils.seed import set_global_seed

def git_sha(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def windows_dyn(episodes,cfg):
    r=cfg['representation']; L=int(r['history_length']); stride=int(r['sample_stride'])
    H=[]; C=[]; Y=[]; labels=[]; ids=[]
    for e in episodes:
        S=e['states']; A=e['actions']; T=len(A)
        if T<=L: continue
        feat=[dynamics_transition_feature(S[t],int(A[t]),S[t+1],5) for t in range(T)]
        for t in range(L,T,stride):
            H.append(np.stack(feat[t-L:t]).astype(np.float32))
            C.append(current_feature(S[t],int(A[t]),5))
            Y.append((S[t+1]-S[t]).astype(np.float32))
            labels.append(e['context']); ids.append(int(e['episode_id']))
    return {'history':np.stack(H).astype(np.float32),'current':np.stack(C).astype(np.float32),'target':np.stack(Y).astype(np.float32),'label':np.asarray(labels),'episode_id':np.asarray(ids,dtype=np.int64)}

def online_windows_dyn(episodes,L,stride):
    H=[]; labels=[]; phase=[]; epids=[]; local=[]
    for e in episodes:
        S=e['states']; A=e['actions']; T=len(A)
        if T<L: continue
        feat=[dynamics_transition_feature(S[t],int(A[t]),S[t+1],5) for t in range(T)]
        for end in range(L,T+1,stride):
            H.append(np.stack(feat[end-L:end]).astype(np.float32)); labels.append(e['context'])
            phase.append(int(e['start_phase_step']+end)); epids.append(int(e['episode_id'])); local.append(int(end))
    return {'history':np.stack(H).astype(np.float32),'label':np.asarray(labels),'phase_step':np.asarray(phase,dtype=np.int64),'episode_id':np.asarray(epids,dtype=np.int64),'episode_local_step':np.asarray(local,dtype=np.int64)}

def train_rep(cfg):
    r=cfg['representation']; nt=int(r['train_episodes_per_context']); nv=int(r['val_episodes_per_context']); nte=int(r['test_episodes_per_context']); total=nt+nv+nte
    comb=cat([windows_dyn(collect('A',total,cfg),cfg),windows_dyn(collect('B',total,cfg),cfg)])
    ids=comb['episode_id']; tr=sub(comb,ids<nt); va=sub(comb,(ids>=nt)&(ids<nt+nv)); te=sub(comb,ids>=nt+nv)
    hs=Std.fit(tr['history'].reshape(-1,tr['history'].shape[-1])); cs=Std.fit(tr['current']); ys=Std.fit(tr['target'])
    def standard(d):
        q={k:v.copy() for k,v in d.items()}; q['history']=hs.t(d['history']); q['current']=cs.t(d['current']); q['target']=ys.t(d['target']); return q
    trs,vas,tes=standard(tr),standard(va),standard(te)
    model,base,hcurve,bcurve,hmeta,bmeta=fit_models(trs,vas,cfg)
    hp,ztest=pred_hist(model,tes,r['batch_size']); bp=pred_base(base,tes,r['batch_size']); hp=ys.inv(hp); bp=ys.inv(bp); y=te['target']
    hmse=float(np.mean((hp-y)**2)); bmse=float(np.mean((bp-y)**2)); rel=float(1-hmse/bmse)
    _,ztrain=pred_hist(model,trs,r['batch_size']); prec,acc,_,_=retrieval(ztrain,tr['label'],ztest,te['label'],10,5000,int(r['seed']))
    model.eval()
    return {'model':model,'history_std':hs,'history_curve':hcurve,'model_meta':hmeta,'offline_history_mse':hmse,'offline_no_history_mse':bmse,'offline_relative_mse_improvement':rel,'offline_precision_at_10':float(prec),'offline_centroid_accuracy':float(acc)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/synthetic_v2/dynamics_isolated_retrieval_pilot.yaml'); args=ap.parse_args()
    cfg=load_config(ROOT/'configs/base.yaml',ROOT/args.config); r=cfg['representation']; q=cfg['retrieval']; seed=int(q['seed']); torch.set_num_threads(1); set_global_seed(seed)
    out=ensure_dir(ROOT/'results/synthetic_v2/dynamics_isolated_retrieval_pilot'); figs=ensure_dir(ROOT/'figures/synthetic_v2/dynamics_isolated_retrieval_pilot')
    rep=train_rep(cfg); model=rep['model']; hs=rep['history_std']
    agent=make_agent(cfg,seed); replay=RecentReplayBuffer(int(q['replay_capacity']),int(q['recent_window']),seed=seed); g=0
    g,A1=run_phase(agent,replay,'A',int(q['A1_steps']),g,seed,0,cfg)
    g,B=run_phase(agent,replay,'B',int(q['B_steps']),g,seed,1,cfg)
    g,A2=run_phase(agent,replay,'A',int(q['A2_steps']),g,seed,2,cfg)
    L=int(r['history_length']); a1=online_windows_dyn(A1,L,int(q['bank_stride'])); b=online_windows_dyn(B,L,int(q['bank_stride'])); a2=online_windows_dyn(A2,L,int(q['query_stride']))
    bank_h=np.concatenate([a1['history'],b['history']]); bank_y=np.concatenate([a1['label'],b['label']]); bank_steps=np.concatenate([a1['phase_step'],b['phase_step']])
    bank_z=encode_histories(model,bank_h,hs); query_z=encode_histories(model,a2['history'],hs)
    bank_z,bank_y,bank_steps=balance_bank(bank_z,bank_y,bank_steps,int(q['max_bank_windows_per_context']),seed)
    r10=topk_label_precision(query_z,bank_z,bank_y,'A',int(q['top_k_primary'])); r50=topk_label_precision(query_z,bank_z,bank_y,'A',int(q['top_k_secondary']))
    df=pd.DataFrame({'phase_step':a2['phase_step'],'episode_id':a2['episode_id'],'episode_local_step':a2['episode_local_step'],'precision_at_10':r10['precision_per_query'],'precision_at_50':r50['precision_per_query'],'best_similarity':r10['best_similarity'],'same_vs_other_margin':r10['same_vs_other_margin']})
    early=df['phase_step']<=int(q['early_query_max_step']); ep=float(df.loc[early,'precision_at_10'].mean()); fp=float(df['precision_at_10'].mean()); fp50=float(df['precision_at_50'].mean()); margin=float(df.loc[early,'same_vs_other_margin'].mean()); bankA=float(np.mean(bank_y=='A'))
    crit=cfg['criteria']; epass=ep>=float(crit['min_early_precision_at_10']); fpass=fp>=float(crit['min_first2k_precision_at_10']); passed=bool(epass and fpass)
    old=pd.read_csv(ROOT/'results/synthetic_v2/learned_relevance_retrieval_pilot/retrieval_summary.csv').iloc[0]
    cmp=pd.DataFrame([
      {'feature':'state_action_delta','early_A2_precision_at_10':float(old['early_A2_precision_at_10']),'first2k_precision_at_10':float(old['first2k_precision_at_10']),'first2k_precision_at_50':float(old['first2k_precision_at_50'])},
      {'feature':'action_delta_only','early_A2_precision_at_10':ep,'first2k_precision_at_10':fp,'first2k_precision_at_50':fp50},
    ]); cmp.to_csv(out/'comparison_to_stateful_feature.csv',index=False)
    summary=pd.DataFrame([{'history_feature':'action_delta_only','history_input_dim':int(bank_h.shape[-1]),'n_bank_windows':int(len(bank_z)),'bank_A_fraction':bankA,'n_A2_queries':int(len(df)),'offline_history_test_mse':rep['offline_history_mse'],'offline_no_history_test_mse':rep['offline_no_history_mse'],'offline_relative_mse_improvement':rep['offline_relative_mse_improvement'],'offline_AB_precision_at_10':rep['offline_precision_at_10'],'offline_AB_centroid_accuracy':rep['offline_centroid_accuracy'],'early_A2_precision_at_10':ep,'first2k_precision_at_10':fp,'first2k_precision_at_50':fp50,'early_same_vs_other_margin':margin,'early_pass':epass,'first2k_pass':fpass,'pilot_pass':passed}]); summary.to_csv(out/'retrieval_summary.csv',index=False); df.to_csv(out/'A2_query_metrics.csv',index=False)
    pd.DataFrame([{'context':lab,'n_windows':int(np.sum(bank_y==lab))} for lab in ['A','B']]).to_csv(out/'historical_bank_counts.csv',index=False)
    bins=[0,64,128,256,512,1024,2000]; labs=['8-64','65-128','129-256','257-512','513-1024','1025-2000']; df['step_bin']=pd.cut(df['phase_step'],bins=bins,labels=labs,include_lowest=True)
    binned=df.groupby('step_bin',observed=False).agg(mean_precision_at_10=('precision_at_10','mean'),mean_precision_at_50=('precision_at_50','mean'),mean_margin=('same_vs_other_margin','mean'),n_queries=('phase_step','size')).reset_index(); binned.to_csv(out/'retrieval_by_A2_step_bin.csv',index=False)
    rows=[]
    for eid,gp in df.groupby('episode_id'):
        cut=int(gp['episode_local_step'].min())+40; eg=gp[gp['episode_local_step']<=cut]
        rows.append({'episode_id':int(eid),'phase_start':int(gp['phase_step'].min()),'phase_end':int(gp['phase_step'].max()),'mean_precision_at_10':float(gp['precision_at_10'].mean()),'first_40ish_precision_at_10':float(eg['precision_at_10'].mean()),'mean_margin':float(gp['same_vs_other_margin'].mean())})
    pd.DataFrame(rows).to_csv(out/'retrieval_by_episode.csv',index=False); rep['history_curve'].to_csv(out/'encoder_training_curve.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.5,5)); ax.plot(df.phase_step,df.precision_at_10.rolling(15,min_periods=1).mean(),label='precision@10'); ax.plot(df.phase_step,df.precision_at_50.rolling(15,min_periods=1).mean(),label='precision@50'); ax.axhline(bankA,ls='--',label='bank A prevalence'); ax.axhline(.9,ls=':',label='go/no-go threshold'); ax.set_xlabel('A2 phase step'); ax.set_ylabel('Retrieval precision'); ax.set_ylim(-.02,1.02); ax.set_title('Dynamics-Isolated Historical Retrieval During A Recurrence'); ax.legend(); fig.tight_layout(); fig.savefig(figs/'A2_retrieval_precision.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.8)); ax.bar(cmp.feature,cmp.early_A2_precision_at_10); ax.axhline(.9,ls='--'); ax.set_ylabel('Early A2 precision@10'); ax.set_ylim(0,1.02); ax.set_title('Effect of Removing Absolute State from History'); fig.tight_layout(); fig.savefig(figs/'feature_comparison_early_precision.png',dpi=180); plt.close(fig)
    save_json({'code_commit':git_sha(),'seed':seed,'only_protocol_change':'history feature: [state, action, delta] -> [action, delta]','history_input_dim':int(bank_h.shape[-1]),'encoder_training_labels_used':False,'retrieval_trajectory_used_for_encoder_training':False,'encoder_best_epoch':int(rep['model_meta']['best_epoch']),'offline_relative_mse_improvement':rep['offline_relative_mse_improvement'],'offline_AB_precision_at_10':rep['offline_precision_at_10'],'offline_AB_centroid_accuracy':rep['offline_centroid_accuracy'],'old_early_A2_precision_at_10':float(old['early_A2_precision_at_10']),'new_early_A2_precision_at_10':ep,'old_first2k_precision_at_10':float(old['first2k_precision_at_10']),'new_first2k_precision_at_10':fp,'criteria':crit,'early_pass':epass,'first2k_pass':fpass,'dynamics_isolated_retrieval_pilot_pass':passed,'next_if_pass':'fork-at-return end-to-end Recent vs Learned Relevant vs Oracle','next_if_fail':'train behavior-diverse dynamics encoder before end-to-end replay integration'},out/'retrieval_metadata.json')
    save_yaml(cfg,out/'resolved_config.yaml')
    print('\n=== Dynamics-isolated retrieval summary ==='); print(summary.to_string(index=False)); print('\n=== Direct comparison ==='); print(cmp.to_string(index=False)); print('\n=== By A2 step bin ==='); print(binned.to_string(index=False)); print('\nDYNAMICS-ISOLATED RETRIEVAL PILOT PASS:',passed)

if __name__=='__main__': main()
