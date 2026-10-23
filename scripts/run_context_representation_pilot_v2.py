from __future__ import annotations
import argparse, copy, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scripts.train_stationary_v2 import make_env
from src.agents.context_encoder import HistoryDynamicsPredictor,NoHistoryDynamicsPredictor
from src.representation.history import transition_feature,current_feature
from src.utils.config import load_config
from src.utils.logger import ensure_dir,save_json,save_yaml
from src.utils.seed import set_global_seed

def git_sha(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
@dataclass
class Std:
    mean:np.ndarray; std:np.ndarray
    @classmethod
    def fit(cls,x):
        m=x.mean(0).astype(np.float32); s=x.std(0).astype(np.float32); s=np.where(s<1e-6,1.,s).astype(np.float32); return cls(m,s)
    def t(self,x): return ((x-self.mean)/self.std).astype(np.float32)
    def inv(self,x): return (x*self.std+self.mean).astype(np.float32)

def collect(ctx,n,cfg,offset=0):
    rep=cfg['representation']; steps=int(rep['collection_steps_per_episode']); seed=int(rep['seed']); out=[]
    for ep in range(n):
        eid=offset+ep; env=make_env(cfg,ctx); state,_=env.reset(seed=100_000_000+seed*1_000_000+eid); arng=np.random.default_rng(110_000_000+seed*1_000_000+eid)
        states=[np.asarray(state,dtype=np.float32)]; actions=[]
        for _ in range(steps):
            a=int(arng.integers(env.action_space.n)); ns,_,term,trunc,_=env.step(a); actions.append(a); states.append(np.asarray(ns,dtype=np.float32)); state=ns
            if term or trunc: break
        env.close(); out.append({'episode_id':eid,'context':ctx,'states':np.asarray(states,dtype=np.float32),'actions':np.asarray(actions,dtype=np.int64)})
    return out

def windows(episodes,cfg):
    rep=cfg['representation']; L=int(rep['history_length']); stride=int(rep['sample_stride']); H=[]; C=[]; Y=[]; labels=[]; ids=[]
    for e in episodes:
        S=e['states']; A=e['actions']; T=len(A)
        if T<=L: continue
        feat=[transition_feature(S[t],int(A[t]),S[t+1],5) for t in range(T)]
        for t in range(L,T,stride):
            H.append(np.stack(feat[t-L:t])); C.append(current_feature(S[t],int(A[t]),5)); Y.append((S[t+1]-S[t]).astype(np.float32)); labels.append(e['context']); ids.append(int(e['episode_id']))
    return {'history':np.stack(H).astype(np.float32),'current':np.stack(C).astype(np.float32),'target':np.stack(Y).astype(np.float32),'label':np.asarray(labels),'episode_id':np.asarray(ids,dtype=np.int64)}

def cat(ds): return {k:np.concatenate([d[k] for d in ds],axis=0) for k in ds[0]}
def sub(d,m): return {k:v[m] for k,v in d.items()}
def loader(d,batch,shuffle): return DataLoader(TensorDataset(torch.from_numpy(d['history']),torch.from_numpy(d['current']),torch.from_numpy(d['target'])),batch_size=int(batch),shuffle=shuffle,num_workers=0)

def fit_models(train,val,cfg):
    r=cfg['representation']; hist=HistoryDynamicsPredictor(train['history'].shape[-1],train['current'].shape[-1],train['target'].shape[-1],int(r['gru_hidden_dim']),int(r['latent_dim']),int(r['predictor_hidden_dim']))
    base=NoHistoryDynamicsPredictor(train['current'].shape[-1],train['target'].shape[-1],int(r['predictor_hidden_dim']))
    tr=loader(train,r['batch_size'],True); va=loader(val,r['batch_size'],False); lossfn=nn.MSELoss()
    def fit(model,use_hist):
        opt=torch.optim.Adam(model.parameters(),lr=float(r['learning_rate']),weight_decay=float(r['weight_decay'])); best=None; bestv=float('inf'); bestep=-1; bad=0; rows=[]
        for ep in range(1,int(r['max_epochs'])+1):
            model.train(); tl=[]
            for h,c,y in tr:
                pred=model(h,c)[0] if use_hist else model(c); loss=lossfn(pred,y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl.append(float(loss.item()))
            model.eval(); vl=[]
            with torch.no_grad():
                for h,c,y in va:
                    pred=model(h,c)[0] if use_hist else model(c); vl.append(float(lossfn(pred,y).item()))
            tv=float(np.mean(tl)); vv=float(np.mean(vl)); rows.append({'epoch':ep,'train_loss':tv,'val_loss':vv})
            if vv<bestv-1e-6: bestv=vv; bestep=ep; best=copy.deepcopy(model.state_dict()); bad=0
            else: bad+=1
            if bad>=int(r['patience']): break
        model.load_state_dict(best); return model,pd.DataFrame(rows),{'best_epoch':bestep,'best_val_loss':bestv}
    hist,hcurve,hmeta=fit(hist,True); base,bcurve,bmeta=fit(base,False); return hist,base,hcurve,bcurve,hmeta,bmeta

def pred_hist(model,d,batch):
    ps=[]; zs=[]; model.eval()
    with torch.no_grad():
        for h,c,_ in loader(d,batch,False):
            p,z=model(h,c); ps.append(p.numpy()); zs.append(z.numpy())
    return np.concatenate(ps),np.concatenate(zs)
def pred_base(model,d,batch):
    ps=[]; model.eval()
    with torch.no_grad():
        for _,c,_ in loader(d,batch,False): ps.append(model(c).numpy())
    return np.concatenate(ps)
def l2(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)

def retrieval(bank_z,bank_y,q_z,q_y,k,max_bank,seed):
    rng=np.random.default_rng(seed)
    if len(bank_z)>max_bank:
        idx=rng.choice(len(bank_z),size=max_bank,replace=False); bank_z=bank_z[idx]; bank_y=bank_y[idx]
    bn=l2(bank_z); qn=l2(q_z); sim=qn@bn.T; top=np.argpartition(-sim,kth=k-1,axis=1)[:,:k]; precision=float(np.mean(bank_y[top]==q_y[:,None]))
    cents=[]
    for lab in ['A','B']:
        c=bank_z[bank_y==lab].mean(0); cents.append(c/max(np.linalg.norm(c),1e-8))
    cents=np.stack(cents); cs=qn@cents.T; pred=np.asarray(['A','B'])[np.argmax(cs,axis=1)]; acc=float(np.mean(pred==q_y)); return precision,acc,cents,np.max(cs,axis=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/synthetic_v2/context_representation_pilot.yaml'); args=ap.parse_args()
    cfg=load_config(ROOT/'configs/base.yaml',ROOT/args.config); r=cfg['representation']; seed=int(r['seed']); torch.set_num_threads(1); set_global_seed(seed)
    out=ensure_dir(ROOT/'results/synthetic_v2/context_representation_pilot'); figs=ensure_dir(ROOT/'figures/synthetic_v2/context_representation_pilot')
    nt=int(r['train_episodes_per_context']); nv=int(r['val_episodes_per_context']); nte=int(r['test_episodes_per_context']); total=nt+nv+nte
    comb=cat([windows(collect('A',total,cfg),cfg),windows(collect('B',total,cfg),cfg)])
    ids=comb['episode_id']; train=sub(comb,ids<nt); val=sub(comb,(ids>=nt)&(ids<nt+nv)); test=sub(comb,ids>=nt+nv)
    hs=Std.fit(train['history'].reshape(-1,train['history'].shape[-1])); cs=Std.fit(train['current']); ys=Std.fit(train['target'])
    def standard(d):
        q={k:v.copy() for k,v in d.items()}; q['history']=hs.t(d['history']); q['current']=cs.t(d['current']); q['target']=ys.t(d['target']); return q
    trs,vas,tes=standard(train),standard(val),standard(test)
    hist,base,hcurve,bcurve,hmeta,bmeta=fit_models(trs,vas,cfg)
    hp,ztest=pred_hist(hist,tes,r['batch_size']); bp=pred_base(base,tes,r['batch_size']); hp=ys.inv(hp); bp=ys.inv(bp); y=test['target']
    hmse=float(np.mean((hp-y)**2)); bmse=float(np.mean((bp-y)**2)); rel=float(1-hmse/bmse)
    _,ztrain=pred_hist(hist,trs,r['batch_size']); prec,acc,cents,ab_sim=retrieval(ztrain,train['label'],ztest,test['label'],int(r['retrieval_k']),int(r['max_retrieval_bank']),seed)
    D=windows(collect('D',int(r['novel_D_episodes']),cfg,offset=50_000),cfg); Ds=standard(D); _,zD=pred_hist(hist,Ds,r['batch_size']); dsim=np.max(l2(zD)@cents.T,axis=1)
    crit=cfg['criteria']; mp=rel>=float(crit['min_relative_mse_improvement']); cp=acc>=float(crit['min_centroid_accuracy']); rp=prec>=float(crit['min_retrieval_precision_at_k']); passed=bool(mp and cp and rp)
    summary=pd.DataFrame([{'history_test_mse':hmse,'no_history_test_mse':bmse,'relative_mse_improvement':rel,'centroid_accuracy':acc,'retrieval_precision_at_10':prec,'AB_nearest_centroid_similarity_mean':float(ab_sim.mean()),'D_nearest_AB_centroid_similarity_mean':float(dsim.mean()),'mse_pass':mp,'centroid_pass':cp,'retrieval_pass':rp,'pilot_pass':passed}]); summary.to_csv(out/'representation_summary.csv',index=False)
    pd.DataFrame([{'context':'A','n_windows':int(np.sum(test['label']=='A')),'nearest_centroid_similarity_mean':float(ab_sim[test['label']=='A'].mean())},{'context':'B','n_windows':int(np.sum(test['label']=='B')),'nearest_centroid_similarity_mean':float(ab_sim[test['label']=='B'].mean())},{'context':'D_unseen','n_windows':len(zD),'nearest_centroid_similarity_mean':float(dsim.mean())}]).to_csv(out/'context_geometry.csv',index=False)
    hcurve.to_csv(out/'history_training_curve.csv',index=False); bcurve.to_csv(out/'baseline_training_curve.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(hcurve.epoch,hcurve.val_loss,label='history GRU'); ax.plot(bcurve.epoch,bcurve.val_loss,label='no-history baseline'); ax.set_xlabel('Epoch'); ax.set_ylabel('Validation MSE'); ax.set_title('Dynamics Prediction Validation Loss'); ax.legend(); fig.tight_layout(); fig.savefig(figs/'prediction_validation_loss.png',dpi=180); plt.close(fig)
    pca=PCA(n_components=2,random_state=seed); coords=pca.fit_transform(np.concatenate([ztest,zD])); labs=np.concatenate([test['label'],np.asarray(['D']*len(zD))]); fig,ax=plt.subplots(figsize=(7,6))
    for lab in ['A','B','D']:
        m=labs==lab; ax.scatter(coords[m,0],coords[m,1],s=10,alpha=.45,label=lab)
    ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2'); ax.set_title('History-Latent Geometry (A/B trained, D unseen)'); ax.legend(); fig.tight_layout(); fig.savefig(figs/'latent_pca_AB_D.png',dpi=180); plt.close(fig)
    save_json({'code_commit':git_sha(),'seed':seed,'train_contexts':['A','B'],'unseen_diagnostic_context':'D','label_used_in_training':False,'history_length':int(r['history_length']),'history_model_best_epoch':int(hmeta['best_epoch']),'baseline_best_epoch':int(bmeta['best_epoch']),'criteria':crit,'relative_mse_improvement':rel,'centroid_accuracy':acc,'retrieval_precision_at_10':prec,'mse_pass':mp,'centroid_pass':cp,'retrieval_pass':rp,'representation_pilot_pass':passed,'next_if_pass':'learned relevance retrieval pilot using latent cosine similarity','next_if_fail':'diagnose representation objective/history before replay integration'},out/'representation_metadata.json')
    save_yaml(cfg,out/'resolved_config.yaml'); print('\n=== Representation summary ==='); print(summary.to_string(index=False)); print('\nREPRESENTATION PILOT PASS:',passed)
if __name__=='__main__': main()
