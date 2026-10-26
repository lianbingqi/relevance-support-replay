"""E1d v2.16 structural checks, not substitutes for a complete Colab run."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import run_e1d_onpolicy_q_representation_ablation_v2 as d


def test_zero_latent_preserves_validity_and_never_mutates_source():
    original=np.arange(17,dtype=np.float32);original[-1]=1
    zero=d.latent_transform(original,"zero")
    full=d.latent_transform(original,"full")
    assert np.array_equal(zero[:-1],np.zeros(16,dtype=np.float32))
    assert zero[-1]==1 and np.array_equal(full,original)
    zero[0]=90;full[1]=90
    assert original[0]==0 and original[1]==1
    assert np.array_equal(d.latent_transform(np.zeros(17,dtype=np.float32),"zero"),np.zeros(17))
    with pytest.raises(ValueError):d.latent_transform(original,"unregistered")


def test_zero_latent_wrapper_uses_exact_original_mask(monkeypatch):
    value=np.arange(17,dtype=np.float32);value[-1]=1
    monkeypatch.setattr(d,"ORIGINAL_CONTEXT",lambda history,rep,L:value)
    f=d.patched_context("zero")
    z=f([],None,8)
    assert np.count_nonzero(z)==1 and z[-1]==1


def test_e1b_eval_seed_matches_existing_protocol():
    assert d._eval_seed("A",2,4)==70_020_004
    assert d._eval_seed("B",2,4)==170_050_004


def test_fixed_q_zero_latent_control_has_no_context_signal():
    net=torch.nn.Linear(21,5)
    with torch.no_grad():
        net.weight.zero_();net.bias.zero_();net.bias[0]=1
    agent=SimpleNamespace(online=net)
    probes={"A":{"state":np.ones((6,4),dtype=np.float32),
                 "latent":np.column_stack([np.ones((6,16),dtype=np.float32)*12,
                                            np.ones((6,1),dtype=np.float32)])}}
    out=d.eval_fixed_q(agent,"history_conditioned","hc_zero_latent",probes,1,"B",20000)
    assert len(out)==1
    assert out[0]["probe_noop_greedy_fraction"]==1
    assert out[0]["probe_noop_margin_mean"]==1


def test_canonical_protocol_guards():
    cfg={"e1":{"development_seeds":[1,2,3],"reserved_confirmation_seeds":[4,5,6]},
         "e1d":{"development_seeds":[1,2,3],"reserved_confirmation_seeds":[4,5,6],
                "arms":list(d.ARMS),"checkpoints_B":list(d.CHECKPOINTS_B)},
         "e1c":{"fixed_probe_seed_base":931000,"fixed_probe_steps_per_episode":64,"fixed_probe_episodes_per_context":6},
         "isolation":{"A1_steps":20000,"B_steps":20000,"eval_episodes":5},
         "representation":{"history_length":8},"experiment":{"train_frequency":4}}
    d.validate_frozen_protocol(cfg)
    cfg["e1d"]["reserved_confirmation_seeds"]=[4,5]
    with pytest.raises(RuntimeError):d.validate_frozen_protocol(cfg)


def test_read_only_snapshot_uses_deepcopy(monkeypatch):
    agent=SimpleNamespace(online=object())
    replay=object()
    monkeypatch.setattr(d.e1c,"_rng_fingerprint",lambda *args:"identical")
    monkeypatch.setattr(d.e1c,"_agent_parameter_fingerprint",lambda *args:"identical")
    seen=[]
    def fake_onpolicy(shadow,*args,**kwargs):
        assert shadow is not agent
        seen.append(args[5]) # context
        return {"ok":True},[{"episode":1}]
    monkeypatch.setattr(d,"eval_onpolicy_q",fake_onpolicy)
    monkeypatch.setattr(d,"eval_fixed_q",lambda *args,**kwargs:[])
    u,v=len(d.PROBE_OUTPUTS),len(d.EPISODES)
    d.snapshot(agent,replay,None,{},None,1,"state_only","state_only","B",0,8)
    assert seen==["A","B"]
    assert len(d.PROBE_OUTPUTS)==u+2 and len(d.EPISODES)==v+2
    del d.PROBE_OUTPUTS[u:];del d.EPISODES[v:]


def test_original_update_only_once_and_predeclared_checkpoint(monkeypatch):
    before=d.ACTIVE
    calls=[]
    def fake(*args,**kwargs):calls.append(1);return "updated"
    def log(*args):calls.append("snapshot")
    monkeypatch.setattr(d,"ORIGINAL_UPDATE",fake)
    monkeypatch.setattr(d,"snapshot",log)
    d.ACTIVE={"update_index":0,"first_update_global":20000,"freq":4,
              "start_global":20000,"snapshot_steps":{1},"rep":None,"cfg":{},
              "probes":None,"seed":1,"arm":"state_only","phase":"B","L":8}
    try:
        assert d.audit_update(None,None,None,"state_only",64)=="updated"
        assert calls==[1,"snapshot"]
        assert d.ACTIVE["update_index"]==1
    finally:
        d.ACTIVE=before
