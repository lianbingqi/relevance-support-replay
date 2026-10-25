"""E1 mechanism-study invariants; run with repo's existing pytest suite."""
from types import SimpleNamespace
import copy
import numpy as np
import torch

from scripts import run_e1_history_conditioning_development_v2 as e1
from src.replay.recent import RecentReplayBuffer
from src.replay.base_buffer import ReplayBatch


def toy_config():
    return {"agent": {"gamma": .99, "learning_rate": .0003,
            "target_update_steps": 1000, "hidden_sizes": [128,128],
            "grad_clip_norm": 10.0}}


def test_frozen_context_requires_complete_episode_local_prefix(monkeypatch):
    fake_encoder = SimpleNamespace(proj=[SimpleNamespace(out_features=16)])
    rep = {"model": SimpleNamespace(encoder=fake_encoder), "history_std": None}
    def fake_encode(model, batch, std):
        assert batch.shape == (1,8,9)
        return np.full((1,16), batch.mean(), dtype=np.float32)
    monkeypatch.setattr(e1, "encode_histories", fake_encode)
    features = [np.full(9,float(i),dtype=np.float32) for i in range(1,9)]
    before = e1.frozen_context(features[:7],rep,8)
    after = e1.frozen_context(features,rep,8)
    assert before.shape == after.shape == (17,)
    assert np.array_equal(before, np.zeros(17))
    assert after[-1] == 1.0
    assert np.allclose(after[:-1], 4.5)
    # Episode-boundary reset is represented by passing an empty prefix.
    assert np.all(e1.frozen_context([], rep, 8) == 0)


def test_latent_sidecar_matches_replay_indices_and_bootstrap_features():
    b = e1.IndexedRecentReplayBuffer(capacity=50, recent_window=50, seed=1)
    store = e1.LatentStore.empty()
    for i in range(20):
        s = np.full(4,float(i), dtype=np.float32)
        b.add(s,1,-1,s+1,False)
        store.add(np.full(17,i,dtype=np.float32),np.full(17,i+0.5,dtype=np.float32))
    batch = b.sample(8)
    store.verify(b)
    a, next_a = store.take(b.last_sample_indices)
    assert np.allclose(a[:,0],batch.states[:,0])
    assert np.allclose(next_a[:,0],batch.states[:,0]+0.5)
    assert len(set(b.last_sample_indices))==8
    fork = copy.deepcopy(store)
    fork.before[0][0] = -123
    assert store.before[0][0] == 0


def test_indexed_recent_matches_existing_recent_rng_and_batch():
    old = RecentReplayBuffer(capacity=100,recent_window=40,seed=7)
    new = e1.IndexedRecentReplayBuffer(capacity=100,recent_window=40,seed=7)
    for i in range(60):
        s = np.array([i,0,0,0],dtype=np.float32)
        old.add(s, 0, -i, s+1, i%19==0)
        new.add(s, 0, -i, s+1, i%19==0)
    o, n = old.sample(32), new.sample(32)
    assert np.array_equal(o.states,n.states)
    assert np.array_equal(o.actions,n.actions)
    assert np.array_equal(o.dones,n.dones)
    assert np.array_equal(new.last_sample_indices,n.states[:,0].astype(int))


def test_ddqn_history_update_uses_different_next_context():
    agent = e1.HistoryConditionedDDQN(toy_config(),seed=3,latent_dim=16)
    n=8
    batch = ReplayBatch(
        states=np.zeros((n,4),dtype=np.float32),
        actions=np.zeros(n,dtype=np.int64),
        rewards=np.ones(n,dtype=np.float32)*-1,
        next_states=np.ones((n,4),dtype=np.float32)*0.1,
        dones=np.zeros(n,dtype=np.float32),
        context_ids=["D"]*n,
        context_vectors=[np.array([0,1],dtype=np.float32)]*n,
    )
    before = np.zeros((n,17),dtype=np.float32)
    after = np.ones((n,17),dtype=np.float32)
    loss = agent.update_context(batch,before,after)
    assert np.isfinite(loss.loss)
    assert agent.update_count==1
    assert agent.online.net[0].in_features == 21
    assert torch.isfinite(next(agent.online.parameters())).all()
