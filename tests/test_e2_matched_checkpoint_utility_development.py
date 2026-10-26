"""Frozen E2 unit tests. Run before collecting outcome data in Colab."""
import numpy as np
import pytest
import scripts.run_e2_matched_checkpoint_utility_development_v2 as e2


def test_stable_seed_is_order_sensitive_and_deterministic():
    assert e2.stable_seed(1, 'ABA', 200) == e2.stable_seed(1, 'ABA', 200)
    assert e2.stable_seed(1, 'ABA', 200) != e2.stable_seed(1, 'ABD', 200)


def test_paired_batches_share_current_subset_without_old_overlap():
    recent=np.arange(40000, 40200, dtype=np.int64)
    old=np.arange(500, 900, dtype=np.int64)
    r1,m1=e2.paired_indices(recent,old,16,64,.8,101)
    r0,m0=e2.paired_indices(recent,None,16,64,.8,101)
    assert m0 == []
    assert len(r1)==len(m1)==len(r0)==16
    assert all(np.array_equal(a,b) for a,b in zip(r0,r1))
    for recent_batch,mixed in zip(r1,m1):
        assert len(recent_batch)==len(mixed)==64
        assert len(np.unique(mixed))==64
        assert sum(mixed<40000)==51
        assert set(recent_batch[:13]).issubset(set(mixed))
    r2,m2=e2.paired_indices(recent,old,16,64,.8,101)
    assert all(np.array_equal(a,b) for a,b in zip(m1,m2))


def test_candidate_selector_uses_embeddings_not_context_labels():
    bank=np.arange(1000, dtype=np.float32).reshape(500,2)
    query=np.array([1.,2.],dtype=np.float32)
    rows,sim=e2.choose_rows(query,bank,50,67)
    assert set(rows)==set(e2.CANDIDATES)
    assert all(len(rows[m])==50 for m in rows)
    assert len(set(rows['uniform50']))==50
    assert sim['top_similarity'] >= sim['bottom_similarity']


def test_candidate_pool_rejects_any_future_experience():
    idx=np.arange(8*100).reshape(100,8)
    idx[-1,:]=np.arange(40000,40008)
    pool=e2.candidate_pool({'replay_indices':idx}, np.arange(60), 40000)
    assert len(pool)>=64 and np.max(pool)<40000
    with pytest.raises(AssertionError):
        e2.candidate_pool({'replay_indices':idx},[99],40000)


def test_training_data_contract_and_seed_guard():
    assert e2.METHODS == ('current_only','retrieved_top50','uniform50','least_relevant50')
    assert e2.CANDIDATES == ('retrieved_top50','uniform50','least_relevant50')


def test_td_loss_without_optimizer_update():
    from src.agents.ddqn import DDQNAgent
    from src.replay.base_buffer import ReplayBatch
    agent=DDQNAgent(4,5,hidden_sizes=(8,8),device='cpu',seed=7)
    state=np.random.default_rng(19).normal(size=(12,4)).astype(np.float32)
    batch=ReplayBatch(states=state,actions=np.zeros(12,dtype=np.int64),
                      rewards=np.zeros(12,dtype=np.float32),next_states=state.copy(),
                      dones=np.zeros(12,dtype=np.float32),context_ids=[None]*12,context_vectors=[None]*12)
    import torch
    params=[p.detach().clone() for p in agent.online.parameters()]
    a=e2.td_loss_value(agent,batch)
    grad=e2.gradient_vector(agent,batch)
    assert np.isfinite(a) and np.all(np.isfinite(grad))
    assert agent.update_count == 0
    assert all(torch.equal(p,q) for p,q in zip(params,agent.online.parameters()))
