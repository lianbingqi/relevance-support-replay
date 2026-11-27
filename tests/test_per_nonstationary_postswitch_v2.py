"""Fast, local correctness checks for PER. No full training run required."""
import copy
import numpy as np
import torch

from src.agents.ddqn import DDQNAgent
from src.replay.prioritized import PrioritizedReplayBuffer
from scripts.run_per_nonstationary_postswitch_v2 import weighted_per_ddqn_update


def _make_buffer(capacity=256, n=128, seed=7):
    b = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6, seed=seed)
    for j in range(n):
        s = np.asarray([j / 100.0, 0.1, 0.2, -0.05], dtype=np.float32)
        ns = s + np.asarray([0.01, -0.02, 0.002, 0.001], dtype=np.float32)
        b.add(s, j % 5, -float(j) / 100, ns, False,
              context_id="A" if j < n // 2 else "B")
    return b


def test_initial_priorities_and_high_priority_sampling():
    b = _make_buffer()
    arr = np.full(len(b), 0.1)
    arr[0] = 1000.0
    b.initialize_priorities(arr)
    hits = 0
    for _ in range(40):
        batch = b.sample(64, beta=0.4)
        hits += int(np.count_nonzero(b.last_sample_indices == 0))
        assert len(batch.actions) == 64
        assert (b.last_is_weights > 0).all()
        assert (b.last_is_weights <= 1).all()
    assert hits > 40  # A top-priority item must be sampled substantially more.
    assert np.isclose(b.tree[1], sum(b.raw_priorities[:len(b)] ** b.alpha))


def test_priority_refresh_and_new_transition_priority():
    b = _make_buffer()
    b.initialize_priorities(np.full(len(b), 0.25))
    b.update_priorities(np.asarray([3, 3, 4]), np.asarray([1., 4., 2.]))
    assert np.isclose(b.raw_priorities[3], 4 + b.priority_epsilon)
    assert np.isclose(b.raw_priorities[4], 2 + b.priority_epsilon)
    old = b.max_priority
    s = np.zeros(4, dtype=np.float32)
    b.add(s, 0, -1, s.copy(), False)
    assert np.isclose(b.raw_priorities[len(b)-1], old)


def test_weighted_update_matches_unweighted_with_unit_weights():
    torch.set_num_threads(1)
    b = _make_buffer(n=128)
    batch = b.sample(64, beta=0.0)
    a = DDQNAgent(observation_dim=4, action_dim=5, seed=11, device="cpu")
    c = copy.deepcopy(a)
    before = {k: v.detach().clone() for k, v in a.online.state_dict().items()}
    a.update(batch)
    td = weighted_per_ddqn_update(c, batch, np.ones(64, dtype=np.float32))
    assert td.shape == (64,)
    assert np.isfinite(td).all()
    assert a.update_count == c.update_count == 1
    for k, val in a.online.state_dict().items():
        assert torch.allclose(val, c.online.state_dict()[k], atol=1e-6, rtol=0), k
    assert any(not torch.equal(before[k], a.online.state_dict()[k]) for k in before)


def test_sampling_reproducibility_and_is_weight_direction():
    a = _make_buffer(seed=1)
    b = _make_buffer(seed=1)
    priorities = np.linspace(0.01, 20.0, len(a))
    a.initialize_priorities(priorities)
    b.initialize_priorities(priorities)
    a.sample(64, beta=1.0)
    b.sample(64, beta=1.0)
    assert np.array_equal(a.last_sample_indices, b.last_sample_indices)
    assert np.array_equal(a.last_is_weights, b.last_is_weights)
    raw = a.raw_priorities[a.last_sample_indices]
    if len(np.unique(raw)) > 1:
        i = np.argmin(raw)
        j = np.argmax(raw)
        assert a.last_is_weights[i] >= a.last_is_weights[j]
