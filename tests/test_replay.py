import numpy as np
from src.replay.uniform import UniformReplayBuffer

def test_uniform_replay_shapes_and_metadata():
    buf = UniformReplayBuffer(capacity=100, seed=1)
    for i in range(20):
        s = np.asarray([i, -i], dtype=np.float32)
        buf.add(
            s, i % 3, -float(i), s + 0.1, False,
            context_id="A",
            context_vector=np.asarray([0.85, 0.18], dtype=np.float32),
        )
    batch = buf.sample(8)
    assert batch.states.shape == (8, 2)
    assert batch.actions.shape == (8,)
    assert batch.rewards.shape == (8,)
    assert batch.next_states.shape == (8, 2)
    assert batch.dones.shape == (8,)
    assert all(c == "A" for c in batch.context_ids)
