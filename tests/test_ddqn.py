import numpy as np
from src.agents.ddqn import DDQNAgent
from src.replay.uniform import UniformReplayBuffer

def test_ddqn_update_runs():
    buf = UniformReplayBuffer(capacity=100, seed=0)
    for i in range(64):
        s = np.asarray([0.1*i, -0.01*i], dtype=np.float32)
        buf.add(
            s, i % 3, -float(np.square(s).sum()), s * 0.9, False,
            context_id="A",
            context_vector=np.asarray([0.85, 0.18], dtype=np.float32),
        )
    agent = DDQNAgent(2, 3, seed=0, device="cpu")
    stats = agent.update(buf.sample(32))
    assert np.isfinite(stats.loss)
    assert stats.loss >= 0.0
