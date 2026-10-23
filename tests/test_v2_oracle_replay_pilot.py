import numpy as np

from src.replay.oracle import OracleRelevantReplayBuffer
from src.replay.recent import RecentReplayBuffer


def add(buf, i, context):
    s = np.asarray([float(i), 0, 0, 0], dtype=np.float32)
    buf.add(
        s, i % 5, -1.0, s + 0.01, False,
        context_id=context,
        context_vector=np.asarray([0.0, 0.0], dtype=np.float32),
    )


def test_recent_uses_only_recent_window():
    buf = RecentReplayBuffer(100, recent_window=10, seed=1)
    for i in range(30):
        add(buf, i, "A")
    for i in range(30, 60):
        add(buf, i, "B")
    batch = buf.sample(8)
    assert all(c == "B" for c in batch.context_ids)
    assert np.all(batch.states[:, 0] >= 50)


def test_oracle_can_retrieve_old_A():
    buf = OracleRelevantReplayBuffer(
        100, recent_window=10, historical_fraction=0.80, seed=2
    )
    for i in range(30):
        add(buf, i, "A")
    for i in range(30, 60):
        add(buf, i, "B")
    buf.set_current_context("A")
    batch = buf.sample(20)
    assert all(c == "A" for c in batch.context_ids)
    assert buf.last_sample_stats["current_context_fraction"] == 1.0
