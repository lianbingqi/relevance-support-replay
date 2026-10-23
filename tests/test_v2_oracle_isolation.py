import numpy as np
from src.replay.oracle import OracleRelevantReplayBuffer


def add(buf, i, context):
    s = np.asarray([float(i), 0.0, 0.0, 0.0], dtype=np.float32)
    buf.add(
        s, i % 5, -1.0, s + 0.01, False,
        context_id=context,
        context_vector=np.asarray([0.0, 0.0], dtype=np.float32),
    )


def test_oracle_uses_historical_A_plus_generic_recent():
    buf = OracleRelevantReplayBuffer(
        capacity=200,
        recent_window=20,
        historical_fraction=0.80,
        seed=7,
    )

    for i in range(60):
        add(buf, i, "A")
    for i in range(60, 120):
        add(buf, i, "B")

    buf.set_current_context("A")
    batch = buf.sample(20)

    assert sum(c == "A" for c in batch.context_ids) == 16
    assert sum(c == "B" for c in batch.context_ids) == 4
    assert np.isclose(
        buf.last_sample_stats["historical_relevant_fraction"],
        0.80,
    )
    assert np.isclose(
        buf.last_sample_stats["recent_fraction"],
        0.20,
    )
