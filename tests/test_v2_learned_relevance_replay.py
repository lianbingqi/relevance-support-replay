import numpy as np

from src.replay.relevance import LearnedRelevantReplayBuffer


def fill(buf, n=20):
    for i in range(n):
        s = np.asarray([float(i)], dtype=np.float32)
        ns = np.asarray([float(i + 1)], dtype=np.float32)
        buf.add(
            s, 0, 0.0, ns, False,
            context_id=("A" if i < 10 else "B"),
        )


def test_without_query_falls_back_to_recent_only():
    buf = LearnedRelevantReplayBuffer(
        capacity=100,
        recent_window=5,
        historical_fraction=0.5,
        retrieval_top_k_windows=1,
        seed=1,
    )
    fill(buf)
    buf.set_historical_bank(
        np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    batch = buf.sample(4)
    sampled_states = batch.states[:, 0].astype(int)
    assert np.all(sampled_states >= 15)
    assert buf.last_sample_stats["historical_fraction"] == 0.0
    assert buf.last_sample_stats["query_available"] is False


def test_query_selects_retrieved_history_plus_recent():
    buf = LearnedRelevantReplayBuffer(
        capacity=100,
        recent_window=5,
        historical_fraction=0.5,
        retrieval_top_k_windows=1,
        seed=2,
    )
    fill(buf)
    buf.set_historical_bank(
        np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    buf.set_query_embedding(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )
    buf.sample(4)
    assert set(buf.last_historical_indices.tolist()).issubset({0, 1})
    assert len(buf.last_historical_indices) == 2
    assert buf.last_sample_stats["historical_fraction"] == 0.5
    assert buf.last_sample_stats["retrieved_bank_windows"] == 1


def test_recent_window_is_not_eligible_as_history():
    buf = LearnedRelevantReplayBuffer(
        capacity=100,
        recent_window=5,
        historical_fraction=0.5,
        retrieval_top_k_windows=1,
        seed=3,
    )
    fill(buf)
    buf.set_historical_bank(
        np.asarray([[0, 1], [17, 18]], dtype=np.int64),
        np.asarray([[0.8, 0.2], [1.0, 0.0]], dtype=np.float32),
    )
    buf.set_query_embedding(
        np.asarray([1.0, 0.0], dtype=np.float32)
    )
    buf.sample(4)
    assert set(buf.last_historical_indices.tolist()).issubset({0, 1})
    assert np.array_equal(
        buf.last_top_bank_rows,
        np.asarray([0], dtype=np.int64),
    )
