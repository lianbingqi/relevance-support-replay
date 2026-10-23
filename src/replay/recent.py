from __future__ import annotations

import numpy as np

from src.replay.base_buffer import ReplayBatch
from src.replay.uniform import UniformReplayBuffer


def _to_batch(items):
    return ReplayBatch(
        states=np.stack([t.state for t in items]).astype(np.float32),
        actions=np.asarray([t.action for t in items], dtype=np.int64),
        rewards=np.asarray([t.reward for t in items], dtype=np.float32),
        next_states=np.stack([t.next_state for t in items]).astype(np.float32),
        dones=np.asarray([t.done for t in items], dtype=np.float32),
        context_ids=[t.context_id for t in items],
        context_vectors=[t.context_vector for t in items],
    )


class RecentReplayBuffer(UniformReplayBuffer):
    def __init__(self, capacity: int, recent_window: int, seed: int = 0):
        super().__init__(capacity=capacity, seed=seed)
        self.recent_window = int(recent_window)

    def sample(self, batch_size: int) -> ReplayBatch:
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError("Not enough samples in replay buffer")

        start = max(0, n - self.recent_window)
        candidates = np.arange(start, n, dtype=np.int64)

        idx = self.rng.choice(
            candidates,
            size=batch_size,
            replace=False,
        )

        return _to_batch([self.buffer[int(i)] for i in idx])
