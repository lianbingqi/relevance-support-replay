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


class OracleRelevantReplayBuffer(UniformReplayBuffer):
    def __init__(
        self,
        capacity: int,
        recent_window: int,
        historical_fraction: float = 0.80,
        seed: int = 0,
    ):
        super().__init__(capacity=capacity, seed=seed)
        self.recent_window = int(recent_window)
        self.historical_fraction = float(historical_fraction)
        self.current_context = None
        self.last_sample_stats = {}

    def set_current_context(self, context_id: str):
        self.current_context = context_id

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.current_context is None:
            raise RuntimeError("set_current_context() must be called first")

        n = len(self.buffer)
        if n < batch_size:
            raise ValueError("Not enough samples in replay buffer")

        recent_start = max(0, n - self.recent_window)

        hist_relevant = [
            i for i in range(recent_start)
            if self.buffer[i].context_id == self.current_context
        ]
        recent_relevant = [
            i for i in range(recent_start, n)
            if self.buffer[i].context_id == self.current_context
        ]

        hist_target = int(round(batch_size * self.historical_fraction))
        recent_target = batch_size - hist_target

        selected = []
        selected_set = set()

        def take(pool, count):
            available = [i for i in pool if i not in selected_set]
            if count <= 0 or not available:
                return
            k = min(count, len(available))
            chosen = self.rng.choice(
                np.asarray(available, dtype=np.int64),
                size=k,
                replace=False,
            ).tolist()
            for i in chosen:
                selected.append(int(i))
                selected_set.add(int(i))

        take(hist_relevant, hist_target)
        take(recent_relevant, recent_target)
        take(hist_relevant + recent_relevant, batch_size - len(selected))
        take(list(range(recent_start, n)), batch_size - len(selected))
        take(list(range(n)), batch_size - len(selected))

        if len(selected) != batch_size:
            raise RuntimeError("Could not assemble a full Oracle batch")

        selected = self.rng.permutation(
            np.asarray(selected, dtype=np.int64)
        ).tolist()

        current_count = sum(
            self.buffer[i].context_id == self.current_context
            for i in selected
        )
        hist_count = sum(
            i < recent_start
            and self.buffer[i].context_id == self.current_context
            for i in selected
        )

        self.last_sample_stats = {
            "current_context_fraction": current_count / batch_size,
            "historical_relevant_fraction": hist_count / batch_size,
        }

        return _to_batch([self.buffer[int(i)] for i in selected])
