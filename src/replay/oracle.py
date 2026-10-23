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
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if not 0.0 <= historical_fraction <= 1.0:
            raise ValueError("historical_fraction must be in [0, 1]")
        self.recent_window = int(recent_window)
        self.historical_fraction = float(historical_fraction)
        self.current_context = None
        self.last_sample_stats = {}

    def set_current_context(self, context_id: str) -> None:
        self.current_context = context_id

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.current_context is None:
            raise RuntimeError("set_current_context() must be called first")

        n = len(self.buffer)
        if n < batch_size:
            raise ValueError("Not enough samples in replay buffer")

        recent_start = max(0, n - self.recent_window)

        historical_relevant = [
            i for i in range(recent_start)
            if self.buffer[i].context_id == self.current_context
        ]
        recent_all = list(range(recent_start, n))

        n_hist_target = int(round(batch_size * self.historical_fraction))
        n_recent_target = batch_size - n_hist_target

        selected = []
        selected_set = set()

        def take(pool, count):
            if count <= 0:
                return
            available = [i for i in pool if i not in selected_set]
            if not available:
                return
            k = min(int(count), len(available))
            chosen = self.rng.choice(
                np.asarray(available, dtype=np.int64),
                size=k,
                replace=False,
            ).tolist()
            for i in chosen:
                selected.append(int(i))
                selected_set.add(int(i))

        take(historical_relevant, n_hist_target)
        take(recent_all, n_recent_target)

        if len(selected) < batch_size:
            take(recent_all, batch_size - len(selected))

        if len(selected) < batch_size:
            take(list(range(n)), batch_size - len(selected))

        if len(selected) != batch_size:
            raise RuntimeError("Oracle sampler could not build a full batch")

        selected = self.rng.permutation(
            np.asarray(selected, dtype=np.int64)
        ).tolist()

        self.last_sample_stats = {
            "historical_relevant_fraction": float(
                sum(
                    i < recent_start
                    and self.buffer[i].context_id == self.current_context
                    for i in selected
                ) / batch_size
            ),
            "recent_fraction": float(
                sum(i >= recent_start for i in selected) / batch_size
            ),
            "current_context_fraction": float(
                sum(
                    self.buffer[i].context_id == self.current_context
                    for i in selected
                ) / batch_size
            ),
        }

        return _to_batch([self.buffer[int(i)] for i in selected])
