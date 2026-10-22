from __future__ import annotations
from collections import deque
import numpy as np
from src.replay.base_buffer import BaseReplayBuffer, ReplayBatch, Transition

class UniformReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done,
            context_id=None, context_vector=None):
        self.buffer.append(Transition(
            state=np.asarray(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32).copy(),
            done=bool(done),
            context_id=context_id,
            context_vector=None if context_vector is None
            else np.asarray(context_vector, dtype=np.float32).copy(),
        ))

    def sample(self, batch_size: int) -> ReplayBatch:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        idx = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        items = [self.buffer[int(i)] for i in idx]
        return ReplayBatch(
            states=np.stack([t.state for t in items]).astype(np.float32),
            actions=np.asarray([t.action for t in items], dtype=np.int64),
            rewards=np.asarray([t.reward for t in items], dtype=np.float32),
            next_states=np.stack([t.next_state for t in items]).astype(np.float32),
            dones=np.asarray([t.done for t in items], dtype=np.float32),
            context_ids=[t.context_id for t in items],
            context_vectors=[t.context_vector for t in items],
        )

    def __len__(self):
        return len(self.buffer)
