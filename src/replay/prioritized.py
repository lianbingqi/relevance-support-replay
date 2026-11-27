"""Proportional prioritized experience replay for the synthetic-v2 DDQN project.

PER uses a sum segment tree and samples with replacement, following the
probability distribution P(i) proportional to (|TD_i| + epsilon)**alpha.
Per-sample importance weights are supplied to the training script.

Design for this project: buffer capacity does not wrap during the experiment.
A new transition receives the highest priority seen so far. Existing
transitions are initialized using pre-switch DDQN TD errors at the fork.
No true context labels participate in sampling or priority updates.
"""
from __future__ import annotations

import numpy as np

from src.replay.recent import _to_batch
from src.replay.uniform import UniformReplayBuffer


class PrioritizedReplayBuffer(UniformReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6,
                 priority_epsilon: float = 1e-6, seed: int = 0):
        super().__init__(capacity=capacity, seed=seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be positive")
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.tree_capacity = 1 << (self.capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)
        self.raw_priorities = np.zeros(self.capacity, dtype=np.float64)
        self.max_priority = 1.0
        self.last_sample_indices = None
        self.last_is_weights = None
        self.last_beta = None

    def _set_priority(self, index: int, raw: float):
        index = int(index)
        if not 0 <= index < len(self.buffer):
            raise IndexError(index)
        raw = float(raw)
        if not np.isfinite(raw) or raw <= 0:
            raise ValueError("priorities must be finite and positive")
        self.raw_priorities[index] = raw
        leaf = self.tree_capacity + index
        self.tree[leaf] = raw ** self.alpha
        while leaf > 1:
            leaf //= 2
            self.tree[leaf] = self.tree[2 * leaf] + self.tree[2 * leaf + 1]
        self.max_priority = max(self.max_priority, raw)

    def initialize_priorities(self, abs_td_errors):
        errors = np.asarray(abs_td_errors, dtype=np.float64)
        if errors.shape != (len(self.buffer),):
            raise ValueError("One initial TD error per stored transition is required")
        if not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("Absolute TD errors must be finite and nonnegative")
        raw = errors + self.priority_epsilon
        self.raw_priorities.fill(0.0)
        self.raw_priorities[:len(raw)] = raw
        self.tree.fill(0.0)
        self.tree[self.tree_capacity:self.tree_capacity + len(raw)] = raw ** self.alpha
        for node in range(self.tree_capacity - 1, 0, -1):
            self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1]
        self.max_priority = max(1.0, float(np.max(raw, initial=0.0)))

    def add(self, state, action, reward, next_state, done,
            context_id=None, context_vector=None):
        # No buffer wrap is permitted in this frozen shared-checkpoint protocol.
        if len(self.buffer) >= self.capacity:
            raise RuntimeError("PER buffer wrap is forbidden by experiment protocol")
        super().add(state, action, reward, next_state, done,
                    context_id=context_id, context_vector=context_vector)
        self._set_priority(len(self.buffer) - 1, self.max_priority)

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError("Not enough stored transitions for a minibatch")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        total = float(self.tree[1])
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError("PER sum tree was not initialized")

        # Vectorized binary search on a proportional-priority segment tree.
        residual = self.rng.random(batch_size) * total
        nodes = np.ones(batch_size, dtype=np.int64)
        for _ in range(self.tree_capacity.bit_length() - 1):
            left = nodes * 2
            left_mass = self.tree[left]
            go_right = residual >= left_mass
            residual -= np.where(go_right, left_mass, 0.0)
            nodes = left + go_right.astype(np.int64)
        indices = nodes - self.tree_capacity
        if np.any(indices >= n) or np.any(indices < 0):
            raise AssertionError("Priority sampling selected an empty leaf")
        probs = self.tree[self.tree_capacity + indices] / total
        weights = np.power(n * probs, -float(beta))
        weights /= weights.max()
        assert np.all(weights > 0) and np.all(weights <= 1 + 1e-12)
        self.last_sample_indices = indices.copy()
        self.last_is_weights = weights.astype(np.float32)
        self.last_beta = float(beta)
        return _to_batch([self.buffer[int(i)] for i in indices])

    def update_priorities(self, indices, abs_td_errors):
        indices = np.asarray(indices, dtype=np.int64)
        errors = np.asarray(abs_td_errors, dtype=np.float64)
        if indices.shape != errors.shape or indices.ndim != 1:
            raise ValueError("Index and TD-error vectors must be one-dimensional and aligned")
        if not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("TD errors must be finite and nonnegative")
        # Duplicate indices in replacement-based sampling receive their maximum
        # observed TD error. This is independent of the duplicate order.
        for index in np.unique(indices):
            self._set_priority(int(index), float(np.max(errors[indices == index]))
                               + self.priority_epsilon)

    def priority_summary(self):
        raw = self.raw_priorities[:len(self.buffer)]
        return {
            "priority_min": float(raw.min()) if len(raw) else float("nan"),
            "priority_mean": float(raw.mean()) if len(raw) else float("nan"),
            "priority_max": float(raw.max()) if len(raw) else float("nan"),
            "total_sampling_mass": float(self.tree[1]),
        }
