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


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (x / denom).astype(np.float32)


class LearnedRelevantReplayBuffer(UniformReplayBuffer):
    """Label-free latent-relevance replay plus a recent fallback."""

    def __init__(
        self,
        capacity: int,
        recent_window: int,
        historical_fraction: float = 0.80,
        retrieval_top_k_windows: int = 50,
        seed: int = 0,
    ):
        super().__init__(capacity=capacity, seed=seed)
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if not 0.0 <= historical_fraction <= 1.0:
            raise ValueError("historical_fraction must be in [0, 1]")
        if retrieval_top_k_windows <= 0:
            raise ValueError("retrieval_top_k_windows must be positive")

        self.recent_window = int(recent_window)
        self.historical_fraction = float(historical_fraction)
        self.retrieval_top_k_windows = int(retrieval_top_k_windows)

        self.bank_transition_indices = None
        self.bank_embeddings = None
        self.bank_embeddings_norm = None
        self.query_embedding = None

        self.last_sample_stats = {}
        self.last_sample_indices = np.asarray([], dtype=np.int64)
        self.last_historical_indices = np.asarray([], dtype=np.int64)
        self.last_top_bank_rows = np.asarray([], dtype=np.int64)

    def set_historical_bank(self, window_transition_indices, embeddings):
        idx = np.asarray(window_transition_indices, dtype=np.int64)
        z = np.asarray(embeddings, dtype=np.float32)

        if idx.ndim != 2:
            raise ValueError("window_transition_indices must be 2D")
        if z.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if len(idx) != len(z):
            raise ValueError("bank index/embedding row count mismatch")
        if len(idx) == 0:
            raise ValueError("historical bank must be non-empty")
        if np.any(idx < 0):
            raise ValueError("historical replay indices must be nonnegative")
        if np.max(idx) >= len(self.buffer):
            raise ValueError("historical bank references unavailable replay entries")

        self.bank_transition_indices = idx.copy()
        self.bank_embeddings = z.copy()
        self.bank_embeddings_norm = _normalize_rows(z)

    def set_query_embedding(self, embedding):
        if self.bank_embeddings is None:
            raise RuntimeError("set_historical_bank() must be called first")

        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.bank_embeddings.shape[1]:
            raise ValueError("query embedding dimension mismatch")
        self.query_embedding = q.copy()

    def clear_query(self):
        self.query_embedding = None

    def _take_unique(self, pool, count, selected, selected_set):
        if count <= 0:
            return

        available = [int(i) for i in pool if int(i) not in selected_set]
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

    def sample(self, batch_size: int) -> ReplayBatch:
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError("Not enough samples in replay buffer")

        recent_start = max(0, n - self.recent_window)
        recent_all = list(range(recent_start, n))

        n_hist_target = int(round(batch_size * self.historical_fraction))
        n_recent_target = batch_size - n_hist_target

        candidate_historical = np.asarray([], dtype=np.int64)
        top_rows = np.asarray([], dtype=np.int64)
        best_similarity = np.nan
        mean_top_similarity = np.nan
        eligible_count = 0

        query_available = (
            self.query_embedding is not None
            and self.bank_transition_indices is not None
            and self.bank_embeddings_norm is not None
        )

        if query_available:
            eligible_mask = (
                np.max(self.bank_transition_indices, axis=1) < recent_start
            )
            eligible_rows = np.flatnonzero(eligible_mask)
            eligible_count = int(len(eligible_rows))

            if eligible_count > 0:
                q = self.query_embedding.astype(np.float32)
                q = q / max(float(np.linalg.norm(q)), 1e-8)

                sim = self.bank_embeddings_norm[eligible_rows] @ q
                k = min(self.retrieval_top_k_windows, len(eligible_rows))

                local_top = np.argpartition(-sim, kth=k - 1)[:k]
                local_top = local_top[np.argsort(-sim[local_top])]
                top_rows = eligible_rows[local_top]

                candidate_historical = np.unique(
                    self.bank_transition_indices[top_rows].reshape(-1)
                )
                candidate_historical = candidate_historical[
                    candidate_historical < recent_start
                ]

                best_similarity = float(sim[local_top[0]])
                mean_top_similarity = float(np.mean(sim[local_top]))

        selected = []
        selected_set = set()
        historical_selected = []

        if len(candidate_historical) > 0:
            before = len(selected)
            self._take_unique(
                candidate_historical.tolist(),
                n_hist_target,
                selected,
                selected_set,
            )
            historical_selected = selected[before:].copy()

        self._take_unique(
            recent_all,
            n_recent_target,
            selected,
            selected_set,
        )

        # If relevance cannot supply the full historical share, prefer Recent
        # rather than forcing unrelated old experience.
        if len(selected) < batch_size:
            self._take_unique(
                recent_all,
                batch_size - len(selected),
                selected,
                selected_set,
            )

        if len(selected) < batch_size:
            self._take_unique(
                list(range(n)),
                batch_size - len(selected),
                selected,
                selected_set,
            )

        if len(selected) != batch_size:
            raise RuntimeError("Could not build a full replay batch")

        selected = self.rng.permutation(
            np.asarray(selected, dtype=np.int64)
        )
        historical_selected = np.asarray(
            historical_selected, dtype=np.int64
        )

        self.last_sample_indices = selected.copy()
        self.last_historical_indices = historical_selected.copy()
        self.last_top_bank_rows = top_rows.copy()

        self.last_sample_stats = {
            "query_available": bool(query_available),
            "eligible_bank_windows": int(eligible_count),
            "retrieved_bank_windows": int(len(top_rows)),
            "candidate_historical_transitions": int(
                len(candidate_historical)
            ),
            "historical_fraction": float(
                len(historical_selected) / batch_size
            ),
            "recent_fraction": float(
                np.mean(selected >= recent_start)
            ),
            "best_similarity": best_similarity,
            "mean_top_similarity": mean_top_similarity,
        }

        return _to_batch([self.buffer[int(i)] for i in selected])
