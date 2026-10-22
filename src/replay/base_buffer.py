from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    context_id: Optional[str] = None
    context_vector: Optional[np.ndarray] = None

@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    context_ids: Sequence[Optional[str]]
    context_vectors: Sequence[Optional[np.ndarray]]

class BaseReplayBuffer(ABC):
    @abstractmethod
    def add(self, state, action, reward, next_state, done,
            context_id=None, context_vector=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> ReplayBatch:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
