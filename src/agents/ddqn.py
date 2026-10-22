from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import nn
from src.agents.networks import QNetwork
from src.replay.base_buffer import ReplayBatch

@dataclass
class UpdateStats:
    loss: float
    q_mean: float
    target_mean: float
    td_abs_mean: float

class DDQNAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        target_update_steps: int = 1000,
        hidden_sizes: Sequence[int] = (128, 128),
        grad_clip_norm: float = 10.0,
        seed: int = 0,
        device: str | None = None,
    ):
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.target_update_steps = int(target_update_steps)
        self.grad_clip_norm = float(grad_clip_norm)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.rng = np.random.default_rng(seed)

        self.online = QNetwork(observation_dim, action_dim, hidden_sizes).to(self.device)
        self.target = QNetwork(observation_dim, action_dim, hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=float(learning_rate)
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.update_count = 0

    @torch.no_grad()
    def act(self, state, epsilon: float = 0.0) -> int:
        if self.rng.random() < float(epsilon):
            return int(self.rng.integers(self.action_dim))
        x = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            device=self.device
        ).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    def update(self, batch: ReplayBatch) -> UpdateStats:
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1)
            next_q = self.target(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        td = target - q.detach()
        return UpdateStats(
            loss=float(loss.item()),
            q_mean=float(q.detach().mean().item()),
            target_mean=float(target.detach().mean().item()),
            td_abs_mean=float(td.abs().mean().item()),
        )
