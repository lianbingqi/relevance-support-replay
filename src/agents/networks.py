from __future__ import annotations
from typing import Sequence
import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = int(observation_dim)
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, int(h)), nn.ReLU()])
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
