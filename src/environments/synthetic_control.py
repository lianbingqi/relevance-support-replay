from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces

@dataclass(frozen=True)
class ContextParams:
    rho: float
    beta: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "ContextParams":
        return cls(rho=float(values["rho"]), beta=float(values["beta"]))

class HiddenContextControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        contexts: Mapping[str, Mapping[str, float]],
        context_name: str = "A",
        max_episode_steps: int = 200,
        noise_std: float = 0.01,
        init_x_range: float = 1.25,
        init_v_range: float = 0.10,
        x_limit: float = 4.0,
        v_limit: float = 3.0,
    ):
        super().__init__()
        if not contexts:
            raise ValueError("At least one context must be provided.")
        self.contexts: Dict[str, ContextParams] = {
            name: ContextParams.from_mapping(params)
            for name, params in contexts.items()
        }
        if context_name not in self.contexts:
            raise KeyError(f"Unknown context: {context_name}")
        self.context_name = context_name
        self.max_episode_steps = int(max_episode_steps)
        self.noise_std = float(noise_std)
        self.init_x_range = float(init_x_range)
        self.init_v_range = float(init_v_range)
        self.x_limit = float(x_limit)
        self.v_limit = float(v_limit)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    @property
    def context(self) -> ContextParams:
        return self.contexts[self.context_name]

    def set_context(self, context_name: str) -> None:
        if context_name not in self.contexts:
            raise KeyError(f"Unknown context: {context_name}")
        self.context_name = context_name

    def context_vector(self) -> np.ndarray:
        c = self.context
        return np.asarray([c.rho, c.beta], dtype=np.float32)

    def _info(self) -> dict:
        return {
            "true_context": self.context_name,
            "context_vector": self.context_vector().copy(),
        }

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        x0 = self.np_random.uniform(-self.init_x_range, self.init_x_range)
        v0 = self.np_random.uniform(-self.init_v_range, self.init_v_range)
        self.state = np.asarray([x0, v0], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        u = float(action - 1)
        x, v = map(float, self.state)
        c = self.context
        noise = float(self.np_random.normal(0.0, self.noise_std))
        v_next = c.rho * v + c.beta * u + noise
        x_next = x + v_next

        self.state = np.asarray([x_next, v_next], dtype=np.float32)
        self.steps += 1

        reward = -(x_next**2 + 0.10 * v_next**2 + 0.01 * u**2)
        terminated = abs(x_next) > self.x_limit or abs(v_next) > self.v_limit
        truncated = self.steps >= self.max_episode_steps

        return self.state.copy(), float(reward), bool(terminated), bool(truncated), self._info()
