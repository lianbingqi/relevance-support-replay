import numpy as np
from src.environments.synthetic_control import HiddenContextControlEnv

def make_env():
    return HiddenContextControlEnv(
        contexts={
            "A": {"rho": 0.85, "beta": 0.18},
            "B": {"rho": 0.65, "beta": 0.25},
        },
        context_name="A",
        max_episode_steps=10,
        noise_std=0.0,
    )

def test_observation_does_not_contain_context():
    env = make_env()
    obs, info = env.reset(seed=123)
    assert obs.shape == (2,)
    assert info["true_context"] == "A"
    assert info["context_vector"].shape == (2,)

def test_reset_is_seed_reproducible():
    env = make_env()
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    np.testing.assert_allclose(a, b)

def test_context_change_preserves_observation_schema():
    env = make_env()
    obs_a, _ = env.reset(seed=123)
    env.set_context("B")
    obs_b, info_b = env.reset(seed=123)
    assert obs_a.shape == obs_b.shape == (2,)
    assert info_b["true_context"] == "B"
