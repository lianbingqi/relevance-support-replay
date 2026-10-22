import numpy as np
from src.environments.synthetic_control import HiddenContextControlEnv

def make_env():
    return HiddenContextControlEnv(
        contexts={
            "A": {"rho": 0.85, "beta": 0.18},
            "B": {"rho": 0.85, "beta": 0.18, "drift": 0.08},
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
    assert info["context_vector"].shape == (3,)

def test_old_context_without_drift_defaults_to_zero():
    env = make_env()
    env.set_context("A")
    np.testing.assert_allclose(
        env.context_vector(),
        np.asarray([0.85, 0.18, 0.0], dtype=np.float32),
    )

def test_reset_is_seed_reproducible():
    env = make_env()
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    np.testing.assert_allclose(a, b)

def test_drift_changes_same_state_same_action_dynamics():
    contexts = {
        "neg": {"rho": 0.85, "beta": 0.18, "drift": -0.08},
        "pos": {"rho": 0.85, "beta": 0.18, "drift": 0.08},
    }
    neg = HiddenContextControlEnv(contexts, "neg", noise_std=0.0)
    pos = HiddenContextControlEnv(contexts, "pos", noise_std=0.0)
    neg.state = np.asarray([0.0, 0.0], dtype=np.float32)
    pos.state = np.asarray([0.0, 0.0], dtype=np.float32)
    n, *_ = neg.step(1)
    p, *_ = pos.step(1)
    assert n[1] < 0.0 < p[1]
    assert n[0] < p[0]

def test_context_change_preserves_observation_schema():
    env = make_env()
    a, _ = env.reset(seed=123)
    env.set_context("B")
    b, info = env.reset(seed=123)
    assert a.shape == b.shape == (2,)
    assert info["true_context"] == "B"
