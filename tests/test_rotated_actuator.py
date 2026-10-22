import numpy as np

from src.environments.rotated_actuator import HiddenRotatedActuatorEnv


CONTEXTS = {
    "A": {"rotation_deg": 0.0},
    "B": {"rotation_deg": 90.0},
    "D": {"rotation_deg": 180.0},
}


def make_env(context_name="A"):
    return HiddenRotatedActuatorEnv(
        contexts=CONTEXTS,
        context_name=context_name,
        rho=0.0,
        beta=1.0,
        noise_std=0.0,
        init_position_range=0.0,
        init_velocity_range=0.0,
        max_episode_steps=10,
    )


def test_context_is_hidden_from_observation():
    env = make_env("A")
    obs, info = env.reset(seed=1)

    assert obs.shape == (4,)
    assert "rotation_deg" not in info
    assert info["true_context"] == "A"


def test_right_action_rotates_from_A_to_B_to_D():
    env_a = make_env("A")
    env_b = make_env("B")
    env_d = make_env("D")

    for env in (env_a, env_b, env_d):
        env.reset(seed=1)

    # action 2 = nominal right
    next_a, *_ = env_a.step(2)
    next_b, *_ = env_b.step(2)
    next_d, *_ = env_d.step(2)

    np.testing.assert_allclose(next_a[2:], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(next_b[2:], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(next_d[2:], [-1.0, 0.0], atol=1e-6)


def test_time_limit_is_truncation_not_terminal():
    env = HiddenRotatedActuatorEnv(
        contexts=CONTEXTS,
        context_name="A",
        noise_std=0.0,
        max_episode_steps=1,
        safety_radius=1e6,
        safety_speed=1e6,
    )

    env.reset(seed=5)
    _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True


def test_contexts_have_same_observation_and_action_spaces():
    env_a = make_env("A")
    env_b = make_env("B")
    env_d = make_env("D")

    assert env_a.observation_space.shape == (4,)
    assert env_b.observation_space.shape == (4,)
    assert env_d.observation_space.shape == (4,)

    assert env_a.action_space.n == 5
    assert env_b.action_space.n == 5
    assert env_d.action_space.n == 5
