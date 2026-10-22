from src.environments.synthetic_control import HiddenContextControlEnv


def test_time_limit_truncation_is_separate_from_termination():
    env = HiddenContextControlEnv(
        contexts={"A": {"rho": 0.85, "beta": 0.18}},
        context_name="A",
        max_episode_steps=1,
        noise_std=0.0,
        init_x_range=0.01,
        init_v_range=0.0,
        x_limit=100.0,
        v_limit=100.0,
    )

    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(1)

    assert terminated is False
    assert truncated is True
