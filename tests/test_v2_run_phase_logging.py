from scripts.run_learned_relevance_retrieval_pilot_v2 import run_phase
from src.replay.recent import RecentReplayBuffer


class FixedAgent:
    def act(self, state, epsilon=0.0):
        return 0

    def update(self, batch):
        raise AssertionError("update should not run in this test")


def make_cfg():
    return {
        "environment": {
            "rho": 0.85,
            "beta": 0.12,
            "noise_std": 0.01,
            "init_position_range": 1.0,
            "init_velocity_range": 0.10,
            "max_episode_steps": 200,
            "safety_radius": 25.0,
            "safety_speed": 10.0,
        },
        "contexts": {
            "A": {"rotation_deg": 0.0},
            "B": {"rotation_deg": 90.0},
            "D": {"rotation_deg": 180.0},
        },
        "agent": {
            "batch_size": 64,
        },
        "experiment": {
            "learning_starts": 10_000_000,
            "train_frequency": 4,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
        },
    }


def n_logged(episodes):
    return sum(len(e["actions"]) for e in episodes)


def test_exact_episode_boundary_is_logged_once():
    cfg = make_cfg()

    replay = RecentReplayBuffer(
        capacity=1000,
        recent_window=1000,
        seed=1,
    )

    global_step, episodes = run_phase(
        agent=FixedAgent(),
        replay=replay,
        context="A",
        phase_steps=200,
        global_step=0,
        seed=1,
        phase_index=0,
        cfg=cfg,
    )

    assert global_step == 200
    assert len(replay) == 200
    assert len(episodes) == 1
    assert n_logged(episodes) == 200
    assert len(episodes[0]["actions"]) == 200
    assert episodes[0]["episode_id"] == 0
    assert episodes[0]["start_phase_step"] == 0


def test_mid_episode_phase_end_flushes_only_partial_tail():
    cfg = make_cfg()

    replay = RecentReplayBuffer(
        capacity=1000,
        recent_window=1000,
        seed=1,
    )

    global_step, episodes = run_phase(
        agent=FixedAgent(),
        replay=replay,
        context="A",
        phase_steps=250,
        global_step=0,
        seed=1,
        phase_index=0,
        cfg=cfg,
    )

    assert global_step == 250
    assert len(replay) == 250
    assert len(episodes) == 2
    assert [len(e["actions"]) for e in episodes] == [200, 50]
    assert n_logged(episodes) == 250
    assert [e["episode_id"] for e in episodes] == [0, 1]
    assert [e["start_phase_step"] for e in episodes] == [0, 200]
