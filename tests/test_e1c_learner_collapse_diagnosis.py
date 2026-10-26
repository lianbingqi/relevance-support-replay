"""E1c diagnostic invariants. No environment training in unit tests."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

import scripts.run_e1c_learner_collapse_diagnosis_v2 as e1c


def cfg():
    return {
        "experiment": {"train_frequency": 4, "learning_starts": 1000},
        "isolation": {"A1_steps": 20000},
        "e1c": {
            "q_snapshot_target_steps_A1": [0, 1001, 10001, 16001, 18001, 20000],
            "q_snapshot_target_steps_B": [0, 201, 1001, 2001, 5001, 10001, 15001, 20000],
        },
    }


def test_exact_original_update_clock_and_no_checkpoint_zero_duplicate():
    a = e1c._begin_phase("A1", 1, "state_only", cfg(), None, None, None, 8, 0)
    b = e1c._begin_phase("B", 1, "state_only", cfg(), None, None, None, 8, 20000)
    assert a["first_update_global"] == 1000
    assert b["first_update_global"] == 20000
    assert a["first_update_global"] - 0 + 1 == 1001
    assert b["first_update_global"] - 20000 + 1 == 1
    assert 0 not in a["update_snapshots"]
    assert 20000 not in b["update_snapshots"]
    assert 201 in b["update_snapshots"]


def test_update_binner_keeps_risk_metrics_distinct():
    d = pd.DataFrame([
        dict(seed=1, architecture="history_conditioned", phase="B", phase_step=i,
             loss=float(i), q_mean=-float(i), target_mean=-2 * float(i),
             td_abs_mean=i/2., sampled_A_fraction_diagnostic=.2,
             batch_history_valid_before=.7, batch_history_valid_after=.8,
             recent_history_valid_fraction=np.nan)
        for i in (1, 5, 501, 505)
    ])
    x = e1c.summarize_update_bins(d)
    assert len(x) == 2
    assert list(x["phase_bin_start"]) == [0, 500]
    assert list(x["n_updates"]) == [2, 2]
    assert x.loc[0, "target_mean"] == -6


def test_fixed_q_probe_detects_noop_margin_and_action_counts(monkeypatch):
    # A model with noop dominating each fixed probe by two Q-value units.
    class ToyNet(torch.nn.Module):
        def forward(self, inputs):
            x = torch.full((len(inputs), 5), -2.0, dtype=torch.float32)
            x[:, 0] = 0.0
            return x
    toy_agent = SimpleNamespace(online=ToyNet(), device=torch.device("cpu"))
    monkeypatch.setattr(e1c, "FIXED_PROBES", {
        "A": {"state": np.zeros((10, 4), dtype=np.float32),
              "latent": np.zeros((10, 17), dtype=np.float32), "n": 10},
        "B": {"state": np.ones((8, 4), dtype=np.float32),
              "latent": np.ones((8, 17), dtype=np.float32), "n": 8},
    })
    e1c.Q_PROBES.clear()
    e1c.fixed_probe_q(toy_agent, "history_conditioned", "B", 1001, 1)
    assert len(e1c.Q_PROBES) == 2
    assert all(x["probe_noop_greedy_fraction"] == 1.0 for x in e1c.Q_PROBES)
    assert all(x["probe_noop_margin_mean"] == 2.0 for x in e1c.Q_PROBES)
    e1c.Q_PROBES.clear()


def test_instrument_update_invokes_e1_once_and_preserves_statistics(monkeypatch):
    replay = SimpleNamespace(last_sample_indices=np.asarray([0, 1]),
                             buffer=[SimpleNamespace(context_id="A"), SimpleNamespace(context_id="B")])
    fake = SimpleNamespace(loss=.5, q_mean=-4, target_mean=-6, td_abs_mean=2)
    calls = []
    def orig(agent, replay, store, arch, bs):
        calls.append((arch, bs))
        return fake
    monkeypatch.setattr(e1c, "ORIGINAL_UPDATE", orig)
    monkeypatch.setattr(e1c, "CURRENT", {
        "update_index": 0, "freq": 4, "first_update_global": 20000,
        "start_global": 20000, "seed": 1, "phase": "B", "update_snapshots": set(),
    })
    e1c.UPDATES.clear()
    r = e1c.instrument_update(None, replay, None, "state_only", 2)
    assert r is fake and calls == [("state_only", 2)]
    row = e1c.UPDATES[-1]
    assert row["phase_step"] == 1
    assert row["sampled_A_fraction_diagnostic"] == .5
    assert row["loss"] == .5
    e1c.UPDATES.clear()


def test_e1b_reference_comparison_flags_drift(tmp_path):
    path = tmp_path / "results/synthetic_v2/e1b_policy_behavior_audit"
    path.mkdir(parents=True)
    pd.DataFrame([
        dict(seed=1, architecture="history_conditioned", phase="A1", phase_step=20000,
             eval_context="A", eval_return_mean=-5., noop_fraction=.3),
        dict(seed=1, architecture="history_conditioned", phase="post_B", phase_step=40000,
             eval_context="A", eval_return_mean=-250., noop_fraction=1.),
        dict(seed=1, architecture="history_conditioned", phase="post_B", phase_step=40000,
             eval_context="B", eval_return_mean=-230., noop_fraction=1.),
    ]).to_csv(path / "policy_evaluation_summaries.csv", index=False)
    current = pd.DataFrame([
        dict(seed=1, architecture="history_conditioned", phase="A1", phase_step=20000,
             eval_context="A", raw_return=-5., eval_noop_fraction=.3),
        dict(seed=1, architecture="history_conditioned", phase="B", phase_step=20000,
             eval_context="A", raw_return=-250., eval_noop_fraction=1.),
        dict(seed=1, architecture="history_conditioned", phase="B", phase_step=20000,
             eval_context="B", raw_return=-230., eval_noop_fraction=1.),
    ])
    d = e1c.compare_e1b(current, tmp_path)
    assert len(d) == 3 and d.return_drift.abs().max() == 0
    current.loc[1, "raw_return"] += 100
    altered = e1c.compare_e1b(current, tmp_path)
    assert altered.return_drift.abs().max() == 100


def test_snapshot_is_read_only_on_live_agent_and_rng(monkeypatch):
    import numpy as np
    import torch
    import scripts.run_e1c_learner_collapse_diagnosis_v2 as diag
    from types import SimpleNamespace

    class ToyAgent:
        def __init__(self):
            self.online = torch.nn.Linear(4, 5)
            self.target = torch.nn.Linear(4, 5)
            self.optimizer = torch.optim.Adam(self.online.parameters(), lr=.001)
            self.rng = np.random.default_rng(100)
            self.update_count = 0
    agent = ToyAgent()
    replay = SimpleNamespace(rng=np.random.default_rng(200))
    monkeypatch.setattr(diag, 'fixed_probe_q', lambda *args: None)
    monkeypatch.setattr(diag.e1b, 'policy_episode', lambda *args: {
        'return': -10., 'episode_length': 200, 'n_noop': 200,
        'mean_distance': 1., 'final_distance': 1.,
        'mean_speed': 0., 'terminated': False,
    })
    monkeypatch.setattr(diag.e1b, 'noop_episode', lambda *args: {'noop_return': -10.})
    diag.SNAPSHOTS.clear()
    a = diag._rng_fingerprint(agent, replay)
    b = diag._agent_parameter_fingerprint(agent)
    diag._snapshot(agent, replay, None, 'state_only', 'B', 0, 1, {}, 8)
    assert a == diag._rng_fingerprint(agent, replay)
    assert b == diag._agent_parameter_fingerprint(agent)
    assert len(diag.SNAPSHOTS) == 2
    diag.SNAPSHOTS.clear()
