"""E1b instrumentation tests; no environment training required."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_e1b_policy_behavior_audit_v2 as audit


class TinyEnv:
    """Independent tiny deterministic episode, useful for testing event accounting."""
    def __init__(self):
        self.n = 0
    def reset(self, seed=None):
        self.n = 0
        return np.array([1., 0., 0., 0.], dtype=np.float32), {}
    def step(self, action):
        self.n += 1
        state = np.array([1. + self.n, 0., 0., 0.], dtype=np.float32)
        return state, -float(action + 1), False, self.n == 3, {}
    def close(self):
        pass


def test_audit_action_counts_and_noop_return(monkeypatch):
    monkeypatch.setattr(audit.e1, "make_env", lambda *a, **k: TinyEnv())
    monkeypatch.setattr(audit.e1, "get_action", lambda *a, **k: 2)
    cfg = {"isolation": {"eval_episodes": 2}, "novelty": {"eval_episodes": 2}}
    obs = audit.policy_episode(None, "A", 1, cfg, "state_only", None, 8, 0)
    noop = audit.noop_episode("A", 1, cfg, 0)
    assert obs["return"] == -9.0
    assert noop["noop_return"] == -3.0
    assert obs["n_right"] == 3 and obs["n_noop"] == 0
    assert obs["episode_length"] == 3
    assert obs["mean_distance"] == 3.0


def test_instrumented_summary_matches_raw_episode_statistics(monkeypatch):
    monkeypatch.setattr(audit.e1, "make_env", lambda *a, **k: TinyEnv())
    monkeypatch.setattr(audit.e1, "get_action", lambda *a, **k: 0)
    monkeypatch.setattr(audit, "ACTIVE_CONTEXT", audit._audit_phase("adaptation", "ABD", "recent", [0]))
    monkeypatch.setattr(audit, "AUDIT_EVALS", [])
    monkeypatch.setattr(audit, "AUDIT_EPISODES", [])
    cfg = {"isolation": {"eval_episodes": 2}, "novelty": {"eval_episodes": 2}}
    result = audit.audited_eval_greedy(None, "D", 1, cfg, "state_only", None, 8)
    assert result["return_mean"] == -3.0
    assert audit.AUDIT_EVALS[0]["action_noop_fraction"] == 1.0
    assert audit.AUDIT_EVALS[0]["advantage_vs_noop"] == 0
    assert len(audit.AUDIT_EPISODES) == 2


def test_instrumented_comparison_rejects_mismatched_labels():
    curves = pd.DataFrame([dict(seed=1, architecture="state_only", schedule="ABA",
                                method="recent", phase_step=0, eval_return_mean=-2., eval_return_sd=1.)])
    summary = curves.copy()
    summary["phase"] = "adaptation"
    assert len(audit.compare_e1_with_current_audit(curves, summary)) == 1
    summary.loc[0, "phase_step"] = 100
    with pytest.raises(AssertionError):
        audit.compare_e1_with_current_audit(curves, summary)


def test_noop_progress_calculation():
    d = pd.DataFrame([dict(seed=1, eval_context="D", noop_return_mean=-200.,
                           advantage_vs_noop=100.)])
    refs = pd.DataFrame([dict(seed=1, A_reference_state_only=-10.,
                              D_reference_state_only=-20.)])
    x = audit.add_control_quality(d, refs)
    assert x.loc[0,"control_progress_from_noop"] == pytest.approx(100/180)


def test_reserved_seeds_are_not_run():
    # Guard is in main(), and the test ensures its literal safe default is intact.
    import inspect
    source = inspect.getsource(audit.main)
    assert "{1, 2, 3}" in source
    assert "reserved_confirmation_seeds" in source
