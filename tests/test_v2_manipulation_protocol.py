from pathlib import Path
from src.utils.config import load_yaml

def test_frozen_context_angles():
    cfg = load_yaml(Path("configs/synthetic_v2/manipulation.yaml"))
    assert cfg["contexts"] == {
        "A": {"rotation_deg": 0.0},
        "B": {"rotation_deg": 90.0},
        "D": {"rotation_deg": 180.0},
    }

def test_preregistered_criteria():
    c = load_yaml(
        Path("configs/synthetic_v2/manipulation.yaml")
    )["criteria"]
    assert c["max_off_context_retention"] == 0.80
    assert c["min_pairwise_action_disagreement"] == 0.25
    assert c["require_diagonal_dominance"] is True
