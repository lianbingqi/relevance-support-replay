from pathlib import Path
from src.utils.config import load_yaml

def test_calibration_candidate_set():
    cfg = load_yaml(Path("configs/synthetic/context_calibration.yaml"))
    assert cfg["reference_context"] == "A"
    assert cfg["contexts"]["A"] == {"rho": 0.85, "beta": 0.18}
    assert len(cfg["contexts"]) == 9

def test_calibration_budget():
    cfg = load_yaml(Path("configs/synthetic/context_calibration.yaml"))
    assert cfg["calibration"]["train_seeds"] == [1, 2, 3]
    assert cfg["calibration"]["training_episodes"] == 300
