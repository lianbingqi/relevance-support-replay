from pathlib import Path
from src.utils.config import load_yaml

def test_fine_candidates_and_reference():
    cfg = load_yaml(Path("configs/synthetic/drift_fine_calibration.yaml"))
    assert cfg["reference_context"] == "A"
    assert list(cfg["contexts"]) == [
        "Dm07", "Dm06", "Dm05", "A", "Dp05", "Dp06", "Dp07"
    ]
    assert cfg["contexts"]["A"]["drift"] == 0.0

def test_preregistered_B_screen():
    s = load_yaml(
        Path("configs/synthetic/drift_fine_calibration.yaml")
    )["selection_screen"]
    assert s["min_late_train_full_length_fraction"] == 0.90
    assert s["min_self_eval_full_length_fraction"] == 0.95
    assert s["min_raw_transfer_penalty"] == 2.0
    assert s["min_action_disagreement_from_A"] == 0.15
