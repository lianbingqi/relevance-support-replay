# relevance-support-replay

Course project on relevance- and support-aware experience replay for non-stationary reinforcement learning.

The active synthetic benchmark uses a fixed hidden rotated-actuator task with:

- recurrence: `A -> B -> A`
- novelty: `A -> B -> D`

Contexts are defined before experiments:

- A: 0 degree actuator rotation
- B: 90 degree actuator rotation
- D: 180 degree actuator rotation

The repository intentionally keeps the active experiment tree compact and excludes abandoned calibration artifacts.

## Reproducibility notes

Superseded exploratory retrieval artifacts are removed from the active tree but
remain available in Git history. The current learned-relevance evidence is the
frozen-trajectory comparison under
`results/synthetic_v2/frozen_trajectory_encoder_comparison/`.

The 05a and 05b result directories intentionally retain only their historical
`retrieval_summary.csv` files because the standalone 05b and 05c scripts read
those summaries for comparison outputs. Those legacy summaries are not the
current evaluation evidence.

The canonical frozen trajectory binary (`canonical_seed1_recent_ABA.npz`) is
generated deterministically by
`scripts/run_frozen_trajectory_control_v2.py --mode freeze` and is intentionally
excluded from version control by the repository binary-artifact policy. Its
manifest, hashes, phase counts, and episode boundaries are tracked under
`results/synthetic_v2/frozen_retrieval_trajectory/`.
