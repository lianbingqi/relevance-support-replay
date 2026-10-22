# Experiment Log

| Date | Commit | Experiment | Change / Observation | Decision |
|---|---|---|---|---|
| 2026-10-22 | pre-run | Phase 1B | Correct time-limit handling so truncation ends the rollout but does not mask DDQN bootstrapping; formal 5-seed validation will be run only after the corrected source is committed. Hyperparameters unchanged. | Re-run seeds 1-5 and freeze the baseline only if behavior is stable. |
