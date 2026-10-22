# Experiment Log

| Date | Commit | Experiment | Change / Observation | Decision |
|---|---|---|---|---|
| 2026-10-22 | pre-run | Phase 1B | Correct time-limit handling so truncation ends the rollout but does not mask DDQN bootstrapping; formal 5-seed validation will be run only after the corrected source is committed. Hyperparameters unchanged. | Re-run seeds 1-5 and freeze the baseline only if behavior is stable. |
| 2026-10-22 | pre-run | Phase 2 calibration | Freeze a 9-context candidate set around A=(0.85, 0.18); 3 stationary training seeds/context; use raw and random-normalized cross-context transfer. | Inspect results before freezing B/D. |
| 2026-10-22 | pre-run | Phase 2B drift recalibration | Prior rho/beta calibration produced near-complete cross-context transfer. Extend hidden dynamics with persistent drift d while preserving A exactly at d=0; pre-register seven drift candidates from -0.12 to +0.12. Add shared-state greedy action disagreement as a mechanism diagnostic. | Use calibration only to select B/D before recurrence/novelty experiments. |
| 2026-10-22 | pre-run | Phase 2C fine drift calibration | Search only drift magnitudes 0.05, 0.06, 0.07 around A. Pre-register B criteria: late full-length >=0.90, self-eval full-length >=0.95, raw transfer penalty >=2.0, action disagreement >=0.15. | Do not change thresholds after viewing results; freeze B only after raw review. |
