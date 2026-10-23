# Experiment Log

| Date | Commit | Experiment | Change / Observation | Decision |
|---|---|---|---|---|
| 2026-10-22 | pre-run | Synthetic v2 reset | Replace the abandoned 1D parameter/drift calibration workflow with a fixed hidden rotated-actuator benchmark. | Freeze context semantics before results: A = 0°, B = 90°, D = 180°. |
| 2026-10-22 | pre-run | Synthetic v2 stationary sanity | Implement fixed hidden rotated-actuator contexts A=0°, B=90°, D=180°. Use identical DDQN settings and five training seeds per context. No context tuning is permitted from this point. | Proceed only if all three contexts learn independently, beat random, and show zero greedy-evaluation safety terminations. |
| 2026-10-22 | pre-run | Synthetic v2 manipulation check | With A=0°, B=90°, D=180° frozen, train five stationary seeds/context and cross-evaluate all policies on all contexts. Pre-register diagonal dominance, best off-context normalized retention <=0.80, and pairwise greedy action disagreement >=0.25 on 4096 fixed probe states. | Go/no-go only; do not tune context angles from results. |
| 2026-10-22 | pre-run | Synthetic v2 oracle recurrence pilot | Replace the computationally heavy 5-seed ABA run with a one-seed pilot. Use A1/B/A2=20k/20k/10k transitions, Recent window=5k, 5 eval episodes every 2k steps, and methods Uniform/Recent/Oracle. | Expand only to paired seeds 2 and 3 if Oracle improves censored recovery by >=2k steps and early-A2 normalized score by >=0.10 versus Recent. |
