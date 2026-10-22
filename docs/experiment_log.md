# Experiment Log

| Date | Commit | Experiment | Change / Observation | Decision |
|---|---|---|---|---|
| 2026-10-22 | pre-run | Synthetic v2 reset | Replace the abandoned 1D parameter/drift calibration workflow with a fixed hidden rotated-actuator benchmark. | Freeze context semantics before results: A = 0°, B = 90°, D = 180°. |
| 2026-10-22 | pre-run | Synthetic v2 stationary sanity | Implement fixed hidden rotated-actuator contexts A=0°, B=90°, D=180°. Use identical DDQN settings and five training seeds per context. No context tuning is permitted from this point. | Proceed only if all three contexts learn independently, beat random, and show zero greedy-evaluation safety terminations. |
