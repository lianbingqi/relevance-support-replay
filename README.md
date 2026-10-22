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
