# Experimental Protocol

## Phase 0: stationary sanity check
Verify that Double DQN with uniform replay learns the stationary synthetic task.

## Phase 1: context calibration
Train/evaluate stationary policies over candidate contexts and estimate the cross-context transfer matrix `T_ij = J(pi_i ; c_j)`. Freeze A/B/D after calibration.

## Phase 2: oracle recurrence
Run `A -> B -> A` with Uniform, Recent, and Oracle Relevant replay. Primary diagnostic: recovery time after returning to A.

**Go/no-go question:** does oracle historical relevance improve recovery relative to recent replay?

## Phase 3: novelty
Run `A -> B -> D` and test behavior when historical support is weak.

## Phase 4: GRU context representation
Evaluate retrieval Precision@k and context separability before using the representation for replay.

## Phase 5: learned relevance replay
Compare learned relevance with oracle relevance.

## Phase 6: historical support
Test whether support distinguishes recurring/covered contexts from novel contexts.

## Phase 7: support-aware replay
Mix relevant historical and recent experience as a function of support.

## Phase 8: full synthetic benchmark
Methods: Uniform, Recent, PER, Learned Relevant, Oracle Relevant, Support-Aware.\nSchedules: `A -> A`, `A -> B -> A`, `A -> B -> D`.

## Phase 9: CARL replication
Calibrate CARL contexts, then repeat recurrence and novelty experiments in CARL CartPole.

## Reproducibility rule
Every formal run records the configuration, seed, Git commit SHA, context schedule, replay method, and summary metrics. Any design change made after inspecting main results must be recorded in `docs/experiment_log.md`.
