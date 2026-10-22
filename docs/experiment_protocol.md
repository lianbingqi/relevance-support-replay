# Experimental Protocol

## Synthetic v2: hidden rotated actuator

The synthetic benchmark is a 2D control task with hidden actuator rotation.

The observable state contains position and velocity. The reward, goal, action labels, and agent architecture remain fixed across contexts. Only the hidden transition mechanism changes.

Contexts are fixed before experiments:

- A: 0° actuator rotation
- B: 90° actuator rotation
- D: 180° actuator rotation

The context label and rotation angle are never included in the agent observation.

## Phase 1: stationary sanity
Train the same DDQN independently in A, B, and D. All three contexts must be stably learnable.

## Phase 2: manipulation check
Cross-evaluate stationary policies across A, B, and D. Confirm clear cross-context performance degradation and shared-state action disagreement. A, B, and D are not tuned using these results.

## Phase 3: oracle recurrence
Run `A -> B -> A` with Uniform, Recent, and Oracle Relevant replay.

Do not reset the network, optimizer, exploration schedule, or replay memory at context switches.

Primary go/no-go question: does oracle access to historically relevant A experience improve recovery after returning to A relative to Recent replay?

## Phase 4: oracle novelty sanity
Run `A -> B -> D` to verify that historical reuse is not automatically beneficial when no historical exact context match exists.

## Phase 5: latent context representation
Add a history encoder and test whether recent transition history yields a useful representation of hidden dynamics.

## Phase 6: learned relevance
Use latent-context similarity to retrieve historically relevant experience and compare against Oracle Relevant.

## Phase 7: historical support
Measure whether the current latent context is well supported by historical experience.

## Phase 8: support-aware replay
Adapt the historical/recent replay mixture as a function of historical support.

## Phase 9: full synthetic benchmark
Compare Uniform, Recent, PER, Learned Relevant, Oracle Relevant, and Support-Aware replay.

## Phase 10: CARL replication
Port recurrence and novelty experiments to CARL CartPole after the synthetic mechanism is validated.

## Reproducibility rule
Formal runs record configuration, seed, Git commit SHA, context schedule, replay method, and summary metrics.
