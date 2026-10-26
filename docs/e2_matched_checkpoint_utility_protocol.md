# E2 v2.17 — Matched-Checkpoint Historical Replay Utility

**Status:** locked, *pre-outcome* diagnostic development protocol. Seeds **1/2/3** only; **4/5/6** untouched. No new replay algorithm or threshold is evaluated. Existing E1/E1b/E1c/E1d and RQ2/RQ3 files remain immutable.

## Question and estimand

Does the dynamics-prediction compatibility of old experiences predict their actual impact on the *current learner*? A useful replay policy is not assumed to exist. For a fixed state-only DDQN checkpoint \(\theta_t\) and a fixed offline update budget, define

\[
U_{t,r}(H)=J_c(\pi_{\theta_t^{16}(0.8H+0.2R)})-J_c(\pi_{\theta_t^{16}(R)}),
\]

where \(R\) contains **only post-switch current-context transitions**, \(H\) contains only earlier A1/B transitions, both forks start from the identical network, target, optimizer, replay contents and update counter, and \(J_c\) is greedy policy return on the same five evaluation episodes of the current context. This is a **local controlled offline-update contrast**, not a full online adaptation outcome or an oracle utility available to a deployment-time gate. The historical-data treatment replaces 80% of current training samples; sample composition can itself induce effects.

**Important protocol difference:** `current_only` is STRICTLY post-switch current-phase sampling (excluding all pre-switch B observations). It is **not** the legacy Recent 5,000-step rolling window of RQ2/RQ3. No cross-project AUC equivalence is claimed.

## Frozen conditions

- HiddenRotatedActuatorEnv: contexts A=0°, B=90°, D=180°. Frozen 05c GRU trained on A/B only and used for **retrieval/compatibility diagnostics, never fed into Q-network**. State-only DDQN as original canonical E1 model.
- A1 20,000 steps then B 20,000 steps, with original E1 collect/recent/DDQN implementation, on canonical CPU. Before collecting any E2 outcomes, raw return and No-op fractions at A1 end and B end on A/B must reproduce E1b for each development seed within `1e-6`.
- From the end of B, run **one** normal E1 Recent 5,000-step adaptation trajectory per seed/schedule (ABA and ABD). Save diagnostic checkpoints **after** 200, 500, 1,000, and 2,000 post-switch transitions, all from the same trajectory. Use these checkpoints solely as starting states for intervention; they are not selected by outcome. We omit phase step zero because no current-phase observations exist yet and a "current-only" control is undefined.
- At a checkpoint, remove 32 randomly selected *current-phase* transitions from the offline training pool and use them **only** to measure held-out TD loss. Do not train on this frozen holdout subset during interventions. The 05c compatibility diagnostic uses the last 16 valid observed one-step transitions with their own within-episode preceding eight-step histories; never reaches across episode boundaries.
- Historical candidate sets based ONLY on learned embedding cosine similarity: `retrieved_top50`, `uniform50` seeded random bank windows, and `least_relevant50`. The 50-window A1/B bank rows produce historical replay transition pools; optional A1/B labels are used **only for post-hoc composition diagnostics**. No D labels, no true context labels in methods, no future evidence.
- For each checkpoint and **three** independent minibatch draws, compare the same `current_only` update arm to each 80% old / 20% current candidate. **16 gradient updates**, batch size **64**, identical target-network synchronization rules and Adam optimizer state. Current-only minibatches are identical across candidate contrasts within a replicate, and 13 current samples in each historical minibatch are a subset of the corresponding current-only batch. All sample pools disjoint in time; within-batch sampling has no replacement. Each branch is a fresh deep copy. No extra environment transitions or post-update adaptation during the intervention.
- Greedy evaluation: **five** exact same-seed episodes in current A or D. Report unnormalized return and improvement against a passive No-op policy, No-op choice rate, mean distance to origin, safety termination, and pre/post held-out current TD loss. No random-baseline-normalized AUC should be used as proof of control.
- For each candidate at each checkpoint, compute mean retrieved-mean predictive MSE over 16 previously observed transitions and cosine alignment between gradients from current and historical minibatches on the **frozen checkpoint**. These are correlates, not learning-utility ground truth and not an implemented gating mechanism.

## Outputs and analysis rules

Emit unfiltered per-seed/ schedule/checkpoint/replicate fork outcomes, per-candidate compatibility and gradient diagnostics, paired `historical minus current_only` raw-return effects, and an all-cell utility summary. Report all three replicate outcomes and all seeds, including negative results. Descriptively analyze conditions where a prediction-compatible candidate harms learning, a less compatible candidate helps, and whether rank alignment is consistent; do not fit/select thresholds or claim p-values from nonindependent checkpoints/replicates. Distinguish optimizer/TD-loss proxies from actual policy return. If 16 updates yield negligible or unstable effects, record a *null/low-sensitivity local intervention* rather than increasing the budget based on observed outcomes. Do not run 4/5/6 unless a future method/protocol is separately frozen.

## Interpretation guardrails

Post-switch learner checkpoints are sampled along a Recent trajectory; results are conditional on those states, not the stationary distribution of all possible learners. A1/B bank and recent-phase data have different distributions, so causal treatment is the **sampling regime**, not intrinsic value of one named transition independent of coverage. If E2 shows a mismatch, this establishes a **specific benchmark-level counterexample**, not an universal law. If no mismatch, do not construct a post-hoc heuristic to rescue the original hypothesis. The E1 architecture collapse is a separate finding and is not mixed into E2.
