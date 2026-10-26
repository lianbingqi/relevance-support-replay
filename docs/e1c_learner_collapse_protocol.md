# E1c v2.15 — History-Conditioned Learner Collapse Diagnosis

**Status:** pre-run, diagnostic development only. Seeds 1/2/3 may be inspected; seeds 4/5/6 remain untouched. This is a follow-up to E1b, which confirmed near-universal No-op choices after B for the history-conditioned DDQN.

## Question and competing hypotheses

Why does the history-conditioned value learner show strong A1 control yet drift to an almost-all-No-op policy by the end of B? Candidate mechanisms include (i) Q-value preference for No-op on matched probe states, (ii) divergence/bias in target or Q estimates, (iii) bootstrapping/optimization instability, (iv) insufficient valid history in training minibatches, and (v) learner representation or coverage mismatch. These are hypotheses, not assumed explanations. TD errors, Q-margins and loss are **diagnostic correlates** and do not by themselves establish causal mechanisms.

## Frozen intervention

Import and invoke the existing E1 `collect_common` for the **A1=20,000, B=20,000 common training phases only**. Do not run any A2 or D adaptation or historical replay branch. Use the same frozen 05c encoder, state-only and history-conditioned DDQNs, recent buffer, model hyperparameters, epsilon schedule, environment context definitions, evaluation seeds, episode semantics and CPU settings. **No algorithm/optimization change** is allowed. Instrument only the call to the existing `update_from_indexed_batch` and perform read-only probes on deep copies. Do not touch E1/E1b/RQ2/RQ3 scripts or results.

## Predeclared observations

1. **Training update traces:** phase step, loss, mean online Q, mean Double-DQN TD target, mean absolute TD error, sampled share of A transitions (diagnostic labels only), sampled fraction of valid history features before and after the action, and intermittently the valid-history fraction of the recent buffer. Save individual updates and pre-specified 500-step-bin means.
2. **Policy rollouts:** exact-seed five-episode evaluation on contexts A and B at fixed A1/B snapshots. Record raw episode return, exact-seed passive No-op return, No-op fraction, mean/final distance, velocity and safety fraction. The No-op baseline is an explicit reference, not a replacement for task-optimal control.
3. **Fixed Q probes:** independent random-action A/B trajectories held out from DDQN training. At each snapshot, record the fraction selecting No-op, mean/median `Q(noop) - max_{a != noop} Q(a)`, mean Q-values and availability of the 05c latent. These probes are researcher-only diagnostic data; do not use context labels in training.
4. **Reproducibility check:** A1 end and B end raw policy returns and action proportions MUST reproduce the previously committed E1b evidence within `1e-6` tolerance. If not, abort result interpretation and investigate drift.

B-phase snapshots at environment phase steps `[0,201,1001,2001,5001,10001,15001,20000]`; A1 `[0,1001,10001,16001,18001,20000]`. Intermediate checkpoints are after the original update at exactly their displayed (1-indexed) phase steps; 0 and 20,000 are phase-boundary probes. Probing the live agent and its RNG is prohibited. The pre-specified fixed Q probe bank uses 6 episodes of at most 64 steps/context, sampled with an independent random policy and local RNG. No new training conditions are permitted after seeing diagnostics.

## Pre-specified descriptive questions

- Does the history-conditioned A1 policy outperform the No-op baseline before B?
- At which B checkpoint does the No-op action fraction become near-universal (descriptively >= 95%)?
- Is the rise in No-op policy actions accompanied by an increase in the fixed-probe `Q(noop)` margin? If not, investigate distribution or state-coverage dependence.
- Do Q means/targets, TD-error or loss exhibit discontinuities or trends near policy collapse? Are valid-history features available to sampled batches?
- Are these associations consistent across seeds 1/2/3 and architectures? Avoid statistical significance claims based on correlated time points.

## Interpretation / publication guardrails

No-op collapse is a **behavioral result**; these correlates do not alone prove its cause. No history-conditioned improvement claim can rely only on random-baseline-normalized AUC. Report raw returns and advantage over No-op. If no single mechanism is supported, record E1c as inconclusive and plan a controlled intervention rather than modifying the outcome metric or model post hoc. Reserved seeds 4/5/6 are not used in E1c. Keep pre-run source commit separate from subsequent outcomes.
