# E1b: Policy Behavior Audit (pre-run protocol)

**Status:** Development/diagnostic only. The prior E1 result remains unchanged. Run only development seeds **1, 2, 3**. Seeds **4, 5, 6** remain reserved.

## Motivation

The E1 history-conditioned learner attains zero-shot normalized scores around 0.95–0.98 for both ABA and ABD. Those scores are normalized against a very poor random baseline, so a passive No-op policy may appear nearly optimal although the stationary control reference has much higher raw return. Before attributing differences to better context representation or replay, test whether greedy policies actually control the system.

## Frozen scientific design

Import the previously merged `run_e1_history_conditioning_development_v2.py` and use its training, representation, replay buffer, agent update, hyperparameters, schedule, fork, evaluation resets, and metrics without modification. Instrument its original `eval_greedy` at the **same calls** rather than adding extra agent evaluations during adaptation. The only additional evaluation is an A/B policy probe after the common B phase on a **deep copy** of the agent, so it cannot advance the training agent's exploration RNG.

Data: the same 2 architectures × 3 replay methods × ABA/ABD × seeds 1/2/3 as E1. No outcome-driven changes to reward, environment, action set, architecture, training schedule, historical fraction, or support calibration. Save all E1b outputs in a new directory.

## Outcomes

For every original evaluation checkpoint, record episode-level greedy action counts (`noop`, `left`, `right`, `down`, `up`), raw return, safety termination, episode length, average and final Euclidean target distance, and average/final speed. Compute exact-seed *always No-op* returns using the same initial state and environment noise seeds; do not use the trained agent to compute this baseline. Report the difference in raw return between the learned policy and No-op.

Define **No-op-anchored control progress** (diagnostic only):

\[ P=\frac{J_\text{policy}-J_\text{noop}}{J_\text{stationary state-only reference}-J_\text{noop}} \]

This diagnostic is 0 for No-op and 1 for the existing stationary control reference on the same seed/context. It is not a replacement for normalized AUC in previously frozen experiments. Compare policy behavior during A1, at the common post-B checkpoint, immediately post-switch and after 200/500/1,000/2,000/10,000 adaptation transitions.

## Checks and interpretation

- E1b's return mean and standard deviation must match the values handed to the original E1 curve construction for **every** corresponding evaluation row.
- E1b rerun vs historical E1 metrics: report any drift, **never** post-hoc tune to historical values. Cross-run bitwise equality is not a precondition for within-run diagnostic interpretation.
- A high No-op action fraction **together with** raw return indistinguishable from exact-seed No-op is evidence for passive policy behavior. High normalized score alone is not.
- If actual performance near the state-only reference and active actions are observed, reject the No-op-collapse explanation for that condition.
- If early No-op collapse is confirmed, do not interpret high zero-shot normalized AUC as recovered high-quality control.
- This is an observational behavior audit, not a test of a new algorithm or a causal identification of replay utility.

## Repository policy

Keep RQ3c/RQ3d negative results intact. Commit the script, tests, and this protocol in a separate **pre-run commit**, then commit new E1b diagnostic results. Never overwrite historical E1 files. Do not run seeds 4/5/6.

## Pre-outcome implementation correction

The initial E1b run stopped before completed results were saved.
E1's evaluation schedule already includes phase step 0, which the
E1b wrapper accidentally counted twice. The wrapper now
deduplicates checkpoint indices. Training, replay, seeds,
evaluation metrics and scientific criteria remain unchanged.
