# E1d v2.16 — on-policy Q diagnostics and frozen-latent ablation

**Status: pre-outcome development and mechanism diagnosis.** Development seeds 1/2/3 only. Seeds 4/5/6 remain held out and must not be run. This protocol must be committed before E1d outcome collection; E1, E1b, E1c, RQ2 and failed RQ3 results must remain unchanged.

## Motivating observations

E1b reproduced near-universal No-op policy behavior in history-conditioned DDQN after A1 -> B. E1c found large positive fixed-probe No-op Q margins in some seeds, but seed 2 ended with nearly all No-op **rollouts** and a negative mean Q margin on **random-action fixed probes**. Its late training residuals also remained elevated. These are descriptive observations, not proof that the frozen 05c representation caused collapse.

## Question A: policy-conditioned value diagnostics

Compare the same value learner at fixed timestamps on two distinct input populations: (i) researcher-only independent random-action A/B fixed probe trajectories, and (ii) actual states **and their preceding episode-local histories** visited during greedy evaluation of that checkpoint's policy. Record `Q(noop) - max(Q(active))`, fraction selecting No-op, relative No-op margin conditional on history availability (first 8 steps vs later) and state distance (norm(position) <1.5 vs >=1.5), raw return, No-op return, and distance. Keep evaluation resets and episode count exactly equal to E1b/E1c. A difference across probe populations may indicate visitation dependence; it does **not** alone establish causality.

## Question B: controlled information ablation

On the unchanged A1=20,000 -> B=20,000 **common-phase Recent replay** pipeline, compare these arms per development seed:

- `state_only`: original 4-dimensional DDQN, included as historical trajectory control.
- `hc_full_05c`: original history-conditioned DDQN with frozen 05c 16-d latent and valid-history flag, historical control.
- `hc_zero_latent`: same 21-dimensional DDQN with identical seed/architecture/optimizer/training/replay/evaluation, but all 16 latent coordinates set to exactly zero for both action inference and replay training; preserve the valid-history flag with original episode-local semantics. Neither current true context labels nor D data are supplied to the learner.

All arms train from the same method-specific initialization seed; each has its own on-policy trajectory, so comparing outcomes is a randomized-seed intervention on the information channel, **not** a counterfactual single-minibatch gradient effect. The zero-latent control tests whether **the 05c signal is necessary for the observed collapse within this setup**. If both full and zero collapse, we cannot conclude that hidden-context representation is benign, but can rule out the strong hypothesis that nonzero frozen 05c coordinates are necessary. If zero does not collapse, that implicates 05c-information use under the same architecture, but further controls are needed to separate information, input-scale and distribution-shift effects.

The 05c encoder training remains A/B behavior-diverse and frozen. No changes to architecture, rewards, epsilon schedule, optimizer, episode semantics, replay capacity, recent window, batch size, training budget or phase switching. No adaptation A2 or D phase and no historical retrieval or gating experiments in E1d.

## Locked evaluation and validity criteria

Pre-specified B checkpoints: `[0, 201, 1001, 2001, 5001, 10001, 15001, 20000]` (positive internal checkpoints are taken immediately after the original update at that 1-indexed phase step). A1 end is evaluated at step 20,000. At each snapshot, evaluate *five greedy episodes* in each A and B using the same E1b seeds and an independent same-seed No-op baseline. Use an agent deep copy for all diagnostics. Check RNG/weights/optimizer noninterference on the live learner.

For `state_only` and `hc_full_05c`, final A1 and B returns and No-op fractions must reproduce the 18 historical E1b cells across 3 seeds within `1e-6`; each seed does six checks. The B-end A/B fixed-probe No-op fraction and Q-margin must reproduce E1c within `1e-4`. If either guard fails, abort interpretation rather than comparing altered trajectories.

## Evaluation and interpretation

Primary diagnostics are (1) end-B raw **advantage versus same-seed No-op**, (2) end-B No-op fraction, and (3) on-policy versus fixed-probe No-op Q margins, especially seed 2. Report all seed-level data; use no pooled significance test at n=3. Do not tune hidden size, latent dimensions, probes, intervention, checkpoints, or thresholds after seeing outcomes. Keep E1d as diagnostic even if the zero-latent arm outperforms Full 05c. Protect seeds 4/5/6.
