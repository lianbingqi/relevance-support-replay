# Synthetic v2 research status (before E1)

This document indexes scientific evidence without removing failed experiments.
Original commits, configs, raw result tables, plots, and diagnostic scripts remain authoritative.

| Experiment | Status | Interpretation | Evidence |
|---|---|---|---|
| Stationary / manipulation | PASS | Fixed A/B/D dynamics are learnable and distinct | `results/synthetic_v2/stationary/`, `manipulation/` |
| Oracle recurrence | Positive diagnostic | A historical match can accelerate return to A | `oracle_recovery_isolation/`, `oracle_recovery_3seed/` |
| 05b random encoder | FAIL for frozen retrieval | Strong offline model is insufficient for early online retrieval | `dynamics_isolated_retrieval_pilot/` |
| 05c behavior-diverse encoder | Informative offline representation | Good offline history prediction and retrieval | `behavior_diverse_retrieval_pilot/` |
| 05d frozen-trajectory threshold | FAIL | Early first-512 retrieval did not reach frozen 0.90 threshold | `frozen_trajectory_encoder_comparison/` |
| RQ2 learned relevance | Positive 3-seed evidence | Early ABA adaptation improves over Recent | `learned_replay_recurrence_3seed/` |
| RQ3a kNN global density | FAIL | Policy-induced visitation confounds global latent density | `novelty_support_pilot/` |
| RQ3a negative-transfer observation | Observed seed-1 failure case | Relevance-only replay may harm ABD adaptation | `novelty_support_pilot/` |
| RQ3b compatibility | Development result | Retrieved mean prediction MSE distinguishes A/D offline | `support_compatibility_development/` |
| RQ3c one-step support | **FORMAL CONFIRMATION FAIL** | Support improved over Relevance in ABD but failed pre-registered primary criteria | `support_aware_heldout_confirmation/` |
| RQ3d window support | **DEVELOPMENT FAIL** | 8-score averaging failed pre-registered six-cell / mean / worst regret criteria | `temporal_support_development/` |
| E1 history-conditioned value learner | **PROPOSED / DEVELOPMENT** | Diagnostic 2 architecture × 3 replay × 2 schedule experiment | `configs/synthetic_v2/e1_history_conditioning_development.yaml` |

## Reproducibility and data provenance

- The historical exact trajectory disagreement across runtimes is documented in `results/synthetic_v2/execution_reconciliation/`. Same-runtime fork identity is required for local paired comparisons.
- Development RL seeds 1/2/3 are not fresh confirmations. Seeds 4/5/6 remain reserved and must not be used before a future protocol is frozen.
- Context labels are for diagnostics/oracle only, never learned encoder, retrieval, Q input, gate, or sampling decisions.
- Relevance, predictive compatibility, and marginal learning utility are distinct concepts. Utility is not yet measured.

## Repository cleanup policy

- **Retain:** failed experiments, immutable raw results, pre-run experiment log, configs, scripts, tests, metadata, and negative results.
- **Exclude from future commits:** runtime caches, `.ipynb_checkpoints`, downloaded dependencies, secrets, temporary scratch outputs, duplicate exported notebooks, oversized checkpoints not required for reproducibility.
- For genuinely superseded non-evidential files, move/label them in a separately documented maintenance PR; never silently overwrite or delete historical evidence.
- GitHub `main` is the source of truth; maintain experimental lineage via commit SHA and PR.
