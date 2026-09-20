# V45 UMK R2 — Corrected Direction

Development candidate only. No file under `versions/` or `candidates/v45_umk_local`
is edited or replaced. The parent is v45_umk_local; the parent's frozen bundle is
untouched. This document supersedes `V45_UMK.md` for the r2 candidate only.

## Motivation

The v45 literal formula `lambda_c = lambda0 * sigmoid(p_c)` makes a **larger prior
give faster decay** — the opposite of the intended "better mastery forgets
slowly" hypothesis. `V45_UMK.md` documented this limitation. An epoch-1..14 full
run of the literal version stayed within +0.0002..+0.0009 AUC of v43 and was
converging; because SSM/attention scalars are unconstrained they can partially
self-compensate, while the RWCE branch (lambda always positive) cannot invert.
The r2 candidate fixes the direction and raises initial modulation strength.

## Kernel and Direction (r2)

```text
lambda_c = lambda0 * (1 - sigmoid(p_c))
K_c(gap) = exp(-lambda_c * gap)
```

Higher prior (better mastery) now gives **slower** decay. `umk_lambda0` remains a
finite positive configuration constant; default changed from `0.1` to `0.3`.
Same concept-prior table, training-folds-only initialization, and three branches
as v45.

## Initial Values (r2)

- SSM: `decay_eff = decay ** (1 + alpha * rate)`, `alpha(initial) = 0.3` (was 0.1)
- Attention: `bias = -beta * lambda_concept[j] * Delta`, `beta(initial) = 1.0` (was 0.5)
- RWCE: `weight = exp(-lambda_target * (distance - 1))`, uses corrected rate
- `alpha`/`beta` remain unconstrained learned scalars; only their initial values
  and the shared rate direction changed. Parameter count unchanged: all-on adds
  exactly `ssm.umk_alpha` + `umk_beta` (5,114,871 total for production config).

## Compatibility

- All constructor defaults remain `0`; all-off stays bitwise-identical to v43
  (verified by `test_disabled_state_rng_and_training_are_exactly_v43`).
- Enabled models require a matching enabled checkpoint for strict loading.
- `tests/test_umk.py` updated: direction assertion now monotone-decreasing,
  initial-value assertions `alpha=0.3`/`beta=1.0`, config contract `0.3`.
  **12/12 tests pass** on the development tree before freezing.

## Commands

Same as `V45_UMK.md` commands; use the `assist2017_v45_umk_*` configs under the
frozen r2 bundle. These checks establish implementation/compatibility, not an
AUC improvement. No positive independently retrained ablation or AUC threshold
claim is made.
