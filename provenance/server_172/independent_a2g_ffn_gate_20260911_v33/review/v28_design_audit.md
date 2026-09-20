# Prospective V28 Structure Review

The full same-five-of-eight objective remains unmet. The latest independently
audited Assist2017 score is V27 at 0.8000407760611609, below the unchanged
strict floor 0.8174. V27's historical-pace contrast passed its registered
retention rule. Preserve that component and all other V27 components.

## Hypothesis

V27 writes item/concept/response embeddings into its trunk and produces a
probability only at the end. It does not use its previous prediction errors
to estimate a learner-specific correction. Equal raw responses to questions
with different predicted success probabilities supply different evidence.

For each target t, let l_j be the uncorrected V27 logit and p_j=sigmoid(l_j).
Construct a bounded feature phi_j from the existing target embedding, with
an intercept and one existing attention head's width of learned coordinates.
Use only observed events j<t:

    H_t = I + sum_j p_j*(1-p_j)*phi_j*phi_j^T
    g_t = sum_j (r_j-p_j)*phi_j
    delta_t = phi_t^T * inverse(H_t) * g_t
    final_logit_t = l_t + alpha*delta_t

This is one unit-ridge Newton/Fisher-scoring step at zero learner offset,
not an exact Bayesian logistic posterior or a mastery estimator. The
reference probabilities are the current model's uncorrected causal
probabilities, not a frozen pretrained model or the corrected probabilities.
Gradients may flow through the differentiable refinement. No optimizer,
validation fitting, extra loss or second trunk pass is used inside forward.

The inverse is updated using the Sherman-Morrison identity. A separate
prefix-by-prefix torch.linalg.solve oracle must verify values and gradients.
The feature vector has squared norm at most 2, so for at most 199 histories
the exact precision matrix has condition number at most 100.5.

## Fixed Design

- Projection rank is d_model/num_heads, inherited from one attention head.
- Features are [1, tanh(W*LayerNorm(target))/sqrt(rank)].
- LayerNorm has no affine parameters; the prior precision is exactly I.
- A single trainable alpha is zero initialized, preserving V27's initial
  predictions, common gradients and stochastic module calls.
- The feature projection uses a forked CPU RNG. All 93 V27 state tensors
  and the post-construction RNG stream must remain identical.
- At d_model=256 and heads=8 this adds 8,193 parameters in two tensors.
- Invalid/padded or unobserved responses make no precision or score update.
- Unknown concept IDs may be real observed events, but padding ID0 is not.
- State resets for every existing width200 segment and every learner.
- Only past responses are read, never shft_rseqs or current/future outcomes.
- V27's history-time boundary and all six original ablation flags remain.

## Bounded Historical Review

The inspected Rasch prototypes alter static item/concept embedding
coordinates. The old causal channel recalibration uses cumulative hidden
channel means. The normalized-innovation prototype subtracts normalized
SSM input from its output. V20 is a concept-specific binary-state Bayesian
filter with separately learned emissions. None of those inspected
computations uses the current trunk's past prediction errors and a
target-conditioned Fisher matrix. The inspected gated-delta prototype
predicts learned value vectors from an associative state and applies
learned retention/write gates; it neither predicts the binary outcome
with the preserved trunk nor accumulates its Bernoulli Fisher matrix.

The historical ALiBi files establish that fixed relative-position attention
is already implemented; do not present it as a new candidate here.
This is a bounded source distinction, not a literature novelty claim.
No new production data, validation labels or trained weights were inspected
to choose this structure.

## Execution And Decision

Only one independent Assist2017 Full training run is proposed: train folds
1/2/3/4, validation fold0, seed42, unchanged batches, optimizer, max200 and
patience20. No baseline or V27 control retraining, pretrained initialization,
sweep, extra seed or automatic retry is allowed.

The planned controls are evidence, SSM, attention, factorized input,
item-attempt, history pace and prequential Newton readout. Register 56
contrasts across the original eight datasets; do not replace a retained
module or count one new positive contrast as all seven effects.
Reuse the independent V27 control only after source, initialization, data,
training and environment parity checks.

An Assist2017 AUC at or below0.8174 closes this candidate and related
projection/rank, prior, score, Fisher weighting, damping, normalization,
feedback, placement, initialization, width or seed retries. Retaining a
supported component is not permission to rerun a closed candidate.
No other dataset may train before a separately audited Assist2017 pass.

Before CUDA, verify both GPUs and every process owner. Only the already
authorized exact UID128/gdm pure-G display processes are exempt. Unknown
ownership, query failure or other-user GPU work blocks use. Keep the
unchanged global lock and two-second own-process-only watchdog.
