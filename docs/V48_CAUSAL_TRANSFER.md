# V48 Causal Residual Transport Gate

V48 keeps the v46/v47 backbone and adds one optional evidence module:
`use_causal_transfer_gate`. The default is `0`; when disabled, the original
forward path and parameter set remain unchanged.

## Motivation

The same-concept history contains two different signals:

1. repeated attempts on the exact item, which belongs to the item-attempt path;
2. evidence that may transfer from a different item testing the same concept.

V48 tests the narrower hypothesis that only the second signal should be used
as an explicit residual correction. It does not add a relation graph, a
relative-position attention bias, an exponential forgetting kernel, or a
persistent learner state. Those mechanisms have close precedents and are not
claimed as new here.

## Formula

For target position `i`, `hist[:, i]` is the observed event at `i-1`. An event
`j` contributes only when `j <= i` in the shifted representation, `c_j = c_i`,
and known item IDs satisfy `q_j != q_i`. The history response is centered by a
train-fold prior:

```text
rho_j = r_j - sigmoid(item_prior[q_j] + concept_prior[c_j])
```

The causal weight separates recency from repeated-item duplication:

```text
w_ij = 1 / (i - j clipped to 1) / sqrt(1 + prefix_count(q_j))
```

The raw peer residual is:

```text
e_i = sum_j(w_ij * rho_j) / (1 + sum_j w_ij)
```

Let `n_i` be the strict-past count of the target item. The final transport
signal is:

```text
novelty_i = 1 / sqrt(1 + n_i)
```

The single module signal is:

```text
z_i = e_i * novelty_i
logit_i <- logit_i + tanh(gamma) * z_i
```

`gamma` is one learnable scalar initialized to `0.35`. No input channel,
embedding, hidden state, or auxiliary loss is added.

The denominator `1 + sum_j(w_ij)` is the sole evidence-mass calibration:
sparse prefixes are shrunk, and contradictory peer responses cancel in the
signed numerator. No separate coverage or agreement gate is applied, because
that would attenuate the same evidence twice.

## Causal and ablation contract

Current and future responses are never read. `hist_r` is the shifted response
sequence, and the diagonal pair is therefore the immediately preceding event,
not the current label. The matched control constructs the scalar in the same
initial state and disables only `use_causal_transfer_gate`; it is independently
retrained. A forward difference, finite gradient, or smoke result is not an
AUC claim.

The contribution claim is deliberately limited to an auditable causal
decomposition of same-item repetition versus cross-item concept residuals.
No first-publication or exclusivity claim is made.
