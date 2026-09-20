# V47 Transfer-Calibrated Residual

V47 is a development candidate layered on the v46 training protocol. It adds
one optional scalar gate, `use_transfer_residual`, to the existing concept
evidence slot. The default is `0`, so the disabled path remains the existing
v43/v46 computation.

For target position `i`, let `j < i` be a strictly earlier history position.
An event is eligible when its concept matches the target and either:

```text
q_i > 0, q_j > 0, q_i != q_j
```

or both item IDs are unavailable, in which case the method falls back to
concept-only transfer. The event contributes the prequential residual

```text
r_j - sigmoid(item_prior_qj + concept_prior_cj)
```

with inverse-distance weight and a causal repeated-item balance
`1 / sqrt(1 + number_of_prior_occurrences_of_qj)`. The aggregate is divided
by `1 + total_weight` and added to the existing evidence feature through
`tanh(transfer_gate)`.

This is a testable hypothesis about separating cross-item concept transfer from
same-item repetition. It is not a novelty claim: exponential temporal kernels,
attention over knowledge histories, and state-space knowledge tracing have
close precedents. A positive result requires a separately retrained matched
control (`no_transfer_residual`), not a smoke result or a post-hoc feature
correlation.

The path is causal by construction: current/future responses and current
responses are never read. Only the scalar `transfer_gate` is added when the
feature is enabled; no input width or legacy parameter tensor is changed.
