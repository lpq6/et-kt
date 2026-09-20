# V45 Unified Memory Kernel

Development candidate only. No file under `versions/` is edited or replaced.
The parent is v43, **not** the v44 aligned-history candidate. The original
history-statistics boundary, input memory, modal residual, graph checkpointing,
data, fold, seed, optimizer and stopping rules are preserved.

## Kernel and Direction

For the existing scalar concept-prior logit `p_c`:

```text
lambda_c = lambda0 * sigmoid(p_c)
K_c(gap) = exp(-lambda_c * gap)
```

`umk_lambda0` is a finite positive **configuration constant**, default `0.1`.
It is not a new parameter, embedding or buffer. The same constant and existing
concept-prior table are used in all three branches; gradients reach that table.
The prior is initialized using training folds only. It is not a student-specific
online mastery estimate.

The implementation follows the requested equations literally. With positive
`lambda0`, higher `p_c` increases `lambda_c`, so it causes **faster**, not slower,
forgetting. The requested SSM exponent also shortens retention initially.
Neither "high mastery forgets slowly" nor "longer SSM memory" follows from these
equations. Alpha and beta are unconstrained learned scalars; changing their
parameterization would be a separate model change.

## SSM: `use_umk_ssm=0`

```text
rate[b,t] = lambda0 * sigmoid(concept_prior[b,t])
decay_eff[b,t,d] = decay[b,t,d] ** (1 + alpha * rate[b,t])
alpha(initial) = 0.1
```

`SelectiveSSMBlock.forward` accepts optional `concept_prior [B,T]` after the
existing arguments. The original clamped retention is raised to this exponent
before the original update. Both v43 wrappers preserve the flag and the same
scalar parameter object. The modal branch derives its slower retentions from
this adjusted base retention, so the actual v43 execution path is covered.

`concept_prior` is required, shape/device/dtype checked when enabled, and ignored
when disabled. All hidden states remain `[B,D]`, outputs `[B,T,D]`.

## Attention: `use_umk_attn=0`

For query position `i` and the concept of key position `j`:

```text
last[i,j] = max { k < i : concept[k] == concept[j], concept[k] > 0 }
Delta[i,j] = i - last[i,j]
bias[i,j] = beta * log K_concept[j](Delta[i,j])
          = -beta * lambda_concept[j] * Delta[i,j]
beta(initial) = 0.5
score[i,j] = dot(Q[i], K[j]) / sqrt(head_dim) + bias[i,j]
```

This is **concept-relative last-seen distance**, not ordinary `i-j`. Repeated
keys for the same concept share the same gap at a given query. The prefix scan
uses O(B*T*T) memory/work and explicitly excludes occurrence `i`.

Keys `j>i`, padding concepts and keys with no prior occurrence are masked.
For an entirely empty row, a neutral self token is the sole fallback to avoid
all-negative-infinity softmax. The self token contains the current target
identity and observations only through `i-1`, never the current answer.
Concept ID1 retains the legacy shared OOV identity; this change does not claim
to distinguish different unseen concepts.

The bias is computed once per forward, broadcast to `[B*heads,T,T]`, and passed
as a floating additive mask to the existing `nn.MultiheadAttention`. No new
attention implementation, Q/K/V tensors, positional embedding, or layer-specific
parameter is introduced. `log K` is computed directly, avoiding exp underflow.
When disabled the original boolean causal mask and MHA call are preserved.

## RWCE: `use_umk_rwce=0`

The history arrays are shifted: `hist[:,j]` represents original event `j-1`.
The existing concept match, valid-response and inclusive shifted-history mask
are unchanged.

```text
distance = i - j + 1
weight_on[i,j]  = exp(-lambda_concept[i] * (distance - 1))
weight_off[i,j] = 1 / distance
evidence = sum(weight * (2*response - 1)) / (1 + sum(weight))
```

Thus the most recently observed event has unit weight. Only the weight changes;
exposure counts and the signed evidence reduction are unchanged. The optional
arguments extend the existing static/class method signatures. Enabled weights
use the prior dtype, including float64; disabled weights keep legacy float32
arithmetic exactly.

## Compatibility and Parameters

- All three constructor defaults are `0`, including `A2GUMK`.
- Configure flags when constructing the model; rebuild it when enabling a
  previously absent scalar, before constructing the optimizer.
- All off: state-dict keys, parameter order and RNG stream remain v43-compatible.
  A v43 checkpoint loads with `strict=True`; no missing-key waiver is used.
- SSM only adds the zero-dimensional `ssm.umk_alpha` parameter.
- Attention only adds the zero-dimensional `umk_beta` parameter.
- RWCE adds no parameters. All on adds two scalar values, total 5,114,871
  parameters for the supplied production configuration, versus 5,114,869.
- Enabled models require a matching enabled checkpoint for strict loading.
  Smoke and subsequent training start fresh; the old trained checkpoint is
  loaded only for the compatibility comparison.
- Existing `A2G`/v33 and `A2GModal`/v43 constructors accept the optional flags too.
  `architecture: "v45_umk"` selects the explicitly named v45 development class.

## Commands

Run from the project root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -B run.py smoke `
  --config configs\assist2017_v45_umk_ssm.json `
  --data-dir data\assist2017 --device cuda `
  --smoke-batch-size 64 --smoke-steps 2 --output work\new_ssm_smoke
```

Replace `ssm` with `attn`, `rwce`, `all`, or `off` for the corresponding saved
configuration. The production training contract still says batch64; the smoke
batch is explicitly recorded. Output directories must not already exist.

```powershell
.\.venv\Scripts\python.exe -B tools\verify_umk.py `
  --device cuda --smoke-batch-size 64 --output work\new_umk_verification
```

This launches isolated frozen/development probes with the real v43_lowmem
checkpoint, checks float32 over the complete validation population, checks
float64 predictions and fused features on two length-200 sequences, then runs
two optimizer steps for each single switch and all-on with CSV-bound audits.
It emits per-process commands, exit codes, stdout/stderr and `verification.json`.
No probe writes to the frozen version directory.

`tests/test_umk.py` additionally checks hand-computed equations, prefix gaps,
empty rows, wrapper fallback paths, current/future-answer isolation, future
concept/timestamp isolation, scalar-only additions, gradient propagation,
float64 checkpoint reload and all existing attention normalization topologies.

These checks establish implementation/compatibility, **not** an AUC improvement.
No v45 full training, positive independently retrained ablation, or AUC threshold
claim is made by these tests or smokes.
