# V45 Independent Control Preparation

This is a non-executable preparation record, not permission to bypass the
Assist2017 Full admission gate. No independent control training has started.

The active Full run uses the frozen `candidates/v45_umk_local` snapshot.
Subsequent development of control construction does not change that snapshot.
The development factory now constructs all UMK parameters before disabling a
requested UMK path. Consequently all controls retain the full 116-tensor
initial-state layout, while disabled scalars receive no gradient. Ordinary
all-off construction still omits the new scalars and strictly loads v43.

## Planned Contrasts

| Control | Directly Disabled Path |
| --- | --- |
| no_evidence | Priors, statistical/direct evidence and support scaling |
| no_ssm | SSM output, including its input memory, modal residual and UMK retention |
| no_attention | Attention output, including FFN gates and UMK attention bias |
| no_factorized_input | Factorized residual input |
| no_item_attempt_stage | Item attempt readout |
| no_history_pace | Historical timestamp modulation |
| no_prequential_newton | Online Newton correction |
| no_concept_graph | Directed concept graph |
| no_ffn_gate | Attention FFN hidden gates |
| no_input_memory | Finite projected-input memory |
| no_multimode_residual | Multimode SSM residual |
| no_umk_ssm | Concept-modulated SSM retention |
| no_umk_attn | Concept-relative attention mask/bias |
| no_umk_rwce | Exponential RWCE weighting, restoring reciprocal distance |
| no_transfer_residual | Prior-calibrated cross-item residual evidence |

These interventions overlap. Their effects must not be added together.
`no_umk_attn` restores the complete legacy attention path, including its
self-token visibility; the contrast cannot attribute any gain solely to beta.
RWCE remains part of the evidence module; `no_umk_rwce` tests the replacement
weighting rule, not the existence of all statistical evidence.

## Required Before Execution

1. The frozen v45 Full run completes normally and its audited Assist2017 AUC
   strictly exceeds 0.8174.
2. A separate control execution snapshot is frozen outside `versions/`.
   Its Full behavior, initial state and stochastic substreams must match the
   active Full snapshot. The current frozen snapshot is not edited in place.
3. Each dataset gets an independently verified CSV, mapping, student-isolation,
   population and resource contract. No test split is silently introduced.
4. Every Full/control run starts from the same freshly generated initial state
   and a new optimizer, never from a trained Full checkpoint.
5. Preserve all 15 controls across the original eight candidate datasets.
   The planned family is 120 comparisons, not the obsolete v43 family of 88.
   Missing or failed comparisons cannot be removed from the family post hoc.
6. On the same set of at least five datasets, Full must exceed the fixed
   historical numeric floor and every matched independently retrained control.
   Report student-cluster paired uncertainty and multiplicity-adjusted intervals.
7. Historical five-fold means remain descriptive references, not matched
   baseline experiments. Their comparability must be resolved separately.

Assist2015 and Statics2011 lack item IDs in the source metadata, so the current
item-attempt module cannot demonstrate a strictly positive contribution there.
This applicability limitation is not a passed ablation or a reason to lower
the original eight-dataset requirement.

The current entrypoint still rejects non-Full `train --variant` invocations.
Tests in `test_umk_controls.py` verify construction and intervention semantics
only; they do not grant admission or establish positive AUC differences.
