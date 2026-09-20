# V33 Prospective Input-Gated FFN

## Scope

Only Assist2017 Full, train folds1-4, validation fold0, seed42, sequence
width200. The strict admission floor remains0.8174. V32 is the completed
predecessor, structural parent, and highest observation, AUC0.8011094815652686.
It remains below the floor. Its graph and all seven earlier modules stay
unchanged. No trained checkpoint initializes the new model.

## Hypothesis

The retained attention stack has three ordinary Linear/GELU/Dropout/Linear/
Dropout FFNs; the fourth attention block has no FFN. A neuron currently
depends on its own affine preactivation before output mixing. Test whether
an independent projection of the same causal FFN input can select the
intermediate nonlinear features for each interaction.

For each retained FFN, preserve its original modules and use:

    a = GELU(W1 x + b1)
    g = 2 sigmoid(U x + v)
    output = Dropout2(W2 Dropout1(a * g) + b2)

U and v are zero-initialized, giving g=1 exactly at construction. They are
created with zeros_like and consume no random numbers. No dropout is added
or moved. All 105 V32 initial state tensors, caller RNG, initial predictions,
dropout traces and common initial gradients must remain identical. The new
full initial-state file is different because it contains six additional
tensors. At D256 and FFN width512, the predicted overhead is394752 parameters,
for5107701 total parameters and111 state tensors; executable checks must
confirm these counts.

The candidate inherits V32.forward and its attention method unchanged.
Only the existing FFN containers receive the gated forward operation.
The model flag propagates to every FFN wrapper. Disabling the gate delegates
each wrapper to its original Sequential computation even with nonzero
gate parameters.

## Distinction And Limits

The bounded historical scan examined341 Python sources and captured19
nearby implementations. Its FFN-gating pattern had no hits. This is not an
exhaustive historical or literature novelty claim.

- Historical token MoE adds a separately routed two-expert residual to the
  first FFN. It does not gate the existing intermediate neurons.
- Historical SDPA and query gates scale attention results. Here Q/K/V,
  attention probabilities and attention output projections stay unchanged.
- Historical depth routing combines whole layer states; branch scales are
  input-independent scalars. The proposed gate is input- and neuron-dependent.
- Historical spline activation modifies a univariate readout activation.
  The proposed gate can depend on an input direction absent from a neuron's
  original preactivation. This distinction has an explicit synthetic witness.
- Historical causal channel recalibration scales the completed attention
  stack using a prefix statistic. No such pooling is added here.

GLU-family feed-forward products are established prior work: Noam Shazeer,
"GLU Variants Improve Transformer", arXiv:2002.05202v1 (2020),
https://arxiv.org/abs/2002.05202v1. Primary bibliographic metadata and the
abstract were checked. No full-text-specific claim or reproduction of that
paper's exact architecture is made. This experiment tests a specific
identity-start product in A2G; it does not claim a newly invented GLU.

The V32 control has fewer parameters. A positive contrast would not by
itself isolate the gate topology from its added capacity. No parameter-
efficiency or parameter-matched superiority claim is registered.

## Causality And Ablations

Each gate uses only the FFN's existing causal input at the same position.
It has no persistent state, new metadata, current response, future response,
new auxiliary objective, normalization, graph update or sequence carry.
The existing per-segment width200 restriction remains.

Removing evidence, SSM, factorized input, item-attempt stage, history pace,
Newton or concept graph must leave the FFN gate active while removing the
specified old effect and gradient. The attention ablation removes the
whole attention/FFN branch and consequently its nested gate. This dependency
is explicit: the full/no-gate contrast isolates the new conditional
component, while no-attention is not a disjoint gate-independent contrast.
Zeroing attention outputs with the FFNs still present must not eliminate
the gate effect, distinguishing it from attention-output scaling.

The eventual same-five-of-eight goal requires nine independently trained
controls on the same qualifying datasets. Register family72, nine modules
times eight datasets. This package still permits only Assist2017 Full;
neither those72 comparisons nor any cross-dataset result is claimed.

## Execution And Decisions

One random-init run, unchanged Adam, learning rate, fixed batch order,
max200 and patience20. No baseline/control retraining, sweep, extra seed
or automatic resource retry. Check both GPUs and all process owners before
CUDA use and every two seconds during execution. Only the previously
authorized UID128/gdm pure-display services are exempt. Unknown ownership
or query failure stops only this task.

After observed zero process exit, independent CPU auditors replay the CSV,
keys, masks, training order, selected epoch, all metrics, source and artifact
hashes, V32 initialization parity, and the six zero gate tensors. Both
audits use10000 learner bootstrap draws. Intervals remain exploratory and
conditional on repeated architecture and validation-checkpoint selection.

AUC<=0.8174 closes this candidate and related FFN-gate activation, gain,
bias, sharing, rank, factorization, normalization, placement, initialization,
width and seed retries. Other datasets cannot rescue a failed Assist2017
pilot. A joint AUC/ACC point gate plus both positive AUC intervals can retain
the component as an exploratory structural base, not authorize a retry or
establish the full research objective.
