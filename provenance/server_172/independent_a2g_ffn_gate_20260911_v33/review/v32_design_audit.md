# V32 Prospective Directed Concept-Graph Recurrence

## Research Scope

Only Assist2017 Full, train folds1-4, validation fold0, seed42, width200.
The strict admission floor remains 0.8174. V28 is both the structural parent
and highest observed result, AUC0.800317550319877. V31 is the closed immediate
predecessor, AUC0.8000950827746682; none of its Gaussian code is retained.
One pilot cannot establish the complete same-five-of-eight objective.

## Structural Hypothesis

The preserved trunk has one mixed event trajectory. Its current concept
evidence uses outcome statistics, while the closed V20 filter evolves each
concept independently. A distinct hypothesis is that a learner's response
at one concept can update a separate latent state at another concept, with
later propagation depending on the source concept's accumulated state.

Keep all seven V28 modules. Before its unchanged SSM, add a zero-start
projection of a learner-local concept-graph state. The graph uses existing
historical concept embeddings, restricted to 2<=ID<declared capacity. The
parent's spare final embedding row is excluded: Assist2017 capacity96
contains 94 known concepts, not 95. Independently
learned source/destination projections produce directed dot-product scores.
Each row's off-diagonal softmax sums to one; the self weight is exactly one.
These are learned computational relations, not verified prerequisites.

For shifted observed event c at column t, after forming the original causal
history embedding:

    u = tanh(W_event LN(history[t]) + b_event + W_state h[c])
    proposed[j] = PyTorch.GRUCell(u, h[j])
    h[j] = (1 - weight[c,j]) * h[j] + weight[c,j] * proposed[j]
    token[t] = original_token[t] + W_output h[target_concept[t]]

States start at zero on every forward call. Padding, unknown concepts and
unobserved responses do not update the graph. Padding/unknown target reads
are exactly zero. The event at column t is event t-1, never response t.
The self update is full; neighbors receive bounded interpolated updates.
State width equals one existing attention head, D/H, with no searched rank.

The graph output projection is zero-initialized; its other nine tensors
use the standard PyTorch constructors within a CPU fork_rng context.
At D256/H8 it adds ten tensors and 40,160 parameters, for 4,712,949 total
parameters and 105 state tensors. These counts require executable checks.
All 95 common initial tensors, constructor RNG, initial function, common
gradients and original dropout trace must match V28. Entire initial weight
files differ and are not required to match.

The new module is independent of evidence, SSM and attention removal.
Every original removal must still remove its own effect and gradient.
Disabling the graph returns the original V28 forward without interception.
With the graph active, disabling Newton must leave the graph active.

## Bounded Historical Distinction

A read-only scan captured 333 Python sources from the remote root, work/
and depth-two candidate files, plus 16 nearby implementations. It found
no explicit adjacency/message-passing/graph-state implementation in this
bounded scope. This is neither an exhaustive review nor a novelty claim.

- V20 is a per-concept two-state filter with identity updates on other skills.
- V29 mixes off-diagonal prefix outcome counts at the readout. It has no
  ordered, coupled source/destination learner states or GRU transition.
- V22 changes attention key/value alignment, not concept-state recurrence.
- CRIC and V14 relate event/role features, not a learner-local node-state graph.
- Concept experts transform an already computed SSM state pointwise.

Required witnesses include order dependence with identical outcome counts,
influence on an unpracticed target only through graph edges, and an ordered
two-hop propagation example. These witnesses establish computational
distinction, not pedagogical transfer or predictive benefit.

## Verification And Execution Gates

Use an independent scalar GRU/graph recurrence oracle and gradient checks.
Verify source/destination orientation, interpolation, invalid-event identity,
known-ID boundaries, learner isolation, no persistent memory, per-prefix
causality, production-width initial parity and every retained ablation.
Independently reconstruct all ten added tensors from seed42 and bound V28
source; never initialize from a trained checkpoint.

One independently initialized max200/patience20 run, with unchanged Adam,
learning rate, batch order and validation selection. No baseline/control
retraining, sweep, extra seed or automatic retry. Both GPUs and all process
owners must pass the resource policy before GPU use, then every two seconds.
Query failure or an unknown owner stops only this job.

Two independent CPU audits use 10,000 learner bootstrap draws. The registered
family remains eight modules times eight datasets, not an assertion that
these 64 comparisons ran. Intervals remain exploratory, conditional on one
seed and adaptive architecture/checkpoint selection.

Failure to exceed0.8174 closes this graph-recurrence candidate and related
edge orientation/normalization, self/neighbor weighting, graph sparsity,
state rank, recurrent-cell/gate, message, readout, placement, initialization,
width or seed retries. Other datasets cannot rescue an Assist2017 failure.

## Unfrozen Capacity Correction

The first 42 small CPU tests passed but did not exercise the declared
vocabulary boundary. A subsequent config/source inspection found that the
parent allocates capacity+1 embedding rows. Preserve that initial test
report and source snapshot as development evidence. Before any freeze,
slice the history table to capacity and add regressions for node count,
spare-row prediction invariance and zero spare-row gradient. This changes
no trainable tensors, initialization or experimental hyperparameters.
