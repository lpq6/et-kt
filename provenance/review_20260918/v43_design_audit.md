# Finite Input Memory: Prospective Prototype

This is a structural hypothesis, not an admitted experiment or a claimed improvement.
Assist2017 remains the only executable dataset in any future separately frozen package.
The existing threshold remains strictly above 0.8174, with independent terminal audit.

## Operation

For the retained SSM projection `z_t = W*x_t + b`, form

`z'_t = z_t + sum_{lag=1..4} w_lag * z_(t-lag)`.

Multiplication is channel-wise. Out-of-window projected inputs are exactly zero.
Only the original, uncorrected projections enter this finite sum. Its output is not
fed back into the input-memory operator. The unchanged SSM tanh, sigmoid, softplus,
retention clipping, state recurrence, boundary read gate, normalization and dropout
then consume `z'_t`.

All current-position coefficients remain one. Four strictly earlier coefficients
are freely signed, zero initialized, and separately parameterized for each of the
candidate, output-gate and step-size channels. One tensor of shape `[768, 1, 4]`
adds 3,072 parameters at the existing width256; there is no added RNG use or dropout.
The initial forward equals V33, and disabling the module dispatches directly to
the original V33 SSM. The full forward and attention methods remain inherited.

## Why Examine It

The retained SSM computes each nonlinear proposal, output gate and retention rate
from the current input projection. Its recurrent state already contains older
information, and attention also has history; this proposal does not claim otherwise.
Explicit finite memory in the transition-driving variables introduces learned local
order-sensitive combinations before their nonlinearities, without using hidden-state
feedback or changing the main residual token. The error profile motivates checking
early window behavior but does not establish that missing local memory causes the
observed AUC gap or that this operation will help.

## Prior Implementations

The bounded source scan checked 395 Python files and preserved 32 relevant files.
It found existing causal convolution: no novelty claim about convolution is valid.

- Bounded cognitive smoothing filters the shared input token with positive,
  normalized width5 kernels and bounded mixing, adds dropout, and affects every
  downstream route. It has four historical Assist2015 variants. None is rerun.
- Robust smoothing uses dense width5 token convolution, a residual scale,
  normalization and dropout. Its jointly installed evidence attention gate is
  also not imported.
- V24 feeds the previous recurrent hidden state into the current projection.
  The proposed finite filter has no hidden-state input and does not recursively
  filter corrected projections. V24 and its closed retries remain excluded.
- V34 filters attention score matrices, not SSM input projections, and has no
  effect on the SSM computation. Its closed stencil variants remain excluded.
- V36 changes attention relative-lag bias; V37 conditions the SSM projection on
  the immediately observed response. Both remain resource-censored, with no retry
  or performance-failure reinterpretation.
- Historical multiscale SSM variants alter recurrent timescales, not the finite
  projected-input drive. Neither their coefficients nor their checkpoints are reused.

This is not an exact Mamba implementation or a claim of pedagogical causality.
Source differences establish a different executable hypothesis, not performance,
novelty, parameter-matched superiority or independence from adaptive selection.

## Controls And Limits

The retained structural control is independently trained V33. All common initial
parameters must be regenerated from seed42 and compared, never loaded to initialize
the new model. The one added tensor must be all zeros.

Removing SSM removes both uses of its state and the nested input-memory effect.
The other eight old-module interventions retain input memory. Removing attention
still also removes its nested FFN gates. These contrasts are not disjoint.

Only one future Full run is contemplated: train folds1-4, validation fold0,
seed42, length200, width256, FFN512, four attention layers/eight heads,
max200/patience20 and the unchanged optimizer/batch/dropout settings.
There is no test access, baseline or retained-control rerun, auxiliary loss,
checkpoint initialization, hyperparameter sweep, dataset expansion or automatic
resource retry. All of those restrictions still require enforcement in a newly
frozen execution package before training.

If a completed Full run fails the Assist2017 floor, this projected-input-memory
candidate and related kernel extent, lag offset, coefficient sign, normalization,
projection-group sharing, current-tap, location, gain, initialization, width and
seed retries close. A resource interruption is incomplete, not a scientific failure.
Neither this CPU prototype nor a future single-dataset pass completes the same-five-
of-eight goal or all independently retrained module requirements.
