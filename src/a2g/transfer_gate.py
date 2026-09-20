"""V48 causal residual transport gate candidate."""

from .candidate import A2GModal


class A2GCausalTransferGate(A2GModal):
    """Separate cross-item concept residuals from same-item repetition."""

    CANDIDATE_ID = "a2g_v48_causal_transfer_gate_candidate"
