import torch

from work.a2g_mambakt_evidence_equivalent_residual_candidate import (
    A2GMambaKT as _EvidenceA2GMambaKT,
)


class A2GMambaKT(_EvidenceA2GMambaKT):
    """Post-norm core with fold-local sparse item residual stabilization."""

    VALID_TOPOLOGIES = {"control", "attention_postnorm", "full_postnorm"}

    def __init__(
        self,
        *args,
        normalization_topology="full_postnorm",
        **kwargs,
    ):
        topology = str(normalization_topology)
        if topology not in self.VALID_TOPOLOGIES:
            raise ValueError(
                f"Unknown normalization topology {topology!r}; "
                f"expected one of {sorted(self.VALID_TOPOLOGIES)}"
            )
        self.normalization_topology = topology
        super().__init__(*args, **kwargs)

    def _attend(self, x):
        if self.normalization_topology == "control":
            return super()._attend(x)

        sequence_length = x.size(1)
        future = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            attended, _ = block["attn"](
                x,
                x,
                x,
                attn_mask=future,
                need_weights=False,
            )
            x = block["norm"](x + block["drop"](attended))
            if "ffn" not in block:
                continue
            if self.normalization_topology == "full_postnorm":
                x = block["ffn_norm"](x + block["ffn"](x))
            else:
                x = x + block["ffn"](block["ffn_norm"](x))
        return x


__all__ = ["A2GMambaKT"]
