"""The audited V43 source candidate, integrated with the consolidated A2G."""

from .model import A2G
from .modules.input_memory import InputMemorySSM
from .modules.modal import MultiModeResidualInputMemorySSM


class A2GModal(A2G):
    CANDIDATE_ID = "a2g_v43_local_candidate"

    def __init__(self, *args, use_input_memory=1, use_multimode_residual=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssm = InputMemorySSM.from_existing(self.ssm, bool(use_input_memory))
        self.ssm = MultiModeResidualInputMemorySSM.from_existing(
            self.ssm,
            mode_count=self.blocks[0]["attn"].num_heads,
            enabled=bool(use_multimode_residual),
        )

    @property
    def use_input_memory(self):
        return self.ssm.use_input_memory

    @use_input_memory.setter
    def use_input_memory(self, value):
        self.ssm.use_input_memory = bool(value)

    @property
    def use_multimode_residual(self):
        return self.ssm.use_multimode_residual

    @use_multimode_residual.setter
    def use_multimode_residual(self, value):
        self.ssm.use_multimode_residual = bool(value)
