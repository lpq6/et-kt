"""Deterministic module-keyed dropout substreams for paired A2G training."""

import hashlib

import torch
from torch import nn


class DropoutSubstreamError(RuntimeError):
    """Raised when deterministic stochastic-module boundaries are violated."""


def _is_stochastic_module(name, module):
    if isinstance(module, (nn.Dropout, nn.MultiheadAttention)):
        return True
    return name == "input.3" and hasattr(module, "warmup_active") and hasattr(
        module, "p"
    )


class A2GDropoutSubstreamAlignment:
    """Reset RNG before each shared stochastic module at one training step."""

    SCHEMA_VERSION = 1

    def __init__(self, model, *, base_seed):
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise ValueError("base_seed must be an integer")
        self.model = model
        self.base_seed = base_seed
        self._modules = tuple(
            (name, module)
            for name, module in model.named_modules()
            if _is_stochastic_module(name, module)
        )
        if not self._modules:
            raise DropoutSubstreamError("model has no bound stochastic modules")
        names = [name for name, _ in self._modules]
        if len(names) != len(set(names)):
            raise DropoutSubstreamError("stochastic module names are not unique")
        self._handles = [
            module.register_forward_pre_hook(self._make_hook(name))
            for name, module in self._modules
        ]
        self._active = False
        self._epoch = None
        self._cursor = None
        self._counts = {}

    @property
    def module_names(self):
        return tuple(name for name, _ in self._modules)

    def _seed(self, name, call_index):
        payload = (
            f"{self.base_seed}|{self._epoch}|{self._cursor}|{name}|{call_index}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
            (1 << 63) - 1
        )

    def _make_hook(self, name):
        def hook(_module, _inputs):
            if not self._active:
                raise DropoutSubstreamError(
                    f"stochastic module {name!r} ran outside an aligned step"
                )
            call_index = self._counts.get(name, 0)
            self._counts[name] = call_index + 1
            torch.manual_seed(self._seed(name, call_index))

        return hook

    def begin_step(self, *, epoch, cursor):
        if self._active:
            raise DropoutSubstreamError("a dropout-substream step is already active")
        for value, name in ((epoch, "epoch"), (cursor, "cursor")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._epoch = epoch
        self._cursor = cursor
        self._counts = {}
        self._active = True
        torch.manual_seed(self._seed("__step_entry__", 0))

    def end_step(self):
        if not self._active:
            raise DropoutSubstreamError("no dropout-substream step is active")
        missing = sorted(set(self.module_names).difference(self._counts))
        if missing:
            self._active = False
            raise DropoutSubstreamError(
                f"bound stochastic modules were not called: {missing}"
            )
        trace = {
            "epoch": self._epoch,
            "cursor": self._cursor,
            "module_call_counts": {
                name: self._counts[name] for name in self.module_names
            },
        }
        self._active = False
        return trace

    def abort_step(self):
        if not self._active:
            raise DropoutSubstreamError("no dropout-substream step is active")
        self._active = False
        self._counts = {}

    def state_dict(self):
        if self._active:
            raise DropoutSubstreamError("cannot serialize an active substream step")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "base_seed": self.base_seed,
            "module_names": list(self.module_names),
            "seed_key": "epoch|cursor|module_name|call_index",
            "algorithm": "sha256-first-63-bits",
            "cuda_contract_complete": False,
        }

    def close(self):
        if self._active:
            raise DropoutSubstreamError("cannot close an active substream step")
        for handle in self._handles:
            handle.remove()
        self._handles = []


__all__ = ["A2GDropoutSubstreamAlignment", "DropoutSubstreamError"]
