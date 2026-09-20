"""CPU/CUDA behavior contracts for finite, strictly earlier SSM input memory."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
from modal_residual_candidate import A2GMambaKT as Candidate, MultiModeResidualInputMemorySSM
from input_memory_candidate import CausalInputMemory, InputMemorySSM
from ffn_gate_candidate import A2GMambaKT as V33
from test_input_memory import (
    BRANCHES, TOLERANCE, activate_model, activate_v33, recurrence_oracle, scalar_memory,
)
from test_input_memory_production import make_batch, parent_kwargs
from test_modal_residual import MODAL_NAMES, activate_modal
from v19_audit_helpers import aligned, build, clone
from v27_audit_helpers import timed_batch
from v28_audit_helpers import has_gradient
from v33_audit_helpers import audit as audit_v33
from work.a2g_mambakt_promoted_no_duplicate_folded_bias_20260715 import SelectiveSSMBlock


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def close(left, right):
    return torch.allclose(left, right, **TOLERANCE)


def gradients_match(parent, candidate):
    actual = dict(candidate.named_parameters())
    return all(
        actual[name].grad is None if value.grad is None else (
            actual[name].grad is not None and bool(torch.isfinite(actual[name].grad).all())
            and close(value.grad, actual[name].grad)
        ) for name, value in parent.named_parameters()
    )


def equation_checks(device, checks):
    torch.manual_seed(42)
    projected = torch.randn(2, 9, 6, dtype=torch.float64, device=device, requires_grad=True)
    memory = CausalInputMemory(6, dtype=torch.float64, device=device)
    with torch.no_grad():
        memory.weight.copy_(torch.linspace(-0.3, 0.5, 24, device=device, dtype=torch.float64).reshape_as(memory.weight))
    actual = memory(projected)
    expected = scalar_memory(projected, memory.weight)
    checks["input_memory_matches_independent_four_lag_scalar_equation"] = torch.allclose(
        actual, expected, atol=1e-12, rtol=1e-12,
    )
    factors = torch.randn_like(actual)
    left = torch.autograd.grad((actual * factors).sum(), (projected, memory.weight))
    right = torch.autograd.grad((expected * factors).sum(), (projected, memory.weight))
    checks["input_memory_input_and_weight_gradients_match_scalar_oracle"] = all(
        torch.allclose(observed, reference, atol=1e-11, rtol=1e-11)
        for observed, reference in zip(left, right)
    )
    checks["input_memory_first_position_exact_zero"] = bool(actual[:, 0].eq(0).all())
    changed = projected.detach().clone()
    changed[:, 4:] *= 13
    altered = memory(changed)
    checks["input_memory_excludes_current_and_future_projection"] = (
        torch.equal(actual[:, :5], altered[:, :5]) and not torch.equal(actual[:, 5:], altered[:, 5:])
    )
    memory(projected)[0, 7].sum().backward()
    checks["input_memory_exact_four_earlier_inputs_and_learner_gradient_scope"] = (
        bool(projected.grad[1].eq(0).all()) and bool(projected.grad[0, :3].eq(0).all())
        and bool(projected.grad[0, 3:7].ne(0).all()) and bool(projected.grad[0, 7:].eq(0).all())
    )
    impulse = projected.detach().new_zeros(1, 12, 6)
    impulse[:, 0] = 1
    output = memory(impulse)
    checks["input_memory_never_recursively_filters_corrected_projection"] = (
        bool(output[:, 1:5].ne(0).all()) and bool(output[:, 5:].eq(0).all())
    )
    checks["input_memory_operator_prefix_consistency"] = all(
        torch.allclose(actual[:, :length], memory(projected[:, :length]), atol=1e-12, rtol=1e-12)
        for length in (1, 2, 4, 6, 8)
    )
    changed = projected.detach().clone()
    changed[1, :, 2] += 17
    altered = memory(changed)
    checks["input_memory_channels_and_learners_isolated"] = (
        torch.equal(actual[0], altered[0])
        and torch.equal(actual[1, :, [0, 1, 3, 4, 5]], altered[1, :, [0, 1, 3, 4, 5]])
    )
    before = torch.get_rng_state().clone()
    empty = CausalInputMemory(6, dtype=torch.float64, device=device)
    checks["input_memory_zero_initialization_consumes_no_rng"] = torch.equal(before, torch.get_rng_state())
    projected.grad = None
    empty(projected).sum().backward()
    checks["input_memory_zero_weights_get_gradient_without_input_gradient"] = (
        bool(empty.weight.grad.ne(0).all()) and bool(projected.grad.eq(0).all())
    )
    checks["input_memory_has_one_parameter_and_no_dropout_or_buffers"] = (
        list(dict(memory.named_parameters())) == ["weight"] and not list(memory.buffers())
        and not any(isinstance(module, nn.Dropout) for module in memory.modules())
    )
    orientation = []
    for lag in range(1, 5):
        with torch.no_grad():
            empty.weight.zero_()
            empty.weight[:, 0, 4 - lag] = 1
        output = empty(projected)
        orientation.append(bool(output[:, :lag].eq(0).all()) and torch.equal(output[:, lag:], projected[:, :-lag]))
    checks["input_memory_weight_indices_read_lags_four_to_one"] = all(orientation)
    with torch.no_grad():
        empty.weight.zero_()
        empty.weight[:, 0, 3] = 1
        empty.weight[:, 0, 2] = -1
    checks["input_memory_can_represent_signed_differences_not_convex_smoothing"] = (
        bool(empty(torch.ones_like(projected))[:, 2:].eq(0).all())
        and torch.equal(empty(projected)[:, 2:], projected[:, 1:-1] - projected[:, :-2])
    )
    original = SelectiveSSMBlock(8, dropout=0.2).to(device=device, dtype=torch.float64).eval()
    recurrent = InputMemorySSM.from_existing(copy.deepcopy(original)).eval()
    inputs = torch.randn(2, 9, 8, device=device, dtype=torch.float64, requires_grad=True)
    scope = torch.rand_like(inputs)
    checks["input_memory_zero_recurrence_exact_original_with_scope"] = torch.equal(
        original(inputs, scope), recurrent(inputs, scope),
    )
    with torch.no_grad():
        recurrent.input_memory.weight.copy_(
            torch.linspace(-0.12, 0.17, 96, device=device, dtype=torch.float64).reshape(24, 1, 4),
        )
    actual = recurrent(inputs, scope)
    expected = recurrence_oracle(recurrent, inputs, scope)
    checks["input_memory_recurrence_matches_independent_oracle"] = torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    factors = torch.randn_like(actual)
    parameters = (inputs, *recurrent.parameters())
    left = torch.autograd.grad((actual * factors).sum(), parameters)
    right = torch.autograd.grad((expected * factors).sum(), parameters)
    checks["input_memory_recurrence_all_gradients_match_independent_oracle"] = all(
        torch.allclose(observed, reference, atol=1e-10, rtol=1e-10)
        for observed, reference in zip(left, right)
    )
    checks["input_memory_scope_zero_keeps_original_direct_path"] = torch.equal(
        recurrent(inputs, torch.zeros_like(scope)), original(inputs, torch.zeros_like(scope)),
    )
    checks["input_memory_recurrence_first_position_unchanged"] = torch.equal(recurrent(inputs)[:, 0], original(inputs)[:, 0])
    checks["input_memory_recurrence_has_effect_after_first_position"] = not torch.equal(
        recurrent(inputs)[:, 1:], original(inputs)[:, 1:],
    )
    checks["input_memory_recurrence_prefix_consistency"] = all(
        torch.allclose(actual[:, :length], recurrent(inputs[:, :length], scope[:, :length]), atol=1e-12, rtol=1e-12)
        for length in (1, 3, 5, 8)
    )
    changed = inputs.detach().clone()
    changed[1] += 100
    changed[0, 5:] -= 100
    checks["input_memory_recurrence_causal_and_call_isolation"] = (
        torch.equal(actual[0, :5], recurrent(changed, scope)[0, :5])
        and torch.equal(actual, recurrent(inputs, scope))
    )
    recurrent.use_input_memory = False
    recurrent.zero_grad(set_to_none=True)
    disabled = recurrent(inputs, scope)
    disabled.sum().backward()
    checks["input_memory_disabled_recurrence_exact_original_and_no_new_gradient"] = (
        torch.equal(disabled, original(inputs, scope)) and recurrent.input_memory.weight.grad is None
    )


def model_checks(device, checks, *, production):
    prefix = "production_" if production else "small_"

    def make(cls):
        if not production:
            return build(cls, device)
        torch.manual_seed(42)
        return cls(96, 3109, **parent_kwargs()).to(device)

    parent = make(V33)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    candidate = make(Candidate)
    checks[prefix + "common_initial_tensors_and_cpu_rng_match_v33"] = (
        torch.equal(cpu_rng, torch.get_rng_state())
        and all(torch.equal(value, candidate.state_dict()[name]) for name, value in parent.state_dict().items())
    )
    if device.type == "cuda":
        checks[prefix + "initial_cuda_rng_matches_v33"] = torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    checks[prefix + "zero_modal_residual_tensors_added"] = (
        sorted(set(candidate.state_dict()) - set(parent.state_dict())) == MODAL_NAMES
        and candidate.ssm.input_memory.weight.shape == (3 * candidate.d_model, 1, 4)
        and bool(candidate.ssm.input_memory.weight.eq(0).all())
        and candidate.ssm.mode_input_weight.shape == (
            candidate.blocks[0]["attn"].num_heads, candidate.d_model
        )
        and candidate.ssm.mode_read_weight.shape == (
            candidate.blocks[0]["attn"].num_heads, candidate.d_model
        )
        and bool(candidate.ssm.mode_input_weight.eq(0).all())
        and bool(candidate.ssm.mode_read_weight.eq(0).all())
    )
    checks[prefix + "forward_attention_statistics_and_input_path_inherited"] = all(
        getattr(Candidate, method) is getattr(V33, method)
        for method in ("forward", "_attend", "_stats", "_sequences", "_boundary_weights", "_factorized_token", "_modulate_history")
    )
    checks[prefix + "original_nonssm_module_types_preserved"] = all(
        type(getattr(candidate, name)) is type(getattr(parent, name))
        for name in ("blocks", "concept_graph", "prequential_newton", "input", "pred")
    ) and all(type(block["attn"]) is nn.MultiheadAttention for block in candidate.blocks)
    checks[prefix + "ssm_wrapped_with_original_projection_normalization_dropout"] = (
        type(parent.ssm) is SelectiveSSMBlock
        and type(candidate.ssm) is MultiModeResidualInputMemorySSM
        and all(type(getattr(candidate.ssm, name)) is type(getattr(parent.ssm, name))
                for name in ("in_proj", "norm", "drop"))
    )
    checks[prefix + "original_dropout_module_names_types_and_counts_unchanged"] = [
        (name, type(value)) for name, value in candidate.named_modules() if isinstance(value, nn.Dropout)
    ] == [(name, type(value)) for name, value in parent.named_modules() if isinstance(value, nn.Dropout)]
    if production:
        checks["production_parameters5114869_new7168_state_tensors114"] = (
            sum(value.numel() for value in candidate.parameters()) == 5114869
            and candidate.ssm.input_memory.weight.numel() == 3072
            and candidate.ssm.mode_input_weight.numel() == 2048
            and candidate.ssm.mode_read_weight.numel() == 2048
            and len(parent.state_dict()) == 111 and len(candidate.state_dict()) == 114
        )
    batch = make_batch(device) if production else timed_batch(device)
    parent.eval()
    candidate.eval()
    with torch.no_grad():
        checks[prefix + "initial_eval_matches_v33"] = close(parent(batch), candidate(batch))
    parent.train()
    candidate.train()
    old, old_trace = aligned(parent, batch)
    new, new_trace = aligned(candidate, batch)
    checks[prefix + "initial_train_matches_v33"] = close(old, new)
    checks[prefix + "initial_dropout_trace_exact_v33"] = old_trace == new_trace
    for prediction in (old, new):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks[prefix + "initial_common_gradients_match_v33"] = gradients_match(parent, candidate)
    gradient = candidate.ssm.input_memory.weight.grad
    checks[prefix + "every_new_coefficient_receives_finite_initial_gradient"] = (
        gradient is not None and bool(torch.isfinite(gradient).all()) and bool(gradient.ne(0).all())
    )
    for model in (parent, candidate):
        activate_v33(model)
        model.zero_grad(set_to_none=True)
    activate_model(candidate)
    activate_modal(candidate)
    active, active_trace = aligned(candidate, batch)
    F.binary_cross_entropy(active[:, 1:], batch["shft_rseqs"].float()).backward()
    checks[prefix + "active_new_and_retained_modules_receive_finite_gradients"] = (
        has_gradient(candidate, tuple(MODAL_NAMES))
        and all(has_gradient(candidate, names) for _, names, _ in BRANCHES)
        and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in candidate.parameters())
    )
    checks[prefix + "active_dropout_trace_exact_v33"] = active_trace == old_trace
    parent.eval()
    candidate.eval()
    with torch.no_grad():
        prediction = candidate(batch)
        checks[prefix + "active_memory_changes_predictions"] = not torch.equal(prediction, parent(batch))
        checks[prefix + "finite_predictions_with_original_shape_and_range"] = (
            prediction.shape == (batch["rseqs"].size(0), batch["rseqs"].size(1) + 1)
            and bool(torch.isfinite(prediction).all()) and bool(((prediction > 0) & (prediction < 1)).all())
        )
        safe = []
        boundaries = (0, 1, 2, 49, 98, 149, 198) if production else range(batch["rseqs"].size(1))
        for boundary in boundaries:
            for key in ("rseqs", "tseqs"):
                changed = clone(batch)
                if key == "rseqs":
                    changed[key][:, boundary:] = 1 - changed[key][:, boundary:]
                else:
                    changed[key][:, boundary:] += 123456789
                safe.append(torch.equal(prediction[:, :boundary + 1], candidate(changed)[:, :boundary + 1]))
        checks[prefix + "current_future_response_and_time_boundaries_safe"] = all(safe)
        changed = clone(batch)
        changed["shft_rseqs"] = 1 - changed["shft_rseqs"]
        for key in ("shft_tseqs", "utseqs", "shft_utseqs"):
            changed[key].fill_(123456789)
        checks[prefix + "shifted_labels_current_times_and_durations_unused"] = torch.equal(prediction, candidate(changed))
        changed = clone(batch)
        changed["rseqs"][1] = 1 - changed["rseqs"][1]
        checks[prefix + "learner_and_call_isolation"] = (
            torch.equal(prediction[0], candidate(changed)[0]) and torch.equal(prediction, candidate(batch))
        )
        changed = clone(batch)
        boundary = 150 if production else 5
        for key in ("qseqs", "cseqs", "shft_qseqs", "shft_cseqs"):
            changed[key][:, boundary:] = 0
        padded = candidate(changed)
        checks[prefix + "future_padding_cannot_change_earlier_predictions"] = (
            bool(torch.isfinite(padded).all()) and torch.equal(prediction[:, :boundary], padded[:, :boundary])
        )
        for group in range(3):
            altered = copy.deepcopy(candidate)
            altered.ssm.input_memory.weight[
                group * candidate.d_model:(group + 1) * candidate.d_model
            ].zero_()
            checks[prefix + f"projection_group_{group}_has_effect"] = not torch.equal(prediction, altered(batch))
    for label, _, flag in BRANCHES:
        left, right = copy.deepcopy(parent), copy.deepcopy(candidate)
        left.train()
        right.train()
        setattr(left, flag, False)
        setattr(right, flag, False)
        right.zero_grad(set_to_none=True)
        expected, before = aligned(left, batch)
        actual, after = aligned(right, batch)
        F.binary_cross_entropy(actual[:, 1:], batch["shft_rseqs"].float()).backward()
        if flag == "use_ssm_branch":
            checks[prefix + "no_ssm_removes_input_memory_effect_and_gradient"] = (
                torch.equal(expected, actual)
                and all(
                    dict(right.named_parameters())[name].grad is None
                    for name in MODAL_NAMES
                )
            )
        else:
            checks[prefix + f"no_{label}_retains_input_memory_effect_and_gradient"] = (
                not torch.equal(expected, actual) and has_gradient(right, tuple(MODAL_NAMES))
            )
        if flag == "use_attention_branch":
            checks[prefix + "no_attention_also_removes_nested_ffn_gate_gradients"] = all(
                parameter.grad is None for name, parameter in right.named_parameters() if ".ffn.gate_" in name
            )
        checks[prefix + f"no_{label}_preserves_original_dropout_trace"] = before == after
    candidate.use_input_memory = False
    candidate.use_multimode_residual = False
    parent.train()
    candidate.train()
    for model in (parent, candidate):
        model.zero_grad(set_to_none=True)
    expected, before = aligned(parent, batch)
    actual, after = aligned(candidate, batch)
    expected.sum().backward()
    actual.sum().backward()
    checks[prefix + "disabled_nonzero_memory_recovers_v33_and_common_gradients"] = (
        torch.equal(expected, actual) and before == after and gradients_match(parent, candidate)
        and all(
            dict(candidate.named_parameters())[name].grad is None
            for name in MODAL_NAMES
        )
    )


def audit(device_name="cpu"):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        from gpu_guard import require_available
        require_available()
    retained = audit_v33(device_name)
    if retained["status"] != "pass":
        raise RuntimeError("retained V33 behavior contract failed")
    checks = {}
    equation_checks(device, checks)
    model_checks(device, checks, production=False)
    model_checks(device, checks, production=True)
    if device.type == "cpu":
        checks["cpu_audit_did_not_initialize_cuda"] = not torch.cuda.is_initialized()
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "fail", "candidate_id": Candidate.CANDIDATE_ID,
        "check_count": len(checks), "checks": checks, "retained_v33_contract": retained,
        "candidate_sha256": digest("modal_residual_candidate.py"),
        "v33_source_sha256": digest("ffn_gate_candidate.py"), "incumbent_sha256": digest("a2g_incumbent.py"),
        "new_parameter_names": MODAL_NAMES, "active_initial_function_equivalent_to_v33": True,
        "ssm_input_memory_ablations_are_nested": True,
        "common_initial_outputs_bit_exact_claimed": False, "common_initial_output_tolerance": TOLERANCE,
        "common_initial_gradients_bit_exact_claimed": False, "common_initial_gradient_tolerance": TOLERANCE,
        "gpu_used": device.type == "cuda", "real_data_access": False, "optimizer_steps": 0,
        "training_started": False, "paper_goal_complete": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("behavior report already exists")
    report = audit(args.device)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
