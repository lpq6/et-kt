"""Synthetic v12 parity, branch isolation and causal checks; no optimizer."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
from a2g_dropout_substream_alignment_20260807 import A2GDropoutSubstreamAlignment
from factorized_input_candidate import A2GMambaKT as V12
from attempt_stage_candidate import A2GMambaKT as Candidate, item_attempt_indices
from test_attempt_stage import oracle_checks


def make_batch(device):
    questions = torch.tensor([[2, 2, 2, 2, 3, 2, 2, 4], [3, 3, 4, 3, 3, 3, 4, 4]], device=device)
    concepts = torch.tensor([[2, 2, 2, 2, 3, 2, 2, 3], [3, 3, 3, 3, 3, 3, 3, 3]], device=device)
    responses = torch.tensor([[0, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1]], device=device)
    return {
        "qseqs": questions[:, :-1], "shft_qseqs": questions[:, 1:],
        "cseqs": concepts[:, :-1], "shft_cseqs": concepts[:, 1:],
        "rseqs": responses[:, :-1], "shft_rseqs": responses[:, 1:],
        "smasks": torch.ones_like(responses[:, 1:], dtype=torch.bool),
    }


def clone(batch):
    return {key: value.clone() for key, value in batch.items()}


def build(cls, device, n_pid=9, **flags):
    torch.manual_seed(42)
    model = cls(
        6, n_pid, d_model=32, d_ff=64, n_blocks=4, num_attn_heads=4,
        dropout=0.2, seq_len=200, prior_scale=0.08, stat_scale=0.12,
        input_decay_scale=0.5, boundary_local_floor=0.5, use_split_boundary=1,
        normalization_topology="full_postnorm", use_evidence_equivalent_residual=1,
        support_smoothing=24.0, use_recency_weighted_concept_evidence=1,
        item_residual_dropout=0.4, item_residual_dropout_schedule="constant", **flags,
    ).to(device)
    with torch.no_grad():
        model.item_support_count.fill_(64)
        model.item_prior.weight.fill_(0.15)
        model.concept_prior.weight.fill_(-0.07)
    return model


def aligned(model, batch):
    align = A2GDropoutSubstreamAlignment(model, base_seed=42)
    try:
        align.begin_step(epoch=0, cursor=0)
        prediction = model(batch, train=True)[0]
        trace = align.end_step()
        return prediction, trace
    finally:
        if align._active:
            align.abort_step()
        align.close()


def forward_scope_matches_v12():
    def function(name):
        tree = ast.parse((ROOT / name).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "A2GMambaKT")
        return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")

    base, candidate = function("factorized_input_candidate.py"), function("attempt_stage_candidate.py")
    guard = candidate.body.pop(0)
    if not (
        isinstance(guard, ast.If) and isinstance(guard.test, ast.UnaryOp)
        and isinstance(guard.test.operand, ast.Attribute)
        and guard.test.operand.attr == "use_item_attempt_stage"
    ):
        return False
    removed = 0
    for node in candidate.body:
        if isinstance(node, ast.Try):
            keep = []
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and isinstance(statement.value, ast.BinOp)
                    and isinstance(statement.value.right, ast.Call)
                    and isinstance(statement.value.right.func, ast.Attribute)
                    and statement.value.right.func.attr == "attempt_readout"
                ):
                    removed += 1
                else:
                    keep.append(statement)
            node.body = keep
    return removed == 1 and ast.dump(base) == ast.dump(candidate)


@torch.no_grad()
def activate(model):
    weight = model.attempt_readout.output.weight
    weight.copy_(0.1 * torch.sin(torch.arange(weight.numel(), device=weight.device, dtype=weight.dtype)).view_as(weight))


def common_gradients_match(left, right):
    parameters = dict(right.named_parameters())
    return all(
        (value.grad is None and parameters[name].grad is None)
        or (value.grad is not None and parameters[name].grad is not None and torch.equal(value.grad, parameters[name].grad))
        for name, value in left.named_parameters()
    )


def audit(device_name="cpu"):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        from gpu_guard import require_available
        require_available()
    batch = make_batch(device)
    base = build(V12, device)
    base_rng = torch.get_rng_state().clone()
    candidate = build(Candidate, device)
    candidate_rng = torch.get_rng_state().clone()
    disabled = build(Candidate, device, use_item_attempt_stage=0)
    added = sorted(set(candidate.state_dict()) - set(base.state_dict()))
    expected = json.loads((ROOT / "protocol.json").read_text())["candidate_structure"]["new_state_tensors"]
    checks = {
        "forward_diff_is_only_attempt_readout_and_disabled_dispatch": forward_scope_matches_v12(),
        "inherited_statistics_and_attention_are_unmodified": Candidate._stats is V12._stats and Candidate._attend is V12._attend,
        "exactly_declared_new_state_tensors": added == expected,
        "all_original_tensors_preserved": set(base.state_dict()).issubset(candidate.state_dict()),
        "common_initial_tensors_identical_to_v12": all(torch.equal(value, candidate.state_dict()[name]) for name, value in base.state_dict().items()),
        "initialization_rng_identical_to_v12": torch.equal(base_rng, candidate_rng),
        "new_output_zero_initialized": bool(candidate.attempt_readout.output.weight.eq(0).all()),
        "all_original_attention_modules_preserved": all(type(block["attn"]) is nn.MultiheadAttention for block in candidate.blocks),
        "readout_has_no_dropout": not any(isinstance(m, nn.Dropout) for m in candidate.attempt_readout.modules()),
    }
    checks.update(oracle_checks(device))
    base.eval()
    candidate.eval()
    disabled.eval()
    with torch.no_grad():
        old = base(batch)
        checks["initial_active_exact_v12_eval"] = torch.equal(old, candidate(batch))
        checks["disabled_exact_v12_eval"] = torch.equal(old, disabled(batch))
    base.train()
    candidate.train()
    disabled.train()
    original_prediction, original_trace = aligned(base, batch)
    candidate_prediction, candidate_trace = aligned(candidate, batch)
    disabled_prediction, disabled_trace = aligned(disabled, batch)
    checks["initial_active_exact_v12_train"] = torch.equal(original_prediction, candidate_prediction)
    checks["disabled_exact_v12_train"] = torch.equal(original_prediction, disabled_prediction)
    checks["active_dropout_trace_identical"] = original_trace == candidate_trace
    checks["disabled_dropout_trace_identical"] = original_trace == disabled_trace
    for model, prediction in ((base, original_prediction), (candidate, candidate_prediction), (disabled, disabled_prediction)):
        torch.nn.functional.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["initial_active_common_gradients_identical"] = common_gradients_match(base, candidate)
    checks["disabled_common_gradients_identical"] = common_gradients_match(base, disabled)
    gradient = candidate.attempt_readout.output.weight.grad
    checks["new_output_receives_gradient_at_zero_start"] = gradient is not None and bool(gradient.abs().sum() > 0)
    checks["disabled_readout_has_no_gradient"] = all(p.grad is None for p in disabled.attempt_readout.parameters())

    activate(candidate)
    activate(disabled)
    candidate.eval()
    base.eval()
    disabled.eval()
    with torch.no_grad():
        prediction = candidate(batch)
        checks["activated_readout_changes_predictions"] = not torch.equal(prediction, base(batch))
        checks["activated_but_disabled_readout_exact_v12"] = torch.equal(disabled(batch), base(batch))
        checks["prediction_shape_range_finite"] = prediction.shape == (2, 8) and bool(torch.isfinite(prediction).all()) and bool(((prediction > 0) & (prediction < 1)).all())
        safe = []
        for position in range(batch["rseqs"].size(1)):
            perturbed = clone(batch)
            perturbed["rseqs"][:, position] = 1 - perturbed["rseqs"][:, position]
            safe.append(torch.equal(prediction[:, :position + 1], candidate(perturbed)[:, :position + 1]))
        checks["every_current_response_boundary_safe"] = all(safe)
        labels = clone(batch)
        labels["shft_rseqs"] = 1 - labels["shft_rseqs"]
        checks["shifted_target_labels_never_read"] = torch.equal(prediction, candidate(labels))
        unused = clone(batch)
        unused.update(utseqs=torch.full_like(batch["rseqs"], 999), shft_utseqs=torch.full_like(batch["rseqs"], -1), tseqs=torch.ones_like(batch["rseqs"]), shft_tseqs=torch.zeros_like(batch["rseqs"]))
        checks["timestamps_and_current_durations_are_unused"] = torch.equal(prediction, candidate(unused))
        future = clone(batch)
        future["rseqs"][:, 4:] = 1 - future["rseqs"][:, 4:]
        future["qseqs"][:, 4:] = 8
        future["shft_qseqs"][:, 3:] = 8
        future["cseqs"][:, 4:] = 5
        future["shft_cseqs"][:, 3:] = 5
        checks["future_events_cannot_change_past_predictions"] = torch.equal(prediction[:, :4], candidate(future)[:, :4])
        prefixes = []
        for length in range(1, batch["rseqs"].size(1)):
            prefix = clone(batch)
            for key, value in prefix.items():
                start = length + 1 if key in {"qseqs", "cseqs", "rseqs"} else length
                value[:, start:] = 0
            prefixes.append(torch.equal(candidate(prefix)[:, :length + 1], prediction[:, :length + 1]))
        checks["fixed_width_prefix_predictions_ignore_future_events"] = all(prefixes)
        changed_learner = clone(batch)
        changed_learner["rseqs"][1] = 1 - changed_learner["rseqs"][1]
        checks["other_learner_responses_do_not_affect_first_learner"] = torch.equal(prediction[0], candidate(changed_learner)[0])
        q, _, hq, _, hr = candidate._sequences(batch)
        first = item_attempt_indices(q, hq, hr)[0].eq(0)
        checks["unseen_item_predictions_exact_v12"] = torch.equal(prediction[first], base(batch)[first])
        no_items = build(Candidate, device, n_pid=0).eval()
        activate(no_items)
        checks["no_item_vocabulary_finite"] = bool(torch.isfinite(no_items(batch)).all())
        missing = clone(batch)
        missing["qseqs"] = missing["qseqs"][:, :0]
        missing["shft_qseqs"] = missing["shft_qseqs"][:, :0]
        checks["missing_item_sequence_finite"] = bool(torch.isfinite(candidate(missing)).all())
        unknown = clone(batch)
        unknown["qseqs"].fill_(1)
        unknown["shft_qseqs"].fill_(1)
        checks["unknown_items_cannot_create_shared_attempt_history"] = torch.equal(candidate(unknown), base(unknown))
        empty = {key: torch.zeros_like(value) for key, value in batch.items()}
        checks["padding_only_finite"] = bool(torch.isfinite(candidate(empty)).all())
        qprediction, fused = candidate(batch, qtest=True)
        checks["qtest_contract"] = torch.equal(prediction, qprediction) and fused.shape[:2] == (2, 8)
        calls = {"ssm": 0, "input": 0, "attention": 0, "attempt": 0}

        def count(name):
            def hook(*_args):
                calls[name] += 1
            return hook

        handles = [
            candidate.ssm.register_forward_hook(count("ssm")),
            candidate.input.register_forward_hook(count("input")),
            candidate.attempt_readout.register_forward_hook(count("attempt")),
        ] + [block["attn"].register_forward_hook(count("attention")) for block in candidate.blocks]
        try:
            candidate(batch)
        finally:
            for handle in handles:
                handle.remove()
        checks["one_input_one_ssm_four_attention_one_readout_calls"] = calls == {"ssm": 1, "input": 1, "attention": 4, "attempt": 1}
        for branch, prefixes in (
            ("ssm", ("ssm.",)), ("attention", ("blocks.",)),
            ("evidence", ("item_prior.", "concept_prior.")),
            ("factorized_input", ("semantic_stream.", "evidence_stream.", "factorized_residual.")),
        ):
            flag = f"use_{branch}_branch" if branch != "factorized_input" else "use_factorized_input"
            masked = build(Candidate, device, **{flag: 0}).eval()
            activate(masked)
            before = masked(batch)
            for name, parameter in masked.named_parameters():
                if name.startswith(prefixes):
                    parameter.add_(0.17)
            if branch == "evidence":
                masked.item_support_count.zero_()
            checks[f"disabled_{branch}_cannot_affect_predictions"] = torch.equal(before, masked(batch))
    candidate.zero_grad(set_to_none=True)
    candidate.train()
    active_prediction, active_trace = aligned(candidate, batch)
    torch.nn.functional.binary_cross_entropy(active_prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["activated_readout_keeps_dropout_trace"] = active_trace == original_trace
    checks["active_gradients_finite"] = all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in candidate.parameters())
    for branch, prefixes in (
        ("ssm", ("ssm.",)), ("attention", ("blocks.",)),
        ("factorized_input", ("factorized_residual.",)), ("attempt", ("attempt_readout.",)),
    ):
        checks[f"{branch}_receives_gradient"] = any(value.grad is not None and bool(value.grad.abs().sum() > 0) for name, value in candidate.named_parameters() if name.startswith(prefixes))
    for name, module in candidate.attempt_readout.named_modules():
        if name.endswith("_embedding"):
            checks[f"{name}_zero_count_row_has_zero_gradient"] = module.weight.grad is not None and bool(module.weight.grad[0].eq(0).all())
    if device.type == "cpu":
        checks["cpu_audit_did_not_initialize_cuda"] = not torch.cuda.is_initialized()
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "candidate_id": Candidate.CANDIDATE_ID,
        "candidate_sha256": hashlib.sha256((ROOT / "attempt_stage_candidate.py").read_bytes()).hexdigest(),
        "v12_source_sha256": hashlib.sha256((ROOT / "factorized_input_candidate.py").read_bytes()).hexdigest(),
        "incumbent_sha256": hashlib.sha256((ROOT / "a2g_incumbent.py").read_bytes()).hexdigest(),
        "checks": checks, "check_count": len(checks), "new_parameter_names": added,
        "active_initial_function_equivalent_to_v12": True,
        "full_model_arbitrary_width_equivalence_claimed": False,
        "production_fixed_width": 200,
        "gpu_used": device.type == "cuda", "real_data_access": False,
        "optimizer_steps": 0, "training_started": False, "paper_goal_complete": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.device)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 2)
