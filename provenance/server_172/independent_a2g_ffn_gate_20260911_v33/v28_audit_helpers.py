"""Synthetic V27 parity, causal Newton readout, and seven-module isolation."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))

from history_pace_candidate import A2GMambaKT as V27
from prequential_newton_candidate import A2GMambaKT as Candidate
from test_prequential_newton import direct_prefix_scores, synthetic
from v19_audit_helpers import activate, aligned, build, clone, common_gradients_match
from v27_audit_helpers import activate_pace, timed_batch

NEW_TENSORS = ["prequential_newton.output_scale", "prequential_newton.projection.weight"]
BRANCHES = (
    ("ssm", ("ssm.",), "use_ssm_branch"),
    ("attention", ("blocks.",), "use_attention_branch"),
    ("evidence", ("item_prior.", "concept_prior."), "use_evidence_branch"),
    ("factorized_input", ("semantic_stream.", "evidence_stream.", "factorized_residual."), "use_factorized_input"),
    ("item_attempt_stage", ("attempt_readout.",), "use_item_attempt_stage"),
    ("history_pace", ("history_pace.",), "use_history_pace"),
)


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def forward_scope_matches_v27():
    def forward(name):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "A2GMambaKT")
        return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "forward")

    base, candidate = forward("history_pace_candidate.py"), forward("prequential_newton_candidate.py")
    for node, flag in ((base, "use_history_pace"), (candidate, "use_prequential_newton")):
        expected = ast.parse(
            f"if not self.{flag}:\n    return super().forward(dcur, train=train, qtest=qtest)"
        ).body[0]
        if ast.dump(node.body.pop(0)) != ast.dump(expected):
            return False
    base_try = next(node for node in base.body if isinstance(node, ast.Try))
    base_prediction = next(
        node for node in base_try.body
        if isinstance(node, ast.Assign) and ast.dump(node.targets[0]) == ast.dump(ast.Name(id="prediction", ctx=ast.Store()))
    )
    expected = ast.parse(
        "base_logits = self.pred(fused).squeeze(-1) + direct_logit\n"
        "correction = self.prequential_newton(base_logits, target, target_c, dcur['rseqs'])\n"
        "prediction = torch.sigmoid(base_logits + correction)"
    ).body
    counts = [0, 0, 0, 0]
    for node in candidate.body:
        if not isinstance(node, ast.Try):
            continue
        retained = []
        for statement in node.body:
            if (
                isinstance(statement, ast.If) and isinstance(statement.test, ast.Attribute)
                and statement.test.attr == "use_history_pace" and not statement.orelse
            ):
                retained.extend(statement.body)
                counts[0] += 1
                continue
            replacement = next((i for i, value in enumerate(expected) if ast.dump(value) == ast.dump(statement)), None)
            if replacement is not None:
                counts[replacement + 1] += 1
                if replacement == 2:
                    retained.append(copy.deepcopy(base_prediction))
            else:
                retained.append(statement)
        node.body = retained
    return counts == [1, 1, 1, 1] and ast.dump(base) == ast.dump(candidate)


@torch.no_grad()
def activate_shared(model):
    activate(model)
    activate_pace(model)


@torch.no_grad()
def activate_newton(model):
    model.prequential_newton.output_scale.fill_(0.2)


def has_gradient(model, prefixes):
    return any(
        p.grad is not None and bool(p.grad.abs().sum() > 0)
        for name, p in model.named_parameters() if name.startswith(prefixes)
    )


def audit(device_name="cpu"):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        from gpu_guard import require_available
        require_available()
    batch = timed_batch(device)
    base = build(V27, device)
    base_rng = torch.get_rng_state().clone()
    base_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    candidate = build(Candidate, device)
    candidate_rng = torch.get_rng_state().clone()
    candidate_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    disabled = build(Candidate, device, use_prequential_newton=0)
    added = sorted(set(candidate.state_dict()) - set(base.state_dict()))
    checks = {
        "forward_diff_only_newton_readout_and_ablation_dispatch": forward_scope_matches_v27(),
        "original_statistics_attention_factorized_time_unchanged": all(
            getattr(Candidate, name) is getattr(V27, name)
            for name in ("_stats", "_attend", "_factorized_token", "_boundary_weights", "_sequences", "_modulate_history")
        ),
        "original_ssm_type_and_tensors_unchanged": type(candidate.ssm) is type(base.ssm) and all(
            torch.equal(value, candidate.ssm.state_dict()[name]) for name, value in base.ssm.state_dict().items()
        ),
        "exactly_two_declared_newton_parameters": added == NEW_TENSORS,
        "parent93_total95_state_tensors": len(base.state_dict()) == 93 and len(candidate.state_dict()) == 95,
        "rank_equals_one_attention_head": candidate.prequential_newton.rank == candidate.d_model // candidate.blocks[0]["attn"].num_heads,
        "new_shapes_and_parameter_count": candidate.prequential_newton.output_scale.shape == ()
        and candidate.prequential_newton.projection.weight.shape == (8, 32)
        and sum(p.numel() for p in candidate.prequential_newton.parameters()) == 257,
        "new_output_zero_initialized": bool(candidate.prequential_newton.output_scale.eq(0)),
        "projection_nonzero_finite": bool(torch.isfinite(candidate.prequential_newton.projection.weight).all())
        and bool(candidate.prequential_newton.projection.weight.abs().sum() > 0),
        "common_initial_tensors_identical_to_v27": all(
            torch.equal(value, candidate.state_dict()[name]) for name, value in base.state_dict().items()
        ),
        "initialization_cpu_rng_identical_to_v27": torch.equal(base_rng, candidate_rng),
        "all_original_attention_modules_preserved": all(type(block["attn"]) is torch.nn.MultiheadAttention for block in candidate.blocks),
        "newton_has_no_dropout_or_persistent_buffers": not list(candidate.prequential_newton.buffers())
        and not any(isinstance(module, torch.nn.Dropout) for module in candidate.prequential_newton.modules()),
    }
    if device.type == "cuda":
        checks["initialization_cuda_rng_identical_to_v27"] = torch.equal(base_cuda_rng, candidate_cuda_rng)
    for model in (base, candidate, disabled):
        model.eval()
    with torch.no_grad():
        old = base(batch)
        checks["initial_active_exact_v27_eval"] = torch.equal(old, candidate(batch))
        checks["disabled_exact_v27_eval"] = torch.equal(old, disabled(batch))
    for model in (base, candidate, disabled):
        model.train()
    old, old_trace = aligned(base, batch)
    new, new_trace = aligned(candidate, batch)
    off, off_trace = aligned(disabled, batch)
    checks["initial_active_exact_v27_train"] = torch.equal(old, new)
    checks["disabled_exact_v27_train"] = torch.equal(old, off)
    checks["initial_active_dropout_trace_identical"] = old_trace == new_trace
    checks["disabled_dropout_trace_identical"] = old_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["initial_active_common_gradients_identical"] = common_gradients_match(base, candidate)
    checks["disabled_common_gradients_identical"] = common_gradients_match(base, disabled)
    checks["new_output_receives_gradient_at_zero"] = has_gradient(candidate, ("prequential_newton.output_scale",))
    gradient = candidate.prequential_newton.projection.weight.grad
    checks["projection_has_zero_gradient_at_zero_scale"] = gradient is not None and bool(gradient.eq(0).all())
    checks["disabled_newton_has_no_gradient"] = all(p.grad is None for p in disabled.prequential_newton.parameters())

    for model in (base, candidate, disabled):
        activate_shared(model)
        model.eval()
    activate_newton(candidate)
    activate_newton(disabled)
    with torch.no_grad():
        captured = []
        handle = candidate.prequential_newton.register_forward_pre_hook(
            lambda _module, inputs: captured.append(tuple(value.clone() for value in inputs))
        )
        try:
            prediction = candidate(batch)
        finally:
            handle.remove()
        logits, target, target_c, response = captured[0]
        features = candidate.prequential_newton.features(target)
        oracle = direct_prefix_scores(logits, features, response, target_c.gt(0))
        checks["full_readout_matches_independent_prefix_solve"] = torch.allclose(
            prediction, torch.sigmoid(logits + candidate.prequential_newton.output_scale * oracle), atol=2e-6, rtol=2e-6,
        )
        checks["uncorrected_reference_is_exact_v27_prediction"] = torch.equal(logits.sigmoid(), base(batch))
        checks["observed_responses_are_unshifted_history_only"] = torch.equal(response, batch["rseqs"])
        checks["activated_newton_changes_predictions"] = not torch.equal(prediction, base(batch))
        checks["disabled_nonzero_newton_exact_v27_eval"] = torch.equal(disabled(batch), base(batch))
        checks["first_target_has_exact_v27_prediction"] = torch.equal(prediction[:, 0], base(batch)[:, 0])
        checks["feature_intercept_and_unit_bound"] = bool(features[..., 0].eq(1).all()) and bool(features.square().sum(-1).le(2).all())
        checks["prediction_shape_range_finite"] = prediction.shape == (2, 8) and bool(torch.isfinite(prediction).all()) and bool(((prediction > 0) & (prediction < 1)).all())
        current_boundaries, response_boundaries = [], []
        for position in range(batch["rseqs"].size(1)):
            changed = clone(batch)
            changed["tseqs"][:, position:] += 123456
            current_boundaries.append(torch.equal(prediction[:, :position + 1], candidate(changed)[:, :position + 1]))
            changed = clone(batch)
            changed["rseqs"][:, position:] = 1 - changed["rseqs"][:, position:]
            response_boundaries.append(torch.equal(prediction[:, :position + 1], candidate(changed)[:, :position + 1]))
        checks["every_current_timestamp_boundary_safe"] = all(current_boundaries)
        checks["every_current_response_boundary_safe"] = all(response_boundaries)
        changed = clone(batch)
        changed["tseqs"][:, -1] += 200000
        checks["latest_observed_interval_changes_next_prediction"] = not torch.equal(prediction[:, -1], candidate(changed)[:, -1])
        changed = clone(batch)
        changed["rseqs"][:, -1] = 1 - changed["rseqs"][:, -1]
        checks["latest_observed_response_changes_next_prediction"] = not torch.equal(prediction[:, -1], candidate(changed)[:, -1])
        changed = clone(batch)
        changed["shft_rseqs"] = 1 - changed["shft_rseqs"]
        checks["shifted_target_labels_never_read"] = torch.equal(prediction, candidate(changed))
        changed = clone(batch)
        changed["shft_tseqs"].fill_(-123)
        changed["utseqs"].fill_(123456789)
        changed["shft_utseqs"].fill_(-9)
        checks["shifted_times_and_all_durations_unused"] = torch.equal(prediction, candidate(changed))
        changed = clone(batch)
        changed["tseqs"] += 3000000000000
        checks["absolute_epoch_offset_does_not_change_predictions"] = torch.equal(prediction, candidate(changed))
        prefixes = []
        for length in range(1, batch["rseqs"].size(1)):
            prefix = clone(batch)
            for key, value in prefix.items():
                start = length + 1 if key in {"qseqs", "cseqs", "rseqs", "tseqs", "utseqs"} else length
                value[:, start:] = 0
            prefixes.append(torch.equal(candidate(prefix)[:, :length + 1], prediction[:, :length + 1]))
        checks["fixed_width_prefix_predictions_ignore_future_events"] = all(prefixes)
        changed = clone(batch)
        changed["rseqs"][1] = 1 - changed["rseqs"][1]
        changed["tseqs"][1].zero_()
        checks["learner_isolation"] = torch.equal(prediction[0], candidate(changed)[0])
        candidate(changed)
        checks["no_state_carry_between_calls"] = torch.equal(prediction, candidate(batch))
        missing = clone(batch)
        missing["qseqs"], missing["shft_qseqs"] = missing["qseqs"][:, :0], missing["shft_qseqs"][:, :0]
        checks["missing_item_sequence_finite"] = bool(torch.isfinite(candidate(missing)).all())
        no_items = build(Candidate, device, n_pid=0).eval()
        activate_shared(no_items)
        activate_newton(no_items)
        checks["no_item_vocabulary_finite"] = bool(torch.isfinite(no_items(batch)).all())
        unknown = clone(batch)
        for name in ("qseqs", "shft_qseqs", "cseqs", "shft_cseqs"):
            unknown[name].fill_(1)
        checks["unknown_ids_are_real_events_with_newton_effect"] = bool(torch.isfinite(candidate(unknown)).all()) and not torch.equal(candidate(unknown), base(unknown))
        empty = {key: torch.zeros_like(value) for key, value in batch.items()}
        checks["padding_only_finite_and_exact_v27"] = bool(torch.isfinite(candidate(empty)).all()) and torch.equal(candidate(empty), base(empty))
        qprediction, fused = candidate(batch, qtest=True)
        checks["qtest_contract"] = torch.equal(prediction, qprediction) and fused.shape[:2] == (2, 8)
        missing_time = {key: value for key, value in batch.items() if key not in {"tseqs", "shft_tseqs"}}
        checks["newton_does_not_require_timestamps"] = not torch.equal(candidate(missing_time), base(missing_time))
        calls = dict.fromkeys(("ssm", "input", "attention", "attempt", "pace", "newton"), 0)

        def count(name):
            def hook(*_args):
                calls[name] += 1
            return hook

        handles = [
            candidate.ssm.register_forward_hook(count("ssm")),
            candidate.input.register_forward_hook(count("input")),
            candidate.attempt_readout.register_forward_hook(count("attempt")),
            candidate.history_pace.register_forward_hook(count("pace")),
            candidate.prequential_newton.register_forward_hook(count("newton")),
        ] + [block["attn"].register_forward_hook(count("attention")) for block in candidate.blocks]
        try:
            candidate(batch)
        finally:
            for handle in handles:
                handle.remove()
        checks["one_input_one_ssm_four_attention_one_attempt_one_pace_one_newton"] = calls == {
            "ssm": 1, "input": 1, "attention": 4, "attempt": 1, "pace": 1, "newton": 1,
        }
        for branch, prefixes, flag in BRANCHES:
            masked = build(Candidate, device, **{flag: 0}).eval()
            parent = build(V27, device, **{flag: 0}).eval()
            activate_shared(masked)
            activate_shared(parent)
            activate_newton(masked)
            before = masked(batch)
            checks[f"newton_survives_no_{branch}"] = not torch.equal(before, parent(batch))
            for name, parameter in masked.named_parameters():
                if name.startswith(prefixes):
                    parameter.add_(0.17)
            if branch == "evidence":
                masked.item_support_count.zero_()
            checks[f"disabled_{branch}_cannot_affect_predictions"] = torch.equal(before, masked(batch))
    for model in (base, candidate, disabled):
        model.zero_grad(set_to_none=True)
        model.train()
    old, old_trace = aligned(base, batch)
    new, new_trace = aligned(candidate, batch)
    off, off_trace = aligned(disabled, batch)
    checks["disabled_nonzero_newton_exact_v27_train"] = torch.equal(old, off)
    checks["activated_newton_keeps_dropout_trace"] = old_trace == new_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["disabled_nonzero_newton_common_gradients_identical"] = common_gradients_match(base, disabled)
    checks["active_gradients_finite"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in candidate.parameters())
    for name, prefixes in (
        ("newton_projection", ("prequential_newton.projection.",)),
        ("newton_output", ("prequential_newton.output_scale",)),
        ("history_pace", ("history_pace.",)), ("ssm", ("ssm.",)),
        ("attention", ("blocks.",)), ("factorized", ("factorized_residual.",)),
        ("attempt", ("attempt_readout.",)),
    ):
        checks[f"active_{name}_receives_gradient"] = has_gradient(candidate, prefixes)
    for branch, prefixes, flag in BRANCHES:
        masked = build(Candidate, device, **{flag: 0}).train()
        activate_shared(masked)
        activate_newton(masked)
        prediction, _ = aligned(masked, batch)
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
        checks[f"disabled_{branch}_has_no_gradient"] = not has_gradient(masked, prefixes)
        checks[f"no_{branch}_retains_newton_projection_gradient"] = has_gradient(masked, ("prequential_newton.projection.",))
    values = tuple(value.to(device) for value in synthetic(batch=1, length=200, rank=33, dtype=torch.float32))
    from prequential_newton_candidate import prefix_newton_scores
    with torch.no_grad():
        local = prefix_newton_scores(*values)
        oracle = direct_prefix_scores(values[0].double(), values[1].double(), values[2], values[3])
    checks["production_length200_rank33_matches_double_prefix_oracle"] = bool(torch.isfinite(local).all()) and torch.allclose(local.double(), oracle, atol=3e-5, rtol=3e-5)
    if device.type == "cpu":
        checks["cpu_audit_did_not_initialize_cuda"] = not torch.cuda.is_initialized()
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "candidate_id": Candidate.CANDIDATE_ID,
        "candidate_sha256": digest("prequential_newton_candidate.py"),
        "v27_source_sha256": digest("history_pace_candidate.py"),
        "incumbent_sha256": digest("a2g_incumbent.py"),
        "checks": checks, "check_count": len(checks), "new_parameter_names": added,
        "active_initial_function_equivalent_to_v27": all(checks[name] for name in (
            "initial_active_exact_v27_eval", "initial_active_exact_v27_train", "initial_active_common_gradients_identical",
        )),
        "full_model_arbitrary_width_equivalence_claimed": False, "production_fixed_width": 200,
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
