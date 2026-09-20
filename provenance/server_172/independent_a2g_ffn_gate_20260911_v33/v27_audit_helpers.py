"""Synthetic model parity, temporal causality, and physical ablation checks."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))

from attempt_stage_candidate import A2GMambaKT as V19
from history_pace_candidate import A2GMambaKT as Candidate, historical_pace_feature
from test_history_pace import scalar_oracle
from v19_audit_helpers import activate, aligned, build, clone, common_gradients_match, make_batch


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def forward_scope_matches_v19():
    def forward(name):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "A2GMambaKT")
        return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "forward")

    base, candidate = forward("attempt_stage_candidate.py"), forward("history_pace_candidate.py")
    for node, flag in ((base, "use_item_attempt_stage"), (candidate, "use_history_pace")):
        guard = node.body.pop(0)
        if not (
            isinstance(guard, ast.If) and isinstance(guard.test, ast.UnaryOp)
            and isinstance(guard.test.op, ast.Not) and isinstance(guard.test.operand, ast.Attribute)
            and guard.test.operand.attr == flag and not guard.orelse
            and len(guard.body) == 1 and isinstance(guard.body[0], ast.Return)
        ):
            return False
    removed, unwrapped = 0, 0
    for node in candidate.body:
        if not isinstance(node, ast.Try):
            continue
        retained = []
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "_modulate_history"
            ):
                removed += 1
            elif (
                isinstance(statement, ast.If) and isinstance(statement.test, ast.Attribute)
                and statement.test.attr == "use_item_attempt_stage" and not statement.orelse
            ):
                retained.extend(statement.body)
                unwrapped += 1
            else:
                retained.append(statement)
        node.body = retained
    return removed == unwrapped == 1 and ast.dump(base) == ast.dump(candidate)


@torch.no_grad()
def activate_pace(model):
    positions = torch.arange(model.d_model, dtype=model.history_pace.scale.dtype, device=model.history_pace.scale.device)
    model.history_pace.scale.copy_(.075 * torch.sin(positions + .3))
    model.history_pace.shift.copy_(.05 * torch.cos(positions + .2))


def timed_batch(device):
    batch = make_batch(device)
    timestamps = torch.tensor([
        [1500000000000, 1500000001000, 1500000004000, 1500000064000, 1500000065000, 1500003665000, 1500003666000, 1500003667000],
        [1500000000000, 1500000000300, 1500000000600, 1500000030600, 1500000030900, 1500000031200, 1500000031500, 1500000031800],
    ], dtype=torch.int64, device=device)
    batch.update(
        tseqs=timestamps[:, :-1], shft_tseqs=timestamps[:, 1:],
        utseqs=torch.full_like(timestamps[:, :-1], 9000),
        shft_utseqs=torch.full_like(timestamps[:, :-1], 9000),
    )
    return batch


def audit(device_name="cpu"):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        from gpu_guard import require_available
        require_available()
    batch = timed_batch(device)
    base = build(V19, device)
    base_rng = torch.get_rng_state().clone()
    candidate = build(Candidate, device)
    candidate_rng = torch.get_rng_state().clone()
    disabled = build(Candidate, device, use_history_pace=0)
    added = sorted(set(candidate.state_dict()) - set(base.state_dict()))
    checks = {
        "forward_diff_only_history_modulation_and_ablation_dispatch": forward_scope_matches_v19(),
        "original_statistics_attention_factorized_input_unchanged": all(
            getattr(Candidate, name) is getattr(V19, name)
            for name in ("_stats", "_attend", "_factorized_token", "_boundary_weights", "_sequences")
        ),
        "original_ssm_type_and_tensors_unchanged": type(candidate.ssm) is type(base.ssm) and all(
            torch.equal(value, candidate.ssm.state_dict()[name]) for name, value in base.ssm.state_dict().items()
        ),
        "exactly_two_declared_history_vectors": added == ["history_pace.scale", "history_pace.shift"],
        "new_vector_shapes_and_parameter_count": all(p.shape == (candidate.d_model,) for p in candidate.history_pace.parameters())
        and sum(p.numel() for p in candidate.history_pace.parameters()) == 2 * candidate.d_model,
        "new_vectors_zero_initialized": all(bool(p.eq(0).all()) for p in candidate.history_pace.parameters()),
        "common_initial_tensors_identical_to_v19": all(
            torch.equal(value, candidate.state_dict()[name]) for name, value in base.state_dict().items()
        ),
        "initialization_rng_identical_to_v19": torch.equal(base_rng, candidate_rng),
        "all_original_attention_modules_preserved": all(type(block["attn"]) is torch.nn.MultiheadAttention for block in candidate.blocks),
        "pace_module_has_no_dropout": not any(isinstance(module, torch.nn.Dropout) for module in candidate.history_pace.modules()),
    }
    hist_c = candidate._sequences(batch)[3]
    feature = historical_pace_feature(batch["tseqs"], hist_c)
    checks["temporal_feature_matches_scalar_rowwise_oracle"] = torch.allclose(
        feature, scalar_oracle(batch["tseqs"], hist_c), atol=1e-6, rtol=1e-6,
    )
    checks["first_two_features_are_zero"] = bool(feature[:, :2].eq(0).all())
    checks["subsecond_gap_survives_epoch_integer_subtraction"] = abs(float(feature[1, 2, 0]) - 0.26236426446749106) < 1e-6
    checks["local_variable_prefix_equivalence"] = all(
        torch.equal(feature[:, :length], historical_pace_feature(batch["tseqs"][:, :length - 1], hist_c[:, :length]))
        for length in range(1, hist_c.size(1))
    )

    for model in (base, candidate, disabled):
        model.eval()
    with torch.no_grad():
        old = base(batch)
        checks["initial_active_exact_v19_eval"] = torch.equal(old, candidate(batch))
        checks["disabled_exact_v19_eval"] = torch.equal(old, disabled(batch))
    for model in (base, candidate, disabled):
        model.train()
    old, old_trace = aligned(base, batch)
    new, new_trace = aligned(candidate, batch)
    off, off_trace = aligned(disabled, batch)
    checks["initial_active_exact_v19_train"] = torch.equal(old, new)
    checks["disabled_exact_v19_train"] = torch.equal(old, off)
    checks["initial_active_dropout_trace_identical"] = old_trace == new_trace
    checks["disabled_dropout_trace_identical"] = old_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["initial_active_common_gradients_identical"] = common_gradients_match(base, candidate)
    checks["disabled_common_gradients_identical"] = common_gradients_match(base, disabled)
    checks["both_new_vectors_receive_gradient_at_zero"] = all(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in candidate.history_pace.parameters()
    )
    checks["disabled_pace_has_no_gradient"] = all(p.grad is None for p in disabled.history_pace.parameters())

    for model in (base, candidate, disabled):
        activate(model)
        model.eval()
    activate_pace(candidate)
    activate_pace(disabled)
    with torch.no_grad():
        prediction = candidate(batch)
        checks["activated_pace_changes_predictions"] = not torch.equal(prediction, base(batch))
        checks["disabled_nonzero_pace_exact_v19_eval"] = torch.equal(disabled(batch), base(batch))
        checks["prediction_shape_range_finite"] = prediction.shape == (2, 8) and bool(torch.isfinite(prediction).all()) and bool(((prediction > 0) & (prediction < 1)).all())
        checks["first_two_predictions_exact_v19"] = torch.equal(prediction[:, :2], base(batch)[:, :2])
        missing_time = {key: value for key, value in batch.items() if key not in {"tseqs", "shft_tseqs"}}
        checks["missing_times_exact_v19_after_activation"] = torch.equal(candidate(missing_time), base(missing_time))
        zero_time = clone(batch)
        zero_time["tseqs"].fill_(1500000000000)
        checks["constant_times_exact_v19_after_activation"] = torch.equal(candidate(zero_time), base(zero_time))
        current_boundaries, response_boundaries = [], []
        for position in range(batch["rseqs"].size(1)):
            changed = clone(batch)
            changed["tseqs"][:, position:] += 123456
            current_boundaries.append(torch.equal(prediction[:, :position + 1], candidate(changed)[:, :position + 1]))
            changed = clone(batch)
            changed["rseqs"][:, position] = 1 - changed["rseqs"][:, position]
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
        activate(no_items)
        activate_pace(no_items)
        checks["no_item_vocabulary_finite"] = bool(torch.isfinite(no_items(batch)).all())
        checks["padding_only_finite"] = bool(torch.isfinite(candidate({key: torch.zeros_like(value) for key, value in batch.items()})).all())
        qprediction, fused = candidate(batch, qtest=True)
        checks["qtest_contract"] = torch.equal(prediction, qprediction) and fused.shape[:2] == (2, 8)
        calls = dict.fromkeys(("ssm", "input", "attention", "attempt", "pace"), 0)

        def count(name):
            def hook(*_args):
                calls[name] += 1
            return hook

        handles = [
            candidate.ssm.register_forward_hook(count("ssm")),
            candidate.input.register_forward_hook(count("input")),
            candidate.attempt_readout.register_forward_hook(count("attempt")),
            candidate.history_pace.register_forward_hook(count("pace")),
        ] + [block["attn"].register_forward_hook(count("attention")) for block in candidate.blocks]
        try:
            candidate(batch)
        finally:
            for handle in handles:
                handle.remove()
        checks["one_input_one_ssm_four_attention_one_readout_one_pace"] = calls == {"ssm": 1, "input": 1, "attention": 4, "attempt": 1, "pace": 1}
        for branch, prefixes, flag in (
            ("ssm", ("ssm.",), "use_ssm_branch"),
            ("attention", ("blocks.",), "use_attention_branch"),
            ("evidence", ("item_prior.", "concept_prior."), "use_evidence_branch"),
            ("factorized_input", ("semantic_stream.", "evidence_stream.", "factorized_residual."), "use_factorized_input"),
            ("attempt", ("attempt_readout.",), "use_item_attempt_stage"),
        ):
            masked = build(Candidate, device, **{flag: 0}).eval()
            parent = build(V19, device, **{flag: 0}).eval()
            activate(masked)
            activate(parent)
            activate_pace(masked)
            before = masked(batch)
            checks[f"pace_survives_no_{branch}"] = not torch.equal(before, parent(batch))
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
    checks["disabled_nonzero_pace_exact_v19_train"] = torch.equal(old, off)
    checks["activated_pace_keeps_dropout_trace"] = old_trace == new_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["disabled_nonzero_pace_common_gradients_identical"] = common_gradients_match(base, disabled)
    checks["active_gradients_finite"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in candidate.parameters())
    for name, prefixes in (
        ("pace", ("history_pace.",)), ("ssm", ("ssm.",)), ("attention", ("blocks.",)),
        ("factorized", ("factorized_residual.",)), ("attempt", ("attempt_readout.",)),
    ):
        checks[f"active_{name}_receives_gradient"] = any(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for key, p in candidate.named_parameters() if key.startswith(prefixes)
        )
    no_attempt = build(Candidate, device, use_item_attempt_stage=0).train()
    activate_pace(no_attempt)
    prediction, _ = aligned(no_attempt, batch)
    prediction.mean().backward()
    checks["no_attempt_disables_ordinal_readout_gradient"] = all(p.grad is None or bool(p.grad.eq(0).all()) for p in no_attempt.attempt_readout.parameters())
    checks["no_attempt_retains_temporal_gradient"] = all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in no_attempt.history_pace.parameters())
    if device.type == "cpu":
        checks["cpu_audit_did_not_initialize_cuda"] = not torch.cuda.is_initialized()
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "candidate_id": Candidate.CANDIDATE_ID,
        "candidate_sha256": digest("history_pace_candidate.py"),
        "v19_source_sha256": digest("attempt_stage_candidate.py"),
        "incumbent_sha256": digest("a2g_incumbent.py"),
        "checks": checks, "check_count": len(checks), "new_parameter_names": added,
        "active_initial_function_equivalent_to_v19": all(checks[name] for name in (
            "initial_active_exact_v19_eval", "initial_active_exact_v19_train", "initial_active_common_gradients_identical",
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
