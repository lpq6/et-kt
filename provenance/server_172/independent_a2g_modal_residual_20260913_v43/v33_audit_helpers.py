"""Synthetic CPU/CUDA gates, V32 identity and retained intervention contract."""

import argparse
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
from concept_graph_candidate import A2GMambaKT as V32
from ffn_gate_candidate import A2GMambaKT as Candidate, InputGatedFeedForward
from test_ffn_gate import (
    BRANCHES, GATE_NAMES, activate_gates, activate_retained, base_ffn,
    ffn_oracle, gate_parameters, gates_active,
)
from v19_audit_helpers import aligned, build, clone, common_gradients_match
from v27_audit_helpers import timed_batch
from v28_audit_helpers import has_gradient
from v32_audit_helpers import audit as audit_retained_v32, production_batch


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def production_checks(device, checks):
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "data_config.train_fold_only.json").read_text(encoding="utf-8"))["assist2017"]
    kwargs = dict(protocol["model_kwargs"])
    torch.manual_seed(42)
    candidate = Candidate(config["num_c"], config["num_q"], **kwargs).to(device)
    candidate_rng = torch.get_rng_state().clone()
    kwargs.pop("use_ffn_gate")
    torch.manual_seed(42)
    parent = V32(config["num_c"], config["num_q"], **kwargs).to(device)
    checks["production_common105_tensors_and_cpu_rng_exact_v32"] = torch.equal(
        candidate_rng, torch.get_rng_state(),
    ) and len(parent.state_dict()) == 105 and all(
        torch.equal(value, candidate.state_dict()[name]) for name, value in parent.state_dict().items()
    )
    checks["production_parameters5107701_new394752_tensors111"] = (
        sum(parameter.numel() for parameter in candidate.parameters()) == 5107701
        and sum(parameter.numel() for parameter in gate_parameters(candidate).values()) == 394752
        and len(candidate.state_dict()) == 111
    )
    checks["production_six_zero_gates_have_original_ffn_shapes"] = all(
        tuple(value.shape) == ((512, 256) if name.endswith("weight") else (512,))
        and bool(value.eq(0).all())
        for name, value in gate_parameters(candidate).items()
    )
    checks["production_retained_graph_capacity96_nodes94_unchanged"] = (
        candidate.n_question == config["num_c"] == 96
        and candidate.hist_concept_emb.num_embeddings == 97
        and candidate.concept_graph.edge_weights(
            candidate.hist_concept_emb.weight[:candidate.n_question],
        ).shape == (94, 94)
    )
    batch = production_batch(device)
    for model in (candidate, parent):
        model.eval()
    with torch.no_grad():
        checks["production_width200_initial_eval_exact_v32"] = torch.equal(candidate(batch), parent(batch))
    for model in (candidate, parent):
        model.train()
    old, old_trace = aligned(parent, batch)
    new, new_trace = aligned(candidate, batch)
    checks["production_width200_initial_train_exact_v32"] = torch.equal(new, old)
    checks["production_width200_dropout_trace_exact_v32"] = new_trace == old_trace
    for prediction in (old, new):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["production_width200_common_gradients_exact_v32"] = common_gradients_match(parent, candidate)
    checks["production_all_six_gates_receive_gradient_at_zero"] = all(
        has_gradient(candidate, (name,)) for name in GATE_NAMES
    )
    candidate.zero_grad(set_to_none=True)
    candidate.eval()
    activate_retained(candidate)
    activate_gates(candidate)
    candidate(batch).sum().backward()
    checks["production_active_all_six_gates_receive_finite_gradients"] = all(
        has_gradient(candidate, (name,)) and bool(torch.isfinite(value.grad).all())
        for name, value in gate_parameters(candidate).items()
    )
    checks["production_active_all_retained_branches_receive_gradients"] = all(
        has_gradient(candidate, prefixes) for _, prefixes, _ in BRANCHES
    )
    module = copy.deepcopy(candidate.blocks[0]["ffn"]).eval()
    inputs = torch.sin(torch.arange(200 * 256, device=device).float() * 0.01).reshape(1, 200, 256)
    with torch.no_grad():
        actual = module(inputs)
        expected = ffn_oracle(copy.deepcopy(module).double(), inputs.double())
    checks["production_width256_hidden512_length200_ffn_matches_double_oracle"] = (
        bool(torch.isfinite(actual).all())
        and torch.allclose(actual.double(), expected, atol=3e-6, rtol=3e-6)
    )


def audit(device_name="cpu"):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        from gpu_guard import require_available
        require_available()
    retained = audit_retained_v32(device_name)
    batch = timed_batch(device)
    parent = build(V32, device)
    parent_rng = torch.get_rng_state().clone()
    parent_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    candidate = build(Candidate, device)
    candidate_rng = torch.get_rng_state().clone()
    candidate_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    disabled = build(Candidate, device, use_ffn_gate=0)
    added = sorted(set(candidate.state_dict()) - set(parent.state_dict()))
    checks = {
        "retained_v32_behavior_contract_all99_pass": retained["status"] == "pass"
        and retained["check_count"] == 99 and all(retained["checks"].values()),
        "forward_and_all_parent_methods_inherited": all(
            getattr(Candidate, name) is getattr(V32, name)
            for name in ("forward", "_attend", "_stats", "_sequences", "_boundary_weights", "_factorized_token", "_modulate_history")
        ),
        "original_ssm_newton_graph_types_and_tensors_unchanged": all(
            type(getattr(candidate, name)) is type(getattr(parent, name))
            and all(torch.equal(value, getattr(candidate, name).state_dict()[key])
                    for key, value in getattr(parent, name).state_dict().items())
            for name in ("ssm", "prequential_newton", "concept_graph")
        ),
        "exactly_six_declared_gate_parameters": added == GATE_NAMES,
        "parent105_total111_state_tensors": len(parent.state_dict()) == 105 and len(candidate.state_dict()) == 111,
        "new_parameter_count6336": sum(p.numel() for p in gate_parameters(candidate).values()) == 6336,
        "all_new_parameters_finite_and_zero": all(
            bool(torch.isfinite(p).all()) and bool(p.eq(0).all()) for p in gate_parameters(candidate).values()
        ),
        "common_initial_tensors_identical_to_v32": all(
            torch.equal(value, candidate.state_dict()[name]) for name, value in parent.state_dict().items()
        ),
        "initialization_cpu_rng_identical_to_v32": torch.equal(parent_rng, candidate_rng),
        "four_original_attention_modules_preserved": len(candidate.blocks) == 4
        and all(type(block["attn"]) is torch.nn.MultiheadAttention for block in candidate.blocks),
        "three_gated_ffns_with_original_five_modules": all(
            isinstance(block["ffn"], InputGatedFeedForward) and len(block["ffn"]) == 5
            and [type(module) for module in block["ffn"].children()]
            == [torch.nn.Linear, torch.nn.GELU, torch.nn.Dropout, torch.nn.Linear, torch.nn.Dropout]
            for block in candidate.blocks[:-1]
        ) and "ffn" not in candidate.blocks[-1],
        "gates_add_no_buffers_or_dropout_modules": len(list(candidate.buffers())) == len(list(parent.buffers()))
        and sum(isinstance(m, torch.nn.Dropout) for m in candidate.modules())
        == sum(isinstance(m, torch.nn.Dropout) for m in parent.modules()),
    }
    if device.type == "cuda":
        checks["initialization_cuda_rng_identical_to_v32"] = torch.equal(parent_cuda_rng, candidate_cuda_rng)
    for model in (parent, candidate, disabled):
        model.eval()
    with torch.no_grad():
        original = parent(batch)
        checks["initial_active_exact_v32_eval"] = torch.equal(original, candidate(batch))
        checks["disabled_exact_v32_eval"] = torch.equal(original, disabled(batch))
    for model in (parent, candidate, disabled):
        model.train()
    old, old_trace = aligned(parent, batch)
    new, new_trace = aligned(candidate, batch)
    off, off_trace = aligned(disabled, batch)
    checks["initial_active_exact_v32_train"] = torch.equal(old, new)
    checks["disabled_exact_v32_train"] = torch.equal(old, off)
    checks["initial_active_dropout_trace_identical"] = old_trace == new_trace
    checks["disabled_dropout_trace_identical"] = old_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["initial_active_common_gradients_identical"] = common_gradients_match(parent, candidate)
    checks["disabled_common_gradients_identical"] = common_gradients_match(parent, disabled)
    checks["all_six_gates_receive_gradient_at_zero"] = all(has_gradient(candidate, (name,)) for name in GATE_NAMES)
    checks["disabled_gates_have_no_gradient"] = all(p.grad is None for p in gate_parameters(disabled).values())
    for model in (parent, candidate, disabled):
        activate_retained(model)
        model.eval()
    activate_gates(candidate)
    activate_gates(disabled)
    with torch.no_grad():
        prediction = candidate(batch)
        checks["activated_gates_change_predictions"] = not torch.equal(prediction, parent(batch))
        checks["disabled_nonzero_gates_exact_v32_eval"] = torch.equal(disabled(batch), parent(batch))
        checks["prediction_shape_range_finite"] = (
            prediction.shape == (2, 8) and bool(torch.isfinite(prediction).all())
            and bool(((prediction > 0) & (prediction < 1)).all())
        )
        times, responses = [], []
        for position in range(batch["rseqs"].size(1)):
            changed = clone(batch)
            changed["tseqs"][:, position:] += 123456
            times.append(torch.equal(prediction[:, :position + 1], candidate(changed)[:, :position + 1]))
            changed = clone(batch)
            changed["rseqs"][:, position:] = 1 - changed["rseqs"][:, position:]
            responses.append(torch.equal(prediction[:, :position + 1], candidate(changed)[:, :position + 1]))
        checks["every_current_timestamp_boundary_safe"] = all(times)
        checks["every_current_response_boundary_safe"] = all(responses)
        changed = clone(batch)
        changed["rseqs"][:, -1] = 1 - changed["rseqs"][:, -1]
        checks["latest_observed_response_changes_next_prediction"] = not torch.equal(prediction[:, -1], candidate(changed)[:, -1])
        changed = clone(batch)
        changed["shft_rseqs"] = 1 - changed["shft_rseqs"]
        checks["shifted_target_labels_never_read"] = torch.equal(prediction, candidate(changed))
        for name in ("shft_tseqs", "utseqs", "shft_utseqs"):
            changed[name].fill_(123456789)
        checks["shifted_times_and_all_durations_unused"] = torch.equal(prediction, candidate(changed))
        changed = clone(batch)
        changed["tseqs"] += 3000000000000
        checks["absolute_epoch_offset_does_not_change_predictions"] = torch.equal(prediction, candidate(changed))
        prefixes = []
        for length in range(1, batch["rseqs"].size(1)):
            changed = clone(batch)
            for key, value in changed.items():
                start = length + 1 if key in {"qseqs", "cseqs", "rseqs", "tseqs", "utseqs"} else length
                value[:, start:] = 0
            prefixes.append(torch.equal(candidate(changed)[:, :length + 1], prediction[:, :length + 1]))
        checks["fixed_width_prefix_predictions_ignore_future_events"] = all(prefixes)
        changed = clone(batch)
        changed["rseqs"][1] = 1 - changed["rseqs"][1]
        checks["learner_isolation"] = torch.equal(prediction[0], candidate(changed)[0])
        candidate(changed)
        checks["no_state_carry_between_calls"] = torch.equal(prediction, candidate(batch))
        missing = clone(batch)
        missing["qseqs"], missing["shft_qseqs"] = missing["qseqs"][:, :0], missing["shft_qseqs"][:, :0]
        checks["missing_item_sequence_finite"] = bool(torch.isfinite(candidate(missing)).all())
        no_items = build(Candidate, device, n_pid=0).eval()
        activate_retained(no_items)
        activate_gates(no_items)
        checks["no_item_vocabulary_finite"] = bool(torch.isfinite(no_items(batch)).all())
        unknown = clone(batch)
        for name in ("qseqs", "shft_qseqs", "cseqs", "shft_cseqs"):
            unknown[name].fill_(1)
        checks["unknown_ids_finite"] = bool(torch.isfinite(candidate(unknown)).all())
        empty = {key: torch.zeros_like(value) for key, value in batch.items()}
        checks["padding_only_finite"] = bool(torch.isfinite(candidate(empty)).all())
        qprediction, fused = candidate(batch, qtest=True)
        checks["qtest_contract"] = torch.equal(prediction, qprediction) and fused.shape[:2] == (2, 8)
        candidate.use_ffn_gate = 0
        checks["runtime_disable_synchronizes_all_wrappers_and_parent_output"] = all(
            not block["ffn"].enabled for block in candidate.blocks if "ffn" in block
        ) and torch.equal(candidate(batch), parent(batch))
        candidate.use_ffn_gate = 1
        checks["runtime_enable_restores_original_active_output"] = torch.equal(prediction, candidate(batch))
        calls = dict.fromkeys(("ssm", "input", "attention", "attempt", "pace", "newton", "graph", "ffn"), 0)

        def count(name):
            def hook(*_args):
                calls[name] += 1
            return hook

        handles = [
            module.register_forward_hook(count(name))
            for module, name in (
                (candidate.ssm, "ssm"), (candidate.input, "input"),
                (candidate.attempt_readout, "attempt"), (candidate.history_pace, "pace"),
                (candidate.prequential_newton, "newton"), (candidate.concept_graph, "graph"),
            )
        ]
        handles += [block["attn"].register_forward_hook(count("attention")) for block in candidate.blocks]
        handles += [block["ffn"].register_forward_hook(count("ffn")) for block in candidate.blocks if "ffn" in block]
        try:
            candidate(batch)
        finally:
            for handle in handles:
                handle.remove()
        checks["original_call_counts_and_three_gated_ffns"] = calls == {
            "ssm": 1, "input": 1, "attention": 4, "attempt": 1, "pace": 1, "newton": 1, "graph": 1, "ffn": 3,
        }
        for branch, prefixes, flag in BRANCHES:
            masked = build(Candidate, device, **{flag: 0}).eval()
            control = build(V32, device, **{flag: 0}).eval()
            activate_retained(masked)
            activate_retained(control)
            activate_gates(masked)
            before = masked(batch)
            if branch == "attention":
                checks["no_attention_also_removes_nested_gate_effect"] = torch.equal(before, control(batch))
            else:
                checks[f"gate_survives_no_{branch}"] = not torch.equal(before, control(batch))
            for name, parameter in masked.named_parameters():
                if name.startswith(prefixes):
                    parameter.add_(0.17)
            if branch == "evidence":
                masked.item_support_count.zero_()
            checks[f"disabled_{branch}_cannot_affect_predictions"] = torch.equal(before, masked(batch))
        zero_parent, zero_candidate = copy.deepcopy(parent), copy.deepcopy(candidate)
        for model in (zero_parent, zero_candidate):
            for block in model.blocks:
                block["attn"].out_proj.weight.zero_()
                block["attn"].out_proj.bias.zero_()
        checks["gate_effect_survives_zero_attention_outputs"] = not torch.equal(zero_parent(batch), zero_candidate(batch))
    for model in (parent, candidate, disabled):
        model.zero_grad(set_to_none=True)
        model.train()
    old, old_trace = aligned(parent, batch)
    new, new_trace = aligned(candidate, batch)
    off, off_trace = aligned(disabled, batch)
    checks["disabled_nonzero_gates_exact_v32_train"] = torch.equal(old, off)
    checks["activated_gates_keep_dropout_trace"] = old_trace == new_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["disabled_nonzero_gates_common_gradients_identical"] = common_gradients_match(parent, disabled)
    checks["active_gradients_finite"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in candidate.parameters())
    checks["each_active_gate_parameter_receives_gradient"] = all(has_gradient(candidate, (name,)) for name in GATE_NAMES)
    for branch, prefixes, flag in BRANCHES:
        checks[f"active_{branch}_receives_gradient"] = has_gradient(candidate, prefixes)
        masked = build(Candidate, device, **{flag: 0}).train()
        activate_retained(masked)
        activate_gates(masked)
        prediction, _ = aligned(masked, batch)
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
        checks[f"disabled_{branch}_has_no_gradient"] = not has_gradient(masked, prefixes)
        if branch == "attention":
            checks["no_attention_has_no_nested_gate_gradients"] = all(p.grad is None for p in gate_parameters(masked).values())
        else:
            checks[f"no_{branch}_retains_all_gate_gradients"] = all(has_gradient(masked, (name,)) for name in GATE_NAMES)
    module = InputGatedFeedForward(base_ffn()).to(device).eval()
    gates_active(module)
    inputs = torch.linspace(-1.7, 2.2, 40, device=device, dtype=torch.float64).reshape(2, 5, 4).requires_grad_(True)
    checks["active_ffn_matches_independent_erf_and_row_sum"] = torch.allclose(
        module(inputs), ffn_oracle(module, inputs), atol=1e-12, rtol=1e-12,
    )
    parameters = (inputs, *module.parameters())
    actual = torch.autograd.grad(module(inputs).square().sum(), parameters)
    expected = torch.autograd.grad(ffn_oracle(module, inputs).square().sum(), parameters)
    checks["active_ffn_gradients_match_independent_equations"] = all(
        torch.allclose(left, right, atol=1e-11, rtol=1e-11) for left, right in zip(actual, expected)
    )
    production_checks(device, checks)
    if device.type == "cpu":
        checks["cpu_audit_did_not_initialize_cuda"] = not torch.cuda.is_initialized()
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "fail", "candidate_id": Candidate.CANDIDATE_ID,
        "candidate_sha256": digest("ffn_gate_candidate.py"), "v32_source_sha256": digest("concept_graph_candidate.py"),
        "incumbent_sha256": digest("a2g_incumbent.py"),
        "checks": checks, "check_count": len(checks), "new_parameter_names": added,
        "retained_v32_contract": retained,
        "active_initial_function_equivalent_to_v32": all(checks[name] for name in (
            "initial_active_exact_v32_eval", "initial_active_exact_v32_train",
            "initial_active_common_gradients_identical", "production_width200_initial_eval_exact_v32",
            "production_width200_initial_train_exact_v32", "production_width200_common_gradients_exact_v32",
        )),
        "attention_gate_ablations_are_nested": True,
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
