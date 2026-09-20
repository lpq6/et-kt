"""CPU/CUDA behavior contract for V28 plus an independent concept-state graph."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
from prequential_newton_candidate import A2GMambaKT as V28
from concept_graph_candidate import A2GMambaKT as Candidate, DirectedConceptGraph
from test_concept_graph import (
    GRAPH_SUFFIXES, activate_graph, direct_graph, forward_scope_matches_v28, synthetic_graph,
)
from v19_audit_helpers import aligned, build, clone, common_gradients_match
from v27_audit_helpers import timed_batch
from v28_audit_helpers import BRANCHES as PARENT_BRANCHES, activate_newton, activate_shared, has_gradient

NEW_TENSORS = ["concept_graph." + name for name in GRAPH_SUFFIXES]
BRANCHES = PARENT_BRANCHES + (
    ("prequential_newton", ("prequential_newton.",), "use_prequential_newton"),
)


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


@torch.no_grad()
def activate_parent(model):
    activate_shared(model)
    activate_newton(model)


def production_batch(device):
    index = torch.arange(200, device=device).unsqueeze(0)
    values = {
        "qseqs": 2 + index % 7, "cseqs": 2 + index % 94,
        "rseqs": (index // 3) % 2, "tseqs": 1500000000000 + index * 37000,
        "utseqs": torch.zeros_like(index),
    }
    batch = {}
    for key, value in values.items():
        batch[key], batch["shft_" + key] = value[:, :-1], value[:, 1:]
    batch["smasks"] = torch.ones_like(batch["rseqs"], dtype=torch.bool)
    return batch


def batched_graph_oracle(module, history, hist_c, hist_r, target_c, table):
    def normalize(value):
        centered = value - value.mean(-1, keepdim=True)
        return centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()

    width, rank = module.d_model, module.rank
    nodes = table.size(0) - 2
    keys = normalize(table[2:])
    source = keys @ module.source_projection.weight.T
    destination = keys @ module.destination_projection.weight.T
    scores = source @ destination.T / math.sqrt(rank)
    diagonal = torch.eye(nodes, device=table.device, dtype=torch.bool)
    scores = scores.masked_fill(diagonal, -torch.inf)
    probabilities = torch.exp(scores - scores.logsumexp(-1, keepdim=True))
    edges = torch.where(diagonal, torch.ones_like(probabilities), probabilities)
    events = normalize(history) @ module.event_projection.weight.T + module.event_projection.bias
    state = history.new_zeros(history.size(0), nodes, rank)
    answers = []
    for position in range(history.size(1)):
        sender = (hist_c[:, position] - 2).clamp(0, nodes - 1)
        source_state = state.gather(1, sender[:, None, None].expand(-1, 1, rank)).squeeze(1)
        message = torch.tanh(events[:, position] + source_state @ module.state_projection.weight.T)
        input_gates = message[:, None] @ module.cell.weight_ih.T + module.cell.bias_ih
        hidden_gates = state @ module.cell.weight_hh.T + module.cell.bias_hh
        ir, iz, inn = input_gates.chunk(3, dim=-1)
        hr, hz, hn = hidden_gates.chunk(3, dim=-1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        proposed = (1 - update) * torch.tanh(inn + reset * hn) + update * state
        valid = hist_c[:, position].ge(2) & hist_c[:, position].lt(nodes + 2)
        valid = valid & (hist_r[:, position].eq(0) | hist_r[:, position].eq(1))
        weights = edges[sender].unsqueeze(-1)
        state = torch.where(valid[:, None, None], (1 - weights) * state + weights * proposed, state)
        target = (target_c[:, position] - 2).clamp(0, nodes - 1)
        reads = state.gather(1, target[:, None, None].expand(-1, 1, rank)).squeeze(1)
        known = target_c[:, position].ge(2) & target_c[:, position].lt(nodes + 2)
        answers.append(torch.where(known[:, None], reads, torch.zeros_like(reads)))
    return torch.stack(answers, dim=1) @ module.output.weight.T


def audit_production_synthetic(device, checks):
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "data_config.train_fold_only.json").read_text(encoding="utf-8"))["assist2017"]
    kwargs = dict(protocol["model_kwargs"])
    torch.manual_seed(42)
    candidate = Candidate(config["num_c"], config["num_q"], **kwargs).to(device)
    candidate_rng = torch.get_rng_state().clone()
    kwargs.pop("use_concept_graph")
    torch.manual_seed(42)
    parent = V28(config["num_c"], config["num_q"], **kwargs).to(device)
    checks["production_constructor_preserves_common_tensors_and_cpu_rng"] = torch.equal(
        candidate_rng, torch.get_rng_state(),
    ) and all(torch.equal(value, candidate.state_dict()[name]) for name, value in parent.state_dict().items())
    checks["production_parameters4712949_new40160_tensors105"] = (
        sum(p.numel() for p in candidate.parameters()) == 4712949
        and sum(p.numel() for p in candidate.concept_graph.parameters()) == 40160
        and len(candidate.state_dict()) == 105
    )
    checks["production_graph_capacity96_known_nodes94_spare_row_excluded"] = (
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
        checks["production_width200_initial_eval_exact_v28"] = torch.equal(candidate(batch), parent(batch))
    for model in (candidate, parent):
        model.train()
    old, old_trace = aligned(parent, batch)
    new, new_trace = aligned(candidate, batch)
    checks["production_width200_initial_train_exact_v28"] = torch.equal(new, old)
    checks["production_width200_dropout_trace_exact_v28"] = new_trace == old_trace
    for prediction in (old, new):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["production_width200_common_gradients_exact_v28"] = common_gradients_match(parent, candidate)
    checks["production_output_projection_receives_gradient_at_zero"] = has_gradient(
        candidate, ("concept_graph.output.weight",),
    )
    candidate.eval()
    candidate.zero_grad(set_to_none=True)
    activate_graph(candidate.concept_graph)
    with torch.no_grad():
        before = candidate(batch)
        spare = candidate.hist_concept_emb.weight[candidate.n_question]
        spare.copy_(100 * torch.sin(torch.arange(spare.numel(), device=device) + 1))
        checks["production_active_prediction_ignores_spare_embedding_row"] = torch.equal(before, candidate(batch))
    candidate(batch).sum().backward()
    checks["production_active_gradient_ignores_spare_embedding_row"] = bool(
        candidate.hist_concept_emb.weight.grad[candidate.n_question].eq(0).all(),
    )
    checks["production_all_ten_graph_parameters_receive_finite_gradients"] = all(
        has_gradient(candidate, (name,)) and bool(torch.isfinite(dict(candidate.named_parameters())[name].grad).all())
        for name in NEW_TENSORS
    )


def audit(device_name="cpu"):
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        from gpu_guard import require_available
        require_available()
    batch = timed_batch(device)
    base = build(V28, device)
    base_rng = torch.get_rng_state().clone()
    base_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    candidate = build(Candidate, device)
    candidate_rng = torch.get_rng_state().clone()
    candidate_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    disabled = build(Candidate, device, use_concept_graph=0)
    added = sorted(set(candidate.state_dict()) - set(base.state_dict()))
    checks = {
        "forward_diff_only_graph_insertion_capacity_bound_and_newton_conditional": forward_scope_matches_v28(),
        "original_statistics_attention_factorized_time_boundaries_unchanged": all(
            getattr(Candidate, name) is getattr(V28, name)
            for name in ("_stats", "_attend", "_factorized_token", "_boundary_weights", "_sequences", "_modulate_history")
        ),
        "original_ssm_type_and_tensors_unchanged": type(candidate.ssm) is type(base.ssm) and all(
            torch.equal(value, candidate.ssm.state_dict()[name]) for name, value in base.ssm.state_dict().items()
        ),
        "original_newton_type_and_tensors_unchanged": type(candidate.prequential_newton) is type(base.prequential_newton)
        and all(torch.equal(value, candidate.prequential_newton.state_dict()[name]) for name, value in base.prequential_newton.state_dict().items()),
        "exactly_ten_declared_graph_parameters": added == NEW_TENSORS,
        "parent95_total105_state_tensors": len(base.state_dict()) == 95 and len(candidate.state_dict()) == 105,
        "graph_state_rank_equals_one_attention_head": candidate.concept_graph.rank == 8
        and candidate.concept_graph.rank == candidate.d_model // candidate.blocks[0]["attn"].num_heads,
        "graph_uses_standard_pytorch_grucell": type(candidate.concept_graph.cell) is torch.nn.GRUCell,
        "new_parameter_count1528": sum(p.numel() for p in candidate.concept_graph.parameters()) == 1528,
        "only_graph_output_zero_initialized": all(
            bool(p.eq(0).all()) == (name == "output.weight") for name, p in candidate.concept_graph.named_parameters()
        ),
        "all_graph_initial_parameters_finite": all(bool(torch.isfinite(p).all()) for p in candidate.concept_graph.parameters()),
        "common_initial_tensors_identical_to_v28": all(
            torch.equal(value, candidate.state_dict()[name]) for name, value in base.state_dict().items()
        ),
        "initialization_cpu_rng_identical_to_v28": torch.equal(base_rng, candidate_rng),
        "all_original_attention_modules_preserved": all(type(b["attn"]) is torch.nn.MultiheadAttention for b in candidate.blocks),
        "graph_has_no_dropout_or_persistent_buffers": not list(candidate.concept_graph.buffers())
        and not any(isinstance(m, torch.nn.Dropout) for m in candidate.concept_graph.modules()),
    }
    if device.type == "cuda":
        checks["initialization_cuda_rng_identical_to_v28"] = torch.equal(base_cuda_rng, candidate_cuda_rng)
    for model in (base, candidate, disabled):
        model.eval()
    with torch.no_grad():
        original = base(batch)
        checks["initial_active_exact_v28_eval"] = torch.equal(original, candidate(batch))
        checks["disabled_exact_v28_eval"] = torch.equal(original, disabled(batch))
    for model in (base, candidate, disabled):
        model.train()
    old, old_trace = aligned(base, batch)
    new, new_trace = aligned(candidate, batch)
    off, off_trace = aligned(disabled, batch)
    checks["initial_active_exact_v28_train"] = torch.equal(old, new)
    checks["disabled_exact_v28_train"] = torch.equal(old, off)
    checks["initial_active_dropout_trace_identical"] = old_trace == new_trace
    checks["disabled_dropout_trace_identical"] = old_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["initial_active_common_gradients_identical"] = common_gradients_match(base, candidate)
    checks["disabled_common_gradients_identical"] = common_gradients_match(base, disabled)
    checks["graph_output_receives_gradient_at_zero"] = has_gradient(candidate, ("concept_graph.output.weight",))
    checks["graph_inner_parameters_have_zero_gradient_at_zero_output"] = all(
        p.grad is not None and bool(p.grad.eq(0).all())
        for name, p in candidate.concept_graph.named_parameters() if name != "output.weight"
    )
    checks["disabled_graph_has_no_gradient"] = all(p.grad is None for p in disabled.concept_graph.parameters())
    for model in (base, candidate, disabled):
        activate_parent(model)
        model.eval()
    activate_graph(candidate.concept_graph)
    activate_graph(disabled.concept_graph)
    with torch.no_grad():
        prediction = candidate(batch)
        checks["activated_graph_changes_predictions"] = not torch.equal(prediction, base(batch))
        checks["disabled_nonzero_graph_exact_v28_eval"] = torch.equal(disabled(batch), base(batch))
        checks["prediction_shape_range_finite"] = (
            prediction.shape == (2, 8) and bool(torch.isfinite(prediction).all())
            and bool(((prediction > 0) & (prediction < 1)).all())
        )
        time_boundaries, response_boundaries = [], []
        for position in range(batch["rseqs"].size(1)):
            changed = clone(batch)
            changed["tseqs"][:, position:] += 123456
            time_boundaries.append(torch.equal(prediction[:, :position + 1], candidate(changed)[:, :position + 1]))
            changed = clone(batch)
            changed["rseqs"][:, position:] = 1 - changed["rseqs"][:, position:]
            response_boundaries.append(torch.equal(prediction[:, :position + 1], candidate(changed)[:, :position + 1]))
        checks["every_current_timestamp_boundary_safe"] = all(time_boundaries)
        checks["every_current_response_boundary_safe"] = all(response_boundaries)
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
            changed = clone(batch)
            for key, value in changed.items():
                start = length + 1 if key in {"qseqs", "cseqs", "rseqs", "tseqs", "utseqs"} else length
                value[:, start:] = 0
            prefixes.append(torch.equal(candidate(changed)[:, :length + 1], prediction[:, :length + 1]))
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
        activate_parent(no_items)
        activate_graph(no_items.concept_graph)
        checks["no_item_vocabulary_finite"] = bool(torch.isfinite(no_items(batch)).all())
        unknown = clone(batch)
        for name in ("qseqs", "shft_qseqs", "cseqs", "shft_cseqs"):
            unknown[name].fill_(1)
        checks["unknown_ids_finite"] = bool(torch.isfinite(candidate(unknown)).all())
        empty = {key: torch.zeros_like(value) for key, value in batch.items()}
        checks["padding_only_finite"] = bool(torch.isfinite(candidate(empty)).all())
        qprediction, fused = candidate(batch, qtest=True)
        checks["qtest_contract"] = torch.equal(prediction, qprediction) and fused.shape[:2] == (2, 8)
        missing_time = {key: value for key, value in batch.items() if key not in {"tseqs", "shft_tseqs"}}
        checks["graph_does_not_require_timestamps"] = not torch.equal(candidate(missing_time), base(missing_time))
        calls = dict.fromkeys(("ssm", "input", "attention", "attempt", "pace", "newton", "graph"), 0)

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
        ] + [block["attn"].register_forward_hook(count("attention")) for block in candidate.blocks]
        try:
            candidate(batch)
        finally:
            for handle in handles:
                handle.remove()
        checks["original_call_counts_plus_one_graph"] = calls == {
            "ssm": 1, "input": 1, "attention": 4, "attempt": 1, "pace": 1, "newton": 1, "graph": 1,
        }
        for branch, prefixes, flag in BRANCHES:
            masked = build(Candidate, device, **{flag: 0}).eval()
            parent = build(V28, device, **{flag: 0}).eval()
            activate_parent(masked)
            activate_parent(parent)
            activate_graph(masked.concept_graph)
            before = masked(batch)
            checks[f"graph_survives_no_{branch}"] = not torch.equal(before, parent(batch))
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
    checks["disabled_nonzero_graph_exact_v28_train"] = torch.equal(old, off)
    checks["activated_graph_keeps_dropout_trace"] = old_trace == new_trace == off_trace
    for prediction in (old, new, off):
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
    checks["disabled_nonzero_graph_common_gradients_identical"] = common_gradients_match(base, disabled)
    checks["active_gradients_finite"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in candidate.parameters())
    checks["each_active_graph_parameter_receives_gradient"] = all(has_gradient(candidate, (name,)) for name in NEW_TENSORS)
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
        activate_parent(masked)
        activate_graph(masked.concept_graph)
        prediction, _ = aligned(masked, batch)
        F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
        checks[f"disabled_{branch}_has_no_gradient"] = not has_gradient(masked, prefixes)
        checks[f"no_{branch}_retains_all_graph_gradients"] = all(has_gradient(masked, (name,)) for name in NEW_TENSORS)
    small_module = DirectedConceptGraph(8, 2).to(device).double()
    activate_graph(small_module)
    small_values = synthetic_graph(device=device)
    with torch.no_grad():
        checks["batched_graph_oracle_matches_independent_scalar_equations"] = torch.allclose(
            batched_graph_oracle(small_module, *small_values),
            direct_graph(small_module, *small_values), atol=3e-12, rtol=3e-12,
        )
    module = DirectedConceptGraph(256, 8).to(device)
    activate_graph(module)
    values = synthetic_graph(batch=1, length=200, width=256, nodes=94, dtype=torch.float32, device=device)
    with torch.no_grad():
        actual = module(*values)
        expected = batched_graph_oracle(
            copy.deepcopy(module).double(),
            *[value.double() if value.is_floating_point() else value for value in values],
        )
    checks["production_length200_width256_nodes94_matches_double_graph_oracle"] = bool(
        torch.isfinite(actual).all(),
    ) and torch.allclose(actual.double(), expected, atol=3e-6, rtol=3e-6)
    audit_production_synthetic(device, checks)
    if device.type == "cpu":
        checks["cpu_audit_did_not_initialize_cuda"] = not torch.cuda.is_initialized()
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "candidate_id": Candidate.CANDIDATE_ID,
        "candidate_sha256": digest("concept_graph_candidate.py"),
        "v28_source_sha256": digest("prequential_newton_candidate.py"),
        "incumbent_sha256": digest("a2g_incumbent.py"),
        "checks": checks, "check_count": len(checks), "new_parameter_names": added,
        "active_initial_function_equivalent_to_v28": all(checks[name] for name in (
            "initial_active_exact_v28_eval", "initial_active_exact_v28_train",
            "initial_active_common_gradients_identical", "production_width200_initial_eval_exact_v28",
            "production_width200_initial_train_exact_v28", "production_width200_common_gradients_exact_v28",
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
