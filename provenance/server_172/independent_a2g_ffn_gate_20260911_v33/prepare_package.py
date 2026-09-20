"""Bind the closed V32 control and prospective Assist2017 FFN-gate protocol."""

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OUTPUTS = WORK.parent / "outputs"
PARENT = WORK / "independent_a2g_concept_graph_20260911_v32"
PARENT_OUTPUT = OUTPUTS / "a2g_concept_graph_v32_20260911"
REVIEW = WORK / "next_structure_review_20260911_v33"
PARENT_ID = "a2g_v28_directed_concept_graph_20260911_v32"
CANDIDATE = "a2g_v32_input_gated_ffn_20260911_v33"
RETAINED = [
    "evidence", "ssm", "attention", "factorized_input", "item_attempt_stage",
    "history_pace", "prequential_newton", "concept_graph",
]
GATE_SHAPES = {
    f"blocks.{block}.ffn.gate_{suffix}": [512, 256] if suffix == "weight" else [512]
    for block in range(3) for suffix in ("weight", "bias")
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def copy_checked(source, target):
    if not target.resolve().is_relative_to(ROOT) or target.exists():
        raise RuntimeError(f"unsafe or existing destination: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha(target) != sha(source):
        raise RuntimeError(f"copy differs: {target}")


def main():
    if (ROOT / "protocol.json").exists() or (ROOT / "manifest.json").exists():
        raise RuntimeError("package already prepared or frozen")
    staged = load(ROOT / "staged_parent_sources.json")
    index_path = OUTPUTS / "research_resume_20260909/latest_stage.json"
    index = load(index_path)
    if (
        sha(index_path) != staged["index_sha256"]
        or any(index[key]["candidate"] != PARENT_ID for key in (
            "latest_completed_candidate", "best_observed_assist2017",
            "structural_working_base_assist2017",
        ))
        or index["next_candidate"]["implemented"] or index["next_candidate"]["training_started"]
        or index["execution_scope"]["other_dataset_training_allowed"]
        or index["execution_scope"]["pilot_auc_must_strictly_exceed"] != 0.8174
        or not load(PARENT_OUTPUT / "post_publication_check.json")["module_joint_gate_pass"]
        or index["best_observed_assist2017"]["auc"] != 0.8011094815652686
        or sha(Path(staged["user_baseline_table"])) != staged["user_baseline_table_sha256"]
    ):
        raise RuntimeError("closed V32 parent or Assist2017-first scope changed")
    for relative, expected in staged["unchanged_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"staged parent dependency changed: {relative}")
    if sha(PARENT / "manifest.json") != sha(PARENT_OUTPUT / "manifest.json"):
        raise RuntimeError("source and publication manifests differ")
    for relative, expected in load(PARENT / "manifest.json")["sha256"].items():
        if sha(PARENT / relative) != expected:
            raise RuntimeError(f"frozen V32 source changed: {relative}")
    for relative, expected in load(PARENT_OUTPUT / "progress_final.json")["evidence_sha256"].items():
        if sha(PARENT_OUTPUT / relative) != expected:
            raise RuntimeError(f"V32 terminal evidence changed: {relative}")
    scan = load(REVIEW / "bounded_source_scan.json")
    tests = load(REVIEW / "initial_ffn_gate_tests.json")
    reference = load(REVIEW / "glu_reference.json")
    if (
        scan["status"] != "bounded_source_review_captured_pending_architecture_decision"
        or scan["hits"]["multiplicative_ffn"] or scan["gpu_used"]
        or scan["weights_read"] or scan["data_files_read"]
        or tests["status"] != "pass" or tests["tests_run"] != 37
        or tests["errors"] or tests["failures"] or tests["skipped"]
        or tests["gpu_used"] or tests["cuda_initialized"] or tests["training_started"]
        or reference["status"] != "primary_arxiv_bibliographic_metadata_verified"
        or not reference["abstract_page_inspected"]
        or reference["full_text_method_details_verified"]
    ):
        raise RuntimeError("bounded source review, primary metadata or first CPU tests incomplete")
    for name, expected in tests["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"initially tested model source changed: {name}")
    for name, binding in scan["sources"].items():
        if sha(REVIEW / "sources" / name) != binding["sha256"]:
            raise RuntimeError(f"historical review source changed: {name}")
    for source, suffix in (
        (PARENT / "manifest.json", "manifest"),
        (PARENT / "protocol.json", "protocol"),
        (PARENT / "cpu_contract.json", "cpu_contract"),
        (PARENT_OUTPUT / "assist2017/seed42/full/result.json", "result"),
        (PARENT_OUTPUT / "assist2017/seed42/provenance.json", "provenance"),
        (PARENT_OUTPUT / "progress_final.json", "progress"),
        (PARENT_OUTPUT / "post_publication_check.json", "closure"),
        (PARENT_OUTPUT / "initialization_closure_check.json", "initialization_closure"),
    ):
        copy_checked(source, ROOT / f"reference_v32_{suffix}.json")
    for path in sorted(REVIEW.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".md"}:
            copy_checked(path, ROOT / "review/ffn_gate" / path.relative_to(REVIEW))
    parent_protocol = load(PARENT / "protocol.json")
    protocol = copy.deepcopy(parent_protocol)
    parent_result = load(ROOT / "reference_v32_result.json")
    binding = {
        "candidate_id": PARENT_ID, "auc": parent_result["metrics"]["auc"],
        "reference_dataset": "assist2017",
        "source_sha256": sha(ROOT / "concept_graph_candidate.py"),
        "result_path": str(PurePosixPath(parent_result["checkpoint"]["path"]).with_name("result.json")),
        "result_sha256": sha(ROOT / "reference_v32_result.json"),
        "initial_state_path": parent_result["initial_state"]["path"],
        "initial_state_sha256": parent_result["initial_state"]["sha256"],
        "initial_state_use": "comparison only; no artifact supplies model initialization",
    }
    protocol.update(
        candidate_id=CANDIDATE, experiment_family=ROOT.name,
        best_candidate=binding, structural_control=copy.deepcopy(binding),
        best_candidate_role="legacy alias: V32 is structural control, immediate predecessor and highest Assist2017 observation",
        best_observed_candidate={
            key: binding[key] for key in ("candidate_id", "auc", "result_path", "result_sha256")
        } | {"same_as_structural_control": True},
        closed_predecessor={
            key: binding[key] for key in ("candidate_id", "auc", "result_path", "result_sha256")
        } | {"required_status": parent_result["status"], "admission_pass": False},
    )
    protocol["model_kwargs"] = {**parent_protocol["model_kwargs"], "use_ffn_gate": 1}
    parent_structure = parent_protocol["candidate_structure"]
    protocol["candidate_structure"] = {
        "base": PARENT_ID,
        "changed_path": "input-dependent neuron gates inside the three retained FFNs",
        "formula": "Dropout2(W2 Dropout1(GELU(W1 x+b1) * (2 sigmoid(U x+v)))+b2)",
        "gate_shapes": GATE_SHAPES, "gated_blocks": [0, 1, 2],
        "new_state_tensors": sorted(GATE_SHAPES),
        "zero_initialized_new_tensors": sorted(GATE_SHAPES),
        "new_parameters": 394752,
        "expected_assist2017_parameters": 5107701,
        "expected_parent_state_tensors": 105, "expected_total_state_tensors": 111,
        "inherited_auxiliary_state_tensors": sorted(
            parent_structure["inherited_auxiliary_state_tensors"] + parent_structure["new_state_tensors"]
        ),
        "all_initial_state_tensors_equal_v32": False,
        "common_initial_parameters_equal_v32": True, "active_initial_function_equals_v32": True,
        "initialization": "zeros_like original first FFN linear weight and bias; consumes no random numbers",
        "disabled_behavior": "use_ffn_gate=0 synchronizes all wrappers and delegates to original Sequential.forward",
        "forward_and_attention_methods_inherited": True,
        "retained_ablation_policy": "seven other removals retain gate effects and gradients; removing the attention/FFN branch also removes its nested gates",
        "ablation_dependencies": {"no_attention": ["attention", "ffn_gate"]},
        "new_path_nested_in_attention": True,
        "ssm_passes": 1, "input_projection_passes": 1, "attention_passes": 4,
        "concept_graph_passes": 1, "ffn_gate_passes": 3,
        "extra_cost": "three B*T*D*d_ff gate projections; same FFN widths and dropout calls",
        "dropout_policy": "both original FFN dropout modules remain in their original order",
        "state_scope": "pointwise stateless FFN gate; retained graph resets learner state on every forward",
        "padding_policy": "unchanged V32 masking and graph unknown/padding semantics",
        "statistics": "unchanged V32 statistics",
        "history_pace": "unchanged V32 historical-time conditioning",
        "prequential_newton": "unchanged V32 Newton readout",
        "concept_graph": "unchanged V32 directed concept-node recurrence",
        "retained_graph_shapes": parent_structure["graph_shapes"],
        "assist2017_capacity": 96, "assist2017_known_graph_nodes": 94,
        "fixed_width_contract": "pointwise gate preserves causal FFN inputs; full model retains width200 statistics",
        "production_fixed_width": 200,
        "potential_same_five_with_all_nine_modules": parent_structure["potential_same_five_with_all_eight_modules"],
        "applicability_limit": "metadata coverage and synthetic tests are not positive module evidence",
        "unused_inputs": parent_structure["unused_inputs"],
        "literature_novelty_claimed": False, "pedagogical_prerequisites_claimed": False,
        "parameter_matched_superiority_claimed": False,
        "full_model_arbitrary_width_equivalence_claimed": False,
    }
    protocol["hypothesis"] = (
        "An independent projection of the existing causal FFN input can condition each "
        "intermediate nonlinear neuron before mixing. Test one identity-start gate "
        "without altering V32 inputs, retained modules or training."
    )
    protocol["planned_ablation_variants"] = {
        **parent_protocol["planned_ablation_variants"], "no_ffn_gate": {"use_ffn_gate": 0},
    }
    protocol["ablation_scope"] = {
        **parent_protocol["ablation_scope"],
        "common": "The same five qualifying datasets must pass all nine independently trained controls; nested attention/FFN-gate interventions are not disjoint.",
        "attention": "Remove the whole original attention/FFN branch, including its nested gates; concept graph stays active.",
        "concept_graph": "Remove only the graph; FFN gates stay active. The older V28 is not a matched no-graph control for V33.",
        "ffn_gate": "Disable only FFN gates; independent V32 is the exact no-gate structural control after complete parity audits.",
    }
    protocol["completion_requirements"][1] = (
        "The same five datasets must pass independently trained evidence, SSM, attention, "
        "factorized-input, item-attempt, history-pace, Newton, concept-graph and FFN-gate controls; "
        "the attention removal also removes its nested gates."
    )
    protocol["uncertainty"]["multiplicity_family_size"] = 72
    protocol["uncertainty"]["multiplicity_family"] = (
        "eight datasets times nine prespecified contrasts: evidence, SSM, attention, "
        "factorized input, item attempt, history pace, Newton, concept graph and FFN gate; "
        "the attention/FFN-gate interventions are nested, not disjoint"
    )
    protocol["point_gate"]["admission"] = (
        "Assist2017 Full strictly exceeds0.8174 and passes independent audit before separately "
        "frozen expansion; V32 is both the structural control and highest observation."
    )
    protocol["point_gate"]["candidate_minus_best_candidate_auc"] = "legacy field compares to independently trained V32"
    protocol["prospective_protocol_changes"] = {
        "previous_candidate": "V32 is closed below the floor; retain its exploratory supported graph without any graph retry.",
        "new_module_family": "Increase family64 to72 for nine controls; record the nested attention/gate dependency.",
        "information_boundary": "no new metadata, labels, fitted transforms, graph update or auxiliary objective",
        "unchanged_scope": "one Assist2017 Full run; other datasets and module retraining disabled",
        "added_capacity_limit": "V32 has394752 fewer parameters; no parameter-matched superiority claim",
    }
    protocol["failure_policy"] = (
        "One Assist2017 Full run, max200/patience20. AUC<=0.8174 closes FFN-gate "
        "activation, gain, bias, sharing, rank, factorization, normalization, placement, "
        "initialization, width and seed retries. No baseline/control rerun, sweep or automatic resource retry."
    )
    write_new(ROOT / "protocol.json", protocol)
    write_new(ROOT / "inherited_dependencies.json", {
        "source_manifest_sha256": sha(PARENT / "manifest.json"),
        "closed_predecessor_manifest_sha256": sha(PARENT / "manifest.json"),
        "unchanged_sha256": staged["unchanged_sha256"],
    })
    write_new(ROOT / "source_review.json", {
        "status": "bounded_source_review_completed",
        "review_sha256": {
            path.relative_to(ROOT).as_posix(): sha(path)
            for path in sorted((ROOT / "review").rglob("*")) if path.is_file()
        },
        "specific_delta": "pointwise multiplicative input gates on existing FFN hidden neurons",
        "historical_overlap": {
            "token_moe": "separate routed expert residual, not gates on original FFN neurons",
            "sdpa_query_gates": "scale attention outputs, not intermediate FFN activations",
            "depth_routing": "combine whole-layer states, not individual hidden neurons",
            "branch_scale": "input-independent scalar, not an input-dependent gate vector",
            "spline": "univariate readout activation, not independent FFN input projection",
        },
        "retained_modules": RETAINED, "new_module": "ffn_gate", "planned_contrasts": 72,
        "real_data_profile_performed": False, "validation_sequences_parsed": False,
        "response_tokens_parsed": False, "trained_weights_read": False,
        "training_started": False, "test_access": False, "literature_novelty_claimed": False,
        "historical_current_time_adapter_reopened": False,
        "closed_newton_route_reopened": False, "closed_cross_count_route_reopened": False,
        "closed_tensor_lift_route_reopened": False, "closed_gaussian_overlap_route_reopened": False,
        "closed_concept_graph_route_reopened": False,
        "full_model_arbitrary_width_equivalence_claimed": False, "production_fixed_width": 200,
        "search_limits": scan["limitations"], "source_files_checked": scan["source_files_checked"],
        "pedagogical_prerequisites_claimed": False, "parameter_matched_superiority_claimed": False,
        "literature_reference": "review/ffn_gate/glu_reference.json",
        "attention_gate_ablations_are_nested": True,
    })
    print(json.dumps({
        "status": "protocol_prepared_requires_ffn_gate_specific_freeze_and_independent_audit",
        "candidate_id": CANDIDATE, "unchanged_files": len(staged["unchanged_sha256"]),
        "new_parameters": 394752, "planned_contrasts": 72,
        "training_started": False, "frozen": False,
    }))


if __name__ == "__main__":
    main()
