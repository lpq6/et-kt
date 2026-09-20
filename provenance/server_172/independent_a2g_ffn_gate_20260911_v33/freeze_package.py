"""Freeze the tested single-run Assist2017 package; refuse any refreeze."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

import torch

from launch_candidate import GUARD_SHA256, validate_package, validate_protocol

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write_new(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_tests(module, source):
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName(module)
    )
    if not result.wasSuccessful() or result.skipped or not result.testsRun:
        raise RuntimeError(stream.getvalue())
    return {
        "status": "pass", "tests_run": result.testsRun, "output": stream.getvalue(),
        "source_sha256": sha(ROOT / source), "tests_sha256": sha(ROOT / (module + ".py")),
        "gpu_used": False, "optimizer_steps": 0,
    }


def validate_science_contract(protocol):
    validate_protocol(protocol, "assist2017", 42, "full")
    expected_controls = {
        "no_attention": {"use_attention_branch": 0},
        "no_evidence": {"use_evidence_branch": 0},
        "no_ssm": {"use_ssm_branch": 0},
        "no_factorized_input": {"use_factorized_input": 0},
        "no_item_attempt_stage": {"use_item_attempt_stage": 0},
        "no_history_pace": {"use_history_pace": 0},
        "no_prequential_newton": {"use_prequential_newton": 0},
        "no_concept_graph": {"use_concept_graph": 0},
        "no_ffn_gate": {"use_ffn_gate": 0},
    }
    if protocol["planned_ablation_variants"] != expected_controls or protocol["uncertainty"]["multiplicity_family_size"] != 72:
        raise RuntimeError("all nine module controls and the72-contrast family are required")
    structure = protocol["candidate_structure"]
    gate_shapes = {
        f"blocks.{block}.ffn.gate_{suffix}": [512, 256] if suffix == "weight" else [512]
        for block in range(3) for suffix in ("weight", "bias")
    }
    expected_tensors = sorted(gate_shapes)
    parent = load("reference_v32_protocol.json")
    if (
        protocol["candidate_id"] != "a2g_v32_input_gated_ffn_20260911_v33"
        or protocol["best_candidate"] != protocol["structural_control"]
        or protocol["best_candidate"]["candidate_id"] != "a2g_v28_directed_concept_graph_20260911_v32"
        or protocol["best_candidate"]["auc"] != 0.8011094815652686
        or any(protocol["best_observed_candidate"][key] != protocol["best_candidate"][key] for key in (
            "candidate_id", "auc", "result_path", "result_sha256",
        ))
        or protocol["best_observed_candidate"].get("same_as_structural_control") is not True
        or structure["new_state_tensors"] != expected_tensors
        or structure["new_parameters"] != 394752
        or structure["gate_shapes"] != gate_shapes
        or structure["gated_blocks"] != [0, 1, 2]
        or structure["retained_graph_shapes"] != parent["candidate_structure"]["graph_shapes"]
        or structure["assist2017_capacity"] != 96 or structure["assist2017_known_graph_nodes"] != 94
        or structure["zero_initialized_new_tensors"] != expected_tensors
        or structure["expected_assist2017_parameters"] != 5107701
        or structure["expected_parent_state_tensors"] != 105
        or structure["expected_total_state_tensors"] != 111
        or structure["pedagogical_prerequisites_claimed"] is not False
        or protocol["model_kwargs"]["use_ffn_gate"] != 1
        or protocol["model_kwargs"]["use_concept_graph"] != 1
        or protocol["model_kwargs"]["use_prequential_newton"] != 1
        or protocol["model_kwargs"]["use_history_pace"] != 1
        or protocol["model_kwargs"]["use_item_attempt_stage"] != 1
        or structure["all_initial_state_tensors_equal_v32"] is not False
        or structure["common_initial_parameters_equal_v32"] is not True
        or structure["active_initial_function_equals_v32"] is not True
        or structure["forward_and_attention_methods_inherited"] is not True
        or structure["ablation_dependencies"] != {"no_attention": ["attention", "ffn_gate"]}
        or structure["new_path_nested_in_attention"] is not True
        or structure["parameter_matched_superiority_claimed"] is not False
        or structure["literature_novelty_claimed"] is not False
    ):
        raise RuntimeError("V33 structure or V32 control role changed")
    previous = load("reference_v32_result.json")
    binding = protocol["closed_predecessor"]
    if (
        binding["candidate_id"] != "a2g_v28_directed_concept_graph_20260911_v32"
        or binding["candidate_id"] != previous["candidate_id"]
        or binding["auc"] != previous["metrics"]["auc"]
        or binding["auc"] != 0.8011094815652686
        or binding["result_sha256"] != sha(ROOT / "reference_v32_result.json")
        or binding["required_status"] != previous["status"]
        or binding["admission_pass"] is not False
    ):
        raise RuntimeError("immediate predecessor must be closed V32, also the structural parent")
    kwargs = dict(protocol["model_kwargs"])
    if kwargs.pop("use_ffn_gate") != 1 or kwargs != parent["model_kwargs"]:
        raise RuntimeError("only the prespecified FFN-gate module may change model arguments")
    for field in ("training", "seeds", "train_folds", "validation_folds", "variants", "item_dropout_override"):
        if protocol[field] != parent[field]:
            raise RuntimeError(f"V32 control protocol differs: {field}")


def main():
    for name in (
        "manifest.json", "gpu_guard_contract.json", "runner_contract.json",
        "scope_contract.json", "attempt_contract.json", "history_pace_contract.json",
        "prequential_newton_contract.json", "concept_graph_contract.json", "ffn_gate_contract.json",
        "freeze_contract.json", "artifacts",
    ):
        if (ROOT / name).exists():
            raise RuntimeError(f"refusing to refreeze/overwrite: {name}")
    protocol = load("protocol.json")
    validate_science_contract(protocol)
    for report in (load("cpu_contract.json"), load("data_contract.json")):
        if report["status"] != "pass" or not report["checks"] or any(x is not True for x in report["checks"].values()):
            raise RuntimeError("CPU contracts failed")
    contract = load("cpu_contract.json")
    for field, name in (
        ("candidate_sha256", "ffn_gate_candidate.py"),
        ("v32_source_sha256", "concept_graph_candidate.py"),
        ("incumbent_sha256", "a2g_incumbent.py"),
    ):
        if contract[field] != sha(ROOT / name):
            raise RuntimeError(f"CPU-audited source changed: {name}")
    if contract["candidate_id"] != protocol["candidate_id"] or contract["gpu_used"] or contract["optimizer_steps"] or contract["training_started"] or contract["real_data_access"]:
        raise RuntimeError("CPU contract identity/scope differs")
    review = load("source_review.json")
    if (
        review["status"] != "bounded_source_review_completed"
        or review["training_started"] or review["literature_novelty_claimed"]
        or review["real_data_profile_performed"] or review["validation_sequences_parsed"]
        or review["response_tokens_parsed"] or review["historical_current_time_adapter_reopened"]
        or review["test_access"] or review["trained_weights_read"]
        or review["closed_newton_route_reopened"] or review["closed_cross_count_route_reopened"]
        or review["closed_tensor_lift_route_reopened"] or review["pedagogical_prerequisites_claimed"]
        or review["closed_gaussian_overlap_route_reopened"]
        or review["closed_concept_graph_route_reopened"]
        or review["parameter_matched_superiority_claimed"]
        or review["attention_gate_ablations_are_nested"] is not True
        or review["new_module"] != "ffn_gate" or review["planned_contrasts"] != 72
        or review["retained_modules"] != ["evidence", "ssm", "attention", "factorized_input", "item_attempt_stage", "history_pace", "prequential_newton", "concept_graph"]
    ):
        raise RuntimeError("source review incomplete")
    for relative, expected in (review["review_sha256"] | load("inherited_dependencies.json")["unchanged_sha256"]).items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise RuntimeError(f"inherited/review source changed: {relative}")
    metadata = load("supplemental_metadata.json")
    parent_manifest = load("reference_v32_manifest.json")
    if metadata["source_manifest_sha256"] != sha(ROOT / "reference_v32_manifest.json"):
        raise RuntimeError("metadata parent binding changed")
    for name in (
        "trainfold_data_summary.json", "reference_snapshot.json", "incumbent_result.json",
        "references/baseline_table.md", "references/baseline_table.csv",
    ):
        if sha(ROOT / name) != metadata["unchanged_sha256"][name] or sha(ROOT / name) != parent_manifest["sha256"][name]:
            raise RuntimeError(f"unchanged runtime/reference metadata differs: {name}")
    if sha(ROOT / "reference_v32_result.json") != protocol["best_candidate"]["result_sha256"]:
        raise RuntimeError("parent result changed")
    if sha(ROOT / "reference_v32_manifest.json") != load("inherited_dependencies.json")["source_manifest_sha256"]:
        raise RuntimeError("parent source manifest changed")
    if sha(ROOT / "reference_v32_manifest.json") != load("inherited_dependencies.json")["closed_predecessor_manifest_sha256"]:
        raise RuntimeError("closed predecessor source manifest changed")
    if sha(ROOT / "gpu_guard.py") != GUARD_SHA256 or sha(ROOT / "launch_candidate.py") != protocol["launcher"]["source_sha256"]:
        raise RuntimeError("resource guard or required launcher changed")
    profile = load("review/history_time_train_profile.json")
    assets = load("review/timestamp_asset_samples.json")
    if (
        profile["counts"]["scored_targets"] != 606414 or profile["training_learners"] != 1093
        or any(report[key] for report in (profile, assets) for key in (
            "response_tokens_parsed", "validation_tokens_parsed", "gpu_used", "model_fitting", "test_access",
        ))
        or assets["nonconstant_timestamp_samples_with_question_ids"]
        != protocol["candidate_structure"]["potential_same_five_with_all_nine_modules"]
        or len(assets["nonconstant_timestamp_samples_with_question_ids"]) < 5
    ):
        raise RuntimeError("temporal metadata scope or potential five-dataset coverage differs")
    from concept_graph_candidate import A2GMambaKT as V32
    from ffn_gate_candidate import A2GMambaKT
    config = load("data_config.train_fold_only.json")["assist2017"]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(42)
        parent_kwargs = dict(protocol["model_kwargs"])
        parent_kwargs.pop("use_ffn_gate")
        parent = V32(config["num_c"], config["num_q"], **parent_kwargs)
        parent_rng = torch.get_rng_state().clone()
        torch.default_generator.manual_seed(42)
        model = A2GMambaKT(config["num_c"], config["num_q"], **protocol["model_kwargs"])
        if not torch.equal(parent_rng, torch.get_rng_state()):
            raise RuntimeError("production constructor changed caller CPU RNG")
    state, parent_state = model.state_dict(), parent.state_dict()
    structure = protocol["candidate_structure"]
    if (
        sum(p.numel() for p in model.parameters()) != structure["expected_assist2017_parameters"]
        or sum(model.state_dict()[name].numel() for name in structure["new_state_tensors"]) != structure["new_parameters"]
        or len(model.state_dict()) != structure["expected_total_state_tensors"]
        or set(state) != set(parent_state) | set(structure["new_state_tensors"])
        or not all(torch.equal(state[name], value) for name, value in parent_state.items())
        or {name: list(state[name].shape) for name in structure["new_state_tensors"]} != structure["gate_shapes"]
        or not all(bool(torch.isfinite(state[name]).all()) and bool(state[name].eq(0).all()) for name in structure["new_state_tensors"])
        or sum("ffn" in block for block in model.blocks) != 3
        or not all(block["ffn"].enabled for block in model.blocks if "ffn" in block)
        or model.n_question != 96
        or model.concept_graph.rank != 32
        or model.concept_graph.edge_weights(model.hist_concept_emb.weight[:model.n_question]).shape != (94, 94)
        or not bool(model.concept_graph.output.weight.eq(0).all())
        or not bool(model.prequential_newton.output_scale.eq(0))
        or model.prequential_newton.projection.weight.shape != (32, 256)
        or not bool(model.prequential_newton.projection.weight.abs().sum() > 0)
        or not bool(torch.isfinite(model.prequential_newton.projection.weight).all())
        or not all(bool(p.eq(0).all()) for p in model.history_pace.parameters())
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("production model dimensions differ")
    reports = {
        "gpu_guard_contract.json": run_tests("test_gpu_guard", "gpu_guard.py"),
        "runner_contract.json": run_tests("test_runner_contract", "run_independent.py"),
        "scope_contract.json": run_tests("test_scope_gate", "launch_candidate.py"),
        "attempt_contract.json": run_tests("test_attempt_stage", "attempt_stage_candidate.py"),
        "history_pace_contract.json": run_tests("test_history_pace", "history_pace_candidate.py"),
        "prequential_newton_contract.json": run_tests("test_prequential_newton", "prequential_newton_candidate.py"),
        "concept_graph_contract.json": run_tests("test_concept_graph", "concept_graph_candidate.py"),
        "ffn_gate_contract.json": run_tests("test_ffn_gate", "ffn_gate_candidate.py"),
        "freeze_contract.json": run_tests("test_freeze_contract", "freeze_package.py"),
    }
    for name, report in reports.items():
        write_new(name, report)
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and path.suffix in (".py", ".json", ".md", ".csv"))
    manifest = {
        "status": "frozen_before_training", "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": protocol["candidate_id"],
        "sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "executable_datasets": ["assist2017"], "eligible_dataset_count": 8,
        "required_dataset_count": 5, "inherited_data_contract_requires_remote_rerun": True,
    }
    write_new("manifest.json", manifest)
    scope = validate_package(ROOT)
    print(json.dumps({
        "status": manifest["status"], "files": len(paths),
        "tests_run": sum(r["tests_run"] for r in reports.values()),
        "manifest_sha256": sha(ROOT / "manifest.json"), "scope": scope,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
