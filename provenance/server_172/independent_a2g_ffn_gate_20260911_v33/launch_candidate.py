"""Fail-closed entry point for a newly frozen Assist2017-only Full pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PILOT = "assist2017"
FLOORS = {
    "assist2009_corrected_collapsed": 0.8566249815040318,
    "assist2012": 0.779282,
    "assist2015": 0.7337756158427048,
    "assist2017": 0.8174,
    "junyi2015": 0.804888,
    "nips_task34": 0.8273,
    "slepemapy": 0.800163,
    "statics2011": 0.8293542042910951,
}
GUARD_SHA256 = "3ae1226cf61851f1df3934a2438ca5d6ed7a4ff68e189467107fba6c9143bd2b"
CLOSED_OR_REFERENCE = {
    "single_trunk_retained_memory_readout_20260907_v1",
    "incumbent_a2g_factorized_input_residual_20260909_v12",
    "a2g_v12_history_only_recurrent_state_20260909_v15",
    "a2g_v12_causal_evidence_alignment_20260909_v16",
}


class ScopeError(ValueError):
    pass


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ScopeError(f"expected a JSON object: {Path(path).name}")
    return value


def validate_protocol(protocol, dataset, seed, variant):
    if dataset != PILOT:
        raise ScopeError("Assist2017 must pass first; other datasets are not executable")
    if type(seed) is not int or seed != 42 or variant != "full":
        raise ScopeError("only candidate Full / seed42 is executable")
    expected = {
        "phase": "assist2017_full_only",
        "datasets": [PILOT],
        "dataset_order": [PILOT],
        "seeds": [42],
        "train_folds": [1, 2, 3, 4],
        "validation_folds": [0],
        "controls": [],
        "execution_order": ["full"],
        "test_access": False,
        "window_test_access": False,
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            raise ScopeError(f"single-dataset protocol restriction violated: {field}")
    if protocol.get("reference_auc") != FLOORS:
        raise ScopeError("all eight reference floors must remain unchanged")
    eligible = protocol.get("eligible_datasets")
    if not isinstance(eligible, list) or len(eligible) != 8 or set(eligible) != set(FLOORS):
        raise ScopeError("the final eight-choose-five dataset scope must remain intact")
    admission = protocol.get("global_admission", {})
    if not isinstance(admission, dict):
        raise ScopeError("global_admission must be a JSON object")
    for field, value in {
        "eligible_dataset_count": 8,
        "required_dataset_count": 5,
        "same_datasets_for_baselines_and_all_modules": True,
        "mandatory_assist2017_precondition": True,
    }.items():
        if admission.get(field) != value:
            raise ScopeError(f"global objective restriction violated: {field}")
    gate = protocol.get("assist2017_first_gate", {})
    if not isinstance(gate, dict):
        raise ScopeError("assist2017_first_gate must be a JSON object")
    for field, value in {
        "strict_auc_floor": 0.8174,
        "independent_terminal_audit_required": True,
        "other_datasets_executable_in_this_package": False,
        "expansion_requires_new_frozen_package": True,
    }.items():
        if gate.get(field) != value:
            raise ScopeError(f"Assist2017 admission restriction violated: {field}")
    if protocol.get("baseline_training") != {"run": False, "incumbent_rerun": False}:
        raise ScopeError("baseline and incumbent retraining are forbidden")
    candidate = protocol.get("candidate_id")
    if not isinstance(candidate, str) or not candidate.strip() or candidate in CLOSED_OR_REFERENCE:
        raise ScopeError("a separately admitted new candidate is required")
    if protocol.get("variants") != {
        "full": {"use_attention_branch": 1, "use_evidence_branch": 1, "use_ssm_branch": 1}
    }:
        raise ScopeError("this phase permits no module-control or baseline training")
    training = protocol.get("training", {})
    if not isinstance(training, dict):
        raise ScopeError("training must be a JSON object")
    for field, value in {
        "pretrained_checkpoint": None,
        "initialization": "random_all_model_parameters_plus_training_only_empirical_priors",
        "max_epochs": 200,
        "patience": 20,
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "batch_size": 64,
        "evaluation_batch_size": 128,
        "batch_order": "fixed_csv_filtered_order",
        "deterministic_algorithms": True,
    }.items():
        if field not in training or training[field] != value:
            raise ScopeError(f"frozen training restriction violated: {field}")


def validate_package(package, dataset=PILOT, seed=42, variant="full"):
    # Reject a forbidden invocation before reading model code or invoking CUDA.
    if dataset != PILOT:
        raise ScopeError("Assist2017 must pass first; other datasets are not executable")
    root = Path(package).resolve(strict=True)
    protocol = load(root / "protocol.json")
    validate_protocol(protocol, dataset, seed, variant)
    manifest = load(root / "manifest.json")
    if (
        manifest.get("candidate_id") != protocol["candidate_id"]
        or manifest.get("status") != "frozen_before_training"
    ):
        raise ScopeError("a matching pretraining manifest is required")
    bindings = manifest.get("sha256")
    required = {
        "protocol.json", "run_independent.py", "a2g_single_trunk_candidate.py",
        "gpu_guard.py", "cpu_contract.json", "data_contract.json",
        "gpu_guard_contract.json", "runner_contract.json",
    }
    if not isinstance(bindings, dict) or not required.issubset(bindings):
        raise ScopeError("manifest is missing a required execution or audit file")
    for relative, expected in bindings.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ScopeError(f"manifest binding failed: {relative}")
    if bindings["gpu_guard.py"] != GUARD_SHA256:
        raise ScopeError("the independently tested GPU ownership guard must be preserved")
    for name in ("cpu_contract", "data_contract", "gpu_guard_contract", "runner_contract"):
        contract = load(root / f"{name}.json")
        if contract.get("status") != "pass":
            raise ScopeError(f"pretraining contract has not passed: {name}")
        checks = contract.get("checks")
        if checks is not None and (
            not isinstance(checks, dict) or not checks
            or any(value is not True for value in checks.values())
        ):
            raise ScopeError(f"pretraining contract contains failed checks: {name}")
    cell = root / "artifacts" / PILOT / "seed42"
    if cell.exists():
        raise ScopeError("pilot artifacts already exist; no silent retry or rerun")
    return {
        "status": "eligible_assist2017_only_execution_scope",
        "package": str(root),
        "candidate_id": protocol["candidate_id"],
        "manifest_sha256": digest(root / "manifest.json"),
        "dataset": PILOT,
        "seed": 42,
        "variant": "full",
        "strict_auc_floor": FLOORS[PILOT],
        "other_datasets_executable": False,
        "independent_terminal_audit_required": True,
        "expansion_requires_new_frozen_package": True,
        "gpu_used": False,
        "training_launched": False,
        "paper_goal_complete": False,
        "scope_only": "This check does not establish a scientific effect or reserve a GPU.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset", default=PILOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="full")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = validate_package(args.package, args.dataset, args.seed, args.variant)
        if args.execute:
            if (
                sys.platform != "linux" or os.getuid() != 1007
                or os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            ):
                raise ScopeError("execution requires the authorized lpq GPU1 environment")
        print(json.dumps(report, sort_keys=True), flush=True)
        if args.check_only:
            return 0
        command = [
            sys.executable, str(Path(report["package"]) / "run_independent.py"),
            "--execute", "--dataset", PILOT, "--seed", "42", "--variant", "full",
        ]
        # The frozen runner retains its host check, global lock and GPU watchdog.
        return subprocess.run(command, cwd=report["package"], check=False).returncode
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({
            "status": "execution_scope_blocked", "reason": str(error),
            "training_launched": False, "gpu_used": False,
            "paper_goal_complete": False,
        }, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
