"""Bind exit, initialization, source, CSV and checkpoint audits for one Full run."""

import argparse
import json
from pathlib import Path

from audit_protocol import compare_metrics
from audit_run import audit, digest
from verify_version import verify


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command_option(command, name):
    if command.count(name) != 1:
        raise ValueError(f"process command must specify {name} exactly once")
    index = command.index(name)
    if index + 1 >= len(command):
        raise ValueError(f"missing command value for {name}")
    return command[index + 1]


def verify_exit(process, directory, version, run):
    if (
        process.get("status") != "exited"
        or type(process.get("return_code")) is not int
        or process["return_code"] != 0
        or process.get("automatic_retries") != 0
        or not process.get("ended_at")
    ):
        raise ValueError("normal observed process exit is required")
    command = process.get("command", [])
    if len(command) < 4 or command[1] != "-u" or command[3] != "train":
        raise ValueError("process did not run a complete training command")
    if Path(command[2]).resolve() != (version / "run.py").resolve():
        raise ValueError("process entrypoint is not the frozen version")
    if Path(process["cwd"]).resolve() != version.resolve():
        raise ValueError("process working directory differs from the version")
    if Path(command_option(command, "--output")).resolve() != directory.resolve():
        raise ValueError("process output is not this run")
    if command_option(command, "--device") != run["runtime"]["device"]:
        raise ValueError("runtime device differs from command")
    command_config = read_json(command_option(command, "--config"))
    if command_config != run["config"]:
        raise ValueError("process configuration differs from run.json")
    return Path(command_option(command, "--data-dir")).resolve()


def verify_supporting_evidence(
    initialization, reevaluation, run, result, *, run_sha256, result_sha256
):
    expected_status = "server_original_production_initialization_exact"
    if run["config"].get("architecture") == "v45_umk":
        flags = run["config"]["model"]
        scalars = {
            name: value for flag, name, value in (
                ("use_umk_ssm", "ssm.umk_alpha", 0.1),
                ("use_umk_attn", "umk_beta", 0.5),
            ) if flags.get(flag, 0)
        }
        expected_status = "server_original_plus_declared_umk_scalars_initialization_exact"
        if (
            initialization.get("server_common_state_exact") is not True
            or initialization.get("declared_new_scalar_parameters") != scalars
        ):
            raise ValueError("UMK scalar initialization is missing or mismatched")
    if (
        initialization.get("status") != expected_status
        or initialization.get("initial_state_sha256") != run["initial_state_sha256"]
        or initialization.get("trained_checkpoint_loaded") is not False
        or initialization.get("gpu_used") is not False
        or initialization.get("prior") != run["prior"]
        or initialization.get("data_binding") != run["data"]
    ):
        raise ValueError("server-derived initialization evidence is missing or mismatched")
    fields = {"label", "probability", "learner_uid", "csv_row_index", "position"}
    equality = reevaluation.get("exact_prediction_fields", {})
    if (
        reevaluation.get("status") != "checkpoint_and_csv_bound_predictions_verified"
        or reevaluation.get("run_mode") != "train"
        or reevaluation.get("data") != run["data"]
        or set(equality) != fields
        or any(value is not True for value in equality.values())
    ):
        raise ValueError("full checkpoint reevaluation evidence is missing or mismatched")
    compare_metrics(result["metrics"], reevaluation["metrics"])
    if reevaluation.get("run_sha256") != run_sha256:
        raise ValueError("reevaluation must bind this run, not just equal metric values")
    for key in ("checkpoint_sha256", "predictions_sha256", "result_sha256"):
        if key not in reevaluation:
            raise ValueError(f"reevaluation is missing {key}")
    for key in ("checkpoint_sha256", "predictions_sha256"):
        if reevaluation[key] != result[key]:
            raise ValueError(f"reevaluation {key} differs from result")
    if reevaluation["result_sha256"] != result_sha256:
        raise ValueError("reevaluation result hash differs")


def terminal(directory, control, version, initialization_path, reevaluation_path):
    directory, control, version = map(
        lambda value: Path(value).resolve(), (directory, control, version)
    )
    version_report = verify(version)
    run, result = read_json(directory / "run.json"), read_json(directory / "result.json")
    if run.get("mode") != "train" or run.get("variant") != "full":
        raise ValueError("only completed Full training can pass the admission gate")
    data = verify_exit(read_json(control / "process.json"), directory, version, run)
    audit_report = audit(directory, version / "src/a2g", data)
    initialization, reevaluation = read_json(initialization_path), read_json(reevaluation_path)
    verify_supporting_evidence(
        initialization, reevaluation, run, result,
        run_sha256=digest(directory / "run.json"),
        result_sha256=digest(directory / "result.json"),
    )
    if not (control / "stdout.log").is_file() or not (control / "stderr.log").is_file():
        raise ValueError("durable process logs are missing")
    return {
        "status": "complete_local_terminal_audit",
        "dataset": run["config"]["dataset"],
        "seed": run["config"]["seed"],
        "validation_folds": run["config"]["validation_folds"],
        "training_protocol": audit_report["training_protocol"],
        "metrics": audit_report["recomputed_metrics"],
        "strict_auc_floor": 0.8174,
        "assist2017_admission_pass": audit_report["point_gate_pass"],
        "normal_process_exit": True,
        "frozen_version": version_report,
        "run_directory": str(directory),
        "run_sha256": digest(directory / "run.json"),
        "result_sha256": digest(directory / "result.json"),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "predictions_sha256": result["predictions_sha256"],
        "initialization_audit_sha256": digest(initialization_path),
        "checkpoint_reevaluation_sha256": digest(reevaluation_path),
        "process_sha256": digest(control / "process.json"),
        "stdout_sha256": digest(control / "stdout.log"),
        "stderr_sha256": digest(control / "stderr.log"),
        "auditor_sha256": digest(__file__),
        "other_datasets_started": False,
        "independent_ablation_runs_verified": 0,
        "test_access": False,
        "paper_goal_complete": False,
        "limits": [
            "Audits one local random-initialized Full run, not repeated-seed robustness.",
            "Validation checkpoint and architecture selection remain exploratory.",
            "Same-five-of-eight baseline and module gates remain separate requirements.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--version", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--reevaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = terminal(
        args.directory, args.control, args.version, args.initialization, args.reevaluation
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
