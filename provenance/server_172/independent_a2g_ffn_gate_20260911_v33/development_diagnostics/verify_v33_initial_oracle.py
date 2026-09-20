"""Independently derive six zero gates from the unchanged V32 FFN geometry."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=WORK / "independent_a2g_ffn_gate_20260911_v33")
    parser.add_argument("--control", type=Path, default=WORK / "independent_a2g_concept_graph_20260911_v32")
    parser.add_argument("--auditors", type=Path, default=WORK / "v33_terminal_audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("independent initializer evidence already exists")
    package, control, auditors = (path.resolve() for path in (args.package, args.control, args.auditors))
    protocol = load(package / "protocol.json")
    parent_manifest = load(control / "manifest.json")
    if sha(control / "manifest.json") != sha(package / "reference_v32_manifest.json"):
        raise RuntimeError("V32 manifest binding differs")
    for name, expected in parent_manifest["sha256"].items():
        if sha(control / name) != expected:
            raise RuntimeError(f"frozen V32 source changed: {name}")
    for name in ("a2g_incumbent.py", "factorized_input_candidate.py", "attempt_stage_candidate.py",
                 "history_pace_candidate.py", "prequential_newton_candidate.py", "concept_graph_candidate.py"):
        if sha(package / name) != parent_manifest["sha256"][name]:
            raise RuntimeError(f"common source changed: {name}")
    sys.path.insert(0, str(auditors))
    import torch
    from audit_ffn_gate_control import (
        CUDA_CHECK_NAMES, RETAINED_CUDA_CHECK_NAMES, reconstruct_new_parameters,
        verify_initial_states, verify_protocol_parity,
    )

    torch.set_num_threads(1)
    verify_protocol_parity(protocol, load(control / "protocol.json"))
    cpu = load(package / "cpu_contract.json")
    if CUDA_CHECK_NAMES != (
        set(cpu["checks"]) - {"cpu_audit_did_not_initialize_cuda"}
    ) | {"initialization_cuda_rng_identical_to_v32"}:
        raise RuntimeError("independent auditor and V33 CPU check names differ")
    if RETAINED_CUDA_CHECK_NAMES != (
        set(cpu["retained_v32_contract"]["checks"]) - {"cpu_audit_did_not_initialize_cuda"}
    ) | {"initialization_cuda_rng_identical_to_v28"}:
        raise RuntimeError("independent auditor and retained V32 CPU check names differ")
    before = torch.get_rng_state().clone()
    expected = reconstruct_new_parameters(control, protocol)
    if not torch.equal(before, torch.get_rng_state()) or "ffn_gate_candidate" in sys.modules:
        raise RuntimeError("independent oracle changed caller RNG or imported the candidate")
    parent_module = importlib.import_module("concept_graph_candidate")
    sys.path.insert(0, str(package / "sources"))
    sys.path.insert(0, str(package))
    from ffn_gate_candidate import A2GMambaKT

    config = load(package / "data_config.train_fold_only.json")["assist2017"]
    kwargs = dict(protocol["model_kwargs"])
    kwargs.pop("use_ffn_gate")
    torch.default_generator.manual_seed(42)
    parent = parent_module.A2GMambaKT(config["num_c"], config["num_q"], **kwargs)
    torch.default_generator.manual_seed(42)
    candidate = A2GMambaKT(config["num_c"], config["num_q"], **protocol["model_kwargs"])
    report = verify_initial_states(candidate.state_dict(), parent.state_dict(), expected, 512)
    if report["shared_tensors"] != 105 or report["new_parameters"] != 394752 or torch.cuda.is_initialized():
        raise RuntimeError("production initialization contract differs")
    report.update({
        "candidate_id": protocol["candidate_id"], "candidate_sha256": sha(package / "ffn_gate_candidate.py"),
        "parent_manifest_sha256": sha(control / "manifest.json"),
        "auditor_source_sha256": sha(auditors / "audit_ffn_gate_control.py"),
        "verification_script_sha256": sha(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False, "trained_weights_read": False, "training_started": False,
        "independent_oracle_imported_candidate": False,
        "scope": "Fresh random initializers without data priors; remote production preflight separately checks empirical priors and the persisted V32 random initial state.",
        "torch": torch.__version__, "real_data_access": False,
    })
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
