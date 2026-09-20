"""Compare a local run's initial tensors to the original server's real-data path."""

import argparse
import ast
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from a2g.experiment import load_data
from a2g.provenance import sha256, write_json


def declared_umk_scalars(config):
    if config.get("architecture") != "v45_umk":
        return {}
    flags = config["model"]
    return {
        name: value
        for flag, name, value in (
            ("use_umk_ssm", "ssm.umk_alpha", 0.1),
            ("use_umk_attn", "umk_beta", 0.5),
        )
        if flags.get(flag, 0)
    }


def compare_production_states(saved, current, config):
    scalars = declared_umk_scalars(config)
    if list(current) != [name for name in saved if name not in scalars]:
        raise ValueError("production common state keys/order changed")
    if set(saved) - set(current) != set(scalars):
        raise ValueError("undeclared or missing UMK state")
    mismatches = [
        name for name, value in current.items() if not torch.equal(value, saved[name])
    ]
    if mismatches:
        raise ValueError(f"production tensors differ: {mismatches}")
    for name, value in scalars.items():
        actual = saved[name]
        if actual.shape != torch.Size([]) or not torch.equal(actual, actual.new_tensor(value)):
            raise ValueError(f"declared scalar initialization differs: {name}")
    return scalars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assist2017"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    reference = root / "provenance/server_172/independent_a2g_ffn_gate_20260911_v33"
    v43 = root / "provenance/server_172/independent_a2g_modal_residual_20260913_v43"
    sys.path[:0] = [str(reference), str(reference / "sources"), str(v43)]
    run = json.loads((args.directory / "run.json").read_text(encoding="utf-8"))
    config = run["config"]
    if config["architecture"] in {"v43", "v44_aligned_history", "v45_umk"}:
        from modal_residual_candidate import A2GMambaKT as ServerModel
    elif config["architecture"] == "v33":
        from ffn_gate_candidate import A2GMambaKT as ServerModel
    else:
        raise ValueError("no server initialization reference for this architecture")
    bundles, binding = load_data(config, args.data_dir)
    original_runner = reference / "run_independent.py"
    tree = ast.parse(original_runner.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_prior"
    )
    namespace = {"torch": torch, "DataLoader": DataLoader}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), str(original_runner), "exec"
        ),
        namespace,
    )
    torch.manual_seed(42)
    model_config = dict(config["model"])
    for key in ("use_umk_ssm", "use_umk_attn", "use_umk_rwce", "umk_lambda0"):
        model_config.pop(key, None)
    model = ServerModel(
        config["data"]["num_c"], config["data"]["num_q"], **model_config
    )
    prior = namespace["initialize_prior"](model, bundles["train"]["base"])
    saved = torch.load(
        args.directory / "initial_state.pt", map_location="cpu", weights_only=True
    )
    current = model.state_dict()
    scalars = compare_production_states(saved, current, config)
    if prior != run["prior"]:
        raise ValueError("training-only prior report differs")
    report = {
        "status": "server_original_production_initialization_exact",
        "state_tensors": len(saved),
        "parameters": sum(parameter.numel() for parameter in model.parameters()) + len(scalars),
        "initial_state_sha256": sha256(args.directory / "initial_state.pt"),
        "original_runner_sha256": sha256(original_runner),
        "data_binding": binding,
        "prior": prior,
        "gpu_used": False,
        "trained_checkpoint_loaded": False,
        "performance_claim": False,
    }
    if config["architecture"] == "v45_umk":
        report.update(
            status="server_original_plus_declared_umk_scalars_initialization_exact",
            parent_state_tensors=len(current),
            server_common_state_exact=True,
            declared_new_scalar_parameters=scalars,
        )
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
