"""Run one richer-Full module ablation under the frozen fold-0 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F


OLD_ROOT = "/home/lpq/a2g_mambakt/boundary_complete_v2_assist2015_screen_20260811_v1"
OLD_RUNNER = Path(OLD_ROOT) / "run_boundary_complete_v2_assist2015_pair_v2_20260811.py"
SOURCE_ROOT = Path(
    "/home/lpq/a2g_mambakt/boundary_complete_richer_full_assist2015_screen_20260813_v2"
)
OLD_BASE_SHA = "0126c790c83cb79d159a2f048dc1001bd93c41c7467e9788474197d7b5624e56"
RICHER_BASE_SHA = "509189b84aeb383d712945ab90b5992e0d76359f292df3f9ddfca726993584b9"
CONTROL_RESULT = SOURCE_ROOT / "artifacts/control/result.json"
CONTROL_SHA = "4af660a5b3d2f2426edf17f2ae9af1f6d4b2e56351a6b1277bf6023b630cbc1d"
SPECS = {
    "no_split_boundary": {"use_split_boundary": 0},
    "no_direct_stat": {"stat_scale": 0.0},
    "no_prior_direct": {"prior_scale": 0.0},
    "no_direct_residuals": {"prior_scale": 0.0, "stat_scale": 0.0},
    "normalization_control": {"normalization_topology": "control"},
    "no_ssm": {"ssm_forward": "zero_noop"},
    "ssm_scale_0_5": {"ssm_output_scale": 0.5},
    "ssm_scale_0_75": {"ssm_output_scale": 0.75},
    "ssm_no_outer_residual": {"ssm_forward": "subtract_input"},
    "ssm_state_only": {"ssm_forward": "state_only"},
    "ssm_normalized_innovation": {"ssm_forward": "normalized_innovation"},
    "auc_rank_loss_0_02": {"loss_mode": "auc_rank", "loss_scale": 0.02},
    "prefix_group_loss_0_15": {
        "loss_mode": "prefix_group",
        "loss_scale": 0.15,
    },
}
METRICS = ("auc", "acc", "nll", "brier", "ece_15bin")
GATE = {
    "auc_min_strict": 0.001,
    "acc_min": -0.0005,
    "nll_max": 0.0005,
    "brier_max": 0.0005,
    "ece_15bin_max": 0.005,
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def record(path: Path, label: str) -> dict:
    path = Path(path)
    return {
        "label": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def checked_replace(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"replacement count mismatch for {old!r}: {count}")
    return source.replace(old, new)


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_control(control: dict) -> None:
    expected = {
        "dataset_id": "assist2015",
        "fold": 0,
        "seed": 42,
        "split": "validation-only",
        "test_access": False,
        "window_test_access": False,
    }
    mismatches = {
        key: {"expected": value, "actual": control.get(key)}
        for key, value in expected.items()
        if control.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"control contract mismatch: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=sorted(SPECS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical_gpu not in {"0", "1"}:
        raise RuntimeError("requires one declared physical GPU (0 or 1)")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires exactly one visible CUDA device")
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise RuntimeError(f"insufficient free GPU memory: {free}")

    if digest(CONTROL_RESULT) != CONTROL_SHA:
        raise RuntimeError("matched richer-Full control SHA changed")
    control = json.loads(CONTROL_RESULT.read_text(encoding="utf-8"))
    validate_control(control)

    candidate_id = f"ordinary_richer_full_ablation_{args.ablation}_v1"
    source = OLD_RUNNER.read_text(encoding="utf-8")
    source = checked_replace(source, OLD_ROOT, str(SOURCE_ROOT))
    source = checked_replace(source, OLD_BASE_SHA, RICHER_BASE_SHA)
    # Bound exploratory runs and expose per-epoch throughput/resource telemetry.
    source = checked_replace(
        source,
        "for epoch in range(1, 41):",
        "for epoch in range(1, int(os.environ.get('A2G_MAX_EPOCHS', '40')) + 1):\n            epoch_started = time.perf_counter()",
    )
    source = checked_replace(
        source,
        '"schedule": sampler.state_dict(),\n            }',
        '"schedule": sampler.state_dict(),\n                "epoch_seconds": time.perf_counter() - epoch_started,\n                "batches": len(losses),\n                "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),\n            }',
    )
    source = checked_replace(
        source,
        "if epoch - best_epoch >= 10:",
        "if epoch - best_epoch >= int(os.environ.get('A2G_PATIENCE', '10')):",
    )
    source = checked_replace(
        source,
        '"segment_local_full_control" if variant == "control"',
        f'"{candidate_id}" if variant == "control"',
    )
    namespace = {
        "__file__": str(OLD_RUNNER),
        "__name__": "a2g_dynamic_full_module_ablation",
    }
    exec(compile(source, str(OLD_RUNNER), "exec"), namespace)

    api, source_records = namespace["setup"]()
    base_class = api["Full"]
    overrides = SPECS[args.ablation]

    class Candidate(base_class):
        def __init__(self, *model_args, **model_kwargs):
            model_kwargs.update(
                {
                    key: value
                    for key, value in overrides.items()
                    if key not in {
                        "loss_mode",
                        "loss_scale",
                        "ssm_forward",
                        "ssm_output_scale",
                    }
                }
            )
            super().__init__(*model_args, **model_kwargs)
            if overrides.get("ssm_forward") == "zero_noop":
                def no_ssm_forward(module, x, scope_weight=None):
                    del scope_weight
                    # Preserve the frozen dropout substream while removing SSM signal.
                    zero = module.norm(torch.zeros_like(x))
                    return module.drop(zero)

                self.ssm.forward = types.MethodType(no_ssm_forward, self.ssm)
            if overrides.get("ssm_forward") == "subtract_input":
                original_ssm_forward = self.ssm.forward

                def subtract_input_ssm_forward(module, x, scope_weight=None):
                    del module
                    return original_ssm_forward(
                        x, scope_weight=scope_weight
                    ) - x

                self.ssm.forward = types.MethodType(
                    subtract_input_ssm_forward, self.ssm
                )
            if overrides.get("ssm_forward") == "state_only":
                def state_only_ssm_forward(module, x, scope_weight=None):
                    h = torch.zeros(
                        x.size(0), x.size(-1), device=x.device, dtype=x.dtype
                    )
                    outputs = []
                    transition_rate = x.new_tensor(math.log(2.0))
                    for index in range(x.size(1)):
                        candidate, gate, delta = module.in_proj(
                            x[:, index]
                        ).chunk(3, dim=-1)
                        candidate = torch.tanh(candidate)
                        gate = torch.sigmoid(gate)
                        step = module.input_decay_scale * F.softplus(delta)
                        retention = torch.exp(-transition_rate * step).clamp(
                            1e-4, 1.0 - 1e-4
                        )
                        h = retention * h + (1.0 - retention) * candidate
                        if scope_weight is not None:
                            gate = gate * scope_weight[:, index]
                        outputs.append(gate * h)
                    return module.drop(module.norm(torch.stack(outputs, dim=1)))

                self.ssm.forward = types.MethodType(
                    state_only_ssm_forward, self.ssm
                )
            if overrides.get("ssm_forward") == "normalized_innovation":
                original_ssm_forward = self.ssm.forward

                def normalized_innovation_ssm_forward(
                    module, x, scope_weight=None
                ):
                    normalized_input = module.norm(x)
                    normalized_state = original_ssm_forward(
                        x, scope_weight=scope_weight
                    )
                    return normalized_state - normalized_input

                self.ssm.forward = types.MethodType(
                    normalized_innovation_ssm_forward, self.ssm
                )
            if "ssm_output_scale" in overrides:
                original_ssm_forward = self.ssm.forward
                output_scale = float(overrides["ssm_output_scale"])

                def scaled_ssm_forward(module, x, scope_weight=None):
                    del module
                    return output_scale * original_ssm_forward(
                        x, scope_weight=scope_weight
                    )

                self.ssm.forward = types.MethodType(
                    scaled_ssm_forward, self.ssm
                )

    device = torch.device("cuda:0")
    data = namespace["data_bundle"](api)
    cfg = data["cfg"]

    namespace["seed_all"]()
    base_model = base_class(
        cfg["num_c"], cfg["num_q"], **namespace["kwargs"](cfg)
    ).to(device)
    base_initialization = namespace["initialize_prior"](
        base_model, data["train_base"]
    )
    namespace["seed_all"]()
    candidate_model = Candidate(
        cfg["num_c"], cfg["num_q"], **namespace["kwargs"](cfg)
    ).to(device)
    candidate_initialization = namespace["initialize_prior"](
        candidate_model, data["train_base"]
    )

    base_parameters = dict(base_model.named_parameters())
    candidate_parameters = dict(candidate_model.named_parameters())
    if base_parameters.keys() != candidate_parameters.keys():
        raise RuntimeError("ablation changed trainable parameter names")
    unequal = [
        name
        for name in base_parameters
        if not torch.equal(base_parameters[name], candidate_parameters[name])
    ]
    if unequal:
        raise RuntimeError(f"ablation changed initial parameters: {unequal[:5]}")
    actual_overrides = {
        key: getattr(candidate_model, key)
        for key in overrides
        if key not in {
            "loss_mode",
            "loss_scale",
            "ssm_forward",
            "ssm_output_scale",
        }
    }
    if "ssm_forward" in overrides:
        actual_overrides["ssm_forward"] = overrides["ssm_forward"]
    if "ssm_output_scale" in overrides:
        actual_overrides["ssm_output_scale"] = float(
            overrides["ssm_output_scale"]
        )
    if "loss_mode" in overrides:
        actual_overrides["loss_mode"] = overrides["loss_mode"]
        actual_overrides["loss_scale"] = float(overrides["loss_scale"])
    expected_overrides = {
        key: bool(value) if key.startswith("use_") else value
        for key, value in overrides.items()
    }
    if actual_overrides != expected_overrides:
        raise RuntimeError(
            f"ablation override mismatch: {actual_overrides} != {expected_overrides}"
        )

    loader, _ = namespace["uid_loader"](
        api, data["valid_uid"], data["valid_meta"], 128, 0
    )
    dcur = namespace["to_device"](next(iter(loader))["dcur"], device)
    base_model.eval()
    candidate_model.eval()
    with torch.no_grad():
        base_prediction = base_model(dcur)
        candidate_prediction = candidate_model(dcur)
    forward_diff = float(
        (base_prediction - candidate_prediction).abs().max().detach().cpu()
    )
    if forward_diff <= 0.0 and "loss_mode" not in overrides:
        raise RuntimeError("ablation has no effect on the checked forward path")

    loss_probe = None
    if "loss_mode" in overrides:
        scale = float(overrides["loss_scale"])

        def candidate_loss(prediction, batch):
            mask = batch["smasks"].bool()
            probability = prediction[:, 1:][mask].double()
            target = batch["shft_rseqs"][mask].double()
            base = F.binary_cross_entropy(probability, target)
            if overrides["loss_mode"] == "auc_rank":
                logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
                positive = logits[target > 0.5]
                negative = logits[target <= 0.5]
                if positive.numel() and negative.numel():
                    rank = 0.5 * (
                        F.softplus(negative.mean() - positive).mean()
                        + F.softplus(negative - positive.mean()).mean()
                    )
                    return base + scale * rank
                return base
            if overrides["loss_mode"] == "prefix_group":
                token_loss = F.binary_cross_entropy(
                    prediction[:, 1:].double(),
                    batch["shft_rseqs"].double(),
                    reduction="none",
                )
                width = token_loss.size(1)
                position = torch.arange(width, device=token_loss.device)
                groups = (
                    position < max(1, width // 4),
                    (position >= max(1, width // 4))
                    & (position < max(2, width // 2)),
                    position >= max(2, width // 2),
                )
                group_losses = []
                for group in groups:
                    selected = mask & group.unsqueeze(0)
                    if selected.any():
                        group_losses.append(token_loss[selected].mean())
                balanced = torch.stack(group_losses).mean()
                return (1.0 - scale) * base + scale * balanced
            raise RuntimeError(f"unknown loss mode: {overrides['loss_mode']}")

        base_loss_probe = namespace["loss_for"](base_prediction, dcur)
        candidate_loss_probe = candidate_loss(candidate_prediction, dcur)
        loss_probe = {
            "base": float(base_loss_probe.detach().cpu()),
            "candidate": float(candidate_loss_probe.detach().cpu()),
            "abs_diff": float(
                (candidate_loss_probe - base_loss_probe).abs().detach().cpu()
            ),
        }
        if loss_probe["abs_diff"] <= 0.0:
            raise RuntimeError("loss ablation has no effect on the checked batch")
        namespace["loss_for"] = candidate_loss

    preflight = {
        "status": "pass_real_forward_ablation_contract",
        "candidate_id": candidate_id,
        "dataset_id": "assist2015",
        "fold": 0,
        "seed": 42,
        "split": "validation-only",
        "test_access": False,
        "window_test_access": False,
        "ablation": args.ablation,
        "overrides": overrides,
        "actual_overrides": actual_overrides,
        "base_sha256": RICHER_BASE_SHA,
        "runner": record(Path(__file__).resolve(), "module ablation runner"),
        "dynamic_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "control_result": record(CONTROL_RESULT, "matched richer-Full control"),
        "gpu": {
            "physical_index": int(physical_gpu),
            "name": torch.cuda.get_device_name(0),
            "free_bytes": int(free),
            "total_bytes": int(total),
        },
        "trainable_parameter_names_equal": True,
        "initial_trainable_parameters_bit_exact": True,
        "forward_max_abs_diff": forward_diff,
        "loss_probe": loss_probe,
        "base_initialization": base_initialization,
        "candidate_initialization": candidate_initialization,
        "source_records": source_records,
    }
    print(json.dumps(preflight, sort_keys=True), flush=True)
    del base_model, candidate_model, dcur, base_prediction, candidate_prediction
    torch.cuda.empty_cache()

    output = args.root.resolve() / "artifacts" / args.ablation
    if (output / "result.json").exists() or (output / "summary.json").exists():
        raise FileExistsError("refusing to overwrite existing candidate artifacts")
    api["Full"] = Candidate
    state, initialization = namespace["initial_state"](api, data, device)
    train_started = time.perf_counter()
    result = namespace["train"](
        api, data, "control", state, output, device
    )
    result.setdefault("runtime", {})["wrapper_elapsed_seconds"] = (
        time.perf_counter() - train_started
    )
    result["runtime"]["gpu_peak_memory_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    result["runtime"]["gpu_peak_memory_reserved_bytes"] = int(
        torch.cuda.max_memory_reserved(device)
    )

    control_metrics = control["metrics"]["overall"]
    candidate_metrics = result["metrics"]["overall"]
    delta = {
        metric: candidate_metrics[metric] - control_metrics[metric]
        for metric in METRICS
    }
    checks = {
        "auc": delta["auc"] > GATE["auc_min_strict"],
        "acc": delta["acc"] >= GATE["acc_min"],
        "nll": delta["nll"] <= GATE["nll_max"],
        "brier": delta["brier"] <= GATE["brier_max"],
        "ece_15bin": delta["ece_15bin"] <= GATE["ece_15bin_max"],
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "status": (
            "pass_strict_assist_first_gate"
            if passed
            else "fail_fast_strict_assist_first_gate"
        ),
        "candidate_id": candidate_id,
        "dataset_id": "assist2015",
        "fold": 0,
        "seed": 42,
        "split": "validation-only",
        "test_access": False,
        "window_test_access": False,
        "ablation": args.ablation,
        "overrides": overrides,
        "control_result": record(CONTROL_RESULT, "matched richer-Full control"),
        "candidate_result": result["result"],
        "delta_candidate_minus_control": delta,
        "gate": GATE,
        "gate_semantics": "strict_delta_auc_greater_than_0.001",
        "gate_checks": checks,
        "gate_pass": passed,
        "next_action": (
            "eligible_for_next_small_dataset"
            if passed
            else "stop_candidate_without_cross_dataset_expansion"
        ),
        "preflight": preflight,
        "initialization": initialization,
    }
    summary_path = output / "summary.json"
    write_new(summary_path, summary)
    print(
        json.dumps(
            {**summary, "summary": record(summary_path, "strict gate summary")},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
