"""Verify v45 against a frozen v43 checkpoint, then run fresh single-switch smokes."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "versions/v43_lowmem"
REFERENCE_RUN = REFERENCE / "runs/full_fold0_seed42_20260918_0325"


def shapes(args):
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from a2g.experiment import load_data, loader, make_model

    torch.set_num_threads(1)
    report = {}
    for suffix in ("ssm", "attn", "rwce", "all"):
        directory = args.output / f"smoke_{suffix}"
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        config = run["config"]
        bundles, _ = load_data(config, ROOT / "data/assist2017")
        state = torch.load(directory / "selected_model.pt", map_location="cpu", weights_only=True)
        initial = torch.load(directory / "initial_state.pt", map_location="cpu", weights_only=True)
        report[suffix] = {
            "scalar_updates": {
                name: {"initial": initial[name].item(), "trained": state[name].item()}
                for name in ("ssm.umk_alpha", "umk_beta") if name in state
            },
            "shapes": {},
        }
        for dtype, size in ((torch.float32, args.smoke_batch_size), (torch.float64, 2)):
            model = make_model(config, "full").to(device=args.device, dtype=dtype).eval()
            model.load_state_dict(state, strict=True)
            batch = next(iter(loader(bundles["validation"], size)))
            dcur = {name: value.to(args.device) for name, value in batch["dcur"].items()}
            captured = {}

            def capture(name, destination=captured):
                def hook(_module, _inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    destination[name] = list(value.shape)
                return hook

            handles = [
                model.input.register_forward_hook(capture("token")),
                model.ssm.register_forward_hook(capture("ssm")),
            ]
            handles.extend(
                block["attn"].register_forward_hook(capture(f"attention_{i}"))
                for i, block in enumerate(model.blocks)
            )
            try:
                with torch.no_grad():
                    prediction, fused = model(dcur, qtest=True)
            finally:
                for handle in handles:
                    handle.remove()
            expected = [size, config["model"]["seq_len"], config["model"]["d_model"]]
            if any(shape != expected for shape in captured.values()):
                raise ValueError(f"invalid {suffix} intermediate geometry: {captured}")
            if list(prediction.shape) != expected[:2] or list(fused.shape) != [*expected[:2], expected[2] + 6]:
                raise ValueError("invalid prediction/fused geometry")
            if prediction.dtype != dtype or not torch.isfinite(prediction).all() or not torch.isfinite(fused).all():
                raise ValueError("nonfinite or wrong-dtype model output")
            captured["prediction"], captured["fused"] = list(prediction.shape), list(fused.shape)
            report[suffix]["shapes"][str(dtype)] = captured
            del model, prediction, fused
    with (args.output / "integration_shapes.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"integration_shapes": "passed", "switches": list(report)}, indent=2))


def probe(args):
    import numpy as np
    import torch

    sys.path.insert(0, str(args.source_root))
    import a2g
    from a2g.experiment import (
        A2GDropoutSubstreamAlignment,
        evaluate,
        load_config,
        load_data,
        loader,
        make_model,
        seed_all,
    )

    if Path(a2g.__file__).resolve().parent != args.source_root.resolve() / "a2g":
        raise RuntimeError("probe imported the wrong source")
    torch.set_num_threads(1)
    seed_all(42)
    config_path = (
        ROOT / "configs/assist2017_v45_umk_off.json"
        if args.source_root.resolve() == ROOT / "src"
        else REFERENCE / "configs/assist2017_v43_lowmem.json"
    )
    config = load_config(config_path)
    bundles, _ = load_data(config, ROOT / "data/assist2017")
    model = make_model(config, "full")
    model.load_state_dict(
        torch.load(REFERENCE_RUN / "selected_model.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    model.to(device=args.device, dtype=dtype).eval()
    batch = next(iter(loader(bundles["validation"], 2)))
    dcur = {name: value.to(args.device) for name, value in batch["dcur"].items()}
    shapes = {}

    def capture(_module, _inputs, result):
        shapes["ssm"] = list(result.shape)

    hook = model.ssm.register_forward_hook(capture)
    with torch.no_grad():
        prediction, fused = model(dcur, qtest=True)
    hook.remove()
    arrays = {
        "prediction": prediction.cpu().numpy(),
        "fused": fused.cpu().numpy(),
    }
    shapes["prediction"], shapes["fused"] = list(prediction.shape), list(fused.shape)
    if args.precision == "float32":
        alignment = A2GDropoutSubstreamAlignment(model, base_seed=42)
        try:
            metrics, values = evaluate(
                model, bundles["validation"], alignment, torch.device(args.device), 128
            )
        finally:
            alignment.close()
        arrays["full_validation_probability"] = values["probability"]
        arrays["full_validation_label"] = values["label"]
        shapes["full_validation"] = len(values["label"])
    else:
        metrics = None
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({
        "source": str(Path(a2g.__file__).resolve()),
        "precision": args.precision, "device": args.device,
        "strict_checkpoint_load": True, "shapes": shapes, "metrics": metrics,
    }, indent=2))


def verify(args):
    import numpy as np

    sys.path.insert(0, str(ROOT / "tools"))
    from audit_run import audit, digest
    from verify_version import verify as verify_frozen

    args.output.mkdir(parents=True, exist_ok=False)
    frozen_report = verify_frozen(REFERENCE)
    result = json.loads((REFERENCE_RUN / "result.json").read_text(encoding="utf-8"))
    if digest(REFERENCE_RUN / "selected_model.pt") != result["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint digest mismatch")
    report = {
        "frozen_reference": frozen_report,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "device": args.device,
        "checkpoint_compatibility": {},
        "smokes": {},
        "full_training_started": False,
        "performance_claim": False,
    }

    def command(arguments, name):
        completed = subprocess.run(
            [sys.executable, "-B", *map(str, arguments)],
            cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True, capture_output=True, timeout=1200, check=False,
        )
        (args.output / f"{name}.log").write_text(
            json.dumps({"command": completed.args, "return_code": completed.returncode})
            + "\n" + completed.stdout + "\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(f"{name} failed: {completed.stderr[-4000:]}")
        return completed.stdout

    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="umk_probe_", dir=work) as temporary:
        temporary = Path(temporary).resolve()
        if not temporary.is_relative_to(work.resolve()):
            raise ValueError("probe scratch directory must stay in project work/")
        for precision in ("float32", "float64"):
            outputs = []
            for name, source in (("reference", REFERENCE / "src"), ("development", ROOT / "src")):
                destination = temporary / f"{name}_{precision}.npz"
                command([
                    __file__, "--probe", "--source-root", source,
                    "--precision", precision, "--device", args.device, "--output", destination,
                ], f"{name}_{precision}")
                with np.load(destination, allow_pickle=False) as archive:
                    outputs.append({key: archive[key] for key in archive.files})
            differences = {}
            for name in outputs[0]:
                expected, actual = outputs[0][name], outputs[1][name]
                if expected.shape != actual.shape:
                    raise ValueError(f"probe shape mismatch: {name}")
                difference = float(np.max(np.abs(expected - actual)))
                if not np.array_equal(expected, actual) or difference >= 1e-9:
                    raise ValueError(f"nonidentical all-off {precision} {name}: {difference}")
                differences[name] = {"max_abs_error": difference, "bitwise_equal": True, "shape": list(actual.shape)}
            report["checkpoint_compatibility"][precision] = differences
            print(f"Checkpoint {precision}: exact", flush=True)

    for suffix in ("ssm", "attn", "rwce", "all"):
        run_directory = args.output / f"smoke_{suffix}"
        command([
            ROOT / "run.py", "smoke", "--config", ROOT / f"configs/assist2017_v45_umk_{suffix}.json",
            "--output", run_directory, "--data-dir", ROOT / "data/assist2017",
            "--device", args.device, "--smoke-batch-size", args.smoke_batch_size,
            "--smoke-steps", 2,
        ], f"smoke_{suffix}")
        smoke = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
        audited = audit(run_directory, ROOT / "src/a2g", ROOT / "data/assist2017")
        report["smokes"][suffix] = {
            "status": smoke["status"], "return_code": 0,
            "batch_size": args.smoke_batch_size, "optimizer_steps": 2,
            "audit": audited,
        }
        print(f"Smoke {suffix}: passed and CSV-bound", flush=True)
    shapes(args)
    report["successful"] = True
    (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"successful": True, "report": str(args.output / "verification.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--shapes-only", action="store_true")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=("float32", "float64"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke_batch_size < 1:
        parser.error("--smoke-batch-size must be positive")
    args.output = args.output.resolve()
    if args.probe:
        probe(args)
    elif args.shapes_only:
        shapes(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
