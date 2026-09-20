"""Reload one local checkpoint and verify predictions against the frozen CSV."""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assist2017"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.version_root.resolve() if args.version_root else Path(__file__).resolve().parents[1]
    if args.version_root:
        from verify_version import verify

        verify(root)
        import importlib.util
        import sys
        import subprocess

        package = importlib.util.find_spec("a2g")
        if (
            package is None
            or package.origin is None
            or Path(package.origin).resolve().parent != root / "src/a2g"
        ):
            # Execute this auditor with the frozen package first on sys.path.
            script = (
                "import runpy,sys; "
                "sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)]; "
                "target=sys.argv.pop(1); sys.argv[0]=target; "
                "runpy.run_path(target,run_name='__main__')"
            )
            command = [
                sys.executable, "-I", "-c", script, str(root / "src"),
                str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve()), *sys.argv[1:],
            ]
            raise SystemExit(subprocess.run(command, check=False).returncode)
    import numpy as np
    import torch

    from a2g.experiment import evaluate, load_data, make_model, seed_all
    from a2g.provenance import write_json
    from a2g.randomness import A2GDropoutSubstreamAlignment
    from audit_run import audit

    report = audit(args.directory, root / "src/a2g", args.data_dir)
    run = json.loads((args.directory / "run.json").read_text(encoding="utf-8"))
    from a2g.provenance import sha256

    for name, expected in run["source"].items():
        if sha256(root / "src/a2g" / name) != expected:
            raise ValueError(
                "active code differs from the run; reevaluate under its original source snapshot"
            )
    result = json.loads((args.directory / "result.json").read_text(encoding="utf-8"))
    if run["mode"] != "train":
        parser.error(
            "only a completed full run can receive a full CSV-bound reevaluation"
        )
    torch.set_num_threads(1)
    seed_all(42)
    bundles, data_report = load_data(run["config"], args.data_dir)
    model = make_model(run["config"], run["variant"])
    model.load_state_dict(
        torch.load(
            args.directory / "selected_model.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    model.to(args.device)
    alignment = A2GDropoutSubstreamAlignment(model, base_seed=42)
    try:
        metrics, arrays = evaluate(
            model,
            bundles["validation"],
            alignment,
            args.device,
            run["effective_evaluation_batch_size"],
            epoch=100,
        )
    finally:
        if alignment._active:
            alignment.abort_step()
        alignment.close()
    with np.load(
        args.directory / "validation_predictions.npz", allow_pickle=False
    ) as saved:
        equality = {
            key: bool(np.array_equal(value, saved[key]))
            for key, value in arrays.items()
        }
    if not all(equality.values()) or metrics != result["metrics"]:
        raise ValueError(
            f"checkpoint reevaluation differs: fields={equality}, metrics={metrics}"
        )
    report.update(
        status="checkpoint_and_csv_bound_predictions_verified",
        data=data_report,
        exact_prediction_fields=equality,
        metrics=metrics,
        run_sha256=sha256(args.directory / "run.json"),
        result_sha256=sha256(args.directory / "result.json"),
        checkpoint_sha256=sha256(args.directory / "selected_model.pt"),
        predictions_sha256=sha256(args.directory / "validation_predictions.npz"),
        limits=[
            "Same runtime and same evaluation implementation; not independent model retraining.",
            "Matched-baseline comparisons and positive retrained ablations remain unverified.",
        ],
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
