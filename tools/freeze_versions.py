"""Copy complete local A2G versions without links, shared code, or overwrites."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from verify_version import bundle_files, sha256, verify


VERSIONS = {
    "v46_compact_r3": {
        "architecture": "v46",
        "config": "assist2017_v46_compact.json",
        "checkpoint_steps": 16,
        "parent": "v46_compact_r2",
        "description": "v46_compact_r2 plus make_model architecture routing for v46 (A2GModal). Complete runnable v46 candidate: shared weight-tied embeddings (-16 pct params), AdamW wd=1e-4, seeded shuffle, 3-epoch warmup + cosine, V46 protocol id. Supersedes stale v46_compact and non-runnable v46_compact_r2. No performance claim.",
        "design_document": "V46_COMPACT.md",
    },
    "v46_compact_r2": {
        "architecture": "v46",
        "config": "assist2017_v46_compact.json",
        "checkpoint_steps": 16,
        "parent": "v46_compact",
        "description": "v46_compact refreshed with registered V46 training protocol (protocol.py) and config training_protocol id; shared weight-tied embeddings, AdamW wd=1e-4, seeded shuffle, 3-epoch warmup + cosine. Supersedes v46_compact (stale, pre-protocol). No performance claim.",
        "design_document": "V46_COMPACT.md",
    },
    "v46_compact": {
        "architecture": "v46",
        "config": "assist2017_v46_compact.json",
        "checkpoint_steps": 16,
        "parent": "v43_lowmem_r3",
        "description": "v43 architecture with shared (weight-tied) target/history embeddings (16 pct fewer params) plus upgraded protocol: AdamW wd=1e-4, seeded per-epoch shuffle, 3-epoch warmup + cosine decay. Keeps all v43 modules; efficiency/generalization target, no performance claim.",
        "design_document": "V46_COMPACT.md",
    },
    "v45_umk_r2": {
        "architecture": "v45_umk",
        "config": "assist2017_v45_umk_all.json",
        "checkpoint_steps": 16,
        "parent": "v45_umk_local",
        "description": (
            "Local v45 candidate with corrected UMK direction "
            "lambda=0.3*(1-sigmoid(prior)); alpha0=0.3, beta0=1.0. "
            "Fixes the faster-forgetting-on-higher-prior defect; "
            "no performance claim."
        ),
        "design_document": "V45_UMK_R2.md",
    },
    "v45_umk_local": {
        "architecture": "v45_umk",
        "config": "assist2017_v45_umk_all.json",
        "checkpoint_steps": 16,
        "parent": "v43_lowmem",
        "description": (
            "Local v45 candidate with all three literal-formula UMK switches on. "
            "Adds only alpha/beta scalars; no performance claim."
        ),
        "design_document": "V45_UMK.md",
    },
    "v33": {
        "architecture": "v33",
        "config": "assist2017_v33.json",
        "checkpoint_steps": 0,
        "parent": "server_172_v33",
        "description": "Consolidated v33 reference with a single complete forward.",
    },
    "v43": {
        "architecture": "v43",
        "config": "assist2017_v43.json",
        "checkpoint_steps": 0,
        "parent": "v33",
        "description": "v33 plus causal input memory and multimode residual SSM.",
    },
    "v43_lowmem": {
        "architecture": "v43",
        "config": "assist2017_v43_lowmem.json",
        "checkpoint_steps": 16,
        "parent": "v43",
        "description": "v43 with deterministic concept-graph activation recomputation.",
    },
    "v43_lowmem_r2": {
        "architecture": "v43",
        "config": "assist2017_v43_lowmem.json",
        "checkpoint_steps": 16,
        "parent": "v43_lowmem",
        "description": (
            "Unchanged Full v43 model; adds separate input-memory/modal controls "
            "and post-selection diagnostics. Full ablation training remains gated."
        ),
    },
    "v43_lowmem_r3": {
        "architecture": "v43",
        "config": "assist2017_v43_lowmem.json",
        "checkpoint_steps": 16,
        "parent": "v43_lowmem_r2",
        "description": (
            "Unchanged Full v43 model; preserves separate controls and diagnostics "
            "and fixes JSON serialization of skipped test reports."
        ),
    },
    "v44_aligned_history_local": {
        "architecture": "v44_aligned_history",
        "config": "assist2017_v44_aligned_history.json",
        "checkpoint_steps": 16,
        "parent": "v43_lowmem_r3",
        "description": (
            "Local candidate: include the latest observed event in base statistics; "
            "same parameters and random initialization as v43. No AUC gain claimed."
        ),
        "design_document": "V44_ALIGNED_HISTORY.md",
    },
    "v48_causal_transfer_r2": {
        "architecture": "v48_causal_transfer",
        "config": "assist2017_v48_causal_transfer.json",
        "checkpoint_steps": 16,
        "parent": "v48_causal_transfer_local",
        "description": (
            "V48 r2 causal residual transport: strict-past cross-item concept "
            "residuals with evidence-mass shrinkage and target-item novelty. "
            "Removes redundant coverage/agreement attenuation; no performance claim."
        ),
        "design_document": "V48_CAUSAL_TRANSFER.md",
    },
}

TOOL_NAMES = (
    "audit_protocol.py", "audit_run.py", "run_logged.py", "run_tests.py",
    "reevaluate_checkpoint.py", "profile_local.py", "snapshot_run_sources.py",
    "verify_sources.py", "verify_production_initialization.py", "verify_version.py",
    "diagnose_validation.py",
    "audit_terminal_local.py",
    "profile_history_alignment.py",
)


def copy_file(source, destination):
    if source.is_symlink():
        raise ValueError(f"source links are not supported: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


def freeze(root, destination, version):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    spec = VERSIONS[version]
    source_manifest = json.loads(
        (root / "provenance/local_manifest.json").read_text(encoding="utf-8")
    )
    actual_source = {
        path.relative_to(root / "src/a2g").as_posix(): sha256(path)
        for path in sorted((root / "src/a2g").rglob("*.py"))
    }
    if destination.exists():
        raise FileExistsError(destination)
    if source_manifest["local_source"] != actual_source:
        raise ValueError("refresh verified source provenance before freezing")
    actual_configs = {
        path.name: sha256(path) for path in sorted((root / "configs").glob("*.json"))
    }
    if source_manifest["local_configuration"] != actual_configs:
        raise ValueError("configuration provenance is stale")
    if destination.is_relative_to(root / "src") or root.is_relative_to(destination):
        raise ValueError("destination must not contain or replace maintained source")
    destination.mkdir(parents=True, exist_ok=False)
    for directory, pattern in (
        ("src/a2g", "*.py"),
        ("configs", "*.json"),
        ("tests", "*.py"),
    ):
        for path in sorted((root / directory).rglob(pattern)):
            if path.name == "test_version_bundles.py":
                continue
            copy_file(path, destination / path.relative_to(root))
    for name in TOOL_NAMES:
        copy_file(root / "tools" / name, destination / "tools" / name)
    for name in ("MODEL_GUIDE.md", "EXPERIMENT_PROTOCOL.md", "V43_CONTROL_REVIEW.md"):
        copy_file(root / "docs" / name, destination / "docs" / name)
    if "design_document" in spec:
        name = spec["design_document"]
        copy_file(root / "docs" / name, destination / "docs" / name)
    for name in ("local_manifest.json", "source_map.json"):
        copy_file(root / "provenance" / name, destination / "provenance" / name)
    review = root / "provenance/review_20260918"
    for path in sorted(review.iterdir()):
        if path.is_file():
            copy_file(path, destination / path.relative_to(root))
    reference = root / "provenance/server_172"
    v33 = reference / "independent_a2g_ffn_gate_20260911_v33"
    for path in sorted(v33.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            copy_file(path, destination / path.relative_to(root))
    for name in (
        "v33_frozen.zip",
        "independent_a2g_modal_residual_20260913_v43/manifest.json",
        "independent_a2g_modal_residual_20260913_v43/input_memory_candidate.py",
        "independent_a2g_modal_residual_20260913_v43/modal_residual_candidate.py",
    ):
        copy_file(reference / name, destination / "provenance/server_172" / name)
    copy_file(root / "pyproject.toml", destination / "pyproject.toml")
    copy_file(root / "tools/version_entrypoint.py", destination / "run.py")
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl")
    }
    (destination / "requirements-runtime.txt").write_text(
        "# Observed runtime; install a platform-appropriate PyTorch CUDA wheel.\n"
        + "".join(f"{name}=={value}\n" for name, value in dependencies.items()),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "version_id": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "description": spec["description"],
        "architecture": spec["architecture"],
        "graph_checkpoint_steps": spec["checkpoint_steps"],
        "parent_for_comparison_only": spec["parent"],
        "default_config": f"configs/{spec['config']}",
        "self_contained_code": True,
        "shared_code_dependencies": [],
        "data_included": False,
        "data_contract": "Explicit --data-dir with frozen CSV and mapping hashes",
        "runtime_dependencies_observed": dependencies,
        "server_v33_manifest_bindings": source_manifest["v33_manifest_bindings"],
        "server_v33_archive_sha256": source_manifest["v33_archive_sha256"],
        "server_v43_verified_files": source_manifest["v43_verified_files"],
        "server_v43_full_package_verified": False,
        "model_parameters_and_training_algorithm_changed_by_export": False,
        "auc_claim": False,
        "paper_goal_complete": False,
        "files": {},
    }
    (destination / "README.md").write_text(
        f"# A2G {version}\n\n{spec['description']}\n\n"
        "This is a complete independent copy of the audited local implementation, "
        "not a wrapper around another version or the parent src directory. "
        "Optional inactive model branches are retained for regression tests. "
        "The default config determines this version's model and execution mode.\n\n"
        "## Run\n\nUse a compatible Python environment with the dependencies in "
        "`requirements-runtime.txt`. Do not install multiple editable A2G packages "
        "into one environment; `run.py` always selects this folder's code.\n\n"
        "```powershell\n"
        "python run.py verify\n"
        "python run.py tests --output runs/tests.json\n"
        'python run.py audit-data --data-dir "<frozen-data-directory>"\n'
        'python run.py smoke --data-dir "<frozen-data-directory>" --output runs/smoke\n'
        'python run.py logged train --data-dir "<frozen-data-directory>" '
        "--output runs/full --control runs/process\n"
        "```\n\n"
        "Data and the Python environment are external inputs, not shared model code. "
        "The launcher works from another working directory and after relocating "
        "this whole folder. Run outputs must use a new path.\n\n"
        "## Read\n\n"
        "- `src/a2g/model.py`: complete base forward.\n"
        "- `src/a2g/candidate.py`: local v43 construction, where enabled.\n"
        "- `src/a2g/aligned_history.py`: local history-alignment candidate, where enabled.\n"
        "- `src/a2g/modules/`: all module implementations.\n"
        "- `src/a2g/experiment.py`: complete train/evaluate/selection pipeline.\n"
        "- `docs/MODEL_GUIDE.md`: Chinese model reading guide.\n"
        "- `docs/V43_CONTROL_REVIEW.md`: independent intervention semantics and limits.\n"
        "- `tests/` and `provenance/server_172/`: bundled regression and source evidence.\n"
        "- `provenance/review_20260918/`: additional server review with a hash receipt.\n"
        "- `VERSION.json`: hashes of every delivered source/config/document.\n\n"
        "Treat this directory as a frozen snapshot. Create a new version for later "
        "edits; never refresh its manifest to disguise a modification. Runs and "
        "bytecode caches are excluded from integrity verification.\n\n"
        "No completed local AUC > 0.8174 or positive retrained ablation is claimed. "
        "v43 is not a verified full copy of the server experiment archive. "
        "Its two new modules and v33 dependencies were source-verified.\n",
        encoding="utf-8",
    )
    manifest["files"] = bundle_files(destination)
    (destination / "VERSION.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    default_folder = "candidates" if args.version == "v45_umk_local" else "versions"
    destination = args.output or root / default_folder / args.version
    print(json.dumps(freeze(root, destination, args.version), indent=2))


if __name__ == "__main__":
    main()
