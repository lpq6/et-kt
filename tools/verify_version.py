"""Verify a standalone version's complete, content-addressed source bundle."""

import argparse
import hashlib
import json
from pathlib import Path


IGNORED_DIRECTORIES = {
    "__pycache__", ".venv", ".ruff_cache", ".pytest_cache", ".git",
    "runs", "data", "work",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_files(root):
    root = Path(root).resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in IGNORED_DIRECTORIES or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"version cannot use external links: {relative}")
        if not path.is_file() or relative.as_posix() == "VERSION.json":
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        files[relative.as_posix()] = sha256(path)
    return files


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = bundle_files(root)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name for name in set(actual) & set(expected) if actual[name] != expected[name]
        )
        raise ValueError(
            f"version hash mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    default = root / manifest["default_config"]
    if not default.resolve().is_relative_to(root):
        raise ValueError("default config must stay in this version")
    config = json.loads(default.read_text(encoding="utf-8"))
    validate_version_config(manifest, config)
    return {
        "status": "independent_source_bundle_verified",
        "version_id": manifest["version_id"],
        "root": str(root),
        "files_verified": len(actual),
        "default_config": manifest["default_config"],
        "architecture": manifest["architecture"],
        "graph_checkpoint_steps": manifest["graph_checkpoint_steps"],
        "manifest_sha256": sha256(root / "VERSION.json"),
        "performance_claim": False,
    }


def validate_version_config(manifest, config):
    if (
        config.get("architecture", "v33") != manifest["architecture"]
        or config["model"].get("graph_checkpoint_steps", 0)
        != manifest["graph_checkpoint_steps"]
    ):
        raise ValueError("config architecture/execution mode differs from this version")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
