"""Standalone version launcher; copied as run.py inside each frozen bundle."""

import importlib
import json
import os
from pathlib import Path
import runpy
import sys


TOOLS = {
    "verify": "verify_version.py",
    "tests": "run_tests.py",
    "audit-run": "audit_run.py",
    "reevaluate": "reevaluate_checkpoint.py",
    "profile": "profile_local.py",
    "logged": "run_logged.py",
    "diagnose": "diagnose_validation.py",
}


def main():
    root = Path(__file__).resolve().parent
    if not (root / "VERSION.json").is_file():
        raise RuntimeError("launch the run.py of a frozen version, not this template")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(1, str(root / "tools"))
    os.chdir(root)
    package = importlib.import_module("a2g")
    if Path(package.__file__).resolve().parent != root / "src/a2g":
        raise RuntimeError("another A2G installation shadowed this version")
    from verify_version import validate_version_config, verify

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("python run.py {audit-data,smoke,train,verify,tests,audit-run,reevaluate,profile,logged,diagnose}")
        print(f"Version source: {package.__file__}")
        return
    verify(root)
    manifest = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
    command = sys.argv[1]
    if command in ("audit-data", "smoke", "train", "profile", "logged"):
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path)
        known, _ = parser.parse_known_args(sys.argv[2:])
        config_path = known.config or root / manifest["default_config"]
        validate_version_config(
            manifest, json.loads(config_path.read_text(encoding="utf-8"))
        )
        if known.config is None:
            sys.argv.extend(["--config", str(config_path)])
    if command in TOOLS:
        path = root / "tools" / TOOLS[command]
        sys.argv = [str(path), *sys.argv[2:]]
        runpy.run_path(str(path), run_name="__main__")
    elif command in ("audit-data", "smoke", "train"):
        runpy.run_module("a2g.experiment", run_name="__main__")
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
