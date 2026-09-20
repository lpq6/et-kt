"""Preserve the exact source files already bound to an existing experiment."""

import argparse
import json
from pathlib import Path
import shutil

from a2g.provenance import sha256


def snapshot(directory, source):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    destination = directory / "source_snapshot"
    for name, expected in run["source"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"refusing to snapshot changed source: {name}")
    for name in run["source"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != run["source"][name]:
            raise FileExistsError(target)
        shutil.copyfile(source / name, target)
    print(f"Preserved {len(run['source'])} source files: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "src/a2g"
    for directory in args.directories:
        snapshot(directory, source)


if __name__ == "__main__":
    main()
