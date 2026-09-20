"""Fetch a read-only, hash-verified source/data snapshot from the user's host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--files", nargs="+", default=["."])
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / (destination.name + ".download.tar.gz")
    command = (
        "tar -czf - --exclude=__pycache__ --exclude='*.pyc' -C "
        + shlex.quote(args.remote)
        + " -- "
        + " ".join(shlex.quote(name) for name in args.files)
    )
    ssh = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        "lpq@172.25.114.0",
        command,
    ]
    with archive.open("wb") as output:
        result = subprocess.run(
            ssh,
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    records = []
    with tarfile.open(archive, "r:gz") as bundle:
        for entry in bundle:
            relative = PurePosixPath(entry.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive entry: {entry.name}")
            if entry.isdir():
                continue
            if not entry.isfile():
                raise ValueError(f"Non-regular archive entry: {entry.name}")
            target = destination.joinpath(*relative.parts).resolve()
            target.relative_to(destination)
            content = bundle.extractfile(entry).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != content:
                raise FileExistsError(
                    f"Refusing to overwrite a different file: {target}"
                )
            target.write_bytes(content)
            records.append(
                {
                    "path": relative.as_posix(),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    report = {
        "host": "lpq@172.25.114.0",
        "remote": args.remote,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": records,
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "sha256", {}
        )
        mismatches = []
        for name, checksum in expected.items():
            file = destination / name
            if (
                not file.is_file()
                or hashlib.sha256(file.read_bytes()).hexdigest() != checksum
            ):
                mismatches.append(name)
        report["frozen_manifest_files"] = len(expected)
        report["frozen_manifest_mismatches"] = mismatches
    out = destination.parent / (destination.name + ".transfer.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    print(f"Transferred {len(records)} files to {destination}")
    if report.get("frozen_manifest_mismatches"):
        raise SystemExit(
            "Frozen manifest mismatch; inspect transfer report before use."
        )


if __name__ == "__main__":
    main()
