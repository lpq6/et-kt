"""Small, explicit primitives for immutable run provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_hashes(root):
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*.py"))
    }
