"""Verify original sources plus the current local source manifest."""

import json
from pathlib import Path

from a2g.provenance import sha256, source_hashes, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    sources = root / "provenance/server_172"
    reference = sources / "independent_a2g_ffn_gate_20260911_v33"
    expected = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))[
        "sha256"
    ]
    for name, checksum in expected.items():
        if sha256(reference / name) != checksum:
            raise ValueError(f"V33 source mismatch: {name}")
    candidate = sources / "independent_a2g_modal_residual_20260913_v43"
    candidate_manifest = json.loads(
        (candidate / "manifest.json").read_text(encoding="utf-8")
    )["sha256"]
    candidate_files = ["input_memory_candidate.py", "modal_residual_candidate.py"]
    for name in candidate_files:
        if sha256(candidate / name) != candidate_manifest[name]:
            raise ValueError(f"V43 source mismatch: {name}")
    report = {
        "status": "verified",
        "v33_manifest_bindings": len(expected),
        "v33_archive_sha256": sha256(sources / "v33_frozen.zip"),
        "v43_verified_files": {
            name: candidate_manifest[name] for name in candidate_files
        },
        "v43_full_package_verified": False,
        "local_source": source_hashes(root / "src/a2g"),
        "local_configuration": {
            path.name: sha256(path)
            for path in sorted((root / "configs").glob("*.json"))
        },
    }
    write_json(root / "provenance/local_manifest.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "local_source"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
