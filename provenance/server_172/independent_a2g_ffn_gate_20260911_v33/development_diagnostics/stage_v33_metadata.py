"""Complete V33's unchanged data/reference metadata before any freeze."""

import hashlib
import json
from pathlib import Path
import shutil

WORK = Path(__file__).resolve().parent
PACKAGE = WORK / "independent_a2g_ffn_gate_20260911_v33"
PARENT = WORK / "independent_a2g_concept_graph_20260911_v32"
NAMES = (
    "trainfold_data_summary.json", "reference_snapshot.json", "incumbent_result.json",
    "references/baseline_table.md", "references/baseline_table.csv",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    report_path = PACKAGE / "supplemental_metadata.json"
    if (PACKAGE / "manifest.json").exists() or report_path.exists():
        raise RuntimeError("metadata already staged or package frozen")
    manifest = json.loads((PARENT / "manifest.json").read_text(encoding="utf-8"))
    if sha(PARENT / "manifest.json") != sha(PACKAGE / "reference_v32_manifest.json"):
        raise RuntimeError("V32 parent manifest changed")
    for name in NAMES:
        if (PACKAGE / name).exists() or sha(PARENT / name) != manifest["sha256"][name]:
            raise RuntimeError(f"unsafe or changed metadata: {name}")
    for name in NAMES:
        target = PACKAGE / name
        if not target.resolve().is_relative_to(PACKAGE):
            raise RuntimeError("metadata destination escaped V33")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PARENT / name, target)
        if sha(target) != manifest["sha256"][name]:
            raise RuntimeError("metadata copy differs")
    report = {
        "status": "unchanged_v32_metadata_staged_before_freeze",
        "reason": "The first dependency stage contained model/review sources but not runtime vocabulary-summary and baseline metadata.",
        "unchanged_sha256": {name: manifest["sha256"][name] for name in NAMES},
        "source_manifest_sha256": sha(PARENT / "manifest.json"),
        "script_sha256": sha(Path(__file__)),
        "raw_data_copied": False, "protocol_changed": False, "model_changed": False,
        "gpu_used": False, "training_started": False,
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
