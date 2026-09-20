"""Mechanically extract the reviewed server implementation using Python's AST.

Only explicitly selected definitions are retained. Forward computation lives in
the handwritten model.py; this script preserves legacy initialization order.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "provenance/server_172/independent_a2g_ffn_gate_20260911_v33"
TARGET = ROOT / "src/a2g"
RECORDS = []


def extract(relative, names, methods=None, rename=None, source_root=SOURCE):
    path = source_root / relative
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    selected = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign)):
            continue
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if name not in names:
            continue
        if methods is not None and isinstance(node, ast.ClassDef):
            node.body = [
                part
                for part in node.body
                if isinstance(part, ast.FunctionDef) and part.name in methods
            ]
        if rename and name in rename:
            node.name = rename[name]
        selected.append(node)
    if len(selected) != len(names):
        raise ValueError(f"Missing definitions in {relative}: {names}")
    RECORDS.append(
        {
            "source": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "definitions": names,
            "methods": methods,
        }
    )
    return "\n\n".join(ast.unparse(node) for node in selected)


def emit(name, header, blocks):
    path = TARGET / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")


def main():
    global TARGET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "work/extracted_reference"
    )
    args = parser.parse_args()
    TARGET = args.output_dir.resolve()
    if TARGET == (ROOT / "src/a2g").resolve():
        raise ValueError("refusing to overwrite the maintained model source")
    TARGET.mkdir(parents=True, exist_ok=False)
    expected = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))[
        "sha256"
    ]
    mismatches = [
        name
        for name, checksum in expected.items()
        if not (SOURCE / name).is_file()
        or hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != checksum
    ]
    if mismatches:
        raise ValueError(f"Frozen source mismatch: {mismatches}")
    work = "sources/work/"
    core = work + "a2g_mambakt_promoted_no_duplicate_folded_bias_20260715.py"
    physical = work + "a2g_mambakt_joint_weak_stat_prune_physical.py"
    novelty = work + "a2g_mambakt_one_epoch_concept_novelty_curriculum_candidate.py"
    evidence = work + "a2g_mambakt_evidence_equivalent_residual_candidate.py"
    recency = work + "a2g_mambakt_recency_weighted_concept_evidence_candidate.py"
    emit("modules/__init__.py", '"""Audited components of the A2G model."""', [])
    emit(
        "modules/core.py",
        '"""Server-equivalent SSM and folded normalization."""\n'
        "import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F",
        [
            extract(core, ["SelectiveSSMBlock", "FoldedZeroNormLinear"]),
            extract(physical, ["FoldedSelectedNormLinear"]),
            extract(novelty, ["OneEpochConceptNoveltyDropout"]),
        ],
    )
    specs = [
        (
            "attempt.py",
            "attempt_stage_candidate.py",
            ["item_attempt_indices", "ItemAttemptReadout"],
        ),
        (
            "pace.py",
            "history_pace_candidate.py",
            ["historical_pace_feature", "HistoricalPaceModulation"],
        ),
        (
            "newton.py",
            "prequential_newton_candidate.py",
            ["prefix_newton_scores", "PrequentialNewtonReadout"],
        ),
        (
            "graph.py",
            "concept_graph_candidate.py",
            ["graph_state_rollout", "DirectedConceptGraph"],
        ),
        ("ffn.py", "ffn_gate_candidate.py", ["InputGatedFeedForward"]),
    ]
    for destination, source, names in specs:
        emit(
            "modules/" + destination,
            f'"""Extracted from the reviewed {source}; see source_map.json."""\n'
            "import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F",
            [extract(source, names)],
        )
    emit(
        "_initialization.py",
        '"""Legacy seed stream and folded initialization; no model forward."""\n'
        "import math\nimport torch\nfrom torch import nn\n"
        "from .modules.core import (SelectiveSSMBlock, FoldedZeroNormLinear, "
        "FoldedSelectedNormLinear, OneEpochConceptNoveltyDropout)",
        [
            "LEARNED_STAT_INDICES = (0, 1, 2, 3, 4, 6)\nREMOVED_STAT_INDICES = (5, 7)",
            extract(
                core,
                ["A2GMambaKT"],
                [
                    "__init__",
                    "reset_parameters",
                    "set_prior_logits",
                    "_sequences",
                    "_stats",
                    "_boundary_weights",
                ],
                {"A2GMambaKT": "LegacyInitialization"},
            ),
        ],
    )
    # AST is also used for method extraction, avoiding textual slicing of classes.
    method_nodes = []
    for relative, methods in [
        (
            novelty,
            ["_concept_novelty_mask", "complete_concept_novelty_warmup", "train"],
        ),
        (
            evidence,
            ["identity_support", "set_prior_support", "_item_residual_reliability"],
        ),
        (
            recency,
            [
                "_recency_weighted_concept_evidence_components",
                "_recency_weighted_concept_evidence",
                "_replace_learned_exposure",
            ],
        ),
        (
            "sources/a2g_mambakt_final.py",
            ["_scheduled_dropout", "_apply_item_residual_dropout"],
        ),
    ]:
        temporary = ast.parse(extract(relative, ["A2GMambaKT"], methods))
        method_nodes.extend(temporary.body[0].body)
    definition = ast.ClassDef(
        name="EvidenceMethods",
        bases=[],
        keywords=[],
        body=method_nodes,
        decorator_list=[],
    )
    emit(
        "modules/evidence.py",
        '"""Causal evidence utilities; legacy lag semantics are preserved."""\n'
        "import torch\nfrom torch import nn\nfrom .core import OneEpochConceptNoveltyDropout\n"
        "EXPOSURE_LOCAL_INDEX = 2",
        [ast.unparse(ast.fix_missing_locations(definition))],
    )
    emit(
        "data.py",
        '"""Server train/validation dataset and deterministic batching."""\n'
        "import csv\nimport hashlib\nimport json\nimport math\nfrom pathlib import Path\n"
        "import numpy as np\nimport torch\nfrom torch.utils.data import Dataset, Sampler\n"
        "from torch.utils.data._utils.collate import default_collate",
        [
            extract(
                "strict_sequence_data.py",
                [
                    "FIELDS",
                    "TrainValidationDataset",
                    "collate_ordered_segments",
                    "FixedOrderBatchSampler",
                ],
            ),
        ],
    )
    # These self-contained utilities are retained byte-for-byte.
    for src, dest in [
        ("sources/a2g_uid_topological_loader_20260807.py", "uid.py"),
        ("sources/a2g_dropout_substream_alignment_20260807.py", "randomness.py"),
    ]:
        text = (SOURCE / src).read_bytes()
        (TARGET / dest).write_bytes(text)
        RECORDS.append(
            {"source": src, "sha256": hashlib.sha256(text).hexdigest(), "copy": dest}
        )
    v43 = SOURCE.parent / "independent_a2g_modal_residual_20260913_v43"
    v43_manifest = json.loads((v43 / "manifest.json").read_text(encoding="utf-8"))[
        "sha256"
    ]
    for relative in ("input_memory_candidate.py", "modal_residual_candidate.py"):
        actual = hashlib.sha256((v43 / relative).read_bytes()).hexdigest()
        if actual != v43_manifest[relative]:
            raise ValueError(f"V43 source mismatch: {relative}")
    emit(
        "modules/input_memory.py",
        '"""V39 candidate: finite causal SSM input memory (not an AUC claim)."""\n'
        "import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n"
        "from .core import SelectiveSSMBlock",
        [
            extract(
                "input_memory_candidate.py",
                ["CausalInputMemory", "InputMemorySSM"],
                source_root=v43,
            ),
        ],
    )
    emit(
        "modules/modal.py",
        '"""V43 candidate: multimode SSM residual (not an AUC claim)."""\n'
        "import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n"
        "from .input_memory import InputMemorySSM",
        [
            extract(
                "modal_residual_candidate.py",
                ["MultiModeResidualInputMemorySSM"],
                source_root=v43,
            ),
        ],
    )
    report = {
        "frozen_manifest_verified_files": len(expected),
        "package": SOURCE.name,
        "transform": "AST-selected definitions, no experiment execution",
        "records": RECORDS,
    }
    (TARGET / "source_map.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(expected)} source bindings; emitted reviewed modules.")


if __name__ == "__main__":
    main()
