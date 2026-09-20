import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from freeze_versions import freeze  # noqa: E402
from verify_version import bundle_files, verify  # noqa: E402


VERSIONS = (
    "v33", "v43", "v43_lowmem", "v43_lowmem_r2", "v43_lowmem_r3",
    "v44_aligned_history_local",
)
PROBE = """
import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
sys.argv = [str(root / "run.py"), "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
import torch
from a2g.experiment import load_config, make_model, seed_all
torch.set_num_threads(1)
manifest = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
config = load_config(root / manifest["default_config"])
seed_all(42)
model = make_model(config, "full")
digest = hashlib.sha256()
for name, tensor in model.state_dict().items():
    digest.update(name.encode())
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
sys.path.insert(0, str(root / "tests"))
from test_model import build, aligned
from v27_audit_helpers import timed_batch
small = build(type(model), graph_checkpoint_steps=manifest["graph_checkpoint_steps"])
prediction, trace = aligned(small, timed_batch("cpu"))
prediction.sum().backward()
modules = {
    name: str(Path(module.__file__).resolve())
    for name, module in sys.modules.items()
    if (name == "a2g" or name.startswith("a2g.")) and hasattr(module, "__file__")
}
assert all(Path(path).is_relative_to(root / "src/a2g") for path in modules.values())
assert all(torch.isfinite(p.grad).all() for p in small.parameters() if p.grad is not None)
print("ISOLATION_REPORT=" + json.dumps({
    "version": manifest["version_id"],
    "class": type(model).__name__,
    "parameters": sum(p.numel() for p in model.parameters()),
    "state_sha256": digest.hexdigest(),
    "prediction_sha256": hashlib.sha256(prediction.detach().numpy().tobytes()).hexdigest(),
    "graph_checkpoint_steps": model.concept_graph.checkpoint_steps,
    "module_files": modules,
}))
"""


class VersionBundleTests(unittest.TestCase):
    def temporary(self):
        work = ROOT / "work"
        work.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="version_test_", dir=work)
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name).resolve()
        self.assertTrue(path.is_relative_to(work.resolve()))
        return path

    def copy_bundle(self, destination, version):
        source = ROOT / "versions" / version
        verify(source)
        return Path(shutil.copytree(
            source,
            destination / version,
            ignore=shutil.ignore_patterns("runs", "__pycache__", "*.pyc"),
        ))

    def command(self, *args, cwd):
        return subprocess.run(
            [sys.executable, "-I", *map(str, args)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def test_all_versions_are_complete_and_hash_verified(self):
        required = {
            "run.py", "src/a2g/model.py", "src/a2g/candidate.py",
            "src/a2g/_initialization.py", "src/a2g/experiment.py",
            "src/a2g/data.py", "src/a2g/uid.py", "src/a2g/randomness.py",
            "tools/run_logged.py", "tools/audit_run.py",
            "tests/test_model.py", "requirements-runtime.txt",
        }
        for version in VERSIONS:
            root = ROOT / "versions" / version
            with self.subTest(version=version):
                report = verify(root)
                self.assertEqual(report["version_id"], version)
                self.assertTrue(required <= bundle_files(root).keys())

    def test_model_files_are_physical_copies_not_shared_links(self):
        paths = [ROOT / "src/a2g/model.py"] + [
            ROOT / "versions" / version / "src/a2g/model.py" for version in VERSIONS
        ]
        for index, first in enumerate(paths):
            for second in paths[index + 1:]:
                self.assertFalse(os.path.samefile(first, second))

    def test_relocated_versions_import_only_their_own_code_and_match(self):
        temporary = self.temporary()
        reports = {}
        for version in VERSIONS:
            root = self.copy_bundle(temporary, version)
            result = self.command("-c", PROBE, root, cwd=temporary)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(
                next(
                    line.removeprefix("ISOLATION_REPORT=")
                    for line in result.stdout.splitlines()
                    if line.startswith("ISOLATION_REPORT=")
                )
            )
            self.assertEqual(report["version"], version)
            self.assertEqual(
                report["graph_checkpoint_steps"], 0 if version in ("v33", "v43") else 16
            )
            expected_class = (
                "A2G" if version == "v33" else
                "A2GAlignedHistory" if version == "v44_aligned_history_local" else "A2GModal"
            )
            self.assertEqual(report["class"], expected_class)
            verify(root)
            reports[version] = report
        self.assertEqual(reports["v43"]["parameters"], 5114869)
        for version in ("v43_lowmem", "v43_lowmem_r2", "v43_lowmem_r3"):
            for name in ("parameters", "state_sha256", "prediction_sha256"):
                self.assertEqual(reports["v43"][name], reports[version][name])
        candidate = reports["v44_aligned_history_local"]
        for name in ("parameters", "state_sha256"):
            self.assertEqual(reports["v43"][name], candidate[name])
        self.assertNotEqual(reports["v43"]["prediction_sha256"], candidate["prediction_sha256"])

    def test_mutation_and_added_code_are_rejected_before_execution(self):
        temporary = self.temporary()
        root = self.copy_bundle(temporary, "v43")
        path = root / "src/a2g/model.py"
        original = path.read_bytes()
        path.write_bytes(original + b"\n# modified test fixture\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify(root)
        result = self.command(root / "run.py", "verify", cwd=temporary)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("version hash mismatch", result.stderr)
        path.write_bytes(original)
        (root / "src/a2g/extra.py").write_text("# test fixture\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra"):
            verify(root)

    def test_cross_version_config_is_rejected_before_training(self):
        temporary = self.temporary()
        for version, wrong in (
            ("v33", "v43"), ("v43", "v43_lowmem"),
            ("v44_aligned_history_local", "v43_lowmem"),
        ):
            root = self.copy_bundle(temporary, version)
            result = self.command(
                root / "run.py", "train", "--config",
                f"configs/assist2017_{wrong}.json",
                "--output", "runs/should_not_exist", cwd=temporary,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from this version", result.stderr)
            self.assertFalse((root / "runs/should_not_exist").exists())

    def test_existing_version_is_never_overwritten(self):
        root = ROOT / "versions/v33"
        before = bundle_files(root)
        with self.assertRaises(FileExistsError):
            freeze(ROOT, root, "v33")
        self.assertEqual(bundle_files(root), before)


if __name__ == "__main__":
    unittest.main()
