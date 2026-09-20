import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReevaluationEntrypointTests(unittest.TestCase):
    def test_frozen_source_dispatch_reaches_the_artifact_audit(self):
        version = ROOT if (ROOT / "VERSION.json").exists() else ROOT / "versions/v43_lowmem"
        with tempfile.TemporaryDirectory(prefix="a2g_reevaluation_") as temporary:
            temporary = Path(temporary)
            other_source = temporary / "unrelated_source"
            (other_source / "a2g").mkdir(parents=True)
            (other_source / "a2g/__init__.py").write_text(
                'raise RuntimeError("unrelated A2G must never be imported")\n',
                encoding="utf-8",
            )
            # Import the auditor from an unrelated location to exercise re-dispatch.
            launcher = (
                "import runpy,sys; "
                "sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)]; "
                "target=sys.argv.pop(1); sys.argv[0]=target; "
                "runpy.run_path(target,run_name='__main__')"
            )
            result = subprocess.run(
                [
                    sys.executable, "-I", "-c", launcher, str(other_source),
                    str(ROOT / "tools"),
                    str(ROOT / "tools/reevaluate_checkpoint.py"),
                    str(temporary / "missing_run"), "--version-root", str(version),
                    "--output", str(temporary / "never_written.json"), "--device", "cpu",
                ],
                cwd=temporary,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("result.json", result.stderr)
            self.assertIn("FileNotFoundError", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse((temporary / "never_written.json").exists())


if __name__ == "__main__":
    unittest.main()
