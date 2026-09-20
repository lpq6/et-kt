"""Run local tests and persist a compact result suitable for handoff."""

import argparse
import io
from pathlib import Path
import platform
import sys
import time
import unittest

from a2g.provenance import source_hashes, write_json


def serializable_skips(skipped):
    return [
        {"test": str(test), "reason": reason}
        for test, reason in skipped
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/tests.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    stream = io.StringIO()
    start = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "successful": result.wasSuccessful(),
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": serializable_skips(result.skipped),
        "duration_seconds": time.monotonic() - start,
        "python": platform.python_version(),
        "source": source_hashes(root / "src/a2g"),
        "test_source": source_hashes(root / "tests"),
        "output": stream.getvalue(),
    }
    write_json(args.output, report)
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
