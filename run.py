"""Development launcher. Frozen versions keep their own independent run.py."""

import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    import a2g

    if Path(a2g.__file__).resolve().parent != root / "src/a2g":
        raise RuntimeError("another A2G installation shadowed development source")
    runpy.run_module("a2g.experiment", run_name="__main__")


if __name__ == "__main__":
    main()
