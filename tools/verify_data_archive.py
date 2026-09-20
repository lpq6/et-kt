"""Decompress downloaded Assist2017 data only when its frozen digest agrees."""

import argparse
import hashlib
import json
import lzma
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/assist2017/train_valid_sequences.csv.xz"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/assist2017_v33.json")
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    content = lzma.decompress(args.archive.read_bytes())
    actual = hashlib.sha256(content).hexdigest()
    if actual != config["data_binding"]["csv_sha256"]:
        raise ValueError("decompressed CSV does not match frozen server data")
    target = args.archive.with_suffix("")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != actual:
        raise FileExistsError(target)
    target.write_bytes(content)
    print(json.dumps({"path": str(target), "bytes": len(content), "sha256": actual}))


if __name__ == "__main__":
    main()
