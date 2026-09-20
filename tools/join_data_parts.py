"""Join explicit nonoverlapping download ranges, then verify the frozen CSV."""

import hashlib
import json
import lzma
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/assist2017"
    prefix = data / "train_valid_sequences.csv.xz.partial"
    parts = [(prefix, 848896)]
    parts.extend(
        (data / f"assist2017.part{index}", size)
        for index, size in enumerate([482304, 482304, 482304, 483140])
    )
    for path, size in parts:
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"missing or incomplete range: {path}, expected {size}")
    content = b"".join(path.read_bytes() for path, _ in parts)
    csv = lzma.decompress(content)
    config = json.loads(
        (root / "configs/assist2017_v33.json").read_text(encoding="utf-8")
    )
    checksum = hashlib.sha256(csv).hexdigest()
    if checksum != config["data_binding"]["csv_sha256"]:
        raise ValueError("joined archive does not match frozen data")
    (data / "train_valid_sequences.csv.xz").write_bytes(content)
    (data / "train_valid_sequences.csv").write_bytes(csv)
    print(json.dumps({"bytes": len(csv), "csv_sha256": checksum, "status": "verified"}))


if __name__ == "__main__":
    main()
