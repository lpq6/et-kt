"""Supervise one local experiment using persistent files, never console pipes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_record(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def supervise(command, control, cwd):
    control = Path(control).resolve()
    control.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = {
        "command": command,
        "cwd": str(Path(cwd).resolve()),
        "supervisor_pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "automatic_retries": 0,
        "stdout_destination": "stdout.log",
        "stderr_destination": "stderr.log",
        "status": "starting",
    }
    write_record(control / "process.json", record)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        with (
            (control / "stdout.log").open("xb") as output,
            (control / "stderr.log").open("xb") as errors,
        ):
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=output,
                stderr=errors,
                stdin=subprocess.DEVNULL,
                env=environment,
            )
            record.update(status="running", child_pid=process.pid)
            write_record(control / "process.json", record)
            return_code = process.wait()
        record.update(status="exited", return_code=return_code)
    except BaseException:
        record.update(status="supervisor_error", traceback=traceback.format_exc())
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        record["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_record(control / "process.json", record)
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "train"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/assist2017"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    entrypoint = root / "run.py"
    command = [sys.executable, "-u"]
    if entrypoint.is_file() and (root / "VERSION.json").is_file():
        command.append(str(entrypoint))
    else:
        command.extend(["-m", "a2g.experiment"])
    command.extend([
        args.mode,
        "--config",
        str((root / args.config).resolve()),
        "--output",
        str(output),
        "--data-dir",
        str((root / args.data_dir).resolve()),
        "--device",
        "cuda",
    ])
    raise SystemExit(supervise(command, root / args.control, root))


if __name__ == "__main__":
    main()
