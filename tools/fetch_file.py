"""Read a single remote file using bounded SFTP requests and no prefetch."""

import argparse
from pathlib import Path
import time

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("remote")
    parser.add_argument("local", type=Path)
    parser.add_argument("--pipeline", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(
        "172.25.114.0",
        username="lpq",
        key_filename=str(Path.home() / ".ssh" / "id_ed25519"),
        timeout=15,
        banner_timeout=20,
        auth_timeout=20,
        allow_agent=False,
        look_for_keys=False,
        compress=True,
    )
    print("Authenticated", flush=True)
    try:
        sftp = paramiko.SFTPClient.from_transport(
            client.get_transport(),
            window_size=32768,
            max_packet_size=1024,
        )
        sftp.get_channel().settimeout(30)
        remote_size = sftp.stat(args.remote).st_size
        size = remote_size - args.start if args.length is None else args.length
        if args.start < 0 or size < 1 or args.start + size > remote_size:
            raise ValueError("invalid remote range")
        args.local.parent.mkdir(parents=True, exist_ok=True)
        partial = args.local.with_suffix(args.local.suffix + ".partial")
        start = time.monotonic()
        position = partial.stat().st_size if partial.exists() else 0
        with (
            sftp.open(args.remote, "rb", bufsize=0) as source,
            partial.open("ab") as dest,
        ):
            while position < size:
                requests = [
                    (args.start + offset, min(1024, size - offset))
                    for offset in range(
                        position, min(size, position + 1024 * args.pipeline), 1024
                    )
                ]
                if args.pipeline > 1:
                    chunks = source.readv(requests)
                else:
                    source.seek(args.start + position)
                    chunks = [source.read(requests[0][1])]
                for chunk, (_, expected_size) in zip(chunks, requests):
                    if len(chunk) != expected_size:
                        raise EOFError(f"Short read: {position}/{size}")
                    dest.write(chunk)
                    position += len(chunk)
                if position % 65536 == 0:
                    print(
                        f"{position}/{size} bytes, {time.monotonic() - start:.1f}s",
                        flush=True,
                    )
        if args.local.exists() and args.local.stat().st_size:
            raise FileExistsError(args.local)
        partial.replace(args.local)
        print(f"Downloaded {size} bytes", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
