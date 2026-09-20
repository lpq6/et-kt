"""Read-only GPU ownership checks; unknown ownership fails closed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

GPU_UUID = "GPU-e1065296-99f9-bf6e-7898-d6755ca37a9d"
SYSTEM_DISPLAY_UID = 128
SYSTEM_DISPLAY_USER = "gdm"
SYSTEM_DISPLAY_EXECUTABLES = {
    "/usr/lib/xorg/Xorg": "Xorg",
    "/usr/bin/gnome-shell": "gnome-shell",
}


def parse_inventory(xml_text, owner_lookup):
    root = ET.fromstring(xml_text)
    if root.tag != "nvidia_smi_log":
        raise ValueError("unexpected nvidia-smi XML root")
    gpus = []
    owners = {}
    for gpu in root.findall("gpu"):
        index = gpu.findtext("minor_number")
        uuid = gpu.findtext("uuid")
        process_list = gpu.find("processes")
        if index is None or uuid is None or process_list is None:
            raise ValueError("incomplete GPU identity or process inventory")
        if process_list.text and process_list.text.strip() not in {"", "None"}:
            raise ValueError("GPU process inventory is unsupported or unreadable")
        processes = []
        for process in process_list.findall("process_info"):
            pid = int(process.findtext("pid", ""))
            kind = process.findtext("type", "").strip()
            if pid <= 0 or kind not in {"C", "G", "C+G", "M", "M+C"}:
                raise ValueError("unknown GPU process identity or type")
            if pid not in owners:
                identity = owner_lookup(pid)
                if (
                    int(identity["pid"]) != pid
                    or int(identity["uid"]) < 0
                    or not identity["user"]
                    or not identity["command"]
                ):
                    raise ValueError("incomplete or inconsistent process identity")
                owners[pid] = {
                    "uid": int(identity["uid"]),
                    "user": identity["user"],
                    "command": identity["command"],
                }
            processes.append({
                "pid": pid,
                **owners[pid],
                "type": kind,
                "name": process.findtext("process_name", ""),
            })
        gpus.append({"index": int(index), "uuid": uuid, "processes": processes})
    if not gpus:
        raise ValueError("empty GPU inventory")
    return gpus


def inspect_resources(host=None):
    prefix = (
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host]
        if host
        else []
    )

    def run(args):
        return subprocess.check_output(prefix + args, text=True, timeout=20)

    uid = int(run(["id", "-u"]).strip()) if host else os.getuid()
    xml_text = run(["nvidia-smi", "-q", "-x"])

    def owner_lookup(pid):
        fields = run([
            "ps", "-o", "pid=,uid=,user:64=,comm=", "-p", str(pid),
        ]).strip().split(maxsplit=3)
        if len(fields) != 4:
            raise ValueError(f"cannot resolve GPU process {pid}")
        return dict(zip(("pid", "uid", "user", "command"), fields))

    inventory = parse_inventory(xml_text, owner_lookup)
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host or "local",
        "current_uid": uid,
        "gpus": inventory,
    }


def assess(snapshot, own_pid=None):
    reasons = []
    exempted = []
    target = [
        gpu for gpu in snapshot["gpus"]
        if gpu["index"] == 1 and gpu["uuid"] == GPU_UUID
    ]
    if len(target) != 1:
        reasons.append({"reason": "physical_gpu1_identity_unverified"})
    for gpu in snapshot["gpus"]:
        for process in gpu["processes"]:
            if process["pid"] == own_pid and process["uid"] == snapshot["current_uid"]:
                continue
            if (
                process["uid"] == SYSTEM_DISPLAY_UID
                and process.get("user") == SYSTEM_DISPLAY_USER
                and process["type"] == "G"
                and process["name"] in SYSTEM_DISPLAY_EXECUTABLES
                and process.get("command") == SYSTEM_DISPLAY_EXECUTABLES[process["name"]]
            ):
                exempted.append({"gpu": gpu["index"], **process})
                continue
            if process["uid"] != snapshot["current_uid"]:
                reasons.append({
                    "reason": "other_user_gpu_process",
                    "gpu": gpu["index"],
                    **process,
                })
            elif "C" in process["type"] or process["type"].startswith("M"):
                reasons.append({
                    "reason": "other_compute_process",
                    "gpu": gpu["index"],
                    **process,
                })
    return {
        **snapshot,
        "status": "available" if not reasons else "blocked",
        "allowed": not reasons,
        "reasons": reasons,
        "system_graphics_exception_authorized": True,
        "resource_policy_revision": "v15_resource_r1",
        "verified_system_display_processes": exempted,
    }


def check(host=None):
    try:
        return assess(inspect_resources(host), None if host else os.getpid())
    except Exception as error:
        return {
            "status": "blocked",
            "allowed": False,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "reasons": [{"reason": "resource_check_incomplete", "error": repr(error)}],
        }


def require_available():
    report = check()
    if not report["allowed"]:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.host)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["allowed"] else 75)
