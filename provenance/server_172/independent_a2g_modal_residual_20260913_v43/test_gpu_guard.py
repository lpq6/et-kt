import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gpu_guard import (
    GPU_UUID, SYSTEM_DISPLAY_EXECUTABLES, SYSTEM_DISPLAY_UID,
    SYSTEM_DISPLAY_USER, assess, parse_inventory,
)


class GPUOwnershipTests(unittest.TestCase):
    def snapshot(self, processes=(), uuid=GPU_UUID):
        return {
            "current_uid": 1007,
            "gpus": [{"index": 1, "uuid": uuid, "processes": list(processes)}],
        }

    def process(self, uid=1008, kind="C", pid=42):
        return {"pid": pid, "uid": uid, "type": kind, "name": "fixture"}

    def display(self, path="/usr/lib/xorg/Xorg", **changes):
        process = {
            "pid": 13877, "uid": SYSTEM_DISPLAY_UID,
            "user": SYSTEM_DISPLAY_USER, "type": "G", "name": path,
            "command": SYSTEM_DISPLAY_EXECUTABLES[path],
        }
        process.update(changes)
        return process

    def test_empty_is_available(self):
        self.assertTrue(assess(self.snapshot())["allowed"])

    def test_other_user_compute_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.process()]))["allowed"])

    def test_other_user_graphics_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.process(uid=121, kind="G")]))["allowed"])

    def test_same_user_other_compute_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.process(uid=1007)]))["allowed"])

    def test_only_actual_own_pid_can_be_exempted(self):
        self.assertTrue(assess(self.snapshot([self.process(uid=1007)]), own_pid=42)["allowed"])
        self.assertFalse(assess(self.snapshot([self.process()]), own_pid=42)["allowed"])

    def test_same_user_graphics_is_allowed(self):
        self.assertTrue(assess(self.snapshot([self.process(uid=1007, kind="G")]))["allowed"])

    def test_verified_gdm_xorg_is_allowed_and_recorded(self):
        result = assess(self.snapshot([self.display()]))
        self.assertTrue(result["allowed"])
        self.assertEqual(len(result["verified_system_display_processes"]), 1)

    def test_verified_gdm_shell_is_allowed(self):
        self.assertTrue(assess(self.snapshot([
            self.display("/usr/bin/gnome-shell"),
        ]))["allowed"])

    def test_gdm_compute_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.display(type="C")]))["allowed"])

    def test_gdm_combined_compute_graphics_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.display(type="C+G")]))["allowed"])

    def test_gdm_mps_is_blocked(self):
        for kind in ("M", "M+C"):
            with self.subTest(kind=kind):
                self.assertFalse(assess(self.snapshot([self.display(type=kind)]))["allowed"])

    def test_gdm_unknown_executable_is_blocked(self):
        self.assertFalse(assess(self.snapshot([
            self.display(name="/tmp/Xorg"),
        ]))["allowed"])

    def test_gdm_basename_only_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.display(name="Xorg")]))["allowed"])

    def test_gdm_command_mismatch_is_blocked(self):
        self.assertFalse(assess(self.snapshot([
            self.display(command="python"),
        ]))["allowed"])

    def test_missing_display_identity_is_blocked(self):
        for field in ("user", "command"):
            process = self.display()
            del process[field]
            with self.subTest(field=field):
                self.assertFalse(assess(self.snapshot([process]))["allowed"])

    def test_display_uid_mismatch_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.display(uid=1008)]))["allowed"])

    def test_display_username_mismatch_is_blocked(self):
        self.assertFalse(assess(self.snapshot([self.display(user="researcher")]))["allowed"])

    def test_root_display_is_not_exempted(self):
        self.assertFalse(assess(self.snapshot([
            self.display(uid=0, user="root"),
        ]))["allowed"])

    def test_display_on_both_gpus_is_allowed(self):
        snapshot = self.snapshot([self.display()])
        snapshot["gpus"].append({
            "index": 0, "uuid": "other", "processes": [self.display()],
        })
        self.assertTrue(assess(snapshot)["allowed"])

    def test_display_does_not_hide_other_job(self):
        snapshot = self.snapshot([self.display()])
        snapshot["gpus"].append({
            "index": 0, "uuid": "other", "processes": [self.process()],
        })
        self.assertFalse(assess(snapshot)["allowed"])

    def test_other_gpu_compute_also_blocks(self):
        snapshot = self.snapshot()
        snapshot["gpus"].append({"index": 0, "uuid": "other", "processes": [self.process()]})
        self.assertFalse(assess(snapshot)["allowed"])

    def test_identity_mismatch_is_blocked(self):
        self.assertFalse(assess(self.snapshot(uuid="unknown"))["allowed"])

    def test_xml_owner_is_resolved(self):
        xml = (
            "<nvidia_smi_log><gpu><minor_number>1</minor_number>"
            f"<uuid>{GPU_UUID}</uuid><processes><process_info>"
            "<pid>42</pid><type>G</type><process_name>Xorg</process_name>"
            "</process_info></processes></gpu></nvidia_smi_log>"
        )
        identity = {"pid": 42, "uid": 121, "user": "fixture", "command": "Xorg"}
        result = parse_inventory(xml, lambda _pid: identity)
        self.assertEqual(result[0]["processes"][0]["uid"], 121)
        self.assertEqual(result[0]["processes"][0]["user"], "fixture")
        with self.assertRaises((ValueError, TypeError, KeyError)):
            parse_inventory(xml, lambda _pid: {})
        with self.assertRaises(ValueError):
            parse_inventory(xml.replace("<type>G</type>", "<type>unknown</type>"), lambda _pid: identity)
        with self.assertRaises(ValueError):
            parse_inventory(xml, lambda _pid: {**identity, "pid": 43})
        with self.assertRaises(ValueError):
            parse_inventory(xml, lambda _pid: {**identity, "uid": -1})
        with self.assertRaises(ValueError):
            parse_inventory(xml, lambda _pid: {**identity, "user": ""})

    def test_incomplete_inventory_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_inventory("<nvidia_smi_log/>", lambda _pid: 1007)
        with self.assertRaises(ValueError):
            parse_inventory(
                f"<nvidia_smi_log><gpu><minor_number>1</minor_number><uuid>{GPU_UUID}</uuid>"
                "</gpu></nvidia_smi_log>",
                lambda _pid: 1007,
            )

    def test_failed_lookup_blocks_check(self):
        from gpu_guard import check
        with patch("gpu_guard.inspect_resources", side_effect=PermissionError("fixture")):
            self.assertFalse(check()["allowed"])

    def test_watchdog_stops_own_job_even_if_recording_fails(self):
        import run_independent
        stop = Mock()
        stop.wait.return_value = False
        with (
            patch("run_independent.check_gpu_resources", return_value={"allowed": False}),
            patch("run_independent.write_new", side_effect=OSError("fixture")),
            patch("run_independent.os._exit", side_effect=SystemExit(75)) as exit_own,
        ):
            with self.assertRaises(SystemExit) as raised:
                run_independent.resource_watchdog(stop, Path("unused"))
            self.assertEqual(raised.exception.code, 75)
            exit_own.assert_called_once_with(75)


if __name__ == "__main__":
    unittest.main()
