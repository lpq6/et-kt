import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_logged import supervise


class LoggedProcessTests(unittest.TestCase):
    def test_output_is_durable_and_does_not_require_a_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "process"
            script = (
                "import sys,time; print('started',flush=True); time.sleep(.1); "
                "sys.stdout.write('x'*1000000); sys.stderr.write('diagnostic\\n'); "
                "sys.stdout.flush(); raise SystemExit(7)"
            )
            code = supervise([sys.executable, "-c", script], control, temporary)
            self.assertEqual(code, 7)
            self.assertGreater((control / "stdout.log").stat().st_size, 1000000)
            self.assertIn("diagnostic", (control / "stderr.log").read_text())
            status = json.loads((control / "process.json").read_text())
            self.assertEqual(status["status"], "exited")
            self.assertEqual(status["return_code"], 7)
            self.assertEqual(status["automatic_retries"], 0)
            with self.assertRaises(FileExistsError):
                supervise([sys.executable, "-c", "pass"], control, temporary)

    def test_launch_failure_is_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "process"
            with self.assertRaises(OSError):
                supervise(
                    [str(Path(temporary) / "does_not_exist.exe")], control, temporary
                )
            status = json.loads((control / "process.json").read_text())
            self.assertEqual(status["status"], "supervisor_error")
            self.assertIn("Traceback", status["traceback"])


if __name__ == "__main__":
    unittest.main()
