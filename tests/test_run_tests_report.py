import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_tests  # noqa: E402


class RunTestsReportTests(unittest.TestCase):
    def test_skipped_tests_are_json_serializable(self):
        case = unittest.FunctionTestCase(lambda: None)
        serialized = run_tests.serializable_skips([(case, "CUDA unavailable")])
        self.assertEqual(serialized, [{
            "test": str(case),
            "reason": "CUDA unavailable",
        }])
        json.dumps(serialized)

    def test_cpu_report_is_written_after_skipped_tests(self):
        with tempfile.TemporaryDirectory(prefix="a2g_test_report_") as directory:
            output = Path(directory) / "tests.json"
            case = unittest.FunctionTestCase(lambda: None)

            class FakeResult:
                testsRun = 1
                failures = []
                errors = []
                skipped = [(case, "CUDA unavailable")]

                @staticmethod
                def wasSuccessful():
                    return True

            with (
                mock.patch.object(
                    run_tests.unittest.defaultTestLoader,
                    "discover",
                    return_value=object(),
                ),
                mock.patch.object(
                    run_tests.unittest,
                    "TextTestRunner",
                    return_value=mock.Mock(
                        run=mock.Mock(return_value=FakeResult())
                    ),
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    ["run_tests.py", "--output", str(output)],
                ),
                self.assertRaises(SystemExit) as exit_info,
            ):
                run_tests.main()
            self.assertEqual(exit_info.exception.code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["successful"])
            self.assertIsInstance(report["skipped"], list)
            self.assertEqual(report["skipped"], [{
                "test": str(case),
                "reason": "CUDA unavailable",
            }])


if __name__ == "__main__":
    unittest.main()
