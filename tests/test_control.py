import sys
import unittest
from unittest.mock import MagicMock, patch

for _mod in (
    "lwa_f",
    "lwa_f.snap2_feng_etcd_client",
    "lwa_f.snap2_fengine",
    "lwa_f.helpers",
    "lwa352_pipeline_control",
    "observing",
    "observing.obsstate",
):
    sys.modules.setdefault(_mod, MagicMock())

from mnc.control import Controller, _recording_path_from_response


class TestRecordingPathFromResponse(unittest.TestCase):
    def test_absolute_filename(self):
        response = {"response": {"filename": "/lustre/ubuntu/beam01/D1_123.dat"}}
        self.assertEqual(
            _recording_path_from_response(response),
            "/lustre/ubuntu/beam01/D1_123.dat",
        )

    def test_relative_filename_volt(self):
        response = {
            "response": {
                "filename": "D1_123.dat",
                "directory": "/lustre/ubuntu/beam01",
            }
        }
        self.assertEqual(
            _recording_path_from_response(response),
            "/lustre/ubuntu/beam01/D1_123.dat",
        )

    def test_relative_filename_power(self):
        response = {
            "response": {
                "filename": "D1_123.dat",
                "directory": "/lustre/pipeline/beam03",
            }
        }
        self.assertEqual(
            _recording_path_from_response(response),
            "/lustre/pipeline/beam03/D1_123.dat",
        )

    def test_missing_filename(self):
        self.assertIsNone(_recording_path_from_response({"response": {}}))


class TestStartDrRecordings(unittest.TestCase):
    def test_start_dr_returns_beam_recording_paths(self):
        controller = Controller.__new__(Controller)
        controller.conf = {"dr": {"recorders": ["dr3"]}}
        controller.drvnums = []

        response = {
            "status": "success",
            "response": {
                "filename": "D1_123.dat",
                "directory": "/lustre/pipeline/beam03",
            },
        }
        controller.drc = MagicMock()
        controller.drc.send_command.return_value = (True, response)
        summary = MagicMock()
        summary.value = "normal"
        controller.drc.read_monitor_point.return_value = summary

        recordings = controller.start_dr(
            recorders=["dr3"],
            duration=60000,
            time_avg=100,
        )

        self.assertEqual(
            recordings,
            {
                "dr3": {
                    "status": "success",
                    "path": "/lustre/pipeline/beam03/D1_123.dat",
                    "filename": "D1_123.dat",
                    "directory": "/lustre/pipeline/beam03",
                }
            },
        )

    def test_start_dr_omits_drvs(self):
        controller = Controller.__new__(Controller)
        controller.conf = {"dr": {"recorders": ["drvs"]}}
        controller.drvnums = []

        response = {"status": "success", "response": {}}
        controller.drc = MagicMock()
        controller.drc.send_command.return_value = (True, response)
        summary = MagicMock()
        summary.value = "normal"
        controller.drc.read_monitor_point.return_value = summary

        with patch.object(controller, "stop_dr"):
            recordings = controller.start_dr(recorders=["drvs"])

        self.assertEqual(recordings, {})


if __name__ == "__main__":
    unittest.main()
