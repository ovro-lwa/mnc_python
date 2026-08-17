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

from mnc.control import (
    Controller,
    _flag_antname_to_correlator,
    _recording_path_from_response,
)


class TestFlagAntnameToCorrelator(unittest.TestCase):
    @patch("mnc.control.mapping.antname_to_correlator")
    def test_accepts_full_polarized_name(self, convert):
        convert.return_value = 17

        self.assertEqual(_flag_antname_to_correlator("LWA-002A"), 17)
        convert.assert_called_once_with("LWA-002")

    @patch("mnc.control.mapping.antname_to_correlator")
    def test_accepts_numeric_polarized_name(self, convert):
        convert.return_value = 18

        self.assertEqual(_flag_antname_to_correlator("003B"), 18)
        convert.assert_called_once_with("LWA-003")


class TestControlBfAntennaHealth(unittest.TestCase):
    @patch("mnc.control._flag_antname_to_correlator", return_value=2)
    @patch("mnc.control.anthealth.get_badants")
    def test_caltable_falls_back_to_selfcorr(self, get_badants, convert):
        get_badants.side_effect = [
            RuntimeError("missing caltable state"),
            (61269.0, ["LWA-002A"]),
        ]
        controller = Controller.__new__(Controller)
        beam = MagicMock()
        beam.cal_set = True
        controller.bfc = {5: beam}

        controller.control_bf(
            num=5,
            targetname="sun",
            track=False,
            flag_ants="caltable",
        )

        self.assertEqual(
            get_badants.call_args_list,
            [unittest.mock.call("caltable"), unittest.mock.call("selfcorr")],
        )
        convert.assert_called_once_with("LWA-002A")
        beam.set_beam_target.assert_called_once_with("sun")


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
