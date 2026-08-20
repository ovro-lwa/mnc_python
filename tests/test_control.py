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

from mnc.control import Controller, FPG_FILE, _recording_path_from_response


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


def _all_snaps(value=True):
    return {i: value for i in range(1, 12)}


class TestStartFengineLoadProgram(unittest.TestCase):
    def setUp(self):
        self.controller = Controller.__new__(Controller)
        self.controller.conf = {
            "fengines": {
                "snap2s_inuse": [
                    "snap01", "snap02", "snap03", "snap04", "snap05",
                    "snap06", "snap07", "snap08", "snap09", "snap10", "snap11",
                ]
            }
        }
        self.controller.config_file = "/tmp/lwa_config.yaml"

    def _patch_etcd(self, send_command):
        etcd = MagicMock()
        etcd.send_command.side_effect = send_command
        return patch.multiple(
            "mnc.control",
            snap2_feng_etcd_client=MagicMock(
                Snap2FengineEtcdControl=MagicMock(return_value=etcd)
            ),
            settings=MagicMock(),
        ), etcd

    def test_program_passes_fpg_file_to_cold_start(self):
        calls = []

        def send_command(fid, block, cmd, **kwargs):
            calls.append((fid, block, cmd, kwargs.get("kwargs", {}), kwargs.get("timeout")))
            return _all_snaps()

        ctx, _etcd = self._patch_etcd(send_command)
        with ctx:
            with patch("mnc.control.time.sleep"):
                self.controller.start_fengine(program=True, loadprogram=False)

        program_calls = [c for c in calls if c[2] == "program"]
        self.assertEqual(len(program_calls), 1)
        self.assertEqual(program_calls[0][0], 0)
        self.assertEqual(program_calls[0][1], "feng")
        self.assertEqual(program_calls[0][3]["fpgfile"], FPG_FILE)
        self.assertTrue(program_calls[0][3]["force"])

        cold_starts = [c for c in calls if c[2] == "cold_start_from_config"]
        self.assertTrue(cold_starts)
        self.assertEqual(cold_starts[0][0], 1)
        self.assertTrue(cold_starts[0][3]["program"])
        self.assertNotIn("fpgfile", cold_starts[0][3])

        poll_restarts = [c for c in calls if c[2] == "start_poll_stats_loop"]
        self.assertEqual(len(poll_restarts), 1)

    def test_program_continues_when_flash_header_unread(self):
        calls = []

        def send_command(fid, block, cmd, **kwargs):
            calls.append(cmd)
            if cmd == "is_programmed" and "start_poll_stats_loop" not in calls:
                return {1: True, 2: True}  # fewer than 11 boards
            return _all_snaps()

        ctx, _etcd = self._patch_etcd(send_command)
        with ctx:
            with patch("mnc.control.time.sleep"):
                self.controller.start_fengine(program=True)

        self.assertIn("program", calls)
        self.assertIn("cold_start_from_config", calls)

    def test_loadprogram_path_is_forwarded(self):
        fpg_path = "/firmware/custom.fpg"
        calls = []

        def send_command(fid, block, cmd, **kwargs):
            calls.append((fid, block, cmd, kwargs.get("kwargs", {})))
            return _all_snaps()

        ctx, _etcd = self._patch_etcd(send_command)
        with ctx:
            with patch("mnc.control.time.sleep"):
                self.controller.start_fengine(loadprogram=True, fpg_file=fpg_path)

        program_calls = [c for c in calls if c[2] == "program"]
        self.assertEqual(program_calls[0][3]["fpgfile"], fpg_path)

    def test_initialize_without_program_does_not_upload_fpg(self):
        calls = []

        def send_command(fid, block, cmd, **kwargs):
            calls.append((fid, block, cmd, kwargs.get("kwargs", {}), kwargs.get("timeout")))
            return _all_snaps()

        ctx, _etcd = self._patch_etcd(send_command)
        with ctx:
            with patch("mnc.control.time.sleep"):
                self.controller.start_fengine(initialize=True, program=False)

        cmds = [c[2] for c in calls]
        self.assertNotIn("program", cmds)
        cold_starts = [c for c in calls if c[2] == "cold_start_from_config"]
        self.assertFalse(cold_starts[0][3]["program"])
        self.assertNotIn("fpgfile", cold_starts[0][3])


if __name__ == "__main__":
    unittest.main()
