import os
import sys
import tempfile
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
    "lwa_antpos",
    "lwa_antpos.mapping",
    "lwautils",
    "lwautils.lwa_arx",
    "dsautils",
    "dsautils.dsa_store",
    "matplotlib",
    "matplotlib.pyplot",
):
    sys.modules.setdefault(_mod, MagicMock())

from click.testing import CliRunner

from mnc.cli import _loadprogram_from_new_fpg, cli


class TestLoadprogramFromNewFpg(unittest.TestCase):
    def test_omitted_does_not_load(self):
        self.assertEqual(_loadprogram_from_new_fpg(None), (False, "default"))

    def test_default_loads_standard_image(self):
        self.assertEqual(_loadprogram_from_new_fpg("default"), (True, "default"))

    def test_existing_path_is_absolutized(self):
        with tempfile.NamedTemporaryFile(suffix=".fpg") as fh:
            loadprogram, fpg_file = _loadprogram_from_new_fpg(fh.name)
        self.assertTrue(loadprogram)
        self.assertTrue(os.path.isabs(fpg_file))
        self.assertTrue(fpg_file.endswith(".fpg"))

    def test_missing_path_raises(self):
        import click

        with self.assertRaises(click.BadParameter):
            _loadprogram_from_new_fpg("/no/such/firmware.fpg")


class TestStartFengineCli(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def _invoke(self, args):
        with patch("mnc.cli.control.Controller") as mock_controller:
            inst = mock_controller.return_value
            result = self.runner.invoke(cli, args, catch_exceptions=False)
        return result, inst

    def test_new_fpg_default_enables_loadprogram(self):
        result, inst = self._invoke(["start-fengine", "--new-fpg", "default"])
        self.assertEqual(result.exit_code, 0, result.output)
        inst.start_fengine.assert_called_once_with(
            program=False,
            initialize=False,
            loadprogram=True,
            fpg_file="default",
        )

    def test_new_fpg_underscore_alias(self):
        result, inst = self._invoke(["start-fengine", "--new_fpg", "default"])
        self.assertEqual(result.exit_code, 0, result.output)
        inst.start_fengine.assert_called_once()
        kwargs = inst.start_fengine.call_args[1]
        self.assertTrue(kwargs["loadprogram"])

    def test_new_fpg_path(self):
        with tempfile.NamedTemporaryFile(suffix=".fpg") as fh:
            result, inst = self._invoke(["start-fengine", "--program", "--new-fpg", fh.name])
            self.assertEqual(result.exit_code, 0, result.output)
            kwargs = inst.start_fengine.call_args[1]
            self.assertTrue(kwargs["program"])
            self.assertTrue(kwargs["loadprogram"])
            self.assertEqual(kwargs["fpg_file"], os.path.abspath(fh.name))

    def test_without_new_fpg_does_not_load(self):
        result, inst = self._invoke(["start-fengine", "--program"])
        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = inst.start_fengine.call_args[1]
        self.assertTrue(kwargs["program"])
        self.assertFalse(kwargs["loadprogram"])


if __name__ == "__main__":
    unittest.main()
