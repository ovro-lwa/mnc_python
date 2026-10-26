import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

# Beamformer imports are optional in this checkout. Install stand-ins before
# importing the tracker so the tip-tilt tests can run without station hardware.
for _mod in (
    "lwa_f",
    "lwa_f.snap2_feng_etcd_client",
    "lwa_f.snap2_fengine",
    "lwa_f.helpers",
    "lwa_antpos",
    "lwa_antpos.station",
    "casacore",
    "casacore.tables",
    "observing",
    "observing.obsstate",
    "lwa352_pipeline_control",
):
    sys.modules.setdefault(_mod, MagicMock())

import numpy
import astropy.units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time

import mnc.tiptilt as tiptilt
from mnc.tiptilt import TIPTILT_ETCD_KEY, get_tiptilt, get_tiptilt_path, set_tiptilt
from mnc.xengine_beamformer_control import BeamPointingControl, BeamTracker


class FakeEtcd(object):
    def __init__(self):
        self.store = {}
        self.closed = False

    def put(self, key, value):
        if isinstance(value, str):
            value = value.encode()
        self.store[key] = value

    def get(self, key):
        return self.store.get(key), None

    def close(self):
        self.closed = True


def _location():
    return EarthLocation(lat=37.23335 * u.deg, lon=-118.28167 * u.deg, height=1222 * u.m)


def _write_map(directory, az_deg, el_deg, dra, ddec, obstime_unix, name="tiptilt.npz"):
    path = os.path.join(directory, name)
    numpy.savez(
        path,
        az_deg=numpy.asarray(az_deg, dtype="<f8"),
        el_deg=numpy.asarray(el_deg, dtype="<f8"),
        dra_arcsec=numpy.asarray(dra, dtype="<f8"),
        ddec_arcsec=numpy.asarray(ddec, dtype="<f8"),
        obstime_unix=numpy.float64(obstime_unix),
    )
    return path


def _grid_maps():
    az = numpy.array([0.0, 90.0, 180.0, 270.0, 360.0])
    el = numpy.array([10.0, 45.0, 80.0])
    dra = numpy.arange(el.size * az.size, dtype="<f8").reshape(el.size, az.size)
    ddec = -2.0 * dra
    return az, el, dra, ddec


def _coord_at(location, obstime_unix, az_deg, el_deg):
    frame = AltAz(
        obstime=Time(obstime_unix, format="unix", scale="utc"),
        location=location,
    )
    return SkyCoord(az=az_deg * u.deg, alt=el_deg * u.deg, frame=frame)


class _Station(object):
    def __init__(self, location):
        x, y, z = location.to_geocentric()
        self.ecef = (x.to_value(u.m), y.to_value(u.m), z.to_value(u.m))


class RecordingControl(BeamPointingControl):
    def __init__(self, location):
        self.station = _Station(location)
        self.pointings = []

    def set_beam_pointing(self, az, alt, degrees=True, load_time=None):
        self.pointings.append((az, alt, degrees, load_time))


class _Clock(object):
    def __init__(self):
        self.t = 1700000000.0

    def time(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class TestTipTilt(unittest.TestCase):
    def setUp(self):
        tiptilt._cache.clear()
        tiptilt._last_shift.clear()
        tiptilt._last_fallback.clear()

    def test_set_rejects_relative_path_and_nonpositive_ttl(self):
        client = FakeEtcd()
        with self.assertRaises(ValueError):
            set_tiptilt("map.npz", 10, 40, client=client)
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt.npz", 0, 40, client=client)
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt.npz", -1, 40, client=client)
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt.npz", 10, 0, client=client)
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt.npz", 10, 40.0, client=client)
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt.npz", 10, True, client=client)
        self.assertEqual(client.store, {})
        self.assertFalse(client.closed)

    def test_set_stores_path_ttl_and_written_unix(self):
        client = FakeEtcd()
        before = time.time()
        set_tiptilt("/tmp/tiptilt.npz", 12.5, 40, client=client)
        after = time.time()

        raw, _meta = client.get(TIPTILT_ETCD_KEY)
        payload = json.loads(raw.decode())
        entry = payload["40"]
        self.assertEqual(entry["path"], "/tmp/tiptilt.npz")
        self.assertEqual(entry["ttl_sec"], 12.5)
        self.assertEqual(entry["frequency_mhz"], 40)
        self.assertGreaterEqual(entry["written_unix"], before)
        self.assertLessEqual(entry["written_unix"], after)
        self.assertFalse(client.closed)

    def test_set_reads_frequency_from_filename_when_omitted(self):
        client = FakeEtcd()
        set_tiptilt("/tmp/tiptilt_40MHz.npz", 12.5, client=client)
        payload = json.loads(client.get(TIPTILT_ETCD_KEY)[0].decode())
        self.assertEqual(payload["40"]["frequency_mhz"], 40)
        self.assertEqual(payload["40"]["path"], "/tmp/tiptilt_40MHz.npz")

        client = FakeEtcd()
        set_tiptilt("/data/obs/shift_73MHz_map.npz", 12.5, frequency_mhz=None, client=client)
        payload = json.loads(client.get(TIPTILT_ETCD_KEY)[0].decode())
        self.assertEqual(payload["73"]["frequency_mhz"], 73)

        client = FakeEtcd()
        set_tiptilt("/tmp/tiptilt_40MHz.npz", 12.5, 80, client=client)
        payload = json.loads(client.get(TIPTILT_ETCD_KEY)[0].decode())
        self.assertEqual(payload["80"]["frequency_mhz"], 80)

    def test_set_rejects_filename_without_one_frequency(self):
        client = FakeEtcd()
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt.npz", 10, client=client)
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt_40MHz_80MHz.npz", 10, client=client)
        with self.assertRaises(ValueError):
            set_tiptilt("/tmp/tiptilt_0MHz.npz", 10, client=client)
        self.assertEqual(client.store, {})

    def test_get_returns_cell_value(self):
        location = _location()
        obstime_unix = Time("2026-01-01T00:00:00", scale="utc").unix
        az, el, dra, ddec = _grid_maps()
        client = FakeEtcd()
        with tempfile.TemporaryDirectory() as directory:
            path = _write_map(directory, az, el, dra, ddec, obstime_unix)
            set_tiptilt(path, 60, 40, client=client)
            coord = _coord_at(location, obstime_unix, 90.0, 45.0)
            got_dra, got_ddec = get_tiptilt(coord, 40, location=location, client=client)
        self.assertAlmostEqual(got_dra, dra[1, 1], delta=1e-6)
        self.assertAlmostEqual(got_ddec, ddec[1, 1], delta=1e-6)

    def test_missing_key_returns_zero(self):
        coord = SkyCoord(0 * u.deg, 0 * u.deg, frame="fk5")
        dra, ddec = get_tiptilt(coord, 40, location=_location(), client=FakeEtcd())
        self.assertEqual((dra, ddec), (0.0, 0.0))

    def test_expired_map_returns_zero(self):
        client = FakeEtcd()
        client.put(TIPTILT_ETCD_KEY, json.dumps({
            "path": "/tmp/tiptilt.npz",
            "ttl_sec": 10.0,
            "written_unix": time.time() - 100.0,
            "frequency_mhz": 40,
        }))
        coord = SkyCoord(0 * u.deg, 0 * u.deg, frame="fk5")
        dra, ddec = get_tiptilt(coord, 40, location=_location(), client=client)
        self.assertEqual((dra, ddec), (0.0, 0.0))

    def test_outside_map_returns_zero(self):
        location = _location()
        obstime_unix = Time("2026-01-01T00:00:00", scale="utc").unix
        az, el, dra, ddec = _grid_maps()
        client = FakeEtcd()
        with tempfile.TemporaryDirectory() as directory:
            path = _write_map(directory, az, el, dra, ddec, obstime_unix)
            set_tiptilt(path, 60, 40, client=client)
            coord = _coord_at(location, obstime_unix, 90.0, 5.0)
            got = get_tiptilt(coord, 40, location=location, client=client)
        self.assertEqual(got, (0.0, 0.0))

    def test_unreadable_file_returns_zero(self):
        client = FakeEtcd()
        missing = os.path.join(tempfile.gettempdir(), "mnc-tiptilt-missing.npz")
        set_tiptilt(missing, 60, 40, client=client)
        coord = SkyCoord(0 * u.deg, 0 * u.deg, frame="fk5")
        dra, ddec = get_tiptilt(coord, 40, location=_location(), client=client)
        self.assertEqual((dra, ddec), (0.0, 0.0))
        self.assertEqual(get_tiptilt_path(40, client=client), missing)

    def test_get_tiptilt_path_is_none_without_a_publication(self):
        self.assertIsNone(get_tiptilt_path(40, client=FakeEtcd()))

    def test_failed_get_logs_reason_and_path(self):
        client = FakeEtcd()
        missing = os.path.join(tempfile.gettempdir(), "mnc-tiptilt-missing.npz")
        set_tiptilt(missing, 60, 40, client=client)
        coord = SkyCoord(0 * u.deg, 0 * u.deg, frame="fk5")
        log = MagicMock()
        with patch("mnc.tiptilt._logger", return_value=log):
            get_tiptilt(coord, 40, location=_location(), client=client)
            get_tiptilt(coord, 40, location=_location(), client=client)
        self.assertEqual(log.warning.call_count, 2)
        for call in log.warning.call_args_list:
            message = call[0][0]
            self.assertIn("unreadable file", message)
            self.assertIn(missing, message)
            self.assertIn("frequency_mhz=40", message)

    def test_shift_change_is_logged_once(self):
        location = _location()
        obstime_unix = Time("2026-01-01T00:00:00", scale="utc").unix
        az, el, dra, ddec = _grid_maps()
        dra2 = dra + 10.0
        ddec2 = ddec - 4.0
        coord = _coord_at(location, obstime_unix, 90.0, 45.0)
        client = FakeEtcd()
        log = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            first = _write_map(directory, az, el, dra, ddec, obstime_unix, name="a.npz")
            second = _write_map(directory, az, el, dra2, ddec2, obstime_unix, name="b.npz")
            set_tiptilt(first, 60, 40, client=client)
            with patch("mnc.tiptilt._logger", return_value=log):
                get_tiptilt(coord, 40, location=location, client=client)
                get_tiptilt(coord, 40, location=location, client=client)
                set_tiptilt(second, 60, 40, client=client)
                get_tiptilt(coord, 40, location=location, client=client)
                self.assertEqual(log.info.call_count, 2)
                client.put(TIPTILT_ETCD_KEY, json.dumps({
                    "40": {
                        "path": second,
                        "ttl_sec": 10.0,
                        "written_unix": time.time() - 100.0,
                        "frequency_mhz": 40,
                    },
                }))
                get_tiptilt(coord, 40, location=location, client=client)
        self.assertEqual(log.warning.call_count, 1)
        self.assertIn("expired", log.warning.call_args[0][0])

    def test_frequencies_are_kept_until_their_own_ttl_expires(self):
        location = _location()
        obstime_unix = Time("2026-01-01T00:00:00", scale="utc").unix
        az, el, dra, ddec = _grid_maps()
        dra_73 = dra + 3.0
        ddec_73 = ddec - 1.0
        coord = _coord_at(location, obstime_unix, 90.0, 45.0)
        client = FakeEtcd()
        with tempfile.TemporaryDirectory() as directory:
            path_40 = _write_map(directory, az, el, dra, ddec, obstime_unix, name="a.npz")
            path_73 = _write_map(directory, az, el, dra_73, ddec_73, obstime_unix, name="b.npz")
            set_tiptilt(path_40, 60, 40, client=client)
            set_tiptilt(path_73, 60, 73, client=client)
            payload = json.loads(client.get(TIPTILT_ETCD_KEY)[0].decode())
            self.assertEqual(set(payload), {"40", "73"})
            self.assertEqual(payload["40"]["path"], path_40)
            self.assertEqual(payload["73"]["path"], path_73)
            self.assertEqual(get_tiptilt_path(40, client=client), path_40)
            self.assertEqual(get_tiptilt_path(73, client=client), path_73)

            got_40 = get_tiptilt(coord, 40, location=location, client=client)
            got_73 = get_tiptilt(coord, 73, location=location, client=client)
            self.assertAlmostEqual(got_40[0], dra[1, 1], delta=1e-6)
            self.assertAlmostEqual(got_73[0], dra_73[1, 1], delta=1e-6)
            self.assertEqual(
                get_tiptilt(coord, 99, location=location, client=client),
                (0.0, 0.0),
            )

            payload["40"]["written_unix"] = time.time() - 100.0
            payload["40"]["ttl_sec"] = 10.0
            client.put(TIPTILT_ETCD_KEY, json.dumps(payload))
            self.assertEqual(
                get_tiptilt(coord, 40, location=location, client=client),
                (0.0, 0.0),
            )
            still_73 = get_tiptilt(coord, 73, location=location, client=client)
            self.assertAlmostEqual(still_73[0], dra_73[1, 1], delta=1e-6)

            set_tiptilt(path_40, 60, 40, client=client)
            payload = json.loads(client.get(TIPTILT_ETCD_KEY)[0].decode())
            self.assertEqual(set(payload), {"40", "73"})
            self.assertGreater(payload["40"]["written_unix"], time.time() - 5.0)

            payload["73"]["written_unix"] = time.time() - 100.0
            payload["73"]["ttl_sec"] = 10.0
            client.put(TIPTILT_ETCD_KEY, json.dumps(payload))
            set_tiptilt(path_40, 60, 80, client=client)
            payload = json.loads(client.get(TIPTILT_ETCD_KEY)[0].decode())
            self.assertEqual(set(payload), {"40", "80"})

    def _run_track(self, shift):
        location = _location()
        control = RecordingControl(EarthLocation.from_geocentric(*_Station(location).ecef, unit=u.m))
        tracker = BeamTracker(control, update_interval=30)
        clock = _Clock()

        def now():
            return Time(clock.t, format="unix", scale="utc")

        with patch("mnc.xengine_beamformer_control.time.time", clock.time), \
             patch("mnc.xengine_beamformer_control.time.sleep", clock.sleep), \
             patch("mnc.xengine_beamformer_control.Time.now", now), \
             patch("mnc.xengine_beamformer_control.get_tiptilt", return_value=shift):
            tracker.track("12:00:00", dec="+30:00:00", duration=0.05, frequency_mhz=40)
        self.assertEqual(len(control.pointings), 1)
        return control.pointings[0], control

    def _expected_altaz(self, load_time, dra_as, ddec_as, control):
        puto = min(0.05, 30.0) / 2.0
        obstime = Time(load_time - 2 + puto, format="unix", scale="utc")
        location = EarthLocation.from_geocentric(*control.station.ecef, unit=u.m)
        sc = SkyCoord(
            Angle("12:00:00", unit="hourangle"),
            Angle("+30:00:00", unit="deg"),
            frame="fk5",
        )
        sc_pt = sc.transform_to("fk5").spherical_offsets_by(dra_as * u.arcsec, ddec_as * u.arcsec)
        return sc_pt.transform_to(AltAz(obstime=obstime, location=location))

    def test_track_zero_shift_matches_uncorrected_pointing(self):
        (az, alt, degrees, load_time), control = self._run_track((0.0, 0.0))
        self.assertTrue(degrees)
        aa = self._expected_altaz(load_time, 0.0, 0.0, control)
        self.assertAlmostEqual(az, aa.az.deg, delta=1e-6)
        self.assertAlmostEqual(alt, aa.alt.deg, delta=1e-6)

    def test_track_nonzero_shift_moves_pointing(self):
        (az, alt, _degrees, load_time), control = self._run_track((3600.0, 0.0))
        corrected = self._expected_altaz(load_time, 3600.0, 0.0, control)
        uncorrected = self._expected_altaz(load_time, 0.0, 0.0, control)
        self.assertAlmostEqual(az, corrected.az.deg, delta=1e-6)
        self.assertAlmostEqual(alt, corrected.alt.deg, delta=1e-6)
        self.assertGreater(
            max(abs(az - uncorrected.az.deg), abs(alt - uncorrected.alt.deg)),
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
