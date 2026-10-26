"""Publish and read a tip-tilt shift map.

Another module writes a NumPy ``.npz`` file and calls :func:`set_tiptilt`.
:func:`get_tiptilt` reads that publication from etcd, samples the map at the
target, and returns a true-angle sky shift in arcseconds.

Etcd key
    ``TIPTILT_ETCD_KEY`` (``/mon/beamformer/tiptilt``). Rename this constant to
    move the key. The JSON value is::

        {"40": {"path": "<absolute path>", "ttl_sec": <float>,
                "written_unix": <float>, "frequency_mhz": 40},
         "73": {"path": "<absolute path>", "ttl_sec": <float>,
                "written_unix": <float>, "frequency_mhz": 73}}

    The object is keyed by the integer subband name in MHz. Each entry is
    valid while ``time.time() < written_unix + ttl_sec``. There is no etcd
    lease. Publishing one frequency replaces only that frequency. Every other
    frequency is kept while its own TTL still has time left, and is dropped
    on the next publish after it expires. ``frequency_mhz`` is also stored
    inside the entry. When the publisher omits it, the integer is taken from
    the one ``_<digits>MHz`` field in the filename, as in ``tiptilt_40MHz.npz``.
    :func:`get_tiptilt` uses ``frequency_mhz`` to select the entry. A request
    for a frequency that is absent or expired does not read a different entry.

Map file (named arrays)
    A ``.npz`` archive or a ``.npy`` dict (via ``numpy.save``) with:

    - ``az_deg`` — shape ``(n_az,)``, strictly increasing Astropy ``AltAz``
      azimuth in degrees (north through east). Azimuth does not wrap. A
      full-circle map must include every endpoint it wants to cover.
    - ``el_deg`` — shape ``(n_el,)``, strictly increasing altitude in degrees.
    - ``dra_arcsec`` — shape ``(n_el, n_az)``. FK5 true-angle longitude offset.
    - ``ddec_arcsec`` — shape ``(n_el, n_az)``. FK5 true-angle latitude offset.
    - ``obstime_unix`` — scalar UTC unix seconds. This is the ``AltAz`` time of
      the grid. Zenith at that time has right ascension equal to the local
      apparent sidereal time and declination equal to the station latitude.

Map file (OVRO-LWA flow ``.npy``)
    A single array of shape ``(n, n, 2)``. Filename
    ``flow_<freq>MHz_<YYYYMMDD>_<HHMMSS>.npy``. Conventions below are
    implemented as stated and **require verification** against the map
    producer:

    - Zenith-centered SIN (direction-cosine) grid; center pixel is zenith;
      the array edge is the horizon (``REQUIRES VERIFICATION``).
    - Component 0 is a west-positive offset in arcseconds; component 1 is
      a north-positive offset in arcseconds (``REQUIRES VERIFICATION``).
    - Those offsets are already FK5 true-angle sky shifts suitable for
      ``SkyCoord.spherical_offsets_by`` after converting west→east
      (``dra = -west``, ``ddec = north``) (``REQUIRES VERIFICATION``).
    - Pixel axes: column increases west, row increases south
      (``REQUIRES VERIFICATION``).
    - Map epoch is the UTC time in the filename; the station is OVRO-LWA.

Lookup and apply
    The sample coordinate is the target transformed to ``AltAz`` at the
    map epoch and the station location. Named-array maps interpolate on
    ``(el_deg, az_deg)``. Flow maps convert az/el to SIN pixels, then
    interpolate. The tracker applies the returned ``(dra, ddec)`` in FK5
    with ``SkyCoord.spherical_offsets_by`` and then transforms that
    corrected coordinate to the pointing time. Callers must not add the
    shift onto a coordinate that was already corrected.

Zero shift
    :func:`get_tiptilt` returns ``(0.0, 0.0)`` and does not raise when the key
    is missing, the value is unusable, the TTL has elapsed, the file cannot be
    read, the coordinate is outside the grid, or the interpolated sample is
    non-finite. The first result is logged, and later results are logged when
    the shift changes by more than ``0.001`` arcsec. Each fallback also logs a
    warning with the reason and, when the publication names a file, that path
    and ``frequency_mhz``.
"""

import json
import os
import re
import time
from threading import RLock
from typing import Optional, Tuple

import numpy
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from scipy.interpolate import RegularGridInterpolator


__all__ = ["TIPTILT_ETCD_KEY", "set_tiptilt", "get_tiptilt", "get_tiptilt_path"]


#: Provisional etcd key. This is the only place the path is spelled.
TIPTILT_ETCD_KEY = "/mon/beamformer/tiptilt"

_SHIFT_LOG_TOL_ARCSEC = 1e-3
_FREQUENCY_IN_NAME = re.compile(r"_(\d+)MHz")
_FLOW_NAME = re.compile(
    r"^flow_(\d+)MHz_(\d{8})_(\d{6})\.npy$", re.IGNORECASE)

_lock = RLock()
_cache = {}
_last_shift = {}
_last_fallback = {}


class _InvalidMap(Exception):
    """The map file opened but does not match a supported shift-map schema."""


class _UnreadableFile(Exception):
    """The map path could not be opened."""


class _Grid(object):
    """Named-array az/el map."""

    def __init__(self, az_deg, el_deg, interp_dra, interp_ddec, obstime_unix):
        self.az_deg = az_deg
        self.el_deg = el_deg
        self.interp_dra = interp_dra
        self.interp_ddec = interp_ddec
        self.obstime_unix = obstime_unix

    def sample(self, az_deg, el_deg):
        if (
            az_deg < self.az_deg[0] or az_deg > self.az_deg[-1]
            or el_deg < self.el_deg[0] or el_deg > self.el_deg[-1]
        ):
            return float("nan"), float("nan"), True
        dra = float(self.interp_dra((el_deg, az_deg)))
        ddec = float(self.interp_ddec((el_deg, az_deg)))
        return dra, ddec, False


class _FlowGrid(object):
    """OVRO-LWA SIN-projection flow map. Conventions require verification."""

    def __init__(self, west_arcsec, north_arcsec, obstime_unix):
        n = west_arcsec.shape[0]
        self.n = n
        self.center = 0.5 * (n - 1)
        rows = numpy.arange(n, dtype="<f8")
        cols = numpy.arange(n, dtype="<f8")
        self.interp_west = RegularGridInterpolator(
            (rows, cols), west_arcsec, bounds_error=False, fill_value=numpy.nan)
        self.interp_north = RegularGridInterpolator(
            (rows, cols), north_arcsec, bounds_error=False, fill_value=numpy.nan)
        self.obstime_unix = obstime_unix

    def sample(self, az_deg, el_deg):
        # REQUIRES VERIFICATION: zenith-centered SIN; horizon at array edge;
        # +x west, +y north; column increases west; row increases south.
        az = numpy.deg2rad(az_deg)
        el = numpy.deg2rad(el_deg)
        cos_el = numpy.cos(el)
        west = -cos_el * numpy.sin(az)
        north = cos_el * numpy.cos(az)
        radius = numpy.hypot(west, north)
        if not numpy.isfinite(radius) or radius > 1.0 + 1e-12 or el_deg < 0.0:
            return float("nan"), float("nan"), True
        col = self.center + west * self.center
        row = self.center - north * self.center
        if (
            row < 0.0 or row > self.n - 1
            or col < 0.0 or col > self.n - 1
        ):
            return float("nan"), float("nan"), True
        west_arcsec = float(self.interp_west((row, col)))
        north_arcsec = float(self.interp_north((row, col)))
        # REQUIRES VERIFICATION: stored vectors are FK5 offsets with +x west,
        # +y north; spherical_offsets_by wants east-positive dra.
        return -west_arcsec, north_arcsec, False


def _logger():
    # Imported lazily. mnc.common imports hardware clients at module scope, and
    # this module must still import on a machine that only has the science stack.
    from mnc.common import get_logger
    return get_logger(__name__)


def _connect():
    import etcd3
    from mnc.common import ETCD_HOST, ETCD_PORT
    return etcd3.client(host=ETCD_HOST, port=ETCD_PORT)


def _use_client(client):
    """Return ``(client, close_when_done)``."""
    if client is not None:
        return client, False
    return _connect(), True


def _frequency_from_filename(path):
    """Return the positive integer between ``_`` and ``MHz`` in the filename."""

    found = _FREQUENCY_IN_NAME.findall(os.path.basename(path))
    if len(found) != 1:
        raise ValueError(
            "frequency_mhz is omitted and the filename must contain one _<integer>MHz field")
    frequency_mhz = int(found[0])
    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive integer")
    return frequency_mhz


def set_tiptilt(path, ttl_sec, frequency_mhz=None, client=None):
    """Publish a shift-map file.

    ``path`` is the absolute path of a ``.npz`` or ``.npy`` map described in
    the module docstring. The file is not copied and does not need to exist yet.
    ``ttl_sec`` is how long the publication stays valid, in seconds. It must
    be greater than zero. ``frequency_mhz`` is the integer subband name in
    MHz. Pass a positive ``int``, or ``None`` to read the one ``_<integer>MHz``
    field from the filename, such as ``40`` in ``tiptilt_40MHz.npz``.

    ``client`` is an etcd-like object with ``put``, ``get``, and ``close``.
    The default connects to ``ETCD_HOST`` and ``ETCD_PORT`` and is closed
    before return. A caller-supplied client is left open. Connection and
    ``put`` errors propagate to the caller.

    The stored JSON object maps each frequency, as a string, to an entry with
    ``path``, ``ttl_sec``, ``written_unix``, and ``frequency_mhz``. Entries
    for other frequencies are copied forward when they are still inside their
    TTL.
    """

    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError("path must be an absolute path")
    try:
        ttl_sec = float(ttl_sec)
    except (TypeError, ValueError):
        raise ValueError("ttl_sec must be positive")
    if not ttl_sec > 0.0:
        raise ValueError("ttl_sec must be positive")
    if frequency_mhz is None:
        frequency_mhz = _frequency_from_filename(path)
    elif type(frequency_mhz) is not int or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive integer")

    record = {
        "path": path,
        "ttl_sec": ttl_sec,
        "written_unix": time.time(),
        "frequency_mhz": frequency_mhz,
    }
    etcd, own = _use_client(client)
    try:
        catalog = _catalog_for_update(etcd)
        catalog[frequency_mhz] = record
        stored = {str(freq): catalog[freq] for freq in sorted(catalog)}
        etcd.put(TIPTILT_ETCD_KEY, json.dumps(stored))
    finally:
        if own:
            etcd.close()


def get_tiptilt(coord: SkyCoord, frequency_mhz, location: Optional[EarthLocation] = None, client=None) -> Tuple[float, float]:
    """Return ``(dra_arcsec, ddec_arcsec)`` at ``coord`` for ``frequency_mhz``.

    ``coord`` is an Astropy ``SkyCoord``. ``frequency_mhz`` is the positive
    integer subband name in MHz and selects which published map is sampled.
    ``location`` is an ``EarthLocation``; the default is the OVRO station used
    by beam tracking. The sample azimuth and elevation are ``coord`` in
    ``AltAz`` at the map's ``obstime_unix``. Returned offsets are FK5
    true-angle longitude and latitude shifts for
    ``SkyCoord.spherical_offsets_by``.

    ``client`` matches :func:`set_tiptilt`. A missing, expired, or unusable
    map for this frequency returns ``(0.0, 0.0)`` and logs the reason. A
    valid map published at a different frequency is not used. etcd errors
    are logged and also return zero. ``frequency_mhz`` must be a positive
    ``int``; any other value raises ``ValueError``.
    """

    if type(frequency_mhz) is not int or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive integer")

    record, reason = _read_record(client, frequency_mhz)
    path, recorded_frequency = _publication_fields(record)
    logged_frequency = recorded_frequency if type(recorded_frequency) is int else frequency_mhz
    if reason is not None:
        return _report(0.0, 0.0, float("nan"), float("nan"), reason, path, logged_frequency)

    try:
        grid = _cached_grid(record["path"], record["written_unix"])
    except _UnreadableFile as exc:
        detail = str(exc) if str(exc) else "unreadable file"
        return _report(0.0, 0.0, float("nan"), float("nan"), detail, path, frequency_mhz)
    except _InvalidMap as exc:
        detail = str(exc) if str(exc) else "invalid map"
        return _report(0.0, 0.0, float("nan"), float("nan"), detail, path, frequency_mhz)

    if location is None:
        location = _default_location()
    obstime = Time(grid.obstime_unix, format="unix", scale="utc")
    aa = coord.transform_to(AltAz(obstime=obstime, location=location))
    az = float(aa.az.deg)
    el = float(aa.alt.deg)
    dra, ddec, outside = grid.sample(az, el)
    if outside:
        return _report(0.0, 0.0, az, el, "outside map", path, frequency_mhz)
    if not (numpy.isfinite(dra) and numpy.isfinite(ddec)):
        return _report(0.0, 0.0, az, el, "invalid sample", path, frequency_mhz)
    return _report(dra, ddec, az, el, None, path, frequency_mhz)


def get_tiptilt_path(frequency_mhz, client=None):
    """Return the absolute path published for ``frequency_mhz``.

    This is the filename :func:`get_tiptilt` opens for that subband when the
    publication is valid. The path is still returned when that frequency's
    publication has expired or the file cannot be read, so a zero shift can
    be traced to that file. The file is not opened. Returns ``None`` when
    the key is missing, etcd cannot be read, or this frequency has no
    absolute ``path``. ``frequency_mhz`` must be a positive ``int``.
    ``client`` matches :func:`set_tiptilt`.
    """

    if type(frequency_mhz) is not int or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive integer")
    record, _reason = _read_record(client, frequency_mhz)
    path, _recorded_frequency = _publication_fields(record)
    if isinstance(path, str) and os.path.isabs(path):
        return path
    return None


def _default_location():
    from lwa_antpos.station import ovro
    import astropy.units as u
    return EarthLocation.from_geocentric(*ovro.ecef, unit=u.m)


def _publication_fields(record):
    """Return ``(path, frequency_mhz)`` from a parsed record, or ``(None, None)``."""

    if not isinstance(record, dict):
        return None, None
    return record.get("path"), record.get("frequency_mhz")


def _catalog_for_update(etcd):
    """Return unexpired entries, then let the caller insert the new one.

    etcd errors propagate. A missing key or an unreadable value starts from
    an empty catalog. An unreadable value is logged and replaced.
    """

    got = etcd.get(TIPTILT_ETCD_KEY)
    value = got[0] if isinstance(got, tuple) else got
    if value is None:
        return {}
    catalog, reason = _decode_catalog(value)
    if reason is not None:
        _logger().warning("tip-tilt record is unreadable; replacing it")
        return {}
    kept = {}
    for freq, entry in catalog.items():
        record, entry_reason = _validate_entry(entry, freq)
        if entry_reason is None:
            kept[freq] = record
    return kept


def _decode_catalog(value):
    """Return ``(catalog, reason)`` from an etcd value.

    ``catalog`` maps ``frequency_mhz`` to the raw JSON entry. A legacy
    single-entry object is accepted as a one-frequency catalog. ``reason``
    is ``None`` when the value can be read as a catalog, including an empty
    one.
    """

    if isinstance(value, bytes):
        try:
            value = value.decode()
        except UnicodeError:
            return None, "invalid record"
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return None, "invalid record"
    if not isinstance(payload, dict):
        return None, "invalid record"
    if "path" in payload or "ttl_sec" in payload or "written_unix" in payload:
        freq = payload.get("frequency_mhz")
        if type(freq) is not int or freq <= 0:
            return None, "invalid record"
        return {freq: payload}, None
    catalog = {}
    for key, entry in payload.items():
        try:
            freq = int(key)
        except (TypeError, ValueError):
            return None, "invalid record"
        if type(freq) is not int or freq <= 0 or str(freq) != str(key):
            return None, "invalid record"
        if not isinstance(entry, dict):
            return None, "invalid record"
        catalog[freq] = entry
    return catalog, None


def _read_record(client, frequency_mhz):
    """Return ``(record, reason)`` for one frequency.

    ``reason`` is ``None`` when that frequency's publication can be used.
    On failure ``record`` is ``None``, or a partial dict carrying ``path``
    and ``frequency_mhz`` when those fields were present, so the fallback
    log can name the file. A frequency that is not in the catalog leaves
    every other frequency untouched.
    """

    try:
        etcd, own = _use_client(client)
    except Exception:
        return None, "etcd error"
    try:
        try:
            got = etcd.get(TIPTILT_ETCD_KEY)
        except Exception:
            return None, "etcd error"
    finally:
        if own:
            try:
                etcd.close()
            except Exception:
                pass

    value = got[0] if isinstance(got, tuple) else got
    if value is None:
        return None, "missing key"
    catalog, reason = _decode_catalog(value)
    if reason is not None:
        return None, reason
    entry = catalog.get(frequency_mhz)
    if entry is None:
        return None, "missing frequency"
    return _validate_entry(entry, frequency_mhz)


def _validate_entry(payload, frequency_mhz):
    """Return ``(record, reason)`` for one catalog entry."""

    path = payload.get("path")
    recorded_frequency = payload.get("frequency_mhz", frequency_mhz)
    partial = {}
    if isinstance(path, str):
        partial["path"] = path
    if type(recorded_frequency) is int:
        partial["frequency_mhz"] = recorded_frequency
    noted = partial or None
    try:
        ttl_sec = float(payload["ttl_sec"])
        written_unix = float(payload["written_unix"])
    except (KeyError, TypeError, ValueError):
        return noted, "invalid record"
    if not isinstance(path, str) or not os.path.isabs(path):
        return noted, "invalid record"
    if type(recorded_frequency) is not int or recorded_frequency != frequency_mhz:
        return noted, "invalid record"
    if not numpy.isfinite(ttl_sec) or not numpy.isfinite(written_unix):
        return noted, "invalid record"
    record = {
        "path": path,
        "ttl_sec": ttl_sec,
        "written_unix": written_unix,
        "frequency_mhz": recorded_frequency,
    }
    if time.time() >= written_unix + ttl_sec:
        return record, "expired"
    return record, None


def _cached_grid(path, written_unix):
    key = (path, float(written_unix))
    with _lock:
        grid = _cache.get(key)
        if grid is None:
            grid = _load_map(path)
            _cache.clear()
            _cache[key] = grid
        return grid


def _parse_flow_obstime_unix(path):
    """Return UTC unix seconds from ``flow_<freq>MHz_<YYYYMMDD>_<HHMMSS>.npy``."""

    matched = _FLOW_NAME.match(os.path.basename(path))
    if matched is None:
        raise _InvalidMap(
            "flow map filename must be flow_<freq>MHz_<YYYYMMDD>_<HHMMSS>.npy")
    stamp = matched.group(2) + matched.group(3)
    try:
        obstime = Time.strptime(stamp, "%Y%m%d%H%M%S", scale="utc")
    except ValueError as exc:
        raise _InvalidMap("flow map timestamp is not a valid UTC time") from exc
    return float(obstime.unix)


def _load_arrays(data):
    """Return the five map arrays from an NpzFile or a dict-like object."""

    try:
        az_deg = numpy.array(data["az_deg"], dtype="<f8", copy=True)
        el_deg = numpy.array(data["el_deg"], dtype="<f8", copy=True)
        dra = numpy.array(data["dra_arcsec"], dtype="<f8", copy=True)
        ddec = numpy.array(data["ddec_arcsec"], dtype="<f8", copy=True)
        obstime = numpy.array(data["obstime_unix"], dtype="<f8", copy=True)
    except KeyError as exc:
        raise _InvalidMap("missing array %s" % exc) from exc
    except (TypeError, ValueError) as exc:
        raise _InvalidMap("%s" % exc) from exc
    return az_deg, el_deg, dra, ddec, obstime


def _load_named_map(data):
    az_deg, el_deg, dra, ddec, obstime = _load_arrays(data)

    if az_deg.ndim != 1 or el_deg.ndim != 1:
        raise _InvalidMap("az_deg and el_deg must be 1-d")
    if az_deg.size < 2 or el_deg.size < 2:
        raise _InvalidMap("az_deg and el_deg must each have at least two points")
    if dra.shape != (el_deg.size, az_deg.size) or ddec.shape != dra.shape:
        raise _InvalidMap("dra_arcsec and ddec_arcsec must have shape (n_el, n_az)")
    if obstime.shape != ():
        raise _InvalidMap("obstime_unix must be a scalar")
    if not numpy.all(numpy.diff(az_deg) > 0.0) or not numpy.all(numpy.diff(el_deg) > 0.0):
        raise _InvalidMap("az_deg and el_deg must be strictly increasing")
    if not numpy.isfinite(obstime):
        raise _InvalidMap("obstime_unix must be finite")

    interp_dra = RegularGridInterpolator(
        (el_deg, az_deg), dra, bounds_error=False, fill_value=numpy.nan)
    interp_ddec = RegularGridInterpolator(
        (el_deg, az_deg), ddec, bounds_error=False, fill_value=numpy.nan)
    return _Grid(az_deg, el_deg, interp_dra, interp_ddec, float(obstime))


def _load_flow_map(path, data):
    """Build a :class:`_FlowGrid` from a ``(n, n, 2)`` array.

    Conventions (all **require verification** with the map producer):

    1. Zenith-centered SIN grid; center pixel is zenith; array edge is the
       horizon.
    2. ``data[..., 0]`` is west-positive FK5 offset in arcseconds;
       ``data[..., 1]`` is north-positive FK5 offset in arcseconds.
    3. Column index increases west; row index increases south.
    4. Returned sample converts to ``spherical_offsets_by`` as
       ``dra = -west``, ``ddec = north``.
    5. Map epoch is parsed from the filename
       ``flow_<freq>MHz_<YYYYMMDD>_<HHMMSS>.npy``.
    """

    if data.ndim != 3 or data.shape[0] != data.shape[1] or data.shape[2] != 2:
        raise _InvalidMap("flow map must have shape (n, n, 2)")
    if data.shape[0] < 2:
        raise _InvalidMap("flow map must be at least 2 pixels on a side")
    obstime_unix = _parse_flow_obstime_unix(path)
    west = numpy.array(data[:, :, 0], dtype="<f8", copy=True)
    north = numpy.array(data[:, :, 1], dtype="<f8", copy=True)
    return _FlowGrid(west, north, obstime_unix)


def _load_map(path):
    try:
        data = numpy.load(path, allow_pickle=True)
    except Exception as exc:
        raise _UnreadableFile("%s: %s" % (type(exc).__name__, exc)) from exc

    try:
        if hasattr(data, "files"):
            return _load_named_map(data)
        if isinstance(data, numpy.ndarray) and data.shape == () and isinstance(data.item(), dict):
            return _load_named_map(data.item())
        if isinstance(data, dict):
            return _load_named_map(data)
        if isinstance(data, numpy.ndarray) and data.ndim == 3 and data.shape[2] == 2:
            return _load_flow_map(path, data)
        raise _InvalidMap(
            "expected a .npz/.npy named-array map or a flow (n, n, 2) .npy")
    finally:
        close = getattr(data, "close", None)
        if close is not None:
            close()


def _failure_detail(path, frequency_mhz):
    parts = []
    if isinstance(path, str) and path:
        parts.append("path=%s" % path)
    if type(frequency_mhz) is int:
        parts.append("frequency_mhz=%d" % frequency_mhz)
    if not parts:
        return ""
    return " " + " ".join(parts)


def _report(dra, ddec, az, el, reason, path=None, frequency_mhz=None):
    dra = float(dra)
    ddec = float(ddec)
    with _lock:
        if reason is not None:
            _logger().warning(
                "tip-tilt unavailable (%s)%s; using zero shift"
                % (reason, _failure_detail(path, frequency_mhz)))
        _last_fallback[frequency_mhz] = reason
        shift = (dra, ddec)
        previous = _last_shift.get(frequency_mhz)
        changed = previous is None
        if previous is not None:
            changed = (
                abs(shift[0] - previous[0]) > _SHIFT_LOG_TOL_ARCSEC
                or abs(shift[1] - previous[1]) > _SHIFT_LOG_TOL_ARCSEC
            )
        if changed:
            _logger().info(
                "tip-tilt shift at az=%.3f deg, el=%.3f deg: dra=%.3f arcsec, ddec=%.3f arcsec"
                % (az, el, dra, ddec))
            _last_shift[frequency_mhz] = shift
        return shift
