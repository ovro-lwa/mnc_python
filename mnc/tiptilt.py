"""Publish and read a tip-tilt shift map.

Another module writes a NumPy ``.npz`` file and calls :func:`set_tiptilt`.
:func:`get_tiptilt` reads that publication from etcd, samples the map at the
target, and returns a true-angle sky shift in arcseconds.

Etcd key
    ``TIPTILT_ETCD_KEY`` (``/mon/beamformer/tiptilt``). Rename this constant to
    move the key. The JSON value is::

        {"path": "<absolute path>", "ttl_sec": <float>, "written_unix": <float>}

    ``written_unix`` is UTC unix seconds from :func:`time.time` at publish
    time. The map is valid while ``time.time() < written_unix + ttl_sec``.
    There is no etcd lease. An expired key is left in place and ignored.

Map file
    Float64 arrays:

    - ``az_deg`` — shape ``(n_az,)``, strictly increasing Astropy ``AltAz``
      azimuth in degrees (north through east). Azimuth does not wrap. A
      full-circle map must include every endpoint it wants to cover.
    - ``el_deg`` — shape ``(n_el,)``, strictly increasing altitude in degrees.
    - ``dra_arcsec`` — shape ``(n_el, n_az)``. FK5 true-angle longitude offset.
    - ``ddec_arcsec`` — shape ``(n_el, n_az)``. FK5 true-angle latitude offset.
    - ``obstime_unix`` — scalar UTC unix seconds. This is the ``AltAz`` time of
      the grid. Zenith at that time has right ascension equal to the local
      apparent sidereal time and declination equal to the station latitude.

Lookup and apply
    The sample coordinate is the target transformed to ``AltAz`` at
    ``obstime_unix`` and the station location. Offsets are bilinearly
    interpolated on ``(el_deg, az_deg)``. The tracker applies them in FK5 with
    ``SkyCoord.spherical_offsets_by`` and then transforms that corrected
    coordinate to the pointing time. Callers must not add the shift onto a
    coordinate that was already corrected.

Zero shift
    :func:`get_tiptilt` returns ``(0.0, 0.0)`` and does not raise when the key
    is missing, the value is unusable, the TTL has elapsed, the file cannot be
    read, the coordinate is outside the grid, or the interpolated sample is
    non-finite. The first result is logged, and later results are logged when
    the shift changes by more than ``0.001`` arcsec. A warning is logged when
    the fallback reason changes.
"""

import json
import os
import time
from threading import RLock
from typing import Optional, Tuple

import numpy
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from scipy.interpolate import RegularGridInterpolator


__all__ = ["TIPTILT_ETCD_KEY", "set_tiptilt", "get_tiptilt"]


#: Provisional etcd key. This is the only place the path is spelled.
TIPTILT_ETCD_KEY = "/mon/beamformer/tiptilt"

_SHIFT_LOG_TOL_ARCSEC = 1e-3

_lock = RLock()
_cache = {}
_last_shift = None
_last_fallback = None


class _InvalidMap(Exception):
    """The .npz file opened but does not match the shift-map schema."""


class _UnreadableFile(Exception):
    """The .npz path could not be opened."""


class _Grid(object):
    def __init__(self, az_deg, el_deg, interp_dra, interp_ddec, obstime_unix):
        self.az_deg = az_deg
        self.el_deg = el_deg
        self.interp_dra = interp_dra
        self.interp_ddec = interp_ddec
        self.obstime_unix = obstime_unix


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


def set_tiptilt(path, ttl_sec, client=None):
    """Publish a shift-map file.

    ``path`` is the absolute path of a ``.npz`` map described in the module
    docstring. The file is not copied and does not need to exist yet.
    ``ttl_sec`` is how long the publication stays valid, in seconds. It must
    be greater than zero.

    ``client`` is an etcd-like object with ``put`` and ``close``. The default
    connects to ``ETCD_HOST`` and ``ETCD_PORT`` and is closed before return.
    A caller-supplied client is left open. Connection and ``put`` errors
    propagate to the caller.

    The stored JSON object has ``path``, ``ttl_sec``, and ``written_unix``.
    """

    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError("path must be an absolute path")
    try:
        ttl_sec = float(ttl_sec)
    except (TypeError, ValueError):
        raise ValueError("ttl_sec must be positive")
    if not ttl_sec > 0.0:
        raise ValueError("ttl_sec must be positive")

    payload = {
        "path": path,
        "ttl_sec": ttl_sec,
        "written_unix": time.time(),
    }
    etcd, own = _use_client(client)
    try:
        etcd.put(TIPTILT_ETCD_KEY, json.dumps(payload))
    finally:
        if own:
            etcd.close()


def get_tiptilt(coord: SkyCoord, location: Optional[EarthLocation] = None, client=None) -> Tuple[float, float]:
    """Return ``(dra_arcsec, ddec_arcsec)`` at ``coord``.

    ``coord`` is an Astropy ``SkyCoord``. ``location`` is an
    ``EarthLocation``; the default is the OVRO station used by beam tracking.
    The sample azimuth and elevation are ``coord`` in ``AltAz`` at the map's
    ``obstime_unix``. Returned offsets are FK5 true-angle longitude and
    latitude shifts for ``SkyCoord.spherical_offsets_by``.

    ``client`` matches :func:`set_tiptilt`. A missing, expired, or unusable
    map returns ``(0.0, 0.0)``. etcd errors are logged and also return zero.
    """

    record, reason = _read_record(client)
    if reason is not None:
        return _report(0.0, 0.0, float("nan"), float("nan"), reason)

    try:
        grid = _cached_grid(record["path"], record["written_unix"])
    except _UnreadableFile:
        return _report(0.0, 0.0, float("nan"), float("nan"), "unreadable file")
    except _InvalidMap:
        return _report(0.0, 0.0, float("nan"), float("nan"), "invalid map")

    if location is None:
        location = _default_location()
    obstime = Time(grid.obstime_unix, format="unix", scale="utc")
    aa = coord.transform_to(AltAz(obstime=obstime, location=location))
    az = float(aa.az.deg)
    el = float(aa.alt.deg)
    if az < grid.az_deg[0] or az > grid.az_deg[-1] or el < grid.el_deg[0] or el > grid.el_deg[-1]:
        return _report(0.0, 0.0, az, el, "outside map")

    dra = float(grid.interp_dra((el, az)))
    ddec = float(grid.interp_ddec((el, az)))
    if not (numpy.isfinite(dra) and numpy.isfinite(ddec)):
        return _report(0.0, 0.0, az, el, "invalid sample")
    return _report(dra, ddec, az, el, None)


def _default_location():
    from lwa_antpos.station import ovro
    import astropy.units as u
    return EarthLocation.from_geocentric(*ovro.ecef, unit=u.m)


def _read_record(client):
    """Return ``(record, reason)``. ``record`` is None when ``reason`` is set."""

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

    path = payload.get("path")
    try:
        ttl_sec = float(payload["ttl_sec"])
        written_unix = float(payload["written_unix"])
    except (KeyError, TypeError, ValueError):
        return None, "invalid record"
    if not isinstance(path, str) or not os.path.isabs(path):
        return None, "invalid record"
    if not numpy.isfinite(ttl_sec) or not numpy.isfinite(written_unix):
        return None, "invalid record"
    if time.time() >= written_unix + ttl_sec:
        return None, "expired"
    return {"path": path, "ttl_sec": ttl_sec, "written_unix": written_unix}, None


def _cached_grid(path, written_unix):
    key = (path, float(written_unix))
    with _lock:
        grid = _cache.get(key)
        if grid is None:
            grid = _load_map(path)
            _cache.clear()
            _cache[key] = grid
        return grid


def _load_map(path):
    try:
        with numpy.load(path) as data:
            try:
                az_deg = numpy.array(data["az_deg"], dtype="<f8", copy=True)
                el_deg = numpy.array(data["el_deg"], dtype="<f8", copy=True)
                dra = numpy.array(data["dra_arcsec"], dtype="<f8", copy=True)
                ddec = numpy.array(data["ddec_arcsec"], dtype="<f8", copy=True)
                obstime = numpy.array(data["obstime_unix"], dtype="<f8", copy=True)
            except KeyError as exc:
                raise _InvalidMap from exc
    except _InvalidMap:
        raise
    except Exception as exc:
        raise _UnreadableFile from exc

    if az_deg.ndim != 1 or el_deg.ndim != 1:
        raise _InvalidMap
    if az_deg.size < 2 or el_deg.size < 2:
        raise _InvalidMap
    if dra.shape != (el_deg.size, az_deg.size) or ddec.shape != dra.shape:
        raise _InvalidMap
    if obstime.shape != ():
        raise _InvalidMap
    if not numpy.all(numpy.diff(az_deg) > 0.0) or not numpy.all(numpy.diff(el_deg) > 0.0):
        raise _InvalidMap
    if not numpy.isfinite(obstime):
        raise _InvalidMap

    interp_dra = RegularGridInterpolator(
        (el_deg, az_deg), dra, bounds_error=False, fill_value=numpy.nan)
    interp_ddec = RegularGridInterpolator(
        (el_deg, az_deg), ddec, bounds_error=False, fill_value=numpy.nan)
    return _Grid(az_deg, el_deg, interp_dra, interp_ddec, float(obstime))


def _report(dra, ddec, az, el, reason):
    global _last_shift, _last_fallback

    dra = float(dra)
    ddec = float(ddec)
    with _lock:
        if reason != _last_fallback:
            if reason is not None:
                _logger().warning(
                    "tip-tilt unavailable (%s); using zero shift" % reason)
            _last_fallback = reason
        shift = (dra, ddec)
        changed = _last_shift is None
        if _last_shift is not None:
            changed = (
                abs(shift[0] - _last_shift[0]) > _SHIFT_LOG_TOL_ARCSEC
                or abs(shift[1] - _last_shift[1]) > _SHIFT_LOG_TOL_ARCSEC
            )
        if changed:
            _logger().info(
                "tip-tilt shift at az=%.3f deg, el=%.3f deg: dra=%.3f arcsec, ddec=%.3f arcsec"
                % (az, el, dra, ddec))
            _last_shift = shift
        return shift
