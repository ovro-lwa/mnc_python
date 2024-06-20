from dsautils import dsa_store
from lwa_antpos import mapping
from mnc import common
from astropy.time import Time
import numpy as np
from casacore import tables

logger = common.get_logger(__name__)
ls = dsa_store.DsaStore()

METHODS = ['selfcorr', 'caltable', 'union_and', 'union_or']


def set_badants(method, badants, time=None, naming='ant'):
    """ Set the antenna status for a given badant method.
    badants should be a list of antenna names as strings "001A" or "100". If no A/B pol, given then both pols assumed bad.
    time refers to the date of the badant measurement (mjd or isot).
    naming defines antenna sequence and can be "antnum" (e.g., number in "LWA-001").
    """

    assert naming == 'ant', 'setting by ant naming only supported currently'
    assert isinstance(badants, list), "badants must be a list"
    assert isinstance(badants[0], str), "badant entries must be str"
    assert 'x' not in badants[0].lower() and 'y' not in badants[0].lower(), "define polarization with 'A' or 'B'"

    if method not in METHODS:
        logger.warning(f"Method {method} not fully supported. Select from: {METHODS}.")
    else:
        if 'union' in method.lower():
            logger.error("Cannot set antenna status with 'union'.")
            raise RuntimeError

    # clean input badant list
    badants2 = []
    for badant in badants:
        assert isinstance(badant, str)

        if badant[-1] not in ['A', 'B']:
            needspol = True
        else:
            needspol = False

        if needspol:
            aa = int(badant)
            badants2.append(f'{aa:03}A')
            badants2.append(f'{aa:03}B')
        elif not needspol:
            aa = int(badant[:-1])
            pp = badant[-1]
            badants2.append(f'{aa:03}{pp}')
    badants = badants2

    antnames, antstatus = zip(*[(a.lstrip('LWA-')+pol, a.lstrip('LWA-')+pol in badants) for a in mapping.filter_df('used', True).index for pol in ['A', 'B']])  # make list of status for all ants in antnum order
    
    if time is None:
        mjd = Time.now().mjd
    else:
        if 'T' in time:
            mjd = Time(time, format='isot').mjd
        else:
            # assume MJD if no "T" indicating ISOT format time
            mjd = time

    assert Time(mjd, format='mjd'), f"Time ({time}) must be parsable into MJD."
    dd = {'time': mjd, 'flagged': antstatus, 'antname': antnames, 'naming': 'ant'}  # this could be expanded beyond booleans
    ls.put_dict(f'/mon/anthealth/{method}', dd)  # maybe influx can ingest from here
    ls.put_dict(f'/mon/anthealth/{method}/{mjd}', dd)


def get_badants(method, time=None, naming='ant'):
    """ Given a badant method, return list of bad antennas
    naming defines antenna sequence and can be "ant" (e.g., number in "LWA-001") or "corr" (i.e., MS/CASA number).
    mjd is the approximate day of the badant list. Defaults to the latest..
    Naming in etcd is in ant, but values can be set/get in either convention.
    """

    assert naming in ['ant', 'corr'], "naming must be 'ant' or 'corr'"

    if method not in METHODS:
        logger.warning(f"Method {method} not fully supported. Select from: {METHODS}.")

    if time is None:
        mjd = Time.now().mjd
    else:
        if 'T' in time:
            mjd = Time(time, format='isot').mjd
        else:
            # assume MJD if no "T" indicating ISOT format time
            mjd = time

    # get times of past badant lists
    et = ls.get_etcd()
    mjds = sorted([kv.key.decode().lstrip('/mon/anthealth/selfcorr/') for _, kv in et.get_prefix('/mon/anthealth/selfcorr')])
    mjds = list(filter(lambda x: x != '', mjds))  # remove blank entries
    if len(mjds) > 0:
        mjd0 = list(filter(lambda x: float(x) < mjd, mjds))[-1]  # get the latest before mjd
    else:
        logger.error("No badant found with mjds. Using default.")
        mjd0 = None

    if 'union' not in method:
        if mjd0 is None:
            dd = ls.get_dict(f'/mon/anthealth/{method}')
        else:
            dd = ls.get_dict(f'/mon/anthealth/{method}/{mjd0}')
        antstatus = dd['flagged']
        antnames = dd['antname']
        mjd = dd['time']
    elif method == 'union_and':
        # iterate over methods and take logical and per ant
        # antstatus = ...
        raise NotImplementedError
    elif method == 'union_or':
        # iterate over methods and take logical or per ant
        # antstatus = ...
        raise NotImplementedError
    else:
        logger.warning(f"method {method} not recognized")

    badants = np.array(antnames)[np.where(antstatus)].tolist()

    if naming is "corr":
        logger.debug("mapping ant to corr naming")
        badants2 = []
        for antname in badants:
            antnum = antname[:-1]
            pol = antname[-1]
            badants2.append(str(mapping.antname_to_correlator(f'LWA-{antnum}'))+pol)
        badants = badants2

    if -1 in badants:
        logger.warning("Correlator number could not be found for some antennas. Something's fishy...")

    return mjd, badants


def caltable_flags(caltable):
    """ Parse a CASA caltable and return list of antennas that are fully flagged
    """

    tab = tables.table(caltable, ack=False)
    flgdata = tab.getcol('FLAG')[...]  # True means flagged
    allflg = flgdata.all(axis=1)  # bool per [corrnum, pol]
    badants = sorted([f'{mapping.correlator_to_antname(corrnum).lstrip("LWA-")}{["A", "B"][pol]}'
                      for (corrnum, pol) in zip(*np.where(allflg))])

    return badants

