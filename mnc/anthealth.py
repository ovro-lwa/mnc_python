from dsautils import dsa_store
from lwa_antpos import mapping
from mnc import common
from astropy.time import Time
import numpy as np

logger = common.get_logger(__name__)
ls = dsa_store.DsaStore()

METHODS = ['selfcorr', 'union_and', 'union_or']


def set_badants(method, badants, numbering='antnum'):
    """ Set the antenna status for a given badant method.
    badants should be a list of integers of bad antannas.
    numbering defines antenna sequence and can be "antnum" (e.g., number in "LWA-001") or "correlator" (i.e., MS/CASA number).
    Numbering in etcd is in antnum order, but values can be set/get in either convention.
    """

    assert isinstance(badants, list), "badants must be a list"
    assert isinstance(badants[0], int), "badants must hold ints"
    assert numbering in ['antnum', 'correlator'], "numbering must be 'antnum' or 'correlator'"

    if method not in METHODS:
        logger.warning(f"method {method} is new. Add it to the fully supported list once validated: {METHODS}.")
    else:
        if 'union' in method.lower():
            logger.error("Cannot set antenna status with 'union'.")
            raise RuntimeError

    if numbering is "correlator":
        logger.debug("mapping correlator number to antnum")
        badants2 = []
        for corrnum in badants:
            badants2.append(mapping.correlator_to_antname(corrnum).lstrip('LWA-'))
        badants = badants2

    antstatus = [a in badants for a in range(352)]  # make list of status for all ants in antnum order

    mjd = Time.now().mjd
    dd = {'time': mjd, 'flagged': antstatus}  # this could be expanded beyond booleans
    ls.put_dict(f'/mon/anthealth/{method}', dd)


def get_badants(method, numbering='antnum'):
    """ Given a badant method, return list of bad antennas
    numbering defines antenna sequence and can be "antnum" (e.g., number in "LWA-001") or "correlator" (i.e., MS/CASA number).
    Numbering in etcd is in antnum order, but values can be set/get in either convention.
    """

    assert numbering in ['antnum', 'correlator'], "numbering must be 'antnum' or 'correlator'"

    if method not in METHODS:
        logger.warning(f"method {method} is experimental. Fully suppoted methods: {METHODS}.")

    if 'union' not in method:
        dd = ls.get_dict(f'/mon/anthealth/{method}')
        antstatus = dd['flagged']
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

    badants = np.where(antstatus)[0].tolist()
        
    if numbering is "correlator":
        logger.debug("mapping antnum to correlator")
        badants2 = []
        for antnum in badants:
            badants2.append(mapping.antname_to_correlator(f'LWA-{antnum}'))
        badants = badants2

    return badants

