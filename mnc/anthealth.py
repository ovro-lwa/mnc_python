from dsautils import dsa_store
from lwa_antpos import mapping
from mnc import common
from astropy.time import Time
import numpy as np

logger = common.get_logger(__name__)
ls = dsa_store.DsaStore()

METHODS = ['selfcorr', 'union_and', 'union_or']


def set_badants(method, antstatus, full=False, numbering='antnum'):
    """ Set the list of bad antennas for a given badant method.
    If providing status for all 352 antennas, use full=True.
    numbering defines antenna sequence and can be "antnum" (e.g., number in "LWA-001") or "correlator" (i.e., MS/CASA number).
    """

    if method not in METHODS:
        logger.warning(f"method {method} is new. Add it to the fully supported list once validated: {METHODS}.")
    else:
        if 'union' in method.lower():
            logger.error("Cannot set antenna status with 'union'.")
            raise RuntimeError

    mjd = Time.now().mjd
    if not full:
        antstatus = [a in antstatus for a in range(352)]

    dd = {'time': mjd, 'flagged': antstatus}  # this could be expanded beyond booleans
    ls.set_dict(f'/mon/anthealth/{method}', dd)


def get_badants(method, full=False, numbering='antnum'):
    """ Given a badant method, return list of bad antennas 
    If full ordered list of 352 antennas required, set "full=True".
    numbering defines antenna sequence and can be "antnum" (e.g., number in "LWA-001") or "correlator" (i.e., MS/CASA number).
    """

    if method not in METHODS:
        logger.warning(f"method {method} is experimental. Fully suppoted methods: {METHODS}.")

    if 'union' not in method:
        dd = ls.get_dict(f'/mon/anthealth/{method}')
        if not full:
            return np.where(dd['flagged'])
        else:
            return dd['flagged']

    if method == 'union_and':
        # iterate over methods and take logical and per ant
        pass
    elif method == 'union_or':
        # iterate over methods and take logical or per ant
        pass
