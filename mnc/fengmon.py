import numpy as np

from lwa_f import snap2_feng_etcd_client
from mnc.common import ETCD_HOST

etcdcontrol = snap2_feng_etcd_client.Snap2FengineEtcdControl(etcdhost=ETCD_HOST)


# wrap up some etcd feng monitoring commands.
# function named after method of a block with suffix of block name (e.g., "_autocorr")

def get_new_spectra_autocorr():
    """ Safe way to get f-engine spectra.
    Returns (n_ant-pol, n_chan) array of autocorrelation spectra.
    """

    spec = np.zeros((704,4096))

    for kk in range(4):
        # Get data from all SNAPs at once
        newspecs = etcdcontrol.send_command(
            0, 'autocorr', 'get_new_spectra', kwargs={'signal_block':kk},
            timeout=30, n_response_expected=11,
            )
        for k in range(11):
            try:
                spec[k*64+kk*16:k*64+(kk+1)*16] = newspecs[k+1]
            except:
                pass

    return spec


def get_overflow_count_pfb():
    """ Get pfb overflow count for all 11 snap2s.
    Returns dict with key of snap2 number and key as the count.
    """

    count = etcdcontrol.send_command(0, 'pfb', 'get_overflow_count', timeout=30,
                                     n_response_expected=11)

    return count


def get_all_histograms_input():
    """ Get input histograms
    """

    vals, hists = etcdcontrol.send_command(0, 'input', 'get_all_histograms', timeout=30,
                                           n_response_expected=11)

    return vals, hists


def clip_count_eq():
    """ Get clip count of the eq block
    Returns a dict with key of snap2 number and value of count.
    """

    count = etcdcontrol.send_command(0, 'eq', 'clip_count', timeout=30,
                                           n_response_expected=11)

    return count
