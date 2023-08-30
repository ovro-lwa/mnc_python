import numpy as np

from lwa_f import snap2_feng_etcd_client
from lwa_f import snap2_fengine
from mnc.common import ETCD_HOST
from lwa_antpos import mapping


etcdcontrol = snap2_feng_etcd_client.Snap2FengineEtcdControl(etcdhost=ETCD_HOST)
isodd = lambda a: bool(a % 2)


# wrap up some etcd feng monitoring commands.
# function named after method of a block with suffix of block name (e.g., "_autocorr")

def get_ant_autocorr(corrnum=None, antnum=None):
    """ Gets pol A/B for a given corrnum or antnum
    """

    antname = None
    if corrnum is None and antnum is not None:
        antname = f'LWA-{antnum:03d}'
    elif corrnum is not None and antnum is None:
        antname = mapping.correlator_to_antname(corrnum)

    assert antname is not None, "must define corrnum or antnum"

    snapnum, snapinp = mapping.antpol_to_fpga(antname, 'A')
    block = snapinp // 16
    newspecs = etcdcontrol.send_command(snapnum, 'autocorr', 'get_new_spectra',
                                        kwargs={'signal_block':block},
                                        timeout=30, n_response_expected=1)

    return np.array(newspecs[snapinp % 16]), np.array(newspecs[(snapinp % 16) + 1])


def get_all_spectra_autocorr():
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


def adc_power(f,sigs):
    """ Get array of f-engine ADC input power
    """

    ff = snap2_fengine.Snap2FengineEtcd('snap'+str(f).zfill(2))
    d = np.array(ff.input.get_bit_stats())
    var = d[1][sigs]-d[0][sigs]**2   #variance
    var = [max(var[i],0) for i in range(len(var))]   # none < 0
    std = np.sqrt(var)                                  #standard deviation
    pd = [var[i]/512/512/100 for i in range(len(var))]  #power in watts
    return((std,pd))


def plotsomething():
    """ Placeholder for plotting like Greg's rfi script
    """

    feng = range(1, 12)
    pd = np.array([adc_power(i, range(64))[1] for i in feng])*1e3
    today = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    raise NotImplementedError


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
