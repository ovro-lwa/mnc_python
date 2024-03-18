import os.path
import glob
import scipy.io as mat
import time
import numpy as np
import getpass

from mnc import myarx as a
from mnc import common
from lwa_antpos import mapping
from observing import obsstate

DATAPATH = '/home/pipeline/opsdata'

LIST_SETTINGS = sorted([(fn, os.path.basename(fn).split('-')[0]) for fn in glob.glob(DATAPATH + '/*settings*mat')], key=lambda x: x[1])
if len(LIST_SETTINGS):
    LATEST_SETTINGS = LIST_SETTINGS[-1][0]
else:
    LATEST_SETTINGS = None

# Constants
DELAY_OFFSET = 10 # minimum delay
ADC_CLOCK = 196000000    # sampling clock frequency, Hz
snaps = range(1,12)

logger = common.get_logger(__name__)

def dsig2feng(digitalSignal): # From digital sig num calculate F-unit location and signal
    funit = int(digitalSignal/64) + 1  # location, 1:11
    fsig = digitalSignal % 64          # FPGA signal number, 0:63
    return (funit,fsig)

def a2arx(asig): # asig to ARX address and ARX channel number
    adr = int(asig/16)
    chan = asig - 16*adr + 1  # channel number is 1-based
    adr += 1                  # address is 1-based
    return(adr,chan)

try:
    from lwa_f import snap2_feng_etcd_client
    ec = snap2_feng_etcd_client.Snap2FengineEtcdControl(common.ETCD_HOST)
    logger.info('Connected to ETCD host %s' % common.ETCD_HOST)
except ImportError:
    logger.warning('No f-eng library found. Skipping.')


class Settings():
    """ Class to handle settings and configuration.
    Initially defined as from reading of matlab file created by Larry.
    Ultimately, can be generalized to include etcd reading.
    """

    def __init__(self, filename=LATEST_SETTINGS):
        """ Read configuration data file """

        self.filename = filename
        self.read_settings(filename=filename)

    def read_settings(self, filename=None, filenum=None):
        """ Reads a settings file defined by full path or a time-sorted index (use list_settings to print them in order)
        if filename is defined, it will be used.
        """

        if filenum is not None and filename is None:
            filename = LIST_SETTINGS[filenum][0]
            self.filename = filename

        if self.filename is not None:
            self.config = mat.loadmat(self.filename, squeeze_me=True)
            logger.info(f'Read data file {self.filename}')
            logger.info(f'Data file internal time: {time.asctime(time.gmtime(self.config["time"]))}')
            self.cfgkeys = self.config.keys()
        else:
#            self.config = <read from etcd>
            pass

    def list_settings(self):
        """ Show time ordered list of settings files.
        
        """

        for j,k in LIST_SETTINGS:
            logger.info(f'{j}: {k[0]}')

    def get_last_settings(self):
        """ Use obsstate to read last settings loaded.
        """

        try:
            return obsstate.read_latest_setting()
        except Exception as exc:
            logger.warn(f"Failed to read settings from database. Using logfile")
            with open(path+'arxAndF-settings.log','r') as f:
                return os.path.join(DATAPATH, f.readlines()[-1].split()[-2])

    def load_feng(self, zero_unused_feng_input=False):
        """ Load settings for f-engine to the SNAP2 boards.
        """
        
        logger.info(f'Loading settings to SNAP2 boards: {snaps}')

        #=================================
        # SET F ENGINE FFT SHIFT SCHEDULE
        #---------------------------------

        if 'fftShift' in self.cfgkeys:
            fftshift = self.config['fftShift']
        else:
            fftshift = 0x1FFC

        for i in snaps:
            ec.send_command(i,'pfb','set_fft_shift',kwargs={'shift':int(fftshift)})
        logger.info(f'All FFT shifts set to {fftshift}')


        #=====================================
        # LOAD F ENGINE EQUALIZATION FUNCTIONS
        #-------------------------------------

        coef = self.config['coef']   # must include this key
        dsigDone = []
        logger.info('LOADING EQUALIZATION COEFFICIENTS')
        print('LOADING EQUALIZATION COEFFICIENTS')

        k = 'eq0'   # coax length = ref+-50m
        if k in self.cfgkeys:
            dsig = self.config[k]
            for i in dsig:
                loc = dsig2feng(i)
                if not loc[0] in snaps: continue
                ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(int(loc[1])),'coeffs':coef[0].tolist()})
                dsigDone.append(i)
            print('eq0:',dsig)

        k = 'eq1'   # coax: shortest
        if k in self.cfgkeys:
            dsig = self.config[k]
            for i in dsig:
                loc = dsig2feng(i)
                if not loc[0] in snaps: continue
                ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(loc[1]),'coeffs':coef[1].tolist()})
                dsigDone.append(i)
            print('eq1:',dsig)

        k = 'eq2'   # coax: next 40m
        if k in self.cfgkeys:
            dsig = self.config[k]
            for i in dsig:
                loc = dsig2feng(i)
                if not loc[0] in snaps: continue
                ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(loc[1]),'coeffs':coef[2].tolist()})
                dsigDone.append(i)
            print('eq2:',dsig)

        k = 'eq3'   # coax: next 40m
        if k in self.cfgkeys:
            dsig = self.config[k]
            for i in dsig:
                loc = dsig2feng(i)
                if not loc[0] in snaps: continue
                ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(loc[1]),'coeffs':coef[3].tolist()})
                dsigDone.append(i)
            print('eq3:',dsig)

        k = 'eq4'   # coax: next 40m
        if k in self.cfgkeys:
            dsig = self.config[k]
            for i in dsig:
                loc = dsig2feng(i)
                ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(loc[1]),'coeffs':coef[4].tolist()})
                dsigDone.append(i)
            print('eq4:',dsig)

        k = 'eq5'   # coax: longest
        if k in self.cfgkeys:
            dsig = self.config[k]
            for i in dsig:
                loc = dsig2feng(i)
                if not loc[0] in snaps: continue        
                ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(loc[1]),'coeffs':coef[5].tolist()})
                dsigDone.append(i)
            print('eq5:',dsig)

        k = 'eq6'   # fiber
        if k in self.cfgkeys:
            dsig = self.config[k]
            for i in dsig:
                loc = dsig2feng(i)
                if not loc[0] in snaps: continue
                ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(loc[1]),'coeffs':coef[6].tolist()})
                dsigDone.append(i)
            print('eq6:',dsig)

        for i in range(704):  # all others
            if i in dsigDone: continue
            loc = dsig2feng(i)
            if not loc[0] in snaps: continue    
            ec.send_command(loc[0],'eq','set_coeffs',kwargs={'stream':int(loc[1]),'coeffs':coef[0].tolist()})
    
        #=============================
        # LOAD F ENGINE DELAY SETTINGS
        #-----------------------------

        if 'delay_dsig' in self.config.keys():
            delays_ns = np.array(self.config['delay_dsig']) # delays in order of digital sig No., nanoseconds
    
            max_delay_ns = delays_ns.max()
            delays_clocks = np.round(delays_ns*1e-9 * ADC_CLOCK).astype(int)
            max_delay_clocks = delays_clocks.max()

            relative_delays_clocks = max_delay_clocks - delays_clocks
            max_relative_delay_clocks = delays_clocks.max()

            delays_to_apply_clocks = relative_delays_clocks + DELAY_OFFSET

            logger.info('LOADING DELAYS')
            print('LOADING DELAYS')
            print('Maximum delay: %f ns' % max_delay_ns)
            print('Maximum delay: %d ADC clocks' % max_delay_clocks)
            print('Maximum relative delay: %d ADC clocks' % max_relative_delay_clocks)
            print('Maximum delay to be applied: %d ADC clocks' % delays_to_apply_clocks.max())
            print('Minimum delay to be applied: %d ADC clocks' % delays_to_apply_clocks.min())

            for dsig in range(len(delays_ns)):
                sig = dsig2feng(dsig)
                if not sig[0] in snaps: continue
                snap_id = sig[0]
                input_id = sig[1]
                ec.send_command(snap_id, 'delay', 'set_delay', kwargs={'stream':input_id, 'delay':int(delays_to_apply_clocks[dsig])})

        #============================
        # SET UNUSED F INPUTS TO ZERO
        #----------------------------

        logger.info('TURNING OFF SPECIFIED SIGNALS')
        print('TURNING OFF SPECIFIED SIGNALS')
        if 'off' in self.cfgkeys:
            off = self.config['off']
            for dsig in off:
                if zero_unused_feng_input:
                    # Set F engine input to zero
                    feng = dsig2feng(dsig)
                    fpga = feng[0]
                    fsig = feng[1]
                    ec.send_command(fpga, 'input', 'use_zero', kwargs={'stream':int(fsig)})

                # Turn off ARX input DC power (FEE or photodiode)
                antname = mapping.correlator_to_antname(dsig//2)
                pol = 'A' if (dsig % 2 == 0) else 'B'
                address, channel = mapping.antpol_to_arx(antname, pol)
                a.feeOff(address, channel)
            print('Turned off',len(off),'signals: dsig=',off)        


    def load_arx(self):
        """ Load settings for ARX
        """
        #======================
        # NOW LOAD ARX SETTINGS
        #----------------------

        adrs = self.config['adrs']
        settings = self.config['settings']
        logger.info('LOADING ARX SETTINGS')
        print('LOADING ARX SETTINGS')
        print('addresses: ',adrs)

        for i in range(len(adrs)):
            codes = ''
            for j in range(16):
                s = settings[i][j]
                codes += a.chanCode(s[0],s[1],s[2],s[3])
            try:
                a.raw(adrs[i],'SETA'+codes)
                print('Loaded: ',adrs[i],codes)
            except:
                continue

    def update_log(self, path=DATAPATH):
        """ Add line to logging file
        """

        with open(path+'/arxAndF-settings.log','a') as f:
            t = time.time()
            print(time.asctime(time.gmtime(t)), t, getpass.getuser(), os.path.basename(self.filename), self.config['time'], sep='\t',file=f)


def update(filename=LATEST_SETTINGS):
    settings = Settings(filename=filename)
    # watch out for the order of load_arx and load_feng, load_feng currently calls feeOff through ARX.
    settings.load_arx()
    settings.update_log()   # TODO: migrate away from this and to obsstate
    settings.load_feng()
    try:
#        t_now = time.asctime(time.gmtime(time.time()))  # let add_settings handle this
        obsstate.add_settings(filename)
    except:
        logger.warning('Could not add settings to obsstate.')
