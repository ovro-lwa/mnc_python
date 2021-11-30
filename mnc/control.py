import numpy as np
import yaml
from lwautils import lwa_arx   # arx
from lwa_f import snap2_fengine, helpers  # fengine
from lwa352_pipeline_control import Lwa352CorrelatorControl  # xengine
from mnc import ezdr  # dr


class Controller():
    """ Parse configuration and control all subsystems in uniform manner.
    Ideally, will also make it easy to monitor basic system status.
    """

    self.logger = logger or helpers.add_default_log_handlers(logging.getLogger(__name__ + ":%s" % (host)))

    def __init__(self, config_file='lwa_config.yaml', **kwargs):
        with open(config_file, 'r') as fh:
            conf = yaml.load(fh, Loader=yaml.CSafeLoader)
        if 'fengines' not in conf:
            self.logger.error("No 'fengines' key in output configuration!")
            raise RuntimeError('Config file missing "fengines" key')
        if 'xengines' not in conf:
            self.logger.error("No 'xengines' key in output configuration!")
            raise RuntimeError('Config file missing "xengines" key')
        if 'arx' not in conf:
            self.logger.error("No 'arx' key in output configuration!")
            raise RuntimeError('Config file missing "arx" key')
        if 'dr' not in conf:
            self.logger.error("No 'dr' key in output configuration!")
            raise RuntimeError('Config file missing "dr" key')

        # TODO: overload with kwargs
        for key, value in kwargs.items():
            print(key, value)
            
        self.conf = conf

    def set_arx(self):
        aconf = self.conf['arx']

        ma = lwa_arx.ARX() 
        for adr in arxadrs: 
            ma.load_cfg(adr, arx_config)

            
    def start_fengine(self, initialize=False, program=False):
        """ 
        """

        fconf = self.conf['fengines']
        snap2names = fconf['snap2_inuse']
        chans_per_packet = fconf['chans_per_packet']

        macs = self.conf['xengines']['arp']
        dests = []
        for xeng, chans in self.conf['xengines']['chans'].items():
            dest_ip = xeng.split('-')[0]
            dest_port = int(xeng.split('-')[1])
            start_chan = chans[0]
            nchan = chans[1] - start_chan
            dests += [{'ip':dest_ip, 'port':dest_port, 'start_chan':start_chan, 'nchan':nchan}]


        for snap2name in snap2names:
            print(f'Starting f-engine on {snap2name}')
            self.logger.info(f'Starting f-engine on {snap2name}')

            localconf = fconf.get(self.hostname, None)
            if localconf is None:
                self.logger.error("No configuration for F-engine host %s" % self.hostname)
                raise RuntimeError("No config found for F-engine host %s" % self.hostname)

            first_stand_index = localconf['ants'][0]  # Should refer to first SNAP in use?
            nstand = localconf['ants'][1] - first_stand_index
            source_ip = localconf['gbe']
            source_port = localconf['source_port']

            self.cold_start(program = program, initialize = initialize, test_vectors = test_vectors, sync = sync,
                            sw_sync = sw_sync, enable_eth = enable_eth, chans_per_packet = chans_per_packet,
                            first_stand_index = first_stand_index, nstand = nstand, macs = macs, source_ip = source_ip,
                            source_port = source_port, dests = dests)

            f = snap2_fengine.Snap2Fengine(snap2name)
            f.print_status_all()

    def start_xengine(self):
        xconf = self.conf['xengines']
# x-eng
p = Lwa352CorrelatorControl(xhosts, npipeline_per_host=xnpipeline)
        
# start them
p.stop_pipelines()   # stop then start
p.start_pipelines() 
p.pipelines_are_up()

# each pipeline pair needs the same destination IP/port numbers, which should be unique from all other pairs
p.configure_corr(dest_ip=x_dest_corr_ip, dest_port=x_dest_corr_port)

# beamforming data recorder
for p in pipelines:
    p.beamform_output.set_destination(x_dest_beam_ip, x_dest_beam_port) # 1 power beam
    for b in range(2):  # two pols
        for i in range(352):
            s0 = 1 if b == 0 and i == 2 else 0
            s1 = 1 if b == 1 and i == 2 else 0
            p.beamform.update_calibration_gains(b, 2*i+0, s0*np.ones(96, dtype=np.complex64))
            p.beamform.update_calibration_gains(b, 2*i+1, s1*np.ones(96, dtype=np.complex64))
            p.beamform.update_delays(b, np.zeros(352*2)) 

p.stop_pipelines()

    def start_dr(self):
        dconf = self.conf['dr']
            
# start ms writing
for recorder in recorders:
    lwa_drc = ezdr.Lwa352RecorderControl(recorder)  # auto-discovery
    lwa_drc.print_status()
    lwa_drc.start()

# stop ms writing
for recorder in recorders:
    lwa_drc = ezdr.Lwa352RecorderControl(recorder)  # auto-discovery 'slow', 'fast', 'power', 'voltage'
    lwa_drc.stop()
