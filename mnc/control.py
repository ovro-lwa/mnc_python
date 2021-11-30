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
        self.config_file = config_file

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

    def set_arx(self, preset=None):
        """ Set ARX to preset config.
        Defaults to preset in configuration file.
        """

        aconf = self.conf['arx']
        if preset is None:
            preset = aconf['preset']

        ma = lwa_arx.ARX() 
        for adr in aconf['adrs']:
            ma.load_cfg(adr, preset)

    def start_fengine(self, initialize=False, program=False, force=False):
        """ Start the fengines on all snap2s.
        Defaults to all listed in "snap2s_inuse" field of configuration file.
        Optionally can initialize and program.
        force will run cold_start method regardless of current state.
        """

        fconf = self.conf['fengines']
        snap2names = fconf['snap2s_inuse']
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
            f = snap2_fengine.Snap2Fengine(snap2name)
            if program and f.fpga.is_programmed():
                print(f'{snap2name} is already programmed.')
            if f.is_connected() and not force:
                print(f'{snap2name} is already connected.')
                continue
            else:
                print(f'Starting f-engine on {snap2name}')
                self.logger.info(f'Starting f-engine on {snap2name}')

                localconf = fconf.get(snap2name, None)
                if localconf is None:
                    self.logger.error(f"No configuration for F-engine host {snap2name}")
                    raise RuntimeError(f"No config found for F-engine host {snap2name}")
                
                first_stand_index = localconf['ants'][0]
                nstand = localconf['ants'][1] - first_stand_index
                source_ip = localconf['gbe']
                source_port = localconf['source_port']

                self.cold_start(program = program, initialize = initialize, test_vectors = test_vectors, sync = sync,
                                sw_sync = sw_sync, enable_eth = enable_eth, chans_per_packet = chans_per_packet,
                                first_stand_index = first_stand_index, nstand = nstand, macs = macs, source_ip = source_ip,
                                source_port = source_port, dests = dests)
                
                f.print_status_all()

    def start_xengine(self):
        """ Start xengines listed in configuration file.
        """

        xconf = self.conf['xengines']
        p = Lwa352CorrelatorControl(xconf['xhosts'], npipeline_per_host=xconf['xnpipeline'])
        ## QUESTION: p controls all pipelines on all hosts?
        
        p.stop_pipelines()   # stop before starting
        p.start_pipelines() 
        print(f'pipelines up? {p.pipelines_are_up()}')
        self.logger.info(f'pipelines up? {p.pipelines_are_up()}')

        # QUESTION: is this standard after start_pipelines?
        p.configure_corr(dest_ip=xconf['x_dest_corr_ip'], dest_port=xconf['x_dest_corr_port'])

        # QUESTION: how to identify whether beamformed data is being produced?
        for p in pipelines:   # QUESTION: how to list all pipelines?
            p.beamform_output.set_destination(xconf['x_dest_beam_ip'], xconf['x_dest_beam_port']) # 1 power beam
            for b in range(2):  # two pols
                for i in range(352):
                    s0 = 1 if b == 0 and i == 2 else 0
                    s1 = 1 if b == 1 and i == 2 else 0
                    p.beamform.update_calibration_gains(b, 2*i+0, s0*np.ones(96, dtype=np.complex64))
                    p.beamform.update_calibration_gains(b, 2*i+1, s1*np.ones(96, dtype=np.complex64))
                    p.beamform.update_delays(b, np.zeros(352*2))

    def stop_xengine(self):
        """ Stop xengines listed in configuration file.
        """

        xconf = self.conf['xengines']
        p = Lwa352CorrelatorControl(xconf['xhosts'], npipeline_per_host=xconf['xnpipeline'])
        p.stop_pipeline()

    def start_dr(self, recorders=None):
        """ Start data recorders listed recorders.
        Defaults to starting those listed in configuration file.
        """

        # uses ezdr auto-discovery 'slow', 'fast', 'power', 'voltage'
        dconf = self.conf['dr']
        if not recorders:
            recorders = dconf['recorders']

        # start ms writing
        for recorder in recorders:
            lwa_drc = ezdr.Lwa352RecorderControl(recorder)
            lwa_drc.print_status()   # TODO: send to self.logger
            lwa_drc.start()

    def stop_dr(self, recorders=None):
        """ Stop data recorders in list recorders.
        Defaults to stopping those listed in configuration file.
        """

        dconf = self.conf['dr']
        if not recorders:
            recorders = dconf['recorders']

        for recorder in recorders:
            lwa_drc = ezdr.Lwa352RecorderControl(recorder)
            lwa_drc.stop()
          
