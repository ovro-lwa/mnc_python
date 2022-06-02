import numpy as np
import yaml
import logging
import matplotlib

from lwautils import lwa_arx   # arx

matplotlib.use('Agg')

try:
    from lwa_f import snap2_fengine, helpers  # fengine
except ImportError:
    print('No f-eng library found. Skipping.')
try:
    from lwa352_pipeline_control import Lwa352CorrelatorControl  # xengine
except ImportError:
    print('No x-eng library found. Skipping.')

from mnc import ezdr, xengine_beamformer_control


class Controller():
    """ Parse configuration and control all subsystems in uniform manner.
    Ideally, will also make it easy to monitor basic system status.
    """

    def __init__(self, config_file='lwa_config.yaml', xhosts=None, npipeline=None):
        try:
            self.logger = helpers.add_default_log_handlers(logging.getLogger(__name__ + ":%s" % (host)))
        except:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.config_file = config_file

        conf = self.parse_config(config_file)

        self.conf = conf

        self.set_properties(xhosts=xhosts, npipeline=npipeline)

    def set_properties(self, xhosts=None, npipeline=None):
        """ Set x-engine hosts and number of pipelines, then recalculate config properties.
        """

        self.xhosts = xhosts
        if self.xhosts is None:
            self.xhosts = self.conf["xengines"]["xhosts"]
        self.nhosts = len(self.xhosts)

        if npipeline is None:
            self.npipeline = self.conf["xengines"]["nxpipeline"]
        else:
            self.npipeline = npipeline

        drip_mapping = self.conf["drip_mapping"]
        drips = [ip for name in self.conf["xengines"]["x_dest_corr_name"] for ip in drip_mapping[name]]
        self.x_dest_corr_ip = list(sorted(drips*(self.npipeline//2)))
        self.x_dest_corr_port = [10001+i//self.npipeline for i in range(self.npipeline*self.nhosts)]

        # beamforming
        self.x_dest_beam_ip = self.conf["xengines"]["x_dest_beam_ip"]
        self.x_dest_beam_port = self.conf["xengines"]["x_dest_beam_port"]

    @staticmethod
    def parse_config(config_file):
        """ Parse yaml format config_file and return dict
        """

        with open(config_file, 'r') as fh:
            conf = yaml.load(fh, Loader=yaml.CSafeLoader)
        if 'fengines' not in conf:
            raise RuntimeError('Config file missing "fengines" key')
        if 'xengines' not in conf:
            raise RuntimeError('Config file missing "xengines" key')
        if 'arx' not in conf:
            raise RuntimeError('Config file missing "arx" key')
        if 'dr' not in conf:
            raise RuntimeError('Config file missing "dr" key')

        return conf

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

                f.cold_start(program = program, initialize = initialize, test_vectors = test_vectors, sync = sync,
                             sw_sync = sw_sync, enable_eth = enable_eth, chans_per_packet = chans_per_packet,
                             first_stand_index = first_stand_index, nstand = nstand, macs = macs, source_ip = source_ip,
                             source_port = source_port, dests = dests)
                
                f.print_status_all()

    def start_xengine(self):
        """ Start xengines listed in configuration file.
        """

        xconf = self.conf['xengines']

        # one p object controls all products on given subband
        p = Lwa352CorrelatorControl(self.xhosts, npipeline_per_host=self.npipeline)
        
        p.stop_pipelines()   # stop before starting
        p.start_pipelines() 
        print(f'pipelines up? {p.pipelines_are_up()}')
        self.logger.info(f'pipelines up? {p.pipelines_are_up()}')

        p.configure_corr(dest_ip=self.x_dest_corr_ip, dest_port=self.x_dest_corr_port)  # iterates over all slow corr outputs
        #p.corr_output_part.set_destination(?)   # fast (partial) correlator output?

        for pipe in p.pipelines:
            pipe.beamform_output.set_destination(self.x_dest_beam_ip, self.x_dest_beam_port) # 1 power beam
#            for b in range(2):  # two pols
#                for i in range(352):
#                    s0 = 1 if b == 0 and i == 2 else 0
#                    s1 = 1 if b == 1 and i == 2 else 0
#                    pipe.beamform.update_calibration_gains(b, 2*i+0, s0*np.ones(96, dtype=np.complex64))
#                    pipe.beamform.update_calibration_gains(b, 2*i+1, s1*np.ones(96, dtype=np.complex64))
#                    pipe.beamform.update_delays(b, np.zeros(352*2))

    def start_xengine_bf(self, num=1, target=None, track=False):
        """ Starts the xengine for beamformer observation.
        num refers to the beamformer number (1 through 4).
        target can be:
         - source name ('zenith', 'sun') or
         - tuple of (ra, dec) in (hourangle, degrees).
        """

        import glob

        if isinstance(target, tuple):
            ra, dec = target
        elif isinstance(target, str):
            ra = target
            dec = None

        c = BeamPointingControl(num)
        calfiles = glob.glob(self.xhosts['calfiles'])
        for calfile in calfiles: 
            try: 
                c.set_beam_calibration(calfile) 
            except Exception as e: 
                print(“ERROR: %s” % str(e))

        # one-time commands to point
        c.set_beam_dest()
        if target is None:
            c.set_beam_pointing(0, 90)
        else:
            c.set_beam_target(ra, dec=dec)

        # track
        if track and target is not None:
            t = BeamTracker(c, update_interval=self.xhosts['update_interval'])
            t.track(target)
        elif track and target is None:
            print("Must input target to track.")

    def stop_xengine(self):
        """ Stop xengines listed in configuration file.
        """

        p = Lwa352CorrelatorControl(self.xhosts, npipeline_per_host=self.npipeline)
        p.stop_pipelines()

    def start_dr(self, recorders=None, duration=None):
        """ Start data recorders listed recorders.
        Defaults to starting those listed in configuration file.
        duration is power beam recording in seconds.
        """

        # uses ezdr auto-discovery 'slow', 'fast', 'power', 'voltage'
        dconf = self.conf['dr']
        if recorders is None:
            recorders = dconf['recorders']

        # start ms writing
        badresults = []
        for recorder in recorders:
            lwa_drc = ezdr.Lwa352RecorderControl(recorder)
            lwa_drc.print_status()   # TODO: send to self.logger
            if recorder == 'power':
                if duration is not None:
                    d.record(duration=duration)
                else:
                    print("power beam needs duration")
            elif recorder == 'slow':
                results = lwa_drc.start()
                for result in results:
                    if result[1]['status'] != 'success':
                        badresults.append(result[1]['response'])

        if len(badresults):
            print("Data recorder not started successfully. Responses:")
            print(badresults)
            

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
          
