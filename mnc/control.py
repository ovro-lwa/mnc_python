import yaml
import logging
import glob
import numpy as np
import matplotlib
import glob

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

from mnc import mcs, xengine_beamformer_control


class Controller():
    """ Parse configuration and control all subsystems in uniform manner.
    Ideally, will also make it easy to monitor basic system status.
    etcdhost is used by x-engine. data recorders use value set in mnc/common.py code.
    """

    def __init__(self, config_file='config/lwa_config.yaml', etcdhost=None, xhosts=None, npipeline=None):
        try:
            self.logger = helpers.add_default_log_handlers(logging.getLogger(f"{__name__}:{host}"))
        except:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.config_file = config_file

        conf = self.parse_config(config_file)

        self.conf = conf
        self.etcdhost = etcdhost
        self.xhosts = xhosts
        self.npipeline = npipeline
        self.set_properties()

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
        if 'etcd' not in conf:
            raise RuntimeError('Config file missing "etcd" key')

        return conf

    def set_properties(self):
        """ Set properties, then recalculate config properties.
        """

        if self.etcdhost is None:
            self.etcdhost = self.conf["etcd"]["host"]
            # TODO: also use port?

        if self.xhosts is None:
            self.xhosts = self.conf["xengines"]["xhosts"]
        self.nhosts = len(self.xhosts)

        if self.npipeline is None:
            self.npipeline = self.conf["xengines"]["nxpipeline"]

        drip_mapping = self.conf["drip_mapping"]
        drips = [ip for name in self.conf["xengines"]["x_dest_corr_name"] for ip in drip_mapping[name]]
        self.x_dest_corr_ip = list(sorted(drips*(self.npipeline)))
        self.x_dest_corr_slow_port = self.conf["xengines"]["x_dest_corr_slow_port"]
        self.x_dest_corr_fast_port = self.conf["xengines"]["x_dest_corr_fast_port"]
        # beamforming
        self.x_dest_beam_ip = self.conf["xengines"]["x_dest_beam_ip"]
        self.x_dest_beam_port = self.conf["xengines"]["x_dest_beam_port"]

        # one p object controls all products on given subband
        p = Lwa352CorrelatorControl(self.xhosts, npipeline_per_host=self.npipeline, etcdhost=self.etcdhost)
        self.pcontroller = p
        self.pipelines = p.pipelines

        # data recorder control client
        self.drc = mcs.Client()

    def set_arx(self, preset=None):
        """ Set ARX to preset config.
        Defaults to preset in configuration file.
        """

        aconf = self.conf['arx']
        if preset is None:
            preset = aconf['preset']

        ma = lwa_arx.ARX()   # TODO: update to use self.etcdhost
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

        self.pcontroller.stop_pipelines()   # stop before starting
        self.pcontroller.start_pipelines() 
        print(f'pipelines up? {self.pcontroller.pipelines_are_up()}')
        self.logger.info(f'pipelines up? {self.pcontroller.pipelines_are_up()}')

        # slow
        if 'drvs' in self.conf['dr']['recorders']:
            print("Configuring x-engine for slow visibilities")
            self.pcontroller.configure_corr(dest_ip=self.x_dest_corr_ip, dest_port=self.x_dest_corr_slow_port)  # iterates over all slow corr outputs
        else:
            print("Not configuring x-engine for slow visibilities")            

        if 'drvf' in self.conf['dr']['recorders']:
            print("Configuring x-engine for fast visibilities")
            for i in range(self.npipeline*self.nhosts):
                self.pcontroller.pipelines[i].corr_output_part.set_destination(self.x_dest_corr_ip[i], self.x_dest_corr_fast_port[i%4])
        else:
            print("Not configuring x-engine for fast visibilities")            

        if 'dr1' in self.conf['dr']['recorders']:  # TODO: generalize name
            print('Start power beams with "start_xengine_bf" method"')

    def start_xengine_bf(self, num=1, target=None, track=True):
        """ Starts the xengine for beamformer observation.
        num refers to the beamformer number (1 through 4).
        target can be:
         - source name ('zenith', 'sun') or
         - tuple of (ra, dec) in (hourangle, degrees).
        """

        if isinstance(target, tuple):
            ra, dec = target
        elif isinstance(target, str):
            ra = target
            dec = None
        else:
            print("No target specified. Pointing at zenith.")
            ra = 0
            dec = 90

        c = xengine_beamformer_control.BeamPointingControl(num, servers=self.xhosts, npipeline_per_server=self.npipeline)
        calfiles = glob.glob(self.conf['xengines']['calfiles'])
        for calfile in calfiles: 
            try: 
                c.set_beam_calibration(calfile) 
            except Exception as e: 
                print(f"ERROR: {e}")

        # one-time commands to point
        c.set_beam_dest(addr==self.conf['xengines']['x_dest_beam_port'][num-1],
                        port=self.conf['xengines']['x_dest_beam_port'][num-1])
        if target is None:
            c.set_beam_pointing(0, 90)
        else:
            c.set_beam_target(ra, dec=dec)

        # track
        if track and target is not None:
            t = xengine_beamformer_control.BeamTracker(c, update_interval=self.conf['xengines']['update_interval'])
            t.track(target)
        elif track and target is None:
            print("Must input target to track.")

    def status_xengine(self):
        """ to be implemented for more detailed monitor point info
        """
        print("Pipeline id, status:")
        for pipeline in self.pipelines:
            print(pipeline.pipeline_id, pipeline.check_connection())

    def stop_xengine(self):
        """ Stop xengines listed in configuration file.
        """

        self.pcontroller.stop_pipelines()

    def start_dr(self, recorders=None, duration=None):
        """ Start data recorders listed recorders.
        Defaults to starting those listed in configuration file.
        duration is power beam recording in seconds.
        """

        dconf = self.conf['dr']
        if recorders is None:
            recorders = dconf['recorders']

        # start ms writing
        for recorder in recorders:
            # power beams
            if recorder in [f'dr{n}' for n in range(1,11)]:
                if duration is not None:
                    self.drc.send_command(recorder, 'record', start_mjd='now', start_mpm='now', duration_ms=1000)
                else:
                    print("power beam needs duration")
            # visibilities
            elif recorder in ['drvs', 'drvf']:
                self.drc.send_command(recorder, 'start', start_mjd='now', start_mpm='now')

            if self.drc.read_monitor_point('summary', recorder) != 'success':
                self.drc.read_monitor_point('info', recorder)

    def status_dr(self, recorders=None):
        """ Print data recorder info monitor point
        """

        dconf = self.conf['dr']
        if recorders is None:
            recorders = dconf['recorders']

        # start ms writing
        statuses = []
        for recorder in recorders:
            statuses.append(self.drc.read_monitor_point('op-type', recorder).value)
            if self.drc.read_monitor_point('summary', recorder) != 'success':
                statuses.append(f"WARNING: {recorder} not fully recording: {self.drc.read_monitor_point('info', recorder).value}")

        return statuses

    def stop_dr(self, recorders=None):
        """ Stop data recorders in list recorders.
        Defaults to stopping those listed in configuration file.
        """

        dconf = self.conf['dr']
        if not recorders:
            recorders = dconf['recorders']

        for recorder in recorders:
            if recorder in ['drvs', 'drvf']:
                self.drc.send_command(recorder, 'stop', stop_mjd='now', stop_mpm='now')                
          
