import os.path
import yaml
import logging
from typing import Union, Callable, List
import warnings
import numpy as np
from dsautils import dsa_store
from lwautils import lwa_arx   # arx
from lwa_antpos import mapping
from astropy.time import Time, TimeDelta
import time
from mnc import settings, common, anthealth

logger = common.get_logger(__name__)

try:
    from lwa_f import snap2_fengine, helpers, snap2_feng_etcd_client  # fengine
except ImportError:
    logger.warning('No f-eng library found. Skipping.')
try:
    from lwa352_pipeline_control import Lwa352CorrelatorControl, BeamPointingControl  # xengine
except ImportError:
    logger.warning('No x-eng library found. Skipping.')

from mnc import mcs, xengine_beamformer_control


CONFIG_FILE = '/home/pipeline/proj/lwa-shell/mnc_python/config/lwa_config_calim.yaml'
FPG_FILE = '/home/pipeline/proj/lwa-shell/caltech-lwa/snap2_f_200msps_64i_4096c/outputs/snap2_f_200msps_64i_4096c.fpg'

CORE_RADIUS_M = 200.0

#: List of T-engine filter codes and their corresponding sample rates in Hz
VOLT_FILTER_CODES = {1:   250000,
                     2:   500000,
                     3:  1000000,
                     4:  2000000,
                     5:  4900000,
                     6:  9800000,
                     7: 19600000}

class Controller():
    """ Parse configuration and control all subsystems in uniform manner.
    Ideally, will also make it easy to monitor basic system status.
    etcdhost is used by x-engine. data recorders use value set in mnc/common.py code.
    Can overload default recorders by providing a list or single string name at instantiation.
    """

    def __init__(self, config_file=CONFIG_FILE, etcdhost=None, xhosts=None, npipeline=None, recorders=None):
        self.config_file = os.path.abspath(config_file)
        conf = self.parse_config(config_file)

        self.conf = conf
        self.etcdhost = etcdhost
        self.xhosts = xhosts
        self.npipeline = npipeline
        self.set_properties()

        allowed = ['drvs', 'drvf'] + [f'dr{n}' for n in range(1,11)] + ['drt1'] # correct input recorder names
        if recorders is not None:
            if isinstance(recorders, str):
                recorders = [recorders]
            # clean input
            disallowed = [recorder for recorder in recorders if recorder not in allowed]
            if len(disallowed):
                logger.info(f'Removing unexpected recorder names: {disallowed}')
                recorders = [recorder for recorder in recorders if recorder in allowed]
            self.conf['dr']['recorders'] = recorders

        # report
        modes = []
        if 'drvs' in self.conf['dr']['recorders']:
            modes.append('slow')
        if 'drvf' in self.conf['dr']['recorders']:
            modes.append('fast')
        if 'drt1' in self.conf['dr']['recorders']:
            modes.append('teng')
        for b in range(1, 11):
            if f"dr{b}" in self.conf['dr']['recorders']:
                modes.append(f"beam{b}")
        logger.info(f"Loaded configuration for {self.nhosts} x-engine host(s) running {self.npipeline} pipeline(s) each")
        logger.info(f"Supported recorder modes are: {','.join(modes)}")
        if 'beam1' in modes or 'teng' in modes or 'beam2' in modes or 'beam3' in modes or 'beam4' in modes:
            logger.info("\t Note: beams 1 and t-engine (VLBI/FRB), beam 2 (Solar), 3 (FRB), and 4 (Jovian) are reserved for specific science applications. Check with those teams before using them.")
        logger.info(f"etcd server being used is: {self.etcdhost}")

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

        # select calim names/ips for selected xhosts
        calim_name = [self.conf["xengines"]["x_dest_corr_name"][gpu_name] for gpu_name in self.xhosts]
        drips = [self.conf["drip_mapping"][name] for name in calim_name]
        self.x_dest_corr_ip = [item for item in drips for i in range(self.npipeline)]
        self.x_dest_corr_slow_port = self.conf["xengines"]["x_dest_corr_slow_port"]
        self.x_dest_corr_fast_port = self.conf["xengines"]["x_dest_corr_fast_port"]
        # beamforming
        self.x_dest_beam_ip = self.conf["xengines"]["x_dest_beam_ip"]
        self.x_dest_beam_port = self.conf["xengines"]["x_dest_beam_port"]

        # one p object controls all products on given subband
        p = Lwa352CorrelatorControl(self.xhosts, npipeline_per_host=self.npipeline, etcdhost=self.etcdhost, log=logger.getChild('Lwa352CorrelatorControl'))
        self.pcontroller = p
        self.pipelines = p.pipelines
        self.xhosts_up = sorted(set([p.host for p in self.pipelines]))

        # data recorder control client
        self.drvnums = [ip[-2:]+str(port)[-2:] for (ip, port) in zip(self.x_dest_corr_ip,
                                                                     self.conf['xengines']['x_dest_corr_slow_port']*self.nhosts)]
        self.drc = mcs.Client()
        self.bfc = {}

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

    def start_fengine(self, snap2names=None, initialize=False, program=False, force=False, updatesettings=True):
        """ Start the fengines on all snap2s.
        snap2names argument allows a list of board names (e.g., ['snap02']), but defaults configuration file.
        Optionally can initialize and program, which will also run cold_start method regardless of current state.
        The config_file used should be available to the pipeline@lwacalimxx user (e.g., /home/pipeline/proj/lwa-shell/mnc_python/config/lwa...*yaml).
        updatesettings will load the default settings file after initialization or programming.
        """

        fconf = self.conf['fengines']
        if snap2names is None:
            snap2names = fconf['snap2s_inuse']
        else:
            raise NotImplementedError("Can't deal with a single snap2 yet. Rerun with snap2names=None to restart all boards.")
        if not isinstance(snap2names, list):
            snap2names = [snap2names]

        ec = snap2_feng_etcd_client.Snap2FengineEtcdControl()
        is_programmed = ec.send_command(0, 'fpga', 'is_programmed', n_response_expected=11)

        if program:
            if (not all(is_programmed.values()) or force):
                resp = ec.send_command(0, 'feng', 'program', timeout=120, n_response_expected=1, kwargs={'fpgfile': FPG_FILE})
                if not resp:
                    raise RuntimeError('Programming failed. Check f-engine etcd service logs as pipeline@calim.')
                logger.info("Waiting 120 seconds (ignore error messages above)")
                time.sleep(120)

                is_programmed = ec.send_command(0, 'fpga', 'is_programmed', n_response_expected=11)
                if not all(is_programmed.values()):
                    raise RuntimeError('Programming of other snaps failed. Check f-engine service logs as pipeline@calim.')
                else:
                    logger.info("All snaps programmed")
                resp = ec.send_command(0, 'controller', 'stop_poll_stats_loop')
            else:
                logger.info('All snaps already programmed.')

        if initialize or program:
            is_programmed = ec.send_command(0, 'fpga', 'is_programmed', n_response_expected=11)
            if len(is_programmed) != 11 or not all(is_programmed.values()) or force:
                raise RuntimeError('Not all SNAPs programmed. Cannot initialize them.')

            resp = ec.send_command(0, 'controller', 'stop_poll_stats_loop')

            resp = ec.send_command(1, 'feng', 'cold_start_from_config',
                                   kwargs={'config_file': self.config_file, 'program': True, 'initialize': True},
                                   timeout=120)
            if resp is None:
                raise RuntimeError('cold_start_from_config failed. Check f-engine etcd service logs as pipeline@calim.')

            for snap2name in snap2names:
                if snap2name == 'snap01':
                    continue
                logger.info(f"Initializing board {snap2name}")
                snap2num = int(snap2name.lstrip('snap'))
                resp = ec.send_command(snap2num, 'feng', 'cold_start_from_config',
                                        kwargs={'config_file': self.config_file, 'program': True, 'initialize': True},
                                        timeout=0)

            logger.info("Waiting 120 seconds (ignore warning messages above)")
            time.sleep(120)
            resp = ec.send_command(0, 'controller', 'start_poll_stats_loop')
            is_programmed = ec.send_command(0, 'fpga', 'is_programmed', n_response_expected=11)
            if len(is_programmed) != 11 or not all(is_programmed.values()):
                raise RuntimeError('Not all SNAPs programmed. Check logs.')

            if updatesettings:
                settings.update()

    def status_fengine(self):
        """ Use snap2 etcd client to poll for stats on each fengine.
        """

        snap2names = self.conf['fengines']['snap2s_inuse']
        stats = {}
        ls = dsa_store.DsaStore()

        for snap2name in snap2names:
            snap2num = snap2name.lstrip('snap')

            try:
                dd = ls.get_dict(f'/mon/snap/{snap2num}')
            except Exception as e:
                raise e

            if dd is not None:
                stats[snap2name] = dd['stats']
                timestamp = dd['timestamp']
            else:
                stats[snap2name] = None
                timestamp = None

        return timestamp, stats

    def configure_xengine(self, recorders=None, calibratebeams=False, full=False, timeout=None, force=False, flag_ants: List[int]=[]):
        """ 
        Configure the xengines for beamforming.

        TODO we should really rearlly really lump this into control_bf.
        
        Args:
            recorders (list or None): List of recorders (drN) to configure.
            calibratebeams (bool): whether to use calibration for the beams
            full (bool): Stop and restart the xengines before configuring.
            force (bool): Force loading of calibration to beam.
        """
        if timeout is not None:
            warnings.warn(
                'timeout argument in configure_xengine is deprecated. It is not needed.',
                DeprecationWarning, 2)

        dconf = self.conf['dr']
        if recorders is None:
            recorders = dconf['recorders']
        elif not isinstance(recorders, (list, tuple)):
            recorders = [recorders,]

        if full:
            warnings.warn(
                'full argument in configure_xengine to be deprecated. '
                'Use stop_xengine and start_xengine instead if you want to restart x engine. '
                'Usually it is not needed.',
                DeprecationWarning, 2)
            self.stop_xengine()
            # Clear the beamformer state
            self.bfc.clear()
            self.start_xengine()

        logger.info(f'pipelines up? {self.pcontroller.pipelines_are_up()}')
        if not self.pcontroller.pipelines_are_up():
            logger.warning(f"Not all X-engine are up. Proceeding with {self.xhosts_up}.")

        if calibratebeams:
            cal_directory = self.conf['xengines']['cal_directory']
        else:
            cal_directory = '/pathshouldnotexist'

        for recorder in recorders:
            # skip vis recorders and assume dr1 sets appropriately for drt1
            if recorder in ['drvs', 'drvf', 'drt1']:
                continue

            num = int(recorder[2:])
            logger.info(f"Configuring x-engine for beam {num} on xhosts {self.xhosts_up}")
            try:
                self.bfc[num] = xengine_beamformer_control.create_and_calibrate(num, servers=self.xhosts_up,
                                                                                nserver=len(self.xhosts_up),
                                                                                npipeline_per_server=self.npipeline,
                                                                                cal_directory=cal_directory,
                                                                                etcdhost=self.etcdhost, force=force,
                                                                                flag_ants=flag_ants)
            except KeyError:
                logger.error("KeyError when creating beamformer control. Are data being sent from f to x-engines?")

            logger.info(f"Done setting calibration gains for beam {num}")
            # overload dest set by default
            if self.conf['xengines']['x_dest_beam_port'] is not None:
                addr = self.conf['xengines']['x_dest_beam_ip']
                port = self.conf['xengines']['x_dest_beam_port']
                self.bfc[num].set_beam_dest(addr=addr[num-1], port=port[num-1])

        # slow
        if 'drvs' in recorders or 'drvf' in recorders:
            logger.warn('fast and slow vis are already configured by default. '
                        'If you changed data destination or antenna selection,'
                        'you need to restart the X-engine with full=True.')

    def control_bf(self, num=1, coord=None, coordtype='celestial', targetname=None,
                   track=True, uvweight: Union[str, Callable[[float], float]]='core',
                   flag_ants: Union[str, List[str]]='caltable',
                   beam_gain=1, duration=0):
        """ Point and track beamformers.
        num refers to the beamformer number (1 through 8).
        If track=True, target is treated as celestial coords or by target name
        If track=False, target is treated as (az, el)
        uvweight can be: 'core', 'natural' or a function (see xengine_beamformer_control.set_beam_weights)
        flag_ants can be name of badant set (e.g., selfcorr or gaincal or list of antnames (not corrnums).
        Note: pol info (A/B) will be stripped to flag whole antenna.
        beam_gain optionally specifies the amplitude scaling for the beam.
        duration is time to track in seconds (0 means 12 hrs).
        target can be:
         - source name ('zenith', 'sun') or
         - tuple of (ra, dec) in (hourangle, degrees).
         - tuple of (az, el) in degrees, if track=False
        """

        az, el, ra, dec = None, None, None, None
        if coord is not None and coordtype is not None and targetname is None:
            assert isinstance(coord, tuple)
            if coordtype == 'azel':
                az, el = coord
            elif coordtype == 'celestial':
                ra, dec = coord
        elif targetname is None:
            logger.warning("Coordinates not fully specified. Pointing at zenith.")
            az = 0
            el = 90

        if num not in self.bfc:
            msg = f"X-engine not configured for beam {num}"
            logger.error(msg)
            raise KeyError(msg)

        # we convert antnames into corr_nums and ignore pol info
        if isinstance(flag_ants, str):
            mjd, flag_ants = anthealth.get_badants(flag_ants)
        assert isinstance(flag_ants, list)
        flag_ants = list({f"{mapping.antname_to_correlator('LWA-'+antname.rstrip('A').rstrip('B')):03}" for antname in flag_ants})

        if (callable(uvweight)):
            assert uvweight.__code__.co_argcount == 1, "weight function must only take one argument"
            self.bfc[num].set_beam_weighting(fnc=weight, flag_ants=flag_ants)
        elif (uvweight == 'core'):
            self.bfc[num].set_beam_weighting(fnc=_core_weight_func, flag_ants=flag_ants)
        elif (uvweight == 'natural'):
            self.bfc[num].set_beam_weighting(flag_ants=flag_ants)
        else:
            raise ValueError(f'Invalid value for weight {weight}')

        if beam_gain:
            self.bfc[num].set_beam_gain(beam_gain, flag_ants=flag_ants)  # TODO: ensure flag_ants matches that used in weighting (single call?)
        if targetname is not None:
            self.bfc[num].set_beam_target(targetname)
        elif ra is not None:
            self.bfc[num].set_beam_target(ra, dec=dec)
        elif az is not None:
            self.bfc[num].set_beam_pointing(az, el)

        if self.bfc[num].cal_set is False:
            logger.info(f'beam {num} calibration not set')

        # track
        if track:
            t = xengine_beamformer_control.BeamTracker(self.bfc[num], update_interval=self.conf['xengines']['update_interval'])
            if targetname is not None:
                t.track(targetname, duration=duration)
            elif ra is not None:
                t.track(ra, dec=dec, duration=duration)
            else:
                logging.info(f'Beam {num}: Not tracking for azel input')
        else:
            logging.info(f'Beam {num}: Not tracking')

    def status_xengine(self):
        """ Print X-engine status
        """
        AGE_THRESHOLD_S = 10
        fmt = '{:<16}{:<8}{:<14}{:<14}'
        logger.info(fmt.format("Pipeline id:", "alive", "capture_gbps", "corr_gbps"))
        for pipeline in self.pipelines:
            capture_status = pipeline.capture.get_bifrost_status()
            corr_status = pipeline.corr.get_bifrost_status()
            if capture_status is None:
                raise RuntimeError("Failed to get X-engine capture block status.")
            alive = (time.time() - corr_status['time'] < AGE_THRESHOLD_S)
            logger.info(fmt.format(f'{pipeline.host}:{pipeline.pipeline_id}',
                             str(bool(alive)),
                             f"{capture_status['gbps']:.1f}",
                             f"{corr_status['gbps']:.1f}"))

    def start_xengine(self, timeout=300):
        if self.pcontroller.pipelines_are_up():
            msg = 'Pipelines are running. Please stop the xengine first.'
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info('Starting X-engine and configuring for visibilities.')
        self.pcontroller.start_pipelines(timeout=timeout)
        try:
            self.pcontroller.configure_corr(dest_ip=self.x_dest_corr_ip, dest_port=self.x_dest_corr_slow_port)  # iterates over all slow corr outputs
        except KeyError:
            msg = "KeyError when configuring correlator. Are data being sent from f to x-engines?"
            logger.error(msg)
            raise RuntimeError(msg)

        xconf = self.conf['xengines']
        fast_antnames = xconf.get('fast_vis_ants', [])

        if len(fast_antnames):
            logger.info("Selecting antennas for fast visibilities.")
        elif len(fast_antnames) == 0:
            logger.warning("No antennas selected for fast visibilities")
            return

        # Empty array for visibility selection indices
        fast_vis_out = np.zeros([self.pcontroller.pipelines[0].corr_subsel.nvis_out, 2, 2], dtype=int)
        # "LWA-007"-style antenna names for fast visibilities
        fast_corrids = []
        for antname in fast_antnames:
            try:
                fast_corrids += [mapping.antname_to_correlator(antname)]
            except KeyError:
                logger.error(f'Couldn\'t convert antenna {antname} to a correlator index')
                logger.info('Continuing without this antenna')
        # Construct list of all baselines (including autos) for antennas in the list
        fast_vis_n = 0
        for corridn_a, corrid_a in enumerate(fast_corrids):
            for corridn_b, corrid_b in enumerate(fast_corrids[corridn_a:]):
                # Add all 4 polarization products -- must match the order expected by the data recorder.
                fast_vis_out[fast_vis_n+0] = [[corrid_a, 0], [corrid_b, 0]]
                fast_vis_out[fast_vis_n+1] = [[corrid_a, 0], [corrid_b, 1]]
                fast_vis_out[fast_vis_n+2] = [[corrid_a, 1], [corrid_b, 0]]
                fast_vis_out[fast_vis_n+3] = [[corrid_a, 1], [corrid_b, 1]]
                fast_vis_n += 4
        for i in range(self.npipeline*self.nhosts):
            self.pcontroller.pipelines[i].corr_subsel.set_baseline_select(fast_vis_out)
            self.pcontroller.pipelines[i].corr_output_part.set_destination(self.x_dest_corr_ip[i], self.x_dest_corr_fast_port[i%4])

    def stop_xengine(self):
        """ Stop xengines listed in configuration file.
        """

        self.pcontroller.stop_pipelines()
        time.sleep(20)

    def start_dr(self, recorders=None, t0='now', duration=None, time_avg=1, teng_f1=None, teng_f2=None, f0=7, gain1=6, gain2=6):
        """ Start data recorders listed recorders.
        Defaults to starting those listed in configuration file.
        Recorder list can be overloaded with 'drvs' (etc) or individual recorders (e.g., 'drvs7601').
        t0 is either 'now' or a start time (astropy Time, mjd float, and isot strings supported).
        duration is length of data recording in milliseconds (required for power beam recording; optional for visibilities).
        time_avg is power beam averaging time in milliseconds (integer converted to next lower power of 2).
        teng_f1/2 are the central frequencies of t-engine tunings in units of Hz.
        f0 sets bandwidth as integer from 1 (250kHz) to 7 (19.6MHz).
        gain1/2 are t-engine re-quantization gains from 0 (most gain) to 15 (least gain).
        """

        dconf = self.conf['dr']
        if recorders is None:
            recorders = dconf['recorders']
        elif not isinstance(recorders, (list, tuple)):
            recorders = [recorders,]

        # set start time arguments
        if isinstance(t0, str) and t0 == 'now':
            mjd = mpm = t0
            start = Time.now()
        else:
            if not isinstance(t0, Time):
                try:
                    start = Time(t0, format='isot')
                except ValueError:
                    start = Time(t0, format='mjd')
            else:
                start = t0

            mjd_dt = start.mjd % 1
            mjd = int((start - TimeDelta(mjd_dt, format='jd')).mjd)
            mpm = int(mjd_dt * 24 * 3600 * 1e3)
                
        # start ms writing
        logger.info(f"Starting recorders {recorders} at {start.mjd} (currently {Time.now().mjd})")
        for recorder in recorders:
            accepted = False

            # power beams
            if recorder in [f'dr{n}' for n in range(1,11)]:
                if duration is not None:
                    assert isinstance(time_avg, int)
                    time_avg = 2 ** int(np.log2(time_avg))  # set to next lower power of 2
                    accepted, response = self.drc.send_command(recorder, 'record', start_mjd=mjd,
                                                               start_mpm=mpm, duration_ms=duration,
                                                               time_avg=time_avg)
                else:
                    logger.warn("Power beam recordings require a duration")

            elif recorder in [f'drt{n}' for n in range(1,3)]:
                assert teng_f1 is not None and teng_f2 is not None, "Need to set teng_f1, teng_f2 frequencies"
                assert f0 in VOLT_FILTER_CODES, "f0 (filter number) should be from 1-7 (inclusive)"
                if duration is not None:
                    assert time_avg in [None, 0, 1], "No time averaging can be done for t-engine"

                beam = int(recorder[3:])

                # Check for valid tunings
                assert teng_f1 > VOLT_FILTER_CODES[f0]/2, "t-engine tuning 1 frequency too low."
                assert teng_f2 > VOLT_FILTER_CODES[f0]/2, "t-engine tuning 2 frequency too low."
                assert teng_f1 < (196e6/2-VOLT_FILTER_CODES[f0]/2), "t-engine tuning 1 frequency too high."
                assert teng_f2 < (196e6/2-VOLT_FILTER_CODES[f0]/2), "t-engine tuning 2 frequency too high."

                accepted1, response = self.drc.send_command(f"drt{beam}", "drx", beam=beam, tuning=1,
                                                            central_freq=teng_f1, filter=f0, gain=gain1)
                if accepted1:
                    accepted2, response = self.drc.send_command(f"drt{beam}", "drx", beam=beam, tuning=2,
                                                                central_freq=teng_f2, filter=f0, gain=gain2)

                if accepted1 and accepted2:
                    accepted, response = self.drc.send_command(recorder, 'record', start_mjd=mjd, beam=beam,
                                                               start_mpm=mpm, duration_ms=duration)
                else:
                    logger.warn("tengine tuning command(s) not successful. Not starting data recorder.")
                    
            elif recorder in ["drvs", "drvf"] + ["drvs" + num for num in self.drvnums]:
                accepted, response = self.drc.send_command(recorder, "start", mjd=mjd, mpm=mpm)
                if duration is not None:
                    if response["status"] != "success":
                        logger.warn("Data recorder not started successfully. Trying to schedule stop...")
                    stop = start + TimeDelta(duration/1e3/24/3600, format="jd")
                    self.stop_dr(recorders=recorder, t0=stop)
            else:
                logger.info(f"recorder name {recorder} not recognized.")
                accepted = False

            if not accepted:
                logger.warn(f"no response from {recorder}")
            elif response['status'] == 'success':
                rec_extra_info = ''
                try:
                    rec_extra_info = f" for {duration/1000.0:.3f} s and {time_avg} ms averaging to file {response['response']['filename']}"
                except (KeyError, TypeError):
                    pass
                logger.info(f"recording on {recorder}{rec_extra_info}")
            else:
                logger.warn(f"recording on {recorder} failed: {response['response']}")
                
            if self.drc.read_monitor_point('summary', recorder).value != 'normal':
                self.drc.read_monitor_point('info', recorder)

    def status_dr(self, recorders=None):
        """ Print data recorder info monitor point
        """

        dconf = self.conf['dr']
        if recorders is None:
            recorders = dconf['recorders']
        elif not isinstance(recorders, (list, tuple)):
            recorders = [recorders,]

        # start ms writing
        statuses = []
        for recorder in recorders:
            statuses.append(self.drc.read_monitor_point('op-type', recorder).value)
            if self.drc.read_monitor_point('summary', recorder).value != 'normal':
                statuses.append(f"WARNING: {recorder} not fully operational: {self.drc.read_monitor_point('info', recorder).value}")

        return statuses

    def stop_dr(self, recorders=None, t0='now', queue_number=0):
        """ Stop data recorders in list recorders.
        Defaults to stopping those listed in configuration file.
        t0 is stop time (astropy Time object, mjd, or isot format supported).
        queue_number is index of queued beamformer observations (default stops most recent == 0).
        """

        dconf = self.conf['dr']
        if not recorders:
            recorders = dconf['recorders']
        elif not isinstance(recorders, (list, tuple)):
            recorders = [recorders,]

        # set start time arguments
        if isinstance(t0, str):
            assert t0 == 'now'
            mjd = mpm = t0
            start = Time.now()
        else:
            if not isinstance(t0, Time):
                try:
                    start = Time(t0, format='isot')
                except ValueError:
                    start = Time(t0, format='mjd')
            else:
                start = t0

            mjd_dt = start.mjd % 1
            mjd = int((start - TimeDelta(mjd_dt, format='jd')).mjd)
            mpm = int(mjd_dt * 24 * 3600 * 1e3)
                
        for recorder in recorders:
            if recorder in ['drvs', 'drvf']:
                accepted, response = self.drc.send_command(recorder, 'stop', mjd=mjd, mpm=mpm)
            else:
                accepted, response = self.drc.send_command(recorder, 'cancel', queue_number=queue_number)

            if not accepted:
                logger.warn(f"no response from {recorder}")
            elif response['status'] == 'success':
                logger.info(f"recording on {recorder} stopped")
            else:
#                badrec = [k for k,v in response['response'].items() if v['status'] != 'success']
                logger.warn(f"stopping recording on {recorder} failed: {response}")

def _core_weight_func(r: float) -> float:
    return 1.0 if r < CORE_RADIUS_M else 0.0

def _single_ant_flags_list(antname: str) -> List[int]:
    flag_list = list(range(352))
    flag_list.remove(mapping.antname_to_correlator(antname))
    return flag_list
