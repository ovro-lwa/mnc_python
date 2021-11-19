import numpy as np
from lwautils import lwa_arx 
from lwa_f import snap2_fengine
from lwa352_pipeline_control import Lwa352CorrelatorControl
from mnc import ezdr

# config
configfile = 'lwa_config.yaml'
# or
arxadrs = list(range(35, 46))  # currently installed
arx_config = 1 # 0, 1, 2 save preset config
snap2names = ['snap09', 'snap10', 'snap11']
xhosts = [f'lxdlwagpu{i:02}' for i in range(1, 9)] # 4.3 MHz subband per pipeline pair
xnpipeline = 4  # x-eng
x_dest_corr_ip = ['10.41.0.25', '10.41.0.25', '10.41.0.41', '10.41.0.41']
x_dest_corr_port = [10001+i//4 for i in range(32)]
x_dest_beam_ip = ['10.41.0.19'] + ['0.0.0.0']*15
x_dest_beam_port = [20001]
recorders = ['slow'] # slow, fast, power, voltage

# arx
ma = lwa_arx.ARX() 
for adr in arxadrs: 
    ma.load_cfg(adr, arx_config)

# f-eng
for snap2name in snap2names:
    f = snap2_fengine.Snap2Fengine(snap2name)
    f.fpga.print_status()  # confirm firmware name is right
    if not f.fpga.is_programmed():
        f.program()
        # sleep 30?
        f.initialize(read_only=False)

    f.cold_start_from_config(configfile)
    status = f.get_status_all() # or f.print_status_all()

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

# dr
# start ms writing
for recorder in recorders:
    lwa_drc = ezdr.Lwa352RecorderControl(recorder)  # auto-discovery
    lwa_drc.print_status()
    lwa_drc.start()

# stop ms writing
for recorder in recorders:
    lwa_drc = ezdr.Lwa352RecorderControl(recorder)  # auto-discovery 'slow', 'fast', 'power', 'voltage'
    lwa_drc.stop()
