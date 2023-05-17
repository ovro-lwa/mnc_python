Example Observations with OVRO-LWA
========================================

Common initial steps:

```
from mnc import control 
con = control.Controller()
con.status_xengine()
con.status_dr()
```

The status methods display details about how subsystems are set up or if they have errors. This info is also displayed on the OVRO-LWA System Health Dashboard.

Next, set up the x-engine. To set it up to route data for both slow visibility recording and beamforming with beam 1:

```
con.configure_xengine(recorders=['drvs', 'dr1'], full=True)
```

After this step, the x-engine status on the dashboard should be green, which means data is being routed from the GPU servers to the data recorders on the Cal-Im nodes. If the x-engine was already set up correctly, then this step will go faster with `full=False`.

Next, you need to turn the data recorders on. To start slow visibility recording:
```
con.start_dr(recorders='drvs')
```

The default is to start immediately and record without end (until the `stop_dr` command). However, the `t0` and `duration` arguments can define the start and length of an observation. `t0` can be in MJD or ISOT format and `duration` must be in milliseconds (e.g., `duration=30*60*1e3` for 30 minutes).

To start beamformer recording:
```
con.start_dr(recorders='dr1', t0='now', duration=None, time_avg=1)
con.control_bf(num=1, targetname=<name or (hourangle, deg)>, track=True)
```

The number in the name of the data recorder is the same as that used by `control_bf`. Note that the `control_bf` command will start a loop in Python to update the beam pointing. You'll need to Ctrl-C to stop tracking and/or open a new terminal to issue other commands (e.g., stop the data recorder).

Finally, to stop observing:
```
con.stop_dr()
con.stop_xengine()
```


* :ref:`genindex`
* :ref:`search`
