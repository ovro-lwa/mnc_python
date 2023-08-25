import click
from mnc import settings, control
from mnc import myarx
from lwa_antpos import mapping

@click.group('lwamnc')
def cli():
    pass


@cli.command()
@click.argument('antpol')
def arx_off(antpol):
    """ Turn off ARX front end for a specific ant-pol.
    """

    assert 'A' in antpol.upper() or 'B' in antpol.upper()
    address, channel = mapping.antpol_to_arx(antpol[:-1], antpol[-1])
    myarx.feeOff(address, channel)


@cli.command()
@click.argument('antpol')
def arx_on(antpol):
    """ Turn on ARX front end for a specific ant-pol.
    """

    assert 'A' in antpol.upper() or 'B' in antpol.upper()
    address, channel = mapping.antpol_to_arx(antpol[:-1], antpol[-1])
    myarx.feeOn(address, channel)


@cli.command()
@click.option('--filename', default=None)
def load_settings(filename):
    """ Load ARX and F-engine settings
    """

    settings.update(filename)


@cli.command()
@click.option('--full', default=False)
def start_xengine(full):
    """ Turn on x-engine with basic mnc-python interface
    """

    con = control.Controller()
    con.configure_xengine(full=full)


@cli.command()
@click.option('--subsystem', default=None)
def print_gonogo(subsystem):
    """ Check all subsystems and print the go/no-go status for each one.
    This is meant to summarize observing state, but for more detail, check the System Health Dashboard.
    May optionally print status of a single subsystem (feng, xeng, dr).
    """

    from mnc import mcs, control
    from astropy import time

    t_stale = 10
    con = None

    if subsystem in ['feng', None]:
        from dateutil.parser import parse
        from dsautils import dsa_store
        ls = dsa_store.DsaStore()
        status_f =  []
        for snapnum in range(1, 12):
            status = ls.get_dict(f'/mon/snap/{snapnum:02}/status')
            t_age = time.Time.now().unix-time.Time(parse(status['timestamp'])).unix
            status_f.append((str(snapnum), (status['ok'] and t_age < t_stale)))
    else:
        status_f = None
    
    if subsystem in ['xeng', None]:
        con = control.Controller()
        status_x = []
        hostids = [f'{pp.host[-2:]}{pp.pipeline_id}' for pp in con.pipelines]
        for host in con.xhosts:
            for ii in range(con.npipeline):
                hostid = host[-2:] + str(ii)
                if hostid in hostids:
                    pp = con.pipelines[hostids.index(hostid)]
                    t_age = time.Time.now().unix-pp.capture.get_bifrost_status()['time']
                    rate = pp.capture.get_bifrost_status()['gbps']
                    status_x.append((hostid, (rate > 10 and t_age < t_stale)))
                else:
                    status_x.append((hostid, False))
    else:
        status_x = None

    if subsystem in ['dr', None]:
        if con is None:
            con = control.Controller()
        recorders = con.conf['dr']['recorders'].copy()
        if 'drvs' in recorders:
            recorders.remove('drvs')
            for num in con.drvnums[::2]:  # one per pair
                recorders.append('drvs'+str(num))

        if 'drvf' in recorders:
            recorders.remove('drvf')
            for num in con.drvnums[::2]:  # one per pair
                recorders.append('drvf'+str(num))
        status_dr = []
        for dr in recorders:
            summary = mcs.Client(dr).read_monitor_point('summary')
            t_age = time.Time.now().unix-summary.timestamp
            status_dr.append((dr.lstrip('dr'), (summary.value == 'normal' and t_age < t_stale)))
    else:
        status_dr = None

    return status_f, status_x, status_dr


@cli.command()
def rfi_summary():
    """ Use f-engine to create summary of antenna and RFI issues
    """

    raise NotImplementedError
