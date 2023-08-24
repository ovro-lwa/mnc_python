import click
from mnc import settings, control
from mnc import myarx as a
import sigtab as s

@click.group('lwamnc')
def cli():
    pass

@cli.command()
@click.argument('antpol')
def arx_off(antpol):
    """ Turn off ARX front end for a specific ant-pol.
    """

    asig = s.name2sig(antpol)[0]
    arx = s.a2arx(asig)
    a.feeOff(arx[0],arx[1])


@cli.command()
@click.option('--filename', default=None)
def load_settings(filename):
    """ Load ARX and F-engine settings
    """

    settings.runall(filename)


@cli.command()
@click.option('--subsystem')
def print_gonogo(subsystem):
    """ Check all subsystems and print the go/no-go status for each one.
    This is meant to summarize observing state, but for more detail, check the System Health Dashboard.
    May optionally print status of a single subsystem (feng, xeng, dr).
    """

    pass


