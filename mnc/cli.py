import click
from mnc import settings, control
from mnc import myarx


# mapping larry's proposed analog and digital numbering
def a2arx(asig):
    """ asig to ARX address and ARX channel number
    """
    
    adr = int(asig/16)
    chan = asig - 16*adr + 1  # channel number is 1-based
    adr += 1                  # address is 1-based
    return(adr,chan)


def name2sig(name):
    i=0
    while i<len(antNames):
        if antNames[i]==name:
            break
        i = i+1
    if i<len(antNames):
        return(sigtab[i])
    else:
        return(None)

@click.group('lwamnc')
def cli():
    pass

@cli.command()
@click.argument('antpol')
def arx_off(antpol):
    """ Turn off ARX front end for a specific ant-pol.
    """

    asig = name2sig(antpol)[0]
    arx = a2arx(asig)
    myarx.feeOff(arx[0], arx[1])


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


@cli.command()
def rfi_summary():
    """ Use f-engine to create summary of antenna and RFI issues
    """

    pass
