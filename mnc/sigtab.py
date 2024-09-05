#!/var/bin python
# LWA signal number conversion module
# Convert among digital signal number (dsig), analog signal number (asig),
# antenna name, and SNAP2 or ARX board.

# 20230601  Initial version.
# 20230627  Fix bug in name2sig if name not found.
# 20240228  Use opsdatapath.py to find sigtab.mat

import scipy.io as mat
# import opsdatapath as dp

d = mat.loadmat('/home/pipeline/opsdata/' +'sigtab.mat')
sigtab = d['sigtab']
antNames = d['antNames']

def d2a(dsig):   # dsig to asig
    return(sigtab[dsig][0])

def a2d(asig):   # asig to dsig
    for i in range(len(sigtab)):
        if sigtab[i][0]==asig:
            return(sigtab[i][1])
    return(None)

def d2name(dsig):# dsig to antenna name
    return(antNames[dsig])

def a2name(asig):# asig to antenna name
    for i in range(len(sigtab)):
        if sigtab[i][0]==asig:
            return(antNames[i])
    return(None)

def d2feng(dsig):# dsig to SNAP2 id and SNAP2 signal number
    snap = int(dsig/64) + 1
    sig = dsig % 64
    return(snap,sig)

def a2arx(asig): # asig to ARX address and ARX channel number
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
