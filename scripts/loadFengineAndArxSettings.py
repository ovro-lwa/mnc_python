# Script to configure F engines and ARX boards
# USAGE:  $python loadFengineAndArxSettings.py <eqFile.mat>

# 20221117 Initial version of F engine setting.
# 20230105 Add 7th coef set for fiber-connected signals. 
# 20230203 Include ARX setting.

import sys
import scipy.io as mat
import time

# Read configuration data file
config = mat.loadmat(sys.argv[1],squeeze_me=True)
print('Read data file',sys.argv[1])
print('Data file internal time: ',time.asctime(time.gmtime(config['time'])))


#=======================
# LOAD F ENGINE SETTINGS
#-----------------------

import myfengines as f
from fengFunctions import dsig2feng
cfgkeys = config.keys()
coef = config['coef']   # must include this key
      
for i in range(11):
    f.f[i].pfb.set_fft_shift(0x1FFF)  # Request shift at all FFT stages
    print('snap%02d:'%(i+1)," fft shift set")

dsigDone = []

k = 'eq0'   # coax length = ref+-50m
if k in cfgkeys:
    dsig = config[k]
    for i in dsig:
        loc = dsig2feng(i)
        f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[0])
        dsigDone.append(i)
    print(dsig)

k = 'eq1'   # coax: shortest
if k in cfgkeys:
    dsig = config[k]
    for i in dsig:
        loc = dsig2feng(i)
        f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[1])
        dsigDone.append(i)
    print(dsig)

k = 'eq2'   # coax: next 40m
if k in cfgkeys:
    dsig = config[k]
    for i in dsig:
        loc = dsig2feng(i)
        f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[2])
        dsigDone.append(i)
    print(dsig)

k = 'eq3'   # coax: next 40m
if k in cfgkeys:
    dsig = config[k]
    for i in dsig:
        loc = dsig2feng(i)
        f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[3])
        dsigDone.append(i)
    print(dsig)

k = 'eq4'   # coax: next 40m
if k in cfgkeys:
    dsig = config[k]
    for i in dsig:
        loc = dsig2feng(i)
        f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[4])
        dsigDone.append(i)
    print(dsig)

k = 'eq5'   # coax: longest
if k in cfgkeys:
    dsig = config[k]
    for i in dsig:
        loc = dsig2feng(i)
        f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[5])
        dsigDone.append(i)
    print(dsig)

k = 'eq6'   # fiber
if k in cfgkeys:
    dsig = config[k]
    for i in dsig:
        loc = dsig2feng(i)
        f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[6])
        dsigDone.append(i)
    print(dsig)

# All others (if any): set same as reference
for i in range(704):  # all others
    if i in dsigDone: continue
    loc = dsig2feng(i)
    f.f[loc[0]-1].eq.set_coeffs(loc[1],coef[0])

    
#======================
# NOW LOAD ARX SETTINGS
#----------------------

import myarx as a

adrs = config['adrs']
settings = config['settings']
print('ARX: ',adrs)

for i in range(len(adrs)):
    codes = ''
    for j in range(16):
        s = settings[i][j]
        codes += a.chanCode(s[0],s[1],s[2],s[3])
    try:
        a.raw(adrs[i],'SETA'+codes)
        print('Loaded: ',adrs[i],codes)
        print(settings[i])
    except:
        continue

