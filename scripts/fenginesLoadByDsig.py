# Script to set all F engines to max FFT shift and load one of 6 sets of equalization coefficients.
# USAGE:  $python fengineLoadSettingsByDsig.py <eqFile.mat>

# 20221117 Initial version.

import sys
import scipy.io as mat

config = mat.loadmat(sys.argv[1],squeeze_me=True)
coef = config['coef']
dsig0 = config['eq0']
dsig1 = config['eq1']
dsig2 = config['eq2']
dsig3 = config['eq3']
dsig4 = config['eq4']
dsig5 = config['eq5']
#print(len(coef), len(coef[0]), len(coef[1]), len(coef[2]), len(coef[3]))

from mnc.fengFunctions import dsig2feng, myfengines

f = myfengines()

for i in range(11):
    f[i].pfb.set_fft_shift(0x1FFF)  # Request shift for all
    print('snap%02d:'%(i+1)," fft shift set")

dsigDone = []

for i in dsig0:
    loc = dsig2feng(i)
    f[loc[0]-1].eq.set_coeffs(loc[1],coef[0])
    dsigDone.append(i)
print(dsig0)
    
for i in dsig1:
    loc = dsig2feng(i)
    f[loc[0]-1].eq.set_coeffs(loc[1],coef[1])
    dsigDone.append(i)
print(dsig1)

#print(coef[2])
for i in dsig2:
    loc = dsig2feng(i)
    #print('dsig2: ',i,loc[0],loc[1])
    f[loc[0]-1].eq.set_coeffs(loc[1],coef[2])
    dsigDone.append(i)
print(dsig2)

#print(coef[3])
for i in dsig3:
    loc = dsig2feng(i)
    f[loc[0]-1].eq.set_coeffs(loc[1],coef[3])
    dsigDone.append(i)
print(dsig3)

for i in dsig4:
    loc = dsig2feng(i)
    f[loc[0]-1].eq.set_coeffs(loc[1],coef[4])
    dsigDone.append(i)
print(dsig4)

for i in dsig5:
    loc = dsig2feng(i)
    f[loc[0]-1].eq.set_coeffs(loc[1],coef[5])
    dsigDone.append(i)
print(dsig5)

for i in range(704):  # all others
    if i in dsigDone: continue
    loc = dsig2feng(i)
    f[loc[0]-1].eq.set_coeffs(loc[1],coef[0])
    

