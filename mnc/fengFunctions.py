# myfengFunctions.py
# Low-level F-engine control
# by Larry D'Addario with small modifications by Casey Law


# 20220310 - add adcPowerSave().
# 20221006 - add get_spectra().

import numpy as np
import math as m

arxsigs = [2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13] #(digitalSigNo) for each ARX signal.
# This works for for A Lower.  For the others:
# AL = 0, AU = 16, BL = 32, BU = 48.  Add these to arxsigs for other digitizer boards.

sigs5=[36,37,*range(40,52)]
sigs6=[*range(0,56)]
sigs7=[*range(0,56),58,59]
sigs8=[*range(0,52)]
sigs9=[*range(0,52),54,55]
sigs10=[*range(0,52),54,55]
sigs11=[*range(0,52),54,55]
sigs=[sigs5,sigs6,sigs7,sigs8,sigs9,sigs10,sigs11]


def myfengines():
    """ Get list of f-eng control instances"""
    snaps = range(1,12)

    import numpy as np
    from lwa_f import snap2_fengine

    arxsigs = [2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13] #(digitalSigNo) for each ARX signal.
    # This works for for A Lower.  For the others:
    # AL = 0, AU = 16, BL = 32, BU = 48.  Add these to arxsigs for other digitizer boards.

    f = []
    for i in range(len(snaps)):
        id='snap'+'%02d'%(snaps[i])
        print("###SNAP2:",id)
        try:
            f.append(snap2_fengine.Snap2FengineEtcd(id))
        except:
            continue

        if not f[i].fpga.is_programmed():
            f[i].program()
            f[i].initialize(False)
            if not f[i].adc.mmcm_is_locked():
                if not f[i].adc.initialize():
                    print('ADC delay calibration failed.')
            f[i].sync.arm_sync()
#            f[i].sync.sw_sync()
    return f


def save_spectra(f,filename):
    # Get spectra for all 64 signals and write to a file.

    f.adc.use_data()

    f.autocorr.set_acc_len(16384)   # T = 16384*4096/98MHz = 0.685s
    s0 = f.autocorr.get_new_spectra(0)
    s1 = f.autocorr.get_new_spectra(1)
    s2 = f.autocorr.get_new_spectra(2)
    s3 = f.autocorr.get_new_spectra(3)
    s = np.vstack((s0,s1,s2,s3))
    print(np.shape(s))
    
    outfile = open(filename,'w')
    [print(str(list(s[i])).strip('[]'), file=outfile) for i in range(len(s))]
    outfile.close()
    return True

def get_spectra(f):
    # Return spectra for all 64 signals of one SNAP2 board
    f.adc.use_data()

    f.autocorr.set_acc_len(16384)   # T = 16384*4096/98MHz = 0.685s
    s0 = f.autocorr.get_new_spectra(0)
    s1 = f.autocorr.get_new_spectra(1)
    s2 = f.autocorr.get_new_spectra(2)
    s3 = f.autocorr.get_new_spectra(3)
    s = np.vstack((s0,s1,s2,s3))
    return s    

def get_spectrum(f,signal):
    # Return the spectrum for one signal.
    if signal<0 or signal>63:
        print('Signal number',signal,'is not valid.')
        return []
    block = int(signal/16)
    s = f.autocorr.get_new_spectra(block)
    return s[signal - 16*block]

def adc_power(f,sigs):
    d = f.input.get_bit_stats()
    var = d[1][sigs]-d[0][sigs]**2                      #variance
    var = [max(var[i],0) for i in range(len(var))]      #none < 0
    std = np.sqrt(var)                                  #standard deviation
    pd = [var[i]/512/512/100 for i in range(len(var))]  #power in watts
    return (std,pd)

def adcPowerSave(f,snaps,fileprefix=''):
    # f is a list of F-engine;
    # snaps is a list of SNAP2 locations (1:11) corresponding to the F-engines;
    # fileprefix is prefix to filename; 'adcPwr'<date>'.csv' added.
    # Gets adc power measurements (watts) from all 64 signals of each F-engine and
    # stores them in a length-704 array indexed by digital signal number.  Signals
    # not measured are set to -1.  Entire array is written to the specified file, one
    # value per line.
    pd = [adc_power(f[i],range(64))[1] for i in range(len(snaps))]
    powerd = -np.zeros(704)
    for i in range(len(snaps)):
        powerd[64*(snaps[i]-1):(64*(snaps[i]-1)+64)] = pd[i]
    filename = fileprefix + 'adcPwr' + time.strftime('%Y%m%d-%H%M',gmtime()) + '.csv'
    file = open(filename, 'w')
    print(str(list(powerd)).strip('[]'),sep='\n',file=file)

def save_selfcorr(f,filename):
    # Get self-correlations from corr module for all signals and write to .csv file.
    with open(filename,'w') as file:
        for i in range(f.input.n_streams):
            x = f.corr.get_new_corr(i,i)
            print(str(list(np.abs(x))).strip('[]'),sep=',',file=file)

def save_crosscorr(f,signals,filename):
    # Get correlations among the given signals (incl self) and write to file.
    # Save magnitudes only.
    #20211117: revised to record baselines of specified signals rather than all with a
    #given signal. Add header line with signal list.
    with open(filename,'w') as file:
        print('#',str(list(signals)).strip('[]'),sep=',',file=file)
        for i in range(len(signals)):
            for j in range(i,len(signals)):
                x = np.abs(f.corr.get_new_corr(i,j))
                print(str(list(x)).strip('[]'),sep=',',file=file)

def dsig2feng(digitalSignal): # From digital sig num calculate F-unit location and signal
    funit = m.trunc(digitalSignal/64) + 1  # location, 1:11
    fsig = digitalSignal % 64              # FPGA signal number, 0:63
    return (funit,fsig)

def adc_power_dsig(dsig,f,pr=True):
    # Get adc rms count and power for one signal.
    # dsig:  digital signal number
    # f:  list of fengine modules, 0:10.
    # pr:  suppress printing if False.
    loc = dsig2feng(dsig)
    p = adc_power(f[loc[0]-1],[loc[1]])
    rms = p[0][0]
    power = p[1][0]
    if pr:  print(dsig,loc,'rms count=',p[0][0],'P/dBm=',10*np.log10(p[1][0])+30)
    return [dsig, loc[0], loc[1], rms, power]

def get_spectrum_dsig(dsig,f):
    loc = dsig2feng(dsig)
    snapindex = loc[0]-1
    s = [get_spectrum(f[snapindex],loc[1]), f[snapindex].corr.get_new_corr(loc[1],loc[1])]
    return s
    
