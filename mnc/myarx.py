# myarx.py ARX control/monitor using raw()
# 20220112 - add functions 'gainAdjust', 'filterSelect'.
# 20220310 - add rfPowerSave().
# 20221014 - revise status_asig() not to get rfPowerOffset if provided by caller.
# 20221017 - fix raw() function so that return value is never undefined.
# 20221029 - change chanCode() to round rather than truncate attenuator values that are not 0.5dB multiples.

import lwautils.lwa_arx
import sys
import numpy as np
import math as m
import time
import warnings
arx = lwautils.lwa_arx.ARX()

adrs = range(15,46)  #addresses of ARX boards (rollout phase 3)

def raw(adr,cmd):
    r = []
    try:
        r = arx.raw(adr,cmd)
    except:
        pass
    return r

def raw2int(adr,cmd):
    r = arx.raw(adr,cmd)
    #print(ord(r[0]),r[1:])
    d = [int(r[i:i+4],16) for i in range(0,len(r),4)]
    return d

def owte(adr):
    r = arx.raw(adr,'OWTE',1.1)
    d = [int(r[i:i+4],16)/16 for i in range(0,len(r),4)]
    return d
def temp(adr):
    r = arx.raw(adr,'TEMP')
    return int(r[1:5],16)/10

def rfPower(adr,offset=[0]*16): 
    #print('myarx.rfPower.adr,offset:',adr,offset)
    r = arx.raw(adr,'POWA',10.0)
    d = [int(r[i:i+4],16) for i in range(0,len(r),4)]
    d = [d[i]-offset[i] for i in range(len(d))]
    p = []
    for i in range(len(d)):
        watts = (d[i]*.004/2.296)**2/50
        p.append(watts)
    return p

def rfPowerOffset(adr):
    set = arx.raw(adr,'GETA')     #save current configuration
    #print(len(set),set)
    #return
    r = arx.raw(adr,'SETS0001')   #set all channels off and max attenuation
    offsets = raw2int(adr,'POWA') #read detectors
    r = arx.raw(adr,'SETA'+set)   #restore original configuration
    return offsets

def rfPowerSave(adr,offsets,fileprefix=''):
    # adr is a list of addresses;
    # offsets is a list of offset arrays of the same length;
    # fileprefix is prefix to filename; 'arxPwr'<date>'.csv' added.
    # RF power (W) is obtained for all channels of these addresses, then converted
    # to an array of length 720 indexed by analog signal number and written one value
    # per line to the given file.  Values not measured are set to -1.
    pa = [rfPower(adr[i],offsets[i]) for i in range(len(adr))]
    powera = -np.ones(720)
    for i in range(len(adr)):
        powera[16*(adr[i]-1):(16*(adr[i]-1)+16)] = pa[i]
    filename = fileprefix + 'arxPwr' + time.strftime('%Y%m%d-%H%M',gmtime()) + '.csv'
    f = open(filename,'w')
    print(str(list(powera)).strip('[]'),sep='\n',file=f)
    f.close()

def chanCode(at1,at2,filter,feePower):
    """chanCode(): construct hex4 channel control code."""
    # at1 and at2 are the attenuator settings in dB.
    # filter is a string that can either represent a number '0' to '7'
    #  or be one of 'LL','LH','HL','HH' where the first character selects
    #  the low-frequency filter (HPF) and the second selects the high-frequency
    #  filter (LPF).  H and L are the higher and lower of the two available
    #  cutoff frequencies.
    # feePower is boolean, True for on and False for off, controlling front
    #  end electronics.
    f = {'LH':0, 'HH':3, 'LL':4, 'HL':7}
    filterNumber = int(f.get(filter,filter)) & 0xF
    a1 = ~int(round(2*at1)) & 0x3F   # 6b attenuator controls
    a2 = ~int(round(2*at2)) & 0x3F
    x = ((a2 << 9) | (a1 << 3) | filterNumber) & 0x7FFF;
    if feePower:
        x |= 0x8000
    h = '%04X'%(x)
    return(h)

def chanDecode(code):
    """chanDecode(): Decode channel control code either as hex4 or int"""
    if isinstance(code,int):
        c = code
    else:
        c = int(code,16)
    at1 = ((~(c>>3))&0x3F)/2
    at2 = ((~(c>>9))&0x3F)/2
    filt = c & 0x0007
    feePower = bool(c & 0x8000)
    return(at1,at2,filt,feePower)

def presenceCheck(first=1,last=126):
    good = []
    for i in range(first,last+1):
        try:
            r = arx.raw(i,'ARXN',2)  # Long timeout is workaround for API bug
            print('Address',i,'response:',r)
            good.append(i)
        except:
            err = sys.exc_info()
            print('Address',i,'NO response.  ',err[1])
            pass
    return(good)

def feeOff(adr):   # Turn off all front ends of an ARX board.
    c = np.array(raw2int(adr,'GETA')) & 0x7FFF
    for i in range(len(c)):
        code = '{:04X}'.format(c[i])
        arx.raw(adr,'SETC'+'%1X'%(i)+code)
    return(True)

def at2add(adr,dB):  # Add given value to AT2 settings for all channels of ARX board
    c = np.array(raw2int(adr,'GETA')) # Get current settings
    at1 = ((~c & 0x01F8)>>3)/2
    at2 = ((~c & 0x7E00)>>9)/2 + dB
    fil = c & 0x7;
    fee = c & 0x8000;
    #print(at1)
    #print(at2)
    #print(fil)
    #print(fee)
    for i in range(len(c)):
        if at2[i] > 31.5:  at2[i]=31.5
        if at2[i] < 0: at2[i]=0.0
        code = chanCode(at1[i],at2[i],fil[i],fee[i])
        arx.raw(adr,'SETC'+'%1X'%(i)+code)
    return(True)

def arxn(adr):
    s = arx.raw(adr,'ARXN')
    sn = int(s[:4],16)
    sw = s[4:8]
    sensors = int(s[12:14],16)
    chans = [int(s[14+i],16)+1 for i in range(sensors)]
    return (sn,sw,chans)

def gainAdjust(adr,target):  # Set AT2 to achieve specified output power
    # target is desired output power in watts.
    # ARX board settings should already be initialized; only AT2 is changed here.
    # Return final AT2 settings (dB), final powers (dBm), final errors (dB).
    
    offsets = rfPowerOffset(adr) # get detector offsets

    # Do 3 iterations of adjustment
    for i in range(2):
        p = np.array(rfPower(adr,offsets)) # get output power
        e = 10*np.log10(p/target)          # compute error in dB
        at2add(adr,e)                      # make adjustment

    c = np.array(raw2int(adr,'GETA')) # Get current settings
    at2 = ((~c & 0x7E00)>>9)/2             # final AT2 settings
    p = np.array(rfPower(adr,offsets))     # final power
    e = 10*np.log10(p/target)              # final error in dB
    p = 10*np.log10(p)+30                  # final power in dBm
    return(at2,p,e)

def filterSelect(adr,filter): # Change the filter setting for all channels.
    # filter must be a list of length 16.
    c = np.array(raw2int(adr,'GETA')) # Get current settings
    at1 = ((~c & 0x01F8)>>3)/2
    at2 = ((~c & 0x7E00)>>9)/2
    fee = c & 0x8000;
    for i in range(len(c)):
        code = chanCode(at1[i],at2[i],filter[i],fee[i])
        arx.raw(adr,'SETC'+'%1X'%(i)+code)
    return(True)

def asig2arx(analogSignal):   # From analog sig num calculate arx location and channel.
    arx = m.trunc(analogSignal/16) + 1   # location==address, 1:45
    chan = analogSignal % 16 + 1         # channel, 1:16
    return (arx,chan)

def dsig2feng(digitalSignal): # From digital sig num calculate F-unit location and signal
    funit = m.trunc(digitalSignal/64) + 1  # location, 1:11
    fsig = digitalSignal % 64              # FPGA signal number, 0:63
    return (funit,fsig)

def status(adr,offsets=[],pr=True):
    # status(): print status report for all channels of one or more ARX boards
    # pr=False to suppress printing
    # returns:
    # [[analogSigNr, arxAdr, arxCh, AT1,AT2,filter,biasOn, current_mA, rfPower_W],...]
    warnings.simplefilter('ignore') # ignore all warnings (esp. divide by 0)
    stat = []
    if pr: print('sig','adr','ch','config','I/mA','P/dBm')
    if not isinstance(adr,list):
        adr=[adr]
    #print('myarx.status.adr:',adr)
    for i in range(len(adr)):
        a = adr[i]
        if len(offsets)<len(adr): o = rfPowerOffset(a)
        else:                     o = offsets[i]
        #print('myarx.status.o:',o)
        p = rfPower(a,o)
        I = 0.4*np.array(raw2int(a,'CURA'))
        r = raw2int(a,'GETA')
        for i in range(len(r)):
            cfg = chanDecode(r[i])
            if pr: print(16*(a-1)+i,a,i+1,chanDecode(r[i]),format(I[i],'.1f'),format(10*np.log10(p[i])+30,'.2f'))
            stat.append([16*(a-1)+i,cfg[0],cfg[1],cfg[2],cfg[3],I[i],p[i]])

    return np.array(stat)                        

def status_asig(asig,pr=True,offsets=[]):
    # Optional parameters:
    #   offsets:  rfPowerOffsets; if not provided, will be measured here.
    #   pr:  print to stdout if True, otherwise don't print.
    adr = asig2arx(asig)
    a = adr[0]
    c = adr[1]-1
    if len(offsets)<16:
        offsets = rfPowerOffset(a)
    p = rfPower(a,offsets)
    I = 0.4*np.array(raw2int(a,'CURA'))
    r = raw2int(a,'GETA')
    cfg = chanDecode(r[c])
    if pr: print('sig','adr','ch','config','I/mA','P/dBm')
    if pr: print(asig,a,c+1,chanDecode(r[c]),format(I[c],'.1f'),format(10*np.log10(p[c])+30,'.2f'))
    return [asig,cfg[0],cfg[1],cfg[2],cfg[3],I[c],p[c]]

