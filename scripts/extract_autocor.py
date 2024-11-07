#
# This will extract self-correlation spectra for all the antennas for a specified day/hr/min/sec.
# All 16 subbands are used to make the spectra. The spectra will be saved in npy array for
# further processing
#
# The script takes three parameters:
# 1. the full path to the parent directory containing the data, e.g., /lustre/pipeline/night-time/
#    or /lustre/pipeline/slow
# 2. the date of the observations in the format YYYYMMDD, e.g, day=20231214
# 3. the hr:min:sec of the observations in the format HHMMSS, e.g, time=024002
#
# NOTES:
#  - If time="" is not provided, the script will extract all the observations taken on the specified day
#  - If sec is not provided (e.g., time=HHMM), the script will extract all the observations taken in the
#    specified day, hr, and minute
#  - if min:sec are not provided (e.g., time=HH), the script will extract all the observations taken on the
#    specified day and hr


import argparse
import glob
import shutil
import logging, sys
import os

from tqdm import tqdm

import numpy as np

from casacore.tables import table
#from casatasks import mstransform
#logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
#logger = logging.getLogger(__name__)



# function that extract self-correlations from ms file
def extract_selfcorr(path: str, date: str, time: str, step: int, workingdir: str):

    #print(path)
    #print(date)
    #print(time)
    #print(step)
    
    # Find the available bands and prepare the path
    p_band = sorted(glob.glob(path+'*'))
    s_band = [p[-5:] for p in p_band]
    #print(s_band)
    
    # Add date to the path
    s_date = '/'+date[0:4]+'-'+date[4:6]+'-'+date[6:8]
    p_date = [p +s_date for p in p_band]
    #print(p_date)
    
    # Add hour to the path
    hh = time[0:2]   
    if (hh):
        s_hh = hh
        p_hh = np.array([p+'/'+s_hh for p in p_date])
        p_hh = np.expand_dims(p_hh, 1)
    else:
        p_hh = np.array([sorted(glob.glob(p+'/*')) for p in p_date])
        p_hh = np.squeeze(p_hh)
        #p_hh = np.expand_dims(p_hh, 1)
        s_hh = [p[-2:] for p in p_hh[0]]
        
        
    #print('p_hh::',p_hh)
    #print(p_hh.shape)
    #print('p_hh[0]::',p_hh[0])
    #print(p_hh[0].shape)
    #print('s_hh::',s_hh)

    # ------------------------------
    # Extract autocorrlations
    
    workdir = workingdir+date+'/'
    os.makedirs(workdir, exist_ok=True)
   
    
    # we work hour by hour
    for k in np.arange(len(p_hh[0])): # iterate on the hour
        #print('k=',k)
        #print('p_hh[0][k]=',p_hh[0][k])
        print(f'> Extracting autocorrelations for {date}, {p_hh[0][k][-2:]} UT')
        # Find the timestamp of the files
        timestamps=np.array(())
        if len(time)==6:
            timestamps = time
        elif len(time)==0:
            #print('p_hh[3][k]::',p_hh[3][k])
            timestamps = glob.glob(p_hh[3][k]+'/*')
            #print(timestamps)
            timestamps = sorted([row[-24:-9] for row in timestamps])
            timestamps = timestamps[0::step]
            #timestamps=sorted(timestamps)
            #print('timestamps:',timestamps)
        else:
            s_time = p_hh[3][0]+'/'+date+'_'+time+'*'
            #print(s_time)
            timestamps = glob.glob(s_time)
            timestamps = sorted([row[-24:-9] for row in timestamps])
            timestamps = timestamps[0::step]
            #print('timestamps:',timestamps)        

        # We now work on each timestamp
        if (len(timestamps)==0):
            print("> no files found")
        
        for i, timestamp in enumerate(timestamps):        
        
            # Extract autocorrelation for each band
            for j, band in enumerate(s_band):

                #print(p_hh[j][k])
                #print(timestamp)
                ms = p_hh[j][k]+'/'+timestamp+'_'+band+'.ms'

                if (os.path.exists(ms)):
                    
                    #printx("> Working on ",ms)
                    
                    # get frequency information
                    with table(ms+'/SPECTRAL_WINDOW', readonly=True, ack=False) as tb:
                        freq = tb.getcol('CHAN_FREQ')

                    # get antenna names
                    with table(ms+'/ANTENNA', readonly=True, ack=False) as tb:
                        antname = tb.getcol('NAME')
                        nant = len(antname)
                
                    with table(ms, readonly=True, ack=False) as tb:
                        data = tb.getcol('DATA')
                        #print(data.shape)
                        #ant1 = np.squeeze(tb.getcol('ANTENNA1'))
                        #ant2 = np.squeeze(tb.getcol('ANTENNA2'))
                        time2 = tb.getcol('TIME')
                
                    # Extract autocorrelations. To do that I define an array with the
                    # indexes of the diagonal elements which corresponds to autocorrelations 
                    ind = np.zeros((nant), dtype=int)
                    for n in range(1,nant):
                        ind[n] = int(ind[n-1]+nant+1-n)
                        autocor = data[ind][:][:]

                    if j==0:
                       comb_freq = freq
                       comb_time = time2
                       comb_autocor = autocor
                    else:
                       comb_freq = np.concatenate((comb_freq, freq), axis=0)
                       comb_time = np.concatenate((comb_time, time2), axis=0)
                       comb_autocor = np.concatenate((comb_autocor, autocor), axis=1)

                    
                    #print("> Writing autocorrelations in ", workdir+timestamp+".npz")
                    np.savez_compressed(workdir+timestamp+".npz",  antname=antname, time=comb_time, freq=comb_freq.flatten(), autocor=comb_autocor)

                else:
                    print(f"> WARNING: file {ms} does not exist. Continuing anyway.") 
               
                 
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract self-correlations')
    parser.add_argument('-p', '--path', type=str, required=True, help='Absolute path to the parent directory containing the ms files')
    parser.add_argument('-d', '--date', type=str, required=True, help='Date of the observations in the format YYYY-MM-DD')
    parser.add_argument('-t', '--time', type=str, required=False, default='', help='Time of the observations in HHMMSS, or HHMM, or HH. Default=none')
    parser.add_argument('-s', '--step', type=int, required=False, default=60, help='Number of files to use (e.g., step=1 means every file in the specified time range is used, step=6 means that 1 every 6 files is used. step=6 means that autocorrlations are extracted once every minute. Step=360 means that autocorrelations are extracted once every hour. Default=60 (extract autocorrelation every 10 minutes)')
    parser.add_argument('-w', '--workingdir', type=str, required=False, default='/lustre/ai/', help='Path to save the autocorrelation files')
    
    args = parser.parse_args()
    extract_selfcorr(args.path, args.date, args.time, args.step, args.workingdir)

    
