import os
import glob
import copy
import argparse
import time
import numpy as np
from matplotlib import pyplot as plt, cm
import matplotlib.colors
import seaborn as sns
from tqdm import tqdm
from scipy import signal
from astropy.time import Time
from mnc import anthealth

sns.set_theme()
sns.set_context("poster")


def read_autocor(path: str):
    
    date = path[-9:-1]
    filenames = sorted(glob.glob(path+date+'*'))
    prefix=[x[-19:-4] for x in filenames] 
    #print(prefix)
    
    # Read numpy files and check for missing channels/subbands
    print(f"> Reading autocorrelations from {len(prefix)} files")
    tt=np.zeros(len(prefix))

    nch  = 0
    ch = 0
    for i, f  in enumerate(tqdm(filenames)): 
        data = np.load(f)
        tt[i] = np.array(data['time'])[0]
        antname = np.array(data['antname'])
        freq = np.array(data['freq'].real)
        #print('> Reading number of channels in', f, freq.shape[0])
        if (freq.shape[0]==3072):
            nch=3072
            good_freq=freq
            ch = 1
        elif (freq.shape[0]>nch and ch == 0):
            #print(freq.shape[0], nch, ch)
            nch = freq.shape[0]
            good_freq=freq

    nch = good_freq.shape[0]
    #print("> The maximum number of channels is ", nch)
    if (nch < 3072):
        print(f"> !!!! Warning: nchan = {nch} < 3072.  There are missing channels")

    autocor=np.full((len(prefix), 352, nch, 4), np.nan, dtype=np.float32)
    #print("> ", end="", flush=True)
    for i, f  in enumerate(tqdm(filenames)):
        data = np.load(f)
        freq = np.array(data['freq'].real)
        #print('.', end="", flush=True)
        if (freq.shape[0]==nch):
            autocor[i,:,:,:] = np.array(data['autocor'].real)
        else: 
            print ('> !!!! Warning: Missing channels in file ', f)
            # find existing channels
            chid = np.where(np.in1d(good_freq, freq))[0]
            tmp = np.array(data['autocor'].real)
            for j in range(chid.shape[0]):
                autocor[i,:,chid[j],:] = tmp[:,j,:] 
   # print()
    return tt, good_freq, antname, prefix, autocor

def calc_flatness(spec):
    #flatness = 0
    #normalize before calculating the flatness
    #print('np.nanmedian(spec):',np.nanmedian(spec))
    spec = spec/np.nanmedian(spec)
    flatness = [ (spec[i+1]-spec[i])**2 for i in range(len(spec)-1)]
    return np.sum(flatness)

def find_once_bad(badants):
    once_bad = []
    for row in badants:
        once_bad.extend(row)
    return np.unique(np.array(once_bad))

def remove_rfi(spectrum, size_kernel):
    """
    Removes narrow band spikes from a spectrum.

    Parameters:
    spectrum (numpy.ndarray): The input spectrum.
    threshold (float): The threshold value for spike removal.

    Returns:
    numpy.ndarray: The spectrum with spikes removed.
    """
    """
    # Compute the median of the spectrum.
    median_spectrum = np.median(spectrum)

    # Compute the standard deviation of the spectrum.
    std_spectrum = np.std(spectrum)

    # Compute the threshold value for spike removal.
    threshold_value = median_spectrum + threshold * std_spectrum

    rfi = np.array(np.where(spectrum > threshold_value)[0])
    print(rfi)
    if (len(rfi)>1):
        for i in range(len(rfi)):
            if (rfi[i]==len(spectrum)-1):
                spectrum[rfi[i]]=spectrum[rfi[i-1]]
            elif (rfi[i]==0):
                spectrum[rfi[i]]=spectrum[rfi[i]+1]
            else: 
                spectrum[rfi[i]] = (spectrum[rfi[i]+1]+spectrum[rfi[i]-1])/2.
                
    if (len(rfi)==1):
        if (rfi==len(spectrum)-1):
            spectrum[rfi]=spectrum[rfi-1]
        elif (rfi==0):
            spectrum[rfi]=spectrum[rfi+1]
        else:
            print(spectrum[rfi])
            spectrum[rfi] = (spectrum[rfi+1]+spectrum[rfi-1])/2.
            print(spectrum[rfi])
    """
    
    # Remove the spikes from the spectrum.
    spectrum = signal.medfilt(spectrum, kernel_size=size_kernel)
    #print(spectrum[rfi])
    #power_spectrum = np.abs(spectrum) ** 2
    #power_spectrum[power_spectrum > threshold_value] = median_power_spectrum

    # Return the spectrum with spikes removed.
    return spectrum


def calc_stats(tt, freqs, antname, prefix, autocor, figdir, statdir, minfreq, maxfreq):

    print('> Calculating antenna flags')

    tmpdir = figdir+'tmp/'
    os.makedirs(tmpdir, exist_ok=True)

    ant_id = np.arange(0, len(antname), 1)
     
    # Select the channel range to calculate statistics
    min_i = np.abs(freqs-minfreq).argmin()
    max_i = np.abs(freqs-maxfreq).argmin()
    st_index = np.r_[min_i:max_i]

    # -----------------------------------------------------------------------------------------------------
    # FIND AND REMOVE ANTENNAS WITH LOW FLUX (DEAD SIGNALS)
    
    # calculate the median flux for each time and each antenna over the selected spectral range
    medflux = np.nanmedian(autocor[:,:,st_index,:], axis=2)       
    
    # calculate the median flux for each antenna over all times
    med_medflux = np.nanmedian(medflux, axis=0)
    #print(med_medflux[:,0])
    #print(med_medflux.shape)
    
    # calculate the median flux over all antennas and all times
    global_medflux = np.nanmedian(med_medflux, axis=0)
    #print(global_medflux[0])
    
    # Find antennas with median lower the global median 
    low_threshold = 0.1
    lowflux_xx = np.array(np.where(med_medflux[:,0]/global_medflux[0]<low_threshold)[0])
    lowflux_yy = np.array(np.where(med_medflux[:,3]/global_medflux[3]<low_threshold)[0])
    #print(lowflux_xx)
    #print(lowflux_xx.shape)
    
    # remove bad antennas from further analysis
    nolowflux_xx = np.delete(ant_id, lowflux_xx)
    nolowflux_yy = np.delete(ant_id, lowflux_yy)

    # -------------------------------------------------------------
    # CHECK FLUX TIME VARIABILITY ON THE REMAINING ANTENNAS

    # Calculate the reference median flux at each time by combining all antennas
    ref_medflux_xx = np.nanmedian(medflux[:, nolowflux_xx, 0], axis=1)
    ref_medflux_yy = np.nanmedian(medflux[:, nolowflux_yy, 3], axis=1)
    
    # Find antennas with large temporal variations of the median flux as compared to the global median flux
    flat_ref_medflux_xx = calc_flatness(remove_rfi(ref_medflux_xx, 3))
    flat_ref_medflux_yy = calc_flatness(remove_rfi(ref_medflux_yy, 3))
    
    flat_medflux_xx = np.array([calc_flatness(remove_rfi(medflux[:,nolowflux_xx[i],0],3)) for i in range(len(nolowflux_xx))])
    flat_medflux_yy = np.array([calc_flatness(remove_rfi(medflux[:,nolowflux_yy[i],3],3)) for i in range(len(nolowflux_yy))])

    #[print(nolowflux_xx[i], flat_medflux_xx[i], flat_ref_medflux_xx, flat_medflux_xx[i]/flat_ref_medflux_xx) for i in range(len(nolowflux_xx))]
    
    timevar_threshold = 10.0
    hightimevar_xx = np.array(np.where(flat_medflux_xx/flat_ref_medflux_xx>timevar_threshold)[0])
    hightimevar_yy = np.array(np.where(flat_medflux_yy/flat_ref_medflux_yy>timevar_threshold)[0])

    hightimevar_xx = nolowflux_xx[hightimevar_xx]
    hightimevar_yy = nolowflux_yy[hightimevar_yy]

    highest_timevar_xx = nolowflux_xx[np.argmax(flat_medflux_xx/flat_ref_medflux_xx)]
    highest_timevar_yy = nolowflux_yy[np.argmax(flat_medflux_yy/flat_ref_medflux_yy)]
    
    # -------------------------------------------------------------
    # CHECK SPECTRAL SHAPE OF REMAINING ANTENNAS
    # calculate median spectra for each antenna over the entire time range
    #print(autocor.shape)
    autocor_nolowflux_xx = autocor[:, nolowflux_xx, :, 0]
    autocor_nolowflux_yy = autocor[:, nolowflux_yy, :, 3]
    #print(autocor_nolowflux_xx.shape)
    medspec_xx = np.nanmedian(autocor_nolowflux_xx[:, :, st_index], axis=1)
    medspec_yy = np.nanmedian(autocor_nolowflux_yy[:, :, st_index], axis=1)
    #print(medspec_xx.shape)
    medspec_norfi_xx = np.array([remove_rfi(medspec_xx[i,:], 25) for i in range(len(nolowflux_xx))])
    medspec_norfi_yy = np.array([remove_rfi(medspec_yy[i,:], 25) for i in range(len(nolowflux_yy))])
    #print(medspec_xx.shape)
    
    # Find antennas characterized by a median spectrum that differs from the global median spectrum
    flat_medspec_xx = np.array([calc_flatness(medspec_norfi_xx[i,:]) for i in range(len(nolowflux_xx))])
    flat_medspec_yy = np.array([calc_flatness(medspec_norfi_yy[i,:]) for i in range(len(nolowflux_yy))])
    #print(flat_medspec_xx)

    # calculate a global median spectrum for all antennas over the entire time range
    global_medspec_norfi_xx = np.nanmedian(medspec_norfi_xx[:,:], axis=0)
    global_medspec_norfi_yy = np.nanmedian(medspec_norfi_yy[:,:], axis=0)

    flat_global_medspec_xx = np.array(calc_flatness(global_medspec_norfi_xx))
    flat_global_medspec_yy = np.array(calc_flatness(global_medspec_norfi_yy))
    #print(flat_med_medspec_xx)
    #print(flat_med_medspec_yy)
    
    #[print(i, flat_medspec_xx[i], flat_med_medspec_xx, flat_medspec_xx[i]/flat_med_medspec_xx) for i in range(len(antname))]
    #plt.plot(med_medspec_norfi_xx/np.median(med_medspec_norfi_xx), 'b')
    #plt.plot(medspec_norfi_xx[40]/np.median(medspec_norfi_xx[40]), 'r')
    #plt.plot(medspec_norfi_xx[41]/np.median(medspec_norfi_xx[41]), 'g')    
    #plt.show()
    
    specshape_threshold = 4.5
    badspecshape_xx = np.array(np.where(flat_medspec_xx/flat_global_medspec_xx>specshape_threshold)[0])
    badspecshape_yy = np.array(np.where(flat_medspec_yy/flat_global_medspec_yy>specshape_threshold)[0])

    badspecshape_xx = nolowflux_xx[badspecshape_xx]
    badspecshape_yy = nolowflux_yy[badspecshape_yy]

    # Create flags    
    flag_xx = np.zeros((len(tt), len(antname)))
    flag_yy = np.zeros((len(tt), len(antname)))

    flag_xx[:, lowflux_xx]=1
    flag_yy[:, lowflux_yy]=1
    flag_xx[:, hightimevar_xx]=2
    flag_yy[:, hightimevar_yy]=2
    flag_xx[:, badspecshape_xx]+=3
    flag_yy[:, badspecshape_yy]+=3
    
    # Check if the remaining antennas misbehave at any times by comparing the
    # antenna spectrum to the median antenna spectrum

    #remove antenna that were already flagged
    bad_xx = np.unique(np.concatenate((lowflux_xx, hightimevar_xx, badspecshape_xx),0))
    bad_yy = np.unique(np.concatenate((lowflux_yy, hightimevar_yy, badspecshape_yy),0))
    nobad_xx = np.delete(ant_id, bad_xx)
    nobad_yy = np.delete(ant_id, bad_yy)

    autocor_nobad_xx = autocor[:,nobad_xx,:,0]
    autocor_nobad_yy = autocor[:,nobad_yy,:,3]
    medspec_nobad_xx = np.nanmedian(autocor_nobad_xx[:, :, st_index], axis=1)
    medspec_nobad_yy = np.nanmedian(autocor_nobad_yy[:, :, st_index], axis=1)
    medspec_nobad_norfi_xx = np.array([remove_rfi(medspec_nobad_xx[i,:], 25) for i in range(len(nobad_xx))])
    medspec_nobad_norfi_yy = np.array([remove_rfi(medspec_nobad_yy[i,:], 25) for i in range(len(nobad_yy))])
    
    # Find antennas characterized by a median spectrum that differs from the global median spectrum
    flat_medspec_nobad_norfi_xx = np.array([calc_flatness(medspec_nobad_norfi_xx[i,:]) for i in range(len(nobad_xx))])
    flat_medspec_nobad_norfi_yy = np.array([calc_flatness(medspec_nobad_norfi_yy[i,:]) for i in range(len(nobad_yy))])
    
    for i in tqdm(range(len(tt))):

        # Compare the antenna spectrum to the median antenna spectrum
        # calculate flatness of the spectrum after removing rfi      
        autocor_nobad_norfi_xx = np.array([remove_rfi(autocor_nobad_xx[j, i, st_index], 25) for j in range(len(nobad_xx))])
        flat_spec_nobad_norfi_xx = np.array([calc_flatness(autocor_nobad_norfi_xx[j, :]) for j in range(len(nobad_xx))])
        autocor_nobad_norfi_yy = np.array([remove_rfi(autocor_nobad_yy[j, i, st_index], 25) for j in range(len(nobad_yy))])
        flat_spec_nobad_norfi_yy = np.array([calc_flatness(autocor_nobad_norfi_yy[j,:]) for j in range(len(nobad_yy))])

        flat_ratio_xx = flat_spec_nobad_norfi_xx/flat_medspec_nobad_norfi_xx
        flat_ratio_yy = flat_spec_nobad_norfi_yy/flat_medspec_nobad_norfi_yy

        #[print(i, prefix[i], nobad_xx[j] , flat_spec_norfi_xx[j], flat_medspec_xx[j], flat_ratio_xx[j]) for j in range(len(nobad_xx))]
        threshold = 2
        noflat_xx = np.array(np.where(flat_ratio_xx>threshold)[0])
        noflat_yy = np.array(np.where(flat_ratio_yy>threshold)[0])

        flag_xx[i, nobad_xx[noflat_xx]]=6
        flag_yy[i, nobad_yy[noflat_yy]]=6
    

    sflag_xx = np.empty((len(tt), len(antname)), dtype='U25')
    sflag_yy = np.empty((len(tt), len(antname)), dtype='U25')
    
    sflag_xx[:, nolowflux_xx] = "Good"
    sflag_xx[:, lowflux_xx] = "LowFlux"
    sflag_xx[:, hightimevar_xx] ="HighTimeVar"
    sflag_xx[:, badspecshape_xx] ="BadSpecShape"
    bad2_xx = np.intersect1d(hightimevar_xx, badspecshape_xx)
    sflag_xx[:, bad2_xx] = "HighTimeVar+BadSpecShape"
    
    sflag_yy[:, nolowflux_yy] ="Good"
    sflag_yy[:, lowflux_yy] ="LowFlux"
    sflag_yy[:, hightimevar_yy] ="HighTimeVar"
    sflag_yy[:, badspecshape_yy] ="BadSpecShape"
    bad2_yy = np.intersect1d(hightimevar_yy, badspecshape_yy)
    sflag_yy[:, bad2_yy] = "HighTimeVar+BadSpecShape"

    # Update list of bad antennas
    bad_ants_xx = np.concatenate((antname[lowflux_xx], antname[hightimevar_xx]))
    bad_ants_xx = list(map(str, bad_ants_xx))
    bad_ants_xx = [ant+'A' for ant in bad_ants_xx]

    bad_ants_yy = np.concatenate((antname[lowflux_yy], antname[hightimevar_yy]))
    bad_ants_yy = list(map(str, bad_ants_yy))
    bad_ants_yy = [ant+'B' for ant in bad_ants_yy]

    bad_ants = bad_ants_xx+bad_ants_yy
    bad2 = [sub.replace('LWA', 'LWA-') for sub in bad_ants]
    print(bad2)

    yy = str(prefix[0][0:4])
    mm = str(prefix[0][4:6])
    dd = str(prefix[0][6:8])
    h = str(prefix[0][9:11])
    m = str(prefix[0][11:13])
    s = str(prefix[0][13:15])
    date =str(yy+'-'+mm+'-'+dd+'T'+h+':'+m+':'+s)
    print('> setting badants for ',date)
    mjd = Time(date, format='isot').mjd
    print('> mjd = ', mjd)

    anthealth.set_badants('selfcorr', bad2, time=mjd)

    # MAKE FIGURE
    minutes = (tt-tt[0])/60
    
    # Color for False and True
    cmap = matplotlib.colors.ListedColormap(['green', 'black', 'red', 'yellow', 'blue', 'purple', 'pink'])

    fig = plt.figure(figsize=(13,35))        
    ax  = plt.subplot(111)
    fig.subplots_adjust(bottom=0.18)
    ax.set_title(f'XX pol Flags - {prefix[0][0:8]}')
    ax.set_xlabel("Time (min) (+%s)"%(prefix[0]))
    ax.set_ylabel('Antenna correlator number')
    ax.set_xlim(minutes[0], minutes[-1])
    ax.set_ylim(-0.5,351.5)
    img = ax.imshow(flag_xx.T, interpolation='none', cmap=cmap, extent=(minutes[0], minutes[-1], 351.5, -0.5), aspect='auto')
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.set_yticks(np.arange(0, len(antname), 5, dtype=int))
    ax.set_yticks(np.arange(0, len(antname), 1, dtype=int), minor=True)
    ax.set_xticks(np.arange(0, minutes[-1], 30, dtype=int))
    ax.set_xticks(np.arange(0, minutes[-1], 10, dtype=int), minor=True)

    text = fig.text(0.1, 0.1, 'Flag description:  Black=Low power;\n Red=Large time variability;\n Yellow=Suspicious spectral shape over the entire time interval;\n Purple=Red+Yellow;\n Pink=Variable spectral shape.', horizontalalignment='left', transform=plt.gcf().transFigure)
    
    fig.savefig(figdir+prefix[0][0:8]+'_antenna_status_xx.png', bbox_inches='tight', dpi=800)
    plt.close(fig)
    
    fig = plt.figure(figsize=(13,35))        
    ax  = plt.subplot(111)
    fig.subplots_adjust(bottom=0.18)
    ax.set_title(f'YY pol Flags - {prefix[0][0:8]}')
    ax.set_xlabel("Time (min) (+%s)"%(prefix[0]))
    ax.set_ylabel('Antenna correlator number')
    ax.set_xlim(minutes[0], minutes[-1])
    ax.set_ylim(-0.5,351.5)
    img = ax.imshow(flag_yy.T, interpolation='none', cmap=cmap, extent=(minutes[0], minutes[-1], 351.5, -0.5), aspect='auto')
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.set_yticks(np.arange(0, len(antname), 5, dtype=int))
    ax.set_yticks(np.arange(0, len(antname), 1, dtype=int), minor=True)
    ax.set_xticks(np.arange(0, minutes[-1], 30, dtype=int))
    ax.set_xticks(np.arange(0, minutes[-1], 10, dtype=int), minor=True)
    text = fig.text(0.1, 0.1,
                    'Flag description: Black=Low power;\n Red=Large time variability;\n Yellow=Suspicious spectral shape over the entire time interval;\n Purple=Red+Yellow;\n  Pink=Variable spectral shape.',
                    horizontalalignment='left', transform=plt.gcf().transFigure)
    fig.savefig(figdir+prefix[0][0:8]+'_antenna_status_yy.png', bbox_inches='tight', dpi=800)
    plt.close(fig)
    
    # write statistics on file
    statfile = statdir+str(prefix[0][0:8])+'.txt'
    f = open(statfile, 'w')
    print(f"> Statistics calculated between {prefix[0]} and {prefix[-1]}", file=f)
    print(f"> Number of files analized = {len(tt)}", file=f)
    print(" ---------------------------------------------", file=f)
    print("> Statistics calculated on XX", file=f)
    print(f"> {len(lowflux_xx)} antennas show low flux: ", file=f)
    print("  Correlator number:",lowflux_xx, file=f)
    print("  Ant name:",antname[lowflux_xx], file=f)
    print(f"> {len(badspecshape_xx)} antennas show anomalous spectral shape:", file=f)
    print("  Correlator number:",badspecshape_xx, file=f)
    print("  Ant name:",antname[badspecshape_xx], file=f)
    print(f"> {len(hightimevar_xx)} antennas show excessive temporal variability:", file=f)
    print("  Correlator number:", hightimevar_xx, file=f)
    print("  Ant name:", antname[hightimevar_xx], file=f)
    print(f"  The antenna showing the largest time variability is {antname[highest_timevar_xx]} ({highest_timevar_xx})", file=f)
    print(" ---------------------------------------------", file=f)
    print("> Statistics calculated on YY", file=f)
    print(f"> {len(lowflux_yy)} antennas show low flux: ", file=f)
    print("  Correlator number:",lowflux_yy, file=f)
    print("  Ant name:",antname[lowflux_yy], file=f)
    print(f"> {len(badspecshape_yy)} antennas show anomalous spectral shape:", file=f)
    print("  Correlator number:",badspecshape_yy, file=f)
    print("  Ant name:",antname[badspecshape_yy], file=f)
    print(f"> {len(hightimevar_yy)} antennas show excessive temporal variability: ", file=f)
    print("  Correlator number:", hightimevar_xx, file=f)
    print("  Ant name:", antname[hightimevar_xx], file=f)
    print(f"  The antenna showing the largest time variability is {antname[highest_timevar_yy]} ({highest_timevar_yy})", file=f)
    print(" ---------------------------------------------", file=f)
    print("> XX and YY combined statistis", file=f)

    lowflux = np.unique(np.concatenate((lowflux_xx, lowflux_yy)),0)
    badspecshape = np.unique(np.concatenate((badspecshape_xx, badspecshape_yy),0))
    hightimevar = np.unique(np.concatenate((hightimevar_xx, hightimevar_yy),0))

    print(f"> {len(lowflux)} antennas out of {len(antname)} ({len(lowflux)/len(antname)*100:.1f}%) show low flux: ", file=f)
    print("  Correlator number:",lowflux, file=f)
    print("  Ant name:",antname[lowflux], file=f)
    print(f"> {len(badspecshape)} antennas out of {len(antname)} ({len(badspecshape)/len(antname)*100:.1f}%) show anomalous spectral shape:", file=f)
    print("  Correlator number:",badspecshape, file=f)
    print("  Ant name:",antname[badspecshape], file=f)
    print(f"> {len(hightimevar)} antennas out of {len(antname)} ({len(hightimevar)/len(antname)*100:.1f}%) show excessive temporal variability: ", file=f)
    print("  Correlator number:", hightimevar_xx, file=f)
    print("  Ant name:", antname[hightimevar_xx], file=f)
    print(" ---------------------------------------------", file=f)
    
    badants = np.unique(np.concatenate((lowflux, badspecshape, hightimevar),0))

    print(f"> {len(badants)} antennas out of {len(antname)} ({len(badants)/len(antname)*100:.1f}%) might have ploblems in either XX and YY.", file=f)

    f.close()    
    
    return (sflag_xx, sflag_yy)
    
    """
    

    

        
        
        for j in range(len(nobad)):
            print('Correlator number = ',nobad[j])
            print('Ant name = ', antname[nobad[j]])
            print('Flatness = ',flat_spec_xx[i,j])
            plt.plot(autocor[i,nobad[j],st_index,0], label="orig", alpha=0.6)
            plt.plot(autocor_nobad[i,j,st_index,0], label="nobad", alpha=0.6)
            plt.plot(autocor_norfi_xx[i,j,st_index], label="norfi", alpha=1)
            plt.plot(med_medspec_xx, label="med")
            plt.legend()
            plt.show()
        

    """
         
    

#plot antenna spectra
def plot_spectra(tt, freqs, antname, prefix, autocor, figdir, minfreq, maxfreq):
    print("> Plotting spectra")

    tmpdir = figdir+'tmp/'
    os.makedirs(tmpdir, exist_ok=True)

    min_i = np.abs(freqs-minfreq).argmin()
    max_i = np.abs(freqs-maxfreq).argmin()
    st_index = np.r_[min_i:max_i]

    # calculate the median spectrum for each antenna over the entire time range
    medspec_xx = np.nanmedian(autocor[:,:,st_index,0], axis=0)
    medspec_yy = np.nanmedian(autocor[:,:,st_index,3], axis=0)

    # calculate a global median spectrum for all antennas over the entire time range
    med_medspec_xx = np.nanmedian(medspec_xx[:,:], axis=0)
    med_medspec_yy = np.nanmedian(medspec_yy[:,:], axis=0)
   
    for i in tqdm(range(len(tt))):
        for j in range(len(antname)):
            fig =plt.figure(figsize=(10,10))
            ax = plt.subplot(111)
            ax.set_title('%s - Correlator# %03i - XX, YY'%(antname[j],j))
            ax.plot(freqs[st_index]*1.e-6, autocor[i, j,st_index, 0]/np.median(autocor[i, j,st_index, 0]), 'k', label='XX')
            ax.plot(freqs[st_index]*1.e-6, medspec_xx[j,:]/np.median(medspec_xx[j,:]), 'k', label='med XX', alpha=0.6)
            ax.plot(freqs[st_index]*1.e-6, med_medspec_xx/np.median(med_medspec_xx), 'k', label='global XX', alpha=0.3)

            ax.plot(freqs[st_index]*1.e-6, 1+autocor[i, j,st_index, 3]/np.median(autocor[i, j,st_index, 3]), 'g', label='YY')
            ax.plot(freqs[st_index]*1.e-6, 1+medspec_yy[j,:]/np.median(medspec_yy[j,:]), 'g', label='med YY', alpha=0.6)
            ax.plot(freqs[st_index]*1.e-6, 1+med_medspec_yy/np.median(med_medspec_yy), 'g', label='global YY', alpha=0.3)
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Normalized Flux (Jy) (+1 for YY)')
            #ax.set_ylim(0, 60)
            ax.legend()
            fig.savefig(tmpdir+'/'+prefix[i]+'_ant'+format(j,'03')+'_spectra.jpg')
            plt.close(fig)
        os.system('convert '+tmpdir+'/'+prefix[i]+'_ant*_spectra.jpg '+figdir+prefix[i]+'_spectra.pdf')

def plot_normSpectra(tt, freqs, antname, prefix, autocor, figdir, minfreq, maxfreq, flag_xx, flag_yy):
    print("> Plotting normalized spectra")

    tmpdir = figdir+'tmp/'
    os.makedirs(tmpdir, exist_ok=True)

    min_i = np.abs(freqs-minfreq).argmin()
    max_i = np.abs(freqs-maxfreq).argmin()
    st_index = np.r_[min_i:max_i]

    # calculate the median spectrum for each antenna over the entire time range
    medspec_xx = np.nanmedian(autocor[:,:,st_index,0], axis=0)
    medspec_yy = np.nanmedian(autocor[:,:,st_index,3], axis=0)

    # calculate a global median spectrum for all antennas over the entire time range
    med_medspec_xx = np.nanmedian(medspec_xx[:,:], axis=0)
    med_medspec_yy = np.nanmedian(medspec_yy[:,:], axis=0)

    med_medflux_xx = np.nanmedian(med_medspec_xx)
    med_medflux_yy = np.nanmedian(med_medspec_yy)
    
    for j in tqdm(range(len(antname))):

        medflux_xx = np.median(medspec_xx[j,:])
        medflux_yy = np.median(medspec_yy[j,:])
        
        fig =plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        ax.set_title('%s - Correlator# %03i - XX - FLAGS:%s'%(antname[j],j, flag_xx[0,j]))
        if (medflux_xx>0):
            [ax.plot(freqs[st_index]*1.e-6, autocor[i, j,st_index, 0]/medflux_xx, 'b', label='', alpha=0.2) for i in range(len(tt))]
            ax.plot(freqs[st_index]*1.e-6, medspec_xx[j,:]/medflux_xx, 'r', label=f'median ({medflux_xx:.1f} Jy)')
        ax.plot(freqs[st_index]*1.e-6, med_medspec_xx/med_medflux_xx, 'k', label=f'global ({med_medflux_xx:.1f} Jy)')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Normalized Flux')
        ax.set_ylim(0.75, 1.25)
        ax.legend()
        fig.savefig(tmpdir+'/ant'+format(j,'03')+'_normspectra_xx.jpg')
        plt.close(fig)

        fig =plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        ax.set_title('%s - Correlator# %03i - YY - FLAGS:%s'%(antname[j],j, flag_yy[0,j]))
        if (medflux_yy>0):
            [ax.plot(freqs[st_index]*1.e-6, autocor[i, j,st_index, 3]/medflux_yy, 'b', label='', alpha=0.2) for i in range(len(tt))]
            ax.plot(freqs[st_index]*1.e-6, medspec_yy[j,:]/medflux_yy, 'r', label=f'median ({medflux_yy:.1f} Jy)')
        ax.plot(freqs[st_index]*1.e-6, med_medspec_yy/med_medflux_yy, 'k', label=f'global ({med_medflux_yy:.1f} Jy)')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Normalized Flux')
        ax.set_ylim(0.75, 1.25)
        ax.legend()
        fig.savefig(tmpdir+'/ant'+format(j,'03')+'_normspectra_yy.jpg')
        plt.close(fig)
        
    os.system('convert '+tmpdir+'/ant*_normspectra_xx.jpg '+figdir+prefix[0][0:8]+'_normSpectra_xx.pdf')
    os.system('convert '+tmpdir+'/ant*_normspectra_yy.jpg '+figdir+prefix[0][0:8]+'_normSpectra_yy.pdf')

    
def plot_medFluxVsTime(tt, freqs, antname, prefix, autocor, figdir, minfreq, maxfreq, flag_xx, flag_yy):

    print("> Plotting median flux Vs time")

    tmpdir = figdir+'tmp/'
    os.makedirs(tmpdir, exist_ok=True)

    # create arrays containing time in hr and minutes
    hr = (tt-tt[0])/3600
    minutes = (tt-tt[0])/60

    min_i = np.abs(freqs-minfreq).argmin()
    max_i = np.abs(freqs-maxfreq).argmin()
    st_index = np.r_[min_i:max_i]

    # calculate the median flux for each antenna and each time in the sprectral direction
    medflux_xx = np.nanmedian(autocor[:,:,st_index,0], axis=2)       
    medflux_yy = np.nanmedian(autocor[:,:,st_index,3], axis=2)
    
    # calculte the median flux for each time across all antennas
    med_medflux_xx = np.nanmedian(medflux_xx[:,:], axis=1)
    med_medflux_yy = np.nanmedian(medflux_yy[:,:], axis=1)

    for i in tqdm(range(len(antname))):
        #print("> plotting antenna ",i, antname[i])
    
        fig = plt.figure(figsize=(20,7))        
        ax  = plt.subplot(111)
         
        ax.plot(minutes, medflux_xx[:,i], 'k', label='XX')
        ax.plot(minutes, medflux_yy[:,i], 'g', label='YY')
        ax.plot(minutes, med_medflux_xx, 'k--', label='median XX')
        ax.plot(minutes, med_medflux_yy, 'g--', label='median YY')

        ax.set_ylabel(f"Median Flux between {minf*1.e-6:.1f}-{maxf*1.e-6:.1f} MHz  (Jy)")
        ax.set_xlabel("Time (min) (+%s)"%(prefix[0]))
    
        ax.legend()
        ax.set_title('%s - Correlator# %03i - XX FLAGS:%s - YY FLAGS:%s '%(antname[i], i, flag_xx[0][i], flag_yy[0][i]))
        
        ax.set_xlim(minutes[0], minutes[-1])
        
        xt = np.linspace(0, minutes[-1], 20, endpoint=True) # one tick every 10 min
        
        ax.set_xticks(xt)
        #ax.set_ylim([0,40])
        
        ax.grid(True, color='k', axis='both', which='major', alpha=0.25)

        fig.savefig(tmpdir+'/ID'+format(i,'03')+'-'+str(antname[i])+'medFluxVsTime.png', bbox_inches='tight')
        plt.close(fig)
        
    print("> Creating pdf file with median fluxes Vs time")
    os.system('convert '+tmpdir+'/*medFluxVsTime.png '+figdir+prefix[0][0:8]+'_medFluxVsTime.pdf')

def plot_dynamicSpectra(tt, antname, prefix, autocor, figdir, flag_xx, flag_yy):

    print("> Plotting dynamic spectra") 

    tmpdir = figdir+'tmp/'
    os.makedirs(tmpdir, exist_ok=True)
    
    cmap = copy.copy(cm.get_cmap('viridis'))
    cmap.set_under('r')
    for i in tqdm(range(len(antname))):
        
        #print("> plotting antenna ",i, antname[i])
        
        fig, ax = plt.subplots(1, figsize=(20,7))    

        ax.set_ylabel("Channel")
        ax.set_xlabel("Time (hr) (+%s)"%(prefix[0]))
        masked = np.ma.masked_where(autocor[:,i,:,0]<0, autocor[:,i,:,0])
        #print(masked.dtype)
        #test = np.random.rand(masked.shape[0], masked.shape[1])
        #print(test.dtype)
        #img = ax.imshow(test.T, interpolation='none', vmin=1.e-15, vmax=40, origin='lower', \
        # aspect='auto', extent=(0, (tt[-1]-tt[0])/3600, -0.5, 3071+0.5), cmap=cmap)
        img = ax.imshow(masked.T, interpolation='none', vmin=1.e-15, vmax=30, origin='lower', \
                        aspect='auto', extent=(0, (tt[-1]-tt[0])/3600, -0.5, 3071+0.5), cmap=cmap)
    
        ax.set_title('%s - Correlator# %03i - XX - FLAGS: %s'%(antname[i], i, flag_xx[0][i]))
        hr = (tt-tt[0])/3600
        ax.set_xlim(hr[0], hr[-1])
        nticks = int(len(hr)/20)
        ax.set_xticks(hr[0::nticks])

        #secax = ax.secondary_xaxis('top', 
        for j in range(16):
            ax.axhline(y=j*192, color='r', linestyle=':')

        cbar = fig.colorbar(img)
        cbar.ax.set_ylabel('XX Amp (Jy)')
        fig.savefig(tmpdir+'/ID'+format(i,'03')+'-'+str(antname[i])+'_XX_dynspec.png', bbox_inches='tight')
        plt.close(fig)

        # YY pol
        fig, ax = plt.subplots(1, figsize=(20,7))    
        ax.set_ylabel("Channel")
        ax.set_xlabel("Time (hr) (+%s)"%(prefix[0]))
        masked = np.ma.masked_where(autocor[:,i,:,3]<0, autocor[:,i,:,3])
        img = ax.imshow(masked.T, interpolation='none', vmin=1.e-15, vmax=30, origin='lower', \
                        aspect='auto', extent=(0, (tt[-1]-tt[0])/3600, -0.5, 3071+0.5), cmap=cmap)
        #ax.set_ylim(freq[0]*1.e-6, freq[-1]*1.e-6)
        ax.set_title('%s - Correlator# %03i - YY - FLAGS: %s'%(antname[i], i, flag_yy[0][i]))

        ax.set_xlim(hr[0], hr[-1])
        ax.set_xticks(hr[0::nticks])
        for j in range(16):
            ax.axhline(y=j*192, color='r', linestyle=':')

        cbar = fig.colorbar(img)
        cbar.ax.set_ylabel('YY Amp (Jy)')
        fig.savefig(tmpdir+'/ID'+format(i,'03')+'-'+str(antname[i])+'_YY_dynspec.png', bbox_inches='tight')
        plt.close(fig)

    print("> Creating pdf file with dynamic spectra")
    os.system('convert '+tmpdir+'/*_dynspec.png '+figdir+prefix[0][0:8]+'_dynSpec.pdf')
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Plot self-correlations')
    parser.add_argument('-p', '--path', type=str, required=True, help='Absolute path to the directory containing the autocorrelations')
    parser.add_argument('-ds', '--dynSpec', type=bool, required=False, default=False, help='Plot dynamic spectra')
    parser.add_argument('-ft', '--fluxvstime', type=bool, required=False, default=False, help='Plot flux vs time')
    parser.add_argument('-ns', '--normSpec', type=bool, required=False, default=False, help='Plot normalized spectra')
    
    args = parser.parse_args()
    
    tt, freqs, antname, prefix, autocor = read_autocor(args.path)
    
    figdir = args.path+'fig/'
    os.makedirs(figdir, exist_ok=True)

    statdir = args.path+'stats/'
    os.makedirs(statdir, exist_ok=True)

    minf = 30e6
    maxf = 80e6

    sflag_xx, sflag_yy = calc_stats(tt, freqs, antname, prefix, autocor, figdir, statdir, minf, maxf)

    if (args.dynSpec):
        plot_dynamicSpectra(tt, antname, prefix, autocor, figdir, sflag_xx, sflag_yy)

    if (args.fluxvstime):
        plot_medFluxVsTime(tt, freqs, antname, prefix, autocor, figdir, minf, maxf, sflag_xx, sflag_yy)

    if (args.normSpec):
        plot_normSpectra(tt, freqs, antname, prefix, autocor, figdir, minf, maxf, sflag_xx, sflag_yy)

    #plot_spectra(tt, freqs, antname, prefix, autocor, figdir, minf, maxf)

