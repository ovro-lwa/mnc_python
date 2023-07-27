import pandas as pd
import numpy as np
from astropy.time import Time
import warnings
from datetime import timedelta
import subprocess
import argparse

def main(args):
    out_name = args.filename
    sdf_text = ''

    print(" ")
    print("Enter a session ID and session mode with the format NNN MODE")
    acceptable_modes = ['POWER','VOLT','FAST']
    correct = False
    while not correct:
        inp = input()
        out = inp.split(' ')
        if len(out) < 2:
            raise Exception(f"{len(out)} inputs instead of the expected 2 were given")
        sess_id = out[0]
        sess_mode = out[1]
        if sess_mode in acceptable_modes:
            correct = True
        elif sess_mode not in acceptable_modes:
            print(f"The session mode provided is not recognized. Please give one of the following: {acceptable_modes}")
    
    if sess_mode != "FAST":
        print("Provide a beam number:")
        inp = input()
        beam_num = int(inp)

        print(" ")
        print("Provide the path to the calibration directory, or hit enter:")
        cal_dir = input()
        if cal_dir == '':
            cal_dir = None
        
    elif sess_mode == "FAST":
        beam_num = None
        cal_dir = None

    print(" ")
    print("Enter a PI ID, or hit enter")
    inp = input()
    if inp == "":
        print("No PI ID provided. Setting the PI ID to 0")
        pi_id = 0
    elif inp != "":
        pi_id = int(inp)

    print(" ")
    print("Enter PI Name, or hit enter")
    pi_name = input()
    if pi_name == '':
        pi_name = 'Observer'
        print("No PI Name provided. Setting the PI Name to Observer")

    print(" ")
    print("Provide the path to the configuration file, or hit enter:")
    config_dir = input()
    if config_dir == "":
        print("No configuration file specified. Assuming the standard path.")
        config_dir = "/home/pipeline/proj/lwa-shell/mnc_python/config/lwa_calim_config.yaml"

    try:
        session_preamble = make_session_preamble(sess_id, sess_mode, pi_id,pi_name,beam_num,config_dir, cal_dir)
        sdf_text += session_preamble
        print(session_preamble)
    except:
        raise Exception("Couldn't make the session preamble")
    
    ephem_modes =  ['TRK_JOV','TRK_SOL','TRK_LUN'] 
    cont = True
    obs_count = 1
    while cont:
        print(f"Making observation {obs_count}")
        print("Enter the observing mode")
        obs_mode = input()
        if obs_mode == '':
            if sess_mode == 'FAST':
                obs_mode = None
            elif sess_mode != 'FAST':
                obs_mode = 'TRK_RADEC'
                print("No obs mode specified for beamformed observation. Assuming TRK_RADEC")

        print(f"Give the start time of the observation in isot format")
        correct = False
        while not correct:
            start_time = input()
            try:
                t = Time(start_time, format = 'isot')
                correct = True
            except:
                print("Time format not isot. Please enter the observation start time in isot format")
               

        print(f"Give the duration of the observation in milliseconds:")
        dur = int(input())

        if sess_mode == 'POWER' or sess_mode == 'VOLT': 
            print(f"Give the integrations time of the observation in milliseconds")
            int_time = int(input())
        elif sess_mode == 'FAST':
            int_time = None
        
        if obs_mode not in ephem_modes:
            print(f"Give the RA and Dec of the object as RA DEC, in degrees. If doing a fast vis observation or giving an object name to resolve instead, hit enter")
            coords = input()
            if coords != '':
                ra,dec = coords.split()
                ra = float(ra)
                dec = float(dec)
            elif coords == '':
                ra = None
                dec = None
   
        if obs_mode not in ephem_modes:
            print("Give the name of the object, or optionally hit enter if you provided an ra/dec")
        obj_name = input()
        if obj_name == '':
            obj_name = None
    
        if obs_mode in ephem_modes:
            ra = 0.
            dec = 0.
            obj_name = obs_mode.strip('TRK_')

        obs_text = make_obs_block(obs_count, start_time,dur,ra,dec,obj_name,int_time,obs_mode)
        sdf_text += obs_text
        print(obs_text)
        print("Add another observation? (Y/N)")
        inp = input()
        if inp == 'Y' or inp == ' ' or inp == 'y':
            cont = True
            obs_count += 1
        elif inp == 'N' or 'n':
            cont = False
            print(f"Writing out to {args.filename}")
            f = open(args.filename,'w')
            f.write(sdf_text)
            f.close()
    return

def make_session_preamble(session_id,session_mode,pi_id = 0, pi_name:str = 'Observer',beam_num = None,config_dir = '/home/pipeline/proj/lwa-shell/mnc_python/config/lwa_calim_config.yaml',cal_dir = None):
    acceptable_modes = ['POWER','VOLT','FAST']
    assert(session_mode in acceptable_modes), f"session mode not an accepted mode ({acceptable_mode})"
    lines = 'PI_ID            {:02d}\n'.format(pi_id)
    lines += f'PI_NAME          {pi_name}\n\n'
    lines += 'PROJECT_ID       0\n'
    lines += f'SESSION_ID       {session_id}\n'
    lines += f'SESSION_MODE     {session_mode}\n'
    if beam_num != None:
        lines += f'SESSION_DRX_BEAM       {beam_num}\n'
    lines += f'CONFIG_FILE      {config_dir}\n'
    if cal_dir != None:
        lines += f'CAL_DIR          {cal_dir}\n'
    lines += '\n'
    return lines


def make_obs_block(obs_id, start_time:str, duration, ra = None, dec = None, obj_name = None, integration_time = 1, obs_mode = None):
    t = Time(start_time, format = 'isot')
    midnight = Time(int(t.mjd), format = 'mjd')
    mjd_start = int(midnight.value)
    mpm_dt = t - midnight
    mpm = int(mpm_dt.sec * 1e3)
    duration_lf = str(timedelta(milliseconds = duration))
    duration_arr = duration_lf.split(':')
    if len(duration_arr[0]) == 1:
        duration_lf = '0' + duration_lf

    lines =  f'OBS_ID          {obs_id}\n'
    if obj_name != None:
        lines += f'OBS_TARGET      {obj_name}\n'
    lines += f'OBS_START_MJD   {mjd_start}\n'
    lines += f'OBS_START_MPM   {mpm}\n'
    lines += f"OBS_START       UTC {start_time.replace('-',' ').replace('T',' ')}\n"
    lines += f"OBS_DUR         {duration}\n"
    lines += f"OBS_INT_TIME    {integration_time}\n"
    lines += f"OBS_DUR+        {duration_lf}\n"

    if obs_mode != None:
        lines += f"OBS_MODE        {obs_mode}\n"

    if ra is not None:
        lines += f"OBS_RA          %.9f\n" % (ra)
    if dec is not None:
        lines += f"OBS_DEC         %+.9f\n" % (dec)

    lines += "OBS_FREQ1       1161394218\n"
    lines += "OBS_FREQ1+      53.000000009 MHz\n"
    lines += "OBS_FREQ2       1599656187\n"
    lines += "OBS_FREQ2+      73.000000010 MHz\n"
    lines += "OBS_BW          7\n"
    lines += "OBS_BW+         19.600 MHz\n\n"
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    main(args)
