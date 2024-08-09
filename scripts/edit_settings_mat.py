import numpy as np
import scipy.io as sio

if __name__ == '__main__':
    INPUT_MAT = '/home/pipeline/opsdata/20240315-settingsAll-day.mat'
    OUTPUT_MAT = '20240315-settingsAll-day.mat'

    config = sio.loadmat(INPUT_MAT, squeeze_me=True)
    
    #### Edit settings here ####
    config['comment'] = 'This file turns off FEE power to both chan of LWA-041,044,120,124,209,235,255,280,310'
    new_off = np.array(
        sorted([158, 159, 160, 161, 416, 417, 448, 449, 602, 603, 618, 619, 66, 67, 82, 83, 34, 35]),
        dtype=config['off'].dtype)
    config['off'] = new_off

    # set AT2=31.5 for arx 16 chan 15 (aka LWA-041A)
    target_arx_chans = [(16, 15)]
    adrs = config['adrs']
    for target_arx, target_chan in target_arx_chans:
        i_arx = np.where(adrs == target_arx)[0]
        config['settings'][i_arx, target_chan-1, 1] = 31.5
    #### End of edit ####

    sio.savemat(OUTPUT_MAT, config)
