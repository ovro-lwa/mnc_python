import numpy as np
import scipy.io as sio

from os import path

if __name__ == '__main__':
    INPUT_MAT = '/home/pipeline/opsdata/20240315-settingsAll-night.mat'
    OUTPUT_MAT = path.basename(INPUT_MAT)

    config = sio.loadmat(INPUT_MAT, squeeze_me=True)
    
    #### Edit settings here ####
    # config['comment'] = 'This file turns off FEE power to both chan of LWA-041,044,235,280'
    config['comment'] = 'This file turns off FEE power to no antennas.'
    new_off = np.array(
        sorted([]),
        dtype=config['off'].dtype)
    config['off'] = new_off

    """
    # set AT2=31.5 for arx 16 chan 15 (aka LWA-041A)
    target_arx_chans = [(16, 15)]
    adrs = config['adrs']
    for target_arx, target_chan in target_arx_chans:
        i_arx = np.where(adrs == target_arx)[0]
        config['settings'][i_arx, target_chan-1, 1] = 31.5
    """
    target_arx_chans = [(16, 15)]
    adrs = config['adrs']
    for target_arx, target_chan in target_arx_chans:
        i_arx = np.where(adrs == target_arx)[0]
        config['settings'][i_arx, target_chan-1, 1] = 8.02284177
    #### End of edit ####

    sio.savemat(OUTPUT_MAT, config)
