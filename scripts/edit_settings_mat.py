import numpy as np
import scipy.io as sio

if __name__ == '__main__':
    INPUT_MAT = '/home/pipeline/opsdata/20240315-settingsAll-day.mat'
    OUTPUT_MAT = '20240315-settingsAll-day.mat'

    config = sio.loadmat(INPUT_MAT, squeeze_me=True)
    
    #### Edit settings here ####
    config['comment'] = 'This file turns off FEE power to both chan of LWA-041,044,052,076,105,111,120,124,142,151,209,232,255,280,310'
    new_off = np.array(
        sorted([66, 67, 158, 159, 160, 161, 356, 357, 402, 403, 602, 603, 614, 615, 416, 417, 448, 449, 34, 35, 234, 235, 422, 423, 436, 437, 82, 83, 386, 387]),
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