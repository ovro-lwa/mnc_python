# Script to load ARX settings to boards from a .mat file.
#   USAGE:  python arxSettingsLoad <filename>

import mnc.myarx as a
import scipy.io as mat
import time
import sys
import numpy as np

if len(sys.argv) < 2:
    print(' USAGE:  python arxSettingsLoad.py <filename>')
    sys.exit()

#adrs = a.presenceCheck(10,45)
#

data = mat.loadmat(sys.argv[1],squeeze_me=True)
adrs = data['adrs']
settings = data['settings']
print('ARX: ',adrs)
print('Settings array: ',np.shape(settings))
print('Saved at: ',time.asctime(time.gmtime(data['time'])))

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
    
