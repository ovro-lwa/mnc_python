#!/bin/bash

# Set OVRO-LWA configuration from settings files
# Casey Law (adapted from Larry D'Addario)

if [ -n "$1" ]
then
    filename=$1
else    
    filename="data/20230105b-settingsAll.mat"
fi

echo 'Restoring ARX and F-Engine settings with' $filename
python scripts/loadFengineAndArxSettings.py $filename

# latest version of Larry's script (23Feb13) does delay setting too
#echo 'Restoring X-Engine settings...'
#python scripts/lwa_load_delays.py data/cable_delays.csv
