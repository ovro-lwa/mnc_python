#!/bin/bash

# Set OVRO-LWA configuration from settings files
# Casey Law (adapted from Larry D'Addario)

if [ -n "$1" ]
then
    filename=$1
else    
    filename="data/20230309-settingsAll-night.mat"
fi

echo 'Restoring ARX and F-Engine settings with' $filename
python scripts/loadFengineAndArxSettings.py $filename

# After 23 Feb13, Larry's script does delay setting too. Alternatively...
#echo 'Restoring X-Engine settings...'
#python scripts/lwa_load_delays.py data/cable_delays.csv
