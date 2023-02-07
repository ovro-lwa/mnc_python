
echo 'Restoring ARX and F-Engine settings...'
python scripts/loadFengineAndArxSettings.py data/20230105b-settingsAll.mat

echo 'Restoring X-Engine settings...'
python scripts/lwa_load_delays.py data/cable_delays.csv
