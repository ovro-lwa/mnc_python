# once per snap2. currently snap01
# execute from anywhere (lxdlwacr preferred) using etcd
lwa_snap_feng_init.py -e -s -m -i -p -o ~/proj/caltech-lwa/control_sw/config/lwa_corr_config.yaml snap01

# once per gpu server (1 through 8)
# execute from gpu server
lwa352-start-pipeline.sh 0 1 2 3

# start correlator output
# execute from anywhere (lxdlwacr preferred) using etcd
lwa352_arm_correlator.py
