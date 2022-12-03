from mnc import control
from lwa_antpos import mapping

con = control.Controller("/home/ubuntu/proj/lwa-shell/mnc_python/config/lwa_config_calim.yaml")

# find bad antennas from threshodl on arx channel power
adrs = con.conf['arx']['adrs']
ma = lwa_arx.ARX()
adr = adrs[0]
chpow = ma.get_all_chan_power(adr)
# threshold then map chpow to antenna

snap2nums = [name.lstrip('snap') for name in con.conf['fengines']['snap2s_inuse']]
num = snap2nums[0]
mp = ls.get_dict(f'/mon/snap/{num}')
print(mp['time'], mp['stats']['pfb']['overflow_count'], mp['stats']['fpga']['programmed'], mp['stats']['fpga']['sys_mon'],
      mp['stats']['sync']['period_fpga_clks'], mp['stats']['eth']['tx_full'],a mp['stats']['eq']['clip_count'])
# threshold then map to bad antenna groups on a snap

statsdict = mp['stats']['input']
#threshold on 'powernn', 'meannn', 'rmsnn'
# map to bad antenna

# reserve /mon/antvisflag/n for flags identified from visibilities
# etcd value: {'time': nnn, 'flagged': True/False}

# temperature?

# store antenna flag info at /mon/anthealth/n
# idea for etcd value: {'time': nnn, 'flagged': True/False, 'arxflag': True/False, 'snapflag', 'overflowflag', 'visflag' etc...}
# "flagged" is result of logical and of other fields
