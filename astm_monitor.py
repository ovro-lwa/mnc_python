import requests
from dsautils import dsa_store
from time import sleep
from astropy import time
from dateutil.parser import parse

store = dsa_store.DsaStore('/home/claw/code/dsa110-pyutils/dsautils/conf/etcdConfig_lwa.yml')
# should connect to 'etcdv3service:2379' via conf file

# Setup by sshing to astm nodes, then
# > glances -w
# TODO: get eth0 ip for each node and put in locs dict

locs = {'astm13': '0.0.0.0:61208', 'astm12': '169.254.50.162:61208'}

def run(wait=1):
    """ Run query loop
    """

    route = '/api/3/'  
    while True:
        t0 = time.Time.now().unix
        for node, ipp in locs.items():
            nodenum = node.lstrip('astm')
            r = requests.get(f'http://{ipp}{route}now')
            if r.status_code == 200:
                dt = parse(r.json())
                mjd = time.Time(dt).mjd
                print(f'checking node {node} at MJD {mjd}')
            else:
                print(f'Cannot connect to glances server at {ipp}')
                continue

            for item in ['cpu', 'mem']:
                r = requests.get(f'http://{ipp}{route}{item}')
                if r.status_code == 200:
                    dd = r.json()
                    dd['node_num'] = nodenum
                    dd['time'] = mjd
                    store.put_dict(f'/mon/astm/{nodenum}/{item}', dd)
                    print(f'Pushing item {item} for node {nodenum}')
                else:
                    print('uh oh')

        dt = time.Time.now().unix - t0

        if dt < wait:
            sleep(wait - dt)
