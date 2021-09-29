import time
import json

from mnc.mcs import Client


__all__ = ['discover_recorders', 'Lwa352RecorderControl']


_CLIENT = Client()


def discover_recorders(type='slow', age_limit=120):
    if type == 'slow':
        prefix = '/mon/drvs'
    elif type == 'fast':
        prefix = '/mon/drvf'
    elif type == 'power':
        prefix = '/mon/dr'
    elif type == 'voltage':
        prefix = '/mon/drt'
    else:
        raise ValueError("Unknown recorder type '%s'" % type)
        
    tnow = time.time()
    found = []
    for entry in _CLIENT.client.get_prefix(prefix):
        value, metadata = entry
        metadata = metadata.key.decode()
        if metadata.find('bifrost/max_acquire') == -1:
            continue
        mcs_id = metadata.split('/')[2]
        if type == 'power' and mcs_id[2] in ('v', 't'):
            continue
            
        value = json.loads(value)
        if value['timestamp'] + age_limit < tnow:
            continue
            
        found.append(mcs_id)
        
    return found


class Lwa352RecorderControl(object):
    def __init__(self, type):
        if type not in ('slow', 'fast', 'power', 'voltage'):
            raise ValueError("Unknown recorder type '%s'" % type)
            
        self.type = type
        self.discover()
        
    def discover(self):
        self.ids = discover_recorders(type=self.type)
        print("Found %i %s recorders: %s" % (len(self.ids), self.type, ' '.join(self.ids)))
        
    def print_status(self):
        tnow = time.time()
        for id in self.ids:
            r = _CLIENT.read_monitor_point('bifrost/rx_rate', id=id)
            age = tnow - r.timestamp
            print("%s - %.0f %s as of %.0f s ago" % (id, r.value, r.unit, age))
            
    def start(self, start_mjd='now', start_mpm=0):
        if self.type not in ('slow', 'fast'):
            raise RuntimeError("Only valid for slow and fast recorders")
            
        responses = []
        for id in self.ids:
            responses.append(_CLIENT.send_command(id, 'start', start_mjd=start_mjd, start_mpm=start_mpm))
            if responses[-1][1]['status'] != 'success':
                print("WARNING: failed to command %s" % id)
                
        return responses
        
    def stop(self, stop_mjd='now', stop_mpm=0):
        if self.type not in ('slow', 'fast'):
            raise RuntimeError("Only valid for slow and fast recorders")
            
        responses = []
        for id in self.ids:
            responses.append(_CLIENT.send_command(id, 'stop', stop_mjd=stop_mjd, stop_mpm=stop_mpm))
            if responses[-1][1]['status'] != 'success':
                print("WARNING: failed to command %s" % id)
                
        return responses
        
    def record(self, start_mjd='now', start_mpm=0, duration=60):
        if self.type in ('slow', 'fast'):
            raise RuntimeError("Only valid for power and voltage beam recorders")
            
        responses = []
        for id in self.ids:
            responses.append(_CLIENT.send_command(id, 'record', start_mjd=start_mjd, start_mpm=start_mpm,
                                                                duration_ms=int(duration*1000)))
            if responses[-1][1]['status'] != 'success':
                print("WARNING: failed to command %s" % id)

