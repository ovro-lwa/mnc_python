from influxdb import DataFrameClient
from astropy.time import Time
influx = DataFrameClient('influxdbservice.sas.pvt', 8086, 'root', 'root', 'dsa110')

MS_PER_SECOND = 1000

def get_rfpower(utctime, dt):
    """ Given utctime (e.g., '2023-07-25T10:00:00') and dt in seconds, return ARX RF power monitor points.
    Returns time, ant_num, power_a, and power_b as a DataFrame.
    """
    
    tu = int(MS_PER_SECOND*Time(utctime, format='isot').unix)
    query = f'SELECT time, ant_num, "power_a" as pa, "power_b" as pb FROM "antmon" WHERE time >= {tu}ms and time < {tu+dt*MS_PER_SECOND}ms'
    result = influx.query(query)['antmon']

    return result


def get_rain(utctime, dt):
    """ Given utctime (e.g., '2023-07-25T10:00:00') and dt in seconds, return rain monitor points from wxmon.
    Returns time, rainrate, and rainhr as a DataFrame.
    """
    
    tu = int(MS_PER_SECOND*Time(utctime, format='isot').unix)
    query = f'SELECT time, rainrate, rainhr FROM "wxmon" WHERE time >= {tu}ms and time < {tu+dt*MS_PER_SECOND}ms'
    result = influx.query(query)['wxmon']

    return result