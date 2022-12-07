from lwa_f import snap2_feng_etcd_client

# TODO: pass arguments, use click?
ec = snap2_feng_etcd_client.Snap2FengineEtcdControl()
ec.send_command(0, 'controller', 'start_poll_stats_loop', kwargs={'pollsecs':60, 'expiresecs': -1})
