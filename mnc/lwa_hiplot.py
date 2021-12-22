import hiplot
import lwa_antpos

def get_exp(uri):
    df = lwa_antpos.lwa_df.reset_index()
    df.drop(0, inplace=True)  # remove antnum=0
    df.antname = df.antname.apply(lambda x: int(x.split('-')[1]))
    df.rename(columns={'antname': 'antnum'}, inplace=True)
    df = df[['antnum', 'pola_fee', 'polb_fee', 'arx_address', 'pola_arx_channel', 'polb_arx_channel', 'snap2_hostname',
             'pola_digitizer_channel', 'polb_digitizer_channel']]

    return hiplot.Experiment.from_dataframe(df)
