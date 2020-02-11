import pandas as pd
import numpy as np
import random as rd
import datetime as dt


from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib.pyplot as plt

df_m = pd.read_excel(r'/Users/viaceslavpavlovskij/zadanie ING/mortgage_data.xlsx', sheet_name='Mortgage data')
df_r = pd.read_excel(r'/Users/viaceslavpavlovskij/zadanie ING/refinancing_rate_data.xlsx', sheet_name= 'Refinancing rate data')

def agregate_pp_rate(df):

    ntd = pd.DataFrame(df)

    ntd['Prep amount'] = ntd['Outstanding_notional'] * ntd['PP_rate (%)']


    grp_df = ntd.groupby(['Date'])['Prep amount', 'Outstanding_notional'].sum()

    grp_df = pd.DataFrame(grp_df.to_records('rows'))
    grp_df['ag_mon_pp_rate1'] =  ntd['Prep amount'] / ntd['Outstanding_notional']

    date  = grp_df['Date']
    m_a_p_rate = grp_df['ag_mon_pp_rate1']

    return date, m_a_p_rate




def ref_rate_cal(df_ref):

    dfr = pd.DataFrame(df_ref)

    ref_rate = dfr['Refinancing rate']


    return ref_rate


if __name__ == '__main__':
    df_m = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/mortgage_data.xlsx', sheet_name='Mortgage data' )
    df_r = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/refinancing_rate_data.xlsx', sheet_name='Refinancing rate data' )

    a = agregate_pp_rate(df_m)
    b = ref_rate_cal(df_r)


    plt.figure()

    plt.subplot(2, 1, 1)

    x = a[0]
    rate1 = a[1]

    plt.plot(x, rate1, color = 'blue')
    plt.xlabel('Czas')
    plt.ylabel('Aggreate monthly prepayment rate (%)')
    plt.title('Aggreate monthly prepayment rate (%)')
    plt.legend()


    plt.subplot(2,1,2)
    z = a[0]
    ref_rt = b

    plt.plot(z, ref_rt, color = 'green')
    plt.xlabel('Czas')
    plt.ylabel('Refinancing rate (%)')
    plt.title('Refinancing rate (%)')
    plt.legend()

    plt.show()

#
