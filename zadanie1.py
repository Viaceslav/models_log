import pandas as pd
import numpy as np
import random as rd
import datetime as dt


from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib.pyplot as plt


def agregate_pp_rate(df):

    ntd = pd.DataFrame(df)

    ntd['Prep amount'] = ntd['Outstanding_notional'] * ntd['PP_rate (%)']
    # ntd['Prep amount_2'] = ntd['Starting_notional'] * ntd['PP_rate (%)']


    grp_df = ntd.groupby(['Date'])['Prep amount', 'Outstanding_notional'].sum()

    grp_df = pd.DataFrame(grp_df.to_records('rows'))
    grp_df['ag_mon_pp_rate1'] =  ntd['Prep amount'] / ntd['Outstanding_notional']
    # grp_df['ag_mon_pp_rate2'] =  ntd['Prep amount_2'] / ntd['Outstanding_notional']

    date  = grp_df['Date']
    m_a_p_rate = grp_df['ag_mon_pp_rate1']

    return date, m_a_p_rate





if __name__ == '__main__':
    df_m = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/mortgage_data.xlsx', sheet_name='Mortgage data' )
    df_r = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/refinancing_rate_data.xlsx', sheet_name='Refinancing rate data' )

    a = agregate_pp_rate(df_m)

    fig, ax = plt.subplots()

    Date = a[0]
    rate1 = a[1]

    plt.plot(Date, rate1, color = 'blue')
    ax.legend()
    plt.xlabel('Czas')
    plt.ylabel('Aggreate monthly prepayment rate (%)')
    plt.title('Aggreate monthly prepayment rate (%)')

    plt.show()


