import pandas as pd
import numpy as np
import random as rd
import datetime as dt
import sys, os
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression as logk

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib.pyplot as plt

df_m = pd.read_excel(r'/Users/viaceslavpavlovskij/zadanie ING/mortgage_data.xlsx', sheet_name='Mortgage data')
df_r = pd.read_excel(r'/Users/viaceslavpavlovskij/zadanie ING/refinancing_rate_data.xlsx', sheet_name= 'Refinancing rate data')

def agregate_pp_rate(df):

    ntd = pd.DataFrame(df)

    ntd['Prep amount'] = ntd['Outstanding_notional'] * ntd['PP_rate (%)']
    ntd['Prep amount_2'] = ntd['Starting_notional'] * ntd['PP_rate (%)']


    grp_df = ntd.groupby(['Date'])['Prep amount','Prep amount_2', 'Starting_notional', 'Outstanding_notional'].sum()

    grp_df = pd.DataFrame(grp_df.to_records('rows'))
    grp_df['ag_mon_pp_rate1'] =  ntd['Prep amount'] / ntd['Outstanding_notional']
    grp_df['ag_mon_pp_rate2'] =  ntd['Prep amount_2'] / ntd['Outstanding_notional']

    date  = grp_df['Date']
    rate = grp_df['ag_mon_pp_rate1']
    rate2 = grp_df['ag_mon_pp_rate2']



    return date, rate, rate2




def ref_rate_cal(df_ref):

    dfr = pd.DataFrame(df_ref)

    ref_rate = dfr['Refinancing rate']


    return ref_rate


def differ_rates(df1, df2):
    # trace1 = pd.DataFrame(df1)
    trace2 = pd.DataFrame ( df2 )
    df_tmp = pd.DataFrame()
    df_tmp['Date'] = trace2['Date']
    df_tmp.set_index('Date')
    df_new = pd.DataFrame()

    colName = ['dif_rt_cl_ {}'.format(i) for i in range(0, 25)]

    for col, i in zip(colName, range(0, 25)):
        trace1 = pd.DataFrame(df1)
        df_new = trace1[trace1['Client_ID'] == i]
        df_new.reset_index(inplace=True)
        df_new.drop('index', axis= 1, inplace=True)
        df_new['roznica'] = np.subtract(df_new['Client_rate (%)'], trace2['Refinancing rate'])
        df_tmp[col] = df_new['roznica']

    return df_tmp

def model_simoid(clasifier, y):

    return 1/(1 + np.exp(-(clasifier.intercept_ + clasifier.coef_ *x)))


if __name__ == '__main__':
    df_m = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/mortgage_data.xlsx', sheet_name='Mortgage data' )
    df_r = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/refinancing_rate_data.xlsx', sheet_name='Refinancing rate data' )



    ind_rate = pd.DataFrame(differ_rates(df_m,df_r), columns= [x for x in differ_rates(df_m,df_r)])

    # c = c.pivot_table(values= ['diff_rate cl n 0', 'diff_rate cl n 1'], index = 'Date', aggfunc=np.sum )
    # print(c)
    ind_rate.plot(x = 'Date', y = [x for x in ind_rate.drop('Date', axis= 1)], kind = 'line')
    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('Aggreate monthly prepayment rate (%)')
    plt.title('Aggreate monthly prepayment rate (%)')


    plt.show()


#
