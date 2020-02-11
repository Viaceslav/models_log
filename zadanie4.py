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


def linear_model(x, m, b):
    return m*x + b

def logarytm_model(x, a, c, d):

    return a +(1/(1 +((np.exp(-(0.4* c + d*x))))))



if __name__ == '__main__':
    df_m = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/mortgage_data.xlsx', sheet_name='Mortgage data' )
    df_r = pd.read_excel ('/Users/viaceslavpavlovskij/zadanie ING/refinancing_rate_data.xlsx', sheet_name='Refinancing rate data' )

    a = agregate_pp_rate(df_m)
    b = ref_rate_cal(df_r)


    plt.figure()


    time = a[0]
    cl_rt= a[1]
    ref_rt = b
    data_sy = np.random.randn ( 36 )
    uncer = pd.DataFrame ( {'s_pos' : [ round ( x, 2 ) for x in data_sy ]} )





    #
    # plt.subplot(2,1,1)
    # plt.scatter(x = cl_rt, y =ref_rt, color = 'red')
    # plt.legend()
    #
    # plt.subplot(2,1,2)
    # plt.plot(time,  cl_rt, color = 'blue')
    # plt.plot(time, ref_rt, color = 'green')
    # plt.legend()



    df_sim = pd.DataFrame()

    df_sim['Time'] = np.linspace(0,10, 36)
    df_sim['client rate'] = cl_rt
    df_sim['ref rt'] = ref_rt
    df_sim['s_pos'] = uncer

    # plot data
    plt.errorbar( df_sim [ 'Time' ], df_sim [ 'client rate' ], df_sim[ 's_pos' ], fmt='b.')
    plt.hlines(0, df_sim['Time'].min(), df_sim['Time'].max())
    plt.ylabel ( " rate" )
    plt.xlabel ( "czas" )




    licz_pocz = [1, 1]
    fit = curve_fit(linear_model, df_sim['ref rt'], df_sim['client rate'] , sigma= uncer['s_pos'], p0= licz_pocz, absolute_sigma= True )

    ans, cov = fit
    fit_m, fit_b = ans

    fit_sm, fit_sb = np.sqrt(np.diag(cov))

    # print the fit results:
    print ( "m: %.2f +/- %.2f" % (fit_m, fit_sm) )
    print ( "b: %.2f +/- %.2f" % (fit_b, fit_sb) )

    # plot the data and fit results
    plt.errorbar ( df_sim[ 'Time' ], df_sim [ 'client rate' ], df_sim [ 's_pos' ], fmt='r.', label="data" )
    plt.ylabel ( 'Odchylenie stopy' )
    plt.xlabel ( "Czas" )
    print ( "covariance:" )
    print ( cov )

    t = np.linspace ( 0, 10 )
    plt.plot ( t, linear_model( t, fit_m, fit_b ), label="model" )
    plt.legend ()

    # compute chi-square
    chisq = sum ( (df_sim [ 'client rate' ] - linear_model ( df_sim ['Time' ], fit_m, fit_b )) ** 2 / df_sim [ 's_pos' ] ** 2 )
    plt.figtext ( 0.5, 0.2, "chi-square linear model: %.2f" % chisq, fontweight="bold" )

    # make a residuals plot
    plt.figure ()
    plt.errorbar ( df_sim [ 'Time' ], df_sim [ 'client rate' ] - linear_model ( df_sim [ 'Time' ], fit_m, fit_b ), df_sim[ 's_pos' ], fmt='b.' )
    plt.hlines ( 0, df_sim [ 'Time' ].min (), df_sim [ 'Time' ].max () )
    plt.ylabel ( "reszty modelu liniowego" )
    plt.xlabel ( "czas" )








#   #########  logistic regression  ##############

    df_sim = pd.DataFrame()

    df_sim['Time'] = np.linspace(0,10, 36)
    df_sim['client rate'] = cl_rt
    df_sim['ref rt'] = ref_rt
    df_sim['s_pos'] = uncer

    # plot data
    plt.errorbar( df_sim [ 'Time' ], df_sim [ 'client rate' ], df_sim[ 's_pos' ], fmt='b.')
    plt.hlines(0, df_sim['Time'].min(), df_sim['Time'].max())
    plt.ylabel ( "stopa" )
    plt.xlabel ( "czas" )




    licz_pocz = [1, 1, 1]
    fit = curve_fit(logarytm_model, df_sim['ref rt'], df_sim['client rate'] , sigma= uncer['s_pos'], p0= licz_pocz, absolute_sigma= True )

    ans, cov = fit
    fit_y0, fit_x0, fit_a = ans
    fit_sy0, fit_sx0, fit_sa = np.sqrt(np.diag(cov))

    # print the fit results
    print ( "y0: %.2f +/- %.2f" % (fit_y0, fit_sy0) )
    print ( "v0: %.2f +/- %.2f" % (fit_x0, fit_sx0) )
    print ( "a: %.2f +/- %.2f" % (fit_a, fit_sa) )
    print ( "kowariancja:" )
    print ( cov )

    # wykres i wyniki dopasowania
    plt.errorbar ( df_sim[ 'Time' ], df_sim [ 'client rate' ], df_sim [ 's_pos' ], fmt='r.', label="dane" )
    plt.ylabel ( 'Odchylenie stopy' )
    plt.xlabel ( "Czas" )
    print ( "kowariancja:" )
    print ( cov )

    t = np.linspace ( 0, 10 )
    plt.plot ( t, linear_model( t, fit_m, fit_b ), label="model" )
    plt.legend ()

    # compute chi-square
    chisq = sum ( (df_sim [ 'client rate' ] - logarytm_model ( df_sim ['Time' ],fit_y0,fit_x0,fit_a )) ** 2 / df_sim [ 's_pos' ] ** 2 )
    plt.figtext ( 0.5, 0.2, "chi-square log - model: %.2f" % chisq, fontweight="bold" )

    # make a residuals plot
    plt.figure ()
    plt.errorbar ( df_sim [ 'Time' ], df_sim [ 'client rate' ] - logarytm_model ( df_sim [ 'Time' ], fit_y0,fit_x0,fit_a ), df_sim[ 's_pos' ], fmt='b.' )
    plt.hlines ( 0, df_sim [ 'Time' ].min (), df_sim [ 'Time' ].max () )
    plt.ylabel ( "reszty (obserwacje - estymacja)" )
    plt.xlabel ( "Czas x" )
    plt.show()

# print(df_m['PP_rate (%)'])


#
# plt.show()
#
