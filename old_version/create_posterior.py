import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.special import gammaincc
from scipy.integrate import dblquad
import pandas as pd 
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap

def weibull(x, k, r):
    # weibull distribution
    first = k/x
    second = (x*r*gamma(1 + k**(-1)))**k
    third = np.exp(-(x*r*gamma(1+k**(-1)))**(k))

    return first*second*third

def weibull_ccdf(x, k, r):
    # weibull complementary cumulative distribution function (1-cdf)
    return np.exp(-(x*r*gamma(1 + k**(-1)))**k)

def Nt_given_params(N, obs, times, k, r):
    # likelihood of observing N bursts at times t1--tN given k and r
    # obs: array like of observation lengths with bursts
    # N: array like of number of bursts in each observation
    # times: list of array like corresponding times of bursts in each observation

    for i, ob_len in enumerate(obs):
        time_array = times[i]
        Ni = N[i]
        term = r*weibull_ccdf(time_array[0], k, r)*weibull_ccdf(ob_len - time_array[-1], k, r)
        if Ni==1:
            return term
        else:
            for j, time in enumerate(time_array):

                if j<=len(time_array) - 2:
                    term *= weibull(time_array[j+1] - time_array[j], k, r)

            return term

def prior(k, r, dist='jeff'):
    # prior distributions
    if dist=='jeff':
        return (k**(-1))*(r**(-1))
    elif dist=='uni':
        return 1

def p_zerobursts(k, r, obsnb):
    # probability of observing zero bursts
    # needs obsnb array which is array of scan times with no bursts
    def g(k, r, obs):
        g = gamma(k**(-1))*gammaincc(k**(-1), (obs*r*gamma(1 + k**(-1)))**(k))/(k*gamma(1 + k**(-1))) 
        return g

    for i, val in enumerate(tqdm(obsnb)):

        if i==0:
            tot = g(k, r, val)
        else:
            tot*=g(k, r, val)

    return tot

def posterior(k, r, obsnb, N, obsb, times, dist='jeff'):
    if len(obsnb)==0:
        return Nt_given_params(N, obsb, times, k, r)*prior(k, r, dist=dist)
    else:

        return Nt_given_params(N, obsb, times, k, r)*p_zerobursts(k, r, obsnb)*prior(k, r, dist=dist)



def read_df(infile):
    # load dataframe
    df = pd.read_pickle(infile)

    # get Wb L-band observations while looking at SGR 1935
    # L-band observations had 128 MHz bandwith, and P band had 64 MHz bandiwdth
    df_wb_l = df[(df["Station"] == "wb") & (df["Source"] == "SGR1935") & (df["Bandwidth"] == 128.0)]

    #get dataframe corresponding to scans with no bursts
    df_wb_l = df_wb_l[(df_wb_l["Code"]!='sgrl21') | (df_wb_l["Scan"]!="no0006")]

    #get onsala L-band scans
    df_o_l = df[(df["Station"] == "o8") & (df["Source"] == "SGR1935")]

    # get observation length of non-overlapping scans at Onsala
    obs_len_o = []
    for row in np.array(df_o_l[['MJD_start','MJD_end', 'Obslen_sec']]):
        mjd_start = row[0]
        mjd_end = row[1]
        obs_len = row[2]
        df_overlap = df_wb_l[((df_wb_l['MJD_start']>= mjd_start) & (df_wb_l['MJD_start']<= mjd_end))
                            | ((df_wb_l['MJD_start']<=mjd_start) & (df_wb_l['MJD_end']>=mjd_start))]
        if df_overlap.empty:
             obs_len_o.append(obs_len)

    # get observation length of non-overlapping scans at Wb
    obs_len_wb = []
    for row in np.array(df_wb_l[['MJD_start','MJD_end', 'Obslen_sec']]):
        mjd_start = row[0]
        mjd_end = row[1]
        obs_len = row[2]
        df_overlap = df_o_l[((df_o_l['MJD_start']>= mjd_start) & (df_o_l['MJD_start']<= mjd_end))
                            | ((df_o_l['MJD_start']<=mjd_start) & (df_o_l['MJD_end']>=mjd_start))]
        if df_overlap.empty:
             obs_len_wb.append(obs_len)

    obs_len_o = np.array(obs_len_o)
    obs_len_wb = np.array(obs_len_wb)



    return np.concatenate((obs_len_o, obs_len_wb))


if __name__=="__main__":
    # obs length with bursts (scan wbno0006, obsid sgrl21) in days
    # if there are multiple scans with bursts, put each scan length 
    # as one entry in the list.
    obs = [595.996736/86400]

    # time of burst arrivals
    # each entry corresponds to a list of arrival times (from beginning
    # of observation) of bursts
    times = [[246.059/86400, 247.455/86400]]

    # num of bursts
    # each entry corresponds to number of bursts in corresponding
    # scan.
    N = [2]

    # grid to evaluate posterior on
    kstart = 0.01
    kstop = 2
    rstart = -5 #-7
    rstop = 5 #-1

    # create k, r arrays
    k = np.linspace(kstart, kstop, 1000)
    r = np.logspace(rstart, rstop, 1000)

    # want to evaluate in the middle of each bin
    k_cent = 0.5*(k[1:] + k[:-1])
    r_cent = 10**(0.5*(np.log10(r[1:]) + np.log10(r[:-1])))

    # create meshgrid on centered values
    km, rm = np.meshgrid(k_cent, r_cent)

    # read in dataframe of observations.
    # returns an array of scan lengths with
    # no bursts

    obslen_nb = read_df('campaign_new.pkl')/86400 


    # calculate posterior ~ likelihood(scan with bursts)*likelihood(scans with no bursts)*prior 
    post = posterior(km, rm, obslen_nb, N, obs, times, dist='uni')


    k_delta = k[1:] - k[:-1]
    r_delta = r[1:] - r[:-1]

    #create meshgrid of bin widths
    kd_mesh, rd_mesh = np.meshgrid(k_delta, r_delta)


    # use bin width meshgrid to normalize posterior
    tot = np.sum(post*kd_mesh*rd_mesh)
    post /= tot

    # save normalized posterior as pickled data
    np.save("wb_onsala_post", [post, k, r], allow_pickle=True)

