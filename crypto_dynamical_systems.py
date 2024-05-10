"""
Module calibrates the three momentum trader and three value traders to 
historical data. 

Module then provides forecast of calibrated system into the future. 

To use the module:
    1. Scroll to main() at the bottom.
    2. Replace start period and end period with different dates. 
    3. Type python crypto_dynamical_systems.py
    4. Look for forecast called cyrpto_dynamical_system_forecast.png in folder. 


"""


from multiprocessing import Value
from this import d
from tkinter import CURRENT
from turtle import clear
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import optimize
from numpy import linalg as LA
import itertools
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Lasso
import time 
import requests
import numpy as np
import scipy.fft
import asyncio
import websockets
import json
import unittest

def form_dynamical_matrix(n_v, n_m, alpha, beta):

    assert(len(alpha) == n_m)
    assert(len(beta) == n_m)

    n = n_v + n_m

    m = [[0.0 for i in range(3 * n)] for j in range(3 * n)]
    m = np.array(m)

    r = 0 
    for i in range(n_m):
        for j in range(3*n):
            if j < n:
                if i == j:
                    id_val = 1.0 + (alpha[i] + beta[i]) / n                    
                    m[r][j] = id_val
                else:
                    m_val = (alpha[i] + beta[i]) / n                    
                    m[r][j] = m_val
            elif j >= n and j < 2 * n:               
                val = -1 * (alpha[i] + 2 * beta[i]) / n                
                m[r][j] = val
            else:
                val = beta[i] / n                
                m[r][j] = val
        r += 1

    for i in range(n_v):
        m[r][n_m + i] = 1 
        r += 1

    for i in range(n):
        m[r][i] = 1
        r += 1

    for i in range(n):
        m[r][n + i] = 1 
        r += 1


    return m


def bounded_non_linear(x, v_min, v_max, n_m):

    for i in range(n_m):
        x[i, 0] = min(max(x[i,0], v_min[i]), v_max[i])
        
    return x


def calculate_volume(x, average_volume, k=0.3):
        
    n = len(x)
    m = int(n / 3)

    if len(x.shape) == 1: 
        x = x[:,np.newaxis]

    v = x[:m, :]
    v_mean = sum(v) / m
    aad = sum([abs(v_i - v_mean) for v_i in v])
    alpha = 2 / (k * v_mean) * average_volume
    s = 0.5 * alpha * aad

    return s 


def f(A, x, v_min, v_max, n_m, average_volume=None, bounded_f = False):
    
    if bounded_f: 
        x = np.matmul(A, x)
        x = bounded_non_linear(x, v_min, v_max, n_m)
        if average_volume is not None:
            v = calculate_volume(x, average_volume, k=0.3)
        else:
            v = None        

        return x, v
    else:
        x = np.matmul(A, x)
        return x


def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


def iterate_map(k=None,
               non_linear_bounded=False,
               actual_prices=None,
               actual_volume=None, 
               average_volume=None, 
               file_name_override=None,
               forecast_plotting=False,
               only_minimize_price_error=True, 
               no_plotting=False):

    if k is None:
        n_v = 3
        n_m = 4
        n = n_v + n_m
        N = 3 * n

        alpha = [3.0, 1.1, 1.3, 1.4]
        beta = [1.4, 1.3, 1.2, 1.1]
        v_min = [50, 20, 30, 20]    
        v_max = [100, 120, 110, 200]
        x_0 = np.array([[np.random.randint(low=40,high=60) for i in range(N)]]).T
        k_steps = 100

    else:
        n_v = 3
        n_m = 3
        n = n_v + n_m
        N = 3 * n

        alpha = k['alpha']
        beta = k['beta']
        v_min = k['v_min']   
        v_max = k['v_max']
        x_0 = k['x_0']
        

        if actual_prices is None:         
            k_steps = 10
        else:
            k_steps = len(actual_prices)

    if x_0.ndim == 1:
        x_0 = x_0[:, np.newaxis]

    if average_volume is None:
        average_volume = sum(actual_volume) / len(actual_volume)
    else:
        pass

    A = form_dynamical_matrix(n_v, n_m, alpha, beta)
    x = x_0
    x_t = np.array([[0.0 for i in range(k_steps)] for j in range(N)])
    p_t = np.array([0.0 for i in range(k_steps)])
    s_t = np.array([0.0 for i in range(k_steps)])
    v1_t = np.array([0.0 for i in range(k_steps)])
    v2_t = np.array([0.0 for i in range(k_steps)])

    for i in range(k_steps):                

        p = sum(x[0:n, 0]) / n
        x, s = f(A, x, v_min, v_max, n_m, average_volume=average_volume, bounded_f = True)
        x_t[:, i] = x[:, 0]
        p_t[i] = p
        s_t[i] = s
        v1 = x[0,0]
        v1_t[i] = v1
        v2 = x[1,0]
        v2_t[i] = v2    

    x_T = x


    now = datetime.now()
    ts = now.strftime("%Y%m%d%H%M%S")

    if not no_plotting:
        if not forecast_plotting: 
            plt.clf()
        plt.plot([i for i in range(len(p_t))], p_t)
        if not forecast_plotting:
            plt.plot(v1_t)
            plt.plot(v2_t)
        if actual_prices is not None:
            plt.plot(actual_prices, 'r+')
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Dynamics with {0} Momentum Trader and {1} Value Trader".format(n_m, n_v))
        if file_name_override is None:
            plt.savefig("crypto_dynamical_system_{0}.png".format(ts))
        else:
            plt.savefig(file_name_override)

        if not forecast_plotting:
            plt.clf()
            plt.plot([i for i in range(len(s_t))], s_t)
            if actual_volume is not None:
                plt.plot(actual_volume, 'r+')

                plt.xlabel("Time")
                plt.ylabel("Volume")
                plt.title("Dynamics with {0} Momentum Trader and {1} Value Trader For Volume".format(n_m, n_v))

            if file_name_override is None:
                plt.savefig("crypto_dynamical_system_for_volume_{0}.png".format(ts))
            else:
                plt.savefig(file_name_override)



    if actual_prices is not None and actual_volume is not None:                
        price_error=MAPE(p_t, actual_prices)
        volume_error=MAPE(s_t, actual_volume)

    elif actual_prices is not None and actual_volume is None:
        #TODO: Consider adding weighting to better calibrate system 
        #to actual prices. 
        print("Calculating price error ... \n")
        weighting = [np.exp(-1 * ((len(p_t) - 1 - i) / (len(p_t) - 1))) for i in range(len(p_t))]
        price_error = np.linalg.norm(np.multiply(p_t - actual_prices, weighting), ord=1)
    else:
        price_error = None


    if only_minimize_price_error:
        error = price_error
    else:
        #Here we look to fit both volume and price,
        #in a certain porportion.         
        error = 0.7 * price_error + 0.3 * volume_error

    return error, x_T, p_t


def transform_x_to_parameters_dictionary(x, prices=None):

    n_v = 3
    n_m = 3
    
    n = n_v + n_m
    N = 3 * n

    k = {}
    k['alpha'] = x[:3]
    k['beta'] = x[3:6]
    k['v_min'] = x[6:9] 
    k['v_max'] = x[9:12]
    k['x_0'] = x[12:30]

    if prices is not None:
        k_steps = len(prices)
        k['k_steps'] = k_steps

    return k


def optimization_function(x, *args):    

    prices = args[0]
    volume = args[1]
    no_plotting = args[2]
    k = transform_x_to_parameters_dictionary(x, prices)
    error, _, _ = iterate_map(k,
                          non_linear_bounded=True, 
                          actual_prices=prices,
                          actual_volume=volume, 
                          only_minimize_price_error=True,
                          no_plotting=no_plotting)
    return error


def load_csv_path():
    cwd = "C:\\Users\\nr282\\OneDrive\\Desktop\\historical_prices"
    filename = "eth_prices.csv"
    csv_path = os.path.join(cwd, filename)
    return csv_path


def load_option_price_csv_path():
    cwd = "C:\\Users\\nr282\\OneDrive\\Desktop\\historical_prices"
    filename = "eth_option_prices.csv"
    csv_path = os.path.join(cwd, filename)
    return csv_path


def get_search_grid(ranges):

    h = 10 
    grid = []
    for i in range(len(ranges)):
        grid.append(list(np.linspace(ranges[i][0], ranges[i][1], num=h)))

    return grid



def brute_force(ranges, prices):

    print("Brute Force ... \n")
    grid = get_search_grid(ranges)
    
    e_min = None
    x_min = None
    c = 0 
    for x_str in itertools.product(*grid):        
        print("Testing candidate number {0} \n".format(c))
        x = np.array(list(x_str))        
        e = optimization_function(x, prices)
        if e_min is None:
            e_min = e 
            x_min = x
        else:
            if e < e_min:
                e_min = e
                x_min = x
            else:
                pass
        c += 1
    return x_min, e_min 

def get_crypto_currency_df(ticker, start_date="2020-01-01", end_date="2023-06-27"):

    r = requests.get(f"https://api.polygon.io/v2/aggs/ticker/X:{ticker}USD/range/1/day/{start_date}/{end_date}?apiKey=5KaYTqFoTFjUIjtv1SUUxP_2TTaAJp2j")
    r_json = r.json()
    
    ds = r_json["results"]
    df = pd.DataFrame(ds)
    number_of_seconds_in_milliseconod = 1000
    df['t'] = df['t'].apply(lambda t: t / number_of_seconds_in_milliseconod)
    df["Date"]  = pd.to_datetime(df['t'], unit='s')
    df["Price"] = df["o"]
    df["Volume"] = df["v"]
    return df

def get_crypto_currency_df(ticker, start_date="2020-01-01", end_date="2023-06-27"):

    r = requests.get(f"https://api.polygon.io/v2/aggs/ticker/X:{ticker}USD/range/1/day/{start_date}/{end_date}?apiKey=5KaYTqFoTFjUIjtv1SUUxP_2TTaAJp2j")
    r_json = r.json()
    
    ds = r_json["results"]
    df = pd.DataFrame(ds)
    number_of_seconds_in_milliseconod = 1000
    df['t'] = df['t'].apply(lambda t: t / number_of_seconds_in_milliseconod)
    df["Date"]  = pd.to_datetime(df['t'], unit='s')
    df["Price"] = df["o"]
    df["Volume"] = df["v"]
    return df

def get_crypto_currency_option_df(ticker):

    #TODO: need to work on getting crypto currency option df
    #aligned properly.

    import asyncio
    import websockets
    import json

    msg = \
        {
          "jsonrpc" : "2.0",
          "id" : 9344,
          "method" : "private/get_order_history_by_instrument",
          "params" : {
            "currency" : ticker,
            "kind" : "option"
          }
        }

    async def call_api(msg):
       async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
           await websocket.send(msg)
           while websocket.open:
               print("Processing web socket ... \n")
               response = await websocket.recv()               
               break           
       return response
    
    result = asyncio.get_event_loop().run_until_complete(call_api(json.dumps(msg)))
    res = json.loads(result)
    df = pd.DataFrame.from_dict(res)

    keys = df["result"][0].keys()
    for k in keys:        
        df[k]= df["result"].apply(lambda x: x[k])


    return df

def download_historical_price_df():


    print("Attempting to download spot historical dataframe \n")
    ticker="ETH"
    spot_df = get_crypto_currency_df(ticker)
    csv_spot_path = load_csv_path()
    spot_df.to_csv(csv_spot_path)
    print("Successfully downloaded spot historical dataframe ... \n")

    print("Attempting to download option historical dataframe \n")
    ticker="ETH"
    csv_option_path = load_option_price_csv_path()
    option_df = get_crypto_currency_option_df(ticker)
    option_df.to_csv(csv_option_path)
    print("Successfully downloaded option historical dataframe ... \n")

    return spot_df, option_df


def get_prices(start_date = "2023-01-01", end_date = "2023-02-01"):

    csv_path = load_csv_path()
    df = pd.read_csv(csv_path)
    df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
    df = df.sort_values(by=["Date"])
    actual_prices = df["Price"].tolist()
    return actual_prices

def get_volume(start_date = "2023-01-01", end_date = "2023-02-01"):

    csv_path = load_csv_path()
    df = pd.read_csv(csv_path)
    df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
    df = df.sort_values(by=["Date"])
    actual_volume = df["Volume"].tolist()
    return actual_volume


def get_option_prices(start_date="2023-01-01", end_date = "2023-02-01"):
    
    csv_option_path = load_option_price_csv_path()
    df = pd.read_csv(csv_option_path)
    df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
    df = df.sort_values(by=["Date"])
    


def create_regression(df, lower_price_bound = None, upper_price_bound = None, weighted_regression=True):

    n = len(df)
    x = []
    y = []
    p = 0 
    for t in range(n): 

        if (t > 1) and (t + 1 < len(df)):

            p_t_minus_two = df['o'][t-2]
            p_t_minus_one = df['o'][t-1]
            p_t = df['o'][t]
            p_t_plus_one = df['o'][t+1]

            if lower_price_bound is not None:
                if (p_t_minus_two > lower_price_bound 
                    and p_t_minus_one > lower_price_bound
                    and p_t > lower_price_bound
                    and p_t_plus_one > lower_price_bound):
                    pass
                else:
                    continue

            if upper_price_bound is not None:
                if (p_t_minus_two < upper_price_bound 
                    and p_t_minus_one < upper_price_bound
                    and p_t < upper_price_bound
                    and p_t_plus_one < upper_price_bound):
                    pass
                else:
                    continue
            
            y_t = p_t_plus_one - p_t
            x_1_t = p_t - p_t_minus_one
            x_2_t = p_t - 2 * p_t_minus_one + p_t_minus_two


            x.append([x_1_t, x_2_t])
            y.append(y_t)
            p += 1


    x = np.array(x)

    reg = Lasso().fit(x, y)
    score = reg.score(x, y)
    coefficients = reg.coef_

    x_0 = x[:,0]
    x_0 = x_0[:, np.newaxis]
    reg = Lasso().fit(x_0, y)
    score = reg.score(x_0, y)
    c = reg.coef_
    alpha = c[0]

    

    x_1 = x[:,1]
    x_1 = x_1[:, np.newaxis]
    reg = Lasso().fit(x_1, y)
    score = reg.score(x_1, y)
    c = reg.coef_
    beta = c[0]


    plt.scatter(list(x[:,0]), y)
    plt.xlabel("p(t) - p(t-1)")
    plt.ylabel("p(t+1) - p(t)")
    plt.savefig("alpha_regression.png")
    plt.clf()
    plt.scatter(list(x[:,1]), y)
    plt.xlabel("p(t) - 2 * p(t-1) + p(t-2)")
    plt.ylabel("p(t+1) - p(t)")
    plt.savefig("beta_regression.png")
    plt.clf()
    
    return score, coefficients, p, alpha, beta       


def develop_ranges_for_calibration(prices, n_v, n_m, use_default=True):

    if use_default:
        ranges = [[2,2.25], 
              [2,2.25],
              [2,2.25],
              [1,1.1],
              [1,1.1],
              [1,1.1],
              [1000,1500],
              [750,1250],
              [500,1000],
              [1750,2000],
              [2000,2250],
              [2250,2500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500],
              [850,1500]]
    else:
        print("Developing ranges ... \n")

        p_max = max(prices)
        p_min = min(prices)
        s = pd.Series(prices, name="o")
        df = pd.DataFrame(data=s, columns=["o"])
        score, coeff, p, _, _  = create_regression(df, lower_price_bound = p_min, upper_price_bound = p_max)
        
        #Override with regression coeffcients only if model is good, 
        #otherwise set so model parameters are intereptable. 

        m = n_v + n_m
        alpha = (coeff[0] * m) / n_m
        beta = (coeff[1] * m )/ n_m

        print("In developing ranges, the score for the regression of alpha {0} and beta {1} is {2} \n".format(alpha, beta, score))
        if score > 0.5: 
            pass
        else:
            alpha = 2.0
            beta = 1
        

        p_0 = prices[0]

        alpha_min = alpha - 0.25
        alpha_max = alpha + 0.25

        beta_min = beta - 0.5
        beta_max = beta + 0.5
           
        V_min_min = 0.5 * p_min
        V_min_max = 1.5 * p_min

        V_max_min = 0.5 * p_max
        V_max_max = 1.5 * p_max

        V_0_min = 0.8 * p_0
        V_0_max = 1.2 * p_0

        ranges = [[alpha_min, alpha_max], 
              [alpha_min, alpha_max],
              [alpha_min, alpha_max],
              [beta_min, beta_max],
              [beta_min, beta_max],
              [beta_min, beta_max],
              [V_min_min,V_min_max],                  #Vmin bounds
              [V_min_min,V_min_max],
              [V_min_min,V_min_max],
              [V_max_min,V_max_max],                  #Vmax bounds
              [V_max_min,V_max_max],
              [V_max_min,V_max_max],
              [V_0_min,V_0_max],                      #V_0 bounds
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max],
              [V_0_min,V_0_max]]

    return ranges


def calibrate_dynamical_system(use_custom_search=False,
                              start_date="2023-01-01", 
                              end_date="2023-02-01",
                              no_plotting=False):

    prices = get_prices(start_date = start_date, end_date = end_date)
    volume = get_volume(start_date = start_date, end_date = end_date)
        
    n_v = 3
    n_m = 3

    ranges = develop_ranges_for_calibration(prices, n_v, n_m, use_default=False)
   
    if use_custom_search:
        x, f = brute_force(ranges, prices)
    else:           
         
         opt = optimize.differential_evolution(optimization_function, ranges, args=(prices, volume, no_plotting), maxiter=1000)
         x = opt['x']
         f = opt['fun']

    return x 

def investigate_dynamical_system():

    alpha = [2, 2]
    beta = [1.5, 1.5]
    n_v = 2
    n_m = 2
    A = form_dynamical_matrix(n_v, n_m, alpha, beta)
    w, v = LA.eig(A)
    n = np.linalg.norm(A)


def get_perturbed_k(k, pct=0.005):

    k_peturbed = {}
    for kv in k:
        v = k[kv]
        k_peturbed[kv] = np.random.normal(v, pct * abs(v))
    return k_peturbed

def calculate_statistical_probabilities(k, no_plotting=False):

        number_of_iterates = 1000
        k_steps = 10
        probability_samples = np.array([[0 for i in range(k_steps)] for j in range(number_of_iterates)])
        
        for it in range(number_of_iterates):
            k_perturbed = get_perturbed_k(k)
            error,_,p_t = iterate_map(k=k_perturbed, 
                                 non_linear_bounded=True,
                                 actual_prices=None,
                                 actual_volume=None, 
                                 average_volume=100,
                                 file_name_override="cyrpto_dynamical_system_forecast_1000.png",
                                 forecast_plotting=True,
                                 no_plotting=no_plotting)
            p_t = np.array(p_t)
            probability_samples[it,:] = p_t

        mean_t = np.mean(probability_samples, axis=0)
        var_t = np.var(probability_samples, axis=0)

        return mean_t, var_t, probability_samples

def calibrate_and_forecast(calibrate_start_period = None, calibrate_end_period = None, forecast_number_of_days = None, no_plotting=False):
        
    print("Calibrate Dynamical System to historical prices ... \n")
    x = calibrate_dynamical_system(use_custom_search=False,
                                  start_date = calibrate_start_period,
                                  end_date = calibrate_end_period,
                                  no_plotting=no_plotting)
    k = transform_x_to_parameters_dictionary(x)
    actual_prices = get_prices(start_date = calibrate_start_period, end_date = calibrate_end_period)
    actual_volume = get_volume(start_date = calibrate_start_period, end_date = calibrate_end_period)
    error, x_T, _ = iterate_map(k=k, non_linear_bounded=True, actual_prices=actual_prices, no_plotting=no_plotting, actual_volume=actual_volume)

    average_volume = sum(actual_volume) / len(actual_volume)

    k['x_0'] = x_T

    #We use calibrated parameters to forecast into the future, 
    #and we can measure against the forecasted period actual 
    #prices, which yields the error.

    print("Calibrated parameters are provided by: \n {0} \n".format(k))

        
    #Forecasted actual prices, may not exist, so we set to zero.    
    forecasted_period_prices = [0 for i in range(forecast_number_of_days)]
    print("Forward Simulating Prices for {0} number of days ... \n".format(forecast_number_of_days))

    plt.clf()
    import copy
    k_actual = copy.copy(k)
    number_iterates = 10
    for it in range(number_iterates):
        print("Iterate number is \n {0} \n".format(it))
        k_perturbed = get_perturbed_k(k)
        error,_,_ = iterate_map(k=k_perturbed, 
                             non_linear_bounded=True,
                             actual_prices=None,
                             actual_volume=None, 
                             average_volume=average_volume,
                             file_name_override="cyrpto_dynamical_system_forecast.png",
                             forecast_plotting=True,
                             no_plotting=no_plotting)


    plt.clf()
    mean_t, var_t, prob_t = calculate_statistical_probabilities(k_actual, no_plotting=no_plotting)

    print("Mean of crypto prices \n {0} \n".format(mean_t))
    print("Variance of crypto prices \n {0} \n".format(var_t))

    return mean_t, var_t, prob_t



def conduct_fourier_analysis():    

    ticker="ETH"
    prices_df = get_crypto_currency_df(ticker, start_date="2023-01-01", end_date="2023-07-02")
    prices = prices_df["Price"]
    prices_df["pct_change"] = prices_df["Price"].pct_change()
    prices_pct_change_np = prices_df["pct_change"].dropna().array
    x = [i for i in range(len(prices))]
    plt.plot(x, prices)
    plt.show()
    plt.clf()

    num_days = 90
    y = prices
    y_np = np.array(y)
    f_analysis_complex = scipy.fft.fft(y_np)
    f_analysis_magnitude = np.abs(f_analysis_complex)
    x = [i for i in range(num_days)]
    y = list(f_analysis_magnitude[:num_days])
    y_mean = sum(y) / len(y)
    y_var = sum([((x - y_mean) ** 2) for x in y]) / len(y)
    y_std = y_var ** 0.5

    print("Standard Deviation is: {0} \n".format(y_std))

    plt.yscale("log")
    plt.scatter(x,y)
    plt.axvline(x=7)
    plt.axvline(x=14)
    plt.axvline(x=21)
    plt.axvline(x=30)
    plt.axvline(x=90)
    plt.axhline(y=y_mean)
    plt.xlabel("Day Frequency")
    plt.ylabel("Magnitude Of Frequency")
    plt.savefig("Crypto_Frequency.png")
    plt.show()
    

def conduct_eth_bitcoin_correlation_analysis():


    ticker="ETH"
    eth_prices_df = get_crypto_currency_df(ticker, start_date="2023-01-01", end_date="2023-07-02")


    ticker="BTC"
    btc_prices_df = get_crypto_currency_df(ticker, start_date="2023-01-01", end_date="2023-07-02")

    df = eth_prices_df.merge(btc_prices_df, how='inner', on="Date", suffixes=('_eth', '_btc'))
    

    df["pct_change_eth"] = df["Price_eth"].pct_change()
    df["pct_change_btc"] = df["Price_btc"].pct_change()

    x = df["pct_change_eth"].dropna()
    y = df["pct_change_btc"].dropna()

    x = np.array(x)
    y = np.array(y)

    x = x[:,np.newaxis]
    y = y[:,np.newaxis]


    s = scipy.stats.pearsonr(x[:,0], y[:,0])    

    print("Pearson Correlation Coefficient: \n {0} \n".format(s))

    x_2 = np.square(x)
    x = np.concatenate((x, x_2), axis=1)

    reg = LinearRegression().fit(x,y)
    score = reg.score(x, y)
    coefficients = reg.coef_

    plt.scatter(df["pct_change_eth"], df["pct_change_btc"])
    plt.xlabel("pct_change_eth")
    plt.ylabel("pct_change_btc")
    plt.title("ETH/BTC Price Correlation, Score: {0}".format(score))
    plt.show()

class Portfolio(object):

    def __init__(self, starting_cash):
        print("Setting up portfolio \n")
        self.starting_cash = starting_cash
        self.current_cash = starting_cash
        self.current_eth_shares = 0
        self.current_spot_price = None
    

    def strategy_1(self, price, mean_t, var_t, prob_t):
        print("Executing strategy 1 ... \n")
        variance_difference = [(var_t[i+1]-var_t[i],i) for i in range(len(var_t)-1)]
        variance_reduction = list(filter(lambda x: x[0] < -1 * 0.05 * np.abs(np.mean(var_t)), variance_difference))
        

        if len(variance_reduction) > 0:
            expected_future_price = mean_t[variance_reduction[-1][1]]
            variance_reduced = True
        else:
            variance_reduced = False

        required_increase = 1.01
        if variance_reduced: 
            if expected_future_price > required_increase * price:
                if self.current_cash > 0:   
                    cash_deployed = (0.2 * self.current_cash)
                    self.current_eth_shares += cash_deployed / price
                    self.current_cash -= cash_deployed
            elif expected_future_price < required_increase * price:
                if self.current_eth_shares > 0:
                    shares_sold = self.current_eth_shares 
                    self.current_cash += shares_sold * price 
                    self.current_eth_shares = 0
            else:
                pass

    def strategy_2(self, price, mean_t, var_t, prob_t):
        print("Executing strategy 2 ... \n")

        #Convert all cash to shares. 
        self.current_eth_shares += self.current_cash / price
        self.current_cash = 0

        if mean_t[0] > price:
            pass
        else:
            #Sell 10% of all shares
            shares_to_sell = int(0.1 * self.current_eth_shares)
            self.current_cash += price * shares_to_sell
            self.current_eth_shares -= shares_to_sell


    def execute_trading_algorithms(self, current_date, mean_t, var_t, prob_t):

        print("Executing trading algorithms for current date of {0} \n".format(current_date))
        print("Mean Estimates for Prices \n {0} \n".format(mean_t))
        print("Variance Estimates for Prices \n {0} \n".format(var_t))

        current_date = datetime.strptime(current_date, "%Y-%m-%d")
        trading_date_minus_one = current_date - timedelta(days=1)
        
        current_date_str = current_date.strftime("%Y-%m-%d")
        trading_date_minus_one_str = trading_date_minus_one.strftime("%Y-%m-%d")

        #Get spot prices 
        prices = get_prices(start_date = trading_date_minus_one_str, end_date = current_date_str)
        if len(prices) > 1: 
            print("Warning...there are multiple prices \n")
        else:
            price = prices[0]

        #Below is the first strategy 
        #strategy_1(self, price, mean_t, var_t, prob_t)

        #Below is the second strategy
        self.strategy_2(price, mean_t, var_t, prob_t)

        self.current_spot_price = price
        

        

    def get_current_cash_position(self):
        return self.current_cash

    def get_current_share_position(self):
        return self.current_eth_shares

    def get_current_spot_position(self):
        if self.current_spot_price is None:
            return 0 
        return self.current_spot_price
        

def run_trading_algo(trading_start_date_str="2022-06-01", 
                     trading_end_date_str="2023-06-01",
                     starting_cash=100000):

    trading_data_start_date_str = "2022-06-01"
    trading_data_start_date = datetime.strptime(trading_data_start_date_str, "%Y-%m-%d")
    trading_start_date = datetime.strptime(trading_start_date_str, "%Y-%m-%d")
    trading_end_date = datetime.strptime(trading_end_date_str, "%Y-%m-%d")

    if trading_start_date - timedelta(days=90) < trading_data_start_date: 
        raise ValueError("Trading Start Date cannot be before when we have data \n")


    p = Portfolio(starting_cash = starting_cash)
    current_date = trading_start_date
    while current_date < trading_end_date:
        
        current_date_minus_one = current_date - timedelta(days = 1)
        current_date_minus_sixty = current_date - timedelta(days = 60)
        current_date_minus_one_str = current_date_minus_one.strftime("%Y-%m-%d")
        current_date_minus_sixty_str = current_date_minus_sixty.strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")
        print("Trading For Date {0} \n".format(current_date_str))


        mean_t, var_t, prob_t = calibrate_and_forecast(calibrate_start_period = current_date_minus_sixty_str,
                                                       calibrate_end_period = current_date_minus_one_str, 
                                                       no_plotting = True,
                                                       forecast_number_of_days = 14)

        current_date = current_date + timedelta(days = 1)
        p.execute_trading_algorithms(current_date_str, mean_t, var_t, prob_t)

        portfolio_position = p.get_current_share_position() * p.get_current_spot_position() + p.get_current_cash_position()

        print("Trading Simulation has finished \n")
        print("Finishing Cash Position is {0} \n".format(p.get_current_cash_position()))
        print("Finishing Share Position is {0} \n".format(p.get_current_share_position()))
        print("Current Spot Price is {0} \n".format(p.get_current_spot_position()))
        print("Finishing Portfolio Position is {0} \n".format(portfolio_position))


class TestDynamicalSystems(unittest.TestCase):

    def test_get_eth_option_data(self):
        ticker = "ETH"
        df = get_crypto_currency_option_df(ticker)


def main():

    print("In Main ... \n")
    run_trading_algo(trading_start_date_str="2023-03-01", trading_end_date_str="2023-04-01")


main()


