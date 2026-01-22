import requests
from datetime import datetime, timedelta
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import johnsonsu
from scipy.optimize import minimize
import csv

#1. Getting prices
def prices(startd,endd,tic):
    info=[]
    key='&apikey=ZKMMTO1ATDBLXH2K'
    ticker='&symbol='+str(tic)
    endpoint='function=TIME_SERIES_MONTHLY_ADJUSTED'
#    endpoint='function=TIME_SERIES_DAILY_ADJUSTED'
    size='&outputsize=full'
    web='https://www.alphavantage.co/query?'
    url =web+endpoint+ticker+size+key

    r = requests.get(url)
    if r.status_code==200:
        print('connection successful')
        data = r.json() #need to convert to json to navigate
#        r1=data.get('Time Series (Daily)', {})
        r1=data.get('Monthly Adjusted Time Series', {})
#        r2=data['Time Series (Daily)']
        r2=data['Monthly Adjusted Time Series']
        
        for date, values in sorted(r1.items()):
            if startd <= date <= endd:
                info.append([tic, date, values['5. adjusted close']])
    return info


#2. Getting the list of relevant prices: 0=daily, 1=weekly, 2=monthly, 3=yearly
def freq_series(prices,freq):
    results=[]
    for i in range(len(prices)):
        date1=prices[i][1]
        month=prices[i][1][5:7]
        day=prices[i][1][8:10]
        date2=datetime.strptime(date1, "%Y-%m-%d").date()
        year, week, weekday = date2.isocalendar()
        prices[i].append(day)
        prices[i].append(week)
        prices[i].append(month)
        prices[i].append(year)
        if freq==0 and i>0:
            if prices[i][3] != prices[i-1][3]:
                results.append([prices[i-1][1],prices[i-1][2]])
        elif freq==1 and i>0:
            if prices[i][4] != prices[i-1][4]:
                results.append([prices[i-1][1],prices[i-1][2]])
        elif freq==2 and i>0:
            if prices[i][5] != prices[i-1][5]:
                results.append([prices[i-1][1],prices[i-1][2]])
        elif freq==3 and i>0:
            if prices[i][6] != prices[i-1][6]:
                results.append([prices[i-1][1],prices[i-1][2]])
    foo=[float(sublist[1]) for sublist in results]
    results1=[]
    for i in range(1,len(foo),1):
        results1.append(np.log(float(foo[i]))-np.log(float(foo[i-1])))

    return {'data': results, 'retx':results1}

#3. Histogram
def histogram(returns, bins):
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=int(bins), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Frequency of Log Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

#4. Moments
def moments(returns):
    mean=np.mean(returns)
    variance=np.var(returns)
    skewness=stats.skew(returns)
    kurtosis=stats.kurtosis(returns, fisher=True)
    return {'first':mean, 'second':variance, 'third':skewness, 'fourth':kurtosis}


#5. Getting the SU Jhonson parameter distribution. Using the method of moments.
def moment_objective(params, sample_moments):
#process: find a combination of parameters of the SU that results in the same moments as the data
#minimize the distance    
    gamma, delta, xi, lam = params
    try:
        theoretical = johnsonsu.stats(gamma, delta, loc=xi, scale=lam, moments='mvsk')
    except Exception:
        return np.inf  # In case parameters lead to numerical issues, penalize heavily
    
    # Compute squared error between sample moments and theoretical moments
    error = np.sum((np.array(theoretical) - np.array(sample_moments))**2)
    return error

def fit_johnson_su_moments(sample_mean, sample_variance, sample_skew, sample_kurt):
    sample_moments = [sample_mean, sample_variance, sample_skew, sample_kurt]
    # An initial guess for the parameters: 
    # gamma: 0, delta: 1, xi: sample_mean, lam: standard deviation.
    initial_guess = [0.0, 1.0, sample_mean, np.sqrt(sample_variance)]
    # You can set bounds if you want to restrict parameters:
    bounds = [(-10, 10), (0.1, 10), (None, None), (0.1, 10)]
    result = minimize(moment_objective, initial_guess, args=(sample_moments,), bounds=bounds)
    if result.success:
        return result.x  # returns gamma, delta, xi, lam
    else:
        raise RuntimeError("Optimization failed: " + result.message)


#6. Moments Johnson SU
def get_johnson_su_moments(gamma, delta, xi, lam):
    mean, variance, skewness, kurtosis = johnsonsu.stats(gamma, delta, loc=xi, scale=lam, moments='mvsk')
    return {'first':mean, 'second':variance, 'third':skewness, 'fourth':kurtosis}

    

#7. Plot Johnson SU vs actual/empirical 
def plot_johnson_vs_actual(returns, gamma, delta, xi, lam, number):
    x = np.linspace(min(returns), max(returns), 1000)
    pdf = stats.johnsonsu.pdf(x, gamma, delta, loc=xi, scale=lam)

    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=int(number), density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Empirical')
    plt.plot(x, pdf, 'r-', lw=2, label='Fitted Distribution')
    plt.title('Fitted Distribution vs Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

#8. Computing cumulative distribution
def compute_cdf(threshold, gamma, delta, xi, lam):
    return stats.johnsonsu.cdf(threshold, gamma, delta, loc=xi, scale=lam)

#9. Import CSV
path=r'C:\Users\cortesf\Dropbox\Teaching\2025- Spring\FINA 4460\Content\sp500hist.csv'
def histprices(path):
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

bins=15
info=prices("2000-01-01","2024-12-31","SPY")
#info=histprices(path)
#0=daily, 1=weekly, 2=monthly, 3=yearly
series=freq_series(info,3)['retx']
fulldata=freq_series(info,3)['data']
histogram(series,bins)
distribution=moments(series)
#
sample_mean=distribution['first']
sample_variance=distribution['second']
sample_skew=distribution['third']
sample_kurt=distribution['fourth']
#
gamma, delta, xi, lam = fit_johnson_su_moments(sample_mean, sample_variance, sample_skew, sample_kurt)
sim_moments=get_johnson_su_moments(gamma, delta, xi, lam)
print(distribution)
print(sim_moments)

plot_johnson_vs_actual(series, gamma, delta, xi, lam,bins)
bottom=compute_cdf(-0.158, gamma, delta, xi, lam)
print(bottom)


