import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(path):
    df = pd.read_csv(path)
    Time = df['Timestamp'].values
    Open = pd.Series(df['Open'].values, index=Time)
    Close = pd.Series(df['Close'].values, index=Time)
    High = pd.Series(df['High'].values, index=Time)
    Low = pd.Series(df['Low'].values, index=Time)
    Volume_BTC = pd.Series(df['Volume_(BTC)'].values, index=Time)
    Volume_Currency = pd.Series(df['Volume_(Currency)'].values, index=Time)
    Weighted_Price = pd.Series(df['Weighted_Price'].values, index=Time)
    return Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price#Series

def plots(data):#ndarray
    plt.figure()
    plt.plot(data)
    plt.show()

def fill_na(data, method='bfill'):#Series, method = 'ffill', 'bfill', 'drop', or replace values
    NaNs = data.isna().sum()
    print('there are ', NaNs, 'NaN ', NaNs/len(data))
    if method=='drop':
        data = data.dropna()
    elif method=='ffill'or method=='bfill':
        data = data.fillna(method=method)
    else:
        data = data.fillna(method=method)
    return data#Series

def pre_process():#only for minute based data
    path = 'data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price = read_data(path)
    raw_data = [Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price]
    cleaned_data = []
    #fillna
    for item in raw_data:
        temp = fill_na(item, 'bfill')
        cleaned_data.append(temp)
    Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price = cleaned_data
    return Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price#Series


if __name__=='__main__':
    Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price = pre_process()

