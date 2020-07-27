import pandas as pd
from environment import *
from DQN import *
from keras import models
import matplotlib.pyplot as plt

def read_test_data():#validation data set
    path = 'data/Bitcoin_2019_1_10_to_2020_6_1.csv'
    df = pd.read_csv(path)
    Date = df['Date'].iloc[::-1]
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    df['Open'] = df['Open'].str.replace(',', '').astype(float)
    df['High'] = df['High'].str.replace(',', '').astype(float)
    df['Low'] = df['Low'].str.replace(',', '').astype(float)
    for i in range(len(df['Vol.'])):
        if 'K' in df['Vol.'].iloc[i]:
            df['Vol.'].iloc[i] = float(df['Vol.'].iloc[i][:-1]) * 1000
        elif 'M' in df['Vol.'].iloc[i]:
            df['Vol.'].iloc[i] = float(df['Vol.'].iloc[i][:-1]) * 1000000
        else:
            df['Vol.'].iloc[i] = float(df['Vol.'].iloc[i])



    Close = pd.Series(df['Price'].values[::-1], index=Date)
    Open = pd.Series(df['Open'].values[::-1], index=Date)
    High = pd.Series(df['High'].values[::-1], index=Date)
    Low = pd.Series(df['Low'].values[::-1], index=Date)
    Volume = pd.Series(df['Vol.'].values[::-1], index=Date)
    return Close, Open, High, Low, Volume#Series

def DQN_validation():
    Close, Open, High, Low, Volume = read_test_data()
    cash =10000
    BTC = 0
    windows = 10
    forecast_size = 2

    env = trading_Env(Open, Close, High, Low, Volume, Volume, Close, windows, cash, BTC, forecast_size)
    DQN_validate = DQN(env)
    DQN_validate.model = models.load_model('models/BTC_DQN')
    state = DQN_validate.env.reset()
    DQN_validate.epsilon = 0
    for i in range(len(Close)-windows-1):
        act = DQN_validate.epsilon_greedy_action(state)
        state_next, rewards, done, info = DQN_validate.env.step(act)
        state = state_next
    DQN_validate.env.render()

if __name__=='__main__':
    DQN_validation()








