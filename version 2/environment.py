import gym
import numpy as np
import keras.models as md
import matplotlib.pyplot as plt
from data_cleaning import *
from pickle import load
from price_prediction_LSTM import *

class trading_Env(gym.Env):
    def __init__(self, Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price, windows, initial_cash, initial_BTC, forecast_size):#Series
        #market data
        self.Open = Open.values.reshape(-1, 1)#ndarray (-1,1)
        self.Close= Close.values.reshape(-1, 1)
        self.High = High.values.reshape(-1, 1)
        self.Low = Low.values.reshape(-1, 1)
        self.Volume_BTC = Volume_BTC.values.reshape(-1, 1)
        self.Volume_Currency = Volume_Currency.values.reshape(-1, 1)
        self.Weighted_Price = Weighted_Price.values.reshape(-1, 1)
        self.market = np.concatenate([self.Open, self.Close, self.High, self.Low, self.Volume_Currency, self.Weighted_Price ], axis=1)

        self.windows = windows
        # observation matrix
        self.obs_next = self.market[:self.windows, :]#ndarray
        #balance matrix
        self.cash_balance = initial_cash
        self.BTC_balance = initial_BTC
        self.total_balance = self.cash_balance + self.BTC_balance * self.obs_next[-1, -1]#used weighted price
        self.balance = np.matrix([[self.cash_balance, self.BTC_balance, self.total_balance] for _ in range(self.windows)])#ndarray
        #history
        self.history_temp = np.concatenate([self.obs_next, self.balance], axis=1)#ndarray
        self.history = np.concatenate([self.obs_next, self.balance], axis=1)#ndarray
        #predictions
        self.forecast_size = forecast_size
        self.prediction_model = md.load_model('models/LSTM_model')
        self.predictions = self.prediciton()#current pred ndarray (1,forecast_size)
        self.prediction_history = self.predictions#pred history
        #env parameters
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(21)#0-9 buy, 10-19 sell, 20 hold
        self.observation_space = gym.spaces.Box(low=0, high=100000, shape=((self.history.shape[0]*self.history.shape[1]) + self.forecast_size, 1))#state shape (-1,1)


    def prediciton(self):#scaler for normalization
        scaler = load(open('STDscaler.pkl', 'rb'))
        test = self.obs_next[:, -1]#weighted price
        test = scaler.transform(test.reshape(-1, 1))  # 可以改进分部rescale
        test = test.reshape(1, 1, -1)
        pred = []
        for i in range(self.forecast_size):
            pred_temp = self.prediction_model.predict(test, batch_size=1)
            test[0, 0, :-1] = test[0, 0, 1:]
            test[0, 0, -1] = pred_temp
            pred.append(scaler.inverse_transform(pred_temp))
        pred = np.array(pred)
        return pred.reshape(1, -1)#ndarray (1,forecast_size)

    def take_action(self, action):
        if action <= 9:#buy BTC
            cash = self.balance[-1, 0] * (1 - ((action+1)/10))
            btc = self.balance[-1, 1] + (self.balance[-1, 0] * ((action+1)/10) / self.obs_next[-1, -1])#use weighted price
            total = cash + btc * self.obs_next[-1, -1]
        elif action == 20:#hold BTC
            cash = self.balance[-1, 0]
            btc = self.balance[-1, 1]
            total = self.balance[-1, 2]
        else:#sell BTC
            cash = self.balance[-1, 0] + (((action-9)/10) * self.balance[-1, 1]) * self.obs_next[-1,-1]
            btc = self.balance[-1, 1] * (1 - (action-9)/10)
            total = cash + btc * self.obs_next[-1, -1]

        #update self.balance
        balance_temp = np.array([cash, btc, total])
        self.balance[:-1, :] = self.balance[1:, :]
        self.balance[-1, :] = balance_temp.reshape(1, 3)
        #update self.obs_next
        self.current_step += 1
        if self.current_step + self.windows >= self.market.shape[0]:
            print('running out of market data')
            self.reset()
        self.obs_next = self.market[self.current_step:self.current_step + self.windows, :]
        #update history
        self.history_temp = np.concatenate([self.obs_next, self.balance], axis=1)
        self.history = np.concatenate([self.history, self.history_temp[-1, :].reshape(1, -1)], axis=0)
        #update self.predictions
        self.predictions = self.prediciton()
        self.prediction_history = np.concatenate([self.prediction_history, self.predictions], axis=0)

    def reward(self):#always calculate after take_action
        rewards = (self.balance[-1, 0] + self.balance[-1, 1] * self.obs_next[-1, -1]) - self.balance[-1,  -1]
        return rewards

    def done(self):#alway after take_action
        if (self.current_step + self.windows >= self.market.shape[0]):
            print('running out of data')
            return True
        elif (self.balance[-1, -1] < 0.3 * self.total_balance):
            print('bankrupt')
            return True
        else:
            return False

    def transform_to_state(self):
        state = self.history_temp.reshape(1, -1)
        state = np.concatenate([state, self.predictions], axis=1)
        return state #ndarray (1,-1)


    def step(self, action):
        self.take_action(action)
        rewards = self.reward()
        state = self.transform_to_state()
        done = self.done()
        return state, rewards, done, {}

    def reset(self):
        self.current_step = 0
        self.obs_next = self.market[:self.windows, :]
        self.balance = np.matrix(
            [[self.cash_balance, self.BTC_balance, self.total_balance] for _ in range(self.windows)])  # ndarray
        self.history_temp = np.concatenate([self.obs_next, self.balance], axis=1)
        self.history = np.concatenate([self.obs_next, self.balance], axis=1)
        self.predictions = self.prediciton()
        self.prediction_history = self.predictions
        state = self.transform_to_state()
        return state

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render(self):
        cash_plt = self.history[:, 6]
        BTC_plt = self.history[:, 7]
        total_plt = self.history[:, 8]
        plt.figure()
        plt.plot(np.linspace(1, len(cash_plt), len(cash_plt)), cash_plt, '--o')
        plt.plot(np.linspace(1, len(cash_plt), len(cash_plt)), total_plt, '-ok')
        plt.legend(['cash in hand', 'total asset'])
        plt.figure()
        plt.plot(np.linspace(1, len(cash_plt), len(cash_plt)), BTC_plt, '--o')
        plt.legend(['# BTC'])

        w = self.history[:, 5]
        pred_plot1 = self.prediction_history[:, 0]
        pred_plot2 = self.prediction_history[:, 1]
        plt.figure()
        plt.plot(np.linspace(1, len(w), len(w)), w, '-o')
        plt.plot(pred_plot1, '-o')
        plt.plot(pred_plot2, '-o')
        plt.legend(['Close', 'pred1', 'pred2'])

        plt.show()

if __name__=='__main__':
    Open, Close, High, Low, Volume_BTC = read_data()
    #Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price = pre_process()
    env = trading_Env(Open, Close, High, Low, Volume_BTC, Volume_BTC, Close, 10, 10000, 0, 2)
    env.reset()
    for i in range(1500):
        env.step(np.random.randint(0,20))
    env.render()
















