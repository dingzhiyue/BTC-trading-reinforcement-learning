import gym
import numpy as np
import matplotlib.pyplot as plt


class trading_environment(gym.Env):

    def __init__(self, market_data, windows, initial_cash_balance,initial_bt_balance):
        self.market = market_data
        self.windows = windows
        self.cash_balance = initial_cash_balance
        self.bitcoin_balance = initial_bt_balance
        #balance [cash, bitcoin, total]
        self.balance = [[self.cash_balance, self.bitcoin_balance, self.cash_balance] for i in range(self.windows)]
        self.history = [self.balance[-1]]
        self.balance = np.matrix(self.balance)
        self.current_step = 0
        #action space
        self.action_space = gym.spaces.Discrete(21)
        #state
        self.observation = gym.spaces.Box(low=0,high=1, shape=(8,))

    def next_observation(self):
        data_next = self.market[self.current_step + 1: self.current_step + 1 + self.windows,:]
        return data_next

    def choose_action(self, action):
        print(action)
        if action<=9:#buy bit coin
            cash = self.balance[-1, 0] * (1 - ((action+1)/10))
            bt = self.balance[-1, 1] + (self.balance[-1, 0] * ((action+1)/10) / self.market[-1,1])
            balance_temp = [cash, bt, cash + bt * self.market[-1,1]]
        elif action==20:
            balance_temp = [self.balance[-1,0],self.balance[-1,1],self.balance[-1,2]]
        else:#sell
            cash = self.balance[-1,0] + (self.balance[-1,1] * ((action-9)/10) / self.market[-1,1])
            bt = self.balance[-1,1] * (1-(action-9)/10)
            balance_temp = [cash, bt, cash + bt * self.market[-1,1]]

        self.history.append(balance_temp)
        self.balance = np.append(self.balance, np.reshape(np.array(balance_temp), (1,3)), axis=0)
        self.balance = self.balance[1:,:]



    def reward(self):
        rewards = self.balance[-1, 2] - self.balance[-2, 2]
        return rewards

    def done(self):
        if (self.current_step + self.windows + 1 >= self.market.shape[0]) or (self.balance[-1, 2] < self.cash_balance):
            return True
        else:
            return False

    def step(self, action):
        self.choose_action(action)
        rewards = self.reward()
        market_next = self.next_observation()
        observation = np.append(market_next, self.balance, axis=1)
        done = self.done()
        self.current_step += 1
        return observation, rewards, done, {}

    def reset(self):
        self.balance = [[self.cash_balance, self.bitcoin_balance, self.cash_balance] for i in range(self.windows)]
        self.history = [self.balance[-1]]
        self.balance = np.matrix(self.balance)
        self.current_step = 0
        observation = np.append(self.market[:self.windows], self.balance, axis=1)

        return observation


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render(self):
        print(self.history)
        cash_plt, bitcoin_plt, total_plt = zip(*self.history)
        plt.figure()
        plt.plot(np.linspace(1,len(cash_plt),len(cash_plt)),cash_plt, '--o')
        plt.plot(np.linspace(1,len(cash_plt),len(cash_plt)), bitcoin_plt, '--o')
        plt.plot(np.linspace(1,len(cash_plt),len(cash_plt)), total_plt, '-ok')
        plt.legend(['cash', 'bitcoin','total'])
        #plt.figure()
        #plt.plot(self.market[0])
        #plt.plot(self.market[1])
        #plt.plot(self.market[2])
        #plt.plot(self.market[3])
        #plt.plot(self.market[4])
        #plt.legend(['O','C','H','L','V'])
        plt.imshow




