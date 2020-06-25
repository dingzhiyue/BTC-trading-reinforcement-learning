import gym
import numpy as np
import random
import pandas as pd
import numpy as np
from envs.environment_gym import trading_environment
import matplotlib.pyplot as plt
import keras as ks
from keras.layers import Dense
import collections

class DQN:
    def __init__(self, env):
        self.env = env
        self.pool_size=200
        self.learning_rate = 0.01
        self.epsilon = 0.1
        self.batch_size = 32
        self.alpha = 0.5
        self.gamma = 0.01
        self.experience_pool = collections.deque(maxlen=self.pool_size)
        self.model = self.build_model()
        self.target_model = self.model

    def build_model(self):
        model = ks.models.Sequential()
        input_size = self.env.observation_space.shape[0] * self.env.windows
        #input_size = 16
        output_size = 21

        model.add(Dense(100, input_dim=input_size, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(output_size))

        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=self.learning_rate))
        return model

    def epsilon_greedy_action(self,state):
        rd = np.random.uniform()
        if rd >= self.epsilon:
            state = self.transfer_state_dim(state)
            act = np.argmax(self.model.predict(state))
        else:
            act = self.env.action_space.sample()
        return act

    def transfer_state_dim(self, state):#out put ndarray (1, 80)
        out = np.reshape(state, [1, state.shape[0] * state.shape[1]])
        return out


    def save_into_pool(self, state, act):
        state_next, rewards, done, info = self.env.step(act)
        new_sample = [state, act, rewards, state_next, done]
        self.experience_pool.append(new_sample)
        return state_next, rewards, done

    def initial_pool(self):
        i = 0
        while True:
            if i > self.pool_size:
                break
            state = self.env.reset()
            while True:
                i += 1
                act = self.env.action_space.sample()
                state_next, rewards, done, info = self.env.step(act)
                self.save_into_pool(state, act)
                state = state_next
                if done or i > self.pool_size:
                    break

    def sample_from_pool(self):
        training_samples = random.sample(self.experience_pool, self.batch_size)
        return training_samples

    def train_batch(self, training_samples):
        x_train = []
        y_train = []
        for sample in training_samples:
            state, act, reward, state_next, done = sample
            state = self.transfer_state_dim(state)
            x_train.append(state)
            state_next = self.transfer_state_dim(state_next)
            target = self.target_model.predict(state)
            if done:
                target[0][act] = reward
            else:
                target[0][act] = reward + self.gamma * np.max(self.target_model.predict(state_next))
            y_train.append(target)
        x_train = np.array(x_train)
        x_train = np.squeeze(x_train)
        y_train = np.array(y_train)
        y_train = np.squeeze(y_train)
        self.model.fit(x_train, y_train, epochs=50, verbose=1)

    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        #self.target_model.set_weights(weights)#simply update by weights
        for i in range(len(weights)):
            target_weights[i] = (1 - self.alpha) * target_weights[i] + self.alpha * weights[i]
        self.target_model.set_weights(target_weights)

    def model_save(self, name):
        self.model.save(name)


def DQN_train():
    market_data_path = 'data/coinbase_daily.csv'
    df = pd.read_csv(market_data_path)

    df_filter = df[['Open','High','Low','Close','Volume BTC']]
    market_data = df_filter.values[::-1]


    initial_cash_balance = 10000
    initial_bt_balance = 0
    windows = 10
    env = trading_environment(market_data, windows=windows, initial_cash_balance=initial_cash_balance, initial_bt_balance=initial_bt_balance)
    DQN_agent = DQN(env)

    trails = 2
    iterations = 600

    DQN_agent.initial_pool()
    print('initial pool finished')
    for trail in range(trails):
        print('trail', trail)
        state = env.reset()
        for iteration in range(iterations):
            print('iteration', iteration)
            act = DQN_agent.epsilon_greedy_action(state)#下一步
            state_next, rewards, done = DQN_agent.save_into_pool(state, act)#save into pool同时做一步step
            training_samples = DQN_agent.sample_from_pool()#sample from pool
            DQN_agent.train_batch(training_samples)#train batch
            if iteration%10==0:#update target NN every10 steps
                DQN_agent.update_target_model()
            state = state_next
            if done:
                break
        DQN_agent.env.render()

    DQN_agent.model_save('BTC_DQN')

if __name__=='__main__':
    DQN_train()































