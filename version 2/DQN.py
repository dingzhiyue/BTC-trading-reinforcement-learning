import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import collections
import random
from environment import trading_Env
import price_prediction_LSTM as daily
import data_cleaning as minute

class DQN():
    def __init__(self, env):
        self.env = env
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.epsilon = 0.1#epsilon greedy for selecting action

        self.pool_size = 200
        self.experience_pool = collections.deque(maxlen=self.pool_size)
        self.batch_size = 32#batch size of each sample from the experience pool

        self.gamma = 0.01#rewards decay
        self.alpha = 0.5#target weight update portion


    def build_model(self):
        model = Sequential()
        input_dim = self.env.observation_space.shape[0]
        output_dim = 21
        model.add(Dense(100, input_shape=(input_dim, ), activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(output_dim))
        model.compile(optimizer='Adam', loss='mean_squared_error')
        return model

    def epsilon_greedy_action(self, state):#state has the same form as state in env
        rd = np.random.uniform()
        if rd > self.epsilon:
            act = np.argmax(self.model.predict(state))
        else:
            act = self.env.action_space.sample()
        return act

    #experience pool
    def save_into_pool(self, state, act):
        state_next, rewards, done, info = self.env.step(act)
        #print(state.shape)
        #print(state_next.shape)
        new_sample = [state, act, rewards, state_next, done]
        self.experience_pool.append(new_sample)
        return state_next, rewards, done

    def initial_experience_pool(self):
        i = 0
        while True:
            if i > self.pool_size:
                break
            state = self.env.reset()
            while True:
                i += 1
                act = self.env.action_space.sample()
                state_next, rewards, done = self.save_into_pool(state, act)
                state = state_next
                if done or i > self.pool_size:
                    break

    def sample_from_pool(self):
        training_samples = random.sample(self.experience_pool, self.batch_size)
        return training_samples

    #batch training
    def train_batch(self):
        training_samples = self.sample_from_pool()
        x_train = []
        y_train = []
        for sample in training_samples:
            state, act, rewards, state_next, done = sample
            x_train.append(state)
            target = self.target_model.predict(state)
            #print(target)
            if done:
                target[0][act] = rewards
            else:
                target[0][act] = rewards + self.gamma * np.max(self.target_model.predict(state_next))
            y_train.append(target)
        x_train = np.squeeze(np.array(x_train))#ndarray(32.92)
        y_train = np.squeeze(np.array(y_train))#ndarray(32.21)
        self.model.fit(x_train, y_train, epochs=10, verbose=0)

    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        #self.target_model.set_weights(weights)#simply update by weights
        for i in range(len(weights)):
            target_weights[i] = (1 - self.alpha) * target_weights[i] + self.alpha * weights[i]
        self.target_model.set_weights(target_weights)

    def model_save(self, name):
        self.model.save(name)


#DQN
def DQN_train():
    #use daily based data
    Open, Close, High, Low, Volume = daily.read_data()
    Volume_BTC = Volume
    Volume_Currency = Volume
    Weighted_price = Close

    #use minute based data
    #Open, Close, High, Low, Volume_BTC, Volume_Currency, Weighted_Price = minute.pre_process()

    start_cash = 10000
    start_BTC = 0
    windows = 10
    forecast_size = 2
    env = trading_Env(Open.iloc[:800], Close.iloc[:800], High.iloc[:800], Low.iloc[:800], Volume_BTC.iloc[:800], Volume_Currency.iloc[:800], Weighted_price.iloc[:800], windows, start_cash, start_BTC, forecast_size)
    DQN_agent = DQN(env)

    epochs = 15
    iterations = 700

    DQN_agent.initial_experience_pool()
    print('initial pool finished')
    for epoch in range(epochs):
        print('epoch: ', epochs)
        state = DQN_agent.env.reset()
        for iteration in range(iterations):
            print('iteration: ', iteration)
            act = DQN_agent.epsilon_greedy_action(state)
            state_next, rewards, done = DQN_agent.save_into_pool(state, act)
            DQN_agent.train_batch()
            if iterations%10 == 0:
                DQN_agent.update_target_model()
            state = state_next
            if done:
                break
    DQN_agent.model_save('models/BTC_DQN')
    DQN_agent.env.render()

if __name__=='__main__':
    DQN_train()
















