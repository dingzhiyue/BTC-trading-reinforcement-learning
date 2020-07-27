# BTC-trading-reinforcement-learning
This project explores the possibility of training a deep reinforcement learning agent to do tradings in the BTC market. 'https://github.com/dingzhiyue/BTC-trading-reinforcement-learning/blob/master/BTC_DQN_Zhiyue_Ding.ipynb' describes a outline 
of the whole project and some results.

The most recently updated version is 'version 2' which adds a time series prediction by a LSTM model to the observation space of the Deep Q Network (DQN). Training data 
set is the historical market price from Dec 2014 to Jan 2019. Validation data set is the data from Jan 2019 to Jun 2020.

In the 'version 2' folder:

'data_cleaning.py' pre-processes and prepared the data for training. 

'price_prediction_LSTM.py' predicts the price based on a LSTM model.

'environment.py' produces a typical OpenAI 'gym' class for the BTC trading environment. 

'DQN' builds the Deep Q Network as a reinforcement learning agent. A delayed 
target network is also used in order to stabilized the training process. 

'validation.py' validates the DQN model on the validation data set. It seems, this model out-performs
market by increasing the total asset by 3.5 times while the market price of the BTC increases only by 2.5 times!

