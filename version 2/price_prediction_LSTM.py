import keras.models as md
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pickle import dump

def read_data():
    path = 'data/coinbase_daily.csv'
    df = pd.read_csv(path)
    date = df['Date']
    Open = pd.Series(df['Open'].iloc[::-1].values, index=date.iloc[::-1])
    Close = pd.Series(df['Close'].iloc[::-1].values, index=date.iloc[::-1])
    High = pd.Series(df['High'].iloc[::-1].values, index=date.iloc[::-1])
    Low = pd.Series(df['Low'].iloc[::-1].values, index=date.iloc[::-1])
    Volume = pd.Series(df['Volume BTC'].iloc[::-1].values, index=date.iloc[::-1])
    return Open, Close, High, Low, Volume#pd.Series

def diff_trans(time_series, inverse=False):#ndarray, diff_trans and its inverse
    if inverse == True:
        trans = [np.sum(time_series[0:i]) for i in range(len(time_series)+1)]
    else:
        trans = np.diff(time_series)
    return trans#ndarray

#normalization by parts
def pre_process(data):#pd.Series
    data = data.values
    #1st diff
    #data = diff_trans(data)
    STDscaler = StandardScaler()
    STDscaler.fit(data.reshape(-1, 1))#uniform scale
    #normalize by regions
    data_part1 = data[:900]
    data_part2 = data[900:1050]
    data_part3 = data[1050:1200]
    data_part4 = data[1200:]
    data_by_parts = [data_part1, data_part2, data_part3, data_part4]
    data_scaled = []
    for part in data_by_parts:
        temp = STDscaler.transform(np.reshape(part, (-1, 1)))#uniform scale
        #temp = STDscaler.fit_transform(np.reshape(part, (-1,1)))#scale by part
        data_scaled.append(temp)
    data_scaled = np.concatenate(data_scaled, axis=0)
    dump(STDscaler, open('STDscaler.pkl', 'wb'))
    return STDscaler, data_scaled#ndarray(-1,1)

#prepare train/test for LSTM
def train_test_split(data, lookback_window):#ndarray(-1,1)
    data_LSTM = np.ndarray(shape=(data.shape[0]-lookback_window, lookback_window+1))
    for i in range(lookback_window+1):
        data_LSTM[:, i] = data[i:data.shape[0]-lookback_window+i, 0]
    train = data_LSTM[:int(0.8*data.shape[0]),:]
    test = data_LSTM[int(0.8*data.shape[0]):,:]
    train_x = train[:,:lookback_window]
    train_y = train[:,-1]
    test_x = test[:,:lookback_window]
    test_y = test[:,-1]

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    train_y = np.reshape(train_y, (len(train_y),1))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    test_y = np.reshape(test_y, (len(test_y), 1))
    return train_x, train_y, test_x, test_y#ndarray(-1,1,window), ndarray(-1,1)

#fit a LSTM model
def LSTM_model(train_x, train_y, test_x, test_y):#ndarray(-1,1,window), ndarray(-1,1)
    model = Sequential()
    model.add(LSTM(3, batch_input_shape=(1, 1, train_x.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    epochs = 30
    test_error = []
    for i in range(epochs):
        print('EPOCH:', i)
        model.fit(train_x, train_y, batch_size=1, verbose=1)
        prediction = model.predict(test_x, batch_size=1)
        test_MSE = (np.dot((prediction[:, 0] - test_y[:, 0]), (prediction[:, 0] - test_y[:, 0]))) ** (0.5) / \
                   prediction.shape[0]
        test_error.append(test_MSE)
        model.reset_states()
    model.save('LSTM_model')
    prediction = model.predict(test_x, batch_size=1)
    #in-sample error
    train_prediction = model.predict(train_x, batch_size=1)

    test_MSE = (np.dot((prediction[:,0]-test_y[:,0]), (prediction[:,0]-test_y[:,0])))**(0.5)/prediction.shape[0]
    train_MSE = (np.dot((train_prediction[:, 0] - train_y[:, 0]), (train_prediction[:, 0] - train_y[:, 0])))**(0.5)/train_prediction.shape[0]

    print('test_MSE: ', test_MSE, 'train_MSE: ', train_MSE)


    plt.figure()
    plt.plot(prediction)
    plt.plot(test_y)
    plt.legend(['predict', 'test_y'])
    plt.title('LSTM_predict')
    plt.figure()
    plt.plot((train_prediction))
    plt.plot(train_y)
    plt.legend(['train_predict','train_y'])
    plt.figure()
    plt.plot(test_error)
    plt.title('performance metric')
    plt.show()

    return prediction

def plot_predictions(train_x, train_y, test_x, test_y, STDscaler):
    model = md.load_model('LSTM_model')
    #in sample error
    train_pred = model.predict(train_x, batch_size=1)
    plt.figure()
    plt.plot(train_pred)
    plt.plot(train_y)
    plt.legend(['in_sample_pred','train_y'])
    train_MSE = (np.dot((train_pred[:,0]-train_y[:,0]), (train_pred[:,0]-train_y[:,0])))**(0.5)/train_pred.shape[0]
    plt.title(['MSE: ', train_MSE])
    #test
    test_pred = model.predict(test_x, batch_size=1)
    plt.figure()
    plt.plot(test_pred)
    plt.plot(test_y)
    plt.legend(['test_pred', 'test_y'])
    test_MSE = (np.dot((test_pred[:, 0] - test_y[:, 0]), (test_pred[:, 0] - test_y[:, 0]))) ** (0.5) / \
                test_pred.shape[0]
    plt.title(['MSE: ', test_MSE])
    plt.show()

    #orig plot
    plt.figure()
    plt.plot(STDscaler.inverse_transform(test_pred))
    plt.plot(STDscaler.inverse_transform(test_y))
    plt.legend(['test_pred', 'test_y'])
    plt.title('original price')
    plt.show()

def plot_accumalted_prediction(train_x, train_y, test_x, test_y):
    model = md.load_model('LSTM_model')
    #in sample
    train_pred = np.zeros(shape=train_y.shape)
    x = np.ndarray((1,1,train_x.shape[2]))
    x[0,0,:] = train_x[0,0,:]
    for i in range(train_y.shape[0]):
        train_pred[i] = model.predict(x, batch_size=1)
        temp = x[0,0,1:]
        x[0,0,-1] = train_pred[i]#+ np.random.normal(0.0,0.1)
        x[0,0,0:-1] = temp
    plt.figure()
    plt.plot(train_pred)
    plt.plot(np.squeeze(train_y))
    train_MSE = (np.dot((np.squeeze(train_pred) - np.squeeze(train_y)), (np.squeeze(train_pred) - np.squeeze(train_y)))) ** (0.5) / \
                train_pred.shape[0]
    plt.title(['MSE: ', train_MSE])
    plt.legend(['train_pred','train_y'])
    plt.show()








if __name__=='__main__':
    Open, Close, High, Low, Volume_BTC = read_data()
    STDscaler, trans = pre_process(Close)
    #trans = np.reshape(Close.values, (-1,1))
    train_x, train_y, test_x, test_y = train_test_split(trans, 10)

    predict = LSTM_model(train_x, train_y, test_x, test_y)
    plot_predictions(train_x, train_y, test_x, test_y, STDscaler)
    #plot_accumalted_prediction(train_x, train_y, test_x, test_y)

