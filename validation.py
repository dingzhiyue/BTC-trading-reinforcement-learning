from train import *
from environment_gym import *
import keras as ks

def DQN_validation():
    market_data_path = 'data/coinbase_daily.csv'
    df = pd.read_csv(market_data_path)

    df_filter = df[['Open','High','Low','Close','Volume BTC']]
    market_data = df_filter.values[::-1]

    initial_cash_balance = 10000
    initial_bt_balance = 0
    windows = 10

    start_time = 800
    env = trading_environment(market_data[start_time:], windows=windows, initial_cash_balance=initial_cash_balance, initial_bt_balance=initial_bt_balance)
    DQN_validate = DQN(env)
    DQN_validate.model = ks.models.load_model('BTC_DQN')

    trading_steps = 700
    state = DQN_validate.env.reset()

    #DQN_validate.epsilon = 0# overwirte epsilon greedy
    for i in range(trading_steps):
        act = DQN_validate.epsilon_greedy_action(state)
        state_next, rewards, done, info = DQN_validate.env.step(act)
        state = state_next
    DQN_validate.env.render()



if __name__=='__main__':
    DQN_validation()

