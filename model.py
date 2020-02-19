from numpy import linalg
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
import logging
import datetime
import pdb
import matplotlib.pyplot as plt



#TODO написать бэктестинг - прогон и оценку оригинальных данных по вычисленным рискам







def get_logger():

    logger = logging.getLogger('base')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'./logs from {datetime.datetime.isoformat(datetime.datetime.now())[:10]}.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


#Глобальные параметры
RISK_FACTORS_NUM=17
NUM_OF_INSTRUMENTS=17
PRICE_OF_INSTRUMENTS = [10**6 for _ in range(10)]+[10**7 for _ in range(5)]+[10**7 for _ in range(2)]
INSTRUMENTS_LOOKBACK=10
RISKS_LOOKBACK=10
SIM_NUM=100

logger=get_logger()


def get_model():
    return LinearRegression() #можем поменять модельку здесь


def train_models(risks_df, instruments_df):
    logger.info('Model training started')
    models = []
    X = np.stack([risks_df.values[i:i+INSTRUMENTS_LOOKBACK].reshape(-1) for i in range(risks_df.shape[0]-INSTRUMENTS_LOOKBACK)])
    for ins in range(NUM_OF_INSTRUMENTS):
        y_ = instruments_df.values[INSTRUMENTS_LOOKBACK:,ins]
        lr = get_model()
        lr.fit(X,y_)
        models.append(lr)
    logger.info('Model training is done')
    return models

def inference_models(risks_df, models):
    X = np.stack([risks_df[i:i+INSTRUMENTS_LOOKBACK].reshape(-1) for i in range(risks_df.shape[0]-INSTRUMENTS_LOOKBACK)])
    pred = np.stack([model.predict(X) for model in models], axis=1)
    return pred




def get_decomp(df_of_risks):
    merged_interpolated_diff_corr = df_of_risks.iloc[1:].corr()
    decomposed = linalg.cholesky(merged_interpolated_diff_corr)
    logger.info('Decomposition calculated')
    return decomposed




def get_data():
    #Здесь - подгрузка и подготовка данных

    shares = ['./shares/AFLT_160101_200101.csv', 
              './shares/GAZP_160101_200101.csv',
              './shares/GMKN_160101_200101.csv', 
              './shares/KMAZ_160101_200101.csv', 
              './shares/LKOH_160101_200101.csv', 
              './shares/PIKK_160101_200101.csv', 
              './shares/MGNT_160101_200101.csv', 
              './shares/RBCM_160101_200101.csv', 
              './shares/ROSN_160101_200101.csv', 
              './shares/SBER_160101_200101.csv']

    shares_name = [i[9:13] for i in shares]

    # Нужно интерполировать
    bonds = ['./bonds/SU26212RMFS9_160101_200101.csv',
             './bonds/SU26205RMFS3_160101_200101.csv',
             './bonds/SU26207RMFS9_160101_200101.csv', 
             './bonds/SU26209RMFS5_160101_200101.csv', 
             './bonds/SU26211RMFS1_160101_200101.csv']

    bonds_name = [i[8:20] for i in bonds]

    currencies = ['./index/USD_RUB.csv', './index/CNY_RUB.csv']

    currencies_name = [i[8:15] for i in currencies]

    indexes = ['./index/IMOEX_160101_200101.csv', 
               './index/RTSI_160101_200101.csv', 
               './index/ICE.BRN_160101_200101.csv']

    indexes_name = ['MOEX', 'RTSI', 'Brent']

    zero_bond = './zerobond.csv'

    act_instruments = pd.read_csv(shares[0], index_col='<DATE>').drop(['<TICKER>', '<PER>', '<TIME>', '<HIGH>', '<LOW>', '<VOL>', '<OPEN>'], axis=1).rename(columns={'<CLOSE>':shares_name[0]})
    act_instruments.index = pd.to_datetime(act_instruments.index, dayfirst=True)
    k = 1
    for i in shares[1:]:
        tmp = pd.read_csv(i, index_col='<DATE>') \
                .drop(['<TICKER>', '<PER>', '<TIME>', '<HIGH>', '<LOW>', '<VOL>', '<OPEN>'], axis=1) \
                .rename(columns={'<CLOSE>': shares_name[k]})
        tmp.index = pd.to_datetime(tmp.index, dayfirst=True)
        act_instruments = act_instruments.join(tmp, how='left')
        k += 1
        
    k = 0
    for i in bonds:
        tmp = pd.read_csv(i, index_col='<DATE>') \
                .drop(['<TICKER>', '<PER>', '<TIME>', '<HIGH>', '<LOW>', '<VOL>', '<OPEN>'], axis=1) \
                .rename(columns={'<CLOSE>' : bonds_name[k]})
        tmp.index = pd.to_datetime(tmp.index, dayfirst=True)
        act_instruments = act_instruments.join(tmp, how='left')
        k += 1

    k=0
    for i in currencies:
        tmp = pd.read_csv(i, index_col='Date') \
                .drop(['Open', 'High', 'Low', 'Change %'], axis=1) \
                .rename(columns={'Price' : currencies_name[k]})
        tmp.index = pd.to_datetime(tmp.index)
        act_instruments = act_instruments.join(tmp, how='left')
        k += 1

    act_instruments = act_instruments.rename(columns={'<CLOSE>' : shares_name[0]})
    act_instruments = act_instruments.fillna(act_instruments.mean(axis=0))
        
    act_risks = pd.read_csv(indexes[0], index_col='<DATE>').drop(['<TICKER>', '<PER>', '<TIME>', '<HIGH>', '<LOW>', '<VOL>', '<OPEN>'], axis=1).rename(columns={'<CLOSE>':indexes_name[0]})
    act_risks.index = pd.to_datetime(act_risks.index, dayfirst=True) 

    k = 1
    for i in indexes[1:]:
        tmp = pd.read_csv(i, index_col='<DATE>') \
                .drop(['<TICKER>', '<PER>', '<TIME>', '<HIGH>', '<LOW>', '<VOL>', '<OPEN>'], axis=1) \
                .rename(columns={'<CLOSE>' : indexes_name[k]})
        tmp.index = pd.to_datetime(tmp.index, dayfirst=True)
        act_risks = act_risks.join(tmp, how='left')
        k += 1

    k = 0
    for i in currencies:
        tmp = pd.read_csv(i, index_col='Date') \
                .drop(['Open', 'High', 'Low', 'Change %'], axis=1) \
                .rename(columns={'Price' : currencies_name[k]})
        tmp.index = pd.to_datetime(tmp.index, dayfirst=True)
        act_risks = act_risks.join(tmp, how='left')
        k += 1
    
    zero_bond_df = pd.read_csv(zero_bond, sep=';', index_col='Date')
    zero_bond_df.index = pd.to_datetime(zero_bond_df.index, dayfirst=True)
    act_risks = act_risks.join(zero_bond_df, how='left')
    act_risks = act_risks.fillna(act_risks.mean(axis=0))

    tmp = act_risks.diff()
    temp = act_risks.iloc[:-1,:]
    temp.index = tmp.iloc[1:,:].index
    r_risks = tmp.iloc[1:,:] / temp
    
    tmp = act_instruments.diff()
    temp = act_instruments.iloc[:-1,:]
    temp.index = tmp.iloc[1:,:].index
    r_instruments = tmp.iloc[1:,:] / temp

    # act_risks = pd.DataFrame() # реальные значения риск-факторов
    # act_instruments = pd.DataFrame() # реальные значения инструментов
    # r_risks = pd.DataFrame() # Риск-факторы, посчитанные в арифметических процентах
    # r_instruments = pd.DataFrame() # доходность инструментов, посчитанная в арифметических процентах

    return    r_risks, r_instruments, act_risks, act_instruments
# all values are diffed already, first values are original (we can cumsum to original series!)

def stoch_wrapper(df_of_risks):
    decomp = get_decomp(df_of_risks)
    def make_stoch(num):
        # sigma=[0.03, 0.0093, 0.11]
        stoch_generator = np.dot(np.random.normal(size=(num,RISK_FACTORS_NUM)),decomp)#*sigma
        return stoch_generator
    return make_stoch


def params_for_gbm(data):
    mu = np.mean(data)
    sigma_sq = (-1 + np.sqrt(1 + ((data - mu) ** 2).sum() / data.shape[0])) / 0.5
    drift = mu - 0.5 * sigma_sq

    return mu, sigma_sq,drift


def generate_gbm_sim(init_array, pred_param, stochs,dt, time_steps):
    #generates one complete simulaiton
    mu, sigma_sq, drift = pred_param
    sim_res = np.zeros(shape=(time_steps+1, RISK_FACTORS_NUM))
    sim_res[0,:] = init_array
    # stochs - np array of (tsteps, factors)
    for t in range(time_steps):

        sim_res[t+1] = sim_res[t] + drift * dt + np.sqrt(sigma_sq) * stochs[t]

    return sim_res[1:]

# Загнал датафрейм, получил графички данных его колонок
def plot_df(df):
    plt.figure(figsize = (16, 40))

    for num, col in enumerate(df.columns):
        plt.subplot(len(df.columns) // 2 + 1, 2, num + 1)
        plt.plot(df[col])
        plt.title(col)
    plt.show()

    plt.figure(figsize = (16, 40))

    for num, col in enumerate(df.columns):
        plt.subplot(len(df.columns) // 2 + 1, 2, num + 1)
        plt.hist(df[col])
        plt.title(col)
    plt.show()

# Нарисовать симуляцию
def plot_sim(simulation):
    plt.figure(figsize=(16,10))
    plt.plot(simulation)
    plt.show()



def calculate_var(prices):# array of shape (instruments_num, simulations_num)
    VaRs = np.percentile(prices, axis=1, q=5)
    
    return VaRs

def calculate_es(prices):# array of shape (instruments_num, simulations_num)
    ESes = [prices[i][prices[i]<np.percentile(prices[i], q=2.5)].mean() for i in range(prices.shape[0]) ]
    return ESes

def calculate_risks(
    instruments_values #array of shape (tsteps,instruments_num,simulations_num, ) 
    ):
    # es and var, 1 and 10 days
    original_prices = np.array(PRICE_OF_INSTRUMENTS,dtype=float)
    original_prices_broadcasted = np.broadcast_to(original_prices.reshape(-1,1), instruments_values.shape[1:])
    prices = original_prices_broadcasted.copy()

    for i in range(instruments_values.shape[0]):
        #move one day forward
        cur_movements = instruments_values[i,:,:]
        prices +=np.multiply(prices, cur_movements)
        if i==0:
            var0=calculate_var(prices)
            es0 = calculate_es(prices)
        
        #rebalance
        prices = np.multiply((prices.sum(axis=0)/original_prices.sum()), prices)

    var10=calculate_var(prices)
    es10=calculate_var(prices)
    return var0,es0,var10,es10



def main():
    timesteps=10
    dt = 1/247
    simulations_num=SIM_NUM
    r_risks, r_instruments, act_risks, act_instruments = get_data()
    logger.info('Data is fetched')
    
    stoch_gen = stoch_wrapper(r_risks)
    models = train_models(r_risks, r_instruments) #мы ограничиваем трейн?
    
    calculated_risks=dict()
    for it in range(max(RISKS_LOOKBACK, INSTRUMENTS_LOOKBACK,1), r_risks.shape[0]-timesteps):
        if it%100==0:
            logger.info(f'iteration {it} is in')
        local_pivot = max(RISKS_LOOKBACK, INSTRUMENTS_LOOKBACK)
        time = r_risks.index[it]
        risk_data = r_risks.iloc[       it-local_pivot:it+timesteps].values
        instr_data = r_instruments.iloc[it-local_pivot:it+timesteps]#what for?
        init_ins = act_instruments.iloc[  INSTRUMENTS_LOOKBACK+it,:].values.reshape(-1)
        instr_results =        np.zeros(shape = ( timesteps, instr_data.shape[1], simulations_num)) #Сюда запишем результат предсказаний по инструментам
        risk_factors_history = np.zeros(shape = ( timesteps, RISK_FACTORS_NUM,    simulations_num)) #Сюда запишем историю симуляции рисков

        #estimation of params for risk sim
        history_to_estimate_params = r_risks.iloc[local_pivot-RISKS_LOOKBACK:-timesteps]
        init_array = r_risks.iloc[-timesteps]

        gbm_params = params_for_gbm(history_to_estimate_params)
        for sim_id in range(simulations_num):
            stochs = stoch_gen(timesteps)

            sim = generate_gbm_sim(init_array, gbm_params, stochs, dt, timesteps)
            risk_factors_history[:,:,sim_id] = sim


        instruments_names = instr_data.columns

        broadcasted_lookback_history = np.stack([risk_data[:INSTRUMENTS_LOOKBACK] for _ in range(simulations_num)], axis=-1)
        # broadcasted_lookback_history = np.broadcast_to(risk_data[:INSTRUMENTS_LOOKBACK],(INSTRUMENTS_LOOKBACK,RISK_FACTORS_NUM,simulations_num))
        appended_history = np.concatenate((broadcasted_lookback_history,risk_factors_history),axis=0)


        simulated_instruments = np.stack([inference_models(appended_history[:,:,sim_id], models)for sim_id in range(simulations_num)], axis=-1)#array of shape (tsteps,instruments_num,simulations_num, )


        risks = calculate_risks(simulated_instruments)
        calculated_risks[time] = risks
    

    var1={k:v[0] for k,v in calculated_risks.items()}
    es1={k:v[1] for k,v in calculated_risks.items()}
    var10={k:v[2] for k,v in calculated_risks.items()}
    es10={k:v[3] for k,v in calculated_risks.items()}

    return var1,es1,var10,es10

 









"""

Халл-уайт из слупов - для референса

def simulate_hull_white(
    sim_number = 10,):
    rub_alpha=0.03
    usd_alpha=0.02
    sigma=[0.03, 0.0093, 0.11]
    k_fx=0.015
    dt=14/365
    timesteps = 26

    (
        curve_rub,
        curve_usd,
        curve_fx,
        curve_rub_df,
        curve_usd_df,
        curve_fx_df,
        init
        ) = get_rates()


    results = np.zeros(shape=(timesteps+1, 3, sim_number))

    passed_time=0

    for sim_ix in range(sim_number):
        results[0,:,sim_ix] = init
        stochs = stoch_generator(timesteps+1)
        for i, (rate_rub, rate_usd, rate_fx,df_rub, df_usd,df_fx, stoch_tuple) in enumerate(zip(curve_rub,curve_usd,curve_fx,curve_rub_df,curve_usd_df, curve_fx_df, stochs)):
            passed_time+=dt

            theta_rub = df_rub + rub_alpha*rate_rub + (sigma[0]**2)*(1-np.exp(-2*rub_alpha*passed_time))/2*rub_alpha
            theta_usd = df_usd + usd_alpha*rate_usd + (sigma[1]**2)*(1-np.exp(-2*usd_alpha*passed_time))/2*usd_alpha

            results[i+1,0,sim_ix] = (theta_rub - rub_alpha* results[:,0,sim_ix].sum())*dt+stoch_tuple[0]
            results[i+1,1,sim_ix] = (theta_usd - usd_alpha* results[:,1,sim_ix].sum())*dt+stoch_tuple[1]
            results[i+1,2,sim_ix] = k_fx*(rate_fx - np.log( results[:,2,sim_ix].sum()))*dt+stoch_tuple[2]

    return results


def perform_simulations_basic_mode(sim_number=1000):
    results = simulate_hull_white(sim_number=sim_number)
    # results - np array of shape(no of timesteps, no of instruments, no of simulations)
    plot_results(results)
    return results
"""
