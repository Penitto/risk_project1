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




#TODO get_data() - написать подгрузку данных (по сути - адаптировать имеющуюся в ноутбуке)
#TODO calculate_var(), calculate_es() - написать калькулятор рисков (можно как для отдельных инструментов, так и для подгрупп в портфеле). формат входа обозначен, формат выхода - какой будет удобнее в дальнейшем
#TODO simulate_risk_factors_once() - написать симулятор для риск-факторов. Функция должна делать один прогон по всем риск-факторам
#TODO написать бэктестинг - прогон и оценку оригинальных данных по вычисленным рискам







def get_logger():

    logger = logging.getLogger('base')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'./{datetime.datetime.ctime(datetime.datetime.now())}.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


#Глобальные параметры
RISK_FACTORS_NUM=4
NUM_OF_INSTRUMENTS=17
PRICE_OF_INSTRUMENTS = [10**6 for _ in range(10)]+[10**7 for _ in range(5)]+[10**7 for _ in range(2)]
LOOKBACK=0
logger=get_logger()


def get_model():
    return LinearRegression() #можем поменять модельку здесь


def train_models(risks_df, instruments_df):
    logger.info('Model training started')
    models = []
    X = np.stack([risks_df.values[i:i+LOOKBACK].reshape(-1) for i in range(risks_df.shape[0]-LOOKBACK)])
    for ins in range(NUM_OF_INSTRUMENTS):
        y_ = instruments_df.values[LOOKBACK:,ins]
        lr = get_model()
        lr.fit(X,y_)
        models.append(lr)
    logger.info('Model training is done')
    return models

def inference_models(risks_df, models):
    X = np.stack([risks_df.values[i:i+LOOKBACK].reshape(-1) for i in range(risks_df.shape[0]-LOOKBACK)])
    pred = np.stack([model.predict(X) for model in models], axis=1)
    return pred




def get_decomp(df_of_risks):
    merged_interpolated_diff_corr = df_of_risks.iloc[1:].corr()
    decomposed = linalg.cholesky(merged_interpolated_diff_corr)
    logger.info('Decomposition calculated')
    return decomposed




def get_data():
    #Здесь - подгрузка и подготовка данных



    r_risks=pd.DataFrame()#Риск-факторы, посчитанные в арифметических процентах
    r_instruments=pd.DataFrame()#доходность инструментов, посчитанная в арифметических процентах
    act_risks=pd.DataFrame()# реальные значения риск-факторов
    act_instruments=pd.DataFrame()# реальные значения инструментов


    return    r_risks, r_instruments, act_risks, act_instruments
# all values are diffed already, first values are original (we can cumsum to original series!)

def stoch_wrapper(df_of_risks):
    decomp = get_decomp(df_of_risks)
    def make_stoch(num):
        # sigma=[0.03, 0.0093, 0.11]
        stoch_generator = np.dot(np.random.normal(size=(num,RISK_FACTORS_NUM)),decomp)#*sigma
        return stoch_generator
    return make_stoch


def simulate_risk_factors_once(
    df_of_risks,
    stoch_gen,
    timesteps,
    dt,):
    stoch = stoch_gen(timesteps)
    res = np.zeros(shape=(RISK_FACTORS_NUM, timesteps))

    # LOOKBACKS # dont forget to use!

    #generate risks once here
    return res



def calculate_var(prices):# array of shape (instruments_num, simulations_num)
    return 0
def calculate_es(prices):# array of shape (instruments_num, simulations_num)
    return 0

def calculate_risks(
    instruments_values #array of shape (tsteps,instruments_num,simulations_num, ) 
    ):
    # es and var, 1 and 10 days
    original_prices = np.array(PRICE_OF_INSTRUMENTS)
    original_prices_broadcasted = np.broadcast_to(original_prices, instruments_values.shape[1:])
    prices = original_prices_broadcasted.copy(deep=True)

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
    simulations_num=10
    r_risks, r_instruments, act_risks, act_instruments = get_data()
    logger.info('Data is fetched')
    stoch_gen = stoch_wrapper(r_risks)
    models = train_models(r_risks, r_instruments) #мы ограничиваем трейн?
    
    calculated_risks=dict()
    for it in range(max(LOOKBACK,1), r_risks.shape[0]-timesteps):
        if it%100==0:
            logger.info(f'iteration {it} is in')

        time = r_risks.index[LOOKBACK+it]
        risk_data = r_risks.iloc[       it-LOOKBACK:it+timesteps].values
        instr_data = r_instruments.iloc[it-LOOKBACK:it+timesteps]#what for?
        init_ins = act_instruments.iloc[                 LOOKBACK+it,:].values.reshape(-1)
        instr_results =        np.zeros(size = ( timesteps, instr_data.shape[1], simulations_num)) #Сюда запишем результат предсказаний по инструментам
        risk_factors_history = np.zeros(size = ( timesteps, RISK_FACTORS_NUM,    simulations_num)) #Сюда запишем историю симуляции рисков
        for sim_id in range(simulations_num):
            sim = simulate_risk_factors_once(
                risk_data[-timesteps:],
                stoch_gen,
                timesteps,
                dt)
            risk_factors_history[:,:,sim_id] = sim


        instruments_names = instr_data.columns
        broadcasted_lookback_history = np.broadcast_to(risk_data.iloc[:LOOKBACK],(LOOKBACK,RISK_FACTORS_NUM,simulations_num))
        appended_history = np.concatenate((broadcasted_lookback_history,risk_factors_history),axis=0)


        simulated_instruments = np.stack([inference_models(appended_history[:,:,sim_id], models)for sim_id in range(simulations_num)], axis=-1)#array of shape (tsteps,instruments_num,simulations_num, )


        risks = calculate_risks(simulated_instruments)
        calculated_risks[time] = risks

    return calculated_risks

 









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
