'''

Артефакты:
    широкий список риск-факторов
    узкий список риск-факторов
    котировки инструментов
    Приведенные к одной гранулярности (день) временные ряды



туду:
    Обозначить используемые риск факторы
        составить список, накидать статистики
        провести исследование (PCA, проч)
        получить финальный список факторов
    Сделать модель симуляции риск-факторов
        разложение холецкого для корректной стохастики
        генератор с заданной гранулярностью
    Сделать модель котировок
        сделать инфраструктуру для трейна/теста модельки
        Натренить модель (какую?) каждой котировки на историю риск-факторов
    оценить риск
        сделать оценщик VaR/ES(?) на произвольные входные данные
    Выкупить за перебалансировку портфеля






1. разложить в холецкого риск-факторы
2. инициализировать генератор стохастики
3. для каждого дня в истории:
    1. насимулировать на 10 дней вперед риск-факторы
    2. с применением модели вычислить цену инструментов (для каждой симуляции!)
    3. по распределению цен портфеля посчитать риски (ес, вар)
    4. красиво зафиксировать
'''
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

RISK_FACTORS_NUM=4
NUM_OF_INSTRUMENTS=17
# ORDER_OF_INSTRUMENTS
LOOKBACK_PERIOD_FOR_INSTRUMENTS=0
logger=get_logger()


def train_model(df_of_risks,instruments_df,base_model,grid_params):
    logger.info(f'Fitting {repr(base_model)}')

    X = df_of_risks
    models = {}
    model_scores={}
    for ins_name in instruments_df.columns:
        gs = GridSearchCV(
            estimator=base_model,
            param_grid=grid_params,
            scoring='r2',
            cv=TimeSeriesSplit()
            )
        y = instruments_df.loc[:,ins_name]
        gs.fit(X,y)

        model_scores[ins_name] = gs.best_score_
        models[ins_name] = gs.best_estimator_
    return models, model_scores


def find_best_model(df_of_risks, instruments_df):
    logger.info('Starting best models fit')
    models_specs = [
        [
            LinearRegression(),
            {
                'fit_intercept':[True, False],
            }
        ],
        [
            RandomForestRegressor(),
            {
                'n_estimators':[10],
                'max_depth':[2,3]
            }
        ]
    ]

    models_list = []
    models_scores_list = []
    for model_spec in models_specs:

        models, model_scores = train_model(
            df_of_risks,
            instruments_df,
            model_spec[0],
            model_spec[1]
            )
        models_list.append(models)
        models_scores_list.append(model_scores)
    
    models_df = pd.DataFrame(models_list)
    scores_df = pd.DataFrame(models_scores_list)
    best_models = {ins_name:models_df.loc[scores_df.idxmax()[ins_name], ins_name] for ins_name in models_df.columns}

    return best_models
            





def get_decomp(df_of_risks):
    merged_interpolated_diff_corr = df_of_risks.iloc[1:].corr()
    decomposed = linalg.cholesky(merged_interpolated_diff_corr)
    return decomposed




def get_data():
    df_of_risks, df_of_instruments = None, None
    return df_of_risks, df_of_instruments# all values are diffed already, first values are original (we can cumsum to original series!)

def stoch_wrapper(df_of_risks):
    decomp = get_decomp(df_of_risks)
    def make_stoch(num):
        # sigma=[0.03, 0.0093, 0.11]
        stoch_generator = np.dot(np.random.normal(size=(num,RISK_FACTORS_NUM)),decomp)#*sigma
        return stoch_generator
    return make_stoch


stoch_generator = stoch_wrapper(get_decomp())




def simulate_risk_factors_once(
    df_of_risks,
    stoch_gen,
    timesteps,
    dt,):
    stoch = stoch_generator(timesteps)
    res = np.zeros(shape=(RISK_FACTORS_NUM, timesteps))

    LOOKBACK_PERIOD_FOR_INSTRUMENTS # dont forget to use!

    #generate risks once here
    return res

def calculate_risks(
    instuments_values #array of shape (simulations_num, tsteps,instruments_num) 
    ):
    
    var1=0
    var2=0
    return var1, var2



def main():
    timesteps=10
    dt = 1/247
    simulations_num=10
    df_of_risks, df_of_instruments = get_data()
    stoch_gen = stoch_wrapper(df_of_risks)
    
    models = find_best_model(df_of_risks, df_of_instruments)



    risks_array=[]
    for it in range(LOOKBACK_PERIOD_FOR_INSTRUMENTS, df_of_risks.shape[0]-timesteps):
        risk_data = df_of_risks.iloc[       it-LOOKBACK_PERIOD_FOR_INSTRUMENTS:it+timesteps].values
        instr_data = df_of_instruments.iloc[it-LOOKBACK_PERIOD_FOR_INSTRUMENTS:it+timesteps]
        init = risk_data.iloc[                 LOOKBACK_PERIOD_FOR_INSTRUMENTS,:].values.reshape(-1)
        instr_results =        np.zeros(size = ( timesteps, instr_data.shape[1], simulations_num))
        risk_factors_history = np.zeros(size = ( timesteps, RISK_FACTORS_NUM,    simulations_num))
        #TODO put init values in results\history if needed
        for sim_id in range(simulations_num):
            sim = simulate_risk_factors_once(
                risk_data[-timesteps:],
                stoch_gen,
                timesteps,
                dt)
            risk_factors_history[:,:,sim_id] = sim


        simulated_instruments=[]
        instruments_names = instr_data.columns
        for sim_id in range(simulations_num):
            instrument_history = defaultdict(list)

            required_history = np.concatenate((risk_data[:LOOKBACK_PERIOD_FOR_INSTRUMENTS],risk_factors_history[:,:,sim_id]), axis=0)
            stacks = np.stack([required_history[i:i+LOOKBACK_PERIOD_FOR_INSTRUMENTS] for i in range(required_history.shape[0]-LOOKBACK_PERIOD_FOR_INSTRUMENTS+1)])
            _ = [instrument_history[name].append(model.predict(stack)) for name,model in models.items() for stack in stacks]
            predicted_instruments_1_sim = pd.DataFrame(instrument_history).loc[:,instruments_names].values

            simulated_instruments.append(predicted_instruments_1_sim)

        simulated_instruments=np.stack(simulated_instruments) #array of shape (simulations_num, tsteps,instruments_num)
        risks = calculate_risks(simulated_instruments)
        risks_array.append(risks)
    return risks_array

 










def simulate_hull_white(
    sim_number = 10,):
    # rub_alpha=0.03
    # usd_alpha=0.02
    # sigma=[0.03, 0.0093, 0.11]
    # k_fx=0.015
    dt=1/247
    timesteps = 10


    results = np.zeros(shape=(timesteps+1, 3, sim_number))

    passed_time=0

    for sim_ix in range(sim_number):
        results[0,:,sim_ix] = init
        stochs = stoch_generator(timesteps+1)
        for i, (rate_rub, rate_usd, rate_fx,df_rub, df_usd,df_fx, stoch_tuple) in enumerate(
            zip(curve_rub,curve_usd,curve_fx,curve_rub_df,curve_usd_df, curve_fx_df, stochs)):
            passed_time+=dt

            theta_rub = df_rub + rub_alpha*rate_rub + (sigma[0]**2)*(1-np.exp(-2*rub_alpha*passed_time))/2*rub_alpha
            theta_usd = df_usd + usd_alpha*rate_usd + (sigma[1]**2)*(1-np.exp(-2*usd_alpha*passed_time))/2*usd_alpha

            results[i+1,0,sim_ix] = (theta_rub - rub_alpha* results[:,0,sim_ix].sum())*dt+stoch_tuple[0]
            results[i+1,1,sim_ix] = (theta_usd - usd_alpha* results[:,1,sim_ix].sum())*dt+stoch_tuple[1]
            results[i+1,2,sim_ix] = k_fx*(rate_fx - np.log( results[:,2,sim_ix].sum()))*dt+stoch_tuple[2]

    return results

# if __name__=='__main__':
#     import numpy as np
#     import pandas as pd
#     df_of_risks = pd.DataFrame(np.random.random(size=(200,5)))
#     df_of_instruments = pd.DataFrame(np.random.random(size=(200,3)))
