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

RISK_FACTORS_NUM=4
NUM_OF_INSTRUMENTS=17
# ORDER_OF_INSTRUMENTS
LOOKBACK_PERIOD_FOR_INSTRUMENTS=0



def train_model(df_of_risks,instruments_df,base_model,grid_params):
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
    merged_interpolated_diff_corr = df_of_risks.diff().iloc[1:].corr()
    decomposed = linalg.cholesky(merged_interpolated_diff_corr)
    return decomposed






def stoch_wrapper(df_of_risks):
    decomp = get_decomp(df_of_risks)
    def make_stoch(num):
        # sigma=[0.03, 0.0093, 0.11]
        stoch_generator = np.dot(np.random.normal(size=(num,RISK_FACTORS_NUM)),decomp)#*sigma
        return stoch_generator
    return make_stoch


stoch_generator = stoch_wrapper(get_decomp())



def make_step(vector_of_current_vals, vector_of_real_vals, **kwargs):
    pass

def main():
    timesteps=10
    dt = 1/247
    simulations_num=10
    df_of_risks, df_of_instruments = get_data()
    stoch_gen = stoch_wrapper(df_of_risks)
    
    models = find_best_model(df_of_risks, instruments_df)




    for it in range(LOOKBACK_PERIOD_FOR_INSTRUMENTS, df_of_risks.shape[0]-timesteps):
        risk_data = df_of_risks.iloc[it-LOOKBACK_PERIOD_FOR_INSTRUMENTS:it+timesteps]
        instr_data = df_of_instruments.iloc[it-LOOKBACK_PERIOD_FOR_INSTRUMENTS:it+timesteps]
        init = risk_data.iloc[LOOKBACK_PERIOD_FOR_INSTRUMENTS,:].values.reshape(-1)

        instr_results = np.zeros(size = (instr_data.shape[1], timesteps, simulations_num))
        risk_factors_history = np.zeros(size = (RISK_FACTORS_NUM, timesteps, simulations_num))
        #TODO put init values in results\history if needed
        for sim_id in range(simulations_num):
            for ts in range(1,timesteps):
                risk_factors_increment = make_step(risk_factors_history[:,ts-1,sim_id], risk_data.iloc[ts+LOOKBACK_PERIOD_FOR_INSTRUMENTS])
                risk_factors_history[:,ts,sim_id] = risk_factors_increment
                #generate risks here
        for sim_id in range(simulations_num):
            #TODO calculate instrunents price here, using  "models"
        #TODO calculate risks here











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
