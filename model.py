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
import arch
import scipy.stats as ss
import multiprocessing as mp
from lightgbm import LGBMRegressor
#TODO написать бэктестинг - прогон и оценку оригинальных данных по вычисленным рискам



from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline,FeatureUnion,make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit





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
RISK_FACTORS_NUM=6
NUM_OF_INSTRUMENTS=17
PRICE_OF_INSTRUMENTS = [10**6 for _ in range(10)]+[10**7 for _ in range(5)]+[10**7 for _ in range(2)]
PORTFOLIOS_INDEX={'STOCKS':np.arange(10),'BONDS':np.arange(5)+10,'CURRENCY':np.array([15,16]),'ALL':np.arange(17)}
INSTRUMENTS_LOOKBACK=100 #>=0
RISKS_LOOKBACK=10 #>=0
SIM_NUM=500

logger=get_logger()


def get_model():
    # return LinearRegression() #можем поменять модельку здесь
    return Pipeline(steps=[
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('rf', LinearRegression()), 
    ])


def select_models(data, y):
    logger.info(f'Started model selection')

    model_params=[]
    for instrument_ix in range(y.shape[1]):
        pipe = get_model()
        param_grid ={
            'pca__n_components':[1,2,3,4,5],
            #'rf__n_estimators':[5,10,30,50,100]
        }
        gs = GridSearchCV(estimator=pipe, cv=TimeSeriesSplit(n_splits=4), param_grid=param_grid)
        gs.fit(data, y.iloc[:,instrument_ix])
        model_params.append(gs.best_params_)
    logger.info(f'Model selection done')
    return model_params

def train_models(risks_df, instruments_df,model_params=None):
    logger.info(f'Started model training')
    models = []
    if not model_params:
        model_params = select_models(risks_df, instruments_df)
    for i in range(instruments_df.shape[1]):
        model = get_model().set_params(**model_params[i])
        model.fit(risks_df, instruments_df.iloc[:,i])
        models.append(model)
    logger.info(f'Model training done')
    return models
        

# def train_models(risks_df, instruments_df):
#     logger.info('Model training started')
#     models = []
#     X = np.stack([risks_df.values[i:i+INSTRUMENTS_LOOKBACK].reshape(-1) for i in range(risks_df.shape[0]-INSTRUMENTS_LOOKBACK)])
#     for ins in range(NUM_OF_INSTRUMENTS):
#         y_ = instruments_df.values[INSTRUMENTS_LOOKBACK:,ins]
#         lr = get_model()
#         print(X.shape, y_.shape)
#         lr.fit(X,y_)
#         models.append(lr)
#     logger.info('Model training is done')
#     return models

def inference_models(risks_df, models):
    # X = np.stack([risks_df[i:i+INSTRUMENTS_LOOKBACK] for i in range(risks_df.shape[0]-INSTRUMENTS_LOOKBACK)], axis=2)
    pred = np.stack([models[i].predict(risks_df) for i in range(len(models))], axis=1)
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

    r_risks = act_risks.pct_change().dropna(how='any',axis=0)
    
    r_instruments = act_instruments.pct_change().dropna(how='any',axis=0)

    # act_risks = pd.DataFrame() # реальные значения риск-факторов
    # act_instruments = pd.DataFrame() # реальные значения инструментов
    # r_risks = pd.DataFrame() # Риск-факторы, посчитанные в арифметических процентах
    # r_instruments = pd.DataFrame() # доходность инструментов, посчитанная в арифметических процентах

    return    r_risks, r_instruments, act_risks, act_instruments

def stoch_wrapper(df_of_risks):
    decomp = get_decomp(df_of_risks)
    def make_stoch(num):
        stoch_generator = np.dot(np.random.normal(size=(num,RISK_FACTORS_NUM)),decomp)
        return stoch_generator
    return make_stoch


# def params_for_gbm(data):
#     mu = np.mean(data)
#     sigma_sq = (-1 + np.sqrt(1 + ((data - mu) ** 2).sum() / data.shape[0])) / 0.5
#     drift = mu - 0.5 * sigma_sq

#     return mu, sigma_sq,drift


# def generate_gbm_sim(init_array, gbm_params,dt, timesteps, stochs):
#     #generates one complete simulaiton

#     mu, sigma_sq, drift = gbm_params
#     sim_res = np.zeros(shape=(timesteps+1, RISK_FACTORS_NUM))
#     sim_res[0,:] = init_array
#     # stochs - np array of (tsteps, factors)
#     for t in range(timesteps):
#         sim_res[t+1] = sim_res[t] + drift * dt + sigma_sq * stochs[t]
#     return sim_res[1:]


def params_for_gbm(data_act, data_r):
    sigma = np.sqrt(data_r.std())
    mu = data_r.mean()
    determ = mu - sigma ** 2 / 2
    return sigma, mu, determ



def generate_gbm_sim( gbm_params,dt, timesteps, stochs):
    #generates one complete simulaiton

    mu, sigma, determ = gbm_params
    return np.exp(determ.values*dt + stochs*sigma.values)-1




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




def calc_VaR(r, level=0.95, kind='normal'):# get 1d array of values
    if kind=='historical':
        return np.percentile(r, q=level*100)
    elif kind=='normal':
        mu, sigma = np.mean(r), np.std(r)
        return mu + ss.norm.ppf(level) * sigma
    elif kind=='garch':
        K = 100
        mdl = arch.arch_model(r * K).fit(disp='off')
        forecast = mdl.forecast()
        mu, sigma = forecast.mean.values[-1] / K, np.sqrt(forecast.variance.values[-1]) / K
        return mu + ss.norm.ppf(level) * sigma


def calculate_var_and_es(r_array):
    var_alpha=0.01
    es_alpha=0.025

    var = calc_VaR(r_array, level=var_alpha)
    var_for_es = calc_VaR(r_array, level=es_alpha)
    es = r_array[r_array<var_for_es].mean()
    # pdb.set_trace()
    return var, es

def calculate_portfolio_risks(r_array, calculate_individuals=True):
    #shape (instruments, simulation)
    ports = []


    for port_name, port_index in PORTFOLIOS_INDEX.items():
        var,es = calculate_var_and_es(r_array[port_index].sum(axis=0))
        ports.append([port_name, 'es',es])
        ports.append([port_name, 'var',var])
    if calculate_individuals:
        for ix, name in enumerate(instruments_names):
            var,es = calculate_var_and_es(r_array[ix])
            ports.append([name, 'es',es])
            ports.append([name, 'var',var])
    return ports




def calculate_risks(
    instruments_values, #array of shape (tsteps,instruments_num,simulations_num, ) 
    # one_day_history, # init and next day
    # ten_day_history # data from init to the 'timesteps' 
    ):
    # es and var, 1 and 10 days
    original_prices = np.array(PRICE_OF_INSTRUMENTS,dtype=float)
    original_prices_broadcasted = np.broadcast_to(original_prices.reshape(-1,1), instruments_values.shape[1:])
    prices = original_prices_broadcasted.copy()#shape (instruments, simulations)
    for i in range(instruments_values.shape[0]):
        #move one day forward
        cur_movements = instruments_values[i,:,:]
        prices +=np.multiply(prices, cur_movements)
        if i==0:
            risks_for_1 = [['1day', *x] for x in calculate_portfolio_risks((prices-original_prices_broadcasted))]
            
        #rebalance
        prices = np.multiply(original_prices_broadcasted,(prices.sum(axis=0)/original_prices_broadcasted.sum(axis=0)))
        # prices = np.multiply((prices.sum(axis=1)/original_prices.sum(axis=1)), original_prices)
    risks_for_10 = [['10day', *x] for x in calculate_portfolio_risks((prices-original_prices_broadcasted))]
    return risks_for_1+risks_for_10



def backtest_main_loop(
    df_risks,
    r_instruments,):
    back_test_results = []
    for instrument in df_risks['portfolio_name'].unique():
        print(instrument)
        if instrument in r_instruments:
            ix = [list(r_instruments.columns).index(instrument)]
        elif instrument in PORTFOLIOS_INDEX.keys():
            ix = PORTFOLIOS_INDEX[instrument]
        else:
            logger.error(f'{instrument} portfolio hasnt been found')
            print(f'{instrument} portfolio hasnt been found')
            continue
        instrument_data = r_instruments.iloc[:,ix]
        risk_pivot = df_risks.loc[df_risks['portfolio_name']==instrument].pivot_table(index='date',columns='kind_and_horizon',values='value')
        prices = [PRICE_OF_INSTRUMENTS[i] for i in ix]
        risk_results = [[instrument, *x] for x in calculate_hits_for_instrument(risk_pivot, instrument_data, prices)]
        back_test_results.extend(risk_results)
    back_test_results_df = pd.DataFrame(back_test_results, columns=['portfolio','risk_kind', 'hits', 'count'])

    back_test_results_df['two_sided_hyp'] = back_test_results_df.apply(lambda x: ss.binom_test(x['hits'],x['count'],0.01,alternative='two-sided'), axis=1)
    back_test_results_df['conservative_hyp'] = back_test_results_df.apply(lambda x: ss.binom_test(x['hits'],x['count'],0.01,alternative='greater'), axis=1)
    back_test_results_df['proportion'] = back_test_results_df['hits']/back_test_results_df['count']
    return back_test_results_df

def calculate_hits_for_instrument(
    risk_pivot, #pivot table: date index, kind+horizon risk columns, risk values 
    instrument_data, #yields for each item in portfolio on each date, df shape (dates, instruments)
    prices, #base prices for instruments in portfolio
    ):
    # vital point - we dont rebalance during backtest
    risks_results = []
    first_day_data = pd.DataFrame(index=instrument_data.index[:-10])
    ten_day_data   = pd.DataFrame(index=instrument_data.index[:-10])

    inst_r = np.stack([instrument_data.shift(-i).values for i in range(10)], axis=-1)[:-10]
    original_prices = np.broadcast_to(prices, instrument_data.shape)[:-10]
    prices = np.broadcast_to(prices, instrument_data.shape)[:-10]

    for i in range(10):
        prices = prices + prices*instrument_data[:,:,i]
        if i==0:
            for instrument_ix, instrument  in enumerate(instrument_data.columns):
                first_day_data[f'{instrument}'] = prices[:,instrument_ix]
        prices = np.multiply((prices.sum(axis=1)/original_prices.sum(axis=1)).values.reshape(-1,1),original_prices)
    for instrument_ix, instrument  in enumerate(instrument_data.columns):
        ten_day_data[f'{instrument}'] = prices[:,instrument_ix]
    # for instrument_ix, instrument  in enumerate(instrument_data.columns):
    #     inst_df = pd.concat([instrument_data.loc[:,instrument].shift(-i) for i in range(10)], axis=1)
    #     price_ins = np.broadcast_to([prices[instrument_ix]], (inst_df.shape[0]))
    #     for timestep in range(10):
    #         price_ins = price_ins+price_ins*inst_df.iloc[:,timestep]
    #         if timestep==0:
    #             first_day_data[f'{instrument}'] = price_ins
        # ten_day_data[f'{instrument}'] = price_ins
    first_day_data=first_day_data.sum(axis=1)
    first_day_data.name = 'fact'
    var_1day_test = pd.merge(
        risk_pivot['var_1day'],
        first_day_data,
        left_index=True,
        right_index=True,
        how='left'
    )
    var_1day_hits = ((var_1day_test['var_1day']-var_1day_test['fact'])>0).sum()
    var_1day_count = var_1day_test.shape[0]
    risks_results.append(['var_1', var_1day_hits, var_1day_count])
    ten_day_data=ten_day_data.sum(axis=1)
    ten_day_data.name = 'fact'
    var_10day_test = pd.merge(
        risk_pivot['var_10day'],
        ten_day_data,
        left_index=True,
        right_index=True,
        how='left'
    )
    var_10day_hits = ((var_10day_test['var_10day']-var_10day_test['fact'])>0).sum()
    var_10day_count = var_10day_test.shape[0]
    risks_results.append(['var_10', var_10day_hits, var_10day_count])
    return risks_results



if __name__=='__main__':
    timesteps=10
    dt = 1/247
    debug=True
    retrain=True
    simulations_num=SIM_NUM
    r_risks, r_instruments, act_risks, act_instruments = get_data()
    r_risks = r_risks.iloc[:,[1,2,3,4,5,16]]
    act_risks = act_risks.iloc[:,[1,2,3,4,5,16]]
    if debug:
        r_risks=r_risks.iloc[:-750]
        r_instruments=r_instruments.iloc[:-750]
        act_risks=act_risks.iloc[:-750]
        act_instruments=act_instruments.iloc[:-750]


    global_volatiles = r_risks.std().values

    logger.info('Data is fetched')
    decomp = get_decomp(r_risks)
    model_params = select_models(r_risks, r_instruments)
    if not retrain:
        models = train_models(r_risks, r_instruments, model_params) #мы ограничиваем трейн?

    # stoch_gen = stoch_wrapper(r_risks)
    global instruments_names
    instruments_names = r_instruments.columns
    
    simsim = np.array([])

    if retrain:

        models = train_models(r_risks.iloc[:INSTRUMENTS_LOOKBACK], r_instruments.iloc[:INSTRUMENTS_LOOKBACK], model_params) #мы ограничиваем трейн?
    i = 0
    calculated_risks=list()# will be list of format ['date', {'1day','10day}, 'portfolio_name',{'es','var'}, value]
    for it in range(max(RISKS_LOOKBACK, INSTRUMENTS_LOOKBACK,0), r_risks.shape[0]-timesteps):
        if it%50==0:
            logger.info(f'iteration {it} is in')
            if retrain:
                models = train_models(r_risks.iloc[:it], r_instruments.iloc[:it], model_params) #мы ограничиваем трейн?
                logger.info(f'Models retrained')

        local_pivot = max(RISKS_LOOKBACK, INSTRUMENTS_LOOKBACK)
        time = r_risks.index[it]
        risk_data = r_risks.iloc[       it-local_pivot:it+timesteps].values
        # instr_data = r_instruments.iloc[it-local_pivot:it+timesteps]#what for?
        # init_ins = act_instruments.iloc[  INSTRUMENTS_LOOKBACK+it,:].values.reshape(-1)

        risk_factors_history = np.zeros(shape = ( timesteps, RISK_FACTORS_NUM,    simulations_num)) #Сюда запишем историю симуляции рисков

        #estimation of params for risk sim
        history_to_estimate_params_act = act_risks.iloc[local_pivot-RISKS_LOOKBACK:-timesteps]
        history_to_estimate_params_r = r_risks.iloc[local_pivot-RISKS_LOOKBACK:-timesteps]
        # history_to_estimate_params = r_risks.iloc[local_pivot-RISKS_LOOKBACK:-timesteps]
        init_array = r_risks.iloc[-timesteps-1]


        gbm_params = params_for_gbm(history_to_estimate_params_act,history_to_estimate_params_r)



        # def apply_pool():
        #     pool = mp.Pool(6)
        #     result = [pool.apply(generate_gbm_sim, kwds = {'init_array':init_array, 'gbm_params':gbm_params,'dt':dt, 'timesteps':timesteps}) for i in range(simulations_num)]
        #     pool.close()
        #     # pool.join()
        #     risk_factors_history = np.stack(result, axis=-1)
        #     return risk_factors_history
        # risk_factors_history = apply_pool()

        for sim_id in range(simulations_num):
            stochs = np.random.normal(size=(timesteps,RISK_FACTORS_NUM))
            # stochs = np.dot(np.random.normal(size=(timesteps,RISK_FACTORS_NUM), scale=np.sqrt(dt)),decomp)#*global_volatiles
            sim = generate_gbm_sim(gbm_params,dt, timesteps, stochs)
            risk_factors_history[:,:,sim_id] = sim

        simulated_instruments = np.stack([inference_models(risk_factors_history[:,:,sim_id], models)for sim_id in range(simulations_num)], axis=-1)#array of shape (tsteps,instruments_num,simulations_num, )
        risks = calculate_risks(simulated_instruments) #format: list of elements - [{'1day','10day}, 'portfolio_name',{'es','var'}, value]
        calculated_risks.extend([[time, *x] for x in risks])

        if i == 0:
            simsim = np.expand_dims(simulated_instruments, axis=0)
        else:
            simsim = np.hstack([simsim, np.expand_dims(simulated_instruments, axis=0)]) 
            # print(simsim.shape)
            # print(simulated_instruments.shape)
            #simsim = np.hstack([simsim, simulated_instruments])
        i += 1
        #simsim = simsim.reshape((it,*simulated_instruments.shape))
        #print(simulated_instruments.shape)
        ##diagnostics graphs
        if it%10==0:
            for i in range(RISK_FACTORS_NUM):
                real=r_risks.iloc[it-1:it+timesteps,i]
                rdf = pd.DataFrame(risk_factors_history[:,i],index=real.index[1:])
                plt.plot(rdf, alpha=0.1)
                r = plt.plot(real, c='black',)
                plt.title(r_risks.columns[i])
                plt.legend(r, ['real'])
                plt.xlabel('Date')
                plt.xticks(real.index)
                plt.savefig(f'./diag_graphs/{it}_risk_{r_risks.columns[i]}.png')
                plt.clf()
                plt.cla()
                plt.close()
            for i in range(RISK_FACTORS_NUM):
                real=r_instruments.iloc[it-1:it+timesteps,i]
                rdf = pd.DataFrame(simulated_instruments[:,i],index=real.index[1:])
                plt.plot(rdf, alpha=0.1)
                r = plt.plot(real, c='black',)
                plt.title(r_instruments.columns[i])
                plt.legend(r, ['real'])
                plt.xlabel('Date')
                plt.xticks(real.index)
                plt.savefig(f'./diag_graphs/{it}_ins_{r_instruments.columns[i]}.png')
                plt.clf()
                plt.cla()
                plt.close()

    # print(simsim.shape)
    # simsim = simsim.reshape((149 ,10,17,500))
    # for i in range(RISK_FACTORS_NUM):
    #     real=r_instruments.iloc[:,i]
    #     rdf = pd.DataFrame(simsim[:,:,i],index=real.index[1:])
    #     plt.plot(rdf, alpha=0.1)
    #     plt.plot(real, c='black',)
    #     plt.title(r_instruments.columns[i])
    #     plt.savefig(f'./diag_graphs/ins_{r_instruments.columns[i]}.png')
    #     plt.clf()
    #     plt.cla()
    #     plt.close()
    df_risks = pd.DataFrame(calculated_risks, columns=['date','horizon', 'portfolio_name','risk_kind', 'value'])
    df_risks['kind_and_horizon'] = df_risks.apply(lambda x:x['risk_kind']+'_'+x['horizon'], axis=1)

    backtest_results = backtest_main_loop(df_risks, r_instruments)
    logger.info('Script done')

"""

def simulate_hull_white(
    act = pd.DataFrame(),
    r = pd.DataFrame(),
    dt=1/247
):
    timesteps=10
    rub_alpha=0.03
    usd_alpha=0.02
    sigma=[0.03, 0.0093, 0.11]

    results = np.ones(shape=(timesteps+1, 3,))

    passed_time=0
    alphas=None#rate of reversion
    sigma=None#
    # for sim_ix in range(sim_number):
        # results[0,:,sim_ix] = init
    stochs = stoch_generator(timesteps+1)
    for i in range(timesteps):
        passed_time+=dt

        thetas = r[i] + alphas*act[i] + (sigma**2)*(1-np.exp(-2*alphas*passed_time))/2*alphas
        results[i+1,:] = (thetas - alphas* results[0]*(results[1:i]+1).cumprod(axis=0))*dt+stochs[i]-1

        # theta_rub = df_rub + rub_alpha*rate_rub + (sigma[0]**2)*(1-np.exp(-2*rub_alpha*passed_time))/2*rub_alpha
        # theta_usd = df_usd + usd_alpha*rate_usd + (sigma[1]**2)*(1-np.exp(-2*usd_alpha*passed_time))/2*usd_alpha

        # results[i+1,0,sim_ix] = (theta_rub - rub_alpha* results[:,0,sim_ix].sum())*dt+stoch_tuple[0]
        # results[i+1,1,sim_ix] = (theta_usd - usd_alpha* results[:,1,sim_ix].sum())*dt+stoch_tuple[1]
        # results[i+1,2,sim_ix] = k_fx*(rate_fx - np.log( results[:,2,sim_ix].sum()))*dt+stoch_tuple[2]

    return results


def perform_simulations_basic_mode(sim_number=1000):
    results = simulate_hull_white(sim_number=sim_number)
    # results - np array of shape(no of timesteps, no of instruments, no of simulations)
    plot_results(results)
    return results
"""
