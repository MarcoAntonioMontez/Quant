import warnings
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# import plotly
# from plotly.offline import iplot
# import plotly.graph_objs as go

# plotly.offline.init_notebook_mode()
from modules import logs_manager as logs
from modules import data_manager as dm
from modules import technical_manager as tm
from modules import visualization_manager as vm
from modules.UserInput import UserInput
from modules.Trader import Trader
from modules.Order import Order
from modules.Statistics import Statistics
from modules import ga
from modules import logs_manager as logs
import json

pd.options.display.max_rows = 50

data_path = '../data/'
# prices = '21_sample_2005_2016.csv'
prices = 'formated_prices_2005-01-01_2018-12-31.csv'
processed_prices = 'pre_processed_prices.csv'
price_field = 'Close'
sp500_name = 'SP500_index_prices_2005-01-01_2018-12-31.csv'
fundamentals = 'fundamental_2002_3.csv'

sp500 = pd.read_csv(data_path + sp500_name, index_col=0)

update_fields = False

if update_fields:
    #     prices = '21_sample_2005_2016.csv'
    dataset = dm.load_csv(data_path + prices, 1)
    ratios = [{'ratio_name': 'ema', 'parameter': 200},
              {'ratio_name': 'atr', 'parameter': 20},
              {'ratio_name': 'cmf', 'parameter': 20},
              {'ratio_name': 'cmo', 'parameter': 14},
              {'ratio_name': 'mfi', 'parameter': 14}
              ]
    #     ratio_names = ['ema','sma']
    #     periods = [20,30,50,100,200]
    #     tm.add_ma(ratios,ratio_names,periods)
    dataset = tm.preprocess_table(dataset, ratios, price_field)
    dataset.to_csv(data_path + processed_prices)
else:
    dataset = dm.load_csv(data_path + processed_prices, 1)

fundamental = dm.load_csv(data_path + fundamentals, 0)

# print(fundamental.head())


td_filename = '../data' + '/screened_tickers.json'

with open(td_filename) as infile:
    screened_tickers_original = json.load(infile)

screened_tickers_original = {int(key):screened_tickers_original[key] for key in screened_tickers_original}
tickers_dict = tm.filter_n_largest(fundamental,screened_tickers_original,15)

# print(tickers_dict_original)
# print(tickers_dict)

tickers = tickers_dict

testing_range = range(2010, 2010 + 1)
training_period = 1

trader_params = {'start_date': '2007-1-1',
                 'end_date': '2018-12-31',
                 'tickers': tickers,
                 'chromosome_list': []
                 }

ga_params = {'pop_size': 20,
             'ga_runs': 50,  # number of iterations
             'ga_reps': 1,  # number of independent simulations
             'hyper_mutation': True
             }



for iter in range(0,5):
    train_results = []
    test_chromosome_list = []
    ga_history = {}
    for year in testing_range:
        start_train_year = year - training_period
        end_train_year = year - 1
        trader_params['start_date'] = str(start_train_year) + '-1-1'
        trader_params['end_date'] = str(end_train_year) + '-12-31'
        print('\n\nYear ' + str(year) + '\n')
        data_params = ga.main(dataset, trader_params, ga_params)

        train_dict = {}
        train_dict['start_date'] = trader_params['start_date']
        train_dict['end_date'] = trader_params['end_date']
        train_dict['train_roi'] = data_params['best_roi']
        train_dict['best_chromosome'] = data_params['best_chromosome']

        test_trader_params = trader_params.copy()
        test_trader_params['start_date'] = str(year) + '-1-1'
        test_trader_params['end_date'] = str(year) + '-12-31'

        chromosome_ex = data_params['best_chromosome']

        test_data = ga.simulate(dataset, test_trader_params, chromosome_ex)
        train_dict['test_roi'] = test_data['roi']
        train_dict['test_year'] = year

        ga_history[str(year-training_period) + ':'+ str(year-1)] = data_params['ga_history']

        chromosome_dict = {'year': year,
                           'chromosome': chromosome_ex}

        test_chromosome_list.append(chromosome_dict)
        train_results.append(train_dict)

    log_path = logs.save_trader_logs(test_chromosome_list, train_results, ga_history, 'sim_hyper' + str(iter))
# log = logs.get_trader_logs(fullpath=log_path)
# print(log['test_chromosome_list'])
# print(log['train_results'])
# print(log['ga_history'])
