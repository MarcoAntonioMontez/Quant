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

print(fundamental.head())
