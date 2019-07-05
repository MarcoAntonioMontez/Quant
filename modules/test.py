import warnings
import pandas as pd
import numpy as np
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from modules import data_manager as dm
from modules import technical_manager as tm
import plotly
# from plotly.offline import iplot
# import plotly.graph_objs as go
# plotly.offline.init_notebook_mode()

data_path = '../data'
prices = '21_sample_2005_2016.csv'

dataset = dm.load_csv(data_path + '/' + prices,1)

print(dataset.head())



# # headers = dataset.columns.get_level_values(0)
# # display(set(headers))
#
# price_field = 'Close'
# ratios = [{'ratio_name':'ema', 'parameter':200},
#           {'ratio_name':'ema', 'parameter':100},
#           {'ratio_name':'ema', 'parameter':50},
#           {'ratio_name':'atr', 'parameter':20}
#          ]
#
# dataset = tm.preprocess_table(dataset, ratios,price_field)


# from modules.UserInput import UserInput
# from modules.Trader import Trader
# from modules.Order import Order
#
# tickers = ['AAPL', 'AKS', 'RF', 'TEX']
# stock_name = 'AAPL'
# tickers = [stock_name]
#
#
# dictionary = {}
# dictionary['start_date'] = '2005-1-1'
# dictionary['end_date'] = '2008-12-31'
# dictionary['initial_capital'] = 10000
# dictionary['tickers'] = tickers
# dictionary['strategy'] = 'crossing_averages'
# dictionary['strategy_params'] = {'big_ema':200,
#                                  'small_ema':20,
#                                  'stop_loss_type':'atr20',
#                                  'stop_loss_parameter':2,
#                                  'take_profit_type':'atr20',
#                                  'take_profit_parameter':10,
#                                  'trailing_stop_type':'atr20',
#                                  'trailing_stop_parameter':4,
#                                  'close_name':price_field
#                                   }
#
# # dictionary['strategy_params'] = {'big_ema':200,
# #                                  'small_ema':20,
# #                                  'stop_loss_type':'percentage',
# #                                  'stop_loss_parameter':0.05,
# #                                  'take_profit_type':'percentage',
# #                                  'take_profit_parameter':10,
# #                                  'trailing_stop_type':'percentage',
# #                                  'trailing_stop_parameter':0.2,
# #                                  'close_name':price_field
# #                                   }
#
# user_input = UserInput(dictionary)
# trader = Trader(dataset,user_input)
#
#
# trader.run_simulation()
# df = trader.portfolio.get_orders_log(stock_name)