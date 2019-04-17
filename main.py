import warnings
import pandas as pd
import numpy as np
import modules.data_manager as dm
# import EconomicData as ed
# # import os
# # cwd = os.getcwd()
# # print(str(cwd))
#
#
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 20
desired_width = 320
pd.set_option('display.width', desired_width)


prices = pd.read_csv('data/aapl_prices.csv', index_col=0)
fundamental = pd.read_csv('data/aapl_fundamental.csv', index_col=0)


df=dm.data_between_dates('2010-01-01','2015-01-01',prices,1)
print(df)

#
# print("Hello world!")
