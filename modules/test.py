import warnings
import pandas as pd
import numpy as np
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from modules import data_manager as dm
from modules import technical_manager as tm
from modules.UserInput import UserInput
from modules.Trader import Trader
from modules.Order import Order
from modules.Statistics import Statistics
from modules import ga
pd.options.display.max_rows = 200


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        from plotly.offline import iplot
        import plotly.graph_objs as go
        import plotly
        plotly.offline.init_notebook_mode()
        plotly.tools.set_credentials_file(username='marco.montez', api_key='FgZQOnOU1P78yrlx0Vwx')
    except NameError:
        return False

data_path = '../data/'
prices = '21_sample_2005_2016.csv'
processed_prices = 'pre_processed_21_sample.csv'
price_field = 'Close'
sp500_name = 'SP500_index_prices_2005-01-01_2018-12-31.csv'

sp500 = pd.read_csv(data_path + sp500_name,index_col=0)

update_fields = True

if update_fields:
    prices = '21_sample_2005_2016.csv'
    dataset = dm.load_csv(data_path + prices,1)
    ratios = [{'ratio_name':'ema', 'parameter':200},
              {'ratio_name':'atr', 'parameter':20},
              {'ratio_name':'atr', 'parameter':14},
              {'ratio_name':'aroon', 'parameter':20},
              {'ratio_name':'aroon', 'parameter':14},
              {'ratio_name':'rsi', 'parameter':14},
              {'ratio_name':'macd_diff', 'parameter':[12,26,9]},
              {'ratio_name':'cmf', 'parameter':20},
              {'ratio_name':'cmo', 'parameter':14},
              {'ratio_name':'mfi', 'parameter':14}
             ]
    ratio_names = ['ema','sma']
    periods = [20,30,50,100,200]
    tm.add_ma(ratios,ratio_names,periods)
    dataset = tm.preprocess_table(dataset, ratios,price_field)
    dataset.to_csv(data_path + processed_prices)
else:
    dataset = dm.load_csv(data_path + processed_prices,1)

# print(dataset.head())

bad_companies = ['AKS','ETFC','LM', 'RF','OI','CLF','DO']
ok_companies = ['TEX','BBT','MOS','CSCO','BC','FLR','FDX']
good_companies = ['LH','PEP','NEE','EW','AOS','AAPL','NVDA']
all_companies = list(set(bad_companies + ok_companies + good_companies))

# tickers = 'MOS'
stock_name = 'LH'
tickers = ['LH']
# tickers = good_companies


dictionary = {}
dictionary['start_date'] = '2010-1-1'
dictionary['end_date'] = '2011-12-31'
dictionary['initial_capital'] = 10000
dictionary['tickers'] = tickers
dictionary['strategy'] = 'modular_strategy'
dictionary['strategy_params'] = {'big_ema':200,
                                 'small_ema':20,
                                 'stop_loss_type':'atr20',
                                 'stop_loss_parameter':2.143,
                                 'take_profit_type':'atr20',
                                 'take_profit_parameter':4.53,
                                 'trailing_stop_type':'atr20',
                                 'trailing_stop_parameter':4.289,
                                 'close_name':price_field,
                                 'scale_out_ratio': 0.5,
                                 'entry_indicator':'aroon',
                                 'entry_indicator_period':14,
                                 'exit_indicator':'None', #ssl
                                 'exit_indicator_period':20,
                                 'volume_ind_1':'cmf20',
                                 'volume_ind_2':'cmo14',
                                 'volume_ind_3':'mfi14',
                                 'weight_vol_1':0.358,
                                 'weight_vol_2':0.246,
                                 'weight_vol_3':0.395,
                                 'buy_limit_vol_1': 0.177,
                                 'buy_limit_vol_2':0.282,
                                 'buy_limit_vol_3':-0.038,
                                 'volume_total_buy_limit':0.259,
                                 'exit_ind_1':'aroon_s',
                                 'exit_ind_2':'ssl_s', #ssl_line
                                 'exit_ind_3':'ema_slope', #sar_line
                                 'exit_ind_1_param': 14.0,
                                 'exit_ind_2_param': 20.0,
                                 'exit_ind_3_param': 30,
                                 'weight_exit_1':0.33,
                                 'weight_exit_2':0.33,
                                 'weight_exit_3':0.33,
                                 'confirmation_total_buy_limit': 0.7,
                                  }
user_input = UserInput(dictionary)
trader = Trader(dataset,user_input)
truncated_dataset = trader.dataset



values = [14, 20, 25, 30, 50]
encoding_period = dict(zip(range(0, len(values)), values))

values = ['None', 'ssl']
encoding_exit_indicator = dict(zip(range(0, len(values)), values))
exit_range = (0, len(values) - 1)

values = ['None', 'macd_s']
encoding_baseline_type = dict(zip(range(0, len(values)), values))
baseline_range = (0, len(values) - 1)

values = [12, 25]
encoding_baseline_period = dict(zip(range(0, len(values)), values))
baseline_period_range = (0, len(values) - 1)

f_min = 0.5
f_max = 10
i_min = 0
i_max = len(encoding_period) - 1
f_range = (f_min, f_max)
period_range = (i_min, i_max)
unit_range = (0, 1)
double_range = (-0.5, 1.5)
volume_limit_range = (-0.5, 0.5)
total_buy_limit_range = (0, 1)
period_range = [10, 50]

weight_names = ['weight_vol_1', 'weight_vol_2', 'weight_vol_3']
exit_names = ['weight_exit_1', 'weight_exit_2', 'weight_exit_3']

master_genes = []
master_genes.append(ga.master_gene("weight_exit_1", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_exit_2", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_exit_3", 0, 'float', unit_range))
master_genes.append(ga.master_gene("exit_ind_1_param", 0, 'float', period_range))
master_genes.append(ga.master_gene("exit_ind_2_param", 0, 'float', period_range))
master_genes.append(ga.master_gene("exit_ind_3_param", 0, 'float', period_range))
master_genes.append(ga.master_gene("confirmation_total_buy_limit", 0, 'float', double_range))

master_genes.append(ga.master_gene("weight_vol_1", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_vol_2", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_vol_3", 0, 'float', unit_range))
master_genes.append(ga.master_gene("buy_limit_vol_1", 0, 'float', volume_limit_range))
master_genes.append(ga.master_gene("buy_limit_vol_2", 0, 'float', volume_limit_range))
master_genes.append(ga.master_gene("buy_limit_vol_3", 0, 'float', volume_limit_range))
master_genes.append(ga.master_gene("volume_total_buy_limit", 0, 'float', double_range))

master_genes.append(ga.master_gene("stop_loss_parameter", 0, 'float', f_range))
master_genes.append(ga.master_gene("trailing_stop_parameter", 0, 'float', f_range))
master_genes.append(ga.master_gene("take_profit_parameter", 0, 'float', f_range))

###GA parameters
pop_size = 20
tournament_size = 2
tournament_co_winners = 1
tour_parents = pop_size / 2
prob_mutation = 0.05
sigma = 1
min_step = 0.05
offspring_size = int(pop_size * 0.9)
number_parents_crossover = 2
crossover_rate = 0.9
elites_size = int(pop_size * 0.1)
ga_runs = 2
if pop_size != (offspring_size + elites_size):
    raise Exception("Size of offspring plus size of elites must equal population size")

ga_simulation_1 = []
for j in range(0, 1):
    ga_results = []
    print("Simulation: " + str(j + 1))

    pop = ga.init_pop(master_genes, pop_size)
    pop = ga.normalize_weights(pop, weight_names, master_genes)
    pop = ga.normalize_weights(pop, exit_names, master_genes)
    # display(pop[0,:])

    print("Init: ")
    fitness_array = ga.fitness_pop(pop, dictionary, master_genes, truncated_dataset)
    most_fit, average_fit = ga.fitness_stats(fitness_array)
    ga_results.append((most_fit, average_fit))

    for i in range(0, ga_runs):
        print("Iteration: " + str(i + 1))
        elites = ga.elite_individuals(pop, fitness_array, elites_size)
        best_4_elites = ga.elite_individuals(pop, fitness_array, 4)
        print(best_4_elites)
        selected_parents = ga.tournament(pop, fitness_array, tournament_size, tournament_co_winners, tour_parents)
        mutated = ga.mutation_pop(selected_parents, master_genes, prob_mutation, sigma, min_step)
        crossed = ga.crossover_pop(mutated, offspring_size, number_parents_crossover, crossover_rate)
        pop = np.concatenate((elites, crossed))
        pop = ga.normalize_weights(pop, weight_names, master_genes)
        pop = ga.normalize_weights(pop, exit_names, master_genes)
        fitness_array = ga.fitness_pop(pop, dictionary, master_genes, truncated_dataset)
        most_fit, average_fit = ga.fitness_stats(fitness_array)

        ga_results.append((most_fit, average_fit))
    ga_simulation_1.append(ga_results)
    best_4_elites = ga.elite_individuals(pop, fitness_array, 4)
    print(best_4_elites)