import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
from modules.UserInput import UserInput
from modules.Order import Order
from modules.Trader import Trader
import random
from collections import OrderedDict
from modules.Statistics import Statistics


def master_gene(name, id, type, range):
    d = {"name": name,
         "encoding": id,
         "type": type,
         "range": range
         }
    return d


def gene(name, value):
    return {'name': name, 'value': value}


def rand_value(type, range):
    if type == 'int':
        value = random.randint(range[0], range[1])
    elif type == 'float':
        value = random.uniform(range[0], range[1])
    else:
        raise Exception("Type '{}' is not valid".format(str(type)))
    return value


def chromossome(master_genes):
    l = len(master_genes)
    arr = np.empty(l)
    for i in range(0, l):
        g = master_genes[i]
        arr[i] = rand_value(g['type'], g['range'])
    return arr


def init_pop(master_genes, pop_size):
    arr = np.empty([pop_size, len(master_genes)])
    for i in range(0, pop_size):
        arr[i, :] = chromossome(master_genes)
    return np.around(arr, decimals=3)


def decoder(arr_row, master_list):
    l = len(arr_row)
    if l != len(master_list):
        raise Exception("Error! Array length must be the same as gene_list length!")
    decoded_params = OrderedDict()
    for i in range(0, l):
        mgene = master_list[i]
        if mgene['type'] == 'int':
            key = mgene["encoding"][int(arr_row[i])]
            decoded_params[mgene['name']] = key
        else:
            decoded_params[mgene['name']] = arr_row[i]
    return decoded_params


def update_params(trader_params, decoded_chromossome):
    trader_params = trader_params.copy()
    d = trader_params['strategy_params']
    for key in decoded_chromossome.keys():
        if key == "atr_period":
            period = decoded_chromossome[key]
            atr = 'atr' + str(period)
            d['stop_loss_type'] = atr
            d['take_profit_type'] = atr
            d['trailing_stop_type'] = atr
        else:
            d[key] = decoded_chromossome[key]
    return trader_params

def update_first_params(trader_params, decoded_chromossome):
    trader_params = trader_params.copy()
    d = trader_params
    for key in decoded_chromossome.keys():
            d[key] = decoded_chromossome[key]
    return trader_params


def calc_roi(holdings):
    start_price = holdings['_net worth'].iloc[0] + holdings['_transaction costs'].iloc[0]
    end_price = holdings['_net worth'].iloc[-1]
    roi = 100 * (end_price - start_price) / start_price
    return roi


def fitness(trader_params, chromossome, master_genes, dataset):
    decoded_chromossome = decoder(chromossome, master_genes)
    trader_params = update_params(trader_params, decoded_chromossome)
    user_input = UserInput(trader_params)
    trader = Trader(dataset, user_input)
    trader.run_simulation()
    holdings = trader.portfolio.get_holdings()
    roi = calc_roi(holdings)
    return roi


def tournament(population, fitness_array, size, co_winners, n_parents):
    if n_parents % co_winners != 0:
        raise Exception(
            "Error!Number of parents '{}' must be a multiple of the co_winners '{}' of each tournament round".format(
                str(n_parents), str(co_winners)))
    parents = []
    for i in range(0, int(n_parents / co_winners)):
        contenders = np.asarray(random.sample(list(enumerate(fitness_array)), size))
        #         display(contenders)
        args_max = np.argpartition(contenders[:, 1], -co_winners)[-co_winners:]
        #         display(args_max)
        args_pop = contenders[args_max, 0].astype(int)
        #         print('args_pop' + str(args_pop))
        #         chosen_parents = population[args_pop,:]
        #         display(chosen_parents)
        for j in range(0, len(args_pop)):
            parents.append(population[args_pop[j], :])
    return np.array(parents)


def elite_individuals(population, fitness_array, size):
    args = np.argpartition(fitness_array, -size)[-size:]
    return population[args, :]


def random_resetting(value, range_values, prob):
    if prob < random.random():
        return value
    options = list(range(range_values[0], range_values[1] + 1))
    options.remove(value)
    return random.choice(options)


def non_uniform_mutation(value, range_values, prob, sigma, min_step):
    if prob < random.random():
        return value
    while True:
        step = sigma * np.random.normal() * value
        if abs(step) < min_step:
            step = min_step * np.sign(step)
        new_value = value + step
        if new_value >= range_values[0] and new_value <= range_values[1]:
            break
    return new_value


def mutation_gene(value, type, range_values, prob, sigma, min_step):
    if type == 'int':
        return random_resetting(value, range_values, prob)
    else:
        return non_uniform_mutation(value, range_values, prob, sigma, min_step)


def mutation_chromossome(master_genes, chromossome, prob, sigma, min_step):
    l = len(master_genes)
    arr = np.empty(l)
    for i in range(0, l):
        g = master_genes[i]
        arr[i] = mutation_gene(chromossome[i], g['type'], g['range'], prob, sigma, min_step)
    return arr


def mutation_pop(pop, master_genes, prob, sigma, min_step):
    n = pop.shape[0]
    m = pop.shape[1]
    arr = np.empty([n, m])
    for i in range(0, n):
        arr[i, :] = mutation_chromossome(master_genes, pop[i, :], prob, sigma, min_step)
    return arr


def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


def uniform_crossover_single_child(pop):
    args = np.random.randint(pop.shape[0], size=pop.shape[1])
    print(args)
    l = len(args)
    arr = np.empty(l)
    for i in range(0, l):
        arr[i] = pop[args[i], i]
    return arr


def uniform_crossover_multi_child(pop, crossover_rate):
    new_pop = pop.copy()
    if crossover_rate < random.random():
        return new_pop
    disarrange(new_pop, axis=-2)
    return new_pop


def crossover_pop(pop, size_offspring, number_parents_crossover, crossover_rate):
    arr = np.empty([size_offspring, pop.shape[1]])
    if size_offspring % number_parents_crossover != 0:
        raise Exception(
            "Size of offspring '{}' must be divisible by number_parents_crossover'{}' ".format(str(size_offspring), str(
                number_parents_crossover)))
    for i in range(0, int(size_offspring / number_parents_crossover)):
        start = i * number_parents_crossover
        end = start + number_parents_crossover
        random_parents = pop[np.random.choice(pop.shape[0], number_parents_crossover, replace=False), :]
        arr[start:end, :] = uniform_crossover_multi_child(random_parents, crossover_rate)
    return arr


def fitness_pop(pop,trader_params, master_list,dataset):
    fitness_array = np.empty(pop.shape[0])
    for i in range(0, pop.shape[0]):
        roi = fitness(trader_params, pop[i, :], master_list, dataset)
        fitness_array[i] = roi
        print('iteration {} of {}'.format(str(i + 1), str(pop.shape[0])), end="\r")
    return fitness_array


def fitness_stats(fitness_array):
    most_fit = np.max(fitness_array)
    average_fit = np.average(fitness_array)
    print("Most fitt individual: " + str(np.round(most_fit, decimals=3)))
    print("Average fitness: " + str(np.round(average_fit, decimals=3)))
    return most_fit, average_fit


def normalize_array(x):
    """Compute softmax values for each sets of scores in x."""
    abs_sum = np.sum(np.abs(x))
    norm = x / abs_sum
    return norm


def encoder(dictionary):
    encoded = np.array([dictionary[x] for x in dictionary.keys()])
    return encoded


def normalize_weights(pop, weight_names, master_list):
    pop = pop.copy()
    for i in range(0, pop.shape[0]):
        individual = pop[i, :]
        decoded = decoder(individual, master_list)
        weights = np.array([decoded[x] for x in weight_names])
        norm_weigths = normalize_array(weights)
        for j in range(0, len(weight_names)):
            decoded[weight_names[j]] = norm_weigths[j]
        pop[i, :] = (encoder(decoded))
        for k in range(0, len(master_list)):
            gene = master_list[k]
            if gene['type'] == 'int':
                for key, value in gene['encoding'].items():
                    if value == pop[i, k]:
                        pop[i, k] = key
    return np.around(pop, decimals=3)


def create_trader_dict():
    dictionary = {}
    dictionary['start_date'] = '2010-1-1'
    dictionary['end_date'] = '2013-12-31'
    dictionary['initial_capital'] = 10000
    dictionary['tickers'] = []
    dictionary['chromosome_list'] = []
    dictionary['strategy'] = 'modular_strategy'
    dictionary['strategy_params'] = {'big_ema': 200,
                                     'small_ema': 20,
                                     'stop_loss_type': 'atr20',
                                     'stop_loss_parameter': 2.143,
                                     'take_profit_type': 'atr20',
                                     'take_profit_parameter': 200,
                                     'trailing_stop_type': 'atr20',
                                     'trailing_stop_parameter': 4.289,
                                     'close_name': 'Close',
                                     'scale_out_ratio': 0.5,
                                     'entry_indicator': 'aroon_s',
                                     'entry_indicator_period': 1,
                                     'exit_indicator': 'None',  # ssl
                                     'exit_indicator_period': 20,
                                     'buy_limit_vol_1': 0,
                                     'buy_limit_vol_2': 0,
                                     'buy_limit_vol_3': 0,
                                     'volume_total_buy_limit': 0,
                                     'exit_ind_1_param': 14.0,
                                     'exit_ind_2_param': 20.0,
                                     'exit_ind_3_param': 100,
                                     'exit_ind_1': 'aroon_s',
                                     'exit_ind_2': 'ssl_s',  # ssl_line
                                     'exit_ind_3': 'ema_slope',  # sar_line
                                     'volume_ind_1': 'cmf20',
                                     'volume_ind_2': 'cmo14',
                                     'volume_ind_3': 'mfi14',
                                     'weight_exit_1': 0.15,
                                     'weight_exit_2': 0.15,
                                     'weight_exit_3': 0.15,
                                     'weight_vol_1': 0.15,
                                     'weight_vol_2': 0.15,
                                     'weight_vol_3': 0.15,
                                     'confirmation_total_buy_limit': 0.7,
                                     }

    return dictionary
def master_genes_calc():
    f_range = (0.5, 10)
    unit_range = (0, 1)
    period_range = [10, 50]

    master_genes = []

    master_genes.append(master_gene("exit_ind_1_param", 0, 'float', period_range))
    master_genes.append(master_gene("exit_ind_2_param", 0, 'float', period_range))
    master_genes.append(master_gene("exit_ind_3_param", 0, 'float', period_range))
    master_genes.append(master_gene("weight_exit_1", 0, 'float', unit_range))
    master_genes.append(master_gene("weight_exit_2", 0, 'float', unit_range))
    master_genes.append(master_gene("weight_exit_3", 0, 'float', unit_range))
    master_genes.append(master_gene("weight_vol_1", 0, 'float', unit_range))
    master_genes.append(master_gene("weight_vol_2", 0, 'float', unit_range))
    master_genes.append(master_gene("weight_vol_3", 0, 'float', unit_range))

    master_genes.append(master_gene("stop_loss_parameter", 0, 'float', f_range))
    master_genes.append(master_gene("trailing_stop_parameter", 0, 'float', f_range))

    return master_genes

def trader_dict_calc(trader_params):
    trader_dictionary = create_trader_dict()
    trader_dictionary = update_first_params(trader_dictionary, trader_params)
    return trader_dictionary


def simulate(dataset, trader_params,chromossome_ex):
    dictionary = trader_dict_calc(trader_params)
    if not trader_params['chromosome_list']:
        master_genes = master_genes_calc()
        decoded = decoder(chromossome_ex, master_genes)
        trader_params_ex = update_params(dictionary, decoded)
        user_input = UserInput(trader_params_ex)
    else :
        # display('\n\n Chromosome ex not included\n\n')
        user_input = UserInput(dictionary)

    # display(user_input.inputs['strategy_params'])
    trader = Trader(dataset, user_input)

    trader.run_simulation()

    orders_log = trader.portfolio.orders_log
    holdings = trader.portfolio.get_holdings()

    statistics = Statistics(orders_log, holdings)
    stats = statistics.get_dict()

    roi = stats['roi']

    return {'roi': roi,'trader': trader}


def main(dataset, trader_params,ga_params):

    ##
    ##
    ##
    ##See if this makes sense !!!
    trader_params['chromosome_list']=[]

    # trader_dictionary = create_trader_dict()
    trader_dictionary = trader_dict_calc(trader_params)

    user_input = UserInput(trader_dictionary)
    trader = Trader(dataset, user_input)
    ###missing ga update params


    truncated_dataset = trader.dataset
    master_genes = master_genes_calc()

    weight_names = ['weight_vol_1', 'weight_vol_2', 'weight_vol_3']
    exit_names = ['weight_exit_1', 'weight_exit_2', 'weight_exit_3']

    ###GA parameters
    pop_size = ga_params['pop_size']
    tournament_size = 2
    tournament_co_winners = 1
    tour_parents = pop_size / 2
    prob_mutation = 0.05
    sigma = 0.5
    hyper_prob_mutation = 1
    hyper_sigma = 2
    min_step = 0.05
    offspring_size = int(pop_size * 0.8)
    number_parents_crossover = 2
    crossover_rate = 0.9
    elites_size = int(pop_size * 0.2)
    ga_runs = ga_params['ga_runs']
    ga_reps = ga_params['ga_reps']
    hyper_mutation = ga_params['hyper_mutation']

    if pop_size != (offspring_size + elites_size):
        raise Exception("Size of offspring plus size of elites must equal population size")

    ga_simulation_1 = []
    for j in range(0, ga_reps):
        ga_history = []
        print("Simulation: " + str(j + 1))

        pop = init_pop(master_genes, pop_size)
        pop = normalize_weights(pop, weight_names + exit_names, master_genes)
        #     pop = normalize_weights(pop,exit_names,master_genes)
        # display(pop[0,:])

        print("Init: ")
        fitness_array = fitness_pop(pop, trader_dictionary, master_genes, dataset)
        most_fit, average_fit = fitness_stats(fitness_array)
        elites = elite_individuals(pop, fitness_array, elites_size)
        best_elite = elite_individuals(pop, fitness_array, 1)
        print(repr(best_elite))
        ga_history.append((most_fit, average_fit, best_elite))

        for i in range(0, ga_runs):
            print("Iteration: " + str(i + 1))
            selected_parents = tournament(pop, fitness_array, tournament_size, tournament_co_winners, tour_parents)
            if (average_fit > (most_fit*0.95)) and hyper_mutation:
                print('Hyper Mutation')
                mutated = mutation_pop(selected_parents, master_genes, hyper_prob_mutation, hyper_sigma, min_step)
            else:
                mutated = mutation_pop(selected_parents, master_genes, prob_mutation, sigma, min_step)
            crossed = crossover_pop(mutated, offspring_size, number_parents_crossover, crossover_rate)
            pop = np.concatenate((elites, crossed))
            pop = normalize_weights(pop, weight_names + exit_names, master_genes)
            #         pop = normalize_weights(pop,exit_names,master_genes)
            fitness_array = fitness_pop(pop, trader_dictionary, master_genes, dataset)
            most_fit, average_fit = fitness_stats(fitness_array)
            elites = elite_individuals(pop, fitness_array, elites_size)
            best_elite = elite_individuals(pop, fitness_array, 1)
            print(repr(best_elite))
            ga_history.append((most_fit, average_fit, best_elite))

        ga_simulation_1.append(ga_history)
        # best_elite = elite_individuals(pop, fitness_array, 1)
        #     display(best_elite)
        # print(repr(best_elite))

        # logs.save_trader_logs(master_genes, trader, best_elite, most_fit, ga_simulation_1, 'sim')
    return {'master_genes': master_genes, 'trader_dictionary': trader_dictionary, 'best_chromosome': best_elite[0], 'best_roi' : most_fit,'ga_history': ga_history}
