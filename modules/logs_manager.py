import json
import pickle
import datetime
import os
import pandas as pd
import numpy as np
from modules.Statistics import Statistics
from modules import visualization_manager as vm

import json
import pickle
import datetime
from collections import OrderedDict


def create_sim_folder(folder_name):
    path = '../data/sim_logs/'
    today = datetime.datetime.now()
    date = today.strftime(" %Y-%m-%d %H-%M-%S")
    folder_path = path + folder_name + str(date)
    os.makedirs(folder_path)
    #os.mkdir(folder_path)
    return folder_path


def save_trader_logs(master_genes,trader,best_chromossome,fitness, ga_simulations, folder_name):
    path = create_sim_folder(folder_name)

    dictionary = trader.user_input.inputs
    # holdings = trader.get_holdings()

    log_dict = OrderedDict()
    log_dict['roi'] = fitness
    log_dict['best_chromossome'] = best_chromossome.tolist()
    # log_dict['trader_dict'] = dictionary
    log_dict['master_genes'] = master_genes

    trader_params = dictionary

    results = path + '/results.json'
    trader_dict = path + '/trader_dict.json'
    ga_sim = path + '/ga_history.json'

    with open(results, 'w') as outfile:
        json.dump(log_dict, outfile)

    with open(trader_dict, 'w') as outfile:
        json.dump(trader_params, outfile)

    with open(ga_sim, 'w') as outfile:
        json.dump(ga_simulations, outfile)
    return path


def get_trader_logs(foldername=None, dirpath=None, fullpath=None):
    if foldername is None and fullpath is None:
        raise Exception('foldername or fullpath parameters were not entered')
    if dirpath is None:
        dirpath = '../data/sim_logs/'

    if fullpath is not None:
        path = fullpath
    else:
        path = dirpath + foldername

    results = path + '/results.json'

    with open(results) as infile:
        results = json.load(infile)


    d = OrderedDict()
    d['roi']=results['roi']
    d['best_chromossome'] = results['best_chromossome']
    d['trader_dictionary'] = results['trader_dict']
    d['master_genes'] = results['master_genes']
    return d
