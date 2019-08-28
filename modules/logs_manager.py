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


def save_trader_logs(test_chromosome_list, train_results, ga_history, folder_name):
    path = create_sim_folder(folder_name)

    ga_history_path = path + '/ga_history.pickle'
    test_chromosome_list_path = path + '/test_chromosome_list.pickle'
    train_results_path = path + '/train_results.pickle'

    outfile = open(ga_history_path, 'wb')
    pickle.dump(ga_history, outfile)
    outfile.close()

    outfile = open(test_chromosome_list_path, 'wb')
    pickle.dump(test_chromosome_list, outfile)
    outfile.close()

    outfile = open(train_results_path, 'wb')
    pickle.dump(train_results, outfile)
    outfile.close()

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

    ga_history_path = path + '/ga_history.pickle'
    test_chromosome_list_path = path + '/test_chromosome_list.pickle'
    train_results_path = path + '/train_results.pickle'

    infile = open(ga_history_path, 'rb')
    ga_history = pickle.load(infile)
    infile.close()

    infile = open(test_chromosome_list_path, 'rb')
    test_chromosome_list = pickle.load(infile)
    infile.close()

    infile = open(train_results_path, 'rb')
    train_results = pickle.load(infile)
    infile.close()

    return {'ga_history': ga_history, 'test_chromosome_list' : test_chromosome_list, 'train_results': train_results}
