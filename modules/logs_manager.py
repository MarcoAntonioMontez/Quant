import json
import pickle
from modules.Statistics import Statistics
import datetime
import os
import pandas as pd


def create_sim_folder(folder_name):
    path = '../data/sim_logs/'
    today = datetime.datetime.now()
    date = today.strftime(" %Y-%m-%d %H-%M-%S")
    folder_path = path + folder_name + str(date)
    os.mkdir(folder_path)
    return folder_path


def save_trader_logs(trader, folder_name):
    path = create_sim_folder(folder_name)
    dictionary = trader.user_input.inputs
    holdings = trader.get_holdings()
    orders_log = trader.portfolio.get_orders_log()
    statistics = Statistics(orders_log, holdings).get_dict()

    trader_dictionary_filename = path + '/trader_dictionary.json'
    holdings_filename = path + '/holdings.csv'
    orders_log_filename = path + '/orders_log.obj'
    statistics_filename = path + '/statistics.json'

    with open(trader_dictionary_filename, 'w') as outfile:
        json.dump(dictionary, outfile)
    holdings.to_csv(holdings_filename)
    filehandler = open(orders_log_filename, 'wb')
    pickle.dump(orders_log, filehandler)
    with open(statistics_filename, 'w') as outfile:
        json.dump(statistics, outfile)

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

    trader_dictionary_filename = path + '/trader_dictionary.json'
    holdings_filename = path + '/holdings.csv'
    orders_log_filename = path + '/orders_log.obj'
    statistics_filename = path + '/statistics.json'

    with open(trader_dictionary_filename) as infile:
        trader_dictionary = json.load(infile)

    holdings = pd.read_csv(holdings_filename, index_col=0)

    filehandler = open(orders_log_filename, 'rb')
    orders_log = pickle.load(filehandler)

    with open(statistics_filename) as infile:
        statistics = json.load(infile)

    d = dict()
    d['trader_dictionary'] = trader_dictionary
    d['statistics'] = statistics
    d['orders_log'] = orders_log
    d['holdings'] = holdings
    return d