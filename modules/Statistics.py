import pandas as pd
import datetime
from modules import data_manager as dm
from modules.Portfolio import Portfolio
from modules.Strategy import Strategy
from modules import UserInput
from modules.Order import Order
from modules import technical_manager as tm


class Statistics:
    def __init__(self, orders, holdings):
        self.orders = orders
        self.holdings = holdings
        self.roi_list = None
        self.roi = None
        self.win_rate = None
        self.avg_win = None
        self.avg_loss = None
        self.calculate_statistics()

    def calc_roi_list(self, orders):
        return tm.roi_order_list(orders)

    def calc_roi(self, df):
        start_price = df['_net worth'].iloc[0]
        end_price = df['_net worth'].iloc[-1]
        roi = 100 * (end_price - start_price) / start_price
        self.roi = roi

    def calc_win_rate(self, roi_list):
        self.win_rate = tm.win_rate(roi_list)

    def calc_avg_win_loss(self, roi_list):
        self.avg_win, self.avg_loss = tm.avg_win_loss(roi_list)

    def calculate_statistics(self):
        self.roi_list = self.calc_roi_list(self.orders)
        self.calc_roi(self.holdings)
        self.calc_win_rate(self.roi_list)
        self.calc_avg_win_loss(self.roi_list)
        return

    def get_dict(self):
        d = dict()
        d['roi'] = self.roi
        d['win_rate'] = self.win_rate
        d['avg_win'] = self.avg_win
        d['avg_loss'] = self.avg_loss
        d['roi_list'] = self.roi_list
        return d

    def __str__(self, show_all=False):
        print('Roi: ' + str(self.roi))
        print('Win rate: ' + str(self.win_rate))
        print('Avg wins: ' + str(self.avg_win))
        print('Avg losses: ' + str(self.avg_loss))
        if show_all:
            print('Roi list: ' + str(self.roi_list))
        return

