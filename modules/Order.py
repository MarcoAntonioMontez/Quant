import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
from modules.Portfolio import Portfolio
from modules.Strategy import Strategy
from modules import UserInput


class Order:
    def __init__(self, trader, stock, shares=None, position = None):
        self.trader = trader
        self.portfolio = self.trader.portfolio
        self.user_input = self.trader.user_input
        self.buy_price = None
        self.sell_price = None
        self.scale_out_price = None
        self.max_price = 0
        self.buy_date = None
        self.sell_date = None
        self.scale_out_date = None
        self.stock = stock
        self.buy_shares= None
        self.position = None
        self.take_profit = None
        self.stop_loss = None
        self.scaled_stop_loss = None
        self.trailing_stop = None
        self.exit_type = None
        self.buy_stock(shares, position)

        # missing to implement
        # trailing stop
        # scaling out

    def buy_stock(self, shares=None, position=None):
        if shares is not None:
            self.portfolio.order('buy', self.stock, shares)
            self.update_buy_metrics('shares',shares)
        elif position is not None:
            self.portfolio.order_money('buy', self.stock, position)
            self.update_buy_metrics('position', position)
        else:
            raise Exception('Shares or position argument is missing')

    def sell_stock(self, exit_type = None):
        self.portfolio.sell_stock(self.stock, 'all')
        self.update_sell_metrics(exit_type)

    def update_buy_metrics(self, buy_type, value):
        price = self.portfolio.get_current_price(self.stock)
        self.buy_price = price
        self.buy_date = self.portfolio.get_date()
        self.stop_loss = self.calc_stop_loss(self.buy_price, self.user_input.stop_loss_type, self.user_input.stop_loss_parameter)
        self.take_profit = self.calc_take_profit(self.buy_price, self.user_input.take_profit_type, self.user_input.take_profit_parameter)
        if buy_type == 'shares':
            self.buy_shares = value
            self.position = value * price
        elif buy_type == 'position':
            self.buy_shares = value/price
            self.position = value
        else:
            raise Exception('Invalid buy_type, should be "shares" or "position"')

    def update_sell_metrics(self,exit_type = None):
        price = self.portfolio.get_current_price(self.stock)
        self.sell_price = price
        self.sell_date = self.portfolio.get_date()
        self.exit_type = exit_type

    def calc_stop_loss(self, price, signal_type, parameter):
        if signal_type.startswith('atr'):
            indicator_value = self.portfolio.get_value(self.stock, signal_type)
            if indicator_value is None:
                raise Exception('Invalid indicator_value for signal_type: atr')
            stop_level = price - indicator_value * parameter
        elif signal_type == 'percentage':
            stop_level = price - price * parameter
        else:
            raise Exception('Stop loss signal type (' + str(signal_type) +  ') not supported')
            return
        return stop_level

    def calc_take_profit(self, price, signal_type, parameter):
        if signal_type.startswith('atr'):
            indicator_value = self.portfolio.get_value(self.stock, signal_type)
            if indicator_value is None:
                raise Exception('Invalid indicator_value for signal_type: atr')
            take_profit = price + indicator_value * parameter
        elif signal_type == 'percentage':
            take_profit = price + price * parameter
        else:
            raise Exception('Take profit signal type (' + str(signal_type) +  ') not supported')
            return
        return take_profit

    def update_order(self):
        price = self.portfolio.get_current_price(self.stock)
        self.max_price = max(self.max_price, price)

    def __str__(self):
        # print('\n -- User Input Class -- ')
        print('\nBuy Price: ' + str(self.buy_price))
        print('\nSell price: ' + str(self.sell_price))
        print('\nScale out price: ' + str(self.scale_out_price))
        print('\nMax Price: ' + str(self.max_price))
        print('\nBuy_date: ' + str(self.buy_date))
        print('\nScale out date: ' + str(self.scale_out_date))
        print('\nSell_date: ' + str(self.sell_date))
        print('\nStock: ' + str(self.stock))
        print('\nBuy shares: ' + str(self.buy_shares))
        print('\nPosition: ' + str(self.position))
        print('\nTake_profit: ' + str(self.take_profit))
        print('\nStop Loss: ' + str(self.stop_loss))
        print('\nScaled Stop Loss: ' + str(self.scaled_stop_loss))
        print('\nTrailing Stop: ' + str(self.trailing_stop))
        print('\nExit Type: ' + str(self.exit_type))
