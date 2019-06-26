import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
from modules import Portfolio as Porfolio
from modules import indicators as ind


class Strategy:
    def __init__(self, name, params,dataset, tradeable_tickers, portfolio):
        self.name = name
        self.params = params
        self.dataset = dataset.copy()
        self.tradeable_tickers = tradeable_tickers
        self.portfolio = portfolio
        self.available_strategies = ['crossing_averages','modular_strategy','crossing_ols','double_crossing_averages','double_ema_ols','double_ema_double_ols']
        if self.name not in self.available_strategies:
            raise Exception('Strategy not available')

        if len(self.tradeable_tickers) == 0:
            raise Exception('Plz choose at least one ticker to trade')

        self.order_list = []
        self.price_field = params['close_name']

    def simulate_day(self, date, new_tickers=[]):
        if new_tickers:
            self.tradeable_tickers = new_tickers
        self.current_date = date
        self.order_list = []

        if self.name == 'crossing_averages':
            self.crossing_averages(self.params)
        elif self.name == 'modular_strategy':
            self.modular_strategy(self.params)
        elif self.name == 'crossing_ols':
            self.crossing_ols(self.params)
        elif self.name =='double_crossing_averages':
            self.double_crossing_averages(self.params)
        elif self.name =='double_ema_ols':
            self.double_ema_ols(self.params)
        elif self.name == 'double_ema_double_ols':
            self.double_ema_double_ols(self.params)
        return self.order_list

    def get_name(self):
        return self.name

    def get_tradeable_tickers(self):
        return self.tradeable_tickers

    def get_current_date(self):
        return self.current_date

    def get_info(self):
        print('\nName: ' + self.name)
        print('\nCurrent Date: ' + str(self.current_date))
        print('\nAvailable strategies: ' + str(self.available_strategies))
        print('\nTradeable Tickers: ' + str(self.tradeable_tickers))

    def is_stock_in_portfolio(self, stock):
        stocks = self.portfolio.get_current_stocks()
        if stock in stocks:
            return True
        return False

    def check_indicators(self,indicators):
        stock = self.tradeable_tickers[0]
        single_stock_data = dm.data_company(stock, self.dataset, 1)
        data_fields = list(single_stock_data.columns.values)
        for indicator in indicators:
            if indicator not in data_fields:
                raise Exception('Error, Indicator: ' + str(indicator) + ' not available!')
                return False
        return True

    def add_buy_order(self, ticker):
        order = {'Date': self.current_date, 'Type': 'buy', 'Stock': ticker }
        self.order_list.append(order)

    def add_sell_order(self, ticker, exit_type):
        order = {'Date': self.current_date, 'Type': 'sell', 'Stock': ticker, 'exit_type': exit_type}
        self.order_list.append(order)

    def add_scale_out_order(self, ticker):
        order = {'Date': self.current_date, 'Type': 'scale_out', 'Stock': ticker }
        self.order_list.append(order)

    def stop_loss(self, price, order ):
        if price <= order.stop_loss:
           return True
        else:
            return False

    def take_profit(self, price, order):
        if price >= order.take_profit:
           return True
        else:
            return False

    def initial_stop_loss(self, ticker, value):
        #value = 0.2 means 20% stop loss
        open_orders = self.portfolio.open_orders
        for order in open_orders:
            if order['stock'] == ticker:
                buy_price = order['price']
                price = dm.get_value(ticker, self.price_field, self.current_date, self.dataset, 1)
                loss = (buy_price - price) / buy_price
                if loss >= value:
                    return True
                else:
                    return False
        raise Exception('Initial_stop_loss - Ticker is not in open_orders!! Ticker[' + str(ticker) + '] not available!!')

    def trailing_stop_loss(self, ticker,params):
        open_orders = self.portfolio.open_orders
        value = params['trailing_stop_parameter']
        stop_type = params['trailing_stop_type']
        if stop_type == 'percentage':
            for order in open_orders:
                if order.stock == ticker:
                    max_price = order.max_price
                    price = dm.get_value(ticker, self.price_field, self.current_date, self.dataset, 1)
                    loss = (max_price - price) / max_price
                    if loss >= value:
                        return True
                    else:
                        return False
            raise Exception(
                'trailing_stop_loss() - Ticker is not in open_orders!! Ticker[' + str(ticker) + '] not available!!')
        elif stop_type.startswith('atr'):
            for order in open_orders:
                if order.stock == ticker:
                    atr = dm.get_value(ticker, stop_type, self.current_date, self.dataset, 1)
                    max_price = order.max_price
                    price = dm.get_value(ticker, self.price_field, self.current_date, self.dataset, 1)
                    loss = (max_price - price)
                    if loss >= atr * value:
                        return True
                    else:
                        return False
            raise Exception(
                'trailing_stop_loss() - Ticker is not in open_orders!! Ticker[' + str(ticker) + '] not available!!')
        else:
            raise Exception('Trailing stop type: ' + str(params['trailing_stop_type']) + ' doesnt exit!')


    def crossing_averages(self,params):
        ema = 'ema' + str(params['big_ema'])
        indicators = [self.price_field, ema]
        close = self.price_field
        self.check_indicators(indicators)
        order = None

        def buy_signal(price, ticker):
            if self.cross(ticker,close,ema) == 'up':
                return True
            else:
                return False

        def sell_signal(price, ticker, order):
            sell_dict = {'flag':False, 'exit_type': None}
            below_baseline = price < ema_value
            stop_loss = self.stop_loss(price,order)
            trailing_stop = self.trailing_stop_loss(ticker,params)
            exit_indicator = (self.cross(ticker,close,ema) == 'down')
            if stop_loss or trailing_stop or exit_indicator or below_baseline:
                if stop_loss:
                    sell_dict['exit_type']='stop_loss'
                elif trailing_stop:
                    sell_dict['exit_type'] = 'trailing_stop'
                elif exit_indicator:
                    sell_dict['exit_type'] = 'exit_indicator'
                elif below_baseline:
                    sell_dict['exit_type'] = 'baseline'
                sell_dict['flag'] = True
                return sell_dict
            else:
                return sell_dict

        def scale_out_signal(price, order):
            if order.state() == 'open':
                take_profit = self.take_profit(price, order)
                if take_profit:
                    return True
            else:
                return False


        for ticker in self.tradeable_tickers:
            price = dm.get_value(ticker, close, self.current_date, self.dataset, 1)
            ema_value = dm.get_value(ticker, ema, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                order = self.portfolio.get_open_order(ticker)
                sell_dict = sell_signal(price,ticker,order)
                if sell_dict['flag']:
                    self.add_sell_order(ticker,sell_dict['exit_type'])
                elif scale_out_signal(price, order):
                    self.add_scale_out_order(ticker)


    def modular_strategy(self,params):
        entry_indicator = params['entry_indicator']
        exit_indicator_field = params['exit_indicator'] + str(params['exit_indicator_period'])
        total_buy_limit = params['total_buy_limit']

        close = self.price_field
        order = None

        def buy_signal(price, ticker):
            confirmation_value = dm.get_value(ticker, 'total_score', self.current_date,
                                              self.portfolio.trader.confirmation_indicators_table, 1)

            if confirmation_value >= total_buy_limit:
                confirmation_flag = True
            else:
                confirmation_flag = False

            if ind.indicator_cross(self, ticker, entry_indicator, params) == 'up' and confirmation_flag:
                    # price >= ema_value and \
                    # confirmation_value > confirmation_param:
                return True
            else:
                return False

        def sell_signal(price, ticker, order):
            sell_dict = {'flag':False, 'exit_type': None}
            stop_loss = self.stop_loss(price,order)
            trailing_stop = self.trailing_stop_loss(ticker,params)
            if params['exit_indicator'] != 'None':
                exit_indicator = (self.cross(ticker, close, exit_indicator_field) == 'down')
            else:
                exit_indicator = False
            if stop_loss or trailing_stop or exit_indicator:
                    # exit_indicator or \
                    # below_baseline:
                if stop_loss:
                    sell_dict['exit_type']='stop_loss'
                elif trailing_stop:
                    sell_dict['exit_type'] = 'trailing_stop'
                elif exit_indicator:
                    sell_dict['exit_type'] = 'exit_indicator'
                sell_dict['flag'] = True
                return sell_dict
            else:
                return sell_dict

        def scale_out_signal(price, order):
            if order.state() == 'open':
                take_profit = self.take_profit(price, order)
                if take_profit:
                    return True
            else:
                return False

        for ticker in self.tradeable_tickers:
            price = dm.get_value(ticker, close, self.current_date, self.dataset, 1)
            # exit_indicator_value = dm.get_value(ticker, exit_indicator_field, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                # confirmation_value = dm.get_value(ticker, confirmation_indicator, self.current_date, self.dataset,1)
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            else:
                order = self.portfolio.get_open_order(ticker)
                sell_dict = sell_signal(price,ticker,order)
                if sell_dict['flag']:
                    self.add_sell_order(ticker,sell_dict['exit_type'])
                elif scale_out_signal(price, order):
                    self.add_scale_out_order(ticker)


    def crossing_ols(self,params):
        indicators = [self.price_field,'ema50','ols50']
        close = self.price_field
        ema = 'ema50'
        ols = 'ols50'
        self.check_indicators(indicators)

        #if fields exist dont exist in data raise exception

        def buy_signal(price, ticker):
            if (price > ema_value) and (ols_value > 1):
                return True
            else:
                return False

        def sell_signal(price, ticker):
            if (price < ema_value):
                return True
            else:
                return False

        for ticker in self.tradeable_tickers:
            price = dm.get_value(ticker, close, self.current_date, self.dataset, 1)
            ema_value = dm.get_value(ticker, ema, self.current_date, self.dataset, 1)
            ols_value = dm.get_value(ticker, ols, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)
    def double_crossing_averages(self,params):
        indicators = [self.price_field,'ema100','ema20']
        close = self.price_field
        big_ema = 'ema100'
        small_ema = 'ema20'
        self.check_indicators(indicators)

        #if fields exist dont exist in data raise exception

        def buy_signal(price, ticker):
            if (small_ema_value > big_ema_value):
                return True
            else:
                return False

        def sell_signal(price, ticker):
            if (small_ema_value < big_ema_value):
                return True
            else:
                return False

        for ticker in self.tradeable_tickers:
            price = dm.get_value(ticker, close, self.current_date, self.dataset, 1)
            big_ema_value = dm.get_value(ticker, big_ema, self.current_date, self.dataset, 1)
            small_ema_value = dm.get_value(ticker, small_ema, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)

    def double_ema_ols(self,params):
        close = self.price_field
        big_ema = 'ema' + str(params['big_ema'])
        small_ema = 'ema' + str(params['small_ema'])
        init_stop_loss_value = params['init_stop_loss']
        trailing_stop_loss_value = params['trailing_stop_loss']
        ols = 'ols' + str(params['ols'])
        ols_error = ols + 'error'
        ols_buy = params['ols_buy']
        ols_error_buy = params['ols_error_buy']
        indicators = [close,big_ema,small_ema,ols,ols_error]
        self.check_indicators(indicators)

        #if fields exist dont exist in data raise exception

        def buy_signal(price, ticker):
            if (small_ema_value > big_ema_value) and (ols_value > ols_buy) and (ols_error_value < ols_error_buy):
                return True
            else:
                return False

        def sell_signal(price, ticker):
            if (small_ema_value < big_ema_value) or self.trailing_stop_loss(ticker, trailing_stop_loss_value) or \
                    self.initial_stop_loss(ticker, init_stop_loss_value):
                return True
            else:
                return False

        for ticker in self.tradeable_tickers:
            ols_error_value = dm.get_value(ticker, ols_error, self.current_date, self.dataset, 1)
            price = dm.get_value(ticker, close, self.current_date, self.dataset, 1)
            big_ema_value = dm.get_value(ticker, big_ema, self.current_date, self.dataset, 1)
            small_ema_value = dm.get_value(ticker, small_ema, self.current_date, self.dataset, 1)
            ols_value = dm.get_value(ticker, ols, self.current_date, self.dataset, 1)

            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)

    def double_ema_double_ols(self,params):
        indicators = [self.price_field,'ema100','ema20','ols100','ols50']
        close = self.price_field
        big_ema = 'ema100'
        small_ema = 'ema20'
        big_ols = 'ols100'
        small_ols = 'ols50'
        self.check_indicators(indicators)

        #if fields exist dont exist in data raise exception

        def buy_signal(price, ticker):
            if (small_ema_value > big_ema_value) and (big_ols_value > 1) and (small_ols_value > 1):
                return True
            else:
                return False

        def sell_signal(price, ticker):
            if (small_ema_value < big_ema_value):
                return True
            else:
                return False

        for ticker in self.tradeable_tickers:
            price = dm.get_value(ticker, close, self.current_date, self.dataset, 1)
            big_ema_value = dm.get_value(ticker, big_ema, self.current_date, self.dataset, 1)
            small_ema_value = dm.get_value(ticker, small_ema, self.current_date, self.dataset, 1)
            big_ols_value = dm.get_value(ticker, big_ols, self.current_date, self.dataset, 1)
            small_ols_value = dm.get_value(ticker, small_ols, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)

    def cross(self, ticker, signal_a,signal_b):
        if self.current_date == self.portfolio.start_day:
            return 'no_cross'

        prev_day = self.portfolio.get_prev_day()
        today = self.current_date
        signal_a_0 = self.portfolio.get_value(ticker, signal_a, date = prev_day)
        signal_b_0 = self.portfolio.get_value(ticker, signal_b, date = prev_day)
        signal_a_1 = self.portfolio.get_value(ticker, signal_a, date = today)
        signal_b_1 = self.portfolio.get_value(ticker, signal_b, date = today)

        if (signal_a_1 >= signal_b_1) and (signal_a_0 < signal_b_0):
            return 'up'
        elif (signal_a_0 > signal_b_0) and (signal_a_1 <= signal_b_1):
            return 'down'
        else:
            return 'no_cross'




