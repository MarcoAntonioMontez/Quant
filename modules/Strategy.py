import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
from modules import Portfolio as Porfolio


class Strategy:
    def __init__(self, name, params,dataset, tradeable_tickers, portfolio):
        self.name = name
        self.params = params
        self.dataset = dataset.copy()
        self.tradeable_tickers = tradeable_tickers
        self.portfolio = portfolio
        self.available_strategies = ['crossing_averages','crossing_ols','double_crossing_averages','double_ema_ols','double_ema_double_ols']
        if self.name not in self.available_strategies:
            raise Exception('Strategy not available')

        if len(self.tradeable_tickers) == 0:
            raise Exception('Plz choose at least one ticker to trade')

        self.order_list = []

    def simulate_day(self,date,new_tickers = []):
        if new_tickers:
            self.tradeable_tickers = new_tickers
        self.current_date = pd.to_datetime(date, format="%Y-%m-%d", errors='coerce').date()
        self.order_list = []

        if self.name == 'crossing_averages':
            self.crossing_averages(self.params)
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

    def add_sell_order(self, ticker):
        order = {'Date': self.current_date, 'Type': 'sell', 'Stock': ticker }
        self.order_list.append(order)

    def initial_stop_loss(self, ticker, value):
        #value = 0.2 means 20% stop loss
        open_orders = self.portfolio.open_orders
        for order in open_orders:
            if order['stock'] == ticker:
                buy_price = order['price']
                price = dm.get_value(ticker, 'Adj Close', self.current_date, self.dataset, 1)
                loss = (buy_price - price) / buy_price
                if loss >= value:
                    return True
                else:
                    return False
        raise Exception('Initial_stop_loss - Ticker is not in open_orders!! Ticker[' + str(ticker) + '] not available!!')

    def trailing_stop_loss(self, ticker, value):
        #value = 0.2 means 20% stop loss
        open_orders = self.portfolio.open_orders
        for order in open_orders:
            if order['stock'] == ticker:
                max_price = order['max_price']
                price = dm.get_value(ticker, 'Adj Close', self.current_date, self.dataset, 1)
                loss = (max_price - price) / max_price
                if loss >= value:
                    return True
                else:
                    return False
        raise Exception('trailing_stop_loss() - Ticker is not in open_orders!! Ticker[' + str(ticker) + '] not available!!')

    def crossing_averages(self,params):
        ema = 'ema' + str(params['big_ema'])
        indicators = ['Adj Close', ema]
        adj_close = 'Adj Close'
        self.check_indicators(indicators)
        #if fields exist dont exist in data raise exception

        def buy_signal(price, ticker):
            if (price > ema_value):
                return True
            else:
                return False

        def sell_signal(price, ticker):
            if (price < ema_value):
                return True
            else:
                return False

        for ticker in self.tradeable_tickers:
            price = dm.get_value(ticker, adj_close, self.current_date, self.dataset, 1)
            ema_value = dm.get_value(ticker, ema, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)


    def crossing_ols(self,params):
        indicators = ['Adj Close','ema50','ols50']
        adj_close = 'Adj Close'
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
            price = dm.get_value(ticker, adj_close, self.current_date, self.dataset, 1)
            ema_value = dm.get_value(ticker, ema, self.current_date, self.dataset, 1)
            ols_value = dm.get_value(ticker, ols, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)
    def double_crossing_averages(self,params):
        indicators = ['Adj Close','ema100','ema20']
        adj_close = 'Adj Close'
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
            price = dm.get_value(ticker, adj_close, self.current_date, self.dataset, 1)
            big_ema_value = dm.get_value(ticker, big_ema, self.current_date, self.dataset, 1)
            small_ema_value = dm.get_value(ticker, small_ema, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)

    def double_ema_ols(self,params):
        adj_close = 'Adj Close'
        big_ema = 'ema' + str(params['big_ema'])
        small_ema = 'ema' + str(params['small_ema'])
        init_stop_loss_value = params['init_stop_loss']
        trailing_stop_loss_value = params['trailing_stop_loss']
        ols = 'ols' + str(params['ols'])
        ols_error = ols + 'error'
        ols_buy = params['ols_buy']
        ols_error_buy = params['ols_error_buy']
        indicators = [adj_close,big_ema,small_ema,ols,ols_error]
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
            price = dm.get_value(ticker, adj_close, self.current_date, self.dataset, 1)
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
        indicators = ['Adj Close','ema100','ema20','ols100','ols50']
        adj_close = 'Adj Close'
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
            price = dm.get_value(ticker, adj_close, self.current_date, self.dataset, 1)
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

    def cross(self, signal_a,signal_b):
        prev_day =
        today = self.current_date
        signal_a_0 =
        signal_b_0 =






