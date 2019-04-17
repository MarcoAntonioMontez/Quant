import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
from modules import Portfolio as Porfolio


class Strategy:
    def __init__(self, name, dataset, tradeable_tickers, portfolio):
        self.name = name
        self.dataset = dataset.copy()
        self.tradeable_tickers = tradeable_tickers
        self.portfolio = portfolio
        self.available_strategies = ['crossing_averages']
        if self.name not in self.available_strategies:
            NameError('Strategy not available')

        if len(self.tradeable_tickers) == 0:
            NameError('Plz choose at least one ticker to trade')

        self.order_list = []


    def simulate_day(self,date,new_tickers = []):
        if new_tickers:
            self.tradeable_tickers = new_tickers
        self.current_date = pd.to_datetime(date, format="%Y-%m-%d", errors='coerce').date()
        self.order_list = []

        if self.name == 'crossing_averages':
            self.crossing_averages()
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
                NameError('Indicator not available')
                return False
        return True

    def add_buy_order(self, ticker):
        order = {'Date': self.current_date, 'Type': 'buy', 'Stock': ticker }
        self.order_list.append(order)

    def add_sell_order(self, ticker):
        order = {'Date': self.current_date, 'Type': 'sell', 'Stock': ticker }
        self.order_list.append(order)

    def crossing_averages(self):
        indicators = ['Adj Close']
        adj_close = 'Adj Close'

        #if fields exist dont exist in data raise exception

        def buy_signal(price, ticker):
            if (price > 3.07):
                return True
            else:
                return False

        def sell_signal(price, ticker):
            if (price > 3.20):
                return True
            else:
                return False

        for ticker in self.tradeable_tickers:
            price = dm.get_value(ticker, adj_close, self.current_date, self.dataset, 1)
            if not self.is_stock_in_portfolio(ticker):
                if buy_signal(price,ticker):
                    self.add_buy_order(ticker)
            elif self.is_stock_in_portfolio(ticker):
                if sell_signal(price,ticker):
                    self.add_sell_order(ticker)










