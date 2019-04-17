import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
import math

#date = datetime date

class Portfolio:
    def __init__(self, initial_capital, initial_day, dataset):
        self.current_cash = initial_capital
        self.dataset = dataset
        self.current_day = pd.to_datetime(initial_day, format="%Y-%m-%d", errors='coerce').date()
        self.holdings = pd.DataFrame(columns=['Date'])
        self.holdings.set_index('Date', inplace=True)
        self.day_count = 0
        self.holdings.at[self.current_day, 'day count'] = self.day_count
        self.price_field='Adj Close'
        self.transaction_cost=0.01
        self.total_transaction_cost=0

    def get_holdings(self):
        return self.holdings

    def get_cash(self):
        return self.current_cash

    def get_current_price(self, ticker):
        price = dm.get_value(ticker, self.price_field, self.current_day, self.dataset,  1)
        return price

    def get_transaction_costs(self):
        return self.total_transaction_costs

    def next_day_date(self, days=1):
        self.day_count = self.day_count + days
        self.current_day = self.current_day + datetime.timedelta(days=days)
        return self.current_day

    def last_row_date_updated(self, df, date):
        df = df.iloc[-1:].reset_index().copy()
        df['Date'] = date
        df.set_index('Date', inplace=True)
        return df

    def update_day_holdings(self):
        self.holdings.at[self.current_day, 'day count'] = self.day_count
        self.holdings.at[self.current_day, 'transaction costs'] = self.total_transaction_cost
        self.holdings.at[self.current_day, 'cash'] = self.current_cash
        return

    def init_day_holdings(self):
        new_row = self.last_row_date_updated(self.holdings,self.current_day)
        self.holdings = pd.concat([self.holdings, new_row])
        self.update_day_holdings()
        return self.holdings

    def next_day(self,days=1):
        self.next_day_date(days)
        self.init_day_holdings()

    def add_stock(self, stock_ticker, number_shares):
        print('buy')
        if stock_ticker not in self.holdings.columns.values:
            self.holdings.at[self.current_day, stock_ticker] = number_shares
        else:
            self.holdings.at[self.current_day, stock_ticker] = self.holdings.at[self.current_day, stock_ticker] + number_shares


    def remove_stock(self, stock_ticker, number_shares):
        print('buy')
        if stock_ticker not in self.holdings.columns.values:
            print('\nERROR: Stock is not held')
            return False
        elif math.isnan(self.holdings.at[self.current_day, stock_ticker]):
            print('\nERROR:Nan value for number of shares')
            return False
        elif self.holdings.at[self.current_day, stock_ticker] < number_shares:
            print('\nERROR:Not enough shares to sell')
            return False
        else:
            self.holdings.at[self.current_day, stock_ticker] = self.holdings.at[self.current_day, stock_ticker] - number_shares
            return True

    def buy_stock(self, stock_ticker, number_shares):
        stock_price = self.get_current_price(stock_ticker)
        order_cost = stock_price * number_shares
        transaction_cost = order_cost * self.transaction_cost
        total_cost = order_cost + transaction_cost

        if total_cost <= self.current_cash:
            self.current_cash = self.current_cash - total_cost
            self.add_stock(stock_ticker,number_shares)
            self.total_transaction_cost = self.total_transaction_cost + transaction_cost
            return True
        else:
            print('\nNot enough cash to buy order')
            return False

    def sell_stock(self, stock_ticker, number_shares):
        stock_price = self.get_current_price(stock_ticker)
        order_return = stock_price * number_shares
        transaction_cost = order_return * self.transaction_cost
        total_return = order_return - transaction_cost

        if self.remove_stock(stock_ticker,number_shares):
            self.current_cash = self.current_cash + total_return
            self.total_transaction_cost = self.total_transaction_cost + transaction_cost
            return True
        else:
            return False

    def order(self,order_type, stock_ticker, number_share):
        flag = False
        if order_type == 'buy':
            flag = self.buy_stock(stock_ticker,number_share)
        elif order_type == 'sell':
            flag = self.sell_stock(stock_ticker, number_share)

        if flag == True:
            self.update_day_holdings()
        return flag


    def __str__(self):
        print('\nI´m a portfolio')
        return


