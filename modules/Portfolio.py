import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
import math

#date = datetime date

class Portfolio:
    def __init__(self, initial_capital, start_day, dataset):
        self.current_cash = initial_capital
        self.dataset = dataset
        self.holdings = pd.DataFrame(columns=['Date'])
        self.holdings.set_index('Date', inplace=True)
        self.day_count = 0
        self.price_field='Adj Close'
        self.transaction_cost=0.003
        self.total_transaction_cost=0
        self.length_dataset = len(dataset)
        self.net_worth = self.current_cash
        self.start_day = self.find_inital_date(start_day, self.dataset)
        self.current_day = self.start_day
        self.end_day = self.dataset.index[-1].date()
        self.update_day_holdings()
        self.orders_log = []
        self.open_orders = []
        # self.remaining_dates = self.create_date_index()


    def get_holdings(self):
        return self.holdings

    def get_date(self):
        return self.current_day

    def get_cash(self):
        return self.current_cash

    def get_current_price(self, ticker):
        price = dm.get_value(ticker, self.price_field, self.current_day, self.dataset,  1)
        return price

    def get_transaction_costs(self):
        return self.total_transaction_costs

    def find_inital_date(self, date, df):
        index = df.index
        while date not in index:
            date = date + datetime.timedelta(1)
        return date

    def last_row_date_updated(self, df, date):
        df = df.iloc[-1:].reset_index().copy()
        df['Date'] = date
        df.set_index('Date', inplace=True)
        return df

    def get_current_stocks(self):
        tickers = []
        current_stocks = []
        df = self.holdings
        last_row = df.iloc[-1:].copy()

        for ticker in list(last_row.columns.values):
            if not ticker.startswith('_'):
                tickers.append(ticker)

        for ticker in tickers:
            number_shares = last_row.loc[last_row.index[-1]].at[ticker]
            if number_shares > 0:
                current_stocks.append(ticker)
        return current_stocks

    def get_num_shares(self, ticker):
        number_shares = 0
        df = self.holdings
        last_row = df.iloc[-1:].copy()

        if ticker not in list(last_row.columns.values):
            raise Exception('Cant get # shares, stock doesnt exist in holding.dataframe')

        number_shares = last_row.loc[last_row.index[-1]].at[ticker]
        return number_shares

    def calc_net_worth(self):
        cash = self.current_cash
        current_stocks = self.get_current_stocks()
        total_net_worth = 0 + cash

        for stock in current_stocks:
            price = dm.get_value(stock, self.price_field, self.current_day, self.dataset, 1)
            stock_total_cost = price * self.get_num_shares(stock)
            total_net_worth = total_net_worth + stock_total_cost

        return total_net_worth

    def update_day_holdings(self):
        self.holdings.at[self.current_day, '_day count'] = self.day_count
        self.holdings.at[self.current_day, '_transaction costs'] = self.total_transaction_cost
        self.holdings.at[self.current_day, '_cash'] = self.current_cash
        self.net_worth = self.calc_net_worth()
        self.holdings.at[self.current_day, '_net worth'] = self.net_worth
        #update open orders
        return

    # def create_date_index(self):
    #     dates = []
    #     start_date = self.start_day
    #     end_date = self.end_day
    #
    #     for indice in self.dataset.index.values:
    #         date_indice = pd.to_datetime(indice, format="%Y-%m-%d", errors='coerce').date()
    #         if date_indice > start_date and date_indice <= end_date:
    #             dates.append(date_indice)
    #
    #     return dates

    def next_day_date(self):
        self.day_count = self.day_count + 1
        df = self.dataset

        if self.current_day == self.end_day:
            print('\nLast day of portfolio simulatio\n No more days to go')
            return False

        idx = df.index.get_loc(self.current_day)
        next_day = df.index[min(idx + 1, self.length_dataset - 1)]
        next_day = pd.to_datetime(next_day, format="%Y-%m-%d", errors='coerce').date()
        return next_day
        # next_day = self.remaining_dates.pop(0)
        # return next_day

    def init_day_holdings(self):
        new_row = self.last_row_date_updated(self.holdings,self.current_day)
        self.holdings = pd.concat([self.holdings, new_row])
        self.update_day_holdings()
        return self.holdings

    def next_day(self):
        #add condition for last day
        self.current_day = self.next_day_date()
        self.init_day_holdings()
        return self.current_day

    def add_stock(self, stock_ticker, number_shares):
        # print('buy')
        if stock_ticker not in self.holdings.columns.values:
            self.holdings.at[self.current_day, stock_ticker] = number_shares
        else:
            self.holdings.at[self.current_day, stock_ticker] = self.holdings.at[self.current_day, stock_ticker] + number_shares

        self.add_order_log('buy', stock_ticker, number_shares)
        return


    # Add error management
    def remove_stock(self, stock_ticker, number_shares):
        # print('sell')
        if stock_ticker not in self.holdings.columns.values:
            print('\nERROR: Stock is not held')
            return False
        elif math.isnan(self.holdings.at[self.current_day, stock_ticker]):
            print('\nERROR:Nan value for number of shares')
            return False
        elif self.holdings.at[self.current_day, stock_ticker] < number_shares:
            if abs(self.holdings.at[self.current_day, stock_ticker] - number_shares) < 0.0000001:
                self.holdings.at[self.current_day, stock_ticker] = 0
                self.add_order_log('sell', stock_ticker, number_shares)
                return True

            print('\nERROR:Not enough shares to sell')
            print('\nDay: ' + str(self.current_day))
            print('Number Shares held: ' + str(self.holdings.at[self.current_day, stock_ticker]))
            print('Number of shares to sell: ' + str(number_shares))
            return False
        else:
            self.holdings.at[self.current_day, stock_ticker] = self.holdings.at[self.current_day, stock_ticker] - number_shares

        if abs(self.holdings.at[self.current_day, stock_ticker]) < 0.0000001:
            self.holdings.at[self.current_day, stock_ticker] = 0

        self.add_order_log('sell', stock_ticker, number_shares)
        return True

    def buy_stock_money(self, stock_ticker, money):
        stock_price = self.get_current_price(stock_ticker)
        number_shares = money / stock_price
        return self.buy_stock(stock_ticker, number_shares)

    def sell_stock_money(self, stock_ticker, money):
        stock_price = self.get_current_price(stock_ticker)
        number_shares = money / stock_price
        return self.sell_stock(stock_ticker, number_shares)

    def buy_stock(self, stock_ticker, number_shares):
        stock_price = self.get_current_price(stock_ticker)
        real_number_shares = number_shares / (1 + self.transaction_cost)
        order_cost = stock_price * real_number_shares
        transaction_cost = order_cost * self.transaction_cost
        total_cost = order_cost + transaction_cost

        if total_cost <= self.current_cash:
            self.current_cash = self.current_cash - total_cost
            self.add_stock(stock_ticker,real_number_shares)
            self.total_transaction_cost = self.total_transaction_cost + transaction_cost
            return True
        elif abs(total_cost - self.current_cash) < 0.000001:
            total_cost = self.current_cash
            self.current_cash = self.current_cash - total_cost
            self.add_stock(stock_ticker, real_number_shares)
            self.total_transaction_cost = self.total_transaction_cost + transaction_cost
            return True
        else:
            print('\nNot enough cash to buy order')
            print('\nDay: ' + str(self.current_day))
            print('Number Share: ' + str(number_shares))
            print('Total cost order: ' + str(total_cost))
            print('Cash: ' + str(self.current_cash))

            return False

    def sell_stock(self, stock_ticker, number_shares):
        stock_price = self.get_current_price(stock_ticker)
        order_return = stock_price * number_shares
        transaction_cost = order_return * self.transaction_cost
        total_return = order_return - transaction_cost

        if self.remove_stock(stock_ticker, number_shares):
            self.current_cash = self.current_cash + total_return
            self.total_transaction_cost = self.total_transaction_cost + transaction_cost
            return True
        else:
            return False

    def order_money(self,order_type, stock_ticker, money):
        flag = False
        if order_type == 'buy':
            flag = self.buy_stock_money(stock_ticker, money)
        elif order_type == 'sell':
            flag = self.sell_stock_money(stock_ticker, money)

        if flag:
            self.update_day_holdings()
        return flag

    def order(self,order_type, stock_ticker, number_share):
        flag = False
        if order_type == 'buy':
            flag = self.buy_stock(stock_ticker, number_share)
        elif order_type == 'sell':
            flag = self.sell_stock(stock_ticker, number_share)

        if flag == True:
            self.update_day_holdings()
        return flag

    def add_order_log(self, order_type, stock, shares):
        order_log = {}
        date = self.current_day
        price = self.get_current_price(stock)

        order_log['date'] = date
        order_log['order_type'] = order_type
        order_log['stock'] = stock
        order_log['shares'] = shares
        order_log['price'] = price
        self.orders_log.append(order_log)

    def get_order_log(self, company=False):
        df = pd.DataFrame(self.orders_log)
        if company:
            if company not in self.holdings.columns.values:
                return None
            df = df.loc[df['stock']==company].reset_index(drop=True)
        return df

    def __str__(self):
        print('\nI´m a portfolio')
        return


