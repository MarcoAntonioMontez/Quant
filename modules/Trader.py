import numpy as np
import pandas as pd
import datetime
from modules import data_manager as dm
from modules.Portfolio import Portfolio
from modules.Strategy import Strategy
from modules import UserInput
from modules.Order import Order


class Trader:
    def __init__(self, dataset, user_input):
        self.user_input = user_input
        self.initial_capital = user_input.initial_capital
        self.tickers = user_input.tickers
        self.strategy_name = user_input.strategy
        dates_dict = self.adjusted_dates(dataset, user_input.start_date, user_input.end_date)
        self.start_date = dates_dict['start_date']
        self.end_date = dates_dict['end_date']
        self.dataset = self.truncate_dataset(dataset)
        self.end_date = self.dataset.index[-1].date()
        self.portfolio = Portfolio(self.initial_capital, self.start_date, self.dataset)
        self.portfolio.set_trader(self)
        self.start_date = self.portfolio.start_day
        self.current_day = self.start_date
        self.strategy = Strategy(self.strategy_name, user_input.strategy_params, self.dataset, self.tickers, self.portfolio)
        self.number_shares = 100
        self.confirmation_indicators_table = self.create_confirmation_ind_table()

    def __str__(self):
        print('\nCurrent Day: ' + str(self.current_day))
        print('\nStart date: ' + str(self.start_date))
        print('\nEnd date: ' + str(self.end_date))
        print('\nInitial Capital: ' + str(self.initial_capital))
        print('\nTickers: ' + str(self.tickers))
        print('\nStrategy: ' + str(self.strategy_name))
        return

    def adjusted_dates(self,df, user_start_date, user_end_date):
        real_start_date = None
        real_end_date = None

        user_start_date = pd.to_datetime(user_start_date, format="%Y-%m-%d", errors='coerce').date()
        user_end_date = pd.to_datetime(user_end_date, format="%Y-%m-%d", errors='coerce').date()

        df_start_date = df.index[0].date()
        df_end_date = df.index[-1].date()

        if user_start_date >= user_end_date:
            raise Exception('Start time bigger than end time!')

        if user_start_date >= df_end_date:
            raise Exception('simulation start date is bigger than dataframe end date!')
        elif user_end_date <= df_start_date:
            raise Exception('simulation end date is lower than dataframe start date!')

        if user_start_date < df_start_date:
            real_start_date = df_start_date
        elif user_start_date >= df_start_date:
            real_start_date = user_start_date

        if user_end_date > df_end_date:
            real_end_date = df_end_date
        elif user_end_date <= df_end_date:
            real_end_date = user_end_date

        dates = {'start_date': real_start_date, 'end_date': real_end_date}
        return dates

    def truncate_dataset(self, df):
        df = df.copy()
        df = df.truncate(before=self.start_date, after=self.end_date)
        return df

    def simulate_day(self):
        orders = self.strategy.simulate_day(self.current_day)
        for order in orders:
            if order['Type'] == 'buy':
                position = min(self.portfolio.current_cash, self.portfolio.net_worth/len(self.tickers))
                new_order = Order(self, order['Stock'], position = position)
                self.portfolio.add_open_order(new_order)
            elif order['Type']=='sell':
                open_order = self.portfolio.get_open_order(order['Stock'])
                open_order.sell_stock(order['exit_type'])
                #remove open_order
                self.portfolio.close_order(order['Stock'])
            elif order['Type'] == 'scale_out':
                open_order = self.portfolio.get_open_order(order['Stock'])
                open_order.scale_out_stock()
        return orders

    def next_day(self):
        self.current_day = self.portfolio.next_day()

    def get_holdings(self):
        return self.portfolio.get_holdings()


    def run_simulation(self):
        while(self.current_day < self.end_date):
            self.simulate_day()
            self.next_day()
        self.portfolio.sell_all_stocks()
            # print('Current day:' + str(curren))
            # print(self.current_day)


    def score_binary(self,ratio, buy_limit):
        # print(ratio.name[1])
        if ratio.name[1] == 'rsi14':
            return (ratio < buy_limit).astype(int)
        else:
            return (ratio > buy_limit).astype(int)


    def create_confirmation_ind_table(self):
        inputs = self.user_input.inputs['confirmation_indicators']
        indicators = inputs['indicators']
        weights = inputs['weights']
        buy_limits = inputs['buy_limits']
        tickers = self.tickers
        idx = pd.IndexSlice
        df1 = self.dataset.loc[:, idx[tickers, indicators]].copy()

        for ticker in tickers:
            df1[ticker, 'total_score'] = 0
            for i in range(0, len(indicators)):
                new_field = 'score_' + indicators[i]
                df1[ticker, new_field] = self.score_binary(df1[ticker, indicators[i]], buy_limits[i])
                df1[ticker, 'total_score'] = df1[ticker, 'total_score'] + df1[ticker, new_field] * weights[i]
        return df1.sort_index(axis=1)
