class UserInput:
    def __init__(self, inputs):
        self.start_date = inputs['start_date']
        self.end_date = inputs['end_date']
        self.initial_capital = inputs['initial_capital']
        self.tickers = inputs['tickers']
        self.strategy = inputs['strategy']
        self.strategy_params = inputs['strategy_params']
        self.stop_loss_type = self.strategy_params['stop_loss_type']
        self.stop_loss_parameter = self.strategy_params['stop_loss_parameter']
        self.take_profit_type = self.strategy_params['take_profit_type']
        self.take_profit_parameter = self.strategy_params['take_profit_parameter']
        self.trailing_stop_type = self.strategy_params['trailing_stop_type']
        self.trailing_stop_parameter = self.strategy_params['trailing_stop_parameter']
        self.scale_out_ratio = self.strategy_params['scale_out_ratio']

    def __str__(self):
        # print('\n -- User Input Class -- ')
        print('\nStart date: ' + str(self.start_date))
        print('\nEnd date: ' + str(self.end_date))
        print('\nInitial Capital: ' + str(self.initial_capital))
        print('\nTickers: ' + str(self.tickers))
        print('\nStrategy: ' + str(self.strategy))
        print('\nStrategy_params: ' + str(self.strategy_params))
