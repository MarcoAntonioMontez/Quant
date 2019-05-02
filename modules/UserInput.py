class UserInput:
    def __init__(self, inputs):
        self.start_date = inputs['start_date']
        self.end_date = inputs['end_date']
        self.initial_capital = inputs['initial_capital']
        self.tickers = inputs['tickers']
        self.strategy = inputs['strategy']
        self.strategy_params = inputs['strategy_params']

    def __str__(self):
        # print('\n -- User Input Class -- ')
        print('\nStart date: ' + str(self.start_date))
        print('\nEnd date: ' + str(self.end_date))
        print('\nInitial Capital: ' + str(self.initial_capital))
        print('\nTickers: ' + str(self.tickers))
        print('\nStrategy: ' + str(self.strategy))
        print('\nStrategy_params: ' + str(self.strategy_params))
