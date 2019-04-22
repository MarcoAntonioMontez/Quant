class UserInput:
    def __init__(self, dict):
        self.start_date = dict['start_date']
        self.end_date = dict['end_date']
        self.initial_capital = dict['initial_capital']
        self.tickers = dict['tickers']
        self.strategy = dict['strategy']

    def __str__(self):
        # print('\n -- User Input Class -- ')
        print('\nStart date: ' + str(self.start_date))
        print('\nEnd date: ' + str(self.end_date))
        print('\nInitial Capital: ' + str(self.initial_capital))
        print('\nTickers: ' + str(self.tickers))
        print('\nStrategy: ' + str(self.strategy))
