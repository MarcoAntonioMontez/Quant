# from modules import Strategy as strat


def indicator_cross(strategy,ticker, indicator, params):
    if indicator == 'aroon':
        return aroon_cross(strategy,ticker, params)
    else:
        raise Exception('Indicator: ' + str(indicator) + ' is not implemented in indicator_cross')


def aroon_cross(strategy,ticker, params):
    aroon_up = 'aroon-up' + str(params['entry_indicator_period'])
    aroon_down = 'aroon-down' + str(params['entry_indicator_period'])

    return strategy.cross(ticker, aroon_up, aroon_down)


def above_baseline(price_value, baseline_value, params= None):
    if price_value >= baseline_value:
        return True
    else:
        return False



