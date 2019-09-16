import pandas as pd
import numpy as np
from modules import data_manager as dm
import math
from sklearn.linear_model import LinearRegression
import sklearn
import talib
import ta
from modules import ratios as ra
import statistics


def add_ratio(df, ratio_name, price_field='Close', parameter=1,new_field_name=-1):
    """
    Loads csv into a dataframe
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """
    ratios = ['ema', 'sma','dema','kama','tema','trima','wma','ols','std','atr','aroon','mfi','ssl','ssl_s','macd_s','macd_diff','rsi','cmf','cmo','aroon_s','sar','sar_s','ema_slope']
    first_level_headers = list(dm.unique_headers(df, 1))

    if new_field_name == -1:
        new_field_name = ratio_name + str(parameter)

    if ratio_name not in ratios:
        raise Exception("\nError! Ratio [{}] doesn't exist.".format(str(ratio_name)))
        return None

    for level in first_level_headers:
        if ratio_name =='sma':
            df[level, new_field_name] = df[level, price_field].rolling(window=parameter).mean()
        elif ratio_name == 'ema':
            df[level, new_field_name] = df[level, price_field].ewm(span=parameter,adjust=False,min_periods=parameter).mean()
        elif ratio_name == 'dema':
            df[level, new_field_name] = talib.DEMA(df[level, price_field],  parameter)
        elif ratio_name == 'kama':
            df[level, new_field_name] = talib.KAMA(df[level, price_field], parameter)
        elif ratio_name == 'tema':
            df[level, new_field_name] = talib.TEMA(df[level, price_field], parameter)
        elif ratio_name == 'trima':
            df[level, new_field_name] = talib.TRIMA(df[level, price_field], parameter)
        elif ratio_name == 'wma':
            df[level, new_field_name] = talib.WMA(df[level, price_field], parameter)
        elif ratio_name =='std':
            df = add_std(df, parameter, level, price_field)
        elif ratio_name == 'ols':
            df = add_average_ols(df, level, 'ema' + str(parameter), divisions=10, length=parameter)
        elif ratio_name == 'atr':
            df = add_atr(df, parameter, level)
        elif ratio_name == 'aroon':
            df = ra.add_aroon(df, parameter, level)
        elif ratio_name == 'mfi':
            df = ra.add_mfi(df, parameter, level)
        elif ratio_name == 'ssl':
            df = ra.add_ssl(df, parameter, level)
        elif ratio_name == 'ssl_s':
            df = ra.add_ssl_s(df, parameter, level)
        elif ratio_name == 'rsi':
            df = ra.add_rsi(df, parameter, level)
        elif ratio_name == 'cmf':
            df = ra.add_cmf(df, parameter, level)
        elif ratio_name == 'cmo':
            df = ra.add_cmo(df, parameter, level)
        elif ratio_name == 'macd_diff':
            df = ra.add_macd_diff(df, parameter, level)
        elif ratio_name == 'aroon_s':
            df = ra.add_aroon_s(df, parameter, level)
        elif ratio_name == 'sar':
            df = ra.add_sar(df, parameter, level)
        elif ratio_name == 'sar_s':
            df = ra.add_sar_s(df, parameter, level)
        elif ratio_name == 'ema_slope':
            df = ra.add_ema_slope(df, parameter, level)
        elif ratio_name == 'macd_s':
            df[level, new_field_name] = ta.trend.macd(df[level, price_field], n_fast=parameter, n_slow=parameter*2, fillna=False)

    df = df.sort_index(axis=1)
    return df

def normalize_y(df):
    normalized_df=(2*(df-df.min()))/(df.max()-df.min())
    normalized_df -= 1
    return normalized_df

def add_atr(dataset,param,first_header):
    field_name = 'atr' + str(param)
    df = dataset[first_header].copy()

    atr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=param)
    column = pd.DataFrame(atr, index=df.index)

    dataset[first_header, field_name] = column
    return dataset

def add_ols(dataset,param,first_header,second_header):

    new_ols_field_name = 'ols' + str(param)
    new_ols_error_field_name = 'ols' + str(param) + 'error'
    subset = dataset[first_header, second_header]
    df = subset

    ols_list = []
    error_list = []
    parameter = param
    n = 0
    length = len(df)

    x = np.linspace(0, parameter, parameter) / parameter
    x_res = x.reshape(-1, 1)

    while n < length:
        if n < parameter - 1:
            ols_list.append(np.nan)
            error_list.append(np.nan)
        else:
            subset = df.iloc[(n - parameter + 1):n + 1]
            if not subset.isnull().values.any():
                subset = normalize_y(subset)
                model = LinearRegression().fit(x_res, subset)
                ols_list.append(model.coef_[0])
                y_predict = model.predict(x_res)
                error = sklearn.metrics.mean_squared_error(subset, y_predict)
                error_list.append(error)
            else:
                ols_list.append(np.nan)
                error_list.append(np.nan)
        n = n + 1

    # print(len(df))
    dataset[first_header, new_ols_error_field_name] = np.array(error_list)
    dataset[first_header, new_ols_field_name] = np.array(ols_list)
    return dataset

def add_std(dataset,param,first_header,second_header):

    new_field_name = 'std' + str(param)
    subset = dataset[first_header, second_header]
    df = subset

    std_list = []
    parameter = param
    n = 0
    length = len(df)

    x = np.linspace(0, parameter, parameter) / parameter

    while n < length:
        if n < parameter - 1:
            std_list.append(np.nan)
        else:
            subset = df.iloc[(n - parameter + 1):n + 1]
            if not subset.isnull().values.any():
                subset = normalize_y(subset)
                std_list.append(subset.var())
            else:
                std_list.append(np.nan)
        n = n + 1

    # print(len(df))
    dataset[first_header, new_field_name] = np.array(std_list)
    return dataset


def delete_rows(df, n):
    """
    Truncates first n rows of table
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """
    df = df.drop(df.index[0:(n)])
    return df


def preprocess_table(df, ratios, price_field = 'Adj Close'):
    """
    Adds ratios and trucates table
    :param df_path: The name of the file if in the same folder or the path to the file
    :returns: returns a Pandas Dataframe with the data of the csv requested
    ratios = [{'ratio_name':'ema', 'parameter':20}, {'ratio_name':'ols', 'parameter':90}]
    """
    max_parameter = 0
    max_index = 0

    for ratio in ratios:
        ratio_name = ratio['ratio_name']
        parameter = ratio['parameter']
        add_ratio(df, ratio_name, price_field, parameter=parameter)
    #     if type(parameter) is not list:
    #         if parameter > max_parameter:
    #             max_parameter = parameter
    #
    # if max_parameter > 0:
    #     max_index = max_parameter - 1
    # df = df.drop(df.index[0:max_index])
    df = df.sort_index(axis=1)
    return df

def add_average_ols(dataset, first_header, second_header, divisions=10, length=100):
    new_ols_field_name = 'ols' + str(length)
    new_error_field_name = new_ols_field_name + 'error'
    delays = np.linspace(0, length, num=divisions, dtype=int)
    x = np.linspace(0, length, divisions) / length
    x_res = x.reshape(-1, 1)

    subset = dataset[first_header, second_header].copy()
    df = subset.reset_index()
    ols_list = []
    length_dataset = len(df)
    empty_list = []
    error_list = []

    for i in range(length, length_dataset):
        values = []
        for delay in delays:
            prev_index = i - delay
            values.insert(0, df[first_header, second_header].loc[prev_index])
        array = np.asarray(values, dtype=float)
        if not np.isnan(array).any():
            # array = (array - min(array)) / (max(array) - min(array))
            array = (array) / ( min(array))
            y_res = array.reshape(-1, 1)
            model = LinearRegression().fit(x_res, y_res)
            ols_list.append(model.coef_[0][0])
            y_predict = model.predict(x_res)
            error = sklearn.metrics.mean_squared_error(x_res, y_predict)
            error_list.append(error)
        else:
            error_list.append(np.nan)
            ols_list.append(np.nan)
    for i in range(0,length):
        empty_list.append(np.nan)
    ols_list = list(empty_list + ols_list)
    error_ols_list = list(empty_list + error_list)

    dataset[first_header, new_ols_field_name] = ols_list
    dataset[first_header, new_error_field_name] = error_ols_list
    return dataset


def get_log_pairs(df):
    index = 0
    row = df.iloc[index]
    if row['order_type']=='sell':
        index = index + 1

    pairs_list = []
    while index < (len(df)-1):
        buy_row = df.iloc[index]
        sell_row = df.iloc[index+1]
        pairs_list.append(tuple([buy_row, sell_row]))
        index = index + 2
    return pairs_list


def percentage_change(order_df):
    log_pairs = get_log_pairs(order_df)
    if len(log_pairs) == 0:
        raise Exception('tuples len is zero!')
        return

    percentages = []
    for buy_order,sell_order in log_pairs:
        percentage_change = (sell_order['price']-buy_order['price'])/ buy_order['price']*100
        percentage_change = truncate(percentage_change,2)

        percentages.append(percentage_change)
    return percentages

def truncate(number, digits):
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper



def roi_order(order):
    buy_price = order.buy_price
    scale_out_price = order.scale_out_price
    sell_price = order.sell_price
    so_ratio = order.scale_out_ratio

    if scale_out_price is None:
        roi = (sell_price - buy_price) / buy_price
    else:
        end_price = so_ratio * scale_out_price + (1 - so_ratio) * sell_price
        roi = (end_price - buy_price) / buy_price
    return roi*100


def roi_order_list(order_list):
    roi_list = []
    for order in order_list:
        roi_list.append(truncate(roi_order(order),4))
    return roi_list


def win_rate(roi_list):
    win_list = []
    for roi in roi_list:
        if roi <= 0:
            win_list.append(0)
        else:
            win_list.append(1)

    l = len(win_list)
    if l == 0:
        return 0
    ratio = float(win_list.count(1))/l*100
    ratio = truncate(ratio,5)
    return ratio

def avg_win_loss(roi_list):
    wins = [i for i in roi_list if i >= 0]
    losses = [i for i in roi_list if i < 1]

    avg_win = 0
    avg_losses = 0

    if len(wins) != 0:
        avg_win = statistics.mean(wins)

    if len(losses) != 0:
        avg_losses = statistics.mean(losses)

    avg_win = truncate(avg_win,5)
    avg_losses = truncate(avg_losses,5)
    return avg_win, avg_losses


def add_ma(ratios,ratios_names,periods):
    for ratio in ratios_names:
        for period in periods:
            d = {}
            d['ratio_name'] = ratio
            d['parameter'] = period
            ratios.append(d)

def buy_hold_roi(dataset,init_year,end_year,tickers_dict):
    time_range= range(init_year,end_year+1)

    last_year_ret = 1
    total_year_returns = pd.Series()
    for year in time_range:
        start_date = str(year)+'-01-01'
        end_date = str(year)+'-12-31'
        aux = dm.data_between_dates(start_date,end_date,dataset.copy(),1)
        tickers = tickers_dict[year]

        first_level = tickers
        second_level = ['Close'] * len(first_level)
        selector = list(zip(first_level, second_level))

        stock_prices = aux[selector]
        stock_prices.columns = stock_prices.columns.droplevel(1)

        df = stock_prices
        df[first_level] = df[first_level].div(df.iloc[0], axis=1)

        returns_year=df.mean(axis=1)
        if list(tickers):
            total_year_returns = total_year_returns.append(returns_year*last_year_ret)
            last_year_ret = total_year_returns.values[-1]
        else:
            new_series = pd.Series(last_year_ret,index = aux.index.values)
            total_year_returns = total_year_returns.append(new_series)

    buy_hold = (total_year_returns -1)
    return buy_hold

def n_largest(dataset,tickers,year,n,field='revt'):
    year = year -1

    revg_df=pd.DataFrame(columns = ['rev_g'])
    for ticker in tickers:
        company_data = dataset[dataset['tic']==ticker]
        company_data = company_data[['fyear',field]].set_index('fyear').dropna()
        col = company_data
        col = col.loc[~col.index.duplicated(keep='first')]
        result = (col.diff()/col.shift().abs())
        try:
            value = result.loc[year][0]
        except KeyError:
            value = np.nan
        new_df = pd.DataFrame([value],index=[ticker],columns = ['rev_g'])
        revg_df = revg_df.append(new_df)

    n_largest_tickers = revg_df.nlargest(n,'rev_g').index.values.tolist()
    return n_largest_tickers

def filter_n_largest(dataset,dictionary,n):
    new_dict = {}
    for year in dictionary.keys():
        new_dict[year] = n_largest(dataset,dictionary[year],year,n)
    return new_dict

def sharpe_ratio(daily_returns, risk_free_rate = 0.02):
    trading_days = daily_returns.count()
    mean_returns = daily_returns.mean()
    mean_annual_returns = mean_returns * trading_days
    std = daily_returns.std()
    std_annual = std * math.sqrt(trading_days)
    sharpe = (mean_annual_returns-risk_free_rate) / std_annual
    return sharpe

def maximum_draw_down(holdings):
    xs = np.array(holdings)
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j = np.argmax(xs[:i]) # start of period
    high = xs[j]
    low = xs[i]
    mdd = (high - low)/high
    return mdd

def beta_stock(stock,market):
    init_stamp = stock.index[0].strftime('%Y-%m-%d')
    final_stamp = stock .index[-1].strftime('%Y-%m-%d')
    market_aux = market[(market.index >= init_stamp) & (market.index <= final_stamp)]
    market = market_aux / market_aux[0]
    market = pd.DataFrame(market).set_index(stock.index)
    market = market.iloc[:,0]
    stock = stock.pct_change()
    market = market.pct_change()
    cov = stock.cov(market)
    var = market.var()
    beta = cov/var
    return beta
