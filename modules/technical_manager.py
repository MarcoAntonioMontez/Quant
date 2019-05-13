import pandas as pd
import numpy as np
from modules import data_manager as dm
import math
from sklearn.linear_model import LinearRegression
import sklearn
import talib
from modules import ratios as ra

def add_ratio(df, ratio_name, price_field, parameter=1,new_field_name=-1):
    """
    Loads csv into a dataframe
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """
    ratios = ['ema', 'sma','ols','std','atr','aroon']
    first_level_headers = list(dm.unique_headers(df, 1))

    if new_field_name == -1:
        new_field_name = ratio_name + str(parameter)

    if ratio_name not in ratios:
        raise Exception("\nError Ratio doesn't exist.")
        return None

    for level in first_level_headers:
        if ratio_name =='sma':
            df[level, new_field_name] = df[level, price_field].rolling(window=parameter).mean()
        elif ratio_name == 'ema':
            df[level, new_field_name] = df[level, price_field].ewm(span=parameter,adjust=False,min_periods=parameter).mean()
        elif ratio_name =='std':
            df = add_std(df, parameter, level, price_field)
        elif ratio_name == 'ols':
            df = add_average_ols(df, level, 'ema100', divisions=10, length=parameter)
        elif ratio_name == 'atr':
            df = add_atr(df, parameter, level)
        elif ratio_name == 'aroon':
            df = ra.add_aroon(df, parameter, level)



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
        if parameter > max_parameter:
            max_parameter = parameter

    if max_parameter > 0:
        max_index = max_parameter - 1
    df = df.drop(df.index[0:max_index])
    df = df.sort_index(axis=1)
    return df


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


def roi(company, dataset):
    start_price = dataset[company, 'Adj Close'].iloc[0]
    end_price = dataset[company, 'Adj Close'].iloc[-1]
    return ((end_price - start_price) / start_price)


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


def win_rate(order_list):
    if len(order_list) == 0:
        print('Cannot calculate winrate, since list is empty')
        return None

    win_list = []
    for order in order_list:
        if order.order_type == 'loss':
            win_list.append(0)
        else:
            win_list.append(1)
    ratio = float(win_list.count(1))/len(win_list)*100
    ratio = truncate(ratio,5)
    return ratio


def mean(lst):
    return sum(lst) / len(lst)

def average(lst):
    return sum(lst) / len(lst)


def add_average_ols(dataset, first_header, second_header, divisions=10, length=100):
    new_ols_field_name = 'ols' + str(length) + '/' + str(divisions)
    delays = np.linspace(0, length, num=divisions, dtype=int)
    x = np.linspace(0, length, divisions) / length
    x_res = x.reshape(-1, 1)

    subset = dataset[first_header, second_header].copy()
    df = subset.reset_index()
    values = []
    ols_list = []
    length_dataset = len(df)
    empty_list = np.empty([length, 1])
    empty_list.fill(np.nan)

    for i in range(length, length_dataset):
        values = []
        for delay in delays:
            prev_index = i - delay
            values.insert(0, df[first_header, second_header].loc[prev_index])
        array = np.asarray(values, dtype=float)
        if not np.isnan(array).any():
            array = (array - min(array)) / (max(array) - min(array))
            y_res = array.reshape(-1, 1)
            model = LinearRegression().fit(x_res, y_res)
            ols_list.append(model.coef_[0])
        else:
            ols_list.append(np.nan)
    #     display(empty_list.tolist())
    ols_list = list(empty_list.tolist() + ols_list)
    dataset[first_header, new_ols_field_name] = np.array(ols_list)
    return dataset