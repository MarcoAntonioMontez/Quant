import pandas as pd
import numpy as np
from modules import data_manager as dm


#Loads csv and checks the type, ie: fundamentals, constituents, and performs adequate date formatting

# def add_ratio(df, new_field_name, ratio_name, parameter=1):
#     """
#     Loads csv into a dataframe
#     :param df_path: The name of the file if in the same folder or the path to the file
#     :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
#     :returns: returns a Pandas Dataframe with the data of the csv requested
#     """
#
#     price_field= 'Adj Close'
#
#     ratios = ['ema', 'sma']
#
#     if ratio_name not in ratios:
#         print("\nError Ratio doesn´t exist.")
#         return None
#
#     if ratio_name =='sma':
#         df[new_field_name]=df[price_field].rolling(window=parameter).mean()
#     elif ratio_name == 'ema':
#         df[new_field_name] = df[price_field].ewm(span=parameter,adjust=False,min_periods=parameter).mean()
#
#     return df

def add_ratio(df, ratio_name, parameter=1,new_field_name=-1):
    """
    Loads csv into a dataframe
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """

    price_field= 'Adj Close'
    ratios = ['ema', 'sma']
    first_level_headers = list(dm.unique_headers(df, 1))

    if new_field_name == -1:
        new_field_name = ratio_name + str(parameter)

    if ratio_name not in ratios:
        print("\nError Ratio doesn´t exist.")
        return None

    for first_level in first_level_headers:
        if ratio_name =='sma':
            df[first_level, new_field_name] = df[first_level, price_field].rolling(window=parameter).mean()
        elif ratio_name == 'ema':
            df[first_level, new_field_name] = df[first_level, price_field].ewm(span=parameter,adjust=False,min_periods=parameter).mean()

    df = df.sort_index(axis=1)
    return df

def delete_rows(df, n):
    """
    Truncates first n rows of table
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """
    df = df.drop(df.index[0:(n)])
    return df

def preprocess_table(df, ratios):
    """
    Adds ratios and trucates table
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """
    max_parameter=0

    for ratio in ratios:
        ratio_name = ratio['ratio_name']
        parameter = ratio['parameter']
        add_ratio(df, ratio_name, parameter)
        if parameter > max_parameter:
            max_parameter = parameter

    df = df.drop(df.index[0:max_parameter])
    df = df.sort_index(axis=1)
    return df