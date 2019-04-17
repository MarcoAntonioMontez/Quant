import pandas as pd
import numpy as np


#Loads csv and checks the type, ie: fundamentals, constituents, and performs adequate date formatting

def add_ratio(df, new_field_name, ratio_name, parameter=1):
    """
    Loads csv into a dataframe
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """

    price_field= 'prccd'

    ratios = ['ema', 'sma']

    if ratio_name not in ratios:
        print("\nError Ratio doesn´t exist.")
        return None

    if ratio_name =='sma':
        df[new_field_name]=df[price_field].rolling(window=parameter).mean()
    elif ratio_name == 'ema':
        df[new_field_name] = df[price_field].ewm(span=parameter,adjust=False,min_periods=parameter).mean()

    return df

