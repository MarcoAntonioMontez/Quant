import pandas as pd
import talib


# def add_atr(dataset,param,first_header):
#     field_name = 'atr' + str(param)
#     df = dataset[first_header].copy()
#
#     atr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=param)
#     column = pd.DataFrame(atr, index=df.index)
#
#     dataset[first_header, field_name] = column
#     return dataset

def add_aroon(dataset,param, first_header):
    field_name_up = 'aroon-up' + str(param)
    field_name_down = 'aroon-down' + str(param)
    df = dataset[first_header].copy()

    aroon_down, aroon_up = talib.AROON(df['High'].values, df['Low'].values, timeperiod=param)

    column_up = pd.DataFrame(aroon_up, index=df.index)
    column_down = pd.DataFrame(aroon_down, index=df.index)

    dataset[first_header, field_name_up] = column_up
    dataset[first_header, field_name_down] = column_down
    return dataset
