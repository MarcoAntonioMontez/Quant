import pandas as pd
import talib
import ta


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


def add_aroon_s(dataset,param, first_header):
    field_name_up = 'aroon_s' + str(param)
    df = dataset[first_header].copy()
    high = df['High'].values
    low = df['Low'].values

    aroon = talib.AROONOSC(high, low, timeperiod=14)/100

    col = pd.DataFrame(aroon, index=df.index)
    dataset[first_header, field_name_up] = col
    return dataset

def add_mfi(dataset,param, first_header):
    field_name_up = 'mfi' + str(param)
    df = dataset[first_header].copy()
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].astype(float).values

    mfi = talib.MFI(high, low, close, volume, timeperiod=param)/100 - 0.5

    col = pd.DataFrame(mfi, index=df.index)
    dataset[first_header, field_name_up] = col
    return dataset

def add_fi(dataset,param, first_header):
    field_name_up = 'fi' + str(param)
    df = dataset[first_header].copy()
    close = df['Close']
    volume = df['Volume']

    ind = ForceIndex(close, volume, timeperiod=param)

    col = pd.DataFrame(ind, index=df.index)
    dataset[first_header, field_name_up] = col
    return dataset

def ForceIndex(close,volume,timeperiod):
    return pd.Series(close.diff(timeperiod) * volume)


def ssl(high, low, close, timeperiod):
    avgH = pd.Series(high, index=high.index).rolling(window=timeperiod).mean().values
    avgL = pd.Series(low, index=low.index).rolling(window=timeperiod).mean().values
    close = pd.Series(close, index=low.index).values

    hilo = 0

    gann_hilos = []

    for i in range(0, len(avgH)):
        if close[i] > avgH[i]:
            hilo = 1
        elif close[i] < avgL[i]:
            hilo = 0

        if hilo:
            gann_hilo = avgL[i]
        else:
            gann_hilo = avgH[i]

        gann_hilos.append(gann_hilo)

    return gann_hilos


def add_ssl(dataset,param, first_header):
    field_name = 'ssl' + str(param)
    df = dataset[first_header].copy()

    ssl_list = ssl(df['High'], df['Low'], df['Close'], param)

    col = pd.DataFrame(ssl_list, index=df.index)

    dataset[first_header, field_name] = col
    return dataset


def add_ssl_s(dataset,param, first_header):
    field_name = 'ssl_s' + str(param)
    df = dataset[first_header].copy()
    close = df['Close']

    ssl_list = ssl(df['High'], df['Low'], df['Close'], param)

    col = pd.Series(ssl_list, index=df.index)

    dataset[first_header, field_name] = close-col
    return dataset


def add_rsi(dataset,param, first_header):
    field_name = 'rsi' + str(param)
    df = dataset[first_header].copy()

    rsi = talib.RSI(df['Close'], timeperiod=param)/100

    col = pd.DataFrame(rsi, index=df.index)

    dataset[first_header, field_name] = col
    return dataset


def add_cmf(dataset,param, first_header):
    field_name = 'cmf' + str(param)
    df = dataset[first_header].copy()
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    cmf = ta.volume.chaikin_money_flow(high, low, close, volume, n=param, fillna=True)

    col = pd.DataFrame(cmf, index=df.index)

    dataset[first_header, field_name] = col
    return dataset


def add_macd_diff(dataset,param, first_header):
    if len(param)!=3:
        raise Exception('This ratio requires three parameters, [0]:fast ema, [1]: slow ema, [2]:sign ema')
    field_name = 'macd_diff' + str(param[0]) + '_' + str(param[1]) + '_' + str(param[2])
    df = dataset[first_header].copy()
    close = df['Close']

    macd = ta.trend.macd_diff(close, n_fast=param[0], n_slow=param[1], n_sign=param[2], fillna=True)

    col = pd.DataFrame(macd, index=df.index)

    dataset[first_header, field_name] = col
    return dataset


def add_cmo(dataset,param, first_header):
    field_name = 'cmo' + str(param)
    df = dataset[first_header].copy()
    close = df['Close']

    cmo = talib.CMO(close, timeperiod=14)/100

    col = pd.DataFrame(cmo, index=df.index)

    dataset[first_header, field_name] = col
    return dataset


def add_sar(dataset,param, first_header):
    if len(param) != 2:
        raise Exception('This ratio requires two parameters, [0]:acceleration, [1]: maximum ')
    field_name = 'sar' + str(param[0]) + '_' + str(param[1])
    df = dataset[first_header].copy()
    high = df['High']
    low = df['Low']

    sar = talib.SAR(high, low, acceleration=param[0], maximum=param[1])

    col = pd.DataFrame(sar, index=df.index)

    dataset[first_header, field_name] = col
    return dataset


def add_sar_s(dataset,param, first_header):
    if len(param) != 2:
        raise Exception('This ratio requires two parameters, [0]:acceleration, [1]: maximum ')
    field_name = 'sar_s' + str(param[0]) + '_' + str(param[1])
    df = dataset[first_header].copy()
    high = df['High']
    low = df['Low']
    close = df['Close']

    sar = talib.SAR(high, low, acceleration=param[0], maximum=param[1])
    col = pd.Series(sar, index=df.index)

    dataset[first_header, field_name] = close-col
    return dataset

# def add_dema()


def add_ema_slope(dataset,param, first_header):
    if len(param) != 2:
        raise Exception('This ratio requires two parameters, [0]:period, [1]: mean parameter ')
    ema_name = 'ema' + str(param[0])
    field_name =  ema_name + '_slope_mean' + str(param[1])
    if ema_name in dataset[first_header].columns.values.tolist():
        ema_col = dataset[first_header, ema_name]
    else:
        ema_col = dataset[first_header, 'Close'].ewm(span=param[0], adjust=False, min_periods=param[0]).mean()

    ema_slope_col = ema_col.diff().rolling(window=param[1]).mean()

    dataset[first_header, field_name] = ema_slope_col
    return dataset


def add_ema_slope_small_dataset(dataset,param):
    if len(param) != 2:
        raise Exception('This ratio requires two parameters, [0]:period, [1]: mean parameter ')
    ema_name = 'ema' + str(param[0])
    field_name =  ema_name + '_slope_mean' + str(param[1])
    if ema_name in dataset.columns.values.tolist():
        ema_col = dataset[ ema_name]
    else:
        ema_col = dataset['Close'].ewm(span=param[0], adjust=False, min_periods=param[0]).mean()

    ema_slope_col = ema_col.diff().rolling(window=param[1]).mean()

    dataset[ field_name] = ema_slope_col
    return dataset