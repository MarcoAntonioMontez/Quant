import pandas as pd
import numpy as np

from plotly.offline import iplot
import plotly.graph_objs as go
import plotly
plotly.offline.init_notebook_mode()
plotly.tools.set_credentials_file(username='marco.montez', api_key='FgZQOnOU1P78yrlx0Vwx')
import os
import time
from shutil import copyfile



def color_bar():
    # import seaborn as sns
    #
    # cm = sns.light_palette("green", as_cmap=True)
    # s = df.style.background_gradient(cmap=cm)

    one = 1
    # to implement
    return one


def plot(dataset, fields=None, plot_type='scatter', title='plot',path = None,show_plot = True):
    df = dataset
    data = []

    if plot_type == 'scatter':
        if fields is None:
            return Exception('No field name chosen')
        for field in fields:
            trace = go.Scatter(x=df.index,
                               y=df[field],
                               name=field)
            data.append(trace)
    elif plot_type == 'ohlc':
        trace = go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name='candlestick')
        data.append(trace)
    else:
        print('Type of plot not supported')

    layout = go.Layout(
        title=title,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )))

    fig = dict(data=data, layout=layout)
    if show_plot:
        iplot(fig, filename=title)
    if path is not None:
        plotly.offline.plot(fig, filename=path, auto_open=False)


def histogram(values, bin_size=None, title='title', start_range=None, end_range=None,path=None,show_plot = True):
    data = []
    xbins={}
    if bin_size is not None :
        xbins['size'] = bin_size

    if start_range is not None :
        xbins['start'] = start_range

    if end_range is not None :
        xbins['end'] = end_range

    trace = go.Histogram(x=values, xbins=xbins)
    layout = go.Layout(
        title=title,
        )
    data.append(trace)
    fig = dict(data=data, layout=layout)
    if show_plot:
        iplot(fig, filename=title)
    if path is not None:
        plotly.offline.plot(fig, filename=path, auto_open=False)

