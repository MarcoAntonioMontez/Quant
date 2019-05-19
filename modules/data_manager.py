import pandas as pd
import numpy as np
from random import sample
import math

#Loads csv and checks the type, ie: fundamentals, constituents, and performs adequate date formatting

def load_csv(df_path, dataset_type=0):
    """
    Loads csv into a dataframe
    :param df_path: The name of the file if in the same folder or the path to the file
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the csv requested
    """


    if (dataset_type == 0):
        df = pd.read_csv(df_path, index_col=0)
        df['datadate'] = pd.to_datetime(df['datadate'], format="%d/%m/%Y", errors='coerce')
        df['rdq'] = pd.to_datetime(df['rdq'], format="%d/%m/%Y", errors='coerce')
    elif (dataset_type == 1):
        df = pd.read_csv(df_path, header=[0, 1], index_col=0)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    elif (dataset_type == 2):
        df = pd.read_csv(df_path, index_col=0)
        df['from'] = pd.to_datetime(df['from'], format="%d/%m/%Y", errors='coerce')
        df['thru'] = pd.to_datetime(df['thru'], format="%d/%m/%Y", errors='coerce')
    return df


def unique_headers( dataset, dataset_type=1):
    """
    Returns the unique headers of a dataset.

    """
    if (dataset_type == 0):  # in case of fundamentals
        print('\n not implmented for fundamental')
        # return dataset[dataset['tic'].str.strip() == company].reset_index(drop=True)
    elif (dataset_type == 1):  # in case of stock prices1
        return dataset.columns.get_level_values(0).unique()
    elif (dataset_type == 2):  # in case of constituents
        print('\n not implmented for constituints')
        # return dataset[dataset['co_conm'].str.strip() == company].reset_index(drop=True)


# Search by company
def data_company(company, dataset, dataset_type=0):
    """
    Returns all available data from a company inside a certain dataset.
    :param company: If the dataset is the fundamental data or the stock price data then this variable must be the ticker of the company. Otherwise, it must be the name of the company.
    :param dataset: Pandas Dataframe with the data to process
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the company requested
    """

    if (dataset_type == 0):  # in case of fundamentals
        return dataset[dataset['tic'].str.strip() == company].reset_index(drop=True)
    elif (dataset_type == 1):  # in case of stock prices
        dataset = dataset.iloc[:, dataset.columns.get_level_values(0) == company]
        dataset.columns = dataset.columns.droplevel(0)
        return dataset
    elif (dataset_type == 2):  # in case of constituents
        return dataset[dataset['co_conm'].str.strip() == company].reset_index(drop=True)

def data_companies(company_list, dataset, dataset_type=0):
    """
    Returns all available data from several companies nside a certain dataset.
    :param company: If the dataset is the fundamental data or the stock price data then this variable must be the ticker of the company. Otherwise, it must be the name of the company.
    :param dataset: Pandas Dataframe with the data to process
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data of the company requested
    """
    df_list=[]

    if (dataset_type == 0):  # in case of fundamentals
        print('\n not implmented for fundamental')
        # return dataset[dataset['tic'].str.strip() == company].reset_index(drop=True)
    elif (dataset_type == 1):  # in case of stock prices
        for ticker in company_list:
            company=data_company(ticker, dataset,1)
            # company.columns = company.columns.droplevel(1)
            df_list.append(company)

        dataset = pd.concat(df_list, axis=1, keys=company_list)
        return dataset
    elif (dataset_type == 2):  # in case of constituents
        print('\n not implmented for constituents')
        # return dataset[dataset['co_conm'].str.strip() == company].reset_index(drop=True)

# Search between dates
def data_between_dates(start_date, end_date, dataset, dataset_type=0):
    """
    Returns all available data inside a certain dataset during a certain interval.
    :param start_date: Starting date of the interval in format (yyyy-mm-dd)
    :param end_date: Ending date of the interval in format (yyyy-mm-dd)
    :param dataset: Pandas Dataframe with the data to process
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices, 2 if constituents.
    :returns: returns a Pandas Dataframe with the data available during the requested interval
    """
    if (dataset_type == 0 ):
        return dataset.loc[(dataset['datadate'] > start_date) & (dataset['datadate'] <= end_date)].reset_index(
            drop=True)
    elif (dataset_type == 1):
        return dataset.loc[start_date:end_date]
    elif (dataset_type == 2):
        return dataset.loc[(dataset['from'] > start_date) & (dataset['thru'] <= end_date)].reset_index(drop=True)

def live_companies_between_dates(start_date, end_date, dataset, dataset_type):
    if dataset_type != 1:
        raise Exception('Only dataset_type = 1 (technical) has been implement')

    df = data_between_dates(start_date, end_date, dataset, dataset_type)
    all_companies = df.columns.levels[0]

    live_companies = []

    for company in all_companies:
        start_price = math.isnan(df.iloc[0][company,'Adj Close'])
        end_price = math.isnan(df.iloc[-1][company,'Adj Close'])
        if not (end_price or start_price):
            live_companies.append(company)

    return live_companies

def equal_date(date, dataset, dataset_type=0):
    """
    Returns all available data in a certain date.
    :param start_date: Starting date of the interval in format (yyyy-mm-dd)
    :param end_date: Ending date of the interval in format (yyyy-mm-dd)
    :param dataset: Pandas Dataframe with the data to process
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices
    :returns: returns a Pandas Dataframe with the data available during the requested interval
    """
    if (dataset_type == 0 ):
        return dataset.loc[(dataset['datadate'] == date)].reset_index(
            drop=True)
    elif (dataset_type == 1):
        return dataset.loc[date]
    elif (dataset_type == 2):
        return dataset.loc[(dataset['from'] == date)].reset_index(drop=True)

def get_value(ticker,field,date,dataset, dataset_type=0):
    """
    Returns all available data in a certain date.
    :param start_date: Starting date of the interval in format (yyyy-mm-dd)
    :param end_date: Ending date of the interval in format (yyyy-mm-dd)
    :param dataset: Pandas Dataframe with the data to process
    :param dataset_type: Integer that tells the dataset inserted. 0 if fundamentals, 1 if stock prices
    :returns: returns a Pandas Dataframe with the data available during the requested interval
    """
    if (dataset_type == 0 ):
        print('\n not implmented for fundamental')
        # df=data_company(ticker,dataset,0)
        # df = df[field]
        # return df.loc[(df['datadate'] == date)].reset_index(
        #     drop=True)
    elif (dataset_type == 1):
        return dataset[ticker,field].loc[date]
    elif (dataset_type == 2):
        print('\n not implemented for constituents')
        # df = data_company(ticker, dataset, 2)
        # df = df[field]
        # return df.loc[(df['from'] == date)].reset_index(drop=True)

# Growth rate of an array
def growth(data, time_interval=1):
    """
    Calculates the growth of the value of a certain index in comparison with the value of 'time_interval' indexes before.
    The calculation was made using (Vi-V_(i-time_interval))/abs(V_(i-time_interval)).
    If you need the result in percentages just multiply by 100.
    :param data: Numpy array with the values to calculate the growth-
    :param time_interval: Number of index before that the user wants to compare to.
    :returns: Returns an array with the same size as 'data'. The value of an index is NaN if there's no value 'time_interval' before that index.
    """

    # create an array with NaN values
    growth = np.empty(data.size)
    growth[:] = np.nan

    for i, value in enumerate(data):

        if (i - time_interval >= 0):
            growth[i] = (value - data[i - time_interval]) / abs(data[i - time_interval])

    return growth


def company_sample(df, sample_size=1):
    """
    Does a random sampling of companies
    :param df: The input dataframe
    :param sample_size: The size of sample
    :returns: returns a Pandas Dataframe with the sampled companies data
    """

    unique_tics = df['tic'].unique()
    max_length = len(unique_tics)

    if (sample_size <= 0) or (sample_size > max_length):
        print("\nError sample size needs to be smaller than dataset size")
        print("dataset has size: " + str(max_length))
        return None

    sample_tics = sample(list(unique_tics), sample_size)

    sample_df = df.loc[df['tic'].isin(sample_tics)]
    return sample_df


def company_double_sample(df1, df2, sample_size=1):
    """
    Does a random sampling of companies from two dataframes, using df1 unique tics as primary keys
    :param df1: The input dataframe1
    :param df2: The input dataframe2
    :param sample_size: The size of sample
    :returns: returns two Pandas Dataframes with the sampled companies data
    """

    unique_tics = df1['tic'].unique()
    max_length = len(unique_tics)

    if (sample_size <= 0) or (sample_size > max_length):
        print("\nError sample size needs to be smaller than dataset size")
        print("dataset has size: " + str(max_length))
        return None

    sample_tics = sample(list(unique_tics), sample_size)

    sample_df1 = df1.loc[df1['tic'].isin(sample_tics)]
    sample_df2 = df2.loc[df2['tic'].isin(sample_tics)]

    return [sample_df1, sample_df2]


def reformat_prices_csv(df):
    company_list = list(df['Open'].columns.values)
    dataset = df

    df_list = []

    for ticker in company_list:
        company = data_company(ticker, dataset, 1)
        company.columns = company.columns.droplevel(1)
        df_list.append(company)

    dataset = pd.concat(df_list, axis=1, keys=company_list)

    # dataset.to_csv('formated_prices_2005-01-01_2018-12-31.csv')
    return dataset
