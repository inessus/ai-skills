import numpy as np
import pandas as pd


def combined_data(train, test):
    """
        Get the combined data
        :param trainer pandas.dataframe:
        :param test pandas.dataframe:
        :return pandas.dataframe:
    """
    A = set(train.columns.values)
    B = set(test.columns.values)
    colToDel = A.difference(B)
    total_df = pd.concat([train.drop(colToDel, axis=1), test], axis=0)
    return total_df


def remove_duplicate_columns(total_df):
    """
        Removing duplicate columns
    """
    colsToRemove = []
    columns = total_df.columns
    for i in range(len(columns) - 1):
        v = total_df[columns[i]].values
        for j in range(i + 1, len(columns)):
            if np.array_equal(v, total_df[columns[j]].values):
                colsToRemove.append(columns[j])
    colsToRemove = list(set(colsToRemove))
    total_df.drop(colsToRemove, axis=1, inplace=True)
    print(f">> Dropped {len(colsToRemove)} duplicate columns")
    return total_df


def merge_data():
    files_to_use = ['bitcoin_price.csv', 'ethereum_price.csv', 'ripple_price.csv', 'litecoin_price.csv']
    cols_to_use = []
    for ind, file_name in enumerate(files_to_use):
        currency_name = file_name.split('_')[0]
        if ind == 0:
            df = pd.read_csv('../input/' + file_name, usecols=['Date', 'Close'], parse_dates=['Date'])
            df.columns = ['Date', currency_name]
        else:
            temp_df = pd.read_csv('../input/' + file_name, usecols=['Date', 'Close'], parse_dates=['Date'])
            temp_df.columns = ['Date', currency_name]
            df = pd.merge(df, temp_df, on='Date')
        cols_to_use.append(currency_name)
    df.head()