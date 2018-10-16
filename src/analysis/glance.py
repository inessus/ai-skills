import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def missing_ratio(df, limit=0.999):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_ratio'] = missing_df['missing_count'] / df.shape[0]
    print(missing_df.loc[missing_df['missing_ratio'] > limit])


def show_dtypes(df, detail=False):
    pd.options.display.max_rows = 65
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ['Count', 'Column Type']
    if detail:
        print(dtype_df)
    return dtype_df.groupby("Column Type").aggregate('count').reset_index()


def cols_type_group(train):
    dtype_df = train.dtypes.reset_index()
    dtype_df.columns = ['Count', 'Column Type']
    dtype_df = dtype_df.groupby('Column Type').aggregate('count').reset_index()
    print(dtype_df)
    plt.figure(figsize=(12, 8))
    sns.barplot(dtype_df['Column Type'].values, dtype_df['Count'].values, alpha=0.8, color=color[4])
    plt.show()


def missing_target_point_plot(train_df, test_df, col):
    train_df['null_count'] = train_df.isnull().sum(axis=1)
    test_df['null_count'] = test_df.isnull().sum(axis=1)

    plt.figure(figsize=(14, 12))
    sns.pointplot(x='null_count', y=col, data=train_df)
    plt.ylabel(col, fontsize=12)
    plt.xlabel('null_count', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()


def missing_show(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count']>0]
    missing_df = missing_df.sort_values(by='missing_count')
    print(missing_df)


def missing_sort_barh_plot(df, create_plot=True):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count'] > 0]
    missing_df = missing_df.sort_values(by='missing_count')

    if create_plot:
        ind = np.arange(missing_df.shape[0])
        width = 0.9
        fig, ax = plt.subplots(figsize=(12, 18))
        rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
        ax.set_yticks(ind)
        ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
        ax.set_xlabel("Count of missing values")
        ax.set_title("Number of missing values in each column")
        plt.show()