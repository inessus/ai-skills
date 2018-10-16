import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()

from prepare.other.zillow import target_sort_scatter_plot
from prepare.other.FeatureEngineering import missing_plot
from prepare.other.zillow import trim_col_vs_target_joint_plot


def target_dist_plot(df, target):
    plt.figure(figsize=(12, 8))
    sns.distplot(df[target], bins=50, kde=True)
    plt.xlabel(target, fontsize=12)
    plt.show()


def target_log_dist_plot(df, target):
    plt.figure(figsize=(12, 8))
    sns.distplot(np.log(df[target].values), bins=50, kde=True)
    plt.xlabel(target, fontsize=12)
    plt.show()


def date_target_median_group_bar_plot(df, col, target):
    df['yearmonth'] = df[col].apply(lambda x: x[:4]+x[5:7])
    grouped_df = df.groupby('yearmonth')[target].aggregate(np.median).reset_index()
    plt.figure(figsize=(12, 8))
    sns.barplot(grouped_df.yearmonth.values, grouped_df[target], alpha=0.8, color=color[2])
    plt.ylabel('Median Target', fontsize=12)
    plt.xlabel('Year Month', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()


def columns_type_group(train):
    dtype_df = train.dtypes.reset_index()
    dtype_df.columns = ['Count', 'Column Type']
    dtype_df = dtype_df.groupby('Column Type').aggregate('count').reset_index()
    print(dtype_df)
    plt.figure(figsize=(12, 8))
    sns.barplot(dtype_df['Column Type'].values, dtype_df['Count'].values, alpha=0.8, color=color[4])
    plt.show()


def col_target_group_median_point_plot(train, col, target):
    grouped_df = train.groupby(col)[target].aggregate(np.median).reset_index()
    plt.figure(figsize=(12, 8))
    sns.pointplot(grouped_df[col].values, grouped_df[target], alpha=0.8, color=color[2])
    plt.ylabel('Median Price', fontsize=12)
    plt.xlabel('Floor number', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()


def col_target_box_plot(train, col, target):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=col, y=target, data=train)
    plt.ylabel(target, fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv('../input/train.csv')
    train_df.describe(train_df, 'price_doc')
    target_sort_scatter_plot(train_df, 'price_doc')
    target_log_dist_plot(train_df, 'price_doc')
    date_target_median_group_bar_plot(train_df, 'timestamp', 'price_doc')
    columns_type_group(train_df)
    missing_plot(train_df)
    trim_col_vs_target_joint_plot(train_df, 'life_sq', 'price_doc', True)
    col_target_group_median_point_plot(train_df, 'floor', 'price_doc')
    col_target_box_plot(train_df, 'max_floor','price_doc')