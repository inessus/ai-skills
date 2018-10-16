import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import warnings


def trim_col(df, cols, ul=99, ll=1):
    for col in cols:
        ulimit = np.percentile(df[col].values, ul)
        llimit = np.percentile(df[col].values, ll)
        df[col].ix[df[col] > ulimit] = ulimit
        df[col].ix[df[col] < llimit] = llimit
    return df


def trim_col_hist_plot(train_df, col, ul=99, ll=1, create_plot=True):
    llimit = np.percentile(train_df[col].values, ll)
    ulimit = np.percentile(train_df[col].values, ul)
    train_df[col].loc[train_df[col] < llimit] = llimit
    train_df[col].loc[train_df[col] > ulimit] = ulimit

    if create_plot:
        plt.figure(figsize=(8, 6))
        sns.distplot(train_df[col].values, bins=50, kde=False)
        plt.xlabel(col, fontsize=12)
        plt.show()


def trim_cols_joint_plot(df, col, target, log1p=False, ul=99.5, ll=0.5, create_plot=True):
    ulimit = np.percentile(df[col].values, ul)
    llimit = np.percentile(df[col].values, ll)
    df[col].loc[df[col] > ulimit] = ulimit
    df[col].loc[df[col] < llimit] = llimit
    color = sns.color_palette()

    if create_plot:
        plt.figure(figsize=(12, 12))
        if log1p:
            sns.jointplot(x=np.log1p(df[col].values), y=np.log1p(df[target].values), size=10, color=color[4])
        else:
            sns.jointplot(x=df[col].values, y=df[target].values, size=10, color=color[4])
        plt.ylabel(target, fontsize=12)
        plt.xlabel(col, fontsize=12)
        plt.title(col + ' VS ' + target, fontsize=12)
        plt.show()