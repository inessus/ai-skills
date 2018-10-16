import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings



def unique_number_values(df, show=True):
    unique_values_dict = {}
    df = df.select_dtypes(include=np.number)
    for col in df.columns:
        unique_value = str(np.sort(df[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist[:]
    for unique_val, columns in unique_values_dict.items():
        if show:
            print('Columns containing the unique values', unique_val)
            print(columns)
            print("-----------------------------------------------")
    return unique_values_dict


def binary_count_bar_plot(df):
    zero_count_list = []
    one_count_list = []
    unique_values_dict = unique_number_values(df, False)

    cols_list = unique_values_dict.get('[0, 1]', [])
    for col in cols_list:
        zero_count_list.append((df[col] == 0).sum())
        one_count_list.append((df[col] == 1).sum())

    N = len(cols_list)
    ind = np.arange(N)
    width = 0.35

    plt.figure(figsize=(6, 100))
    p1 = plt.barh(ind, zero_count_list, width, color='red')
    p2 = plt.barh(ind, one_count_list, width, left=zero_count_list, color='blue')
    plt.yticks(ind, cols_list)
    plt.legend((p1[0], p2[0]), ('zero count', 'One count'))
    plt.show()


def binary_target_mean_bar_plot(df, target):
    zero_mean_list = []
    one_mean_list = []
    unique_values_dict = unique_number_values(df, False)
    cols_list = unique_values_dict.get('[0, 1]', [])

    for col in cols_list:
        zero_mean_list.append(df.loc[df[col] == 0][target].mean())
        one_mean_list.append(df.loc[df[col] == 1][target].mean())

    new_df = pd.DataFrame({'column_name': cols_list + cols_list, 'value': [0] * len(cols_list) + [1] * len(cols_list),
                           'y_mean': zero_mean_list + one_mean_list})
    new_df = new_df.pivot('column_name', 'value', 'y_mean')

    plt.figure(figsize=(8, 80))
    sns.heatmap(new_df)
    plt.title('Mean of y value across binary variables', fontsize=15)
    plt.show()
    return new_df

def unique_count_group(df, col1, col2):
    """
        根据col2分类，计算col1不重复的个数
    :param df:   pd.DataFrame
    :param col1: count to
    :param col2: by col
    :return:
    """
    print(df.groupby(col2)[col1].aggregate(lambda x: len(np.unique(x))))