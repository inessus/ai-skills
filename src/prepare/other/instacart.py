import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

# %matplotlib inline

pd.options.mode.chained_assignment=None


def col_count_plot_v1(df, col, create_plot=True):
    cnt_srs = df[col].value_counts()
    color=sns.color_palette()
    plt.figure(figsize=(12, 8))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
    plt.ylabel('Number of Occurences', fontsize=12)
    plt.xlabel('Eval set type', fontsize=12)
    plt.title('Count of rows in each dataset', fontsize=15)
    plt.xticks(rotation='vertical')
    plt.show()


def col_count_plot_bar(df, col):
    cnt_srs = df[col].value_counts()
    plt.figure(figsize=(8, 4))
    color = sns.color_palette()
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
    plt.ylabel('Number of Occurences', fontsize=12)
    plt.xlabel('bathrooms', fontsize=12)
    plt.show()


# 统计常值的列个数
def constant_unique(train_df):
    unique_df = train_df.nunique().reset_index()
    unique_df.columns = ['col_name', 'unique_count']
    constant_df = unique_df[unique_df['unique_count'] == 1]
    return constant_df


def unique_count_group(df, col1, col2):
    """
        根据col2分类，计算col1不重复的个数
    :param df:   pd.DataFrame
    :param col1: count to
    :param col2: by col
    :return:
    """
    print(df.groupby(col2)[col1].aggregate(lambda x: len(np.unique(x))))


def max_value_group(orders_df, col1, col2, create_plot=True):
    cnt_srs = orders_df.groupby(col2)[col1].aggregate(np.max).reset_index()
    cnt_srs = cnt_srs[col1].value_counts()
    color = sns.color_palette()

    if create_plot:
        plt.figure(figsize=(12, 8))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Maximum order number', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.show()


def mean_value_group(orders_df, col1, col2, create_plot=True):
    cnt_srs = orders_df.groupby(col2)[col1].aggregate('mean').reset_index()
    cnt_srs = cnt_srs[col1].value_counts()
    color = sns.color_palette()

    if create_plot:
        plt.figure(figsize=(12, 8))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Mean order number', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.show()


def count_value_group_pivot(df, col1, col2, create_plot=True):
    """
        :param df:
        :param col1: 需要统计的列名
        :param col2: 两个以上列名
        :param create_plot:
    :return:
    """
    grouped_df = df.groupby(col2)[col1].aggregate('count').reset_index()
    grouped_df = grouped_df.pivot(col2[0], col2[1], col1)

    if create_plot:
        plt.figure(figsize=(12, 6))
        sns.heatmap(grouped_df)
        plt.title('Frequency of Day of week Vs Hour of day')
        plt.show()


if __name__ == "__main__":
    print('ok')

    order_products_train_df = pd.read_csv("../input/order_products__train.csv")
    order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")
    orders_df = pd.read_csv("../input/orders.csv")
    products_df = pd.read_csv("../input/products.csv")
    aisles_df = pd.read_csv("../input/aisles.csv")
    departments_df = pd.read_csv("../input/departments.csv")

    col_count_plot_v1(orders_df, 'eval_set')
    unique_count_group(orders_df, 'user_id', 'eval_set')
    max_value_group(orders_df, 'order_number', 'user_id')
    count_value_group_pivot(orders_df, 'order_number', ['order_dow', 'order_hour_of_day'])
