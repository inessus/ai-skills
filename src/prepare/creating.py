import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from copy import deepcopy
from nltk.corpus import stopwords


def date_col_bar_plot(df, col, create_plot=True):
    df['created'] = pd.to_datetime(df[col])
    df['date_created'] = df['created'].dt.date
    cnt_srs = df['date_created'].value_counts()

    if create_plot:
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(111)
        ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
        ax.xaxis_date()
        plt.xticks(rotation='vertical')
        plt.show()


def hour_col_bar_plot(df, col, create_plot=True):
    df['hour_created'] = df[col].dt.hour
    cnt_srs = df['hour_created'].value_counts()

    if create_plot:
        color = sns.color_palette()
        plt.figure(figsize=(12, 6))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
        plt.xticks(rotation='vertical')
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


def create_month_from_date(df, col, create_plot=True):
    df['transaction_month'] = df[col].dt.month
    cnt_srs = df['transaction_month'].value_counts()
    color = sns.color_palette()
    if create_plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
        plt.xticks(rotation='vertical')
        plt.xlabel('Month of transactions', fontsize=12)
        plt.xlabel('Number of Occurences', fontsize=12)
        plt.show()


def create_features_from_text(train, col):
    """
        1. Number of words in the text
        2. Number of unique words in the text
        3. Number of characters in the text
        4. Number of stopwords
        5. Number of punctuations
        6. Number of upper case words
        7. Number of title case words
        8. Average length of the words
    :param train:
    :param col:
    :return:
    """
    train_df = deepcopy(train)
    eng_stopwords = stopwords['english']
    # Number of words in the text
    train_df['num_words'] = train_df[col].apply(lambda x: len(str(x).split()))

    # Number of unique words in the text
    train_df['num_unique_words'] = train_df[col].apply(lambda x: len(set(str(x).split())))

    # Number of characters in the text
    train_df['num_chars'] = train_df[col].apply(lambda x: len(str(x)))

    # Number of stopwords in the text
    train_df['num_stopwords'] = train_df[col].apply(
        lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

    # Number of punctuations in the text
    train_df['num_punctuations'] = train_df[col].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # Number of upper case words in the text
    train_df['num_words_upper'] = train_df[col].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    # Number of title case words in the text
    train_df['num_words_title'] = train_df[col].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    # Average length of the words in the text
    train_df['maen_word_len'] = train_df[col].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    return train_df


def create_feature_from_datetime(train, col):
    """
    # Let us extract some features like year, month, day, hour from date columns #

    :param train:
    :param col:
    :return:
    """
    train_df = deepcopy(train)
    train_df[col] = pd.to_datetime(train_df[col])
    train_df['create_year'] = train_df[col].dt.year
    train_df["created_month"] = train_df["created"].dt.month
    train_df["created_day"] = train_df["created"].dt.day
    train_df["created_hour"] = train_df["created"].dt.hour
    return train_df