import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

color = sns.color_palette()

# %matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'


def col_count_hue_plot(df, col1, col2):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col1, hue=col2, data=df)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('bedrooms', fontsize=12)
    plt.show()


def col_scatter_plot(train_df, col):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df[col].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.show()


def trunc_col_hist_plot(train_df, col, ul=99, ll=1, create_plot=True):
    llimit = np.percentile(train_df[col].values, ll)
    ulimit = np.percentile(train_df[col].values, ul)
    train_df[col].loc[train_df[col] < llimit] = llimit
    train_df[col].loc[train_df[col] > ulimit] = ulimit

    if create_plot:
        plt.figure(figsize=(8, 6))
        sns.distplot(train_df[col].values, bins=50, kde=False)
        plt.xlabel(col, fontsize=12)
        plt.show()


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


def logcount_group_hist_plot(df, col1, col2, create_plot=True):
    cnt_srs = df.groupby(col2)[col1].count()
    for i in [2, 10, 50, 100, 500]:
        print('Display_address that appear less than {} times: {}%'.format(i, round((cnt_srs < i).mean() * 100, 2)))

    if create_plot:
        plt.figure(figsize=(12, 6))
        plt.hist(cnt_srs.values, bins=100, log=True, alpha=0.9)
        plt.xlabel('Number of times ', fontsize=12)
        plt.ylabel('log(Cout)', fontsize=12)
        plt.show()


def len_bar_plot(df, col, create_plot=True):
    df['num_' + col] = df[col].apply(len)
    cnt_srs = df['num_' + col].value_counts()

    if create_plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
        plt.xlabel('Number', fontsize=12)
        plt.ylabel('Number of Occurences', fontsize=12)
        plt.show()


def word_cloud_plot(df, col):
    text = ""
    for ind, row in df.iterrows():
        for feature in row[col]:
            text = " ".join([text, "_".join(feature.strip().split(" "))])
    text = text.strip()
    plt.figure(figsize=(12, 6))
    wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)
    wordcloud.recolor(random_state=0)
    plt.imshow(wordcloud)
    plt.title("Wordcloud for features", fontsize=30)
    plt.axis('off')
    plt.show()


def word_cloud_plot_v1(df, col):
    text = ""
    for ind, row in df.iterrows():
        text = " ".join([text, "_".join(row[col].strip().split(" "))])
    text = text.strip()
    plt.figure(figsize=(12, 6))
    wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(
        text)
    wordcloud.recolor(random_state=0)
    plt.imshow(wordcloud)
    plt.title("Wordcloud for features", fontsize=30)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_json('../input/train.json')
    test_df = pd.read_json('../input/test.json')
    col_count_hue_plot(train_df, 'bedrooms', 'interest_level')
    col_scatter_plot(train_df, 'price')
    trunc_col_hist_plot(train_df, 'latitude')
    date_col_bar_plot(train_df, 'created')
    hour_col_bar_plot(train_df, 'created')
    logcount_group_hist_plot(train_df, 'display_address', 'display_address')
    len_bar_plot(train_df, 'photos')
    word_cloud_plot(train_df, 'features')
    word_cloud_plot(train_df, 'display_address')
    # word_cloud_plot(train_df, 'features')
    word_cloud_plot_v1(train_df, 'display_address')
