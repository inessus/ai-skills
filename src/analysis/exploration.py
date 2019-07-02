import gc
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import operator


from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from matplotlib.ticker import NullFormatter
from scipy.stats import spearmanr

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls



"""
T3ACKYHDVF-eyJsaWNlbnNlSWQiOiJUM0FDS1lIRFZGIiwibGljZW5zZWVOYW1lIjoi5bCP6bifIOeoi+W6j+WRmCIsImFzc2lnbmVlTmFtZSI6IiIsImFzc2lnbmVlRW1haWwiOiIiLCJsaWNlbnNlUmVzdHJpY3Rpb24iOiIiLCJjaGVja0NvbmN1cnJlbnRVc2UiOmZhbHNlLCJwcm9kdWN0cyI6W3siY29kZSI6IklJIiwiZmFsbGJhY2tEYXRlIjoiMjAxOS0wNi0xMyIsInBhaWRVcFRvIjoiMjAyMC0wNi0xMiJ9LHsiY29kZSI6IkFDIiwiZmFsbGJhY2tEYXRlIjoiMjAxOS0wNi0xMyIsInBhaWRVcFRvIjoiMjAyMC0wNi0xMiJ9LHsiY29kZSI6IkRQTiIsImZhbGxiYWNrRGF0ZSI6IjIwMTktMDYtMTMiLCJwYWlkVXBUbyI6IjIwMjAtMDYtMTIifSx7ImNvZGUiOiJQUyIsImZhbGxiYWNrRGF0ZSI6IjIwMTktMDYtMTMiLCJwYWlkVXBUbyI6IjIwMjAtMDYtMTIifSx7ImNvZGUiOiJHTyIsImZhbGxiYWNrRGF0ZSI6IjIwMTktMDYtMTMiLCJwYWlkVXBUbyI6IjIwMjAtMDYtMTIifSx7ImNvZGUiOiJETSIsImZhbGxiYWNrRGF0ZSI6IjIwMTktMDYtMTMiLCJwYWlkVXBUbyI6IjIwMjAtMDYtMTIifSx7ImNvZGUiOiJDTCIsImZhbGxiYWNrRGF0ZSI6IjIwMTktMDYtMTMiLCJwYWlkVXBUbyI6IjIwMjAtMDYtMTIifSx7ImNvZGUiOiJSUzAiLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiUkMiLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiUkQiLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiUEMiLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiUk0iLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiV1MiLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiREIiLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiREMiLCJmYWxsYmFja0RhdGUiOiIyMDE5LTA2LTEzIiwicGFpZFVwVG8iOiIyMDIwLTA2LTEyIn0seyJjb2RlIjoiUlNVIiwiZmFsbGJhY2tEYXRlIjoiMjAxOS0wNi0xMyIsInBhaWRVcFRvIjoiMjAyMC0wNi0xMiJ9XSwiaGFzaCI6IjEzMzgwMDA0LzAiLCJncmFjZVBlcmlvZERheXMiOjcsImF1dG9Qcm9sb25nYXRlZCI6ZmFsc2UsImlzQXV0b1Byb2xvbmdhdGVkIjpmYWxzZX0=-nTBuZDiAOuM4IHXNkS7GbCvZVZFo4EcHf9hHzfhaPYsaCGQjuCVJFEboopbPuEHn16yT9Zvf7yRuM5WGlGmpcOJnWLpCmGm65S6wHtZdX0kfSNIqnqdS1MhIHpftsAGxSswuQksrm09tltbO4nATeavGs1BIMafsCJVen+BvDFvYL7+3crkRI7AwdyMb2miLLYJcEVPhiVKZnzJUzT9uA8/4Q02BqsvX5oSJg8cLw3w7Cd0ISrn1i8uENe/1z3T/Ede0STM7eOekFaVEdO9cgzYME3iIFzi2TZXMSqIuBpJqF4NFb6M0039tEGy6EHqcksMyDTdCAASquqcDcHrUUA==-MIIElTCCAn2gAwIBAgIBCTANBgkqhkiG9w0BAQsFADAYMRYwFAYDVQQDDA1KZXRQcm9maWxlIENBMB4XDTE4MTEwMTEyMjk0NloXDTIwMTEwMjEyMjk0NlowaDELMAkGA1UEBhMCQ1oxDjAMBgNVBAgMBU51c2xlMQ8wDQYDVQQHDAZQcmFndWUxGTAXBgNVBAoMEEpldEJyYWlucyBzLnIuby4xHTAbBgNVBAMMFHByb2QzeS1mcm9tLTIwMTgxMTAxMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxcQkq+zdxlR2mmRYBPzGbUNdMN6OaXiXzxIWtMEkrJMO/5oUfQJbLLuMSMK0QHFmaI37WShyxZcfRCidwXjot4zmNBKnlyHodDij/78TmVqFl8nOeD5+07B8VEaIu7c3E1N+e1doC6wht4I4+IEmtsPAdoaj5WCQVQbrI8KeT8M9VcBIWX7fD0fhexfg3ZRt0xqwMcXGNp3DdJHiO0rCdU+Itv7EmtnSVq9jBG1usMSFvMowR25mju2JcPFp1+I4ZI+FqgR8gyG8oiNDyNEoAbsR3lOpI7grUYSvkB/xVy/VoklPCK2h0f0GJxFjnye8NT1PAywoyl7RmiAVRE/EKwIDAQABo4GZMIGWMAkGA1UdEwQCMAAwHQYDVR0OBBYEFGEpG9oZGcfLMGNBkY7SgHiMGgTcMEgGA1UdIwRBMD+AFKOetkhnQhI2Qb1t4Lm0oFKLl/GzoRykGjAYMRYwFAYDVQQDDA1KZXRQcm9maWxlIENBggkA0myxg7KDeeEwEwYDVR0lBAwwCgYIKwYBBQUHAwEwCwYDVR0PBAQDAgWgMA0GCSqGSIb3DQEBCwUAA4ICAQAF8uc+YJOHHwOFcPzmbjcxNDuGoOUIP+2h1R75Lecswb7ru2LWWSUMtXVKQzChLNPn/72W0k+oI056tgiwuG7M49LXp4zQVlQnFmWU1wwGvVhq5R63Rpjx1zjGUhcXgayu7+9zMUW596Lbomsg8qVve6euqsrFicYkIIuUu4zYPndJwfe0YkS5nY72SHnNdbPhEnN8wcB2Kz+OIG0lih3yz5EqFhld03bGp222ZQCIghCTVL6QBNadGsiN/lWLl4JdR3lJkZzlpFdiHijoVRdWeSWqM4y0t23c92HXKrgppoSV18XMxrWVdoSM3nuMHwxGhFyde05OdDtLpCv+jlWf5REAHHA201pAU6bJSZINyHDUTB+Beo28rRXSwSh3OUIvYwKNVeoBY+KwOJ7WnuTCUq1meE6GkKc4D/cXmgpOyW/1SmBz3XjVIi/zprZ0zf3qH5mkphtg6ksjKgKjmx1cXfZAAX6wcDBNaCL+Ortep1Dh8xDUbqbBVNBL4jbiL3i3xsfNiyJgaZ5sX7i8tmStEpLbPwvHcByuf59qJhV/bZOl8KqJBETCDJcY6O2aqhTUy+9x93ThKs1GKrRPePrWPluud7ttlgtRveit/pcBrnQcXOl1rHq7ByB8CFAxNotRUYL9IF5n3wJOgkPojMy6jetQA5Ogc8Sm7RG6vg1yow==
"""

color = sns.color_palette()
warnings.filterwarnings('ignore')


# single col exploration
def col_bar_plot(train, col):
    cnt_srs = train[col].value_counts()
    plt.figure(figsize=(8, 4))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Author Name', fontsize=12)
    plt.show()


def col_count_plot(df, col):
    plt.figure(figsize=(12, 8))
    sns.countplot(x=col, data=df)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title('Frequency of ' + col + " Count ", fontsize=15)
    plt.show()


def col_dist_plot(df, target, flag='none'):
    df[target].fillna(np.nanmean(df[target].values), inplace=True)
    plt.figure(figsize=(12, 8))
    if flag == 'log1p':
        sns.distplot(np.log1p(df[target].values), bins=50, kde=True)
    elif flag == 'log':
        sns.distplot(np.log(df[target].values), bins=50, kde=True)
    else:
        sns.distplot(df[target], bins=50, kde=True)
    plt.xlabel(target, fontsize=12)
    plt.show()


def col_ts_plot(df, target, date):
    import matplotlib.dates as mdates
    df['Date_mpl'] = df[date].apply(lambda x: mdates.date2num(x))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.tsplot(df[target].values, time=df.Date_mpl.values, alpha=0.8, color=color[3], ax=ax)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
    fig.autofmt_xdate()
    plt.xlabel(date, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title(target + ' distribution', fontsize=15)
    plt.show()


def col_sort_scatter_plot(train_df, col):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df[col].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.show()


def col_pie_go(train_df, col):
    temp_series = train_df[col].value_counts()
    labels = (np.array(temp_series.index))
    sizes = (np.array((temp_series / temp_series.sum()) * 100))

    trace = go.Pie(labels=labels, values=sizes)
    layout = go.Layout(
        title=col + " Distribution",
        width=900,
        height=900
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='region')


def col_bar_go(train, col, top=20, ori='h'):
    if top > 0:
        cnt_srs = train[col].value_counts().head(top)
    else:
        cnt_srs = train[col].value_counts()

    trace = go.Bar(
        y=cnt_srs.index[::-1],
        x=cnt_srs.values[::-1],
        orientation=ori,
        marker=dict(
            color=cnt_srs.values[::-1],
            colorscale=np.random.choice(['Blues', 'Portland', 'Rainbow', 'Picnic']),
            reversescale=True
        ),
    )
    layout = dict(
        title= col+ " distribution",
    )
    data = [trace]
    fig=go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=col)


def col_maps_go(df, col):
    con_df = pd.DataFrame(df[col].value_counts()).reset_index()
    con_df.columns = ['country', 'num_loans']
    con_df = con_df.reset_index().drop('index', axis=1)

    data = [dict(
        type='choropleth',
        locations=con_df[col],
        locationmode='country names',
        z=con_df['num_loans'],
        text=con_df['country'],
        colorscale=[[0, "rgb(5, 10, 172)"], [0.85, "rgb(40, 60, 190)"], [0.9, "rgb(70, 100, 245)"], \
                    [0.94, "rgb(90, 120, 245)"], [0.97, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]],
        autocolorscale=False,
        reversescale=True,
        marker=dict(
            line=dict(
                color='rgb(180,180,180)',
                width=0.5
            )),
        colorbar=dict(
            autotick=False,
            tickprefix='',
            title='Number of Loans'),
    )]
    layout = dict(
        title='Number of loans by Country',
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection=dict(
                type='Mercator'
            )
        )
    )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, validate=False, filename='loans-world-map')


# single column convert
def len_bar_plot(df, col, create_plot=True):
    df['num_' + col] = df[col].apply(len)
    cnt_srs = df['num_' + col].value_counts()

    if create_plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
        plt.xlabel('Number', fontsize=12)
        plt.ylabel('Number of Occurences', fontsize=12)
        plt.show()


def col_split_len_bar_go(train_df, col):
    ## Filling missing values ##
    train_df[col].fillna("NA", inplace=True)
    train_df["desc_nwords"] = train_df[col].apply(lambda x: len(x.split()))
    cnt_srs = train_df['desc_nwords'].value_counts().head(100)

    trace = go.Bar(
        x=cnt_srs.index,
        y=cnt_srs.values,
        marker=dict(
            color="blue",
            #colorscale = 'Blues',
            reversescale = True
        ),
    )

    layout = go.Layout(
        title='Number of words in Description column'
    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename="desc_nwords")


def col_count_bar_plot(df, col):
    cnt_srs = df[col].value_counts()
    plt.figure(figsize=(8, 4))
    color = sns.color_palette()
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
    plt.ylabel('Number of Occurences', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.show()


def col_count_bar_plot_v1(df, col):
    cnt_srs = df[col].value_counts()
    color=sns.color_palette()
    plt.figure(figsize=(12, 8))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
    plt.ylabel('Number of Occurences', fontsize=12)
    plt.xlabel('Eval set type', fontsize=12)
    plt.title('Count of rows in each dataset', fontsize=15)
    plt.xticks(rotation='vertical')
    plt.show()


# single string column
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


# two column relation

def two_cols_box_plot(train, col, target):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=col, y=target, data=train)
    plt.ylabel(target, fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()


def two_cols_join_plot(df, col1, col2):
    plt.figure(figsize=(12, 12))
    sns.jointplot(x=df[col1].values, y=df[col2].values, size=10)
    plt.ylabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)
    plt.show()


def col_count_hue_plot(df, col1, col2):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col1, hue=col2, data=df)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('bedrooms', fontsize=12)
    plt.show()


def two_cols_strip_plot(df, col, target):
    col_order = np.sort(df[col].unique()).tolist()
    plt.figure(figsize=(12, 6))
    sns.stripplot(x=col, y=target, data=df, order=col_order)
    plt.xlabel(col, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title('Distribution of y variable with'+col, fontsize=15)
    plt.show()


def two_cols_reg_plot(df, col, target):
    plt.figure(figsize=(12, 6))
    sns.regplot(x=col, y=target, data=df, scatter_kws={'alpha':0.5, 's':30})
    plt.xlabel(col, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title('Distribution of target variable with'+col, fontsize=12)
    plt.show()


def two_cols_violin_plot(train_df, col, target):
    # train_df['num_words'].loc[train_df['num_words']>80] = 80 #truncation for better visuals
    plt.figure(figsize=(12,8))
    sns.violinplot(x=target, y=col, data=train_df)
    plt.xlabel(target + ' Name', fontsize=12)
    plt.ylabel('Number of words in text', fontsize=12)
    plt.title("Number of words ", fontsize=15)
    plt.show()


def col_concat_violin_plot(train, test, col):
    train['eval_set'] = 'trainer'
    test['eval_set'] = 'test'
    full_df = pd.concat([train[[col, 'eval_set']], test[[col, 'eval_set']]], axis=0)
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='eval_set', y=col, data=full_df)
    plt.xlabel('eval_set', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Distribution of ID variable with evaluation set', fontsize=12)
    plt.show()


def col_target_correlation(train_df, target, plot='barh'):
    labels = []
    values = []
    train = train_df.select_dtypes(include=np.number)
    for col in train.columns:
        if col not in [target]:
            labels.append(col)
            values.append(spearmanr(train[col].values, train[target])[0])
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')

    if plot == 'barh':
        corr_df = corr_df[(corr_df['corr_values'] > 0.1) | (corr_df['corr_values'] < -0.1)]
        ind = np.arange(corr_df.shape[0])
        width = 0.9
        fig, ax = plt.subplots(figsize=(12, 30))
        rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='b')
        ax.set_yticks(ind)
        ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
        ax.set_xlabel('Correlation coefficient')
        ax.set_title('Correlation coeffient of the variables')
        plt.show()
    elif plot == 'heatmap':
        cols_to_use = corr_df[(corr_df['corr_values'] > 0.11) | (corr_df['corr_values'] < -0.11)].col_labels.tolist()

        temp_df = train_df[cols_to_use]
        corrmat = temp_df.corr(method='spearman')
        f, ax = plt.subplots(figsize=(20, 20))

        # Draw the heatmap using seaborn
        sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)
        plt.title("Important variables correlation map", fontsize=15)
        plt.show()
    return corr_df


    # two column group
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


def max_value_group_bar_plot(orders_df, col1, col2, create_plot=True):
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


def mean_value_group_bar_plot(orders_df, col1, col2, create_plot=True):
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


def count_value_group_pivot_heatmap(df, col1, col2, create_plot=True):
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


def two_cols_group_median_point_plot(train, col, target):
    grouped_df = train.groupby(col)[target].aggregate(np.median).reset_index()
    plt.figure(figsize=(12, 8))
    sns.pointplot(grouped_df[col].values, grouped_df[target], alpha=0.8, color=color[2])
    plt.ylabel('Median Price', fontsize=12)
    plt.xlabel('Floor number', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()


def merge(train, prop):
    return pd.merge(train, prop, on='parcelid', how='left')


def univariate_analysis(df, target, create_plot=True):
    # Let us just impute the missing values with mean values to compute correlation coefficients #
    mean_values = df.mean(axis=0)
    df.fillna(mean_values, inplace=True)

    # Now let us look at the correlation coefficient of each of these variables
    x_cols = [col for col in df.columns if col not in [target] if df[col].dtype == 'float64']

    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(df[col].values, df[target].values)[0, 1])
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')

    if create_plot:
        ind = np.arange(len(labels))
        width = 0.9
        fig, ax = plt.subplots(figsize=(12, 40))
        rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
        ax.set_yticks(ind)
        ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
        ax.set_xlabel('Correlation coefficient')
        ax.set_ylabel('Correlation coefficient of the variables')
        plt.show()

    # There are few variables at the top of this graph without any correlation values.
    # I guess they have only one unique value and hence no correlation value. Let us confirm the same.
    corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt',
                      'decktypeid', 'buildingclasstypeid']
    for col in corr_zero_cols:
        print(col, len(df[col].unique()))

    if create_plot:
        # The important variables themselves are very highly correlated.! Let us now look at each of them.
        corr_df_sel = corr_df.loc[(corr_df['corr_values'] > 0.02) | (corr_df['corr_values'] < -0.01)]
        cols_to_use = corr_df_sel.col_labels.tolist()
        print(cols_to_use)
        temp_df = df[cols_to_use]
        corrmat = temp_df.corr(method='spearman')
        f, ax = plt.subplots(figsize=(8, 8))

        # Draw the heatmap using seaborn
        sns.heatmap(corrmat, vmax=1., square=True)
        plt.title("Important variables correlation map", fontsize=15)
        plt.show()


# two datasets analysis
def test_pca(data, train_idx, test_idx, create_plots=True):
    """
        data, panda.DataFrame
        train_idx = range(0, len(train_df))
        test_idx = range(len(train_df), len(total_df))
        Run PCA analysis, return embeding
    """
    data = data.select_dtypes(include=[np.number])
    data = data.fillna(0)
    # Create a PCA object, specifying how many components we wish to keep
    pca = PCA(n_components=len(data.columns))

    # Run PCA on scaled numeric dataframe, and retrieve the projected data
    pca_trafo = pca.fit_transform(data)

    # The transformed data is in a numpy matrix. This may be inconvenient if we want to further
    # process the data, and have a more visual impression of what each column is etc. We therefore
    # put transformed/projected data into new dataframe, where we specify column names and index
    pca_df = pd.DataFrame(
        pca_trafo,
        index=data.index,
        columns=['PC' + str(i + 1) for i in range(pca_trafo.shape[1])]
    )

    if create_plots:
        # Create two plots next to each other
        _, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = list(itertools.chain.from_iterable(axes))

        # Plot the explained variance# Plot t
        axes[0].plot(
            pca.explained_variance_ratio_, "--o", linewidth=2,
            label="Explained variance ratio"
        )

        # Plot the explained variance# Plot t
        axes[0].plot(
            pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
            label="Cumulative explained variance ratio"
        )

        # show legend
        axes[0].legend(loc='best', frameon=True)

        # show biplots
        for i in range(1, 4):
            # Components to be plottet
            x, y = "PC" + str(i), "PC" + str(i + 1)

            # plot biplots
            settings = {'kind': 'scatter', 'ax': axes[i], 'alpha': 0.2, 'x': x, 'y': y}

            pca_df.iloc[train_idx].plot(label='Train', c='#ff7f0e', **settings)
            pca_df.iloc[test_idx].plot(label='Test', c='#1f77b4', **settings)
    return pca_df


# tsne_df = test_tsne(pca_df, train_idx, test_idx,title='t-SNE: Scaling on non-zeros')
def test_tsne(data, train_idx, test_idx, title='t-SNE', create_plots=True):
    """
        Run t-SNE return embedind
        train_idx = range(0, len(train_df))
        test_idx = range(len(train_df), len(total_df))
    """

    # run t-SNE
    tsne = TSNE(n_components=2, init='pca')
    Y = tsne.fit_transform(data)

    # Create plot
    # Run t-SNE on PCA embedding
    if create_plots:
        _, axes = plt.subplots(1, 1, figsize=(10, 8))
        for name, idx in zip(['Train', 'Test'], [train_idx, test_idx]):
            axes.scatter(Y[idx, 0], Y[idx, 1], label=name, alpha=0.2)
            axes.set_title(title)
            axes.xaxis.set_major_formatter(NullFormatter())
            axes.yaxis.set_major_formatter(NullFormatter())
        axes.legend()
        plt.axis('tight')
        plt.show()
    return Y


def color_plot_sne(tsne_df):
    """
        t-SNE color plot
        :param tsne_df:
        :return:
    """
    gc.collect()
    # Get our color map
    cm = plt.cm.get_cmap('RdYlBu')

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sc = axes[0].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=range(len(tsne_df)), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[0])
    cbar.set_label('Entry index')
    axes[0].set_title("t-SNE colored by index")
    axes[0].xaxis.set_major_formatter(NullFormatter())
    axes[0].yaxis.set_major_formatter(NullFormatter())

    zero_count = (tsne_df == 0).sum(axis=1)
    sc = axes[1].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=zero_count, cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label('#sparse entries')
    axes[1].set_title("t-SNE colored by number of zeros")
    axes[1].xaxis.set_major_formatter(NullFormatter())
    axes[1].yaxis.set_major_formatter(NullFormatter())


def target_color_plot(tsne_df, target_col, train_idx):
    """
        target_col = train_df['Survived']
        target_color_plot(tsne_df, target_col)
        :param tsne_df:
        :param target_col:
        :param train_idx:
        :return:
    """
    # Create plot
    cm = plt.cm.get_cmap('RdYlBu')
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    sc = axes.scatter(tsne_df[train_idx, 0], tsne_df[train_idx, 1], alpha=0.2, c=np.log1p(target_col), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label('Log1p(target)')
    axes.set_title("t-SNE colored by target")
    axes.xaxis.set_major_formatter(NullFormatter())
    axes.yaxis.set_major_formatter(NullFormatter())


def perplexity_tsne_plot(pca_df, train_idx, test_idx):
    """
    t-SNE can give some pretty tricky to intepret results depending on the perplexity parameter used.
    So just to be sure in the following I check for a few different values of the perplexity parameter.
    :param pca_df:
    :param train_idx:
    :param test_idx:
    :return:
    """
    _, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, perplexity in enumerate([5, 30, 50, 100]):

        # Create projection
        Y = TSNE(init='pca', perplexity=perplexity).fit_transform(pca_df)

        # Plot t-SNE
        for name, idx in zip(["Train", "Test"], [train_idx, test_idx]):
            axes[i].scatter(Y[idx, 0], Y[idx, 1], label=name, alpha=0.2)
        axes[i].set_title("Perplexity=%d" % perplexity)
        axes[i].xaxis.set_major_formatter(NullFormatter())
        axes[i].yaxis.set_major_formatter(NullFormatter())
        axes[i].legend()

    plt.show()


def all_method(total_df, train_idx, test_idx, target_col):
    """

    :param total_df:
    :param train_idx:
    :param test_idx:
    :param target_col: target_col=train_df['Survived'])
    :return:
    """
    COMPONENTS = 5

    # List of decomposition methods to use
    methods = [
        TruncatedSVD(n_components=COMPONENTS),
        PCA(n_components=COMPONENTS),
        FastICA(n_components=COMPONENTS),
        GaussianRandomProjection(n_components=COMPONENTS, eps=0.1),
        SparseRandomProjection(n_components=COMPONENTS, dense_output=True)
    ]

    # Run all the methods
    embeddings = []
    for method in methods:
        name = method.__class__.__name__
        embeddings.append(
            pd.DataFrame(method.fit_transform(total_df), columns=[f"{name}_{i}" for i in range(COMPONENTS)])
        )
        print(f">> Ran {name}")

    # Put all components into one dataframe
    components_df = pd.concat(embeddings, axis=1)

    # Prepare plot
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Run t-SNE on components
    tsne_df = test_tsne(
        components_df, train_idx, test_idx,
        title='t-SNE: with decomposition features'
    )

    # Color by index
    fig, axes = plt.subplots(2, 1, figsize=(20, 20))
    cm = plt.cm.get_cmap('RdYlBu')
    sc = axes[0].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=range(len(tsne_df)), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[0])
    cbar.set_label('Entry index')
    axes[0].set_title("t-SNE colored by index")
    axes[0].xaxis.set_major_formatter(NullFormatter())
    axes[0].yaxis.set_major_formatter(NullFormatter())

    # Color by target

    sc = axes[1].scatter(tsne_df[train_idx, 0], tsne_df[train_idx, 1], alpha=0.2, c=np.log1p(target_col), cmap=cm)
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label('Log1p(target)')
    axes[1].set_title("t-SNE colored by target")
    axes[1].xaxis.set_major_formatter(NullFormatter())
    axes[1].yaxis.set_major_formatter(NullFormatter())

    plt.axis('tight')
    plt.show()
    return components_df


if __name__ == "__main__":
    train_df = pd.read_json('../input/trainer.json')
    test_df = pd.read_json('../input/test.json')
    # col_count_hue_plot(train_df, 'bedrooms', 'interest_level')
    # col_scatter_plot(train_df, 'price')
    # trim_col_hist_plot(train_df, 'latitude')
    # date_col_bar_plot(train_df, 'created')
    # hour_col_bar_plot(train_df, 'created')
    # logcount_group_hist_plot(train_df, 'display_address', 'display_address')
    # len_bar_plot(train_df, 'photos')
    # word_cloud_plot(train_df, 'features')
    # word_cloud_plot(train_df, 'display_address')
    # # word_cloud_plot(train_df, 'features')
    # word_cloud_plot_v1(train_df, 'display_address')
