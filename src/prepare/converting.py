import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn import model_selection, preprocessing
import xgboost as xgb
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def categorical_to_label_encoding(train_df, test_df):
    print('   >>> ', end='')
    for f in train_df.columns:
        if train_df[f].dtype == 'object':
            print(f, end=' ')
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))
            train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
            test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))
    print('')
    return train_df, test_df


def cols_to_LabelEncoder(train_df, cols):
    print('   >>> ', end='')
    for f in cols:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
    return train_df


def object_to_label(df):
    df_object = df.select_dtypes(include=object)
    for f in df_object.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[f].values))
        df[f] = lbl.transform(list(df[f].values))
    return df


def converting(train_df):
    """
        构建自己的分类对象，然后可以进行统计
    :return:
    """
    from io import StringIO

    temp_data = StringIO("""
    region,region_en
    Свердловская область, Sverdlovsk oblast
    Самарская область, Samara oblast
    Ростовская область, Rostov oblast
    Татарстан, Tatarstan
    Волгоградская область, Volgograd oblast
    Нижегородская область, Nizhny Novgorod oblast
    Пермский край, Perm Krai
    Оренбургская область, Orenburg oblast
    Ханты-Мансийский АО, Khanty-Mansi Autonomous Okrug
    Тюменская область, Tyumen oblast
    Башкортостан, Bashkortostan
    Краснодарский край, Krasnodar Krai
    Новосибирская область, Novosibirsk oblast
    Омская область, Omsk oblast
    Белгородская область, Belgorod oblast
    Челябинская область, Chelyabinsk oblast
    Воронежская область, Voronezh oblast
    Кемеровская область, Kemerovo oblast
    Саратовская область, Saratov oblast
    Владимирская область, Vladimir oblast
    Калининградская область, Kaliningrad oblast
    Красноярский край, Krasnoyarsk Krai
    Ярославская область, Yaroslavl oblast
    Удмуртия, Udmurtia
    Алтайский край, Altai Krai
    Иркутская область, Irkutsk oblast
    Ставропольский край, Stavropol Krai
    Тульская область, Tula oblast
    """)

    region_df = pd.read_csv(temp_data)
    train_df = pd.merge(train_df, region_df, how="left", on="region")
    return train_df


def col_tfidf(train, test, col):
    # tfidf_vec = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    full_tfidf = tfidf_vec.fit_transform(train[col].values.tolist() + test[col].values.tolist())
    train_tfidf = tfidf_vec.transform(train[col].values.tolist())
    test_tfidf = tfidf_vec.transform(test[col].values.tolist())
    return full_tfidf, train_tfidf, test_tfidf


def col_map(train, col, mapping):
    # author_mapping_dict = {'EAP': 0, 'HPL': 1, 'MWS': 2}
    return train[col].map(mapping)