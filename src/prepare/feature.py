import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import warnings

color = sns.color_palette()
warnings.filterwarnings('ignore')


def extract_features_from_timestamp(train_df, test_df, col='timestamp', target='price_doc', create_plots=True):
    """
        We have a timestamp variable in the dataset and
        time could be one of an important factor determining the price.
        So let us extract some features out of the timestamp variable.
    """
    # year and month
    train_df['yearmonth'] = train_df[col].dt.year * 100 + train_df[col].dt.month
    test_df['yearmonth'] = test_df[col].dt.year * 100 + test_df[col].dt.month

    # year and week
    train_df['yearweek'] = train_df[col].dt.year * 100 + train_df[col].dt.weekofyear
    test_df['yearweek'] = test_df[col].dt.year * 100 + test_df[col].dt.weekofyear

    # year
    train_df['year'] = train_df[col].dt.year
    test_df['year'] = test_df[col].dt.year

    # month of year
    train_df['month_of_year'] = train_df[col].dt.month
    test_df['month_of_year'] = test_df[col].dt.month

    # week of year
    train_df['week_of_year'] = train_df[col].dt.weekofyear
    test_df['week_of_year'] = test_df[col].dt.weekofyear

    # day of week
    train_df['day_of_week'] = train_df[col].dt.weekday
    test_df['day_of_week'] = test_df[col].dt.weekday

    if create_plots:
        plt.figure(figsize=(12, 8))
        sns.pointplot(x='yearweek', y=target, data=train_df)
        plt.ylabel(target, fontsize=12)
        plt.xlabel('yearweek', fontsize=12)
        plt.title('Median Price distribution by year and week_num')
        plt.xticks(rotation='vertical')
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='month_of_year', y=target, data=train_df)
        plt.ylabel(target, fontsize=12)
        plt.xlabel('month_of_year', fontsize=12)
        plt.title('Median Price distribution by month_of_year')
        plt.xticks(rotation='vertical')
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.pointplot(x='week_of_year', y=target, data=train_df)
        plt.ylabel(target, fontsize=12)
        plt.xlabel('week of the year', fontsize=12)
        plt.title('Median Price distribution by week of year')
        plt.xticks(rotation='vertical')
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='day_of_week', y=target, data=train_df)
        plt.ylabel(target, fontsize=12)
        plt.xlabel('day_of_week', fontsize=12)
        plt.title('Median Price distribution by day of week')
        plt.xticks(rotation='vertical')
        plt.show()


def create_feature_from_ratio(train_df, test_df, create_plots=True):
    """
        Let us create some ratio variables around it.
    """
    # ratio of living area to full area #
    train_df["ratio_life_sq_full_sq"] = train_df["life_sq"] / np.maximum(train_df["full_sq"].astype("float"), 1)
    test_df["ratio_life_sq_full_sq"] = test_df["life_sq"] / np.maximum(test_df["full_sq"].astype("float"), 1)
    train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"] < 0] = 0
    train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"] > 1] = 1
    test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"] < 0] = 0
    test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"] > 1] = 1

    # ratio of kitchen area to living area #
    train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"), 1)
    test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"), 1)
    train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"] < 0] = 0
    train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"] > 1] = 1
    test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"] < 0] = 0
    test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"] > 1] = 1

    # ratio of kitchen area to full area #
    train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"), 1)
    test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"), 1)
    train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"] < 0] = 0
    train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"] > 1] = 1
    test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"] < 0] = 0
    test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"] > 1] = 1

    if create_plots:
        plt.figure(figsize=(12, 12))
        sns.jointplot(x=train_df.ratio_life_sq_full_sq.values, y=np.log1p(train_df.price_doc.values), size=10)
        plt.ylabel('Log of Price', fontsize=12)
        plt.xlabel('Ratio of living area to full area', fontsize=12)
        plt.title("Joint plot on log of living price to ratio_life_sq_full_sq")
        plt.show()

        plt.figure(figsize=(12, 12))
        sns.jointplot(x=train_df.ratio_life_sq_full_sq.values, y=np.log1p(train_df.price_doc.values),
                      kind='kde', size=10)
        plt.ylabel('Log of Price', fontsize=12)
        plt.xlabel('Ratio of kitchen area to living area', fontsize=12)
        plt.title("Joint plot on log of living price to ratio_kitch_sq_life_sq")
        plt.show()

        plt.figure(figsize=(12, 12))
        sns.jointplot(x=train_df.ratio_life_sq_full_sq.values, y=np.log1p(train_df.price_doc.values),
                      kind='kde', size=10)
        plt.ylabel('Log of Price', fontsize=12)
        plt.xlabel('Ratio of kitchen area to full area', fontsize=12)
        plt.title("Joint plot on log of living price to ratio_kitch_sq_full_sq")
        plt.show()


def create_feature_from_floor(train_df, test_df):
    """
        Also the next important variables from EDA are floor and max_floor. So let us create two variables
         1. Floor number of the house to the total number of floors
         2. Number of floor from the top
    """
    # floor of the house to the total number of floors in the house #
    train_df["ratio_floor_max_floor"] = train_df["floor"] / train_df["max_floor"].astype("float")
    test_df["ratio_floor_max_floor"] = test_df["floor"] / test_df["max_floor"].astype("float")

    # num of floor from top #
    train_df["floor_from_top"] = train_df["max_floor"] - train_df["floor"]
    test_df["floor_from_top"] = test_df["max_floor"] - test_df["floor"]


def create_feature_from_area(train_df, test_df):
    """
    One more variable from floor area could be the difference between full area and living area.
    """
    train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]
    test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]


def create_feature_from_building_year(train_df, test_df):
    """
        Age of building might have an impact in the rental price and so we can add that one as well.
    """
    train_df["age_of_building"] = train_df["build_year"] - train_df["year"]
    test_df["age_of_building"] = test_df["build_year"] - test_df["year"]


def add_count(df, group_col):
    """
        Price of the house could also be affected by the availability of other
        houses at the same time period. So creating a count variable on the
        number of houses at the given time period might help.
    """
    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()
    grouped_df.columns = [group_col, "count_"+group_col]
    df = pd.merge(df, grouped_df, on=group_col, how="left")
    return df


def create_feature_from_school(train_df, test_df):
    """
        Since schools generally play an important role in house hunting, let us create some variables around school.
    """
    train_df["ratio_preschool"] = train_df["children_preschool"] / train_df["preschool_quota"].astype("float")
    test_df["ratio_preschool"] = test_df["children_preschool"] / test_df["preschool_quota"].astype("float")

    train_df["ratio_school"] = train_df["children_school"] / train_df["school_quota"].astype("float")
    test_df["ratio_school"] = test_df["children_school"] / test_df["school_quota"].astype("float")


def model_data(train_df, test_df):
    """
        We could potentially add more variables like this.
        But for now let us start with model building using these additional variables.
        Let us drop the variables which are not needed in model building.
        :param train_df:
        :param test_df:
        :return:
    """
    train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
    test_X = test_df.drop(["id", "timestamp"], axis=1)

    # Since our metric is "RMSLE",
    # let us use log of the target variable for model building rather than using the actual target variable.
    train_y = np.log1p(train_df.price_doc.values)
    return train_X, test_X, train_y


def Validation_Methodology(train_X, train_y, test_X):
    """
    In this competition, the trainer and test set are from different time periods and so
    let us use the last 1 year as validation set for building our models and rest as model
    development set.
    :return:
    """
    val_time = 201407
    dev_indices = np.where(train_X["yearmonth"] < val_time)
    val_indices = np.where(train_X["yearmonth"] >= val_time)
    dev_X = train_X.ix[dev_indices]
    val_X = train_X.ix[val_indices]
    dev_y = train_y[dev_indices]
    val_y = train_y[val_indices]
    print(dev_X.shape, val_X.shape)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'min_child_weight':1,
        'silent': 1,
        'seed':0
    }

    xgtrain = xgb.DMatrix(dev_X, dev_y, feature_names=dev_X.columns)
    xgtest = xgb.DMatrix(val_X, val_y, feature_names=val_X.columns)
    watchlist = [ (xgtrain,'trainer'), (xgtest, 'test') ]
    num_rounds = 100 # Increase the number of rounds while running in local
    model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=5)

    # plot the important features #
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()