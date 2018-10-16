import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
color = sns.color_palette()

# %matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 999

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


def col_target_strip_plot(df, col, target):
    col_order = np.sort(df[col].unique()).tolist()
    plt.figure(figsize=(12, 6))
    sns.stripplot(x=col, y=target, data=df, order=col_order)
    plt.xlabel(col, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title('Distribution of y variable with'+col, fontsize=15)
    plt.show()


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


def col_target_reg_plot(df, col, target):
    plt.figure(figsize=(12, 6))
    sns.regplot(x=col, y=target, data=df, scatter_kws={'alpha':0.5, 's':30})
    plt.xlabel(col, fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.title('Distribution of target variable with'+col, fontsize=12)
    plt.show()


def col_concat_violin_plot(train, test, col):
    train['eval_set'] = 'train'
    test['eval_set'] = 'test'
    full_df = pd.concat([train[[col, 'eval_set']], test[[col, 'eval_set']]], axis=0)
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='eval_set', y=col, data=full_df)
    plt.xlabel('eval_set', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Distribution of ID variable with evaluation set', fontsize=12)
    plt.show()


def object_to_label(df):
    df_object = df.select_dtypes(include=object)
    for f in df_object.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[f].values))
        df[f] = lbl.transform(list(df[f].values))
    return df


# Thanks to anokas for this #
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


def important_variables_xgb(train_X, train_y):
    xgb_params = {
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    # model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, maximize=True)

    # plot the important features #
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()


def important_variables_ensemble(train_X, train_y):
    from sklearn import ensemble
    train_X.fillna(0, inplace=True)
    model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
    model.fit(train_X, train_y)
    feat_names = train_X.columns.values

    # plot the importances
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(12,12))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", align="center")
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()


if __name__ == "__main__":
    print('')
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    col_target_strip_plot(train_df, 'X0', 'y')
    unique_number_values(train_df, False)
    binary_count_bar_plot(train_df)
    binary_target_mean_bar_plot(train_df, 'y')
    col_target_reg_plot(train_df, 'ID', 'y')
    col_concat_violin_plot(train_df, test_df, 'ID')
    trina_df = object_to_label(train_df)
    train_y = train_df['y'].values
    train_X = train_df.drop(["ID", "y"], axis=1)
    important_variables_xgb(train_X, train_y)
    important_variables_ensemble(train_X, train_y)